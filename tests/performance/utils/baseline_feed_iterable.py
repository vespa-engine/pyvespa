# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Benchmark only: feed_iterable exactly as on master before the completion-event
rewrite, so one session can compare both on the same runner and instance."""

import sys
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Queue
from time import sleep
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from vespa.application import Vespa, VespaSync, _response_json
from vespa.io import VespaResponse


class BaselineVespa(Vespa):
    def feed_iterable(
        self,
        iter: Iterable[Dict],
        schema: Optional[str] = None,
        namespace: Optional[str] = None,
        callback: Optional[Callable[[VespaResponse, str], None]] = None,
        operation_type: Optional[str] = "feed",
        max_queue_size: int = 1000,
        max_workers: int = 8,
        max_connections: int = 16,
        compress: Union[str, bool] = "auto",
        num_retries_429: int = 10,
        **kwargs,
    ):
        """
        Feed data from an Iterable of Dict with the keys 'id' and 'fields' to be used in the `feed_data_point` function.

        Uses a queue to feed data in parallel with a thread pool. The result of each operation is forwarded
        to the user-provided callback function that can process the returned `VespaResponse`.

        Example usage:
            ```python
            app = Vespa(url="localhost", port=8080)
            data = [
                {"id": "1", "fields": {"field1": "value1"}},
                {"id": "2", "fields": {"field1": "value2"}},
            ]
            def callback(response, id):
                print(f"Response for id {id}: {response.status_code}")
            app.feed_iterable(data, schema="schema_name", callback=callback)
            ```

        Args:
            iter (Iterable[dict]): An iterable of Dict containing the keys 'id' and 'fields' to be used in the `feed_data_point`.
                Note that this 'id' is only the last part of the full document id, which will be generated automatically by pyvespa.
            schema (str): The Vespa schema name that we are sending data to.
            namespace (str, optional): The Vespa document id namespace. If no namespace is provided, the schema is used.
            callback (function): A callback function to be called on each result. Signature `callback(response: VespaResponse, id: str)`.
            operation_type (str, optional): The operation to perform. Defaults to `feed`. Valid values are `feed`, `update`, or `delete`.
            max_queue_size (int, optional): The maximum size of the blocking queue and max in-flight operations.
            max_workers (int, optional): The maximum number of workers in the threadpool executor.
            max_connections (int, optional): The maximum number of persisted connections to the Vespa endpoint.
            compress (Union[str, bool], optional): Whether to compress the request body. Defaults to "auto", which will compress if the body is larger than 1024 bytes.
            num_retries_429 (int, optional): Retries per request on a 429 response or connection error. Defaults to 10; 0 passes every response to the callback unchanged.
            **kwargs (dict, optional): Additional parameters passed to the respective operation type specific function (`_data_point`).

        Returns:
            None
        """

        if operation_type not in ["feed", "update", "delete"]:
            raise ValueError(
                "Invalid operation type. Valid are `feed`, `update` or `delete`."
            )

        if namespace is None:
            namespace = schema
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        def _consumer(
            queue: Queue,
            executor: ThreadPoolExecutor,
            sync_session: VespaSync,
            max_in_flight=2 * max_queue_size,
        ):
            in_flight = 0  # Single threaded consumer
            futures: List[Future] = []
            while True:
                try:
                    doc = queue.get(timeout=5)
                except Empty:
                    continue  # producer has not produced anything
                if doc is None:  # producer is done
                    queue.task_done()
                    break  # Break and wait for all futures to complete

                completed_futures = [future for future in futures if future.done()]
                for future in completed_futures:
                    futures.remove(future)
                    in_flight -= 1
                    _handle_result_callback(future, callback=callback)

                while in_flight >= max_in_flight:
                    # Check for completed tasks and reduce in-flight tasks
                    for future in futures:
                        if future.done():
                            futures.remove(future)
                            in_flight -= 1
                            _handle_result_callback(future, callback=callback)
                    sleep(0.01)  # wait a bit for more futures to complete

                # we can submit a new doc to Vespa
                future: Future = executor.submit(_submit, doc, sync_session)
                futures.append(future)
                in_flight += 1
                queue.task_done()  # signal that we have consumed the doc from queue

            # make sure callback is called for all pending operations before
            # exiting the consumer thread
            for future in futures:
                _handle_result_callback(future, callback)

        def _submit(
            doc: dict, sync_session: VespaSync
        ) -> Tuple[str, Union[VespaResponse, Exception]]:
            id = doc.get("id", None)
            if id is None:
                return id, VespaResponse(
                    status_code=499,
                    json={"id": id, "message": "Missing id in input dict"},
                    url="n/a",
                    operation_type=operation_type,
                )
            fields = doc.get("fields", None)
            if fields is None and operation_type != "delete":
                return id, VespaResponse(
                    status_code=499,
                    json={"id": id, "message": "Missing fields in input dict"},
                    url="n/a",
                    operation_type=operation_type,
                )
            groupname = doc.get("groupname", None)
            try:
                if operation_type == "feed":
                    response: VespaResponse = sync_session.feed_data_point(
                        schema=schema,
                        namespace=namespace,
                        groupname=groupname,
                        data_id=id,
                        fields=fields,
                        **kwargs,
                    )
                    return (id, response)
                elif operation_type == "update":
                    response: VespaResponse = sync_session.update_data(
                        schema=schema,
                        namespace=namespace,
                        groupname=groupname,
                        data_id=id,
                        fields=fields,
                        **kwargs,
                    )
                    return (id, response)
                elif operation_type == "delete":
                    response: VespaResponse = sync_session.delete_data(
                        schema=schema,
                        namespace=namespace,
                        data_id=id,
                        groupname=groupname,
                        **kwargs,
                    )
                    return (id, response)
            except Exception as e:
                return (id, e)

        def _handle_result_callback(
            future: Future, callback: Optional[Callable[[VespaResponse, str], None]]
        ):
            id, response = future.result()
            if isinstance(response, Exception):
                # The HTTPError from raise_for_status may be the VespaError's cause.
                http_response = getattr(response, "response", None)
                if http_response is None:
                    http_response = getattr(response.__cause__, "response", None)
                if isinstance(getattr(http_response, "status_code", None), int):
                    # A non-2xx response: pass it on as it came from Vespa.
                    response = VespaResponse(
                        json=_response_json(http_response),
                        status_code=http_response.status_code,
                        url=str(http_response.url),
                        operation_type=operation_type,
                    )
                else:
                    response = VespaResponse(
                        status_code=599,
                        json={
                            "Exception": str(response),
                            "id": id,
                            "message": "Exception during feed_data_point",
                        },
                        url="n/a",
                        operation_type=operation_type,
                    )
            if callback is not None:
                try:
                    callback(response, id)
                except Exception as e:
                    print(f"Exception in user callback for id {id}", file=sys.stderr)
                    traceback.print_exception(
                        type(e), e, e.__traceback__, file=sys.stderr
                    )

        with VespaSync(
            app=self,
            pool_maxsize=max_connections,
            pool_connections=max_connections,
            compress=compress,
            num_retries_429=num_retries_429,
        ) as session:
            queue = Queue(maxsize=max_queue_size)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                consumer_thread = threading.Thread(
                    target=_consumer, args=(queue, executor, session, max_queue_size)
                )
                consumer_thread.start()
                for doc in iter:
                    queue.put(doc, block=True)
                queue.put(None, block=True)
                queue.join()
                consumer_thread.join()
