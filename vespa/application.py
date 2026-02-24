# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import asyncio
import traceback
import concurrent.futures
import warnings
from typing import Optional, Dict, Generator, List, IO, Iterable, Callable, Tuple, Union, TypeVar
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from queue import Queue, Empty
import threading
import httpr

from requests import Session
from requests.models import Response
from requests.exceptions import ConnectionError, HTTPError, JSONDecodeError
from tenacity import (
    retry,
    wait_exponential,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_result,
    retry_if_exception,
    retry_if_exception_type,
    retry_any,
    RetryCallState,
)
from time import sleep
from urllib.parse import quote
import random
import time

from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse, VespaResponse, VespaVisitResponse
from vespa.package import ApplicationPackage
from vespa.throttling import AdaptiveThrottler

import httpx
import vespa
import gzip

from io import BytesIO
import logging
import json

logging.getLogger("urllib3").setLevel(logging.ERROR)

VESPA_CLOUD_SECRET_TOKEN: str = "VESPA_CLOUD_SECRET_TOKEN"


_T = TypeVar("_T")


async def bounded_gather(
    *awaitables: Awaitable[_T],
    max_concurrency: int,
) -> list[_T]:
    """Gather awaitables with a sliding-window concurrency limit.

    Unlike batching (gather N, wait for all N, then start the next N),
    this starts a new awaitable as soon as any one completes, so a single
    slow request does not block the remaining work.

    Args:
        *awaitables: The awaitables to run concurrently.
        max_concurrency: Maximum number of awaitables allowed to run at the same time.

    Returns:
        List of results in the same order as the input awaitables.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_under_semaphore(awaitable: Awaitable[_T]) -> _T:
        async with semaphore:
            return await awaitable

    return list(await asyncio.gather(*(run_under_semaphore(a) for a in awaitables)))


def get_profiling_params() -> Dict[str, str]:
    """
    Get the profiling parameters to add to a query.

    When profiling is enabled, these parameters add detailed trace and timing
    information to the query response. Note that the response may be significantly
    larger when profiling is enabled.

    Returns:
        Dict[str, str]: Dictionary of profiling parameters to merge with query params.
    """
    return {
        "trace.level": "1",
        "trace.explainLevel": "1",
        "trace.profileDepth": "100",
        "trace.timestamps": "true",
        "presentation.timing": "true",
    }


def _is_connection_error(e: Exception) -> bool:
    """
    Check if an exception is a connection-related error.

    This handles both requests.ConnectionError and httpr exceptions
    (RequestError, ConnectError) as well as generic network errors.

    Args:
        e: The exception to check

    Returns:
        True if this is a connection/network error, False otherwise
    """
    error_str = str(e).lower()
    return (
        isinstance(e, ConnectionError)
        or isinstance(e, ConnectionResetError)
        or (hasattr(httpr, "RequestError") and isinstance(e, httpr.RequestError))
        or (hasattr(httpr, "ConnectError") and isinstance(e, httpr.ConnectError))
        or "error sending request" in error_str
        or "connection" in error_str
        or type(e).__name__ == "RequestError"
    )


def _prepare_mtls_cert_data(cert: Optional[str], key: Optional[str]) -> Optional[bytes]:
    """
    Prepare mTLS certificate data for httpr.

    Reads certificate and key files and combines them into a single bytes object
    for use with httpr's client_pem_data parameter.

    Args:
        cert: Path to certificate file (may also contain key)
        key: Path to key file (optional if cert contains both)

    Returns:
        Combined certificate and key data as bytes, or None if no cert provided
    """
    if not cert:
        return None

    # Read cert file
    with open(cert, "rb") as f:
        cert_content = f.read()

    if not key or key == cert:
        # Single file with both cert and key
        return cert_content

    # Read key file and combine
    with open(key, "rb") as f:
        key_content = f.read()

    # Combine cert and key
    if not cert_content.endswith(b"\n"):
        return cert_content + b"\n" + key_content
    return cert_content + key_content


def _prepare_request_body(
    method: str,
    json_data=None,
    data=None,
    compress: Union[str, bool] = "auto",
    compress_larger_than: int = 1024,
) -> Tuple[any, Dict[str, str]]:
    """
    Prepare request body with optional compression.

    Args:
        method: HTTP method (POST, PUT, GET, DELETE, etc.)
        json_data: JSON data to send
        data: Raw data to send
        compress: Whether to compress ("auto", True, or False)
        compress_larger_than: Threshold in bytes for auto compression

    Returns:
        Tuple of (body_data, extra_headers)
        - body_data: The data to send (could be original or compressed)
        - extra_headers: Dict of additional headers to include
    """
    if method not in ["POST", "PUT"]:
        return json_data, {}

    if compress is False:
        return json_data or data, {}

    # Determine body content
    if json_data is not None:
        body_bytes = json.dumps(json_data).encode("utf-8")
        content_type = "application/json"
    elif data is not None:
        body_bytes = data if isinstance(data, bytes) else data.encode("utf-8")
        content_type = "application/octet-stream"
    else:
        return json_data, {}

    # Check if compression should be applied
    should_compress = compress is True or (
        compress == "auto" and len(body_bytes) > compress_larger_than
    )

    if should_compress:
        # Compress the body
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as f:
            f.write(body_bytes)
        compressed_body = buf.getvalue()

        # Return raw bytes and headers for httpr
        return compressed_body, {
            "Content-Encoding": "gzip",
            "Content-Type": content_type,
        }
    else:
        return json_data, {}


def raise_for_status(
    response: Response, raise_on_not_found: Optional[bool] = False
) -> None:
    """
    Raises an appropriate error if necessary.

    If the response contains an error message, `VespaError` is raised along with `HTTPError` to provide more details.

    Args:
        response (Response): Response object from the Vespa API (requests or httpr).
        raise_on_not_found (bool): If True, raises `HTTPError` if status_code is 404.

    Raises:
        HTTPError: If status_code is between 400 and 599.
        VespaError: If the response JSON contains an error message.
    """

    # Check if response has raise_for_status method (requests/httpx)
    # or if we need to check manually (httpr)
    has_error = False
    http_error = None

    if hasattr(response, "raise_for_status"):
        # requests/httpx Response - use built-in method
        try:
            response.raise_for_status()
            return  # No error, return early
        except HTTPError as e:
            has_error = True
            http_error = e
    else:
        # httpr Response - check status code manually
        if 400 <= response.status_code < 600:
            has_error = True
            # Try to format error message with JSON if available
            try:
                error_json = response.json()
                http_error = HTTPError(
                    f"HTTP {response.status_code}: {json.dumps(error_json)}"
                )
            except Exception:
                # Fall back to text if JSON parsing fails
                http_error = HTTPError(f"HTTP {response.status_code}: {response.text}")

    if has_error:
        # Handle 404 special case
        if response.status_code == 404 and not raise_on_not_found:
            return

        # Try to extract error details from JSON
        try:
            response_json = response.json()
            errors = response_json.get("root", {}).get("errors", [])
            error_message = response_json.get("message", None)
            if errors:
                raise VespaError(errors) from http_error
            if error_message:
                raise VespaError(error_message) from http_error
        except JSONDecodeError:
            # If we can't parse JSON, just raise the HTTP error
            pass

        raise http_error


class Vespa(object):
    def __init__(
        self,
        url: str,
        port: Optional[int] = None,
        deployment_message: Optional[List[str]] = None,
        cert: Optional[str] = None,
        key: Optional[str] = None,
        vespa_cloud_secret_token: Optional[str] = None,
        output_file: IO = sys.stdout,
        application_package: Optional[ApplicationPackage] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Establish a connection with an existing Vespa application.

        Args:
            url (str): Vespa endpoint URL.
            port (int): Vespa endpoint port.
            deployment_message (str): Message returned by Vespa engine after deployment. Used internally by deploy methods.
            cert (str): Path to data plane certificate and key file in case the 'key' parameter is None. If 'key' is not None, this should be the path of the certificate file. Typically generated by Vespa-cli with 'vespa auth cert'.
            key (str): Path to the data plane key file. Typically generated by Vespa-cli with 'vespa auth cert'.
            vespa_cloud_secret_token (str): Vespa Cloud data plane secret token.
            output_file (str): Output file to write output messages.
            application_package (str): Application package definition used to deploy the application.
            additional_headers (dict): Additional headers to be sent to the Vespa application.

        Example usage:
            ```python
            Vespa(url="https://cord19.vespa.ai")   # doctest: +SKIP

            Vespa(url="http://localhost", port=8080)
            Vespa(http://localhost, 8080)

            Vespa(url="https://token-endpoint..z.vespa-app.cloud", vespa_cloud_secret_token="your_token")  # doctest: +SKIP

            Vespa(url="https://mtls-endpoint..z.vespa-app.cloud", cert="/path/to/cert.pem", key="/path/to/key.pem")  # doctest: +SKIP

            Vespa(url="https://mtls-endpoint..z.vespa-app.cloud", cert="/path/to/cert.pem", key="/path/to/key.pem", additional_headers={"X-Custom-Header": "test"})  # doctest: +SKIP
            ```
        """
        self.output_file = output_file
        self.url = url
        self.port = port
        self.deployment_message = deployment_message
        self.cert = cert
        self.key = key
        self.vespa_cloud_secret_token = vespa_cloud_secret_token
        self._application_package = application_package
        self.pyvespa_version = vespa.__version__
        self.base_headers = {"User-Agent": f"pyvespa/{self.pyvespa_version}"}
        if additional_headers is not None:
            self.base_headers.update(additional_headers)
        if port is None:
            self.end_point = self.url
        else:
            self.end_point = str(url).rstrip("/") + ":" + str(port)
        self.search_end_point = self.end_point + "/search/"
        if self.vespa_cloud_secret_token is not None:
            self.auth_method = "token"
            self.base_headers.update(
                {"Authorization": f"Bearer {self.vespa_cloud_secret_token}"}
            )
        else:
            self.auth_method = "mtls"

    def asyncio(
        self,
        connections: Optional[int] = 1,
        total_timeout: Optional[int] = None,
        timeout: Union[httpx.Timeout, int, float] = 30.0,
        client: Optional[httpx.AsyncClient] = None,
        **kwargs,
    ) -> "VespaAsync":
        """
        Access Vespa asynchronous connection layer.
        Should be used as a context manager.

        Example usage:
            ```python
            async with app.asyncio() as async_app:
                response = await async_app.query(body=body)

            # passing kwargs with custom timeout
            async with app.asyncio(connections=5, timeout=60.0) as async_app:
                response = await async_app.query(body=body)

            ```
        See `VespaAsync` for more details on the parameters.

        Args:
            connections (int): Number of maximum_keepalive_connections.
            total_timeout (int, optional): Deprecated. Will be ignored. Use timeout instead.
            timeout (float | int | httpx.Timeout, optional): Timeout in seconds. Defaults to 30.0.
                httpx.Timeout is deprecated but still supported for backward compatibility.
            client (httpx.AsyncClient, optional): Reusable httpx.AsyncClient to use instead of creating a new
                one. When provided, the caller is responsible for closing the client.
            **kwargs (dict, optional): Additional arguments to be passed to the httpx.AsyncClient.

        Returns:
            VespaAsync: Instance of Vespa asynchronous layer.
        """

        return VespaAsync(
            app=self,
            connections=connections,
            total_timeout=total_timeout,
            timeout=timeout,
            client=client,
            **kwargs,
        )

    def get_async_session(
        self,
        connections: Optional[int] = 1,
        total_timeout: Optional[int] = None,
        timeout: Union[httpx.Timeout, int, float] = 30.0,
        **kwargs,
    ) -> httpx.AsyncClient:
        """Return a configured `httpx.AsyncClient` for reuse.

        The client is created with the same configuration as `VespaAsync` and is HTTP/2
        enabled by default. Callers are responsible for closing the client via
        `await client.aclose()` when finished.

        Args:
            connections (int, optional): Number of logical connections to keep alive.
            timeout (float | int | httpx.Timeout, optional): Timeout in seconds. Defaults to 30.0.
                httpx.Timeout is deprecated but still supported for backward compatibility.
            **kwargs: Additional keyword arguments forwarded to `httpx.AsyncClient`.

        Returns:
            httpx.AsyncClient: Configured asynchronous HTTP client.
        """

        async_layer = VespaAsync(
            app=self,
            connections=connections,
            total_timeout=total_timeout,
            timeout=timeout,
            client=None,
            **kwargs,
        )
        client = async_layer._open_httpr_client()
        async_layer._owns_client = False
        return client

    def syncio(
        self,
        connections: Optional[int] = 8,
        compress: Union[str, bool] = "auto",
        session: Optional[Session] = None,
    ) -> "VespaSync":
        """
        Access Vespa synchronous connection layer.
        Should be used as a context manager.

        Example usage:

            ```python
            with app.syncio() as sync_app:
                response = sync_app.query(body=body)
            ```

        See <class.VespaSync> for more details.

        Args:
            connections (int): Number of allowed concurrent connections.
            total_timeout (float): Total timeout in seconds.
            compress (Union[str, bool], optional): Whether to compress the request body. Defaults to "auto",
                which will compress if the body is larger than 1024 bytes.
            session (requests.Session, optional): Reusable requests session to utilise for all requests made
                within the context manager. When provided, the caller is responsible for closing the session.

        Returns:
            VespaAsyncLayer: Instance of Vespa asynchronous layer.
        """
        return VespaSync(
            app=self,
            pool_connections=connections,
            pool_maxsize=connections,
            compress=compress,
            session=session,
        )

    def get_sync_session(
        self,
        connections: Optional[int] = 8,
        compress: Union[str, bool] = "auto",
    ) -> httpr.Client:
        """Return a configured httpr.Client for reuse.

        The returned client is configured with the same headers, authentication, and
        mTLS certificates as the VespaSync context manager. Callers are responsible
        for closing the client when it is no longer needed.

        Args:
            connections (int, optional): Kept for API compatibility (httpr manages pooling).
            compress (Union[str, bool], optional): Whether to compress request bodies.

        Returns:
            httpr.Client: Configured HTTP client.
        """
        sync_layer = VespaSync(
            app=self,
            pool_connections=connections,
            pool_maxsize=connections,
            compress=compress,
            session=None,  # Let VespaSync create httpr.Client
        )
        session = sync_layer._open_http_client()
        sync_layer._owns_client = False
        return session

    @staticmethod
    def _run_coroutine_new_event_loop(loop, coro):
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    @staticmethod
    def _check_for_running_loop_and_run_coroutine(coro):
        try:
            _ = asyncio.get_running_loop()
            new_loop = asyncio.new_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    Vespa._run_coroutine_new_event_loop, new_loop, coro
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    def http(self, pool_maxsize: int = 10):
        return VespaSync(
            app=self, pool_maxsize=pool_maxsize, pool_connections=pool_maxsize
        )

    def __repr__(self) -> str:
        if self.port:
            return "Vespa({}, {})".format(self.url, self.port)
        else:
            return "Vespa({})".format(self.url)

    def _infer_schema_name(self):
        if not self._application_package:
            raise ValueError(
                "Application Package not available. Not possible to infer schema name."
            )

        try:
            schema = self._application_package.schema
        except AssertionError:
            raise ValueError(
                "Application has more than one schema. Not possible to infer schema name."
            )

        if not schema:
            raise ValueError(
                "Application has no schema. Not possible to infer schema name."
            )

        return schema.name

    def wait_for_application_up(self, max_wait: int = 300) -> None:
        """
        Wait for application endpoint ready (/ApplicationStatus).

        Args:
            max_wait (int): Seconds to wait for the application endpoint.

        Raises:
            RuntimeError: If not able to reach endpoint within `max_wait` or the client fails to authenticate.

        Returns:
            None
        """
        for wait_sec in range(max_wait):
            sleep(1)
            try:
                response = self.get_application_status()
                if not response:
                    continue
                if response.status_code == 200:
                    print("Application is up!", file=self.output_file)
                    return
            except Exception as e:
                # During startup, connection errors are expected
                # Catch both requests.ConnectionError and httpr exceptions
                if not _is_connection_error(e):
                    # If it's not a connection error, re-raise
                    raise
                # Otherwise, silently continue waiting
                pass

            if wait_sec % 5 == 0:
                print(
                    f"Waiting for application to come up, {wait_sec}/{max_wait} seconds.",
                    file=self.output_file,
                )
        else:
            raise RuntimeError(
                "Could not connect to endpoint {0} using any of the available auth methods within {1} seconds.".format(
                    self.end_point, max_wait
                )
            )

    def get_application_status(self) -> Optional[Response]:
        """
        Get application status (/ApplicationStatus).

        Returns:
            None
        """
        endpoint = f"{self.end_point}/ApplicationStatus"
        with self.syncio() as sync_sess:
            response = sync_sess._request_with_retry("GET", endpoint)
        return response

    def get_model_endpoint(self, model_id: Optional[str] = None) -> Optional[Response]:
        """Get stateless model evaluation endpoints."""

        with VespaSync(self, pool_connections=1, pool_maxsize=1) as sync_app:
            return sync_app.get_model_endpoint(model_id=model_id)

    def query(
        self,
        body: Optional[Dict] = None,
        groupname: str = None,
        streaming: bool = False,
        profile: bool = False,
        **kwargs,
    ) -> Union[VespaQueryResponse, Generator[str, None, None]]:
        """
        Send a query request to the Vespa application.

        Send 'body' containing all the request parameters.

        Args:
            body (dict): Dictionary containing request parameters.
            groupname (str, optional): The groupname used with streaming search.
            streaming (bool, optional): Whether to use streaming mode (SSE). Defaults to False.
            profile (bool, optional): Add profiling parameters to the query (response may be large). Defaults to False.
            **kwargs (dict, optional): Extra Vespa Query API parameters.

        Returns:
            VespaQueryResponse when streaming=False, or a generator of decoded lines when streaming=True.
        """

        # Use one connection as this is a single query
        with VespaSync(self, pool_maxsize=1, pool_connections=1) as sync_app:
            return sync_app.query(
                body=body,
                groupname=groupname,
                streaming=streaming,
                profile=profile,
                **kwargs,
            )

    def feed_data_point(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        namespace: str = None,
        groupname: str = None,
        compress: Union[str, bool] = "auto",
        **kwargs,
    ) -> VespaResponse:
        """
        Feed a data point to a Vespa app. Will create a new VespaSync with
        connection overhead.

        Example usage:
            ```python
            app = Vespa(url="localhost", port=8080)
            data_id = "1",
            fields = {
                    "field1": "value1",
                }
            with VespaSync(app) as sync_app:
                response = sync_app.feed_data_point(
                    schema="schema_name",
                    data_id=data_id,
                    fields=fields
                )
            print(response)
            ```

        Args:
            schema (str): The schema that we are sending data to.
            data_id (str): Unique id associated with this data point.
            fields (dict): Dictionary containing all the fields required by the `schema`.
            namespace (str, optional): The namespace that we are sending data to.
            groupname (str, optional): The groupname that we are sending data to.
            compress (Union[str, bool], optional): Whether to compress the request body. Defaults to "auto", which will compress if the body is larger than 1024 bytes.

        Returns:
            VespaResponse: The response of the HTTP POST request.
        """

        if not namespace:
            namespace = schema
        # Use low low connection settings to avoid too much overhead for a
        # single data point
        with VespaSync(
            app=self, pool_connections=1, pool_maxsize=1, compress=compress
        ) as sync_app:
            return sync_app.feed_data_point(
                schema=schema,
                data_id=data_id,
                fields=fields,
                namespace=namespace,
                groupname=groupname,
                **kwargs,
            )

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

    def feed_async_iterable(
        self,
        iter: Iterable[Dict],
        schema: Optional[str] = None,
        namespace: Optional[str] = None,
        callback: Optional[Callable[[VespaResponse, str], None]] = None,
        operation_type: Optional[str] = "feed",
        max_queue_size: int = 1000,
        max_workers: int = 64,
        max_connections: int = 1,
        **kwargs,
    ):
        """
        Feed data asynchronously using httpx.AsyncClient with HTTP/2. Feed from an Iterable of Dict with the keys 'id' and 'fields' to be used in the `feed_data_point` function.
        The result of each operation is forwarded to the user-provided callback function that can process the returned `VespaResponse`.
        Prefer using this method over `feed_iterable` when the operation is I/O bound from the client side.

        Example usage:
            ```python
            app = Vespa(url="localhost", port=8080)
            data = [
                {"id": "1", "fields": {"field1": "value1"}},
                {"id": "2", "fields": {"field1": "value2"}},
            ]
            def callback(response, id):
                print(f"Response for id {id}: {response.status_code}")
            app.feed_async_iterable(data, schema="schema_name", callback=callback)
            ```

        Args:
            iter (Iterable[dict]): An iterable of Dict containing the keys 'id' and 'fields' to be used in the `feed_data_point`.
                Note that this 'id' is only the last part of the full document id, which will be generated automatically by pyvespa.
            schema (str): The Vespa schema name that we are sending data to.
            namespace (str, optional): The Vespa document id namespace. If no namespace is provided, the schema is used.
            callback (function): A callback function to be called on each result. Signature `callback(response: VespaResponse, id: str)`.
            operation_type (str, optional): The operation to perform. Defaults to `feed`. Valid values are `feed`, `update`, or `delete`.
            max_queue_size (int, optional): Deprecated and no longer used. Kept for backwards compatibility. Default is 1000.
            max_workers (int, optional): Maximum number of concurrent in-flight requests (sliding-window concurrency limit). A new request starts as soon as any in-flight request completes. Increase if the server is scaled to handle more requests. Default is 64.
            max_connections (int, optional): The maximum number of connections passed to httpx.AsyncClient to the Vespa endpoint. As HTTP/2 is used, only one connection is needed.
            **kwargs (dict, optional): Additional parameters passed to the respective operation type-specific function (`_data_point`).

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

        # Wrapping in async function to be able to use asyncio.run, and avoid that the feed_async_iterable have to be async
        async def run():
            async with self.asyncio(connections=max_connections) as async_session:

                async def process_document(doc: Dict) -> None:
                    doc_id = doc.get("id")
                    fields = doc.get("fields")
                    groupname = doc.get("groupname")

                    if doc_id is None:
                        response = VespaResponse(
                            status_code=499,
                            json={
                                "id": doc_id,
                                "message": "Missing id in input dict",
                            },
                            url="n/a",
                            operation_type=operation_type,
                        )
                        if callback is not None:
                            callback(response, doc_id)
                        return
                    if fields is None and operation_type != "delete":
                        response = VespaResponse(
                            status_code=499,
                            json={
                                "id": doc_id,
                                "message": "Missing fields in input dict",
                            },
                            url="n/a",
                            operation_type=operation_type,
                        )
                        if callback is not None:
                            callback(response, doc_id)
                        return

                    try:
                        if operation_type == "feed":
                            response = await async_session.feed_data_point(
                                schema=schema,
                                namespace=namespace,
                                groupname=groupname,
                                data_id=doc_id,
                                fields=fields,
                                **kwargs,
                            )
                        elif operation_type == "update":
                            response = await async_session.update_data(
                                schema=schema,
                                namespace=namespace,
                                groupname=groupname,
                                data_id=doc_id,
                                fields=fields,
                                **kwargs,
                            )
                        elif operation_type == "delete":
                            response = await async_session.delete_data(
                                schema=schema,
                                namespace=namespace,
                                data_id=doc_id,
                                groupname=groupname,
                                **kwargs,
                            )
                    except Exception as e:
                        response = VespaResponse(
                            status_code=599,
                            json={
                                "Exception": str(e),
                                "id": doc_id,
                                "message": "Exception during feed_data_point",
                            },
                            url="n/a",
                            operation_type=operation_type,
                        )

                    if callback is not None:
                        try:
                            callback(response, doc_id)
                        except Exception as e:
                            print(
                                f"Exception in user callback for id {doc_id}",
                                file=sys.stderr,
                            )
                            traceback.print_exception(
                                type(e), e, e.__traceback__, file=sys.stderr
                            )

                await bounded_gather(
                    *(process_document(doc) for doc in iter),
                    max_concurrency=max_workers,
                )

        asyncio.run(run())
        return

    async def query_many_async(
        self,
        queries: Iterable[Dict],
        num_connections: int = 1,
        max_concurrent: int = 100,
        adaptive: bool = True,
        client_kwargs: Dict = {},
        **query_kwargs,
    ) -> List[VespaQueryResponse]:
        """
        Execute many queries asynchronously using httpx.AsyncClient.
        Number of concurrent requests is controlled by the `max_concurrent` parameter.
        Each query will be retried up to 3 times using an exponential backoff strategy.

        When adaptive=True (default), an AdaptiveThrottler is used that starts with
        a conservative concurrency limit and automatically adjusts based on server
        responses to prevent overloading Vespa with expensive operations.

        Args:
            queries (Iterable[dict]): Iterable of query bodies (dictionaries) to be sent.
            num_connections (int, optional): Number of connections to be used in the asynchronous client (uses HTTP/2). Defaults to 1.
            max_concurrent (int, optional): Maximum concurrent requests to be sent. Defaults to 100. Be careful with increasing too much.
            adaptive (bool, optional): Use adaptive throttling. Defaults to True. When True, starts with lower concurrency and adjusts based on error rates.
            client_kwargs (dict, optional): Additional arguments to be passed to the httpx.AsyncClient.
            **query_kwargs (dict, optional): Additional arguments to be passed to the query method.

        Returns:
            List[VespaQueryResponse]: List of `VespaQueryResponse` objects.
        """

        results = []
        # Use the asynchronous client from VespaAsync (created via self.asyncio).
        async with self.asyncio(connections=num_connections, **client_kwargs) as client:
            if adaptive:
                throttler = AdaptiveThrottler(
                    initial_concurrent=min(10, max_concurrent),
                    max_concurrent=max_concurrent,
                )
            else:
                throttler = None
                sem = asyncio.Semaphore(max_concurrent)

            async def query_wrapper(query_body: Dict) -> VespaQueryResponse:
                # Access semaphore dynamically to pick up throttler adjustments
                async with throttler.semaphore if throttler else sem:
                    response = await client.query(query_body, **query_kwargs)
                    if throttler:
                        await throttler.record_result(response.status_code)
                    return response

            tasks = [query_wrapper(q) for q in queries]
            results = await asyncio.gather(*tasks)
        return results

    def query_many(
        self,
        queries: Iterable[Dict],
        num_connections: int = 1,
        max_concurrent: int = 100,
        adaptive: bool = True,
        client_kwargs: Dict = {},
        **query_kwargs,
    ) -> List[VespaQueryResponse]:
        """
        Execute many queries asynchronously using httpx.AsyncClient.
        This method is a wrapper around the `query_many_async` method that uses the asyncio event loop to run the coroutine.
        Number of concurrent requests is controlled by the `max_concurrent` parameter.
        Each query will be retried up to 3 times using an exponential backoff strategy.

        When adaptive=True (default), an AdaptiveThrottler is used that starts with
        a conservative concurrency limit and automatically adjusts based on server
        responses to prevent overloading Vespa with expensive operations.

        Args:
            queries (Iterable[dict]): Iterable of query bodies (dictionaries) to be sent.
            num_connections (int, optional): Number of connections to be used in the asynchronous client (uses HTTP/2). Defaults to 1.
            max_concurrent (int, optional): Maximum concurrent requests to be sent. Defaults to 100. Be careful with increasing too much.
            adaptive (bool, optional): Use adaptive throttling. Defaults to True. When True, starts with lower concurrency and adjusts based on error rates.
            client_kwargs (dict, optional): Additional arguments to be passed to the httpx.AsyncClient.
            **query_kwargs (dict, optional): Additional arguments to be passed to the query method.

        Returns:
            List[VespaQueryResponse]: List of `VespaQueryResponse` objects.
        """
        return self._check_for_running_loop_and_run_coroutine(
            self.query_many_async(
                queries=queries,
                num_connections=num_connections,
                max_concurrent=max_concurrent,
                adaptive=adaptive,
                client_kwargs=client_kwargs,
                **query_kwargs,
            )
        )

    def delete_data(
        self,
        schema: str,
        data_id: str,
        namespace: str = None,
        groupname: str = None,
        **kwargs,
    ) -> VespaResponse:
        """
        Delete a data point from a Vespa app.

        Example usage:
            ```python
            app = Vespa(url="localhost", port=8080)
            response = app.delete_data(schema="schema_name", data_id="1")
            print(response)
            ```

        Args:
            schema (str): The schema that we are deleting data from.
            data_id (str): Unique id associated with this data point.
            namespace (str, optional): The namespace that we are deleting data from. If no namespace is provided, the schema is used.
            groupname (str, optional): The groupname that we are deleting data from.
            **kwargs (dict, optional): Additional arguments to be passed to the HTTP DELETE request.
                See [Vespa API documentation](https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters) for more details.

        Returns:
            Response: The response of the HTTP DELETE request.
        """

        with VespaSync(self, pool_connections=1, pool_maxsize=1) as sync_app:
            return sync_app.delete_data(
                schema=schema,
                data_id=data_id,
                namespace=namespace,
                groupname=groupname,
                **kwargs,
            )

    def delete_all_docs(
        self,
        content_cluster_name: str,
        schema: str,
        namespace: str = None,
        slices: int = 1,
        **kwargs,
    ) -> Response:
        """
        Delete all documents associated with the schema. This might block for a long time as
        it requires sending multiple delete requests to complete.

        Args:
            content_cluster_name (str): Name of content cluster to GET from, or visit.
            schema (str): The schema that we are deleting data from.
            namespace (str, optional): The namespace that we are deleting data from. If no namespace is provided, the schema is used.
            slices (int, optional): Number of slices to use for parallel delete requests. Defaults to 1.
            **kwargs (dict, optional): Additional arguments to be passed to the HTTP DELETE request.
                See [Vespa API documentation](https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters) for more details.

        Returns:
            Response: The response of the HTTP DELETE request.
        """

        with VespaSync(self, pool_connections=slices, pool_maxsize=slices) as sync_app:
            return sync_app.delete_all_docs(
                content_cluster_name=content_cluster_name,
                namespace=namespace,
                schema=schema,
                slices=slices,
                **kwargs,
            )

    def visit(
        self,
        content_cluster_name: str,
        schema: Optional[str] = None,
        namespace: Optional[str] = None,
        slices: int = 1,
        selection: str = "true",
        wanted_document_count: int = 500,
        slice_id: Optional[int] = None,
        **kwargs,
    ) -> Generator[Generator[VespaVisitResponse, None, None], None, None]:
        """
        Visit all documents associated with the schema and matching the selection.

        Will run each slice on a separate thread, for each slice yields the
        response for each page.

        Example usage:
            ```python
            for slice in app.visit(schema="schema_name", slices=2):
                for response in slice:
                    print(response.json)
            ```

        Args:
            content_cluster_name (str): Name of content cluster to GET from.
            schema (str): The schema that we are visiting data from.
            namespace (str, optional): The namespace that we are visiting data from.
            slices (int, optional): Number of slices to use for parallel GET.
            selection (str, optional): Selection expression to filter documents.
            wanted_document_count (int, optional): Best effort number of documents to retrieve for each request. May contain less if there are not enough documents left.
            slice_id (int, optional): Slice id to use for the visit. If None, all slices will be used.
            **kwargs (dict, optional): Additional HTTP request parameters.
                See [Vespa API documentation](https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters).

        Yields:
            Generator[Generator[Response]]: A generator of slices, each containing a generator of responses.

        Raises:
            HTTPError: If an HTTP error occurred.
        """
        with VespaSync(self, pool_connections=slices, pool_maxsize=slices) as sync_app:
            return sync_app.visit(
                content_cluster_name=content_cluster_name,
                namespace=namespace,
                schema=schema,
                slices=slices,
                selection=selection,
                wanted_document_count=wanted_document_count,
                slice_id=slice_id,
                **kwargs,
            )

    def get_data(
        self,
        schema: str,
        data_id: str,
        namespace: str = None,
        groupname: str = None,
        raise_on_not_found: Optional[bool] = False,
        **kwargs,
    ) -> VespaResponse:
        """
        Get a data point from a Vespa app.

        Args:
            data_id (str): Unique id associated with this data point.
            schema (str, optional): The schema that we are getting data from. Will attempt to infer schema name if not provided.
            namespace (str, optional): The namespace that we are getting data from. If no namespace is provided, the schema is used.
            groupname (str, optional): The groupname that we are getting data from.
            raise_on_not_found (bool, optional): Raise an exception if the data_id is not found. Default is False.
            **kwargs (dict, optional): Additional arguments to be passed to the HTTP GET request.
                See [Vespa API documentation](https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters).

        Returns:
            Response: The response of the HTTP GET request.
        """

        with VespaSync(self, pool_connections=1, pool_maxsize=1) as sync_app:
            return sync_app.get_data(
                schema=schema,
                data_id=data_id,
                namespace=namespace,
                groupname=groupname,
                raise_on_not_found=raise_on_not_found,
                **kwargs,
            )

    def update_data(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        create: bool = False,
        namespace: str = None,
        groupname: str = None,
        compress: Union[str, bool] = "auto",
        **kwargs,
    ) -> VespaResponse:
        """
        Update a data point in a Vespa app.

        Example usage:
            ```python
            vespa = Vespa(url="localhost", port=8080)

            fields = {"mystringfield": "value1", "myintfield": 42}
            response = vespa.update_data(schema="schema_name", data_id="id1", fields=fields)
            # or, with partial update, setting auto_assign=False
            fields = {"myintfield": {"increment": 1}}
            response = vespa.update_data(schema="schema_name", data_id="id1", fields=fields, auto_assign=False)
            print(response.json)
            ```

        Args:
            schema (str): The schema that we are updating data.
            data_id (str): Unique id associated with this data point.
            fields (dict): Dict containing all the fields you want to update.
            create (bool, optional): If true, updates to non-existent documents will create an empty document to update.
            auto_assign (bool, optional): Assumes `fields`-parameter is an assignment operation. If set to false, the fields parameter should be a dictionary including the update operation.
            namespace (str, optional): The namespace that we are updating data. If no namespace is provided, the schema is used.
            groupname (str, optional): The groupname that we are updating data.
            compress (Union[str, bool], optional): Whether to compress the request body. Defaults to "auto", which will compress if the body is larger than 1024 bytes.
            **kwargs (dict, optional): Additional arguments to be passed to the HTTP PUT request.
                See [Vespa API documentation](https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters).

        Returns:
            Response: The response of the HTTP PUT request.
        """

        with VespaSync(
            self, pool_connections=1, pool_maxsize=1, compress=compress
        ) as sync_app:
            return sync_app.update_data(
                schema=schema,
                data_id=data_id,
                fields=fields,
                create=create,
                namespace=namespace,
                groupname=groupname,
                **kwargs,
            )

    @property
    def application_package(self):
        """Get application package definition, if available."""
        if not self._application_package:
            raise ValueError("Application package not available.")
        else:
            return self._application_package

    def get_model_from_application_package(self, model_name: str):
        """Get model definition from application package, if available."""
        app_package = self.application_package
        model = app_package.get_model(model_id=model_name)
        return model

    def predict(self, x, model_id, function_name="output_0"):
        """
        Obtain a stateless model evaluation.

        Args:
            x (various): Input where the format depends on the task that the model is serving.
            model_id (str): The id of the model used to serve the prediction.
            function_name (str): The name of the output function to be evaluated.

        Returns:
            var: Model prediction.
        """

        model = self.get_model_from_application_package(model_id)
        encoded_tokens = model.create_url_encoded_tokens(x=x)
        with VespaSync(self) as sync_app:
            return model.parse_vespa_prediction(
                sync_app.predict(
                    model_id=model_id,
                    function_name=function_name,
                    encoded_tokens=encoded_tokens,
                )
            )

    def get_document_v1_path(
        self,
        id: str,
        schema: Optional[str] = None,
        namespace: Optional[str] = None,
        group: Optional[str] = None,
        number: Optional[str] = None,
    ) -> str:
        """
        Convert to document v1 path.

        Args:
            id (str): The id of the document.
            namespace (str): The namespace of the document.
            schema (str): The schema of the document.
            group (str): The group of the document.
            number (int): The number of the document.

        Returns:
            str: The path to the document v1 endpoint.
        """

        # Make sure `id` is properly quoted, e.g. myid#123 -> myid%23123
        id = quote(str(id))
        if not schema:
            print("schema is not provided. Attempting to infer schema name.")
            schema = self._infer_schema_name()
        if not namespace:
            namespace = schema
        if number:
            return f"/document/v1/{namespace}/{schema}/number/{number}/{id}"
        if group:
            return f"/document/v1/{namespace}/{schema}/group/{group}/{id}"
        return f"/document/v1/{namespace}/{schema}/docid/{id}"


# CustomHTTPAdapter removed - functionality moved into VespaSync helper methods


class VespaSync(object):
    def __init__(
        self,
        app: Vespa,
        pool_maxsize: int = 10,
        pool_connections: int = 10,
        compress: Union[str, bool] = "auto",
        session: Optional[Union[Session, httpr.Client]] = None,
    ) -> None:
        """
        Class to handle synchronous requests to Vespa.
        This class is intended to be used as a context manager.

        Example usage:
            ```python
            with VespaSync(app) as sync_app:
                response = sync_app.query(body=body)
            print(response)
            ```

            Can also be accessed directly through `Vespa.syncio`:
                ```python
                app = Vespa(url="localhost", port=8080)
                with app.syncio() as sync_app:
                    response = sync_app.query(body=body)
                ```

            **Reusing a client across multiple contexts** (avoids TLS handshake overhead):
                ```python
                # Get a reusable client
                client = app.get_sync_session()
                try:
                    # Use it multiple times
                    with app.syncio(session=client) as sync_app:
                        response1 = sync_app.query(body=body1)
                    with app.syncio(session=client) as sync_app:
                        response2 = sync_app.query(body=body2)
                finally:
                    # User is responsible for closing
                    client.close()
                ```

            See also `Vespa.feed_iterable` for a convenient way to feed data synchronously.

        Args:
            app (Vespa): Vespa app object.
            pool_maxsize (int, optional): The maximum number of connections to save in the pool. Defaults to 10. (Note: httpr manages connection pooling automatically)
            pool_connections (int, optional): The number of connection pools to cache. Defaults to 10. (Note: httpr manages connection pooling automatically)
            compress (Union[str, bool], optional): Whether to compress the request body. Defaults to "auto", which will compress if the body is larger than 1024 bytes.
            session (httpr.Client, optional): An externally managed httpr client to reuse. When provided, the caller is responsible for closing it. Defaults to None.
        """

        if compress not in ["auto", True, False]:
            raise ValueError(
                f"compress must be 'auto', True, or False. Got {compress} instead."
            )
        self.app = app
        self.cert = self.app.cert
        self.key = self.app.key
        self.headers = self.app.base_headers.copy()
        if self.app.auth_method == "token" and self.app.vespa_cloud_secret_token:
            # Bearer and user-agent
            self.headers.update(
                {"Authorization": f"Bearer {self.app.vespa_cloud_secret_token}"}
            )
        self.compress = compress
        self.compress_larger_than = 1024
        self.num_retries_429 = 10
        self.http_client = (
            session  # For backward compatibility, parameter is still called "session"
        )
        # Automatically determine ownership based on whether client was provided
        self._owns_client = session is None
        self._client_configured = False
        # pool_maxsize and pool_connections are kept for API compatibility but httpr manages pooling automatically

    def _prepare_mtls_cert(self) -> Optional[bytes]:
        """Prepare mTLS certificate data for httpr."""
        return _prepare_mtls_cert_data(self.cert, self.key)

    def _request_with_retry(self, method: str, url: str, json_data=None, **kwargs):
        """
        Execute HTTP request with 429 retry logic using exponential backoff.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: URL to request
            json_data: JSON data to send (will be compressed if configured)
            **kwargs: Additional arguments to pass to httpr.Client request method

        Returns:
            httpr.Response object
        """
        # Handle compression if json_data provided
        if json_data is not None:
            prepared_body, extra_headers = _prepare_request_body(
                method, json_data, None, self.compress, self.compress_larger_than
            )
            if extra_headers:
                kwargs["content"] = prepared_body
                kwargs["headers"] = {**kwargs.get("headers", {}), **extra_headers}
            else:
                kwargs["json"] = prepared_body

        for attempt in range(self.num_retries_429 + 1):
            try:
                # Make the request using httpr.Client
                response = getattr(self.http_client, method.lower())(url, **kwargs)

                if response.status_code == 429 and attempt < self.num_retries_429:
                    # Exponential backoff for 429 (same formula as CustomHTTPAdapter)
                    wait_time = 0.1 * 1.618**attempt + random.uniform(0, 1)
                    time.sleep(wait_time)
                    continue

                return response
            except (ConnectionResetError, Exception) as e:
                # Check if it's a connection/network error that should be retried
                # This includes httpr.RequestError, httpr.ConnectError, ConnectionResetError, and other network errors
                if _is_connection_error(e) and attempt < self.num_retries_429:
                    wait_time = 0.1 * 1.618**attempt + random.uniform(0, 1)
                    time.sleep(wait_time)
                elif _is_connection_error(e):
                    raise
                else:
                    # Not a connection error, re-raise immediately
                    raise

        return response

    def __enter__(self):
        self._open_http_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_http_client()

    def _open_http_client(self):
        """Create and configure httpr.Client for making HTTP requests."""
        if self.http_client is None:
            # Prepare client configuration
            client_config = {
                "headers": self.headers,
                "timeout": 120,  # Default timeout in seconds
                "follow_redirects": True,
                "http2_only": False,  # Allow HTTP/1.1 for compatibility
            }

            # Handle mTLS if cert/key are provided
            if self.cert:
                client_pem_data = self._prepare_mtls_cert()
                if client_pem_data:
                    client_config["client_pem_data"] = client_pem_data

            # Handle token authentication (already in headers)
            # httpr will use headers automatically

            self.http_client = httpr.Client(**client_config)
            self._owns_client = True
            self._client_configured = True

        return self.http_client

    def _close_http_client(self):
        """Close the httpr.Client."""
        if self.http_client is None:
            return
        if self._owns_client:
            self.http_client.close()
        self._client_configured = False

    def get_model_endpoint(self, model_id: Optional[str] = None) -> Optional[dict]:
        """Get model evaluation endpoints."""
        end_point = "{}/model-evaluation/v1/".format(self.app.end_point)
        if model_id:
            end_point = end_point + model_id
        try:
            response = self._request_with_retry("GET", end_point)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status_code": response.status_code, "message": response.text}
        except Exception as e:
            # Catch both requests.ConnectionError and httpr exceptions
            if _is_connection_error(e):
                return None
            else:
                raise
        return None

    def predict(self, model_id, function_name, encoded_tokens):
        """
        Obtain a stateless model evaluation.

        Args:
            model_id (str): The id of the model used to serve the prediction.
            function_name (str): The name of the output function to be evaluated.
            encoded_tokens (str): URL-encoded input to the model.

        Returns:
            The model prediction.
        """

        end_point = "{}/model-evaluation/v1/{}/{}/eval?{}".format(
            self.app.end_point, model_id, function_name, encoded_tokens
        )
        try:
            response = self._request_with_retry("GET", end_point)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status_code": response.status_code,
                    "body": response.json(),
                    "message": response.text,
                }
        except Exception as e:
            # Catch both requests.ConnectionError and httpr exceptions
            if _is_connection_error(e):
                return None
            else:
                raise
        return None

    def feed_data_point(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        namespace: str = None,
        groupname: str = None,
        **kwargs,
    ) -> VespaResponse:
        """
        Feed a data point to a Vespa app.

        Args:
            schema (str): The schema that we are sending data to.
            data_id (str): Unique id associated with this data point.
            fields (dict): Dict containing all the fields required by the `schema`.
            namespace (str, optional): The namespace that we are sending data to. If no namespace is provided, the schema is used.
            groupname (str, optional): The group that we are sending data to.
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.

        Returns:
            The response of the HTTP POST request.

        Raises:
            HTTPError: If one occurred.
        """

        path = self.app.get_document_v1_path(
            id=data_id, schema=schema, namespace=namespace, group=groupname
        )
        end_point = "{}{}".format(self.app.end_point, path)
        vespa_format = {"fields": fields}

        response = self._request_with_retry(
            "POST", end_point, json_data=vespa_format, params=kwargs
        )
        raise_for_status(response)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="feed",
        )

    def query(
        self,
        body: Optional[Dict] = None,
        groupname: str = None,
        streaming: bool = False,
        profile: bool = False,
        **kwargs,
    ) -> Union[VespaQueryResponse, Generator[str, None, None]]:
        """
        Send a query request to the Vespa application.

        Args:
            body (dict): Dict containing all the request parameters.
            groupname (str, optional): The groupname used in streaming search.
            streaming (bool, optional): Whether to use streaming mode (SSE). Defaults to False.
            profile (bool, optional): Add profiling parameters to the query (response may be large). Defaults to False.
            **kwargs (dict, optional): Additional valid Vespa HTTP Query API parameters. See: <https://docs.vespa.ai/en/reference/query-api-reference.html>.

        Returns:
            VespaQueryResponse when streaming=False, or a generator of decoded lines when streaming=True.

        Raises:
            HTTPError: If one occurred.
        """

        if groupname:
            kwargs["streaming.groupname"] = groupname
        if profile:
            kwargs.update(get_profiling_params())
        if streaming:
            return self._query_streaming(body, **kwargs)
        else:
            response = self._request_with_retry(
                "POST",
                self.app.search_end_point,
                json_data=body,
                params=kwargs,
                headers={"Accept": "application/cbor"},
            )
            raise_for_status(response)

            return VespaQueryResponse(
                json=response.json(),
                status_code=response.status_code,
                url=str(response.url),
            )

    def _query_streaming(
        self, body: Optional[Dict] = None, **kwargs
    ) -> Generator[str, None, None]:
        """Helper method for streaming queries to avoid generator function issues."""
        # Prepare request body with optional compression
        prepared_body, extra_headers = _prepare_request_body(
            "POST", body, None, self.compress, self.compress_larger_than
        )
        request_kwargs = {"params": kwargs}

        if extra_headers:
            request_kwargs["content"] = prepared_body
            request_kwargs["headers"] = {
                **request_kwargs.get("headers", {}),
                **extra_headers,
            }
        else:
            request_kwargs["json"] = prepared_body

        with self.http_client.stream(
            "POST", self.app.search_end_point, **request_kwargs
        ) as stream:
            for line in stream.iter_lines():
                if line:
                    yield line

    def delete_data(
        self,
        schema: str,
        data_id: str,
        namespace: str = None,
        groupname: str = None,
        **kwargs,
    ) -> VespaResponse:
        """
        Delete a data point from a Vespa app.

        Args:
            schema (str): The schema that we are deleting data from.
            data_id (str): Unique id associated with this data point.
            namespace (str, optional): The namespace that we are deleting data from.
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.

        Returns:
            Response: The response of the HTTP DELETE request.

        Raises:
            HTTPError: If one occurred.
        """

        path = self.app.get_document_v1_path(
            id=data_id, schema=schema, namespace=namespace, group=groupname
        )
        end_point = "{}{}".format(self.app.end_point, path)
        response = self._request_with_retry("DELETE", end_point, params=kwargs)
        raise_for_status(response)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="delete",
        )

    def delete_all_docs(
        self,
        content_cluster_name: str,
        schema: str,
        namespace: str = None,
        slices: int = 1,
        **kwargs,
    ) -> None:
        """
        Delete all documents associated with the schema.

        Args:
            content_cluster_name (str): Name of content cluster to GET from or visit.
            schema (str): The schema that we are deleting data from.
            namespace (str, optional): The namespace that we are deleting data from.
            slices (int, optional): Number of slices to use for parallel delete.
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.

        Returns:
            Response: The response of the HTTP DELETE request.

        Raises:
            HTTPError: If one occurred.
        """

        if not namespace:
            namespace = schema

        def delete_slice(slice_id):
            end_point = "{}/document/v1/{}/{}/docid/?cluster={}&selection=true&slices={}&sliceId={}".format(
                self.app.end_point,
                namespace,
                schema,
                content_cluster_name,
                slices,
                slice_id,
            )
            request_endpoint = end_point
            count = 0
            errors = 0
            while True:
                try:
                    count += 1
                    response = self._request_with_retry(
                        "DELETE", request_endpoint, params=kwargs
                    )
                    result = response.json()
                    if "continuation" in result:
                        request_endpoint = "{}&continuation={}".format(
                            end_point, result["continuation"]
                        )
                    else:
                        break
                except Exception as e:
                    errors += 1
                    error_rate = errors / count
                    if error_rate > 0.1:
                        raise Exception(
                            "Too many errors for slice delete requests"
                        ) from e
                    sleep(1)

        with ThreadPoolExecutor(max_workers=slices) as executor:
            executor.map(delete_slice, range(slices))

    def visit(
        self,
        content_cluster_name: str,
        schema: Optional[str] = None,
        namespace: Optional[str] = None,
        slices: int = 1,
        selection: str = "true",
        wanted_document_count: int = 500,
        slice_id: Optional[int] = None,
        **kwargs,
    ) -> Generator[Generator[VespaVisitResponse, None, None], None, None]:
        """
        Visit all documents associated with the schema and matching the selection.

        This method will run each slice on a separate thread, yielding the response for each page for each slice.

        Args:
            content_cluster_name (str): Name of content cluster to GET from.
            schema (str): The schema that we are visiting data from.
            namespace (str, optional): The namespace that we are visiting data from.
            slices (int): Number of slices to use for parallel GET.
            wanted_document_count (int, optional): Best effort number of documents to retrieve for each request. May contain fewer if there are not enough documents left.
            selection (str, optional): Selection expression to use. Defaults to "true". See: <https://docs.vespa.ai/en/reference/document-select-language.html>.
            slice_id (int, optional): Slice id to use. Defaults to -1, which means all slices.
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.

        Returns:
            generator: A generator of slices, each containing a generator of responses.

        Raises:
            HTTPError: If one occurred.
        """

        if not namespace:
            namespace = schema

        if schema:
            target = "{}/{}/docid/".format(
                namespace,
                schema,
            )
        else:
            target = ""

        end_point = "{}/document/v1/{}".format(
            self.app.end_point,
            target,
        )
        # Validate that if slice_id is provided, it's in range [0, slices)
        if slice_id is not None and slice_id >= slices:
            raise ValueError(
                f"slice_id must be in range [0, {slices - 1}]. Got {slice_id} instead."
            )

        @retry(retry=retry_if_exception_type(HTTPError), stop=stop_after_attempt(3))
        def visit_request(end_point: str, params: Dict[str, str]):
            r = self._request_with_retry("GET", end_point, params=params)
            raise_for_status(r)
            return VespaVisitResponse(
                json=r.json(), status_code=r.status_code, url=str(r.url)
            )

        def visit_slice(slice_id):
            params = {
                "cluster": content_cluster_name,
                "selection": selection,
                "wantedDocumentCount": wanted_document_count,
                "slices": slices,
                "sliceId": slice_id,
                **kwargs,
            }

            while True:
                result = visit_request(end_point, params=params)
                yield result
                if result.continuation:
                    params["continuation"] = result.continuation
                else:
                    break

        if slice_id is None:
            with ThreadPoolExecutor(max_workers=slices) as executor:
                futures = [
                    executor.submit(visit_slice, slice) for slice in range(slices)
                ]
                for future in as_completed(futures):
                    yield future.result()
        else:
            yield visit_slice(slice_id)

    def get_data(
        self,
        schema: str,
        data_id: str,
        namespace: str = None,
        groupname: str = None,
        raise_on_not_found: Optional[bool] = False,
        **kwargs,
    ) -> VespaResponse:
        """
        Get a data point from a Vespa app.

        Args:
            schema (str): The schema that we are getting data from.
            data_id (str): Unique id associated with this data point.
            namespace (str, optional): The namespace that we are getting data from.
            groupname (str, optional): The groupname used to get data.
            raise_on_not_found (bool, optional): Raise an exception if the document is not found. Default is False.
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.

        Returns:
            Response: The response of the HTTP GET request.

        Raises:
            HTTPError: If one occurred.
        """

        path = self.app.get_document_v1_path(
            id=data_id, schema=schema, namespace=namespace, group=groupname
        )
        end_point = "{}{}".format(self.app.end_point, path)

        response = self._request_with_retry("GET", end_point, params=kwargs)
        raise_for_status(response, raise_on_not_found=raise_on_not_found)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="get",
        )

    def update_data(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        create: bool = False,
        auto_assign: bool = True,
        namespace: str = None,
        groupname: str = None,
        **kwargs,
    ) -> VespaResponse:
        """
        Update a data point in a Vespa app.

        Args:
            schema (str): The schema that we are updating data in.
            data_id (str): Unique id associated with this data point.
            fields (dict): Dict containing all the fields you want to update.
            create (bool, optional): If true, updates to non-existent documents will create an empty document to update. Default is False.
            auto_assign (bool, optional): Assumes `fields`-parameter is an assignment operation. If set to False, the fields parameter should include the update operation. Default is True.
            namespace (str, optional): The namespace that we are updating data in.
            groupname (str, optional): The groupname used to update data.
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters.

        Returns:
            Response: The response of the HTTP PUT request.

        Raises:
            HTTPError: If one occurred.
        """

        path = self.app.get_document_v1_path(
            id=data_id, schema=schema, namespace=namespace, group=groupname
        )
        end_point = "{}{}?create={}".format(
            self.app.end_point, path, str(create).lower()
        )
        if auto_assign:
            vespa_format = {"fields": {k: {"assign": v} for k, v in fields.items()}}
        else:
            # Can not send 'id' in fields for partial update
            vespa_format = {"fields": {k: v for k, v in fields.items() if k != "id"}}

        response = self._request_with_retry(
            "PUT", end_point, json_data=vespa_format, params=kwargs
        )
        raise_for_status(response)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="update",
        )


class VespaAsync(object):
    def __init__(
        self,
        app: Vespa,
        connections: Optional[int] = 1,
        total_timeout: Optional[int] = None,
        timeout: Union[httpx.Timeout, int, float] = 30.0,
        compress: Union[str, bool] = "auto",
        client: Optional[Union[httpx.AsyncClient, httpr.AsyncClient]] = None,
        **kwargs,
    ) -> None:
        """
        Class to handle asynchronous HTTP connections to Vespa.

        Uses `httpr` as the async HTTP client, and HTTP/2 by default.
        This class is intended to be used as a context manager.

        **Basic usage**:
            ```python
            async with VespaAsync(app) as async_app:
                response = await async_app.query(
                    body={"yql": "select * from sources * where title contains 'music';"}
                )
            ```

        **Passing custom timeout**:
            ```python
            # Simple timeout in seconds
            async with VespaAsync(app, timeout=60) as async_app:
                response = await async_app.query(
                    body={"yql": "select * from sources * where title contains 'music';"}
                )
            ```

        **Using additional kwargs (e.g., proxies)**:
            ```python
            proxies = "http://localhost:8080"

            async with VespaAsync(app, proxies=proxies) as async_app:
                response = await async_app.query(
                    body={"yql": "select * from sources * where title contains 'music';"}
                )
            ```

        **Accessing via `Vespa.asyncio`**:
            ```python
            app = Vespa(url="localhost", port=8080)
            async with app.asyncio(timeout=timeout, limits=limits) as async_app:
                response = await async_app.query(
                    body={"yql": "select * from sources * where title contains 'music';"}
                )
            ```

        **Reusing a client across multiple contexts** (avoids TLS handshake overhead):
            ```python
            # Get a reusable client
            client = app.get_async_session()
            try:
                # Use it multiple times
                async with app.asyncio(client=client) as async_app:
                    response1 = await async_app.query(body=body1)
                async with app.asyncio(client=client) as async_app:
                    response2 = await async_app.query(body=body2)
            finally:
                # User is responsible for closing
                await client.aclose()
            ```

        See also `Vespa.feed_async_iterable` for a convenient interface to async data feeding.

        Args:
            app (Vespa): Vespa application object.
            connections (Optional[int], optional): Number of connections. Defaults to 1 as HTTP/2 is multiplexed. (Note: httpr manages connection pooling automatically)
            total_timeout (int, optional): **Deprecated**. Will be ignored and removed in future versions.
            timeout (float, optional): Timeout in seconds for the `httpr.AsyncClient`. Defaults to 30 seconds.
            compress (Union[str, bool], optional): Whether to compress the request body. Defaults to "auto", which will compress if the body is larger than 1024 bytes.
            client (httpr.AsyncClient, optional): An externally managed async client to reuse. When provided, the caller is responsible for closing it. Defaults to None.
            **kwargs: Additional arguments to be passed to the `httpr.AsyncClient`.

        Note:
            - `timeout` is specified in seconds as a float
            - httpr manages connection pooling and HTTP/2 automatically
        """
        self.app = app
        self.httpr_client = client  # Renamed from httpx_client
        # Automatically determine ownership based on whether client was provided
        self._owns_client = client is None
        self.connections = connections
        self.total_timeout = total_timeout
        if self.total_timeout is not None:
            # issue DeprecationWarning
            warnings.warn(
                "total_timeout is deprecated, will be ignored and will be removed in future versions.",
                category=DeprecationWarning,
            )

        # Convert timeout to float if needed (for backward compatibility with httpx.Timeout)
        if isinstance(timeout, httpx.Timeout):
            # Extract read timeout from httpx.Timeout for backward compatibility
            self.timeout = timeout.read if timeout.read else 30.0
            warnings.warn(
                "Passing httpx.Timeout is deprecated. Please pass a float (seconds) instead.",
                category=DeprecationWarning,
            )
        elif isinstance(timeout, (int, float)):
            self.timeout = float(timeout)
        else:
            self.timeout = 30.0

        self.kwargs = kwargs
        self.headers = self.app.base_headers.copy()

        # Initialize compression settings
        if compress not in ["auto", True, False]:
            raise ValueError(
                f"compress must be 'auto', True, or False. Got {compress} instead."
            )
        self.compress = compress
        self.compress_larger_than = 1024

        # Note: httpr manages connection pooling automatically, limits parameter is kept for compatibility but not used
        if "limits" in kwargs:
            warnings.warn(
                "The 'limits' parameter is no longer used with httpr. Connection pooling is managed automatically.",
                category=DeprecationWarning,
            )

        if self.app.auth_method == "token" and self.app.vespa_cloud_secret_token:
            # Bearer and user-agent
            self.headers.update(
                {"Authorization": f"Bearer {self.app.vespa_cloud_secret_token}"}
            )

    def _prepare_mtls_cert(self) -> Optional[bytes]:
        """Prepare mTLS certificate data for httpr."""
        return _prepare_mtls_cert_data(self.app.cert, self.app.key)

    async def __aenter__(self):
        self._open_httpr_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_httpr_client()

    def _open_httpr_client(self):
        """Create and configure httpr.AsyncClient for making async HTTP requests."""
        if self.httpr_client is not None:
            return self.httpr_client

        # Prepare client configuration
        client_config = {
            "headers": self.headers,
            "timeout": self.timeout,
            "follow_redirects": True,
            "http2_only": True,  # HTTP/2 only for async (matches httpx default)
        }

        # Handle mTLS if cert/key are provided
        if self.app.cert:
            client_pem_data = self._prepare_mtls_cert()
            if client_pem_data:
                client_config["client_pem_data"] = client_pem_data

        # Handle token authentication (already in headers)
        # httpr will use headers automatically

        # Handle additional kwargs (like proxies)
        # Filter out httpx-specific kwargs that don't apply to httpr
        filtered_kwargs = {
            k: v for k, v in self.kwargs.items() if k not in ["limits", "verify"]
        }
        client_config.update(filtered_kwargs)

        self.httpr_client = httpr.AsyncClient(**client_config)
        self._owns_client = True
        return self.httpr_client

    async def _close_httpr_client(self):
        """Close the httpr.AsyncClient."""
        if self.httpr_client is None:
            return
        if self._owns_client:
            await self.httpr_client.aclose()

    async def _make_request(
        self,
        method: str,
        url: str,
        json_data=None,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs,
    ):
        """
        Execute async HTTP request with optional compression and rate-limiting.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: URL to request
            json_data: JSON data to send (will be compressed if configured)
            semaphore: Optional semaphore for rate-limiting
            **kwargs: Additional arguments to pass to httpr.AsyncClient request method

        Returns:
            httpr.Response object
        """
        if json_data is not None:
            prepared_body, extra_headers = _prepare_request_body(
                method, json_data, None, self.compress, self.compress_larger_than
            )
            if extra_headers:
                kwargs["content"] = prepared_body
                kwargs["headers"] = {**kwargs.get("headers", {}), **extra_headers}
            else:
                kwargs["json"] = prepared_body

        http_method = getattr(self.httpr_client, method.lower())

        if semaphore:
            async with semaphore:
                return await http_method(url, **kwargs)
        return await http_method(url, **kwargs)

    async def _wait(f, args, **kwargs):
        tasks = [asyncio.create_task(f(*arg, **kwargs)) for arg in args]
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        return [result for result in map(lambda task: task.result(), tasks)]

    def callback_docv1(state: RetryCallState) -> VespaResponse:
        if state.outcome.failed:
            raise state.outcome.exception()
        return state.outcome.result()

    @retry(
        wait=wait_random_exponential(multiplier=1.5, max=60), stop=stop_after_attempt(5)
    )
    async def query(
        self,
        body: Optional[Dict] = None,
        groupname: str = None,
        profile: bool = False,
        **kwargs,
    ) -> VespaQueryResponse:
        """
        Send a query request to the Vespa application.

        Args:
            body (dict): Dict containing all the request parameters.
            groupname (str, optional): The groupname used in streaming search.
            profile (bool, optional): Add profiling parameters to the query (response may be large). Defaults to False.
            **kwargs (dict, optional): Additional valid Vespa HTTP Query API parameters.

        Returns:
            VespaQueryResponse: The response from the query.
        """
        if groupname:
            kwargs["streaming.groupname"] = groupname
        if profile:
            kwargs.update(get_profiling_params())

        r = await self._make_request(
            "POST",
            self.app.search_end_point,
            json_data=body,
            params=kwargs,
            headers={"Accept": "application/cbor"},
        )

        return VespaQueryResponse(
            json=r.json(), status_code=r.status_code, url=str(r.url)
        )

    @retry(
        wait=wait_exponential(multiplier=1),
        retry=retry_any(
            retry_if_exception(lambda x: True),
            retry_if_result(lambda x: x.get_status_code() == 503),
        ),
        stop=stop_after_attempt(3),
        retry_error_callback=callback_docv1,
    )
    @retry(
        wait=wait_random_exponential(multiplier=1, max=3),
        retry=retry_if_result(lambda x: x.get_status_code() == 429),
    )
    async def feed_data_point(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        namespace: Optional[str] = None,
        groupname: Optional[str] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs,
    ) -> VespaResponse:
        path = self.app.get_document_v1_path(
            id=data_id, schema=schema, namespace=namespace, group=groupname
        )
        end_point = "{}{}".format(self.app.end_point, path)
        vespa_format = {"fields": fields}

        response = await self._make_request(
            "POST",
            end_point,
            json_data=vespa_format,
            semaphore=semaphore,
            params=kwargs,
        )

        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="feed",
        )

    @retry(
        wait=wait_exponential(multiplier=1),
        retry=retry_any(
            retry_if_exception(lambda x: True),
            retry_if_result(lambda x: x.get_status_code() == 503),
        ),
        stop=stop_after_attempt(3),
        retry_error_callback=callback_docv1,
    )
    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_result(lambda x: x.get_status_code() == 429),
    )
    async def delete_data(
        self,
        schema: str,
        data_id: str,
        namespace: str = None,
        groupname: str = None,
        semaphore: asyncio.Semaphore = None,
        **kwargs,
    ) -> VespaResponse:
        path = self.app.get_document_v1_path(
            id=data_id, schema=schema, namespace=namespace, group=groupname
        )
        end_point = "{}{}".format(self.app.end_point, path)
        if semaphore:
            async with semaphore:
                response = await self.httpr_client.delete(end_point, params=kwargs)
        else:
            response = await self.httpr_client.delete(end_point, params=kwargs)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="delete",
        )

    @retry(
        wait=wait_exponential(multiplier=1),
        retry=retry_any(
            retry_if_exception(lambda x: True),
            retry_if_result(lambda x: x.get_status_code() == 503),
        ),
        stop=stop_after_attempt(3),
        retry_error_callback=callback_docv1,
    )
    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_result(lambda x: x.get_status_code() == 429),
    )
    async def get_data(
        self,
        schema: str,
        data_id: str,
        namespace: str = None,
        groupname: str = None,
        semaphore: asyncio.Semaphore = None,
        **kwargs,
    ) -> VespaResponse:
        path = self.app.get_document_v1_path(
            id=data_id, schema=schema, namespace=namespace, group=groupname
        )
        end_point = "{}{}".format(self.app.end_point, path)
        if semaphore:
            async with semaphore:
                response = await self.httpr_client.get(end_point, params=kwargs)
        else:
            response = await self.httpr_client.get(end_point, params=kwargs)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="get",
        )

    @retry(
        wait=wait_exponential(multiplier=1),
        retry=retry_any(
            retry_if_exception(lambda x: True),
            retry_if_result(lambda x: x.get_status_code() == 503),
        ),
        stop=stop_after_attempt(3),
        retry_error_callback=callback_docv1,
    )
    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_result(lambda x: x.get_status_code() == 429),
    )
    async def update_data(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        create: bool = False,
        auto_assign: bool = True,
        namespace: str = None,
        groupname: str = None,
        semaphore: asyncio.Semaphore = None,
        **kwargs,
    ) -> VespaResponse:
        path = self.app.get_document_v1_path(
            id=data_id, schema=schema, namespace=namespace, group=groupname
        )
        end_point = "{}{}".format(self.app.end_point, path)
        if create:
            kwargs["create"] = str(create).lower()
        if auto_assign:
            vespa_format = {"fields": {k: {"assign": v} for k, v in fields.items()}}
        else:
            # Can not send 'id' in fields for partial update
            vespa_format = {"fields": {k: v for k, v in fields.items() if k != "id"}}

        response = await self._make_request(
            "PUT",
            end_point,
            json_data=vespa_format,
            semaphore=semaphore,
            params=kwargs,
        )
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="update",
        )
