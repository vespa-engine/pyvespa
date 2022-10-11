# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import ssl
import aiohttp
import asyncio
import concurrent.futures
from collections import Counter
from typing import Optional, Dict, List, IO, Union

import requests
from pandas import DataFrame
from requests import Session
from requests.models import Response
from requests.exceptions import ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tenacity import retry, wait_exponential, stop_after_attempt
from time import sleep

from vespa.io import VespaQueryResponse, VespaResponse
from vespa.package import ApplicationPackage

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["POST", "GET", "DELETE", "PUT"],
)


def parse_feed_df(df: DataFrame, include_id, id_field="id"):
    """
    Convert a df into batch format for feeding

    :param df: DataFrame with the following required columns ["id"]. Additional columns are assumed to be fields.
    :param include_id: Include id on the fields to be fed.
    :param id_field: Name of the column containing the id field.
    :return: List of Dict containing 'id' and 'fields'.
    """
    required_columns = [id_field]
    assert all(
        [x in list(df.columns) for x in required_columns]
    ), "DataFrame needs at least the following columns: {}".format(required_columns)
    records = df.to_dict(orient="records")
    batch = [
        {
            "id": record[id_field],
            "fields": record
            if include_id
            else {k: v for k, v in record.items() if k not in [id_field]},
        }
        for record in records
    ]
    return batch


class Vespa(object):
    def __init__(
        self,
        url: str,
        port: Optional[int] = None,
        deployment_message: Optional[List[str]] = None,
        cert: Optional[str] = None,
        key: Optional[str] = None,
        output_file: IO = sys.stdout,
        application_package: Optional[ApplicationPackage] = None,
    ) -> None:
        """
        Establish a connection with an existing Vespa application.

        :param url: Vespa instance URL.
        :param port: Vespa instance port.
        :param deployment_message: Message returned by Vespa engine after deployment. Used internally by deploy methods.
        :param cert: Path to certificate and key file in case the 'key' parameter is none. If 'key' is not None, this
            should be the path of the certificate file.
        :param key: Path to the key file.
        :param output_file: Output file to write output messages.
        :param application_package: Application package definition used to deploy the application.

        >>> Vespa(url = "https://cord19.vespa.ai")  # doctest: +SKIP

        >>> Vespa(url = "http://localhost", port = 8080)
        Vespa(http://localhost, 8080)

        >>> Vespa(url = "https://api.vespa-external.aws.oath.cloud", port = 4443, cert = "/path/to/cert-and-key.pem")  # doctest: +SKIP

        """
        self.output_file = output_file
        self.url = url
        self.port = port
        self.deployment_message = deployment_message
        self.cert = cert
        self.key = key
        self._application_package = application_package

        if port is None:
            self.end_point = self.url
        else:
            self.end_point = str(url).rstrip("/") + ":" + str(port)
        self.search_end_point = self.end_point + "/search/"

    def asyncio(
        self, connections: Optional[int] = 100, total_timeout: int = 10
    ) -> "VespaAsync":
        """
        Access Vespa asynchronous connection layer

        :param connections: Number of allowed concurrent connections
        :param total_timeout: Total timeout in secs.
        :return: Instance of Vespa asynchronous layer.
        """
        return VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        )

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
                return_value = future.result()
                return return_value
        except RuntimeError:
            return asyncio.run(coro)

    def http(self, pool_maxsize: int = 10):
        return VespaSync(app=self, pool_maxsize=pool_maxsize)

    def __repr__(self):
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

    def wait_for_application_up(self, max_wait):
        """
        Wait for application ready.

        :param max_wait: Seconds to wait for the application endpoint
        :return:
        """
        try_interval = 5
        waited = 0
        while not self.get_application_status() and (waited < max_wait):
            print(
                "Waiting for application status, {0}/{1} seconds...".format(
                    waited, max_wait
                ),
                file=self.output_file,
            )
            sleep(try_interval)
            waited += try_interval
        if waited >= max_wait:
            raise RuntimeError(
                "Application did not start, waited for {0} seconds.".format(max_wait)
            )

    def get_application_status(self) -> Optional[Response]:
        """
        Get application status.

        :return:
        """
        endpoint = "{}/ApplicationStatus".format(self.end_point)
        try:
            if self.key:
                response = requests.get(endpoint, cert=(self.cert, self.key))
            else:
                response = requests.get(endpoint, cert=self.cert)
        except ConnectionError:
            response = None
        return response

    def get_model_endpoint(self, model_id: Optional[str] = None) -> Optional[Response]:
        """Get model evaluation endpoints."""

        with VespaSync(self) as sync_app:
            return sync_app.get_model_endpoint(model_id=model_id)

    def query(
        self,
        body: Optional[Dict] = None,
    ) -> VespaQueryResponse:
        """
        Send a query request to the Vespa application.

        Send 'body' containing all the request parameters.

        :param body: Dict containing all the request parameters.
        :return: The response from the Vespa application.
        """
        with VespaSync(self) as sync_app:
            return sync_app.query(
                body=body,
            )

    def _query_batch_sync(
        self,
        body_batch: Optional[List[Dict]],
    ):
        return [self.query(body=body) for body in body_batch]

    async def _query_batch_async(
        self,
        body_batch: Optional[List[Dict]],
        connections,
        total_timeout,
        **kwargs,
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.query_batch(
                body_batch=body_batch,
                **kwargs,
            )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    def query_batch(
        self,
        body_batch: Optional[List[Dict]] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
        **kwargs,
    ):
        """
        Send queries in batch to a Vespa app.

        :param body_batch: A list of dict containing all the request parameters. Set to None if using 'query_batch'.
            of the field to use to recall and the second element is a list of the values to be recalled.
        :param asynchronous: Set True to send data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :param kwargs: Additional parameters to be sent along the request.
        :return: List of HTTP POST responses
        """
        if asynchronous:
            coro = self._query_batch_async(
                body_batch=body_batch,
                connections=connections,
                total_timeout=total_timeout,
                **kwargs,
            )
            return Vespa._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._query_batch_sync(
                body_batch=body_batch,
            )

    def feed_data_point(
        self, schema: str, data_id: str, fields: Dict, namespace: str = None
    ) -> VespaResponse:
        """
        Feed a data point to a Vespa app.

        :param schema: The schema that we are sending data to.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields required by the `schema`.
        :param namespace: The namespace that we are sending data to.
        :return: Response of the HTTP POST request.
        """
        if not namespace:
            namespace = schema

        with VespaSync(app=self) as sync_app:
            return sync_app.feed_data_point(
                schema=schema, data_id=data_id, fields=fields, namespace=namespace
            )

    def _feed_batch_sync(
        self, schema: str, batch: List[Dict], namespace: str
    ) -> List[VespaResponse]:
        return [
            self.feed_data_point(
                schema, data_point["id"], data_point["fields"], namespace
            )
            for data_point in batch
        ]

    async def _feed_batch_async(
        self, schema: str, batch: List[Dict], connections, total_timeout, namespace: str
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.feed_batch(
                schema=schema, batch=batch, namespace=namespace
            )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    def _feed_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
        namespace: Optional[str] = None,
    ):
        """
        Feed a batch of data to a Vespa app.

        :param batch: A list of dict containing the keys 'id' and 'fields' to be used in the :func:`feed_data_point`.
        :param schema: The schema that we are sending data to. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to send data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :param namespace: The namespace that we are sending data to. If no namespace is provided the schema is used.
        :return: List of HTTP POST responses
        """
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        if not namespace:
            namespace = schema

        if asynchronous:
            coro = self._feed_batch_async(
                schema=schema,
                batch=batch,
                connections=connections,
                total_timeout=total_timeout,
                namespace=namespace,
            )
            return Vespa._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._feed_batch_sync(
                schema=schema, batch=batch, namespace=namespace
            )

    def feed_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
        namespace: Optional[str] = None,
        batch_size=1000,
    ):
        """
        Feed a batch of data to a Vespa app.

        :param batch: A list of dict containing the keys 'id' and 'fields' to be used in the :func:`feed_data_point`.
        :param schema: The schema that we are sending data to. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to send data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :param namespace: The namespace that we are sending data to. If no namespace is provided the schema is used.
        :param batch_size: The number of documents to feed per batch.
        :return: List of HTTP POST responses
        """
        mini_batches = [
            batch[i : i + batch_size] for i in range(0, len(batch), batch_size)
        ]
        batch_http_responses = []
        for idx, mini_batch in enumerate(mini_batches):
            feed_results = self._feed_batch(
                batch=mini_batch,
                schema=schema,
                asynchronous=asynchronous,
                connections=connections,
                total_timeout=total_timeout,
                namespace=namespace,
            )
            batch_http_responses.extend(feed_results)
            status_code_summary = Counter([x.status_code for x in feed_results])
            print(
                "Successful documents fed: {}/{}.\nBatch progress: {}/{}.".format(
                    status_code_summary[200],
                    len(mini_batch),
                    idx + 1,
                    len(mini_batches),
                )
            )
        return batch_http_responses

    def feed_df(self, df: DataFrame, include_id: bool = True, id_field="id", **kwargs):
        """
        Feed data contained in a DataFrame.

        :param df: A DataFrame containing a required 'id' column and the remaining fields to be fed.
        :param include_id: Include id on the fields to be fed. Default to True.
        :param id_field: Name of the column containing the id field.
        :param kwargs: Additional parameters are passed to :func:`feed_batch`.
        :return: List of HTTP POST responses
        """
        batch = parse_feed_df(df=df, include_id=include_id, id_field=id_field)
        return self.feed_batch(batch=batch, **kwargs)

    def delete_data(
        self, schema: str, data_id: str, namespace: str = None
    ) -> VespaResponse:
        """
        Delete a data point from a Vespa app.

        :param schema: The schema that we are deleting data from.
        :param data_id: Unique id associated with this data point.
        :param namespace: The namespace that we are deleting data from. If no namespace is provided the schema is used.
        :return: Response of the HTTP DELETE request.
        """
        if not namespace:
            namespace = schema

        with VespaSync(self) as sync_app:
            return sync_app.delete_data(
                schema=schema, data_id=data_id, namespace=namespace
            )

    def _delete_batch_sync(self, schema: str, batch: List[Dict], namespace: str):
        return [
            self.delete_data(schema, data_point["id"], namespace)
            for data_point in batch
        ]

    async def _delete_batch_async(
        self, schema: str, batch: List[Dict], connections, total_timeout, namespace: str
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.delete_batch(
                schema=schema, batch=batch, namespace=namespace
            )

    def delete_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
        namespace: Optional[str] = None,
    ):
        """
        Delete a batch of data from a Vespa app.

        :param batch: A list of dict containing the key 'id'.
        :param schema: The schema that we are deleting data from. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to get data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :param namespace: The namespace that we are deleting data from. If no namespace is provided the schema is used.
        :return: List of HTTP POST responses
        """
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        if not namespace:
            namespace = schema

        if asynchronous:
            coro = self._delete_batch_async(
                schema=schema,
                batch=batch,
                connections=connections,
                total_timeout=total_timeout,
                namespace=namespace,
            )
            return Vespa._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._delete_batch_sync(
                schema=schema, batch=batch, namespace=namespace
            )

    def delete_all_docs(
        self, content_cluster_name: str, schema: str, namespace: str = None
    ) -> Response:
        """
        Delete all documents associated with the schema

        :param content_cluster_name: Name of content cluster to GET from, or visit.
        :param schema: The schema that we are deleting data from.
        :param namespace: The  namespace that we are deleting data from. If no namespace is provided the schema is used.
        :return: Response of the HTTP DELETE request.
        """
        if not namespace:
            namespace = schema

        with VespaSync(self) as sync_app:
            return sync_app.delete_all_docs(
                content_cluster_name=content_cluster_name,
                namespace=namespace,
                schema=schema,
            )

    def get_data(
        self, schema: str, data_id: str, namespace: str = None
    ) -> VespaResponse:
        """
        Get a data point from a Vespa app.

        :param schema: The schema that we are getting data from.
        :param data_id: Unique id associated with this data point.
        :param namespace: The namespace that we are getting data from. If no namespace is provided the schema is used.
        :return: Response of the HTTP GET request.
        """
        if not namespace:
            namespace = schema

        with VespaSync(self) as sync_app:
            return sync_app.get_data(
                schema=schema, data_id=data_id, namespace=namespace
            )

    def _get_batch_sync(self, schema: str, batch: List[Dict], namespace: str):
        return [
            self.get_data(schema, data_point["id"], namespace) for data_point in batch
        ]

    async def _get_batch_async(
        self, schema: str, batch: List[Dict], connections, total_timeout, namespace: str
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.get_batch(
                schema=schema, batch=batch, namespace=namespace
            )

    def get_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
        namespace: Optional[str] = None,
    ):
        """
        Get a batch of data from a Vespa app.

        :param batch: A list of dict containing the key 'id'.
        :param schema: The schema that we are getting data from. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to get data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :param namespace: The namespace that we are getting data from. If no namespace is provided the schema is used.
        :return: List of HTTP POST responses
        """
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        if not namespace:
            namespace = schema

        if asynchronous:
            coro = self._get_batch_async(
                schema=schema,
                batch=batch,
                connections=connections,
                total_timeout=total_timeout,
                namespace=namespace,
            )
            return Vespa._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._get_batch_sync(schema=schema, batch=batch, namespace=namespace)

    def update_data(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        create: bool = False,
        namespace: str = None,
    ) -> VespaResponse:
        """
        Update a data point in a Vespa app.

        :param schema: The schema that we are updating data.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields you want to update.
        :param create: If true, updates to non-existent documents will create an empty document to update
        :param namespace: The namespace that we are updating data. If no namespace is provided the schema is used.
        :return: Response of the HTTP PUT request.
        """
        if not namespace:
            namespace = schema

        with VespaSync(self) as sync_app:
            return sync_app.update_data(
                schema=schema,
                data_id=data_id,
                fields=fields,
                create=create,
                namespace=namespace,
            )

    def _update_batch_sync(self, schema: str, batch: List[Dict], namespace: str):
        return [
            self.update_data(
                schema,
                data_point["id"],
                data_point["fields"],
                data_point.get("create", False),
                namespace,
            )
            for data_point in batch
        ]

    async def _update_batch_async(
        self, schema: str, batch: List[Dict], connections, total_timeout, namespace: str
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.update_batch(
                schema=schema, batch=batch, namespace=namespace
            )

    def update_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
        namespace: Optional[str] = None,
    ):
        """
        Update a batch of data in a Vespa app.

        :param batch: A list of dict containing the keys 'id', 'fields' and 'create' (create defaults to False).
        :param schema: The schema that we are updating data to. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to update data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :param namespace: The namespace that we are updating data. If no namespace is provided the schema is used.
        :return: List of HTTP POST responses
        """
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        if not namespace:
            namespace = schema

        if asynchronous:
            coro = self._update_batch_async(
                schema=schema,
                batch=batch,
                connections=connections,
                total_timeout=total_timeout,
                namespace=namespace,
            )
            return Vespa._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._update_batch_sync(
                schema=schema, batch=batch, namespace=namespace
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

        :param x: Input where the format depends on the task that the model is serving.
        :param model_id: The id of the model used to serve the prediction.
        :param function_name: The name of the output function to be evaluated.
        :return: Model prediction.
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


class VespaSync(object):
    def __init__(self, app: Vespa, pool_maxsize: int = 10) -> None:
        self.app = app
        if self.app.key:
            self.cert = (self.app.cert, self.app.key)
        else:
            self.cert = self.app.cert
        self.http_session = None
        self.adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_maxsize=pool_maxsize
        )

    def __enter__(self):
        self._open_http_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_http_session()

    def _open_http_session(self):
        if self.http_session is not None:
            return

        self.http_session = Session()
        self.http_session.mount("https://", self.adapter)
        self.http_session.mount("http://", self.adapter)
        return self.http_session

    def _close_http_session(self):
        if self.http_session is None:
            return
        self.http_session.close()

    def get_model_endpoint(self, model_id: Optional[str] = None) -> Optional[dict]:
        """Get model evaluation endpoints."""
        end_point = "{}/model-evaluation/v1/".format(self.app.end_point)
        if model_id:
            end_point = end_point + model_id
        try:
            response = self.http_session.get(end_point, cert=self.cert)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status_code": response.status_code, "message": response.reason}
        except ConnectionError:
            response = None
        return response

    def predict(self, model_id, function_name, encoded_tokens):
        """
        Obtain a stateless model evaluation.

        :param model_id: The id of the model used to serve the prediction.
        :param function_name: The name of the output function to be evaluated.
        :param encoded_tokens: URL-encoded input to the model
        :return: Model prediction.
        """
        end_point = "{}/model-evaluation/v1/{}/{}/eval?{}".format(
            self.app.end_point, model_id, function_name, encoded_tokens
        )
        try:
            response = self.http_session.get(end_point, cert=self.cert)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status_code": response.status_code, "message": response.reason}
        except ConnectionError:
            response = None
        return response

    def feed_data_point(
        self, schema: str, data_id: str, fields: Dict, namespace: str = None
    ) -> VespaResponse:
        """
        Feed a data point to a Vespa app.

        :param schema: The schema that we are sending data to.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields required by the `schema`.
        :param namespace: The namespace that we are sending data to. If no namespace is provided the schema is used.
        :return: Response of the HTTP POST request.
        """

        if not namespace:
            namespace = schema

        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, namespace, schema, str(data_id)
        )
        vespa_format = {"fields": fields}
        response = self.http_session.post(end_point, json=vespa_format, cert=self.cert)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="feed",
        )

    def query(
        self,
        body: Optional[Dict] = None,
    ) -> VespaQueryResponse:
        """
        Send a query request to the Vespa application.

        Send 'body' containing all the request parameters.

        :param body: Dict containing all the request parameters.
        :return: Either the request body if debug_request is True or the result from the Vespa application
        """
        r = self.http_session.post(self.app.search_end_point, json=body, cert=self.cert)
        return VespaQueryResponse(
            json=r.json(), status_code=r.status_code, url=str(r.url)
        )

    def delete_data(
        self, schema: str, data_id: str, namespace: str = None
    ) -> VespaResponse:
        """
        Delete a data point from a Vespa app.

        :param schema: The schema that we are deleting data from.
        :param data_id: Unique id associated with this data point.
        :param namespace: The namespace that we are deleting data from.
        :return: Response of the HTTP DELETE request.
        """
        if not namespace:
            namespace = schema

        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, namespace, schema, str(data_id)
        )
        response = self.http_session.delete(end_point, cert=self.cert)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="delete",
        )

    def delete_all_docs(
        self, content_cluster_name: str, schema: str, namespace: str = None
    ) -> Response:
        """
        Delete all documents associated with the schema

        :param content_cluster_name: Name of content cluster to GET from, or visit.
        :param schema: The schema that we are deleting data from.
        :param namespace: The namespace that we are deleting data from.
        :return: Response of the HTTP DELETE request.
        """
        if not namespace:
            namespace = schema

        end_point = "{}/document/v1/{}/{}/docid/?cluster={}&selection=true".format(
            self.app.end_point, namespace, schema, content_cluster_name
        )
        response = self.http_session.delete(end_point, cert=self.cert)
        return response

    def get_data(
        self, schema: str, data_id: str, namespace: str = None
    ) -> VespaResponse:
        """
        Get a data point from a Vespa app.

        :param schema: The schema that we are getting data from.
        :param data_id: Unique id associated with this data point.
        :param namespace: The namespace that we are getting data from.
        :return: Response of the HTTP GET request.
        """
        if not namespace:
            namespace = schema

        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, namespace, schema, str(data_id)
        )
        response = self.http_session.get(end_point, cert=self.cert)
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
        namespace: str = None,
    ) -> VespaResponse:
        """
        Update a data point in a Vespa app.

        :param schema: The schema that we are updating data.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields you want to update.
        :param create: If true, updates to non-existent documents will create an empty document to update
        :param namespace: The namespace that we are updating data.
        :return: Response of the HTTP PUT request.
        """
        if not namespace:
            namespace = schema

        end_point = "{}/document/v1/{}/{}/docid/{}?create={}".format(
            self.app.end_point, namespace, schema, str(data_id), str(create).lower()
        )
        vespa_format = {"fields": {k: {"assign": v} for k, v in fields.items()}}
        response = self.http_session.put(end_point, json=vespa_format, cert=self.cert)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="update",
        )


class VespaAsync(object):
    def __init__(
        self, app: Vespa, connections: Optional[int] = 100, total_timeout: int = 10
    ) -> None:
        self.app = app
        self.aiohttp_session = None
        self.connections = connections
        self.total_timeout = total_timeout

    async def __aenter__(self):
        await self._open_aiohttp_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_aiohttp_session()

    async def _open_aiohttp_session(self):
        if self.aiohttp_session is not None and not self.aiohttp_session.closed:
            return
        sslcontext = False
        if self.app.cert is not None:
            sslcontext = ssl.create_default_context()
            sslcontext.load_cert_chain(self.app.cert, self.app.key)
        conn = aiohttp.TCPConnector(ssl=sslcontext, limit=self.connections)
        self.aiohttp_session = aiohttp.ClientSession(
            connector=conn, timeout=aiohttp.ClientTimeout(total=self.total_timeout)
        )
        return self.aiohttp_session

    async def _close_aiohttp_session(self):
        if self.aiohttp_session is None:
            return
        return await self.aiohttp_session.close()

    @staticmethod
    async def _wait(f, args, **kwargs):
        tasks = [asyncio.create_task(f(*arg, **kwargs)) for arg in args]
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        return [result for result in map(lambda task: task.result(), tasks)]

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def query(
        self,
        body: Optional[Dict] = None,
    ):
        r = await self.aiohttp_session.post(self.app.search_end_point, json=body)
        return VespaQueryResponse(
            json=await r.json(), status_code=r.status, url=str(r.url)
        )

    async def _query_semaphore(
        self,
        body: Optional[Dict],
        semaphore: asyncio.Semaphore,
    ):
        async with semaphore:
            return await self.query(body=body)

    async def query_batch(
        self,
        body_batch: Optional[List[Dict]],
        **kwargs,
    ):
        sem = asyncio.Semaphore(self.connections)
        return await VespaAsync._wait(
            self._query_semaphore,
            [(body, sem) for body in body_batch],
            **kwargs,
        )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def feed_data_point(
        self, schema: str, data_id: str, fields: Dict, namespace: str = None
    ) -> VespaResponse:
        if not namespace:
            namespace = schema
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, namespace, schema, str(data_id)
        )
        vespa_format = {"fields": fields}
        response = await self.aiohttp_session.post(end_point, json=vespa_format)
        return VespaResponse(
            json=await response.json(),
            status_code=response.status,
            url=str(response.url),
            operation_type="feed",
        )

    async def _feed_data_point_semaphore(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        semaphore: asyncio.Semaphore,
        namespace: str = None,
    ):
        if not namespace:
            namespace = schema

        async with semaphore:
            return await self.feed_data_point(
                schema=schema, data_id=data_id, fields=fields, namespace=namespace
            )

    async def feed_batch(self, schema: str, batch: List[Dict], namespace=None):
        if not namespace:
            namespace = schema
        sem = asyncio.Semaphore(self.connections)
        return await VespaAsync._wait(
            self._feed_data_point_semaphore,
            [
                (schema, data_point["id"], data_point["fields"], sem, namespace)
                for data_point in batch
            ],
        )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def delete_data(
        self, schema: str, data_id: str, namespace: str = None
    ) -> VespaResponse:
        if not namespace:
            namespace = schema

        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, namespace, schema, str(data_id)
        )
        response = await self.aiohttp_session.delete(end_point)
        return VespaResponse(
            json=await response.json(),
            status_code=response.status,
            url=str(response.url),
            operation_type="delete",
        )

    async def _delete_data_semaphore(
        self,
        schema: str,
        data_id: str,
        semaphore: asyncio.Semaphore,
        namespace: str = None,
    ):
        if not namespace:
            namespace = schema

        async with semaphore:
            return await self.delete_data(
                schema=schema, data_id=data_id, namespace=namespace
            )

    async def delete_batch(self, schema: str, batch: List[Dict], namespace: str = None):
        sem = asyncio.Semaphore(self.connections)
        if not namespace:
            namespace = schema
        return await VespaAsync._wait(
            self._delete_data_semaphore,
            [(schema, data_point["id"], sem, namespace) for data_point in batch],
        )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def get_data(
        self, schema: str, data_id: str, namespace: str = None
    ) -> VespaResponse:
        if not namespace:
            namespace = schema
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, namespace, schema, str(data_id)
        )
        response = await self.aiohttp_session.get(end_point)
        return VespaResponse(
            json=await response.json(),
            status_code=response.status,
            url=str(response.url),
            operation_type="get",
        )

    async def _get_data_semaphore(
        self,
        schema: str,
        data_id: str,
        semaphore: asyncio.Semaphore,
        namespace: str = None,
    ):
        if not namespace:
            namespace = schema

        async with semaphore:
            return await self.get_data(
                schema=schema, data_id=data_id, namespace=namespace
            )

    async def get_batch(self, schema: str, batch: List[Dict], namespace: str = None):
        if not namespace:
            namespace = schema

        sem = asyncio.Semaphore(self.connections)
        return await VespaAsync._wait(
            self._get_data_semaphore,
            [(schema, data_point["id"], sem, namespace) for data_point in batch],
        )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def update_data(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        create: bool = False,
        namespace: str = None,
    ) -> VespaResponse:
        if not namespace:
            namespace = schema

        end_point = "{}/document/v1/{}/{}/docid/{}?create={}".format(
            self.app.end_point, namespace, schema, str(data_id), str(create).lower()
        )
        vespa_format = {"fields": {k: {"assign": v} for k, v in fields.items()}}
        response = await self.aiohttp_session.put(end_point, json=vespa_format)
        return VespaResponse(
            json=await response.json(),
            status_code=response.status,
            url=str(response.url),
            operation_type="update",
        )

    async def _update_data_semaphore(
        self,
        schema: str,
        data_id: str,
        fields: Dict,
        semaphore: asyncio.Semaphore,
        create: bool = False,
        namespace: str = None,
    ):
        if not namespace:
            namespace = schema

        async with semaphore:
            return await self.update_data(
                schema=schema,
                data_id=data_id,
                fields=fields,
                create=create,
                namespace=namespace,
            )

    async def update_batch(self, schema: str, batch: List[Dict], namespace: str = None):
        if not namespace:
            namespace = schema
        sem = asyncio.Semaphore(self.connections)
        return await VespaAsync._wait(
            self._update_data_semaphore,
            [
                (
                    schema,
                    data_point["id"],
                    data_point["fields"],
                    sem,
                    data_point.get("create", False),
                    namespace,
                )
                for data_point in batch
            ],
        )
