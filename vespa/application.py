# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import ssl
import aiohttp
import asyncio
import concurrent.futures

from typing import Optional, Dict, Tuple, List, IO, Union
from pandas import DataFrame
from requests import Session
from requests.models import Response
from requests.exceptions import ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tenacity import retry, wait_exponential, stop_after_attempt

from vespa.io import VespaQueryResponse, VespaResponse
from vespa.query import QueryModel
from vespa.evaluation import EvalMetric
from vespa.package import ApplicationPackage

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["POST", "GET", "DELETE", "PUT"],
)


def parse_labeled_data(df):
    """
    Convert a DataFrame with labeled data to format used internally

    :param df: DataFrame with the following required columns ["qid", "query", "doc_id", "relevance"].
    :return: List of Dict containing a concise representation of the labeled data, grouped by query_id and query.
    """
    required_columns = ["qid", "query", "doc_id", "relevance"]
    assert all(
        [x in list(df.columns) for x in required_columns]
    ), "DataFrame needs at least the following columns: {}".format(required_columns)
    qid_query = (
        df[["qid", "query"]].drop_duplicates(["qid", "query"]).to_dict(orient="records")
    )
    labeled_data = []
    for q in qid_query:
        docid_relevance = df[(df["qid"] == q["qid"]) & (df["query"] == q["query"])][
            ["doc_id", "relevance"]
        ]
        relevant_docs = []
        for idx, row in docid_relevance.iterrows():
            relevant_docs.append({"id": row["doc_id"], "score": row["relevance"]})
        data_point = {
            "query_id": q["qid"],
            "query": q["query"],
            "relevant_docs": relevant_docs,
        }
        labeled_data.append(data_point)
    return labeled_data


def parse_feed_df(df: DataFrame, include_id):
    """
    Convert a df into batch format for feeding

    :param df: DataFrame with the following required columns ["id"]. Additional columns are assumed to be fields.
    :param include_id: Include id on the fields to be fed.
    :return: List of Dict containing 'id' and 'fields'.
    """
    required_columns = ["id"]
    assert all(
        [x in list(df.columns) for x in required_columns]
    ), "DataFrame needs at least the following columns: {}".format(required_columns)
    records = df.to_dict(orient="records")
    batch = [
        {
            "id": record["id"],
            "fields": record
            if include_id
            else {k: v for k, v in record.items() if k not in ["id"]},
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
        Establish a connection with a Vespa application.

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

    def _run_coroutine_new_event_loop(self, loop, coro):
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def _check_for_running_loop_and_run_coroutine(self, coro):
        try:
            _ = asyncio.get_running_loop()
            new_loop = asyncio.new_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self._run_coroutine_new_event_loop, new_loop, coro
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

    def get_application_status(self) -> Optional[Response]:
        """
        Get application status.

        :return:
        """
        with VespaSync(self) as sync_app:
            return sync_app.get_application_status()

    def get_model_endpoint(self, model_id: Optional[str] = None) -> Optional[Response]:
        """Get model evaluation endpoints."""

        with VespaSync(self) as sync_app:
            return sync_app.get_model_endpoint(model_id=model_id)

    def _build_query_body(
        self,
        query: Optional[str] = None,
        query_model: Optional[QueryModel] = None,
        recall: Optional[Tuple] = None,
        **kwargs,
    ) -> Dict:
        assert query is not None, "No 'query' specified."
        if not query_model:
            query_model = self.get_default_query_model()
        assert query_model is not None, "No 'query_model' specified."
        body = query_model.create_body(query=query)
        if recall is not None:
            body.update(
                {
                    "recall": "+("
                    + " ".join(
                        ["{}:{}".format(recall[0], str(doc)) for doc in recall[1]]
                    )
                    + ")"
                }
            )
        body.update(kwargs)
        return body

    def query(
        self,
        body: Optional[Dict] = None,
        query: Optional[str] = None,
        query_model: Optional[QueryModel] = None,
        debug_request: bool = False,
        recall: Optional[Tuple] = None,
        **kwargs,
    ) -> VespaQueryResponse:
        """
        Send a query request to the Vespa application.

        Either send 'body' containing all the request parameters or specify 'query' and 'query_model'.

        :param body: Dict containing all the request parameters.
        :param query: Query string
        :param query_model: Query model
        :param debug_request: return request body for debugging instead of sending the request.
        :param recall: Tuple of size 2 where the first element is the name of the field to use to recall and the
            second element is a list of the values to be recalled.
        :param kwargs: Additional parameters to be sent along the request.
        :return: Either the request body if debug_request is True or the result from the Vespa application
        """
        with VespaSync(self) as sync_app:
            return sync_app.query(
                body=body,
                query=query,
                query_model=query_model,
                debug_request=debug_request,
                recall=recall,
                **kwargs,
            )

    def _query_batch_sync(
        self,
        body_batch: Optional[List[Dict]],
        query_batch: Optional[List[str]],
        query_model: Optional[QueryModel],
        recall: Optional[List[Tuple]],
        **kwargs,
    ):
        if body_batch:
            return [self.query(body=body, **kwargs) for body in body_batch]
        else:
            if recall:
                return [
                    self.query(query=q, query_model=query_model, recall=r, **kwargs)
                    for (q, r) in zip(query_batch, recall)
                ]
            else:
                return [
                    self.query(query=query, query_model=query_model, **kwargs)
                    for query in query_batch
                ]

    async def _query_batch_async(
        self,
        body_batch: Optional[List[Dict]],
        query_batch: Optional[List[str]],
        query_model: Optional[QueryModel],
        recall: Optional[List[Tuple]],
        connections,
        total_timeout,
        **kwargs,
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.query_batch(
                body_batch=body_batch,
                query_batch=query_batch,
                query_model=query_model,
                recall=recall,
                **kwargs,
            )

    def query_batch(
        self,
        body_batch: Optional[List[Dict]] = None,
        query_batch: Optional[List[str]] = None,
        query_model: Optional[QueryModel] = None,
        recall: Optional[List[Tuple]] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
        **kwargs,
    ):
        """
        Send queries in batch to a Vespa app.

        :param body_batch: A list of dict containing all the request parameters. Set to None if using 'query_batch'.
        :param query_batch: A list of query strings. Set to None if using 'body_batch'.
        :param query_model: Query model to use when sending query strings. Set to None if using 'body_batch'.
        :param recall: List of tuples, one for each query. Tuple of size 2 where the first element is the name
            of the field to use to recall and the second element is a list of the values to be recalled.
        :param asynchronous: Set True to send data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :param kwargs: Additional parameters to be sent along the request.
        :return: List of HTTP POST responses
        """

        if body_batch:
            assert (
                query_batch is None
            ), "'query_batch' has no effect if 'body_batch' is not None."
        elif query_batch:
            assert (
                body_batch is None
            ), "'body_batch' has no effect if 'query_batch' is not None."
            assert (
                query_model is not None
            ), "Specify a 'query_model' when using 'query_batch' argument."
            number_of_queries = len(query_batch)

            if recall:
                assert (
                    len(recall) == number_of_queries
                ), "Specify one recall tuple for each query in the batch."
        else:
            ValueError("Specify either 'query_batch' or 'body_batch'.")

        if asynchronous:
            coro = self._query_batch_async(
                body_batch=body_batch,
                query_batch=query_batch,
                query_model=query_model,
                recall=recall,
                connections=connections,
                total_timeout=total_timeout,
                **kwargs,
            )
            return self._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._query_batch_sync(
                body_batch=body_batch,
                query_batch=query_batch,
                query_model=query_model,
                recall=recall,
                **kwargs,
            )

    def feed_data_point(self, schema: str, data_id: str, fields: Dict) -> VespaResponse:
        """
        Feed a data point to a Vespa app.

        :param schema: The schema that we are sending data to.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields required by the `schema`.
        :return: Response of the HTTP POST request.
        """
        with VespaSync(app=self) as sync_app:
            return sync_app.feed_data_point(
                schema=schema, data_id=data_id, fields=fields
            )

    def _feed_batch_sync(self, schema: str, batch: List[Dict]):
        return [
            self.feed_data_point(schema, data_point["id"], data_point["fields"])
            for data_point in batch
        ]

    async def _feed_batch_async(
        self, schema: str, batch: List[Dict], connections, total_timeout
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.feed_batch(schema=schema, batch=batch)

    def feed_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
    ):
        """
        Feed a batch of data to a Vespa app.

        :param batch: A list of dict containing the keys 'id' and 'fields' to be used in the :func:`feed_data_point`.
        :param schema: The schema that we are sending data to. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to send data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :return: List of HTTP POST responses
        """
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        if asynchronous:
            coro = self._feed_batch_async(
                schema=schema,
                batch=batch,
                connections=connections,
                total_timeout=total_timeout,
            )
            return self._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._feed_batch_sync(schema=schema, batch=batch)

    def feed_df(self, df: DataFrame, include_id: bool = True, **kwargs):
        """
        Feed data contained in a DataFrame.

        :param df: A DataFrame containing a required 'id' column and the remaining fields to be fed.
        :param include_id: Include id on the fields to be fed. Default to True.
        :param kwargs: Additional parameters are passed to :func:`feed_batch`.
        :return: List of HTTP POST responses
        """
        batch = parse_feed_df(df=df, include_id=include_id)
        return self.feed_batch(batch=batch, **kwargs)

    def delete_data(self, schema: str, data_id: str) -> VespaResponse:
        """
        Delete a data point from a Vespa app.

        :param schema: The schema that we are deleting data from.
        :param data_id: Unique id associated with this data point.
        :return: Response of the HTTP DELETE request.
        """
        with VespaSync(self) as sync_app:
            return sync_app.delete_data(schema=schema, data_id=data_id)

    def _delete_batch_sync(self, schema: str, batch: List[Dict]):
        return [self.delete_data(schema, data_point["id"]) for data_point in batch]

    async def _delete_batch_async(
        self, schema: str, batch: List[Dict], connections, total_timeout
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.delete_batch(schema=schema, batch=batch)

    def delete_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
    ):
        """
        Delete a batch of data from a Vespa app.

        :param batch: A list of dict containing the key 'id'.
        :param schema: The schema that we are deleting data from. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to get data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :return: List of HTTP POST responses
        """
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        if asynchronous:
            coro = self._delete_batch_async(
                schema=schema,
                batch=batch,
                connections=connections,
                total_timeout=total_timeout,
            )
            return self._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._delete_batch_sync(schema=schema, batch=batch)

    def delete_all_docs(self, content_cluster_name: str, schema: str) -> Response:
        """
        Delete all documents associated with the schema

        :param content_cluster_name: Name of content cluster to GET from, or visit.
        :param schema: The schema that we are deleting data from.
        :return: Response of the HTTP DELETE request.
        """
        with VespaSync(self) as sync_app:
            return sync_app.delete_all_docs(
                content_cluster_name=content_cluster_name, schema=schema
            )

    def get_data(self, schema: str, data_id: str) -> VespaResponse:
        """
        Get a data point from a Vespa app.

        :param schema: The schema that we are getting data from.
        :param data_id: Unique id associated with this data point.
        :return: Response of the HTTP GET request.
        """
        with VespaSync(self) as sync_app:
            return sync_app.get_data(schema=schema, data_id=data_id)

    def _get_batch_sync(self, schema: str, batch: List[Dict]):
        return [self.get_data(schema, data_point["id"]) for data_point in batch]

    async def _get_batch_async(
        self, schema: str, batch: List[Dict], connections, total_timeout
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.get_batch(schema=schema, batch=batch)

    def get_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
    ):
        """
        Get a batch of data from a Vespa app.

        :param batch: A list of dict containing the key 'id'.
        :param schema: The schema that we are getting data from. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to get data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :return: List of HTTP POST responses
        """
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        if asynchronous:
            coro = self._get_batch_async(
                schema=schema,
                batch=batch,
                connections=connections,
                total_timeout=total_timeout,
            )
            return self._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._get_batch_sync(schema=schema, batch=batch)

    def update_data(
        self, schema: str, data_id: str, fields: Dict, create: bool = False
    ) -> VespaResponse:
        """
        Update a data point in a Vespa app.

        :param schema: The schema that we are updating data.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields you want to update.
        :param create: If true, updates to non-existent documents will create an empty document to update
        :return: Response of the HTTP PUT request.
        """
        with VespaSync(self) as sync_app:
            return sync_app.update_data(
                schema=schema, data_id=data_id, fields=fields, create=create
            )

    def _update_batch_sync(self, schema: str, batch: List[Dict]):
        return [
            self.update_data(
                schema,
                data_point["id"],
                data_point["fields"],
                data_point.get("create", False),
            )
            for data_point in batch
        ]

    async def _update_batch_async(
        self, schema: str, batch: List[Dict], connections, total_timeout
    ):
        async with VespaAsync(
            app=self, connections=connections, total_timeout=total_timeout
        ) as async_app:
            return await async_app.update_batch(schema=schema, batch=batch)

    def update_batch(
        self,
        batch: List[Dict],
        schema: Optional[str] = None,
        asynchronous=True,
        connections: Optional[int] = 100,
        total_timeout: int = 100,
    ):
        """
        Update a batch of data in a Vespa app.

        :param batch: A list of dict containing the keys 'id', 'fields' and 'create' (create defaults to False).
        :param schema: The schema that we are updating data to. The schema is optional in case it is possible to infer
            the schema from the application package.
        :param asynchronous: Set True to update data in async mode. Default to True.
        :param connections: Number of allowed concurrent connections, valid only if `asynchronous=True`.
        :param total_timeout: Total timeout in secs for each of the concurrent requests when using `asynchronous=True`.
        :return: List of HTTP POST responses
        """
        if not schema:
            try:
                schema = self._infer_schema_name()
            except ValueError:
                raise ValueError(
                    "Not possible to infer schema name. Specify schema parameter."
                )

        if asynchronous:
            coro = self._update_batch_async(
                schema=schema,
                batch=batch,
                connections=connections,
                total_timeout=total_timeout,
            )
            return self._check_for_running_loop_and_run_coroutine(coro=coro)
        else:
            return self._update_batch_sync(schema=schema, batch=batch)

    @staticmethod
    def annotate_data(
        hits, query_id, id_field, relevant_id, fields, relevant_score, default_score
    ):
        data = []
        for h in hits:
            record = {}
            record.update({"document_id": h["fields"][id_field]})
            record.update({"query_id": query_id})
            record.update(
                {
                    "label": relevant_score
                    if h["fields"][id_field] == relevant_id
                    else default_score
                }
            )
            for field in fields:
                field_value = h["fields"].get(field, None)
                if field_value:
                    if isinstance(field_value, dict):
                        record.update(field_value)
                    else:
                        record.update({field: field_value})
            data.append(record)
        return data

    def collect_training_data_point(
        self,
        query: str,
        query_id: str,
        relevant_id: str,
        id_field: str,
        query_model: QueryModel,
        number_additional_docs: int,
        fields: List[str],
        relevant_score: int = 1,
        default_score: int = 0,
        **kwargs,
    ) -> List[Dict]:
        """
        Collect training data based on a single query

        :param query: Query string.
        :param query_id: Query id represented as str.
        :param relevant_id: Relevant id represented as a str.
        :param id_field: The Vespa field representing the document id.
        :param query_model: Query model.
        :param number_additional_docs: Number of additional documents to retrieve for each relevant document.
        :param fields: Which fields should be retrieved.
        :param relevant_score: Score to assign to relevant documents. Default to 1.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param kwargs: Extra keyword arguments to be included in the Vespa Query.
        :return: List of dicts containing the document id (document_id), query id (query_id), scores (relevant)
            and vespa rank features returned by the Query model RankProfile used.
        """

        relevant_id_result = self.query(
            query=query,
            query_model=query_model,
            recall=(id_field, [relevant_id]),
            **kwargs,
        )
        hits = relevant_id_result.hits
        features = []
        if len(hits) == 1 and hits[0]["fields"][id_field] == relevant_id:
            if number_additional_docs > 0:
                random_hits_result = self.query(
                    query=query,
                    query_model=query_model,
                    hits=number_additional_docs,
                    **kwargs,
                )
                hits.extend(random_hits_result.hits)

            features = self.annotate_data(
                hits=hits,
                query_id=query_id,
                id_field=id_field,
                relevant_id=relevant_id,
                fields=fields,
                relevant_score=relevant_score,
                default_score=default_score,
            )
        return features

    def collect_training_data(
        self,
        labeled_data: Union[List[Dict], DataFrame],
        id_field: str,
        query_model: QueryModel,
        number_additional_docs: int,
        relevant_score: int = 1,
        default_score: int = 0,
        show_progress: Optional[int] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Collect training data based on a set of labelled data.

        labeled_data can be a DataFrame or a List of Dict:

        >>> labeled_data_df = DataFrame(
        ...     data={
        ...         "qid": [0, 0, 1, 1],
        ...         "query": ["Intrauterine virus infections and congenital heart disease", "Intrauterine virus infections and congenital heart disease", "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus", "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus"],
        ...         "doc_id": [0, 3, 1, 5],
        ...         "relevance": [1,1,1,1]
        ...     }
        ... )

        >>> labeled_data = [
        ...     {
        ...         "query_id": 0,
        ...         "query": "Intrauterine virus infections and congenital heart disease",
        ...         "relevant_docs": [{"id": 0, "score": 1}, {"id": 3, "score": 1}]
        ...     },
        ...     {
        ...         "query_id": 1,
        ...         "query": "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus",
        ...         "relevant_docs": [{"id": 1, "score": 1}, {"id": 5, "score": 1}]
        ...     }
        ... ]

        :param labeled_data: Labelled data containing query, query_id and relevant ids. See details about data format.
        :param id_field: The Vespa field representing the document id.
        :param query_model: Query model.
        :param number_additional_docs: Number of additional documents to retrieve for each relevant document.
        :param relevant_score: Score to assign to relevant documents. Default to 1.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param show_progress: Prints the the current point being collected every `show_progress` step. Default to None,
            in which case progress is not printed.
        :param kwargs: Extra keyword arguments to be included in the Vespa Query.
        :return: DataFrame containing document id (document_id), query id (query_id), scores (relevant)
            and vespa rank features returned by the Query model RankProfile used.
        """

        if isinstance(labeled_data, DataFrame):
            labeled_data = parse_labeled_data(df=labeled_data)

        training_data = []
        number_queries = len(labeled_data)
        idx_total = 0
        for query_idx, query_data in enumerate(labeled_data):
            number_relevant_docs = len(query_data["relevant_docs"])
            for doc_idx, doc_data in enumerate(query_data["relevant_docs"]):
                idx_total += 1
                if (show_progress is not None) and (idx_total % show_progress == 0):
                    print(
                        "Query {}/{}, Doc {}/{}. Query id: {}. Doc id: {}".format(
                            query_idx,
                            number_queries,
                            doc_idx,
                            number_relevant_docs,
                            query_data["query_id"],
                            doc_data["id"],
                        ),
                        file=self.output_file,
                    )
                training_data_point = self.collect_training_data_point(
                    query=query_data["query"],
                    query_id=query_data["query_id"],
                    relevant_id=doc_data["id"],
                    id_field=id_field,
                    query_model=query_model,
                    number_additional_docs=number_additional_docs,
                    relevant_score=doc_data.get("score", relevant_score),
                    default_score=default_score,
                    **kwargs,
                )
                training_data.extend(training_data_point)
        training_data = DataFrame.from_records(training_data)
        return training_data

    def evaluate_query(
        self,
        eval_metrics: List[EvalMetric],
        query_model: QueryModel,
        query_id: str,
        query: str,
        id_field: str,
        relevant_docs: List[Dict],
        default_score: int = 0,
        detailed_metrics=False,
        **kwargs,
    ) -> Dict:
        """
        Evaluate a query according to evaluation metrics

        :param eval_metrics: A list of evaluation metrics.
        :param query_model: Query model.
        :param query_id: Query id represented as str.
        :param query: Query string.
        :param id_field: The Vespa field representing the document id.
        :param relevant_docs: A list with dicts where each dict contains a doc id a optionally a doc score.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param detailed_metrics: Return intermediate computations if available.
        :param kwargs: Extra keyword arguments to be included in the Vespa Query.
        :return: Dict containing query_id and metrics according to the selected evaluation metrics.
        """

        query_results = self.query(query=query, query_model=query_model, **kwargs)
        evaluation = {"model": query_model.name, "query_id": query_id}
        for evaluator in eval_metrics:
            evaluation.update(
                evaluator.evaluate_query(
                    query_results,
                    relevant_docs,
                    id_field,
                    default_score,
                    detailed_metrics,
                )
            )
        return evaluation

    def evaluate(
        self,
        labeled_data: Union[List[Dict], DataFrame],
        eval_metrics: List[EvalMetric],
        query_model: Union[QueryModel, List[QueryModel]],
        id_field: str,
        default_score: int = 0,
        detailed_metrics=False,
        per_query=False,
        aggregators=None,
        **kwargs,
    ) -> DataFrame:
        """
        Evaluate a :class:`QueryModel` according to a list of :class:`EvalMetric`.

        labeled_data can be a DataFrame or a List of Dict:

        >>> labeled_data_df = DataFrame(
        ...     data={
        ...         "qid": [0, 0, 1, 1],
        ...         "query": ["Intrauterine virus infections and congenital heart disease", "Intrauterine virus infections and congenital heart disease", "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus", "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus"],
        ...         "doc_id": [0, 3, 1, 5],
        ...         "relevance": [1,1,1,1]
        ...     }
        ... )

        >>> labeled_data = [
        ...     {
        ...         "query_id": 0,
        ...         "query": "Intrauterine virus infections and congenital heart disease",
        ...         "relevant_docs": [{"id": 0, "score": 1}, {"id": 3, "score": 1}]
        ...     },
        ...     {
        ...         "query_id": 1,
        ...         "query": "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus",
        ...         "relevant_docs": [{"id": 1, "score": 1}, {"id": 5, "score": 1}]
        ...     }
        ... ]

        :param labeled_data: Labelled data containing query, query_id and relevant ids. See details about data format.
        :param eval_metrics: A list of evaluation metrics.
        :param query_model: Accept a Query model or a list of Query Models.
        :param id_field: The Vespa field representing the document id.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param detailed_metrics: Return intermediate computations if available.
        :param per_query: Set to True to return evaluation metrics per query.
        :param aggregators: Used only if `per_query=False`. List of pandas friendly aggregators to summarize per model
            metrics. We use ["mean", "median", "std"] by default.
        :param kwargs: Extra keyword arguments to be included in the Vespa Query.
        :return: DataFrame containing query_id and metrics according to the selected evaluation metrics.
        """
        if isinstance(labeled_data, DataFrame):
            labeled_data = parse_labeled_data(df=labeled_data)

        if isinstance(query_model, QueryModel):
            query_model = [query_model]

        model_names = [model.name for model in query_model]
        assert len(model_names) == len(
            set(model_names)
        ), "Duplicate model names. Choose unique model names."

        evaluation = []
        for query_data in labeled_data:
            for model in query_model:
                evaluation_query = self.evaluate_query(
                    eval_metrics=eval_metrics,
                    query_model=model,
                    query_id=query_data["query_id"],
                    query=query_data["query"],
                    id_field=id_field,
                    relevant_docs=query_data["relevant_docs"],
                    default_score=default_score,
                    detailed_metrics=detailed_metrics,
                    **kwargs,
                )
                evaluation.append(evaluation_query)
        evaluation = DataFrame.from_records(evaluation)
        if not per_query:
            if not aggregators:
                aggregators = ["mean", "median", "std"]
            evaluation = (
                evaluation[[x for x in evaluation.columns if x != "query_id"]]
                .groupby(by="model")
                .agg(aggregators)
                .T
            )
        return evaluation

    @property
    def application_package(self):
        """Get application package definition, if available."""
        if not self._application_package:
            raise ValueError("Application package not available.")
        else:
            return self._application_package

    def get_default_query_model(self):
        try:
            app_package = self.application_package
        except ValueError:
            return None
        return app_package.default_query_model

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

    def get_application_status(self) -> Optional[Response]:
        """
        Get application status.

        :return:
        """
        end_point = "{}/ApplicationStatus".format(self.app.end_point)
        try:
            response = self.http_session.get(end_point, cert=self.cert)
        except ConnectionError:
            response = None
        return response

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

    def feed_data_point(self, schema: str, data_id: str, fields: Dict) -> VespaResponse:
        """
        Feed a data point to a Vespa app.

        :param schema: The schema that we are sending data to.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields required by the `schema`.
        :return: Response of the HTTP POST request.
        """
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, schema, schema, str(data_id)
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
        query: Optional[str] = None,
        query_model: Optional[QueryModel] = None,
        debug_request: bool = False,
        recall: Optional[Tuple] = None,
        **kwargs,
    ) -> VespaQueryResponse:
        """
        Send a query request to the Vespa application.

        Either send 'body' containing all the request parameters or specify 'query' and 'query_model'.

        :param body: Dict containing all the request parameters.
        :param query: Query string
        :param query_model: Query model
        :param debug_request: return request body for debugging instead of sending the request.
        :param recall: Tuple of size 2 where the first element is the name of the field to use to recall and the
            second element is a list of the values to be recalled.
        :param kwargs: Additional parameters to be sent along the request.
        :return: Either the request body if debug_request is True or the result from the Vespa application
        """
        body = (
            self.app._build_query_body(query, query_model, recall, **kwargs)
            if body is None
            else body
        )
        if debug_request:
            return VespaQueryResponse(
                json={}, status_code=None, url=None, request_body=body
            )
        else:
            r = self.http_session.post(
                self.app.search_end_point, json=body, cert=self.cert
            )
        return VespaQueryResponse(
            json=r.json(), status_code=r.status_code, url=str(r.url)
        )

    def delete_data(self, schema: str, data_id: str) -> VespaResponse:
        """
        Delete a data point from a Vespa app.

        :param schema: The schema that we are deleting data from.
        :param data_id: Unique id associated with this data point.
        :return: Response of the HTTP DELETE request.
        """
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, schema, schema, str(data_id)
        )
        response = self.http_session.delete(end_point, cert=self.cert)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="delete",
        )

    def delete_all_docs(self, content_cluster_name: str, schema: str) -> Response:
        """
        Delete all documents associated with the schema

        :param content_cluster_name: Name of content cluster to GET from, or visit.
        :param schema: The schema that we are deleting data from.
        :return: Response of the HTTP DELETE request.
        """
        end_point = "{}/document/v1/{}/{}/docid/?cluster={}&selection=true".format(
            self.app.end_point, schema, schema, content_cluster_name
        )
        response = self.http_session.delete(end_point, cert=self.cert)
        return response

    def get_data(self, schema: str, data_id: str) -> VespaResponse:
        """
        Get a data point from a Vespa app.

        :param schema: The schema that we are getting data from.
        :param data_id: Unique id associated with this data point.
        :return: Response of the HTTP GET request.
        """
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, schema, schema, str(data_id)
        )
        response = self.http_session.get(end_point, cert=self.cert)
        return VespaResponse(
            json=response.json(),
            status_code=response.status_code,
            url=str(response.url),
            operation_type="get",
        )

    def update_data(
        self, schema: str, data_id: str, fields: Dict, create: bool = False
    ) -> VespaResponse:
        """
        Update a data point in a Vespa app.

        :param schema: The schema that we are updating data.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields you want to update.
        :param create: If true, updates to non-existent documents will create an empty document to update
        :return: Response of the HTTP PUT request.
        """
        end_point = "{}/document/v1/{}/{}/docid/{}?create={}".format(
            self.app.end_point, schema, schema, str(data_id), str(create).lower()
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

    async def _wait(self, f, args, **kwargs):
        tasks = [asyncio.create_task(f(*arg, **kwargs)) for arg in args]
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        return [result for result in map(lambda task: task.result(), tasks)]

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def query(
        self,
        body: Optional[Dict] = None,
        query: Optional[str] = None,
        query_model: Optional[QueryModel] = None,
        debug_request: bool = False,
        recall: Optional[Tuple] = None,
        **kwargs,
    ):
        if debug_request:
            return self.app.query(
                body, query, query_model, debug_request, recall, **kwargs
            )
        body = (
            self.app._build_query_body(query, query_model, recall, **kwargs)
            if body is None
            else body
        )
        r = await self.aiohttp_session.post(self.app.search_end_point, json=body)
        return VespaQueryResponse(
            json=await r.json(), status_code=r.status, url=str(r.url)
        )

    async def _query_semaphore(
        self,
        body: Optional[Dict],
        query: Optional[str],
        query_model: Optional[QueryModel],
        recall: Optional[Tuple],
        semaphore: asyncio.Semaphore,
        **kwargs,
    ):
        async with semaphore:
            return await self.query(
                body=body, query=query, query_model=query_model, recall=recall, **kwargs
            )

    async def query_batch(
        self,
        body_batch: Optional[List[Dict]],
        query_batch: Optional[List[str]],
        query_model: Optional[QueryModel],
        recall: Optional[List[Tuple]],
        **kwargs,
    ):
        sem = asyncio.Semaphore(self.connections)
        if body_batch:
            return await self._wait(
                self._query_semaphore,
                [(body, None, None, None, sem) for body in body_batch],
                **kwargs,
            )
        else:
            if recall:
                return await self._wait(
                    self._query_semaphore,
                    [
                        (None, q, query_model, r, sem)
                        for (q, r) in zip(query_batch, recall)
                    ],
                    **kwargs,
                )
            else:
                return await self._wait(
                    self._query_semaphore,
                    [(None, q, query_model, None, sem) for q in query_batch],
                    **kwargs,
                )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def feed_data_point(
        self, schema: str, data_id: str, fields: Dict
    ) -> VespaResponse:
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, schema, schema, str(data_id)
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
        self, schema: str, data_id: str, fields: Dict, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            return await self.feed_data_point(
                schema=schema, data_id=data_id, fields=fields
            )

    async def feed_batch(self, schema: str, batch: List[Dict]):
        sem = asyncio.Semaphore(self.connections)
        return await self._wait(
            self._feed_data_point_semaphore,
            [
                (schema, data_point["id"], data_point["fields"], sem)
                for data_point in batch
            ],
        )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def delete_data(self, schema: str, data_id: str) -> VespaResponse:
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, schema, schema, str(data_id)
        )
        response = await self.aiohttp_session.delete(end_point)
        return VespaResponse(
            json=await response.json(),
            status_code=response.status,
            url=str(response.url),
            operation_type="delete",
        )

    async def _delete_data_semaphore(
        self, schema: str, data_id: str, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            return await self.delete_data(schema=schema, data_id=data_id)

    async def delete_batch(self, schema: str, batch: List[Dict]):
        sem = asyncio.Semaphore(self.connections)
        return await self._wait(
            self._delete_data_semaphore,
            [(schema, data_point["id"], sem) for data_point in batch],
        )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def get_data(self, schema: str, data_id: str):
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.app.end_point, schema, schema, str(data_id)
        )
        response = await self.aiohttp_session.get(end_point)
        return VespaResponse(
            json=await response.json(),
            status_code=response.status,
            url=str(response.url),
            operation_type="get",
        )

    async def _get_data_semaphore(
        self, schema: str, data_id: str, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            return await self.get_data(schema=schema, data_id=data_id)

    async def get_batch(self, schema: str, batch: List[Dict]):
        sem = asyncio.Semaphore(self.connections)
        return await self._wait(
            self._get_data_semaphore,
            [(schema, data_point["id"], sem) for data_point in batch],
        )

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(3))
    async def update_data(
        self, schema: str, data_id: str, fields: Dict, create: bool = False
    ) -> VespaResponse:
        end_point = "{}/document/v1/{}/{}/docid/{}?create={}".format(
            self.app.end_point, schema, schema, str(data_id), str(create).lower()
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
    ):
        async with semaphore:
            return await self.update_data(
                schema=schema, data_id=data_id, fields=fields, create=create
            )

    async def update_batch(self, schema: str, batch: List[Dict]):
        sem = asyncio.Semaphore(self.connections)
        return await self._wait(
            self._update_data_semaphore,
            [
                (
                    schema,
                    data_point["id"],
                    data_point["fields"],
                    sem,
                    data_point.get("create", False),
                )
                for data_point in batch
            ],
        )
