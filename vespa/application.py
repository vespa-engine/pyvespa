# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import ssl
import aiohttp
import asyncio

from typing import Optional, Dict, Tuple, List, IO
from pandas import DataFrame
from requests import Session
from requests.models import Response
from requests.exceptions import ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from vespa.query import QueryModel, VespaResult
from vespa.evaluation import EvalMetric

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["POST", "GET", "DELETE", "PUT"],
)


async def await_coroutine(f, args=None, kwargs=None):
    task = asyncio.ensure_future(f() if (args is None and kwargs is None) else f(*args, **kwargs))
    await asyncio.wait([task], return_when=asyncio.ALL_COMPLETED)  # this is because aiohttp sometimes complains about the need for this being a task
    return task.result()


def wrap_coroutine(f, args=None, kwargs=None):
    return asyncio.run(await_coroutine(f, args, kwargs)) if asyncio._get_running_loop() is None else await_coroutine(f, args, kwargs)


def wrap_coroutine_with(await_func, f, args=None, kwargs=None):
    return asyncio.run(await_func(f, args, kwargs)) if asyncio._get_running_loop() is None else await_func(f, args, kwargs)


class ClientResponseProxy(object):
    """
    Wraps the coroutines of an aiohttp.ClientResponse object.
    If in an event loop, awaits the response. If not, run them in an ad-hoc event loop.
    """
    def __init__(self, response):
        self.response = response
        self.status_code = response.status

    def __getattr__(self, attr):
        return getattr(self.response, attr)

    def read(self, *args, **kwargs):
        return wrap_coroutine(self.response.read, args, kwargs)

    def release(self, *args, **kwargs):
        return wrap_coroutine(self.response.release, args, kwargs)

    def json(self, *args, **kwargs):
        return wrap_coroutine(self.response.json, args, kwargs)

    def text(self, *args, **kwargs):
        return wrap_coroutine(self.response.text, args, kwargs)


class Vespa(object):
    def __init__(
        self,
        url: str,
        port: Optional[int] = None,
        deployment_message: Optional[List[str]] = None,
        cert: Optional[str] = None,
        output_file: IO = sys.stdout,
    ) -> None:
        """
        Establish a connection with a Vespa application.

        :param url: Vespa instance URL.
        :param port: Vespa instance port.
        :param deployment_message: Message returned by Vespa engine after deployment. Used internally by deploy methods.
        :param cert: Path to certificate and key file.
        :param output_file: Output file to write output messages.

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
        self.http_session = None
        self.aiohttp_session = None

        if port is None:
            self.end_point = self.url
        else:
            self.end_point = str(url).rstrip("/") + ":" + str(port)
        self.search_end_point = self.end_point + "/search/"

        self._open_http_session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        await self._open_aiohttp_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_aiohttp_session()

    def _open_http_session(self):
        if self.http_session is not None:
            return
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.http_session = Session()
        self.http_session.mount("https://", adapter)
        self.http_session.mount("http://", adapter)
        return self.http_session

    def _close_http_session(self):
        if self.http_session is None:
            return
        self.http_session.close()

    async def _open_aiohttp_session(self):
        if self.aiohttp_session is not None and not self.aiohttp_session.closed:
            return
        sslcontext = False
        if self.cert is not None:
            sslcontext = ssl.create_default_context().load_cert_chain(self.cert)
        conn = aiohttp.TCPConnector(ssl=sslcontext)
        self.aiohttp_session = aiohttp.ClientSession(connector=conn)
        return self.aiohttp_session

    def _close_aiohttp_session(self):
        if self.aiohttp_session is None:
            return
        return self.aiohttp_session.close()

    def close(self):
        self._close_http_session()

    def __repr__(self):
        if self.port:
            return "Vespa({}, {})".format(self.url, self.port)
        else:
            return "Vespa({})".format(self.url)

    async def _wrap_async(self, f, args, kwargs=None):
        session_closed = self.aiohttp_session is None or self.aiohttp_session.closed
        try:
            await self._open_aiohttp_session()

            if type(args) == tuple:
                return await f(*args)

            if type(args) == list:
                tasks = [asyncio.create_task(f(*arg)) for arg in args]
                await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
                return [result for result in map(lambda task: task.result(), tasks)]

            raise ValueError("Unknown argument type to _wrap_async")

        finally:
            if session_closed:
                await self._close_aiohttp_session()  # Only close if session was opened in this function

    def get_application_status(self) -> Optional[Response]:
        """
        Get application status.

        :return:
        """
        end_point = "{}/ApplicationStatus".format(self.end_point)
        try:
            response = self.http_session.get(end_point, cert=self.cert)
        except ConnectionError:
            response = None
        return response

    def query(
        self,
        body: Optional[Dict] = None,
        query: Optional[str] = None,
        query_model: Optional[QueryModel] = None,
        debug_request: bool = False,
        recall: Optional[Tuple] = None,
        **kwargs
    ) -> VespaResult:
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

        if body is None:
            assert query is not None, "No 'query' specified."
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

        if debug_request:
            return VespaResult(vespa_result={}, request_body=body)
        else:
            return wrap_coroutine_with(self._wrap_async, self._query, (body,))

    async def _query(self, body: Dict) -> VespaResult:
        r = await self.aiohttp_session.post(self.search_end_point, json=body)
        return VespaResult(vespa_result=await r.json())

    def feed_data_point(self, schema: str, data_id: str, fields: Dict):
        """
        Feed a data point to a Vespa app.

        :param schema: The schema that we are sending data to.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields required by the `schema`.
        :return: Response of the HTTP POST request.
        """
        return wrap_coroutine_with(self._wrap_async, self._feed_data_point, (schema, data_id, fields))

    def feed_batch(self, batch):
        """
        Feed a batch of data to a Vespa app.

        :param batch: A list of tuples with 'schema', 'id' and 'fields'.
        :return: List of HTTP POST responses
        """
        return wrap_coroutine_with(self._wrap_async, self._feed_data_point, batch)

    async def _feed_data_point(self, schema: str, data_id: str, fields: Dict):
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.end_point, schema, schema, str(data_id)
        )
        vespa_format = {"fields": fields}
        return ClientResponseProxy(await self.aiohttp_session.post(end_point, json=vespa_format))

    def delete_data(self, schema: str, data_id: str) -> Response:
        """
        Delete a data point from a Vespa app.

        :param schema: The schema that we are deleting data from.
        :param data_id: Unique id associated with this data point.
        :return: Response of the HTTP DELETE request.
        """
        return wrap_coroutine_with(self._wrap_async, self._delete_data, (schema, data_id))

    def delete_batch(self, batch: List):
        """
        Async delete a batch of data from a Vespa app.

        :param batch: A list of tuples with 'schema' and 'id'
        :return:
        """
        return wrap_coroutine_with(self._wrap_async, self._delete_data, batch)

    async def _delete_data(self, schema: str, data_id: str):
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.end_point, schema, schema, str(data_id)
        )
        return ClientResponseProxy(await self.aiohttp_session.delete(end_point))

    def get_data(self, schema: str, data_id: str) -> Response:
        """
        Get a data point from a Vespa app.

        :param schema: The schema that we are getting data from.
        :param data_id: Unique id associated with this data point.
        :return: Response of the HTTP GET request.
        """
        return wrap_coroutine_with(self._wrap_async, self._get_data, (schema, data_id))

    def get_batch(self, batch: List):
        """
        Async get a batch of data from a Vespa app.

        :param batch: A list of tuples with 'schema' and 'id'.
        :return:
        """
        return wrap_coroutine_with(self._wrap_async, self._get_data, batch)

    async def _get_data(self, schema: str, data_id: str):
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.end_point, schema, schema, str(data_id)
        )
        return ClientResponseProxy(await self.aiohttp_session.get(end_point))

    def update_data(
        self, schema: str, data_id: str, fields: Dict, create: bool = False
    ) -> Response:
        """
        Update a data point in a Vespa app.

        :param schema: The schema that we are updating data.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields you want to update.
        :param create: If true, updates to non-existent documents will create an empty document to update
        :return: Response of the HTTP PUT request.
        """
        return wrap_coroutine_with(self._wrap_async, self._update_data, (schema, data_id, fields, create))

    def update_batch(self, batch: List):
        """
        Update a batch of data points.

        :param batch: A list of tuples with 'schema', 'id', 'fields', and 'create'
        :return:
        """
        return wrap_coroutine_with(self._wrap_async, self._update_data, batch)

    async def _update_data(self, schema: str, data_id: str, fields: Dict, create: bool = False):
        end_point = "{}/document/v1/{}/{}/docid/{}?create={}".format(
            self.end_point, schema, schema, str(data_id), str(create).lower()
        )
        vespa_format = {"fields": {k: {"assign": v} for k, v in fields.items()}}
        return ClientResponseProxy(await self.aiohttp_session.put(end_point, json=vespa_format))

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
        **kwargs
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
            **kwargs
        )
        hits = relevant_id_result.hits
        features = []
        if len(hits) == 1 and hits[0]["fields"][id_field] == relevant_id:
            if number_additional_docs > 0:
                random_hits_result = self.query(
                    query=query,
                    query_model=query_model,
                    hits=number_additional_docs,
                    **kwargs
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
        labeled_data: List[Dict],
        id_field: str,
        query_model: QueryModel,
        number_additional_docs: int,
        relevant_score: int = 1,
        default_score: int = 0,
        show_progress: Optional[int] = None,
        **kwargs
    ) -> DataFrame:
        """
        Collect training data based on a set of labelled data.

        :param labeled_data: Labelled data containing query, query_id and relevant ids.
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
                    **kwargs
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
        **kwargs
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
        :param kwargs: Extra keyword arguments to be included in the Vespa Query.
        :return: Dict containing query_id and metrics according to the selected evaluation metrics.
        """

        query_results = self.query(query=query, query_model=query_model, **kwargs)
        evaluation = {"query_id": query_id}
        for evaluator in eval_metrics:
            evaluation.update(
                evaluator.evaluate_query(
                    query_results, relevant_docs, id_field, default_score
                )
            )
        return evaluation

    def evaluate(
        self,
        labeled_data: List[Dict],
        eval_metrics: List[EvalMetric],
        query_model: QueryModel,
        id_field: str,
        default_score: int = 0,
        **kwargs
    ) -> DataFrame:
        """

        :param labeled_data: Labelled data containing query, query_id and relevant ids.
        :param eval_metrics: A list of evaluation metrics.
        :param query_model: Query model.
        :param id_field: The Vespa field representing the document id.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param kwargs: Extra keyword arguments to be included in the Vespa Query.
        :return: DataFrame containing query_id and metrics according to the selected evaluation metrics.
        """
        evaluation = []
        for query_data in labeled_data:
            evaluation_query = self.evaluate_query(
                eval_metrics=eval_metrics,
                query_model=query_model,
                query_id=query_data["query_id"],
                query=query_data["query"],
                id_field=id_field,
                relevant_docs=query_data["relevant_docs"],
                default_score=default_score,
                **kwargs
            )
            evaluation.append(evaluation_query)
        evaluation = DataFrame.from_records(evaluation)
        return evaluation
