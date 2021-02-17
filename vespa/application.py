# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
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
adapter = HTTPAdapter(max_retries=retry_strategy)
http = Session()
http.mount("https://", adapter)
http.mount("http://", adapter)


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

        if port is None:
            self.end_point = self.url
        else:
            self.end_point = str(url).rstrip("/") + ":" + str(port)
        self.search_end_point = self.end_point + "/search/"

    def __repr__(self):
        if self.port:
            return "Vespa({}, {})".format(self.url, self.port)
        else:
            return "Vespa({})".format(self.url)

    def get_application_status(self) -> Optional[Response]:
        """
        Get application status.

        :return:
        """
        end_point = "{}/ApplicationStatus".format(self.end_point)
        try:
            response = http.get(end_point, cert=self.cert)
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
            r = http.post(self.search_end_point, json=body, cert=self.cert)
            return VespaResult(vespa_result=r.json())

    def feed_data_point(self, schema: str, data_id: str, fields: Dict) -> Response:
        """
        Feed a data point to a Vespa app.

        :param schema: The schema that we are sending data to.
        :param data_id: Unique id associated with this data point.
        :param fields: Dict containing all the fields required by the `schema`.
        :return: Response of the HTTP POST request.
        """
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.end_point, schema, schema, str(data_id)
        )
        vespa_format = {"fields": fields}
        response = http.post(end_point, json=vespa_format, cert=self.cert)
        return response

    def delete_data(self, schema: str, data_id: str) -> Response:
        """
        Delete a data point from a Vespa app.

        :param schema: The schema that we are deleting data from.
        :param data_id: Unique id associated with this data point.
        :return: Response of the HTTP DELETE request.
        """
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.end_point, schema, schema, str(data_id)
        )
        response = http.delete(end_point, cert=self.cert)
        return response

    def get_data(self, schema: str, data_id: str) -> Response:
        """
        Get a data point from a Vespa app.

        :param schema: The schema that we are getting data from.
        :param data_id: Unique id associated with this data point.
        :return: Response of the HTTP GET request.
        """
        end_point = "{}/document/v1/{}/{}/docid/{}".format(
            self.end_point, schema, schema, str(data_id)
        )
        response = http.get(end_point, cert=self.cert)
        return response

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
        end_point = "{}/document/v1/{}/{}/docid/{}?create={}".format(
            self.end_point, schema, schema, str(data_id), str(create).lower()
        )

        vespa_format = {"fields": {k: {"assign": v} for k, v in fields.items()}}
        response = http.put(end_point, json=vespa_format, cert=self.cert)
        return response

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
