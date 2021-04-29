# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
from typing import Optional, Dict, Tuple, List, IO, Union
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
        detailed_metrics=False,
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
        **kwargs
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
                    **kwargs
                )
                evaluation.append(evaluation_query)
        evaluation = DataFrame.from_records(evaluation)
        if not per_query:
            if not aggregators:
                aggregators = ["mean", "median", "std"]
            evaluation = (
                evaluation[[x for x in evaluation.columns if x != "query_id"]]
                .groupby(by="model")
                .agg(aggregators).T
            )
        return evaluation
