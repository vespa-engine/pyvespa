from __future__ import annotations
import os
import csv
import logging
from typing import (
    Dict,
    Set,
    Callable,
    List,
    Optional,
    Union,
    Tuple,
    Sequence,
    Mapping,
    Any,
)
import math
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import re
import urllib.parse
from vespa.application import Vespa
from vespa.io import VespaQueryResponse


logger = logging.getLogger(__name__)

# Set default logging level to INFO and use StreamHandler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RandomHitsSamplingStrategy(Enum):
    """
    Enum for different random hits sampling strategies.

    - RATIO: Sample random hits as a ratio of relevant docs (e.g., 1.0 = equal number, 2.0 = twice as many)
    - FIXED: Sample a fixed number of random hits per query
    """

    RATIO = "ratio"
    FIXED = "fixed"


def mean(values: List[float]) -> float:
    """
    Compute the mean of a list of numbers without using numpy.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: List[float], p: float) -> float:
    """
    Compute the p-th percentile of a list of values (0 <= p <= 100).
    This approximates numpy.percentile's behavior.
    """
    if not values:
        return 0.0
    if p < 0:
        p = 0
    if p > 100:
        p = 100
    values_sorted = sorted(values)
    index = (len(values_sorted) - 1) * p / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values_sorted[int(index)]
    # Linear interpolation between the two closest ranks
    fraction = index - lower
    return (
        values_sorted[lower] + (values_sorted[upper] - values_sorted[lower]) * fraction
    )


def validate_queries(queries: Dict[Union[str, int], str]) -> Dict[str, str]:
    """
    Validate and normalize queries.
    Converts query IDs to strings if they are ints.
    """
    if not isinstance(queries, dict):
        raise ValueError("queries must be a dict of query_id => query_text")
    normalized_queries = {}
    for qid, query_text in queries.items():
        if not isinstance(qid, (str, int)):
            raise ValueError("Query ID must be a string or an int.", qid)
        if not isinstance(query_text, str):
            raise ValueError("Query text must be a string.", query_text)
        normalized_queries[str(qid)] = query_text
    return normalized_queries


def validate_qrels(
    qrels: Union[
        Dict[Union[str, int], Union[Set[str], Dict[str, float]]],
        Dict[Union[str, int], str],
    ],
) -> Dict[str, Union[Set[str], Dict[str, float]]]:
    """
    Validate and normalize qrels.
    Converts query IDs to strings if they are ints.
    """
    if not isinstance(qrels, dict):
        raise ValueError(
            "qrels must be a dict of query_id => set/dict of relevant doc_ids or a single doc_id string"
        )
    new_qrels: Dict[str, Union[Set[str], Dict[str, float]]] = {}
    for qid, relevant_docs in qrels.items():
        if not isinstance(qid, (str, int)):
            raise ValueError(
                "Query ID in qrels must be a string or an int.", qid, relevant_docs
            )
        normalized_qid = str(qid)
        if isinstance(relevant_docs, str):
            new_qrels[normalized_qid] = {relevant_docs}
        elif isinstance(relevant_docs, set):
            new_qrels[normalized_qid] = relevant_docs
        elif isinstance(relevant_docs, dict):
            for doc_id, relevance in relevant_docs.items():
                if not isinstance(doc_id, str) or not isinstance(
                    relevance, (int, float)
                ):
                    raise ValueError(
                        f"Relevance scores for query {normalized_qid} must be a dict of string doc_id => numeric relevance."
                    )
                if not 0 <= relevance <= 1:
                    raise ValueError(
                        f"Relevance scores for query {normalized_qid} must be between 0 and 1."
                    )
            new_qrels[normalized_qid] = relevant_docs
        else:
            raise ValueError(
                f"Relevant docs for query {normalized_qid} must be a set, string, or dict."
            )
    return new_qrels


def validate_vespa_query_fn(fn: Callable) -> bool:
    """
    Validates the vespa_query_fn function.

    The function must be callable and accept either 2 or 3 parameters:
        - (query_text: str, top_k: int)
        - or (query_text: str, top_k: int, query_id: Optional[str])

    It must return a dictionary when called with test inputs.

    Returns True if the function takes a query_id parameter, False otherwise.
    """
    if not callable(fn):
        raise ValueError("vespa_query_fn must be callable")

    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if len(params) not in (2, 3):
        raise TypeError("vespa_query_fn must take 2 or 3 parameters")

    # Validate first parameter: query_text
    if (
        params[0].annotation is not inspect.Parameter.empty
        and params[0].annotation is not str
    ):
        raise TypeError("Parameter 'query_text' must be of type str")

    # Validate second parameter: top_k
    if (
        params[1].annotation is not inspect.Parameter.empty
        and params[1].annotation is not int
    ):
        raise TypeError("Parameter 'top_k' must be of type int")

    # If there's a third parameter, validate query_id
    if len(params) == 3:
        third = params[2]
        if third.annotation is not inspect.Parameter.empty and third.annotation not in (
            str,
            Optional[str],
        ):
            raise TypeError("Parameter 'query_id' must be of type str or Optional[str]")
        return True
    else:
        return False


def filter_queries(
    queries: Dict[str, str], relevant_docs: Dict[str, Set[str]]
) -> List[str]:
    """Filter out queries that have no relevant docs"""
    filtered = []
    for qid in queries:
        if qid in relevant_docs and len(relevant_docs[qid]) > 0:
            filtered.append(qid)
    return filtered


def extract_doc_id_from_hit(hit: dict, id_field: str) -> str:
    """Extract document ID from a Vespa hit."""
    if id_field == "":
        full_id = hit.get("id", "")
        if "::" not in full_id:
            # vespa internal id - eg. index:content/0/35c332d6bc52ae1f8378f7b3
            # Trying 'id' field as a fallback
            doc_id = str(hit.get("fields", {}).get("id", ""))
        else:
            doc_id = full_id.split("::")[-1]
    else:
        # doc_id extraction logic
        doc_id = str(hit.get("fields", {}).get(id_field, ""))

    if not doc_id:
        raise ValueError(f"Could not extract doc_id from hit: {hit}")

    return doc_id


def get_id_field_from_hit(hit: dict, id_field: str) -> str:
    """Get the ID field from a Vespa hit."""
    id_val = hit.get("fields", {}).get(id_field, None)
    if id_val is not None:
        return str(id_val)
    else:
        raise ValueError(f"ID field '{id_field}' not found in hit: {hit}")


def calculate_searchtime_stats(searchtimes: List[float]) -> Dict[str, float]:
    """Calculate search time statistics."""
    if not searchtimes:
        return {}
    return {
        "searchtime_avg": mean(searchtimes),
        "searchtime_q50": percentile(searchtimes, 50),
        "searchtime_q90": percentile(searchtimes, 90),
        "searchtime_q95": percentile(searchtimes, 95),
    }


def execute_queries(
    app: Vespa,
    query_bodies: List[dict],
    max_concurrent: int = 10,
) -> Tuple[List[VespaQueryResponse], List[float]]:
    """
    Execute queries and collect timing information.
    Returns the responses and a list of search times.
    """
    responses: List[VespaQueryResponse] = app.query_many(
        query_bodies, max_concurrent=max_concurrent
    )
    extracted_searchtimes: List[float] = []

    for resp in responses:
        if resp.status_code != 200:
            raise ValueError(
                f"Vespa query failed with status code {resp.status_code}, response: {resp.get_json()}"
            )
        # Extract search timing information
        timing = resp.get_json().get("timing", {}).get("searchtime", 0)
        extracted_searchtimes.append(timing)

    return responses, extracted_searchtimes


def write_csv(
    metrics: Dict[str, float],
    searchtime_stats: Dict[str, float],
    csv_file: str,
    csv_dir: Optional[str],
    name: str,
) -> None:
    """Write metrics to CSV file."""
    csv_path = csv_file
    if csv_dir is not None:
        csv_path = os.path.join(csv_dir, csv_path)

    combined_metrics = {**metrics, **searchtime_stats}
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if write_header:
            header = sorted(combined_metrics.keys())
            header.insert(0, "name")  # extra column for "run name"
            writer.writerow(header)
        row_keys = sorted(combined_metrics.keys())
        row = [name] + [combined_metrics[k] for k in row_keys]
        writer.writerow(row)

    logger.info(f"Wrote IR evaluation metrics and search times to {csv_path}")


def log_metrics(name: str, metrics: Dict[str, float]) -> None:
    """Log metrics with appropriate formatting."""
    logger.info(f"Vespa IR evaluation on {name}")
    for metric_name, value in metrics.items():
        if (
            metric_name.startswith("accuracy")
            or metric_name.startswith("precision")
            or metric_name.startswith("recall")
        ):
            logger.info(f"{metric_name}: {value * 100:.2f}%")
        else:
            logger.info(f"{metric_name}: {value:.4f}")


class VespaEvaluatorBase(ABC):
    """
    Abstract base class for Vespa evaluators providing initialization and interface.
    """

    def __init__(
        self,
        queries: Dict[str, str],
        relevant_docs: Union[
            Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]
        ],
        vespa_query_fn: Callable[[str, int, Optional[str]], dict],
        app: Vespa,
        name: str = "",
        id_field: str = "",
        write_csv: bool = False,
        csv_dir: Optional[str] = None,
    ):
        self.id_field = id_field
        validated_queries = validate_queries(queries)
        self._vespa_query_fn_takes_query_id = validate_vespa_query_fn(vespa_query_fn)
        validated_relevant_docs = validate_qrels(relevant_docs)

        # Filter out any queries that have no relevant docs
        self.queries_ids = filter_queries(validated_queries, validated_relevant_docs)

        self.queries = [validated_queries[qid] for qid in self.queries_ids]
        self.relevant_docs = validated_relevant_docs

        self.searchtimes: List[float] = []
        self.vespa_query_fn: Callable = vespa_query_fn
        self.app = app

        self.name = name
        self.write_csv = write_csv
        self.csv_dir = csv_dir

        self.primary_metric: Optional[str] = None

        # Generate datetime string for filenames
        now = datetime.now()
        self.dt_string = now.strftime("%Y%m%d_%H%M%S")

        self.csv_file: str = f"Vespa-evaluation_{name}_{self.dt_string}_results.csv"

    @property
    def default_body(self):
        return {
            "timeout": "5s",
            "presentation.timing": True,
        }

    @abstractmethod
    def run(self) -> Dict[str, float]:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the run method")

    def __call__(self) -> Dict[str, float]:
        """Make the evaluator callable."""
        return self.run()


class VespaEvaluator(VespaEvaluatorBase):
    """
    Evaluate retrieval performance on a Vespa application.

    This class:

    - Iterates over queries and issues them against your Vespa application.
    - Retrieves top-k documents per query (with k = max of your IR metrics).
    - Compares the retrieved documents with a set of relevant document ids.
    - Computes IR metrics: Accuracy@k, Precision@k, Recall@k, MRR@k, NDCG@k, MAP@k.
    - Logs vespa search times for each query.
    - Logs/returns these metrics.
    - Optionally writes out to CSV.

    Note: The 'id_field' needs to be marked as an attribute in your Vespa schema, so filtering can be done on it.


    Example usage:
        ```python
        from vespa.application import Vespa
        from vespa.evaluation import VespaEvaluator

        queries = {
            "q1": "What is the best GPU for gaming?",
            "q2": "How to bake sourdough bread?",
            # ...
        }
        relevant_docs = {
            "q1": {"d12", "d99"},
            "q2": {"d101"},
            # ...
        }
        # relevant_docs can also be a dict of query_id => single relevant doc_id
        # relevant_docs = {
        #     "q1": "d12",
        #     "q2": "d101",
        #     # ...
        # }
        # Or, relevant_docs can be a dict of query_id => map of doc_id => relevance
        # relevant_docs = {
        #     "q1": {"d12": 1, "d99": 0.1},
        #     "q2": {"d101": 0.01},
        #     # ...
        # Note that for non-binary relevance, the relevance values should be in [0, 1], and that
        # only the nDCG metric will be computed.

        def my_vespa_query_fn(query_text: str, top_k: int) -> dict:
            return {
                "yql": 'select * from sources * where userInput("' + query_text + '");',
                "hits": top_k,
                "ranking": "your_ranking_profile",
            }

        app = Vespa(url="http://localhost", port=8080)

        evaluator = VespaEvaluator(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=my_vespa_query_fn,
            app=app,
            name="test-run",
            accuracy_at_k=[1, 3, 5],
            precision_recall_at_k=[1, 3, 5],
            mrr_at_k=[10],
            ndcg_at_k=[10],
            map_at_k=[100],
            write_csv=True
        )

        results = evaluator()
        print("Primary metric:", evaluator.primary_metric)
        print("All results:", results)
        ```

    Args:
        queries (Dict[str, str]): A dictionary where keys are query IDs and values are query strings.
        relevant_docs (Union[Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]]):
            A dictionary mapping query IDs to their relevant document IDs.
            Can be a set of doc IDs for binary relevance, a dict of doc_id to relevance score (float between 0 and 1)
            for graded relevance, or a single doc_id string.
        vespa_query_fn (Callable[[str, int, Optional[str]], dict]): A function that takes a query string,
            the number of hits to retrieve (top_k), and an optional query_id, and returns a Vespa query body dictionary.
        app (Vespa): An instance of the Vespa application.
        name (str, optional): A name for this evaluation run. Defaults to "".
        id_field (str, optional): The field name in the Vespa hit that contains the document ID.
            If empty, it tries to infer the ID from the 'id' field or 'fields.id'. Defaults to "".
        accuracy_at_k (List[int], optional): List of k values for which to compute Accuracy@k.
            Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int], optional): List of k values for which to compute Precision@k and Recall@k.
            Defaults to [1, 3, 5, 10].
        mrr_at_k (List[int], optional): List of k values for which to compute MRR@k. Defaults to [10].
        ndcg_at_k (List[int], optional): List of k values for which to compute NDCG@k. Defaults to [10].
        map_at_k (List[int], optional): List of k values for which to compute MAP@k. Defaults to [100].
        write_csv (bool, optional): Whether to write the evaluation results to a CSV file. Defaults to False.
        csv_dir (Optional[str], optional): Directory to save the CSV file. Defaults to None (current directory).
    """

    def __init__(
        self,
        queries: Dict[str, str],
        relevant_docs: Union[
            Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]
        ],
        vespa_query_fn: Callable[[str, int, Optional[str]], dict],
        app: Vespa,
        name: str = "",
        id_field: str = "",
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        map_at_k: List[int] = [100],
        write_csv: bool = False,
        csv_dir: Optional[str] = None,
    ):
        super().__init__(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=vespa_query_fn,
            app=app,
            name=name,
            id_field=id_field,
            write_csv=write_csv,
            csv_dir=csv_dir,
        )

        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.map_at_k = map_at_k

        # We'll collect metrics in a single pass, so define them up front.
        self.csv_headers = [
            "accuracy@{}",
            "precision@{}",
            "recall@{}",
            "mrr@{}",
            "ndcg@{}",
            "map@{}",
        ]

    def _find_max_k(self):
        """Find the maximum k value across all metrics."""
        return max(
            max(self.accuracy_at_k) if self.accuracy_at_k else 0,
            max(self.precision_recall_at_k) if self.precision_recall_at_k else 0,
            max(self.mrr_at_k) if self.mrr_at_k else 0,
            max(self.ndcg_at_k) if self.ndcg_at_k else 0,
            max(self.map_at_k) if self.map_at_k else 0,
        )

    def run(self) -> Dict[str, float]:
        """
        Executes the evaluation by running queries and computing IR metrics.

        This method:
        1. Executes all configured queries against the Vespa application.
        2. Collects search results and timing information.
        3. Computes the configured IR metrics (Accuracy@k, Precision@k, Recall@k, MRR@k, NDCG@k, MAP@k).
        4. Records search timing statistics.
        5. Logs results and optionally writes them to CSV.

        Returns:
            dict: A dictionary containing:
                - IR metrics with names like "accuracy@k", "precision@k", etc.
                - Search time statistics ("searchtime_avg", "searchtime_q50", etc.).
                The values are floats between 0 and 1 for metrics and in seconds for timing.

        Example:
            ```python
            {
                "accuracy@1": 0.75,
                "ndcg@10": 0.68,
                "searchtime_avg": 0.0123,
                ...
            }
            ```
        """
        max_k = self._find_max_k()

        logger.info(f"Starting VespaEvaluator on {self.name}")
        logger.info(f"Number of queries: {len(self.queries_ids)}; max_k = {max_k}")

        # Build query bodies using the provided vespa_query_fn
        query_bodies = []

        for qid, query_text in zip(self.queries_ids, self.queries):
            if self._vespa_query_fn_takes_query_id:
                query_body: dict = self.vespa_query_fn(query_text, max_k, qid)
            else:
                query_body: dict = self.vespa_query_fn(query_text, max_k)
            if not isinstance(query_body, dict):
                raise ValueError(
                    f"vespa_query_fn must return a dict, got: {type(query_body)}"
                )
            # Add default body parameters only if not already specified
            for key, value in self.default_body.items():
                if key not in query_body:
                    query_body[key] = value
            query_bodies.append(query_body)
            logger.debug(f"Querying Vespa with: {query_body}")

        responses, new_searchtimes = execute_queries(self.app, query_bodies)
        self.searchtimes.extend(new_searchtimes)

        queries_result_list = []
        for resp in responses:
            hits = resp.hits or []
            top_hit_list = []
            for hit in hits[:max_k]:
                doc_id = extract_doc_id_from_hit(hit, self.id_field)
                score = float(hit.get("relevance", float("nan")))
                if math.isnan(score):
                    raise ValueError(f"Could not extract relevance from hit: {hit}")
                top_hit_list.append((doc_id, score))

            queries_result_list.append(top_hit_list)
        metrics = self._compute_metrics(queries_result_list)
        searchtime_stats = calculate_searchtime_stats(self.searchtimes)
        metrics.update(searchtime_stats)

        if not self.primary_metric:
            if self.ndcg_at_k:
                best_k = max(self.ndcg_at_k)
                self.primary_metric = f"ndcg@{best_k}"
            else:
                # fallback to some default
                self.primary_metric = "accuracy@1" if self.accuracy_at_k else "map@100"

        log_metrics(self.name, metrics)

        if self.write_csv:
            write_csv(metrics, searchtime_stats, self.csv_file, self.csv_dir, self.name)

        return metrics

    def _compute_metrics(self, queries_result_list):
        num_queries = len(queries_result_list)
        # Infer graded relevance on the fly instead of storing as a class variable.
        graded = bool(self.relevant_docs) and isinstance(
            next(iter(self.relevant_docs.values())), dict
        )

        if graded:
            ndcg_at_k_list = {k: [] for k in self.ndcg_at_k}
            for query_idx, top_hits in enumerate(queries_result_list):
                qid = self.queries_ids[query_idx]
                # For graded relevance, 'relevant_docs' is a list of dicts: doc_id -> grade
                relevant: Dict[str, float] = self.relevant_docs[qid]
                for k_val in self.ndcg_at_k:
                    predicted_relevance = [
                        relevant.get(doc_id, 0.0) for doc_id, _ in top_hits[:k_val]
                    ]
                    dcg_pred = self._dcg_at_k(predicted_relevance, k_val)
                    # Ideal ranking is the sorted graded scores in descending order
                    ideal_relevances = sorted(relevant.values(), reverse=True)[:k_val]
                    dcg_true = self._dcg_at_k(ideal_relevances, k_val)
                    ndcg_val = dcg_pred / dcg_true if dcg_true > 0 else 0.0
                    ndcg_at_k_list[k_val].append(ndcg_val)
            metrics = {f"ndcg@{k}": mean(ndcg_at_k_list[k]) for k in self.ndcg_at_k}
            return metrics

        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precision_at_k_list = {k: [] for k in self.precision_recall_at_k}
        recall_at_k_list = {k: [] for k in self.precision_recall_at_k}
        mrr_at_k = {k: 0.0 for k in self.mrr_at_k}
        ndcg_at_k_list = {k: [] for k in self.ndcg_at_k}
        map_at_k_list = {k: [] for k in self.map_at_k}

        for query_idx, top_hits in enumerate(queries_result_list):
            qid = self.queries_ids[query_idx]
            relevant = self.relevant_docs[qid]

            # Accuracy@K
            for k_val in self.accuracy_at_k:
                found_correct = any(
                    doc_id in relevant for doc_id, _ in top_hits[:k_val]
                )
                if found_correct:
                    num_hits_at_k[k_val] += 1

            # Precision@K, Recall@K
            for k_val in self.precision_recall_at_k:
                k_hits = top_hits[:k_val]
                num_correct = sum(1 for doc_id, _ in k_hits if doc_id in relevant)
                precision_at_k_list[k_val].append(num_correct / k_val)
                recall_at_k_list[k_val].append(
                    num_correct / len(relevant) if len(relevant) > 0 else 0.0
                )

            # MRR@K
            for k_val in self.mrr_at_k:
                reciprocal_rank = 0.0
                for rank, (doc_id, _) in enumerate(top_hits[:k_val]):
                    if doc_id in relevant:
                        reciprocal_rank = 1.0 / (rank + 1)
                        break
                mrr_at_k[k_val] += reciprocal_rank

            # NDCG@K
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if doc_id in relevant else 0 for doc_id, _ in top_hits[:k_val]
                ]
                # Here we assume each relevant doc has relevance=1
                true_relevances = [1] * len(relevant)
                dcg_pred = self._dcg_at_k(predicted_relevance, k_val)
                dcg_true = self._dcg_at_k(
                    true_relevances, min(k_val, len(true_relevances))
                )
                ndcg_val = dcg_pred / dcg_true if dcg_true > 0 else 0.0
                ndcg_at_k_list[k_val].append(ndcg_val)

            # MAP@K
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0.0
                top_k_hits = top_hits[:k_val]
                for rank, (doc_id, _) in enumerate(top_k_hits, start=1):
                    if isinstance(relevant, dict):
                        if doc_id in relevant:
                            num_correct += 1
                            sum_precisions += (
                                relevant[doc_id] / rank
                            )  # Use relevance score
                    elif doc_id in relevant:
                        num_correct += 1
                        sum_precisions += num_correct / rank
                denom = min(
                    k_val,
                    len(relevant)
                    if isinstance(relevant, set)
                    else len(relevant.keys()),
                )
                avg_precision = sum_precisions / denom if denom > 0 else 0.0
                map_at_k_list[k_val].append(avg_precision)

        # Final metric averages
        metrics = {}
        # accuracy
        for k_val in self.accuracy_at_k:
            metrics[f"accuracy@{k_val}"] = num_hits_at_k[k_val] / num_queries

        # precision and recall
        for k_val in self.precision_recall_at_k:
            p_val = mean(precision_at_k_list[k_val])
            r_val = mean(recall_at_k_list[k_val])
            metrics[f"precision@{k_val}"] = p_val
            metrics[f"recall@{k_val}"] = r_val

        # MRR
        for k_val in self.mrr_at_k:
            metrics[f"mrr@{k_val}"] = mrr_at_k[k_val] / num_queries

        # nDCG
        for k_val in self.ndcg_at_k:
            metrics[f"ndcg@{k_val}"] = mean(ndcg_at_k_list[k_val])

        # MAP
        for k_val in self.map_at_k:
            metrics[f"map@{k_val}"] = mean(map_at_k_list[k_val])

        return metrics

    def _dcg_at_k(self, relevances, k):
        """
        Compute Discounted Cumulative Gain for the top-k.
        """
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], start=1):
            # Use math.log2 instead of np.log2
            dcg += rel / (math.log2(i + 1) if i > 1 else 1.0)
        return dcg


class VespaMatchEvaluator(VespaEvaluatorBase):
    """
    Evaluate recall in the match-phase over a set of queries for a Vespa application.

    This class:

    - Iterates over queries and issues them against your Vespa application.
    - Sends one query with limit 0 to get the number of matched documents.
    - Sends one query with recall-parameter set according to the provided relevant documents.
    - Compares the retrieved documents with a set of relevant document ids.
    - Logs vespa search times for each query.
    - Logs/returns these metrics.
    - Optionally writes out to CSV.

    Note: It is recommended to use a rank profile without any first-phase (and second-phase) ranking if you care about speed of evaluation run.
    If you do so, you need to make sure that the rank profile you use has the same inputs. For example, if you want to evaluate a YQL query including nearestNeighbor-operator, your rank-profile needs to define the corresponding input tensor.
    You must also either provide the query tensor or define it as input (e.g 'input.query(embedding)=embed(@query)') in your Vespa query function.
    Also note that the 'id_field' needs to be marked as an attribute in your Vespa schema, so filtering can be done on it.
    Example usage:
        ```python
        from vespa.application import Vespa
        from vespa.evaluation import VespaEvaluator

        queries = {
            "q1": "What is the best GPU for gaming?",
            "q2": "How to bake sourdough bread?",
            # ...
        }
        relevant_docs = {
            "q1": {"d12", "d99"},
            "q2": {"d101"},
            # ...
        }
        # relevant_docs can also be a dict of query_id => single relevant doc_id
        # relevant_docs = {
        #     "q1": "d12",
        #     "q2": "d101",
        #     # ...
        # }
        # Or, relevant_docs can be a dict of query_id => map of doc_id => relevance
        # relevant_docs = {
        #     "q1": {"d12": 1, "d99": 0.1},
        #     "q2": {"d101": 0.01},
        #     # ...

        def my_vespa_query_fn(query_text: str, top_k: int) -> dict:
            return {
                "yql": 'select * from sources * where userInput("' + query_text + '");',
                "hits": top_k,
                "ranking": "your_ranking_profile",
            }

        app = Vespa(url="http://localhost", port=8080)

        evaluator = VespaMatchEvaluator(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=my_vespa_query_fn,
            app=app,
            name="test-run",
            id_field="id",
            write_csv=True,
            write_verbose=True,
        )

        results = evaluator()
        print("Primary metric:", evaluator.primary_metric)
        print("All results:", results)
        ```

    Args:
        queries (Dict[str, str]): A dictionary where keys are query IDs and values are query strings.
        relevant_docs (Union[Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]]):
            A dictionary mapping query IDs to their relevant document IDs.
            Can be a set of doc IDs for binary relevance, or a single doc_id string.
            Graded relevance (dict of doc_id to relevance score) is not supported for match evaluation.
        vespa_query_fn (Callable[[str, int, Optional[str]], dict]): A function that takes a query string,
            the number of hits to retrieve (top_k), and an optional query_id, and returns a Vespa query body dictionary.
        app (Vespa): An instance of the Vespa application.
        name (str, optional): A name for this evaluation run. Defaults to "".
        id_field (str, optional): The field name in the Vespa hit that contains the document ID.
            If empty, it tries to infer the ID from the 'id' field or 'fields.id'. Defaults to "".
        write_csv (bool, optional): Whether to write the summary evaluation results to a CSV file. Defaults to False.
        write_verbose (bool, optional): Whether to write detailed query-level results to a separate CSV file.
            Defaults to False.
        csv_dir (Optional[str], optional): Directory to save the CSV files. Defaults to None (current directory).
    """

    def __init__(
        self,
        queries: Dict[str, str],
        relevant_docs: Union[
            Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]
        ],
        vespa_query_fn: Callable[[str, int, Optional[str]], dict],
        app: Vespa,
        id_field: str,
        name: str = "",
        rank_profile: str = "unranked",
        write_csv: bool = False,
        write_verbose: bool = False,
        csv_dir: Optional[str] = None,
    ):
        if not id_field:
            raise ValueError(
                "The 'id_field' parameter is required for VespaMatchEvaluator. "
                "Please specify the field name that contains document IDs in your Vespa schema "
                "(e.g., id_field='id'). This field must be defined as an attribute in your schema."
            )

        super().__init__(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=vespa_query_fn,
            app=app,
            name=name,
            id_field=id_field,
            write_csv=write_csv,
            csv_dir=csv_dir,
        )
        self.write_verbose = write_verbose
        if self.write_verbose:
            # Use the dt_string from the base class
            self.verbose_csv_file = (
                f"Vespa-match-evaluation_{name}_{self.dt_string}_verbose.csv"
            )

    def _write_verbose_csv(self, verbose_data: List[Dict]):
        """Write verbose query-level results to CSV file."""
        if not verbose_data:
            return

        csv_path = self.verbose_csv_file
        if self.csv_dir is not None:
            csv_path = os.path.join(self.csv_dir, csv_path)

        write_header = not os.path.exists(csv_path)

        with open(csv_path, mode="a", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            if write_header:
                header = [
                    "name",
                    "query_id",
                    "query_text",
                    "yql",
                    "num_matched",  # Changed from total_matched
                    "relevant_docs_count",
                    "matched_relevant_count",
                    "recall",
                    "ids_matched",
                    "ids_not_matched",
                    "searchtime_limit_query",  # Added
                    "searchtime_recall_query",  # Added
                ]
                writer.writerow(header)

            for row_data in verbose_data:
                writer.writerow(
                    [
                        self.name,
                        row_data["query_id"],
                        row_data["query_text"],
                        row_data["yql"],
                        row_data[
                            "total_matched"
                        ],  # This corresponds to num_matched (totalCount)
                        row_data["relevant_docs_count"],
                        row_data["matched_relevant_count"],
                        row_data["recall"],
                        "; ".join(row_data["ids_matched"]),
                        "; ".join(row_data["ids_not_matched"]),
                        row_data.get("searchtime_limit_query", float("nan")),  # Added
                        row_data.get("searchtime_recall_query", float("nan")),  # Added
                    ]
                )

        logger.info(f"Wrote verbose match evaluation results to {csv_path}")

    @staticmethod
    def create_grouping_filter(
        yql: str, id_field: str, relevant_ids: Union[str, List[str]]
    ) -> str:
        """
        Create a grouping filter to append Vespa YQL queries to limit results to relevant documents.
        | all( group(id_field) filter(regex("<regex matching all ids>", id_field)) each(output(count())))

        Parameters:
        yql (str): The base YQL query string.
        id_field (str): The field name in the Vespa hit that contains the document ID.
        relevant_ids (list[str]): List of relevant document IDs to include in the filter.

        Returns:
        str: The modified YQL query string with the grouping filter applied.
        """
        ids = [relevant_ids] if isinstance(relevant_ids, str) else list(relevant_ids)
        if not ids:
            raise ValueError("relevant_ids must contain at least one value.")
        escaped = [re.escape(doc_id) for doc_id in ids]
        pattern = "|".join(escaped)
        if len(escaped) > 1:
            pattern = f"(?:{pattern})"
        # Need to escape backslashes again for YQL string
        pattern = pattern.replace("\\", "\\\\")
        grouping_clause = f' | all( group({id_field}) filter(regex("^{pattern}$", {id_field})) each(output(count())) )'
        modified_yql = yql.strip().rstrip(";")
        return modified_yql + grouping_clause

    @staticmethod
    def extract_matched_ids(resp: VespaQueryResponse, id_field: str) -> Set[str]:
        """
        Extract matched document IDs from Vespa query response hits.
        Parameters:
        resp (VespaQueryResponse): The Vespa query response object.
        id_field (str): The field name in the Vespa hit that contains the document ID

        Returns:
        Set[str]: A set of matched document IDs.
        """
        # Navigate safely through the nested structure to get group hits
        root = resp.get_json().get("root", {})
        first_level_children = root.get("children", [])
        if not first_level_children:
            group_hits = []
        else:
            second_level_children = first_level_children[0].get("children", [])
            if not second_level_children:
                group_hits = []
            else:
                group_hits = second_level_children[0].get("children", [])
        matched_ids = set()
        for child in group_hits:
            val = child.get("value", None)
            if val is not None:
                matched_ids.add(val)
        return matched_ids

    def run(self) -> Dict[str, float]:
        """
        Executes the match-phase recall evaluation.

        This method:
        1. Sends a grouping query to see which of the relevant documents were matched, and get totalCount.
        3. Computes recall metrics and match statistics.
        4. Logs results and optionally writes them to CSV.

        Returns:
            dict: A dictionary containing recall metrics, match statistics, and search time statistics.

        Example:
            ```python
            {
                "match_recall": 0.85,
                "total_relevant_docs": 150,
                "total_matched_relevant": 128,
                "avg_matched_per_query": 45.2,
                "searchtime_avg": 0.015,
                ...
            }
            ```
        """
        logger.info(f"Starting VespaMatchEvaluator on {self.name}")
        logger.info(f"Number of queries: {len(self.queries_ids)}")

        # Clear searchtimes from any previous runs if the instance is reused,
        # though typically a new instance is created for each evaluation.
        self.searchtimes: List[float] = []

        # Step 2: Recall query with relevant documents
        recall_query_bodies = []
        for qid, query_text in zip(self.queries_ids, self.queries):
            relevant_docs = self.relevant_docs[qid]
            if self._vespa_query_fn_takes_query_id:
                query_body: dict = self.vespa_query_fn(
                    query_text, len(relevant_docs), qid
                )
            else:
                query_body: dict = self.vespa_query_fn(query_text, len(relevant_docs))

            if not isinstance(query_body, dict):
                raise ValueError(
                    f"vespa_query_fn must return a dict, got: {type(query_body)}"
                )

            # Add default body parameters only if not already specified
            for key, value in self.default_body.items():
                if key not in query_body:
                    query_body[key] = value
            # See https://docs.vespa.ai/en/reference/query-api-reference.html#grouping.defaultMaxGroups
            query_body["grouping.defaultMaxHits"] = -1  # Disable hits
            query_body["grouping.defaultMaxGroups"] = len(relevant_docs)
            if "hits" in query_body:
                del query_body["hits"]  # Remove hits parameter if present
            # Add grouping clause based on relevant docs
            relevant_docs = list(relevant_docs)
            query_body["yql"] = self.create_grouping_filter(
                query_body["yql"], self.id_field, relevant_docs
            )
            # Set rank_profile
            if "ranking" not in query_body:
                query_body["ranking"] = "unranked"  # Set to unranked if not specified
            recall_query_bodies.append(query_body)
        # Execute recall queries
        recall_responses, searchtimes_recall_queries = execute_queries(
            self.app, recall_query_bodies
        )
        matched_docs_counts = [
            resp.get_json().get("root", {}).get("fields", {}).get("totalCount", 0)
            for resp in recall_responses
        ]
        logger.info(f"Total matched documents per query: {matched_docs_counts}")

        self.searchtimes.extend(searchtimes_recall_queries)

        # Compute comprehensive match metrics
        total_relevant_docs = 0
        total_matched_relevant = 0
        all_recalls = []
        verbose_data = []

        for idx, (qid, resp) in enumerate(zip(self.queries_ids, recall_responses)):
            relevant_docs = self.relevant_docs[qid]
            query_text = self.queries[idx]
            if isinstance(relevant_docs, set):
                # Binary relevance
                num_relevant = len(relevant_docs)
                total_relevant_docs += num_relevant
                # Extract retrieved document IDs
                retrieved_ids = self.extract_matched_ids(resp, self.id_field)
                # Calculate matches
                matched_relevant_ids = retrieved_ids & relevant_docs
                not_matched_ids = relevant_docs - retrieved_ids
                matched_relevant_count = len(matched_relevant_ids)
                total_matched_relevant += matched_relevant_count

                # Calculate recall for this query
                query_recall = (
                    matched_relevant_count / num_relevant if num_relevant > 0 else 0.0
                )
                all_recalls.append(query_recall)

                # Store verbose data if requested
                if self.write_verbose:
                    verbose_data.append(
                        {
                            "query_id": qid,
                            "query_text": query_text,
                            "yql": recall_query_bodies[idx]["yql"],
                            "total_matched": matched_docs_counts[idx],
                            "relevant_docs_count": num_relevant,
                            "matched_relevant_count": matched_relevant_count,
                            "recall": query_recall,
                            "ids_matched": sorted(list(matched_relevant_ids)),
                            "ids_not_matched": sorted(list(not_matched_ids)),
                            "searchtime_recall_query": searchtimes_recall_queries[idx]
                            if idx < len(searchtimes_recall_queries)
                            else float("nan"),
                        }
                    )

            elif isinstance(relevant_docs, dict):
                # Graded relevance - not supported for match evaluation
                raise ValueError(
                    "Graded relevance is not supported in VespaMatchEvaluator. "
                    "Please use VespaEvaluator for graded relevance evaluation."
                )
            else:
                raise ValueError(
                    f"Unsupported type of relevant docs for query {qid}: {type(relevant_docs)}"
                )

        # Calculate comprehensive metrics
        metrics = {
            "match_recall": total_matched_relevant / total_relevant_docs
            if total_relevant_docs > 0
            else 0.0,
            "avg_recall_per_query": mean(all_recalls),
            "total_relevant_docs": total_relevant_docs,
            "total_matched_relevant": total_matched_relevant,
            "avg_matched_per_query": mean(matched_docs_counts),
            "total_queries": len(self.queries_ids),
        }

        # Add search time statistics
        searchtime_stats = calculate_searchtime_stats(self.searchtimes)
        metrics.update(searchtime_stats)

        # Set primary metric
        self.primary_metric = "match_recall"

        # Log comprehensive results
        logger.info(f"Match Evaluation Results for {self.name}:")
        logger.info(f"  Overall Match Recall: {metrics['match_recall']:.4f}")
        logger.info(
            f"  Average Recall per Query: {metrics['avg_recall_per_query']:.4f}"
        )
        logger.info(f"  Total Relevant Documents: {metrics['total_relevant_docs']}")
        logger.info(f"  Total Matched Relevant: {metrics['total_matched_relevant']}")
        logger.info(
            f"  Average Matched per Query: {metrics['avg_matched_per_query']:.2f}"
        )

        log_metrics(self.name, searchtime_stats)

        if self.write_csv:
            write_csv(metrics, searchtime_stats, self.csv_file, self.csv_dir, self.name)

        if self.write_verbose:
            self._write_verbose_csv(verbose_data)

        return metrics


class VespaCollectorBase(ABC):
    """
    Abstract base class for Vespa training data collectors providing initialization and interface.
    """

    def __init__(
        self,
        queries: Dict[str, str],
        relevant_docs: Union[
            Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]
        ],
        vespa_query_fn: Callable[[str, int, Optional[str]], dict],
        app: Vespa,
        id_field: str,
        name: str = "",
        csv_dir: Optional[str] = None,
        random_hits_strategy: Union[
            RandomHitsSamplingStrategy, str
        ] = RandomHitsSamplingStrategy.RATIO,
        random_hits_value: Union[float, int] = 1.0,
        max_random_hits_per_query: Optional[int] = None,
        collect_matchfeatures: bool = True,
        collect_rankfeatures: bool = False,
        collect_summaryfeatures: bool = False,
        write_csv: bool = True,
    ):
        """
        Initialize the VespaFeatureCollector.

        Args:
            queries: Dictionary mapping query IDs to query strings
            relevant_docs: Dictionary mapping query IDs to relevant document IDs
            vespa_query_fn: Function to generate Vespa query bodies
            app: Vespa application instance
            id_field: Field name containing document IDs in Vespa hits (must be defined as an attribute in the schema)
            name: Name for this collection run
            csv_dir: Directory to save CSV files
            random_hits_strategy: Strategy for sampling random hits - either "ratio" or "fixed"
                - RATIO: Sample random hits as a ratio of relevant docs
                - FIXED: Sample a fixed number of random hits per query
            random_hits_value: Value for the sampling strategy
                - For RATIO: Ratio value (e.g., 1.0 = equal, 2.0 = twice as many random hits)
                - For FIXED: Fixed number of random hits per query
            max_random_hits_per_query: Optional maximum limit on random hits per query
                (only applies when using RATIO strategy to prevent excessive sampling)
            collect_matchfeatures: Whether to collect match features
            collect_rankfeatures: Whether to collect rank features
            collect_summaryfeatures: Whether to collect summary features
            write_csv: Whether to write results to CSV file
        """
        if not id_field:
            raise ValueError(
                "id_field is required and cannot be empty. "
                "Please specify the field name that contains document IDs in your Vespa hits. "
                "This field must be defined as an attribute in your Vespa schema."
            )

        self.id_field = id_field

        # Handle strategy parameter - support both enum and string
        if isinstance(random_hits_strategy, str):
            try:
                self.random_hits_strategy = RandomHitsSamplingStrategy(
                    random_hits_strategy.lower()
                )
            except ValueError:
                raise ValueError(
                    f"Invalid random_hits_strategy '{random_hits_strategy}'. "
                    f"Must be one of: {[s.value for s in RandomHitsSamplingStrategy]}"
                )
        else:
            self.random_hits_strategy = random_hits_strategy

        # Validate random_hits_value based on strategy
        if self.random_hits_strategy == RandomHitsSamplingStrategy.RATIO:
            if not isinstance(random_hits_value, (int, float)) or random_hits_value < 0:
                raise ValueError(
                    "For RATIO strategy, random_hits_value must be a non-negative number"
                )
            self.random_hits_ratio = float(random_hits_value)
        elif self.random_hits_strategy == RandomHitsSamplingStrategy.FIXED:
            if not isinstance(random_hits_value, int) or random_hits_value < 0:
                raise ValueError(
                    "For FIXED strategy, random_hits_value must be a non-negative integer"
                )
            self.random_hits_fixed = int(random_hits_value)

        self.max_random_hits_per_query = max_random_hits_per_query
        if (
            self.max_random_hits_per_query is not None
            and self.max_random_hits_per_query < 0
        ):
            raise ValueError("max_random_hits_per_query must be non-negative")

        self.collect_matchfeatures = collect_matchfeatures
        self.collect_rankfeatures = collect_rankfeatures
        self.collect_summaryfeatures = collect_summaryfeatures
        self.write_csv = write_csv

        # Log the sampling strategy for user understanding
        if self.random_hits_strategy == RandomHitsSamplingStrategy.RATIO:
            strategy_desc = f"RATIO strategy with ratio={self.random_hits_ratio}"
            if self.max_random_hits_per_query is not None:
                strategy_desc += (
                    f" (max {self.max_random_hits_per_query} random hits per query)"
                )
        else:
            strategy_desc = (
                f"FIXED strategy with {self.random_hits_fixed} random hits per query"
            )
        logger.info(f"Random hits sampling strategy: {strategy_desc}")
        validated_queries = validate_queries(queries)
        self._vespa_query_fn_takes_query_id = validate_vespa_query_fn(vespa_query_fn)
        validated_relevant_docs = validate_qrels(relevant_docs)

        # Filter out any queries that have no relevant docs
        self.queries_ids = filter_queries(validated_queries, validated_relevant_docs)

        self.queries = [validated_queries[qid] for qid in self.queries_ids]
        self.relevant_docs = validated_relevant_docs

        self.searchtimes: List[float] = []
        self.vespa_query_fn: Callable = vespa_query_fn
        self.app = app

        self.name = name
        self.csv_dir = csv_dir

        # Generate datetime string for filenames
        now = datetime.now()
        self.dt_string = now.strftime("%Y%m%d_%H%M%S")

        csv_filename = f"Vespa-training-data_{name}_{self.dt_string}.csv"
        if csv_dir:
            import os

            self.csv_file: str = os.path.join(csv_dir, csv_filename)
        else:
            self.csv_file: str = csv_filename

    @property
    def default_body(self):
        return {
            "timeout": "5s",
            "presentation.timing": True,
        }

    @abstractmethod
    def collect(self) -> Union[None, Dict[str, Union[List[Dict], List[str]]]]:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the collect method")

    def __call__(self) -> None:
        """Make the collector callable."""
        self.collect()


class VespaFeatureCollector(VespaCollectorBase):
    """
    Collects training data for retrieval tasks from a Vespa application.

    This class:

    - Iterates over queries and issues them against your Vespa application.
    - Retrieves top-k documents per query.
    - Samples random hits based on the specified strategy.
    - Compiles a CSV file with query-document pairs and their relevance labels.

    Important: If you want to sample random hits, you need to make sure that the rank profile you define in your `vespa_query_fn` has a ranking expression that
    reflects this. See [docs](https://docs.vespa.ai/en/tutorials/text-search-ml.html#get-random-hits) for example.
    In this case, be aware that the `relevance_score` value in the returned results (or CSV) will be of no value.
    This will only have meaning if you use this to collect features for relevant documents only.

    Example usage:
        ```python
        from vespa.application import Vespa
        from vespa.evaluation import VespaFeatureCollector

        queries = {
            "q1": "What is the best GPU for gaming?",
            "q2": "How to bake sourdough bread?",
            # ...
        }
        relevant_docs = {
            "q1": {"d12", "d99"},
            "q2": {"d101"},
            # ...
        }

        def my_vespa_query_fn(query_text: str, top_k: int) -> dict:
            return {
                "yql": 'select * from sources * where userInput("' + query_text + '");',
                "hits": 10,  # Do not make use of top_k here.
                "ranking": "your_ranking_profile", # This should have `random` as ranking expression
            }

        app = Vespa(url="http://localhost", port=8080)

        collector = VespaFeatureCollector(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=my_vespa_query_fn,
            app=app,
            id_field="id",  # Field in Vespa hit that contains the document ID (must be an attribute)
            name="retrieval-data-collection",
            csv_dir="/path/to/save/csv",
            random_hits_strategy="ratio",  # or RandomHitsSamplingStrategy.RATIO
            random_hits_value=1.0,  # Sample equal number of random hits to relevant docs
            max_random_hits_per_query=100,  # Optional: cap random hits per query
            collect_matchfeatures=True,  # Collect match features from rank profile
            collect_rankfeatures=False,  # Skip traditional rank features
            collect_summaryfeatures=False,  # Skip summary features
        )

        collector()
        ```

    **Alternative Usage Examples:**

    ```python
    # Example 1: Fixed number of random hits per query
    collector = VespaFeatureCollector(
        queries=queries,
        relevant_docs=relevant_docs,
        vespa_query_fn=my_vespa_query_fn,
        app=app,
        id_field="id",  # Required field name
        random_hits_strategy="fixed",
        random_hits_value=50,  # Always sample 50 random hits per query
    )

    # Example 2: Ratio-based with a cap
    collector = VespaFeatureCollector(
        queries=queries,
        relevant_docs=relevant_docs,
        vespa_query_fn=my_vespa_query_fn,
        app=app,
        id_field="id",  # Required field name
        random_hits_strategy="ratio",
        random_hits_value=2.0,  # Sample twice as many random hits as relevant docs
        max_random_hits_per_query=200,  # But never more than 200 per query
    )
    ```

    Args:
        queries (Dict[str, str]): A dictionary where keys are query IDs and values are query strings.
        relevant_docs (Union[Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]]):
            A dictionary mapping query IDs to their relevant document IDs.
            Can be a set of doc IDs for binary relevance, a dict of doc_id to relevance score (float between 0 and 1)
            for graded relevance, or a single doc_id string.
        vespa_query_fn (Callable[[str, int, Optional[str]], dict]): A function that takes a query string,
            the number of hits to retrieve (top_k), and an optional query_id, and returns a Vespa query body dictionary.
        app (Vespa): An instance of the Vespa application.
        id_field (str): The field name in the Vespa hit that contains the document ID. This field must be defined as an attribute in your Vespa schema.
        name (str, optional): A name for this data collection run. Defaults to "".
        csv_dir (Optional[str], optional): Directory to save the CSV file. Defaults to None (current directory).
        random_hits_strategy (Union[RandomHitsSamplingStrategy, str], optional): Strategy for sampling random hits.
            Can be "ratio" (or RandomHitsSamplingStrategy.RATIO) to sample as a ratio of relevant docs,
            or "fixed" (or RandomHitsSamplingStrategy.FIXED) to sample a fixed number per query. Defaults to "ratio".
        random_hits_value (Union[float, int], optional): Value for the sampling strategy.
            For RATIO strategy: ratio value (e.g., 1.0 = equal number, 2.0 = twice as many random hits).
            For FIXED strategy: fixed number of random hits per query. Defaults to 1.0.
        max_random_hits_per_query (Optional[int], optional): Maximum limit on random hits per query.
            Only applies to RATIO strategy to prevent excessive sampling. Defaults to None (no limit).
        collect_matchfeatures (bool, optional): Whether to collect match features defined in rank profile's match-features section. Defaults to True.
        collect_rankfeatures (bool, optional): Whether to collect rank features using ranking.listFeatures=true. Defaults to False.
        collect_summaryfeatures (bool, optional): Whether to collect summary features from document summaries. Defaults to False.
        write_csv (bool, optional): Whether to write results to CSV file. Defaults to True.
    """

    def get_recall_param(self, relevant_doc_ids: set, get_relevant: bool) -> dict:
        """
        Adds the recall parameter to the Vespa query body based on relevant document IDs.

        Args:
            relevant_doc_ids (set): A set of relevant document IDs.
            get_relevant (bool): Whether to retrieve relevant documents.

        Returns:
            dict: The updated Vespa query body with the recall parameter.
        """
        recall_string_base = " ".join(
            [f"{self.id_field}:{doc_id}" for doc_id in relevant_doc_ids]
        )
        if get_relevant:
            recall_string = f"+({recall_string_base})"
        else:
            recall_string = f"-({recall_string_base})"
        return {"recall": recall_string}

    def calculate_random_hits_count(self, num_relevant_docs: int) -> int:
        """
        Calculate the number of random hits to sample based on the configured strategy.

        Args:
            num_relevant_docs: Number of relevant documents for the query

        Returns:
            Number of random hits to sample
        """
        if self.random_hits_strategy == RandomHitsSamplingStrategy.RATIO:
            calculated_count = int(num_relevant_docs * self.random_hits_ratio)
            if self.max_random_hits_per_query is not None:
                calculated_count = min(calculated_count, self.max_random_hits_per_query)
            return calculated_count
        else:  # FIXED strategy
            return self.random_hits_fixed

    def collect(self) -> Dict[str, List[Dict]]:
        """
        Collects training data by executing queries and saving results to CSV.

        This method:
        1. Executes all configured queries against the Vespa application.
        2. Collects the top-k document IDs and their relevance labels.
        3. Optionally writes the data to a CSV file for training purposes.
        4. Returns the collected data as a single dictionary with results.

        Returns:
            Dict containing:
            - 'results': List of dictionaries, each containing all data for a query-document pair
              (query_id, doc_id, relevance_label, relevance_score, and all extracted features)
        """
        logger.info(f"Starting VespaFeatureCollector on {self.name}")
        logger.info(f"Number of queries: {len(self.queries_ids)}")

        # Build query bodies using the provided vespa_query_fn
        query_bodies = []
        query_metadata = []  # Track metadata for each query body

        for qid, query_text in zip(self.queries_ids, self.queries):
            relevant_docs = self.relevant_docs.get(qid, set())
            if len(relevant_docs) > 0:
                num_relevant = len(relevant_docs)
                num_random = self.calculate_random_hits_count(num_relevant)
                logger.info(
                    f"Query {qid}: {num_relevant} relevant docs, {num_random} random hits to sample"
                )
            else:
                logger.info(f"No relevant documents for query {qid}, skipping.")
                continue
            if self._vespa_query_fn_takes_query_id:
                query_body: dict = self.vespa_query_fn(query_text, num_relevant, qid)
            else:
                query_body: dict = self.vespa_query_fn(query_text, num_relevant)
            if not isinstance(query_body, dict):
                raise ValueError(
                    f"vespa_query_fn must return a dict, got: {type(query_body)}"
                )
            # Add default body parameters only if not already specified
            for key, value in self.default_body.items():
                if key not in query_body:
                    query_body[key] = value

            # Add feature collection parameters based on configuration
            if self.collect_rankfeatures:
                if "ranking" not in query_body:
                    query_body["ranking"] = {}
                if isinstance(query_body["ranking"], str):
                    # Convert string ranking profile to dict
                    query_body["ranking"] = {"profile": query_body["ranking"]}
                elif not isinstance(query_body["ranking"], dict):
                    query_body["ranking"] = {}
                query_body["ranking"]["listFeatures"] = "true"

            # Note: match-features and summary-features are controlled via the rank profile configuration
            # and YQL select clause respectively, so they don't need query body modifications
            # Add recall parameter for both True and False
            get_relevant_flags = [(True, num_relevant), (False, num_random)]
            for get_relevant, max_k in get_relevant_flags:
                recall_param = self.get_recall_param(relevant_docs, get_relevant)
                # Update hits parameter to match the max_k for this query
                query_body_with_recall = query_body | recall_param | {"hits": max_k}
                # Add the modified query body to the list
                query_bodies.append(query_body_with_recall)
                # Track metadata for this query body
                query_metadata.append(
                    {
                        "qid": qid,
                        "query_text": query_text,
                        "relevant_docs": relevant_docs,
                        "max_k": max_k,
                        "get_relevant": get_relevant,
                    }
                )
                logger.info(f"Querying Vespa with: {query_body_with_recall}")

        responses, new_searchtimes = execute_queries(self.app, query_bodies)
        self.searchtimes.extend(new_searchtimes)

        # Collect all data and determine feature column names
        all_rows = []
        all_feature_names = set()

        for i, (metadata, resp) in enumerate(zip(query_metadata, responses)):
            hits = resp.hits or []
            qid = metadata["qid"]
            query_text = metadata["query_text"]
            relevant_docs = metadata["relevant_docs"]
            max_k = metadata["max_k"]
            get_relevant = metadata["get_relevant"]

            for hit in hits[:max_k]:
                doc_id = get_id_field_from_hit(hit, self.id_field)
                relevance_score = hit.get("relevance", 0.0)

                # Determine relevance label based on whether docs are binary or graded
                if isinstance(relevant_docs, dict):
                    # Graded relevance - use the score directly, or 0.0 if not found
                    relevance_label = relevant_docs.get(doc_id, 0.0)
                else:
                    # Binary relevance - check if doc is in the set
                    relevance_label = 1.0 if doc_id in relevant_docs else 0.0

                # Extract features based on configuration
                features = extract_features_from_hit(
                    hit,
                    self.collect_matchfeatures,
                    self.collect_rankfeatures,
                    self.collect_summaryfeatures,
                )
                all_feature_names.update(features.keys())

                # Create complete row with basic info and features (excluding query_text)
                row_data = {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "relevance_score": relevance_score,
                    "relevance_label": relevance_label,
                }
                # Add all features to the row
                row_data.update(features)
                all_rows.append(row_data)

        # Prepare sorted feature column names
        feature_columns = sorted(list(all_feature_names))

        # Optionally write CSV file
        if self.write_csv:
            # All columns: basic info + features (excluding query_text)
            all_columns = [
                "query_id",
                "doc_id",
                "relevance_label",
                "relevance_score",
            ] + feature_columns

            with open(self.csv_file, "w", newline="") as f:
                if all_rows:
                    writer = csv.DictWriter(f, fieldnames=all_columns)
                    writer.writeheader()
                    for row in all_rows:
                        # Fill missing feature values with empty string
                        complete_row = row.copy()
                        for feature_name in feature_columns:
                            if feature_name not in complete_row:
                                complete_row[feature_name] = ""
                        writer.writerow(complete_row)

            logger.info(
                f"Collected retrieval training data with {len(feature_columns)} features and wrote to {self.csv_file}"
            )
        else:
            logger.info(
                f"Collected retrieval training data with {len(feature_columns)} features"
            )

        # Prepare return data - single dictionary with all results like CSV output
        results = []

        for row in all_rows:
            # Create complete row with all data (same as CSV format)
            complete_row = {
                "query_id": row["query_id"],
                "doc_id": row["doc_id"],
                "relevance_label": row["relevance_label"],
                "relevance_score": row["relevance_score"],
            }

            # Add all features to the row, filling missing values with NaN
            for feature_name in feature_columns:
                complete_row[feature_name] = row.get(feature_name, math.nan)

            results.append(complete_row)

        return {"results": results}


def extract_features_from_hit(
    hit: dict,
    collect_matchfeatures: bool,
    collect_rankfeatures: bool,
    collect_summaryfeatures: bool,
) -> Dict[str, float]:
    """
    Extract features from a Vespa hit based on the collection configuration.

    Args:
        hit: The Vespa hit dictionary
        collect_matchfeatures: Whether to collect match features
        collect_rankfeatures: Whether to collect rank features
        collect_summaryfeatures: Whether to collect summary features

    Returns:
        Dict mapping feature names to values
    """
    features = {}

    if collect_matchfeatures:
        # Try multiple locations for match features
        matchfeatures = hit.get("fields", {}).get("matchfeatures", {})
        for feature_name, feature_value in matchfeatures.items():
            try:
                features[f"match_{feature_name}"] = float(feature_value)
            except (ValueError, TypeError):
                logger.error(
                    f"Skipping non-numeric match feature '{feature_name}': {feature_value}"
                )
                continue

    if collect_rankfeatures:
        # Try multiple locations for rank features
        rankfeatures = hit.get("fields", {}).get("rankfeatures", {})

        for feature_name, feature_value in rankfeatures.items():
            try:
                features[f"rank_{feature_name}"] = float(feature_value)
            except (ValueError, TypeError):
                # Skip non-numeric features
                logger.error(
                    f"Skipping non-numeric rank feature '{feature_name}': {feature_value}"
                )
                continue

    if collect_summaryfeatures:
        # Try multiple locations for summary features
        summaryfeatures = hit.get("fields", {}).get("summaryfeatures", {})

        for feature_name, feature_value in summaryfeatures.items():
            try:
                features[f"summary_{feature_name}"] = float(feature_value)
            except (ValueError, TypeError):
                # Skip non-numeric features
                logger.error(
                    f"Skipping non-numeric summary feature '{feature_name}': {feature_value}"
                )
                continue

    return features


class VespaNNParameters:
    """
    Collection of nearest-neighbor query parameters used in nearest-neighbor classes.
    """

    TIMEOUT = {
        "timeout": "20s",
    }
    HNSW = {
        "ranking.matching.approximateThreshold": 0.00,
        "ranking.matching.filterFirstThreshold": 0.00,
        "ranking.matching.postFilterThreshold": 1.00,
    }
    FILTER_FIRST = {
        "ranking.matching.approximateThreshold": 0.00,
        "ranking.matching.filterFirstThreshold": 1.00,
        "ranking.matching.postFilterThreshold": 1.00,
    }
    EXACT = {
        "ranking.matching.approximateThreshold": 1.00,
        "ranking.matching.filterFirstThreshold": 0.00,
        "ranking.matching.postFilterThreshold": 1.00,
    }
    POST = {
        "ranking.matching.approximateThreshold": 0.00,
        "ranking.matching.filterFirstThreshold": 0.00,
        "ranking.matching.postFilterThreshold": 0.00,
    }


class VespaNNUnsuccessfulQueryError(Exception):
    """
    Exception raised when trying to determine the hit ratio or compute the recall of an unsuccessful query.
    """

    pass


class VespaNNGlobalFilterHitratioEvaluator:
    """
    Determine the hit ratio of the global filter in ANN queries. This hit ratio determines the search strategy
    used to perform the nearest-neighbor search and is essential to understanding and optimizing the behavior
    of Vespa on these queries.

    This class:

    - Takes a list of queries.
    - Runs the queries with tracing.
    - Determines the hit ratio by examining the trace.

    Args:
        queries (Sequence[Mapping[str, str]]): List of ANN queries.
        app (Vespa): An instance of the Vespa application.
    """

    def __init__(
        self,
        queries: Sequence[Mapping[str, Any]],
        app: Vespa,
        verify_target_hits: int | None = None,
    ):
        self.queries = queries
        self.app = app
        self.verify_target_hits = verify_target_hits
        self.searchable_copies = None

    def run(self):
        """
        Determines the hit ratios of the global filters in the supplied ANN queries.

        Returns:
            List[List[float]]: List of lists of hit ratios, which are values from the interval [0.0, 1.0], corresponding to the supplied queries.
        """
        query_parameters = dict(
            dict(VespaNNParameters.TIMEOUT, **VespaNNParameters.HNSW),
            **{
                "trace.explainLevel": "1",
                "trace.level": "1",
                "trace.profileDepth": "100",
            },
        )

        queries_with_parameters = list(
            map(lambda query: dict(query, **query_parameters), self.queries)
        )
        responses, response_times = execute_queries(self.app, queries_with_parameters)

        def extract_from_trace(obj: dict, type_name: str):
            results = []

            if "[type]" in obj and obj["[type]"] == type_name:
                results.append(obj)

            for k, v in obj.items():
                if isinstance(v, dict):
                    results += extract_from_trace(v, type_name)

                elif isinstance(v, list):
                    for i in v:
                        if isinstance(i, dict):
                            results += extract_from_trace(i, type_name)

            return results

        all_hit_ratios = []
        for response in responses:
            if not response.is_successful():
                raise VespaNNUnsuccessfulQueryError()
            trace = response.get_json()["trace"]
            nearest_neighbor_blueprints = extract_from_trace(
                trace, "search::queryeval::NearestNeighborBlueprint"
            )
            hit_ratios = []
            for blueprint in nearest_neighbor_blueprints:
                if (
                    "global_filter" in blueprint
                    and blueprint["global_filter"]["calculated"]
                ):
                    hit_ratios.append(blueprint["global_filter"]["hit_ratio"])
                    actual_upper_limit = blueprint["global_filter"]["upper_limit"]
                    if actual_upper_limit is not None and actual_upper_limit > 0.0:
                        searchable_copies = round(1.0 / actual_upper_limit)
                        if self.searchable_copies is None:
                            self.searchable_copies = searchable_copies
                        else:
                            if self.searchable_copies != searchable_copies:
                                print(
                                    f"Searchable copies mismatch: {searchable_copies} vs. {self.searchable_copies} found earlier"
                                )

                if (
                    self.verify_target_hits is not None
                    and int(blueprint["target_hits"]) != self.verify_target_hits
                ):
                    print(
                        f"Warning: Number of targetHits of query is not {self.verify_target_hits}"
                    )

            all_hit_ratios.append(hit_ratios)

        return all_hit_ratios

    def get_searchable_copies(self) -> int | None:
        """
        Returns number of searchable copies determined during hit-ratio computation.

        Returns:
            int: Number of searchable copies used by Vespa application.
        """
        return self.searchable_copies


class VespaNNRecallEvaluator:
    """
    Determine recall of ANN queries. The recall of an ANN query with k hits is the number of hits
    that actually are among the k nearest neighbors of the query vector.

    This class:

    - Takes a list of queries.
    - First runs the queries as is (with the supplied HTTP parameters).
    - Then runs the queries with the supplied HTTP parameters and an additional parameter enforcing an exact nearest neighbor search.
    - Determines the recall by comparing the results.

    Args:
        queries (Sequence[Mapping[str, Any]]): List of ANN queries.
        hits (int): Number of hits to use. Should match the parameter targetHits in the used ANN queries.
        app (Vespa): An instance of the Vespa application.
        query_limit (int): Maximum number of queries to determine the recall for. Defaults to 20.
        id_field (str): Name of the field containing a unique id. Defaults to "id".
        **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.
    """

    def __init__(
        self,
        queries: Sequence[Mapping[str, Any]],
        hits: int,
        app: Vespa,
        query_limit: int = 20,
        id_field: str = "id",
        **kwargs,
    ):
        self.queries = queries
        self.hits = hits
        self.app = app
        self.query_limit = query_limit
        self.id_field = id_field
        self.parameters = kwargs

    def _compute_recall(
        self, response_exact: VespaQueryResponse, response_approx: VespaQueryResponse
    ) -> float:
        """
        Computes the recall from the given responses, one from an exact search and one from an approximate search.

        Returns:
            float: Recall value from the interval [0.0, 1.0].
        """
        if not (response_exact.is_successful() and response_approx.is_successful()):
            raise VespaNNUnsuccessfulQueryError()

        try:
            results_exact = response_exact.get_json()["root"]["children"]
        except KeyError:
            results_exact = []

        try:
            results_approx = response_approx.get_json()["root"]["children"]
        except KeyError:
            results_approx = []

        def extract_id(hit: dict, id_field: str) -> Tuple[str, str]:
            """Extract document ID from a Vespa hit."""

            # id as specified by field
            fields = hit.get("fields", {})
            if id_field in fields:
                return fields[id_field], "id_field"

            # document id
            id = hit.get("id", "")
            if "::" in id:
                return id, "document_id"

            # internal id
            if id.startswith(
                "index:"
            ):  # id is an internal id of the form index:[source]/[node-index]/[hex-gid], return hex-gid
                return id.split("/", 2)[2], "internal_id"

            raise ValueError(f"Could not extract a document id from hit: {hit}")

        ids_exact = list(map(lambda x: extract_id(x, self.id_field)[0], results_exact))
        ids_approx = list(
            map(lambda x: extract_id(x, self.id_field)[0], results_approx)
        )

        id_types = set()
        id_types.update(map(lambda x: extract_id(x, self.id_field)[1], results_exact))
        id_types.update(map(lambda x: extract_id(x, self.id_field)[1], results_approx))
        if len(id_types) > 1:
            print(
                f"Warning: Multiple id types obtained for hits: {id_types}. The recall computation will not be reliable. Please specify id_field correctly."
            )

        if len(ids_exact) != self.hits:
            print(
                f"Warning: Exact query did not return {self.hits} hits ({len(ids_exact)} hits)."
            )

        recall = sum(map(lambda x: 1 if x in ids_exact else 0, ids_approx))

        if len(ids_exact) > 0:
            return recall / len(ids_exact)
        else:
            return 1.0

    def run(self) -> List[float]:
        """
        Computes the recall of the supplied queries.

        Returns:
            List[float]: List of recall values from the interval [0.0, 1.0] corresponding to the supplied queries.
        """
        query_parameters = dict(
            dict(self.parameters, **VespaNNParameters.TIMEOUT), **{"hits": self.hits}
        )
        query_parameters_exact = dict(query_parameters, **VespaNNParameters.EXACT)

        queries_with_parameters_exact = list(
            map(
                lambda query: dict(query, **query_parameters_exact),
                self.queries[0 : self.query_limit],
            )
        )
        responses_exact, _ = execute_queries(self.app, queries_with_parameters_exact)

        queries_with_parameters = list(
            map(
                lambda query: dict(query, **query_parameters),
                self.queries[0 : self.query_limit],
            )
        )
        responses, _ = execute_queries(self.app, queries_with_parameters)

        return list(
            map(
                lambda pair: self._compute_recall(pair[0], pair[1]),
                zip(responses_exact, responses),
            )
        )


class VespaQueryBenchmarker:
    """
    Determine the searchtime of queries by running them multiple times and taking the average.
    Using the searchtime has the advantage of not including network latency.

    This class:

    - Takes a list of queries.
    - Runs the queries for the given amount of time.
    - Determines the average searchtime of these runs.

    Args:
        queries (Sequence[Mapping[str, Any]]): List of queries.
        app (Vespa): An instance of the Vespa application.
        time_limit(int, optional): Time to run the benchmark for (in milliseconds).
        **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.
    """

    def __init__(
        self,
        queries: Sequence[Mapping[str, Any]],
        app: Vespa,
        time_limit: int = 2000,
        max_concurrent: int = 10,
        **kwargs,
    ):
        self.queries = queries
        self.app = app
        self.time_limit = time_limit
        self.max_concurrent = max_concurrent
        self.parameters = kwargs

        self.queries_with_parameters = list(
            map(
                lambda query: dict(
                    query, **self.parameters, **{"presentation.timing": True}
                ),
                self.queries,
            )
        )
        self.query_chunks = [
            self.queries_with_parameters[x : x + self.max_concurrent]
            for x in range(0, len(self.queries_with_parameters), self.max_concurrent)
        ]

    def _run_benchmark(self, time_limit) -> List[float]:
        """
        Run all queries once and extract the searchtime.

        Returns:
            List[float]: List of searchtimes, corresponding to the supplied queries.
        """
        all_response_times = []
        time_taken = 0

        current_chunk = 0
        while time_taken < time_limit:
            _, response_times = execute_queries(
                self.app,
                self.query_chunks[current_chunk],
                max_concurrent=self.max_concurrent,
            )

            response_times_ms = list(map(lambda x: 1000 * x, response_times))
            all_response_times.extend(response_times_ms)
            time_taken += max(
                sum(response_times_ms), 1
            )  # At least add something in every iteration

            current_chunk = (current_chunk + 1) % len(self.query_chunks)

        return all_response_times

    def run(self) -> List[float]:
        """
        Runs the benchmark (including a warm-up run not included in the result).

        Returns:
            List[float]: List of searchtimes, corresponding to the supplied queries.
        """
        # Warmup run for 100ms
        _ = self._run_benchmark(100)

        # Actual benchmark
        return self._run_benchmark(self.time_limit)


class BucketedMetricResults:
    """
    Stores aggregated statistics for a metric across query buckets.

    Computes mean and various percentiles for values grouped by bucket,
    where each bucket contains multiple measurements (e.g., response times or recall values).

    Args:
        metric_name: Name of the metric being measured (e.g., "searchtime", "recall")
        buckets: List of bucket indices that contain data
        values: List of lists containing measurements, one list per bucket
        filtered_out_ratios: Pre-computed filtered-out ratios for each bucket
    """

    def __init__(
        self,
        metric_name: str,
        buckets: List[int],
        values: List[List[float]],
        filtered_out_ratios: List[float],
    ):
        if len(buckets) != len(values) or len(buckets) != len(filtered_out_ratios):
            raise ValueError(
                "buckets, values, and filtered_out_ratios must have the same length"
            )

        self.metric_name = metric_name
        self.buckets = buckets
        self.filtered_out_ratios = filtered_out_ratios

        # Compute statistics
        self.mean = list(map(mean, values))
        self.median = list(map(lambda x: percentile(x, 50), values))
        self.p95 = list(map(lambda x: percentile(x, 95), values))
        self.p99 = list(map(lambda x: percentile(x, 99), values))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to dictionary format.

        Returns:
            Dictionary containing bucket information and all statistics
        """
        return {
            "metric_name": self.metric_name,
            "buckets": self.buckets,
            "filtered_out_ratios": self.filtered_out_ratios,
            "statistics": {
                "mean": self.mean,
                "median": self.median,
                "p95": self.p95,
                "p99": self.p99,
            },
            "summary": {
                "overall_mean": mean(self.mean) if self.mean else 0.0,
                "overall_median": percentile(self.mean, 50) if self.mean else 0.0,
            },
        }


class VespaNNParameterOptimizer:
    """
    Get suggestions for configuring the nearest-neighbor parameters of a Vespa application.

    This class:

    - Sorts ANN queries into buckets based on the hit-ratio of their global filter.
    - For every bucket, can determine the average response time of the queries in this bucket.
    - For every bucket, can determine the average recall of the queries in this bucket.
    - Can suggest a value for postFilterThreshold.
    - Can suggest a value for filterFirstThreshold.
    - Can suggest a value for filterFirstExploration.
    - Can suggest a value for approximateThreshold.

    Args:
        app (Vespa): An instance of the Vespa application.
        queries (Sequence[Mapping[str, Any]]): Queries to optimize for.
        hits (int): Number of hits to use in recall computations. Has to match the parameter targetHits in the used ANN queries.
        buckets_per_percent (int, optional): How many buckets are created for every percent point, "resolution" of the suggestions. Defaults to 2.
        print_progress (bool, optional): Whether to print progress information while determining suggestions. Defaults to False.
        benchmark_time_limit (int): Time in milliseconds to spend per bucket benchmark. Defaults to 5000.
        recall_query_limit(int): Number of queries per bucket to compute the recall for. Defaults to 20.
        max_concurrent(int): Number of queries to execute concurrently during benchmark/recall calculation. Defaults to 10.
        id_field (str): Name of the field containing a unique id for recall computation. Defaults to "id".
    """

    def __init__(
        self,
        app: Vespa,
        queries: Sequence[Mapping[str, Any]],
        hits: int,
        buckets_per_percent: int = 2,
        print_progress: bool = False,
        benchmark_time_limit: int = 5000,
        recall_query_limit: int = 20,
        max_concurrent: int = 10,
        id_field: str = "id",
    ):
        self.app = app
        self.queries = queries
        self.hits = hits

        # Every bucket represents an interval of length 1/buckets_per_percent. It contains a list of queries with hit-ratios (or rather 1-hit-ratios) in that interval.
        self.buckets_per_percent = buckets_per_percent
        self.buckets = [[] for _ in range(100 * buckets_per_percent)]

        self.print_progress = print_progress
        self.benchmark_time_limit = benchmark_time_limit
        self.recall_query_limit = recall_query_limit
        self.max_concurrent = max_concurrent
        self.id_field = id_field

        self.searchable_copies = None

    def get_bucket_interval_width(self) -> float:
        """
        Gets the width of the interval represented by a single bucket.

        Returns:
            float: Width of the interval represented by a single bucket.
        """
        return 0.01 / self.buckets_per_percent

    def get_number_of_buckets(self) -> int:
        """
        Gets the number of buckets.

        Returns:
            int: Number of buckets.
        """
        return len(self.buckets)

    def get_number_of_nonempty_buckets(self) -> int:
        """
        Counts the number of buckets that contain at least one query.

        Returns:
            int: The number of buckets that contain at least one query.
        """
        return sum(map(lambda x: 1 if x else 0, self.buckets))

    def get_non_empty_buckets(self) -> List[int]:
        """
        Gets the indices of the non-empty buckets.

        Returns:
            List[int]: List of indices of the non-empty buckets.
        """
        non_empty_buckets = []
        for i in range(0, len(self.buckets)):
            if self.buckets[i]:
                non_empty_buckets.append(i)

        return non_empty_buckets

    def get_filtered_out_ratios(self) -> List[float]:
        """
        Gets the (lower interval ends of the) filtered-out ratios of the non-empty buckets.

        Returns:
            List[float]: List of the (lower interval ends of the) filtered-out ratios of the non-empty buckets.
        """
        return list(
            map(lambda x: self.bucket_to_filtered_out(x), self.get_non_empty_buckets())
        )

    def get_number_of_queries(self):
        """
        Gets the number of queries contained in the buckets.

        Returns:
            int: Number of queries contained in the buckets.
        """
        return sum(map(len, self.buckets))

    def bucket_to_hitratio(self, bucket: int) -> float:
        """
        Gets the hit ratio (upper endpoint of interval) corresponding to the given bucket index.

        Args:
            bucket (int): Index of a bucket.

        Returns:
            float: Hit ratio corresponding to the given bucket index.
        """
        return (len(self.buckets) - bucket) / len(self.buckets)

    def bucket_to_filtered_out(self, bucket: int) -> float:
        """
        Gets the filtered-out ratio (1 - hit ratio, lower endpoint of interval) corresponding to the given bucket index.

        Args:
            bucket (int): Index of a bucket.

        Returns:
            float: Filtered-out ratio corresponding to the given bucket index.
        """
        return bucket / len(self.buckets)

    def buckets_to_filtered_out(self, buckets: List[int]) -> List[float]:
        """
        Applies bucket_to_filtered_out to list of bucket indices.

        Args:
            buckets (List[int]): List of bucket indices.

        Returns:
            List[float]: Filtered-out ratios corresponding to the given bucket indices.
        """
        return list(map(lambda b: self.bucket_to_filtered_out(b), buckets))

    def filtered_out_to_bucket(self, percent: float) -> int:
        """
        Gets the index of the bucket containing the given filtered-out ratio.

        Args:
            percent (float): Filtered-out ratio.

        Returns:
            int: Index of bucket containing the given filtered-out ratio.
        """
        bucket = math.floor(percent * 100 * self.buckets_per_percent)
        return max(0, min(bucket, self.get_number_of_buckets() - 1))

    def distribute_to_buckets(
        self, queries_with_hitratios: Sequence[Tuple[Mapping[str, Any], float]]
    ) -> List[List[str]]:
        """
        Distributes the given queries to buckets according to their given hit ratios.

        Args:
            queries_with_hitratios (List[(Dict[str,str],float)]): Queries with hit ratios.

        Returns:
            List[List[str]]: List of buckets.
        """
        for query, hitratio in queries_with_hitratios:
            if hitratio is not None:
                filtered_out = 1.0 - hitratio
                bucket_number = self.filtered_out_to_bucket(filtered_out)
                self.buckets[bucket_number].append(query)
            else:
                print(f"Warning: Query {query} has no hitratio.")

        return self.buckets

    def determine_hit_ratios_and_distribute_to_buckets(
        self, queries: Sequence[Mapping[str, Any]]
    ) -> List[List[str]]:
        """
        Distributes the given queries to buckets by determining their hit ratios.

        Args:
            queries (Sequence[Mapping[str, Any]]): Queries.

        Returns:
            List[List[str]]: List of buckets.
        """
        hitratio_evaluator = VespaNNGlobalFilterHitratioEvaluator(
            queries, self.app, verify_target_hits=self.hits
        )
        hitratio_list = hitratio_evaluator.run()
        self.searchable_copies = hitratio_evaluator.get_searchable_copies()

        for i in range(0, len(hitratio_list)):
            hitratios = hitratio_list[i]
            if len(hitratios) == 0:
                print(
                    f"Warning: No hit ratios found for query #{i}, skipping query (No nearestNeighbor operator?)"
                )

        # Only keep queries for which we found exactly one hit ratio
        queries_with_hitratios = [
            (q, mean(h)) for q, h in zip(queries, hitratio_list) if len(h) > 0
        ]

        return self.distribute_to_buckets(queries_with_hitratios)

    @staticmethod
    def query_from_get_string(get_query: str) -> Dict[str, str]:
        """
        Parses a query in GET format.

        Args:
            get_query (str): Query as a single-line GET string.

        Returns:
            Dict[str,str]: Query as a dict.
        """
        url = urllib.parse.urlparse(get_query)
        assert url.path == "/search/"
        parsed_query = urllib.parse.parse_qs(url.query)
        query = {}
        for key in parsed_query.keys():
            query[key] = parsed_query[key][0]

        assert "yql" in query
        return query

    def distribute_file_to_buckets(self, filename: str) -> List[List[str]]:
        """
        Distributes the queries from the given file to buckets according to their given hit ratios.

        Args:
            filename str: Name of file with GET queries (one per line).

        Returns:
            List[List[str]]: List of buckets.
        """
        if self.print_progress:
            print("Determining hit ratios of queries")

        # Read query file with get queries
        with open(filename) as file:
            get_queries = file.read().splitlines()

        # Parse get queries
        queries = list(map(self.query_from_get_string, get_queries))

        return self.determine_hit_ratios_and_distribute_to_buckets(queries)

    def _has_query_with_filtered_out(self, lower: float, upper: float) -> bool:
        """
        Checks whether there are queries falling into the buckets specified by the given interval.

        Args:
            lower (float): Lower end of interval (inclusive).
            upper (float): Upper end of interval (exclusive).

        Return:
            bool: Whether there are queries falling into the buckets.
        """
        lower_bucket = self.filtered_out_to_bucket(lower)
        upper_bucket = self.filtered_out_to_bucket(upper)

        for bucket_num in range(lower_bucket, upper_bucket):
            if self.buckets[bucket_num]:
                return True

        return False

    def has_sufficient_queries(self) -> bool:
        """
        Checks whether the given queries are deemed sufficient to give meaningful suggestions.

        Returns:
            bool: Whether the given queries are deemed sufficient to give meaningful suggestions.
        """
        check_intervals = [
            (0.00, 0.25),
            (0.25, 0.50),
            (0.50, 0.75),
            (0.75, 0.90),
            (0.90, 0.95),
            (0.95, 1.00),
        ]

        for lower, upper in check_intervals:
            if not self._has_query_with_filtered_out(lower, upper):
                if self.print_progress:
                    print(
                        f"  No queries found with filtered-out ratio in [{lower},{upper})"
                    )

                return False

        return True

    def buckets_sufficiently_filled(self) -> bool:
        """
        Checks whether all non-empty buckets have at least 10 queries.

        Returns:
            bool: Whether all non-empty buckets have at least 10 queries.
        """

        for bucket_num in range(0, len(self.buckets)):
            bucket = self.buckets[bucket_num]

            if bucket and len(bucket) < 10:
                if self.print_progress:
                    print(
                        f"  Bucket for filtered-out ratios in [{self.bucket_to_filtered_out(bucket_num)},{self.bucket_to_filtered_out(bucket_num + 1)}) only has {len(bucket)} queries."
                    )
                return False

        return True

    def get_query_distribution(self):
        """
        Gets the distribution of queries across all buckets.

        Returns:
            List[float]: List of filtered-out ratios corresponding to non-empty buckets.
            List[int]: List of numbers of queries.
        """
        x = []
        y = []
        for i in range(0, len(self.buckets)):
            bucket = self.buckets[i]
            if bucket:  # not empty
                x.append(self.bucket_to_filtered_out(i))
                y.append(len(bucket))

        return x, y

    def benchmark(self, **kwargs) -> BucketedMetricResults:
        """
        For each non-empty bucket, determine the average searchtime.

        Args:
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.

        Returns:
            BucketedMetricResults: The benchmark results.
        """
        if self.print_progress:
            print("->Benchmarking", end="")
        results = []
        non_empty_buckets = []
        processed_buckets = 0
        for i in range(0, self.get_number_of_buckets()):
            bucket = self.buckets[i]
            if bucket:
                if self.print_progress:
                    print(
                        f"\r->Benchmarking: {round(processed_buckets * 100 / self.get_number_of_nonempty_buckets(), 2)}%",
                        end="",
                    )
                processed_buckets += 1
                benchmarker = VespaQueryBenchmarker(
                    bucket,
                    self.app,
                    time_limit=self.benchmark_time_limit,
                    max_concurrent=self.max_concurrent,
                    **kwargs,
                )
                response_times = benchmarker.run()
                results.append(response_times)
                non_empty_buckets.append(i)

        print("\r  Benchmarking: 100.0%")

        return BucketedMetricResults(
            metric_name="searchtime",
            buckets=non_empty_buckets,
            values=results,
            filtered_out_ratios=[
                self.bucket_to_filtered_out(b) for b in non_empty_buckets
            ],
        )

    def compute_average_recalls(self, **kwargs) -> BucketedMetricResults:
        """
        For each non-empty bucket, determine the average recall.

        Args:
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.

        Returns:
            BucketedMetricResults: The recall results.
        """
        if self.print_progress:
            print("->Computing recall", end="")
        results = []
        non_empty_buckets = []
        processed_buckets = 0
        for i in range(0, self.get_number_of_buckets()):
            bucket = self.buckets[i]
            if bucket:
                if self.print_progress:
                    print(
                        f"\r->Computing recall: {round(processed_buckets * 100 / self.get_number_of_nonempty_buckets(), 2)}%",
                        end="",
                    )
                recall_evaluator = VespaNNRecallEvaluator(
                    bucket,
                    self.hits,
                    self.app,
                    self.recall_query_limit,
                    self.id_field,
                    **kwargs,
                )
                recall_list = recall_evaluator.run()
                results.append(recall_list)
                non_empty_buckets.append(i)
                processed_buckets += 1

        print("\r  Computing recall: 100.0%")

        return BucketedMetricResults(
            metric_name="recall",
            buckets=non_empty_buckets,
            values=results,
            filtered_out_ratios=[
                self.bucket_to_filtered_out(b) for b in non_empty_buckets
            ],
        )

    def suggest_filter_first_threshold(
        self, **kwargs
    ) -> dict[str, float | dict[str, List[float]]]:
        """
        Suggests a value for [filterFirstThreshold](https://docs.vespa.ai/en/reference/query-api-reference.html#ranking.matching) based on performed benchmarks.

        Args:
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>. Should contain ranking.matching.filterFirstExploration!

        Returns:
            float: Suggested value for filterFirstThreshold.
        """
        hnsw_parameters = dict(
            dict(VespaNNParameters.TIMEOUT, **kwargs), **VespaNNParameters.HNSW
        )
        benchmark_hnsw = self.benchmark(**hnsw_parameters)
        recall_hnsw = self.compute_average_recalls(**hnsw_parameters)

        filter_first_parameters = dict(
            dict(VespaNNParameters.TIMEOUT, **kwargs), **VespaNNParameters.FILTER_FIRST
        )
        benchmark_filter_first = self.benchmark(**filter_first_parameters)
        recall_filter_first = self.compute_average_recalls(**filter_first_parameters)

        suggestion = self._suggest_filter_first_threshold(
            benchmark_hnsw, recall_hnsw, benchmark_filter_first, recall_filter_first
        )

        report = {
            "suggestion": suggestion,
            "benchmarks": {
                "hnsw": benchmark_hnsw.mean,
                "filter_first": benchmark_filter_first.mean,
            },
            "recall_measurements": {
                "hnsw": recall_hnsw.mean,
                "filter_first": recall_filter_first.mean,
            },
        }

        return report

    @staticmethod
    def _interpolate(x, y, intended_length):
        interpolated_y = [0] * intended_length

        if len(x) == 0:
            return interpolated_y

        for i in range(0, x[0] + 1):
            interpolated_y[i] = y[0]

        for i in range(x[-1], intended_length):
            interpolated_y[i] = y[-1]

        for i in range(0, len(x) - 1):
            length = x[i + 1] - x[i]

            for j in range(0, length):
                interpolated_y[x[i] + j] = (
                    j / length * y[i + 1] + (1 - j / length) * y[i]
                )

        return interpolated_y

    def _suggest_filter_first_threshold(
        self,
        benchmark_hnsw: BucketedMetricResults,
        recall_hnsw: BucketedMetricResults,
        benchmark_filter_first: BucketedMetricResults,
        recall_filter_first: BucketedMetricResults,
    ) -> float:
        """
        Suggests a value for [filterFirstThreshold](https://docs.vespa.ai/en/reference/query-api-reference.html#ranking.matching) based on the two given benchmarks (using HNSW only, using HNSW with filter first only).

        Args:
            benchmark_hnsw (BucketedMetricResults): Benchmark using HNSW only obtained from benchmark().
            recall_hnsw (BucketedMetricResults): Recall measurement using HNSW only obtained from compute_average_recalls().
            benchmark_filter_first (BucketedMetricResults): Benchmark using HNSW with filter first only obtained from benchmark().
            recall_filter_first (BucketedMetricResults): Recall measurement using HNSW with filter first only obtained from compute_average_recalls().

        Returns:
            float: Suggested value for filterFirstThreshold.
        """
        # Interpolate benchmark values for empty buckets
        interpolated_hnsw_y = self._interpolate(
            self.get_non_empty_buckets(),
            benchmark_hnsw.mean,
            self.get_number_of_buckets(),
        )
        interpolated_filter_first_y = self._interpolate(
            self.get_non_empty_buckets(),
            benchmark_filter_first.mean,
            self.get_number_of_buckets(),
        )

        # Start at last bucket
        threshold = self.get_number_of_buckets()
        threshold_diff = 0
        for i in range(0, self.get_number_of_buckets()):
            # Check what happens when we use "i" as the threshold
            # current_diff is the "overhead" of using HNSW instead of HNSW (filter first)
            current_diff = 0
            for j in range(i, self.get_number_of_buckets()):
                current_diff += interpolated_hnsw_y[j] - interpolated_filter_first_y[j]

            # Larger overhead of using HNSW?
            # We find the sport where the overhead of using HNSW is the largest
            if current_diff > threshold_diff:
                threshold = i
                threshold_diff = current_diff

        return self.bucket_to_hitratio(threshold)

    def suggest_approximate_threshold(
        self, **kwargs
    ) -> dict[str, float | dict[str, List[float]]]:
        """
        Suggests a value for [approximateThreshold](https://docs.vespa.ai/en/reference/query-api-reference.html#ranking.matching) based on performed benchmarks.

        Args:
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>. Should contain ranking.matching.filterFirstExploration and ranking.matching.filterFirstThreshold!

        Returns:
            float: Suggested value for approximateThreshold.
        """
        exact_parameters = dict(
            dict(VespaNNParameters.TIMEOUT, **kwargs), **VespaNNParameters.EXACT
        )
        benchmark_exact = self.benchmark(**exact_parameters)

        filter_first_copy = VespaNNParameters.FILTER_FIRST.copy()
        del filter_first_copy["ranking.matching.filterFirstThreshold"]
        filter_first_parameters = dict(
            dict(VespaNNParameters.TIMEOUT, **kwargs), **filter_first_copy
        )
        benchmark_filter_first = self.benchmark(**filter_first_parameters)
        recall_filter_first = self.compute_average_recalls(**filter_first_parameters)

        suggestion = self._suggest_approximate_threshold(
            benchmark_exact, benchmark_filter_first, recall_filter_first
        )

        report = {
            "suggestion": suggestion,
            "benchmarks": {
                "exact": benchmark_exact.mean,
                "filter_first": benchmark_filter_first.mean,
            },
            "recall_measurements": {
                "exact": [1.0] * self.get_number_of_nonempty_buckets(),
                "filter_first": recall_filter_first.mean,
            },
        }

        return report

    def _suggest_approximate_threshold(
        self,
        benchmark_exact: BucketedMetricResults,
        benchmark_ann: BucketedMetricResults,
        recall_ann: BucketedMetricResults,
    ) -> float:
        """
        Suggests a value for [approximateThreshold](https://docs.vespa.ai/en/reference/query-api-reference.html#ranking.matching) based on the two given benchmarks (using exact search only, using HNSW with tuned filter-first parameters).

        Args:
            benchmark_exact (BucketedMetricResults): Benchmark using exact search only obtained from benchmark().
            benchmark_ann (BucketedMetricResults): Benchmark using HNSW with tuned filter-first parameters obtained from benchmark().
            recall_ann (BucketedMetricResults): Recall measurements using HNSW with tuned filter-first parameters obtained from compute_average_recalls().

        Returns:
            float: Suggested value for approximateThreshold.
        """
        # Interpolate benchmark values for empty buckets
        int_bench_exact = self._interpolate(
            self.get_non_empty_buckets(),
            benchmark_exact.mean,
            self.get_number_of_buckets(),
        )
        int_bench_ann = self._interpolate(
            self.get_non_empty_buckets(),
            benchmark_ann.mean,
            self.get_number_of_buckets(),
        )

        # Start at last bucket
        approximate_threshold = self.get_number_of_buckets() - 1

        # Fast-foward to last non-empty bucket
        while approximate_threshold > 0 and not self.buckets[approximate_threshold]:
            approximate_threshold -= 1

        while approximate_threshold > 0:
            # Get response time for ANN
            ann_time = int_bench_ann[approximate_threshold]

            # Is response time for an exact search lower than for ANN?
            # If yes, then use an exact search!
            if int_bench_exact[approximate_threshold] < ann_time:
                approximate_threshold -= 1
            else:
                break

        return self.bucket_to_hitratio(approximate_threshold)

    def suggest_post_filter_threshold(
        self, **kwargs
    ) -> dict[str, float | dict[str, List[float]]]:
        """
        Suggests a value for [postFilterThreshold](https://docs.vespa.ai/en/reference/query-api-reference.html#ranking.matching) based on performed benchmarks and recall measurements.

        Args:
            **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>. Should contain ranking.matching.filterFirstExploration, ranking.matching.filterFirstThreshold, and ranking.matching.approximateThreshold!

        Returns:
            float: Suggested value for postFilterThreshold.
        """
        post_filtering_parameters = dict(
            dict(VespaNNParameters.TIMEOUT, **kwargs), **VespaNNParameters.POST
        )
        benchmark_post_filtering = self.benchmark(**post_filtering_parameters)

        filter_first_copy = VespaNNParameters.FILTER_FIRST.copy()
        del filter_first_copy["ranking.matching.filterFirstThreshold"]
        del filter_first_copy["ranking.matching.approximateThreshold"]
        filter_first_parameters = dict(
            dict(VespaNNParameters.TIMEOUT, **kwargs), **filter_first_copy
        )
        benchmark_filter_first = self.benchmark(**filter_first_parameters)

        recall_post_filtering = self.compute_average_recalls(
            **post_filtering_parameters
        )
        recall_filter_first = self.compute_average_recalls(**filter_first_parameters)

        suggestion = self._suggest_post_filter_threshold(
            benchmark_post_filtering,
            recall_post_filtering,
            benchmark_filter_first,
            recall_filter_first,
        )

        report = {
            "suggestion": suggestion,
            "benchmarks": {
                "post_filtering": benchmark_post_filtering.mean,
                "filter_first": benchmark_filter_first.mean,
            },
            "recall_measurements": {
                "post_filtering": recall_post_filtering.mean,
                "filter_first": recall_filter_first.mean,
            },
        }

        return report

    def _suggest_post_filter_threshold(
        self,
        benchmark_post_filtering: BucketedMetricResults,
        recall_post_filtering: BucketedMetricResults,
        benchmark_pre_filtering: BucketedMetricResults,
        recall_pre_filtering: BucketedMetricResults,
    ) -> float:
        """
        Suggests a value for [postFilterThreshold](https://docs.vespa.ai/en/reference/query-api-reference.html#ranking.matching) based on the two given pairs of a benchmark and a recall measurement (using post-filtering only, using HNSW with tuned parameters only).

        Args:
            benchmark_post_filtering (BucketedMetricResults): Benchmark using post-filtering only obtained from benchmark().
            recall_post_filtering (BucketedMetricResults): Recall measurement using post-filtering only obtained from compute_average_recalls().
            benchmark_pre_filtering (BucketedMetricResults): Benchmark using HNSW with tuned parameters only obtained from benchmark().
            recall_pre_filtering (BucketedMetricResults): Recall measurement using HNSW with tuned parameters only obtained from compute_average_recalls().

        Returns:
            float: Suggested value for postFilterThreshold.
        """
        # Interpolate benchmark values for empty buckets
        int_bench_post = self._interpolate(
            self.get_non_empty_buckets(),
            benchmark_post_filtering.mean,
            self.get_number_of_buckets(),
        )
        int_bench_pre = self._interpolate(
            self.get_non_empty_buckets(),
            benchmark_pre_filtering.mean,
            self.get_number_of_buckets(),
        )

        int_recall_post = self._interpolate(
            self.get_non_empty_buckets(),
            recall_post_filtering.mean,
            self.get_number_of_buckets(),
        )
        int_recall_pre = self._interpolate(
            self.get_non_empty_buckets(),
            recall_pre_filtering.mean,
            self.get_number_of_buckets(),
        )

        threshold = 0
        response_time_gain = 0
        for i in range(0, self.get_number_of_buckets()):
            # Gain we get from using post filtering until i
            current_gain = 0
            for j in range(0, i):
                current_gain += int_bench_pre[j] - int_bench_post[j]

            no_recall_loss = True
            for j in range(0, i):
                recall_loss = int_recall_pre[j] - int_recall_post[j]
                if recall_loss >= 0.01:
                    no_recall_loss = False
                    break

            if no_recall_loss and current_gain > response_time_gain:
                threshold = i
                response_time_gain = current_gain

        suggestion = self.bucket_to_hitratio(threshold)
        if self.searchable_copies is not None:
            suggestion = suggestion * self.searchable_copies
            suggestion = min(suggestion, 1.0)

        return suggestion

    def _test_filter_first_exploration(
        self, filter_first_exploration: float
    ) -> Tuple[BucketedMetricResults, BucketedMetricResults]:
        parameters_candidate = dict(
            dict(VespaNNParameters.TIMEOUT, **VespaNNParameters.FILTER_FIRST),
            **{"ranking.matching.filterFirstExploration": filter_first_exploration},
        )
        benchmark = self.benchmark(**parameters_candidate)
        recall = self.compute_average_recalls(**parameters_candidate)

        return benchmark, recall

    def suggest_filter_first_exploration(
        self,
    ) -> dict[str, float | dict[str, List[float]]]:
        """
        Suggests a value for [filterFirstExploration](https://docs.vespa.ai/en/reference/query-api-reference.html#ranking.matching) based on benchmarks and recall measurements performed on the supplied Vespa app.

        Returns:
            dict: A dictionary containing the suggested value, benchmarks, and recall measurements.
        """
        benchmark_no_exploration, recall_no_exploration = (
            self._test_filter_first_exploration(0.0)
        )
        benchmark_no_exploration_int = self._interpolate(
            self.get_non_empty_buckets(),
            benchmark_no_exploration.mean,
            self.get_number_of_buckets(),
        )

        benchmark_full_exploration, recall_full_exploration = (
            self._test_filter_first_exploration(1.0)
        )
        recall_full_exploration_int = self._interpolate(
            self.get_non_empty_buckets(),
            recall_full_exploration.mean,
            self.get_number_of_buckets(),
        )
        assert mean(benchmark_no_exploration_int) > 0
        assert mean(recall_full_exploration_int) > 0

        benchmarks = {
            0.0: benchmark_no_exploration.mean,
            1.0: benchmark_full_exploration.mean,
        }

        recall_measurements = {
            0.0: recall_no_exploration.mean,
            1.0: recall_full_exploration.mean,
        }

        # Find tradeoff between increase in response time and drop in recall by using binary search
        left = 0.0
        right = 1.0
        filter_first_exploration = (right - left) / 2
        for i in range(0, 7):
            if self.print_progress:
                print(f"  Testing {filter_first_exploration}")
            benchmark_candidate, recall_candidate = self._test_filter_first_exploration(
                filter_first_exploration
            )
            benchmark_candidate_int = self._interpolate(
                self.get_non_empty_buckets(),
                benchmark_candidate.mean,
                self.get_number_of_buckets(),
            )
            recall_candidate_int = self._interpolate(
                self.get_non_empty_buckets(),
                recall_candidate.mean,
                self.get_number_of_buckets(),
            )
            benchmarks[filter_first_exploration] = benchmark_candidate.mean
            recall_measurements[filter_first_exploration] = recall_candidate.mean

            # How much does the response time increase compared to no exploration?
            # One could also try to compare the values for every bucket, but this might be a bit unstable:
            # response_time_deviation = max([x/y - 1 for x, y in zip(benchmark_candidate, benchmark_no_exploration)])
            response_time_deviation = max(
                [
                    x / mean(benchmark_no_exploration_int) - 1
                    for x in benchmark_candidate_int
                ]
            )

            # How much does the recall drop compared to full exploration?
            # One could also try to compare the values for every bucket, but this might be a bit unstable:
            # recall_deviation = max([1 - y/x for x, y in zip(recall_full_exploration, recall_candidate)])
            recall_deviation = max(
                [
                    1 - y / mean(recall_full_exploration_int)
                    for y in recall_candidate_int
                ]
            )

            # Check how increase in response time compares to drop in recall
            # (One could try to use weights here, e.g., make recall matter more)
            if (
                response_time_deviation > 1.5 * recall_deviation
            ):  # Increase in response time is larger than drop in recall, decrease exploration
                right = filter_first_exploration
            else:  # Increase in response time is smaller than drop in recall, increase exploration
                left = filter_first_exploration

            filter_first_exploration = left + (right - left) / 2

        report = {
            "suggestion": filter_first_exploration,
            "benchmarks": benchmarks,
            "recall_measurements": recall_measurements,
        }

        return report

    def run(self) -> Dict[str, Any]:
        """
        Determines suggestions for all parameters supported by this class.

        This method:
        1. Determines the hit-ratios of supplied ANN queries.
        2. Sorts these queries into buckets based on the determined hit-ratio.
        3. Determines a suggestion for filterFirstExploration.
        4. Determines a suggestion for filterFirstThreshold.
        5. Determines a suggestion for approximateThreshold.
        6. Determines a suggestion for postFilterThreshold.
        7. Reports the determined suggestions and all benchmarks and recall measurements performed.

        Returns:
            dict: A dictionary containing the suggested values, information about the query distribution, performed benchmarks, and recall measurements.

        Example:
            ```python
            {
                "buckets": {
                    "buckets_per_percent": 2,
                    "bucket_interval_width": 0.005,
                    "non_empty_buckets": [
                        2,
                        20,
                        100,
                        180,
                        190,
                        198
                    ],
                    "filtered_out_ratios": [
                        0.01,
                        0.1,
                        0.5,
                        0.9,
                        0.95,
                        0.99
                    ],
                    "hit_ratios": [
                        0.99,
                        0.9,
                        0.5,
                        0.09999999999999998,
                        0.050000000000000044,
                        0.010000000000000009
                    ],
                    "query_distribution": [
                        100,
                        100,
                        100,
                        100,
                        100,
                        100
                    ]
                },
                "filterFirstExploration": {
                    "suggestion": 0.39453125,
                    "benchmarks": {
                        "0.0": [
                            4.265999999999999,
                            4.256000000000001,
                            3.9430000000000005,
                            3.246999999999998,
                            2.4610000000000003,
                            1.768
                        ],
                        "1.0": [
                            3.9259999999999984,
                            3.6010000000000004,
                            3.290999999999999,
                            3.78,
                            4.927000000000002,
                            8.415000000000001
                        ],
                        "0.5": [
                            3.6299999999999977,
                            3.417,
                            3.4490000000000007,
                            3.752,
                            4.257,
                            5.99
                        ],
                        "0.25": [
                            3.5830000000000006,
                            3.616,
                            3.3239999999999985,
                            3.3200000000000016,
                            2.654999999999999,
                            2.3789999999999996
                        ],
                        "0.375": [
                            3.465,
                            3.4289999999999994,
                            3.196999999999997,
                            3.228999999999999,
                            3.167,
                            3.700999999999999
                        ],
                        "0.4375": [
                            3.9880000000000013,
                            3.463000000000002,
                            3.4650000000000007,
                            3.5000000000000013,
                            3.7499999999999982,
                            4.724000000000001
                        ],
                        "0.40625": [
                            3.4990000000000006,
                            3.3680000000000003,
                            3.147000000000001,
                            3.33,
                            3.381,
                            4.083999999999998
                        ],
                        "0.390625": [
                            3.6060000000000008,
                            3.5269999999999992,
                            3.2820000000000005,
                            3.433999999999998,
                            3.2880000000000007,
                            3.8609999999999984
                        ],
                        "0.3984375": [
                            3.6870000000000016,
                            3.386000000000001,
                            3.336000000000001,
                            3.316999999999999,
                            3.5329999999999973,
                            4.719000000000002
                        ]
                    },
                    "recall_measurements": {
                        "0.0": [
                            0.8758,
                            0.8768999999999997,
                            0.8915,
                            0.9489999999999994,
                            0.9045999999999998,
                            0.64
                        ],
                        "1.0": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.9675999999999998,
                            0.9852999999999996,
                            0.9957999999999998
                        ],
                        "0.5": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.9660999999999998,
                            0.9759999999999996,
                            0.9903
                        ],
                        "0.25": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.9553999999999995,
                            0.9323999999999996,
                            0.8123000000000004
                        ],
                        "0.375": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.9615999999999997,
                            0.9599999999999999,
                            0.9626000000000002
                        ],
                        "0.4375": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.9642999999999999,
                            0.9697999999999999,
                            0.9832
                        ],
                        "0.40625": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.9632,
                            0.9642999999999999,
                            0.9763999999999997
                        ],
                        "0.390625": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.9625999999999999,
                            0.9617999999999999,
                            0.9688999999999998
                        ],
                        "0.3984375": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.963,
                            0.9635000000000001,
                            0.9738999999999999
                        ]
                    }
                },
                "filterFirstThreshold": {
                    "suggestion": 0.47,
                    "benchmarks": {
                        "hnsw": [
                            2.779,
                            2.725000000000001,
                            3.151999999999999,
                            7.138999999999998,
                            11.362,
                            32.599999999999994
                        ],
                        "filter_first": [
                            3.543999999999999,
                            3.454,
                            3.443999999999999,
                            3.4129999999999994,
                            3.4090000000000003,
                            4.602999999999998
                        ]
                    },
                    "recall_measurements": {
                        "hnsw": [
                            0.8284999999999996,
                            0.8368999999999996,
                            0.9007999999999996,
                            0.9740999999999996,
                            0.9852999999999993,
                            0.9937999999999992
                        ],
                        "filter_first": [
                            0.8757,
                            0.8768999999999997,
                            0.8909999999999999,
                            0.9627999999999999,
                            0.9630000000000001,
                            0.9718999999999994
                        ]
                    }
                },
                "approximateThreshold": {
                    "suggestion": 0.03,
                    "benchmarks": {
                        "exact": [
                            33.072,
                            31.99600000000001,
                            23.256,
                            9.155,
                            6.069000000000001,
                            2.0949999999999984
                        ],
                        "filter_first": [
                            2.9570000000000003,
                            2.91,
                            3.165000000000001,
                            3.396999999999998,
                            3.3310000000000004,
                            4.046
                        ]
                    },
                    "recall_measurements": {
                        "exact": [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0
                        ],
                        "filter_first": [
                            0.8284999999999996,
                            0.8368999999999996,
                            0.9007999999999996,
                            0.9627999999999999,
                            0.9630000000000001,
                            0.9718999999999994
                        ]
                    }
                },
                "postFilterThreshold": {
                    "suggestion": 0.49,
                    "benchmarks": {
                        "post_filtering": [
                            2.0609999999999995,
                            2.448,
                            3.097999999999999,
                            7.200999999999999,
                            11.463000000000006,
                            11.622999999999996
                        ],
                        "filter_first": [
                            3.177999999999999,
                            2.717000000000001,
                            3.177,
                            3.5000000000000004,
                            3.455,
                            2.1159999999999997
                        ]
                    },
                    "recall_measurements": {
                        "post_filtering": [
                            0.8288999999999995,
                            0.8355,
                            0.8967999999999998,
                            0.9519999999999997,
                            0.9512999999999994,
                            0.19180000000000003
                        ],
                        "filter_first": [
                            0.8284999999999996,
                            0.8368999999999996,
                            0.9007999999999996,
                            0.9627999999999999,
                            0.9630000000000001,
                            1.0
                        ]
                    }
                }
            }
            ```
        """
        print("Distributing queries to buckets")
        # Distribute queries to buckets
        self.determine_hit_ratios_and_distribute_to_buckets(self.queries)

        # Check if the queries we have are deemed sufficient
        if not self.has_sufficient_queries():
            print(
                "  Warning: Selection of queries might not cover enough hit ratios to get meaningful results."
            )

        if not self.buckets_sufficiently_filled():
            print("  Warning: Only few queries for a specific hit ratio.")

        bucket_report = {
            "buckets_per_percent": self.buckets_per_percent,
            "bucket_interval_width": self.get_bucket_interval_width(),
            "non_empty_buckets": self.get_non_empty_buckets(),
            "filtered_out_ratios": self.get_filtered_out_ratios(),
            "hit_ratios": list(map(lambda x: 1 - x, self.get_filtered_out_ratios())),
            "query_distribution": self.get_query_distribution()[1],
        }
        if self.print_progress:
            print(bucket_report)

        # Determine filter-first parameters first
        # filterFirstExploration
        if self.print_progress:
            print("Determining suggestion for filterFirstExploration")
        filter_first_exploration_report = self.suggest_filter_first_exploration()
        filter_first_exploration = filter_first_exploration_report["suggestion"]
        if self.print_progress:
            print(filter_first_exploration_report)

        # filterFirstThreshold
        if self.print_progress:
            print("Determining suggestion for filterFirstThreshold")
        filter_first_threshold_report = self.suggest_filter_first_threshold(
            **{"ranking.matching.filterFirstExploration": filter_first_exploration}
        )
        filter_first_threshold = filter_first_threshold_report["suggestion"]
        if self.print_progress:
            print(filter_first_threshold_report)

        # approximateThreshold
        if self.print_progress:
            print("Determining suggestion for approximateThreshold")
        approximate_threshold_report = self.suggest_approximate_threshold(
            **{
                "ranking.matching.filterFirstThreshold": filter_first_threshold,
                "ranking.matching.filterFirstExploration": filter_first_exploration,
            }
        )
        approximate_threshold = approximate_threshold_report["suggestion"]
        if self.print_progress:
            print(approximate_threshold_report)

        # postFilterThreshold
        if self.print_progress:
            print("Determining suggestion for postFilterThreshold")
        post_filter_threshold_report = self.suggest_post_filter_threshold(
            **{
                "ranking.matching.approximateThreshold": approximate_threshold,
                "ranking.matching.filterFirstThreshold": filter_first_threshold,
                "ranking.matching.filterFirstExploration": filter_first_exploration,
            }
        )
        if self.print_progress:
            print(post_filter_threshold_report)

        report = {
            "buckets": bucket_report,
            "filterFirstExploration": filter_first_exploration_report,
            "filterFirstThreshold": filter_first_threshold_report,
            "approximateThreshold": approximate_threshold_report,
            "postFilterThreshold": post_filter_threshold_report,
        }

        return report
