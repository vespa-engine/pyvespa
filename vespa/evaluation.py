from __future__ import annotations
import os
import csv
import logging
from typing import Dict, Set, Callable, List, Optional, Union, Tuple
import math
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import re
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
    app: Vespa, query_bodies: List[dict]
) -> Tuple[List[VespaQueryResponse], List[float]]:
    """
    Execute queries and collect timing information.
    Returns the responses and a list of search times.
    """
    responses: List[VespaQueryResponse] = app.query_many(query_bodies)
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
        queries (List[Dict[str, str]]): List of ANN queries.
        app (Vespa): An instance of the Vespa application.
    """

    def __init__(self, queries: List[Dict[str, str]], app: Vespa):
        self.queries = queries
        self.app = app

    def run(self):
        """
        Determines the hit ratios of the global filters in the supplied ANN queries.

        Returns:
            List[List[float]]: List of lists of hit ratios, which are values from the intervall [0.0, 1.0], corresponding to the supplied queries.
        """
        query_parameters = {
            "timeout": "20s",
            "trace.explainLevel": "1",
            "trace.level": "1",
            "trace.profileDepth": "100",
            "ranking.matching.approximateThreshold": "0.00",
        }

        queries_with_parameters = list(
            map(lambda query: dict(query, **query_parameters), self.queries)
        )
        responses, response_times = execute_queries(self.app, queries_with_parameters)

        def extract_from_trace(obj: dict, type: str):
            results = []

            if "[type]" in obj and obj["[type]"] == type:
                results.append(obj)

            for k, v in obj.items():
                if isinstance(v, dict):
                    results += extract_from_trace(v, type)

                elif isinstance(v, list):
                    for i in v:
                        if isinstance(i, dict):
                            results += extract_from_trace(i, type)

            return results

        results = []
        for response in responses:
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

            results.append(hit_ratios)

        return results


class VespaNNRecallRelevanceMismatchError(Exception):
    """
    Exception raised when the reported relevance between exact and approximate query differs.
    """

    pass


class VespaNNRecallUnsuccessfulQueryError(Exception):
    """
    Exception raised when trying to compute the recall of an unsuccessful query.
    """

    pass


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
        queries (List[Dict[str, str]]): List of ANN queries.
        hits (int): Number of targetHits determined by the ANN queries.
        app (Vespa): An instance of the Vespa application.
        **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.
    """

    def __init__(self, queries: List[Dict[str, str]], hits: int, app: Vespa, **kwargs):
        self.queries = queries
        self.hits = hits
        self.app = app
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
            raise VespaNNRecallUnsuccessfulQueryError()

        try:
            results_exact = response_exact.get_json()["root"]["children"]
        except KeyError:
            results_exact = {}

        try:
            results_approx = response_approx.get_json()["root"]["children"]
        except KeyError:
            results_approx = {}

        size_exact = len(results_exact)
        size_approx = len(results_approx)

        recall = 0
        i = 0
        j = 0
        while i < size_exact and j < size_approx:
            exact = results_exact[i]
            approx = results_approx[j]
            relevance_exact = float(exact["relevance"])
            relevance_approx = float(approx["relevance"])
            if exact["id"] == approx["id"]:
                if abs(relevance_exact - relevance_approx) > 1e-5:
                    raise VespaNNRecallRelevanceMismatchError(
                        f"Results have the same id {exact['id']}, "
                        f"but relevances {relevance_exact} and {relevance_approx} do not match"
                    )
                recall += 1
                i += 1
                j += 1
            elif relevance_exact > relevance_approx:
                i += 1
            else:
                j += 1

        return recall / self.hits

    def run(self) -> List[float]:
        """
        Computes the recall of the supplied queries.

        Returns:
            List[float]: List of recall values from the interval [0.0, 1.0] corresponding to the supplied queries.
        """
        query_parameters = dict(
            self.parameters, **{"hits": self.hits, "timeout": "20s"}
        )
        query_parameters_exact = dict(
            query_parameters, **{"ranking.matching.approximateThreshold": 1.00}
        )

        queries_with_parameters = list(
            map(lambda query: dict(query, **query_parameters), self.queries)
        )
        responses, _ = execute_queries(self.app, queries_with_parameters)

        queries_with_parameters_exact = list(
            map(lambda query: dict(query, **query_parameters_exact), self.queries)
        )
        responses_exact, _ = execute_queries(self.app, queries_with_parameters_exact)

        return list(
            map(
                lambda pair: self._compute_recall(pair[0], pair[1]),
                zip(responses, responses_exact),
            )
        )


class VespaQueryBenchmarker:
    """
    Determine the searchtime of queries by running them multiple times and taking the average.
    Using the searchtime has the advantage of not including network latency.

    This class:

    - Takes a list of queries.
    - Runs the queries multiple times.
    - Determines the average searchtime of these runs.

    Args:
        queries (List[Dict[str, str]]): List of queries.
        app (Vespa): An instance of the Vespa application.
        repetitions (int, optional): Number of times to repeat the queries.
        **kwargs (dict, optional): Additional HTTP request parameters. See: <https://docs.vespa.ai/en/reference/document-v1-api-reference.html#request-parameters>.
    """

    def __init__(
        self, queries: List[Dict[str, str]], app: Vespa, repetitions: int = 5, **kwargs
    ):
        self.queries = queries
        self.app = app
        self.repetitions = repetitions
        self.parameters = kwargs

    def _run_benchmark(self) -> List[float]:
        """
        Run all queries once and extract the searchtime.

        Returns:
            List[float]: List of searchtimes, corresponding to the supplied queries.
        """
        queries_with_parameters = list(
            map(
                lambda query: dict(
                    query, **self.parameters, **{"presentation.timing": True}
                ),
                self.queries,
            )
        )
        _, response_times = execute_queries(self.app, queries_with_parameters)
        return response_times

    def run(self) -> List[float]:
        """
        Runs the benchmark (including a warm-up run not included in the result).

        Returns:
            List[float]: List of searchtimes, corresponding to the supplied queries.
        """
        # Two warmup runs
        for i in range(0, 2):
            self._run_benchmark()

        # Actual benchmark runs
        response_times_sum = [0] * len(self.queries)
        for i in range(0, self.repetitions):
            response_times = self._run_benchmark()
            response_times_ms = list(map(lambda x: 1000 * x, response_times))
            response_times_sum = list(
                map(
                    lambda pair: pair[0] + pair[1],
                    zip(response_times_sum, response_times_ms),
                )
            )

        return list(map(lambda x: x / self.repetitions, response_times_sum))
