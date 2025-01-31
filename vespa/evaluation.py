from __future__ import annotations
import os
import csv
import logging
from typing import Dict, Set, Callable, List, Optional, Union
import math

from vespa.application import Vespa
from vespa.io import VespaQueryResponse

logger = logging.getLogger(__name__)


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


class VespaEvaluator:
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

    Example usage::

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
    """

    def __init__(
        self,
        queries: Dict[str, str],
        relevant_docs: Union[Dict[str, Set[str]], Dict[str, str]],
        vespa_query_fn: Callable[[str, int], dict],
        app: Vespa,
        name: str = "",
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        map_at_k: List[int] = [100],
        write_csv: bool = False,
        csv_dir: Optional[str] = None,
    ):
        """
        :param queries: Dict of query_id => query text
        :param relevant_docs: Dict of query_id => set of relevant doc_ids (the user-specified part of `id:<namespace>:<document-type>:<key/value-pair>:<user-specified>` in Vespa, see https://docs.vespa.ai/en/documents.html#document-ids)
        :param vespa_query_fn: Callable, with signature: my_func(query:str, top_k: int)-> dict: Given a query string and top_k, returns a Vespa query body (dict).
        :param app: A `vespa.application.Vespa` instance.
        :param name: A name or tag for this evaluation run.
        :param accuracy_at_k: list of k-values for Accuracy@k
        :param precision_recall_at_k: list of k-values for Precision@k and Recall@k
        :param mrr_at_k: list of k-values for MRR@k
        :param ndcg_at_k: list of k-values for NDCG@k
        :param map_at_k: list of k-values for MAP@k
        :param write_csv: If True, writes results to CSV
        :param csv_dir: Path in which to write the CSV file (default: current working dir).
        """
        self._validate_queries(queries)
        self._validate_vespa_query_fn(
            vespa_query_fn
        )  # Add this line before _validate_qrels
        relevant_docs = self._validate_qrels(relevant_docs)

        # Filter out any queries that have no relevant docs
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]
        self.relevant_docs = relevant_docs

        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.map_at_k = map_at_k
        self.searchtimes: List[float] = []

        self.vespa_query_fn = vespa_query_fn
        self.app = app

        self.name = name
        self.write_csv = write_csv
        self.csv_dir = csv_dir

        self.primary_metric: Optional[str] = None

        self.csv_file: str = f"Vespa-evaluation_{name}_results.csv"

        # We'll collect metrics in a single pass, so define them up front.
        self.csv_headers = [
            "accuracy@{}",
            "precision@{}",
            "recall@{}",
            "mrr@{}",
            "ndcg@{}",
            "map@{}",
        ]

    def _validate_queries(self, queries: Dict[str, str]):
        if not isinstance(queries, dict):
            raise ValueError("queries must be a dict of query_id => query_text")
        for qid, query_text in queries.items():
            if not isinstance(qid, str) or not isinstance(query_text, str):
                raise ValueError("Each query must be a string.", qid, query_text)

    def _validate_qrels(
        self, qrels: Union[Dict[str, Set[str]], Dict[str, str]]
    ) -> Dict[str, Set[str]]:
        if not isinstance(qrels, dict):
            raise ValueError(
                "qrels must be a dict of query_id => set of relevant doc_ids"
            )
        new_qrels: Dict[str, Set[str]] = {}
        for qid, relevant_docs in qrels.items():
            if not isinstance(qid, str):
                raise ValueError(
                    "Each qrel must be a string query_id and a set of doc_ids.",
                    qid,
                    relevant_docs,
                )
            if isinstance(relevant_docs, str):
                new_qrels[qid] = {relevant_docs}
            elif isinstance(relevant_docs, set):
                new_qrels[qid] = relevant_docs
            else:
                raise ValueError(
                    f"Relevant docs for query {qid} must be a set or string."
                )
        return new_qrels

    def _validate_vespa_query_fn(self, fn: Callable[[str, int], dict]) -> None:
        """
        Validate that vespa_query_fn is callable and has correct signature.

        :param fn: Function to validate
        :raises ValueError: If function doesn't meet requirements
        :raises TypeError: If function signature is incorrect
        """
        if not callable(fn):
            raise ValueError("vespa_query_fn must be a callable")

        import inspect

        sig = inspect.signature(fn)
        params = list(sig.parameters.items())

        # Check number of parameters
        if len(params) != 2:
            raise TypeError(
                f"vespa_query_fn must take exactly 2 parameters (query_text, top_k), got {len(params)}"
            )

        # Check parameter types from type hints
        param_types = {name: param.annotation for name, param in params}

        expected_types = {params[0][0]: str, params[1][0]: int}

        for param_name, expected_type in expected_types.items():
            if param_types.get(param_name) not in (
                expected_type,
                inspect.Parameter.empty,
            ):
                raise TypeError(
                    f"Parameter '{param_name}' must be of type {expected_type.__name__}"
                )

        # Validate the function can actually be called with test inputs
        try:
            result = fn("test query", 10)
            if not isinstance(result, dict):
                raise TypeError("vespa_query_fn must return a dict")
        except Exception as e:
            raise ValueError(f"Error calling vespa_query_fn with test inputs: {str(e)}")

    def run(self) -> Dict[str, float]:
        """
        Execute the evaluation by running queries and computing IR metrics.

        This method:
        1. Executes all configured queries against the Vespa application
        2. Collects search results and timing information
        3. Computes configured IR metrics (Accuracy@k, Precision@k, Recall@k, MRR@k, NDCG@k, MAP@k)
        4. Records search timing statistics
        5. Logs results and optionally writes them to CSV

        Returns:
            Dict[str, float]: Dictionary containing:
                - IR metrics with names like "accuracy@k", "precision@k", etc.
                - Search time statistics ("searchtime_avg", "searchtime_q50", etc.)
                where values are floats between 0 and 1 for metrics, seconds for timing.

        Example returned metrics:
            {
                "accuracy@1": 0.75,
                "ndcg@10": 0.68,
                "searchtime_avg": 0.0123,
                ...
            }
        """
        max_k = max(
            max(self.accuracy_at_k) if self.accuracy_at_k else 0,
            max(self.precision_recall_at_k) if self.precision_recall_at_k else 0,
            max(self.mrr_at_k) if self.mrr_at_k else 0,
            max(self.ndcg_at_k) if self.ndcg_at_k else 0,
            max(self.map_at_k) if self.map_at_k else 0,
        )

        logger.info(f"Starting VespaEvaluator on {self.name}")
        logger.info(f"Number of queries: {len(self.queries_ids)}; max_k = {max_k}")

        queries_result_list = []
        for idx, qid in enumerate(self.queries_ids):
            query_text = self.queries[idx]
            query_body = self.vespa_query_fn(query_text, max_k)
            if "presentation.timing" not in query_body.keys():
                logger.warning(
                    "Timing information is not included in the query body. "
                    'Please include `"presentation.timing": True` in the query body to log search times.'
                )
            logger.debug(f"Querying Vespa with: {query_body}")
            vespa_response: VespaQueryResponse = self.app.query(body=query_body)

            # Attempt to get search time from Vespa's JSON
            timing = vespa_response.get_json().get("timing", {}).get("searchtime", 0)
            self.searchtimes.append(timing)

            hits = vespa_response.hits or []
            top_hit_list = []
            for hit in hits[:max_k]:
                # doc_id extraction logic
                doc_id = str(hit.get("id", "").split("::")[-1])
                if not doc_id:
                    raise ValueError(f"Could not extract doc_id from hit: {hit}")
                score = float(hit.get("relevance", float("nan")))
                if math.isnan(score):
                    raise ValueError(f"Could not extract relevance from hit: {hit}")
                top_hit_list.append((doc_id, score))

            queries_result_list.append(top_hit_list)

        metrics = self._compute_metrics(queries_result_list)
        searchtime_stats = self._calculate_searchtime_stats()
        metrics.update(searchtime_stats)

        if not self.primary_metric:
            if self.ndcg_at_k:
                best_k = max(self.ndcg_at_k)
                self.primary_metric = f"ndcg@{best_k}"
            else:
                # fallback to some default
                self.primary_metric = "accuracy@1" if self.accuracy_at_k else "map@100"

        self._log_metrics(metrics)

        if self.write_csv:
            self._write_csv(metrics, searchtime_stats)

        return metrics

    def _calculate_searchtime_stats(self) -> Dict[str, float]:
        if not self.searchtimes:
            return {}
        return {
            "searchtime_avg": mean(self.searchtimes),
            "searchtime_q50": percentile(self.searchtimes, 50),
            "searchtime_q90": percentile(self.searchtimes, 90),
            "searchtime_q95": percentile(self.searchtimes, 95),
        }

    def _compute_metrics(self, queries_result_list):
        num_queries = len(queries_result_list)
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
                    if doc_id in relevant:
                        num_correct += 1
                        sum_precisions += num_correct / rank
                denom = min(k_val, len(relevant))
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

    def _log_metrics(self, metrics: Dict[str, float]):
        logger.info(f"Vespa IR evaluation on {self.name}")
        for metric_name, value in metrics.items():
            if (
                metric_name.startswith("accuracy")
                or metric_name.startswith("precision")
                or metric_name.startswith("recall")
            ):
                logger.info(f"{metric_name}: {value * 100:.2f}%")
            else:
                logger.info(f"{metric_name}: {value:.4f}")

    def _write_csv(self, metrics: Dict[str, float], searchtime_stats: Dict[str, float]):
        csv_path = self.csv_file
        if self.csv_dir is not None:
            csv_path = os.path.join(self.csv_dir, csv_path)

        combined_metrics = {**metrics, **searchtime_stats}
        write_header = not os.path.exists(csv_path)

        with open(csv_path, mode="a", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            if write_header:
                header = sorted(combined_metrics.keys())
                header.insert(0, "name")  # extra column for "run name"
                writer.writerow(header)
            row_keys = sorted(combined_metrics.keys())
            row = [self.name] + [combined_metrics[k] for k in row_keys]
            writer.writerow(row)

        logger.info(f"Wrote IR evaluation metrics and search times to {csv_path}")
