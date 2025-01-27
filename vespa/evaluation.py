from __future__ import annotations
import os
import csv
import logging
from typing import Dict, Set, Callable, List, Optional, Union
import numpy as np
from vespa.application import Vespa

logger = logging.getLogger(__name__)


class VespaEvaluator:
    """
    Evaluate retrieval performance on a Vespa application.

    This class:
    - Iterates over queries and issues them against your Vespa application.
    - Retrieves top-k documents per query (with k = max of your IR metrics).
    - Compares the retrieved documents with a set of relevant documents.
    - Computes IR metrics: Accuracy@k, Precision@k, Recall@k, MRR@k, NDCG@k, MAP@k.
    - Logs/returns these metrics.
    - Optionally writes out to CSV.

    Example usage::

        from vespa.application import Vespa
        from vespa.evaluation import VespaEvaluator

        #
        # 1) Define your queries, relevant docs, etc.
        #
        my_queries = {
            "q1": "What is the best GPU for gaming?",
            "q2": "How to bake sourdough bread?",
            # ...
        }
        my_relevant_docs = {
            "q1": {"d12", "d99"},
            "q2": {"d101"},
            # ...
        }


        #
        # 2) Define a function that, given a query string, returns the proper
        #    Vespa body for app.query().
        #
        # def my_vespa_query_fn(query_text: str, top_k: int) -> dict:
        #     '''
        #     Convert a plain text user query to a Vespa query body dict.
        #     The example below uses a YQL statement and requests a fixed number of hits.
        #     Adapt this for your ranking profile, filters, etc.
        #     '''
        #     return {
        #         "yql": 'select * from sources * where userInput("' + query_text + '");',
        #         "hits": top_k,
        #         "ranking": "your_ranking_profile",
        #         # add other parameters, e.g. "presentation.summary": "your_summary_class"
        #     }
        #
        # 3) Instantiate the evaluator with the chosen IR metrics and run it.
        #

        app = Vespa(url="http://localhost", port=8080)  # or your Vespa endpoint

        evaluator = VespaEvaluator(
            queries=my_queries,
            relevant_docs=my_relevant_docs,
            vespa_query_fn=my_vespa_query_fn,
            app=app,
            name="test-run",
            accuracy_at_k=[1, 3, 5],
            precision_recall_at_k=[1, 3, 5],
            mrr_at_k=[10],
            ndcg_at_k=[10],
            map_at_k=[100],
            write_csv=True,  # optionally write metrics to CSV
        )

        results = evaluator()
        # logs metrics such as:
        #   Accuracy@1, @3, @5
        #   Precision@1, @3, @5
        #   Recall@1, @3, @5
        #   MRR@10, NDCG@10, MAP@100
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
        :param relevant_docs: Dict of query_id => set of relevant doc_ids
        :param vespa_query_fn: Given a query string and top_k, returns a Vespa query body (dict) suitable for app.query(...).
        :param app: A `vespa.application.Vespa` instance.
        :param name: A name or tag for this evaluation run.
        :param accuracy_at_k: list of k-values for Accuracy@k
        :param precision_recall_at_k: list of k-values for Precision@k and Recall@k
        :param mrr_at_k: list of k-values for MRR@k
        :param ndcg_at_k: list of k-values for NDCG@k
        :param map_at_k: list of k-values for MAP@k
        :param write_csv: If True, writes results to CSV
        :param csv_dir: If provided, path in which to write the CSV file. By default, current working dir.
        """
        # Validate inputs
        self._validate_queries(queries)
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
        # We'll expand them into actual columns when writing.

    def _validate_queries(self, queries: Dict[str, str]):
        """
        Ensure that queries are proper format and type.
        """
        if not isinstance(queries, dict):
            raise ValueError("queries must be a dict of query_id => query_text")
        for qid, query_text in queries.items():
            if not isinstance(qid, str) or not isinstance(query_text, str):
                raise ValueError("Each query must be a string.", qid, query_text)

    def _validate_qrels(
        self, qrels: Union[Dict[str, Set[str]], Dict[str, str]]
    ) -> Dict[str, Set[str]]:
        """
        Ensure that qrels are proper format and type.
        Returns normalized qrels where all values are sets.
        """
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
            assert isinstance(new_qrels[qid], set)
        return new_qrels

    def __call__(self) -> Dict[str, float]:
        """
        Perform the evaluation: run queries against Vespa, compute IR metrics, and
        optionally write CSV.
        Returns a dict with all metrics, e.g.:
          {
            "accuracy@1": 0.75,
            "accuracy@3": 0.82,
            ...,
            "ndcg@10": 0.66,
            ...
          }
        """
        # Step 1: figure out the maximum K we need to retrieve from Vespa
        max_k = max(
            max(self.accuracy_at_k) if self.accuracy_at_k else 0,
            max(self.precision_recall_at_k) if self.precision_recall_at_k else 0,
            max(self.mrr_at_k) if self.mrr_at_k else 0,
            max(self.ndcg_at_k) if self.ndcg_at_k else 0,
            max(self.map_at_k) if self.map_at_k else 0,
        )
        logger.info(f"Starting VespaEvaluator on {self.name}")
        logger.info(f"Number of queries: {len(self.queries_ids)}; max_k = {max_k}")

        # Step 2: Collect top hits for each query (using the user-provided vespa_query_fn)
        # We'll store them in a structure: results_list[q_idx] = list of (doc_id, score)
        #   sorted by whatever order Vespa returns. (We assume top hits are in rank order.)
        queries_result_list = []
        for idx, qid in enumerate(self.queries_ids):
            query_text = self.queries[idx]
            # Build the query body with max_k
            query_body = self.vespa_query_fn(query_text, max_k)
            logger.debug(f"Querying Vespa with: {query_body}")
            vespa_response = self.app.query(body=query_body)
            # The vespa_response is typically a `VespaQueryResponse`.
            # You can parse hits via: vespa_response.hits (pyvespa adds hits, fields, etc.)
            hits = vespa_response.hits or []
            # hits is a list of dict, each with typical structure: {"id": "...", "relevance": "...", ...}.
            # We'll store doc_id, and also keep a "score" if needed. For ranking metrics we only need the order.
            top_hit_list = []
            for hit in hits[:max_k]:
                doc_id = str(
                    hit.get("id", "").split("::")[-1]
                )  # doc IDs from Vespa. Adjust if your doc id is in a sub-field.
                score = float(hit.get("relevance", 1.0))  # or 1.0 if missing
                top_hit_list.append((doc_id, score))

            queries_result_list.append(top_hit_list)

        # Step 3: compute metrics
        metrics = self._compute_metrics(queries_result_list)

        # Step 4: determine primary metric if needed
        if not self.primary_metric:
            # For example, pick the largest ndcg@K
            if self.ndcg_at_k:
                best_k = max(self.ndcg_at_k)
                self.primary_metric = f"ndcg@{best_k}"
            else:
                # fallback
                self.primary_metric = "accuracy@1" if self.accuracy_at_k else "map@100"

        # Step 5: log and optionally write CSV
        self._log_metrics(metrics)
        if self.write_csv:
            self._write_csv(metrics)

        return metrics

    def _compute_metrics(self, queries_result_list):
        """
        queries_result_list: List of lists, each entry for one query.
          Each sub-list is a list of (doc_id, score), sorted from most relevant to least.
        """
        # Initialize accumulators
        num_queries = len(queries_result_list)

        # For each metric, we keep either a running sum or a list to compute average
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precision_at_k_list = {k: [] for k in self.precision_recall_at_k}
        recall_at_k_list = {k: [] for k in self.precision_recall_at_k}
        mrr_at_k = {k: 0.0 for k in self.mrr_at_k}
        ndcg_at_k_list = {k: [] for k in self.ndcg_at_k}
        map_at_k_list = {k: [] for k in self.map_at_k}

        for query_idx, top_hits in enumerate(queries_result_list):
            qid = self.queries_ids[query_idx]
            relevant = self.relevant_docs[qid]

            #
            # Accuracy@K
            #
            for k_val in self.accuracy_at_k:
                found_correct = False
                for doc_id, _score in top_hits[:k_val]:
                    if doc_id in relevant:
                        found_correct = True
                        break
                if found_correct:
                    num_hits_at_k[k_val] += 1

            #
            # Precision@K, Recall@K
            #
            for k_val in self.precision_recall_at_k:
                k_hits = top_hits[:k_val]
                num_correct = sum(1 for doc_id, _ in k_hits if doc_id in relevant)
                precision_at_k_list[k_val].append(num_correct / k_val)
                recall_at_k_list[k_val].append(num_correct / len(relevant))

            #
            # MRR@K
            #
            for k_val in self.mrr_at_k:
                reciprocal_rank = 0.0
                for rank, (doc_id, _) in enumerate(top_hits[:k_val]):
                    if doc_id in relevant:
                        reciprocal_rank = 1.0 / (rank + 1)
                        break
                mrr_at_k[k_val] += reciprocal_rank

            #
            # NDCG@K
            #
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if doc_id in relevant else 0 for doc_id, _ in top_hits[:k_val]
                ]
                true_relevances = [1] * len(
                    relevant
                )  # for that query, all relevant docs have rel=1
                ndcg_val = self._dcg_at_k(predicted_relevance, k_val) / self._dcg_at_k(
                    true_relevances, min(k_val, len(true_relevances))
                )
                ndcg_at_k_list[k_val].append(ndcg_val)

            #
            # MAP@K
            #
            for k_val in self.map_at_k:
                # We'll measure average precision across relevant docs in top_k.
                num_correct = 0
                sum_precisions = 0.0
                top_k_hits = top_hits[:k_val]
                for rank, (doc_id, _) in enumerate(top_k_hits, start=1):
                    if doc_id in relevant:
                        num_correct += 1
                        sum_precisions += num_correct / rank
                # If we have R relevant docs overall, we compute average precision as:
                #   sum(precision at each relevant doc) / min(k, R)
                # This is a common approach, but can vary by definition.
                denom = min(k_val, len(relevant))
                if denom > 0:
                    avg_precision = sum_precisions / denom
                else:
                    avg_precision = 0.0
                map_at_k_list[k_val].append(avg_precision)

        # Compute means
        metrics = {}
        # Accuracy@k
        for k_val in self.accuracy_at_k:
            metrics[f"accuracy@{k_val}"] = num_hits_at_k[k_val] / num_queries
        # Precision@k
        for k_val in self.precision_recall_at_k:
            metrics[f"precision@{k_val}"] = (
                float(np.mean(precision_at_k_list[k_val]))
                if precision_at_k_list[k_val]
                else 0.0
            )
        # Recall@k
        for k_val in self.precision_recall_at_k:
            metrics[f"recall@{k_val}"] = (
                float(np.mean(recall_at_k_list[k_val]))
                if recall_at_k_list[k_val]
                else 0.0
            )
        # MRR@k
        for k_val in self.mrr_at_k:
            metrics[f"mrr@{k_val}"] = mrr_at_k[k_val] / num_queries
        # nDCG@k
        for k_val in self.ndcg_at_k:
            metrics[f"ndcg@{k_val}"] = (
                float(np.mean(ndcg_at_k_list[k_val])) if ndcg_at_k_list[k_val] else 0.0
            )
        # MAP@k
        for k_val in self.map_at_k:
            metrics[f"map@{k_val}"] = (
                float(np.mean(map_at_k_list[k_val])) if map_at_k_list[k_val] else 0.0
            )

        return metrics

    def _dcg_at_k(self, relevances, k):
        """
        Discounted Cumulative Gain.
        relevances: list of 0/1 indicating relevant or not
        k: top-k
        """
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], start=1):
            dcg += rel / np.log2(i + 1)
        return dcg

    def _log_metrics(self, metrics: Dict[str, float]):
        logger.info(f"Vespa IR evaluation on {self.name}")
        for metric_name, value in metrics.items():
            if (
                metric_name.startswith("accuracy")
                or metric_name.startswith("precision")
                or metric_name.startswith("recall")
            ):
                logger.info(f"{metric_name}: {value*100:.2f}%")
            else:
                logger.info(f"{metric_name}: {value:.4f}")

    def _write_csv(self, metrics: Dict[str, float]):
        csv_path = self.csv_file
        if self.csv_dir is not None:
            csv_path = os.path.join(self.csv_dir, csv_path)

        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode="a", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            if write_header:
                # Example: we can store a row of metric keys as header
                header = sorted(metrics.keys())
                header.insert(0, "name")  # an extra column for "run name"
                writer.writerow(header)
            row_keys = sorted(metrics.keys())
            row = [self.name] + [metrics[k] for k in row_keys]
            writer.writerow(row)
        logger.info(f"Wrote IR metrics to {csv_path}")
