# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import math
from typing import Dict, List
from vespa.io import VespaQueryResponse


class EvalMetric(object):
    def __init__(self) -> None:
        pass

    def evaluate_query(
        self,
        query_results,
        relevant_docs,
        id_field,
        default_score,
        detailed_metrics=False,
    ) -> Dict:
        raise NotImplementedError


class MatchRatio(EvalMetric):
    def __init__(self) -> None:
        """
        Computes the ratio of documents retrieved by the match phase.
        """
        super().__init__()
        self.name = "match_ratio"

    def evaluate_query(
        self,
        query_results: VespaQueryResponse,
        relevant_docs: List[Dict],
        id_field: str,
        default_score: int,
        detailed_metrics=False,
    ) -> Dict:
        """
        Evaluate query results.

        :param query_results: Raw query results returned by Vespa.
        :param relevant_docs: A list with dicts where each dict contains a doc id a optionally a doc score.
        :param id_field: The Vespa field representing the document id.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param detailed_metrics: Return intermediate computations if available.
        :return: Dict containing the number of retrieved docs (_retrieved_docs), the number of docs available in
            the corpus (_docs_available) and the match ratio.
        """
        retrieved_docs = query_results.number_documents_retrieved
        docs_available = query_results.number_documents_indexed
        value = 0
        if docs_available > 0:
            value = retrieved_docs / docs_available
        metrics = {
            str(self.name): value,
        }
        if detailed_metrics:
            metrics.update(
                {
                    str(self.name) + "_retrieved_docs": retrieved_docs,
                    str(self.name) + "_docs_available": docs_available,
                }
            )
        return metrics


class Recall(EvalMetric):
    def __init__(self, at: int) -> None:
        """
        Compute the recall at position `at`

        :param at: Maximum position on the resulting list to look for relevant docs.
        """
        super().__init__()
        self.name = "recall_" + str(at)
        self.at = at

    def evaluate_query(
        self,
        query_results: VespaQueryResponse,
        relevant_docs: List[Dict],
        id_field: str,
        default_score: int,
        detailed_metrics=False,
    ) -> Dict:
        """
        Evaluate query results.

        There is an assumption that only documents with score > 0 are relevant. Recall is equal to zero in case no
        relevant documents with score > 0 is provided.

        :param query_results: Raw query results returned by Vespa.
        :param relevant_docs: A list with dicts where each dict contains a doc id a optionally a doc score.
        :param id_field: The Vespa field representing the document id.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param detailed_metrics: Return intermediate computations if available.
        :return: Dict containing the recall value.
        """

        relevant_ids = {str(doc["id"]) for doc in relevant_docs if doc.get("score", default_score) > 0}
        try:
            retrieved_ids = {
                str(hit["fields"][id_field]) for hit in query_results.hits[: self.at]
            }
        except KeyError:
            retrieved_ids = set()

        return {str(self.name): len(relevant_ids & retrieved_ids) / len(relevant_ids) if len(relevant_ids) > 0 else 0}


class ReciprocalRank(EvalMetric):
    def __init__(self, at: int):
        """
        Compute the reciprocal rank at position `at`

        :param at: Maximum position on the resulting list to look for relevant docs.
        """
        super().__init__()
        self.name = "reciprocal_rank_" + str(at)
        self.at = at

    def evaluate_query(
        self,
        query_results: VespaQueryResponse,
        relevant_docs: List[Dict],
        id_field: str,
        default_score: int,
        detailed_metrics=False,
    ) -> Dict:
        """
        Evaluate query results.

        There is an assumption that only documents with score > 0 are relevant.

        :param query_results: Raw query results returned by Vespa.
        :param relevant_docs: A list with dicts where each dict contains a doc id a optionally a doc score.
        :param id_field: The Vespa field representing the document id.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param detailed_metrics: Return intermediate computations if available.
        :return: Dict containing the reciprocal rank value.
        """

        relevant_ids = {str(doc["id"]) for doc in relevant_docs if doc.get("score", default_score) > 0}
        rr = 0
        hits = query_results.hits[: self.at]
        for index, hit in enumerate(hits):
            if str(hit["fields"][id_field]) in relevant_ids:
                rr = 1 / (index + 1)
                break

        return {str(self.name): rr}


class NormalizedDiscountedCumulativeGain(EvalMetric):
    def __init__(self, at: int):
        """
        Compute the normalized discounted cumulative gain at position `at`.

        :param at: Maximum position on the resulting list to look for relevant docs.
        """
        super().__init__()
        self.name = "ndcg_" + str(at)
        self.at = at

    @staticmethod
    def _compute_dcg(scores: List[int]) -> float:
        return sum([score / math.log2(idx + 2) for idx, score in enumerate(scores)])

    def evaluate_query(
        self,
        query_results: VespaQueryResponse,
        relevant_docs: List[Dict],
        id_field: str,
        default_score: int,
        detailed_metrics=False,
    ) -> Dict:
        """
        Evaluate query results.

        There is an assumption that documents returned by the query that are not included in the set of relevant
        documents have score equal to zero. Similarly, if the query returns a number `N < at` documents, we will
        assume that those `N - at` missing scores are equal to zero.

        :param query_results: Raw query results returned by Vespa.
        :param relevant_docs: A list with dicts where each dict contains a doc id a optionally a doc score.
        :param id_field: The Vespa field representing the document id.
        :param default_score: Score to assign to the additional documents that are not relevant. Default to 0.
        :param detailed_metrics: Return intermediate computations if available.
        :return: Dict containing the ideal discounted cumulative gain (_ideal_dcg), the discounted cumulative
            gain (_dcg) and the normalized discounted cumulative gain.
        """

        at = self.at
        relevant_scores = {str(doc["id"]): doc["score"] for doc in relevant_docs}
        assert default_score == 0, "NDCG default score should be zero."
        search_scores = [default_score] * at
        ideal_scores = [default_score] * at

        hits = query_results.hits[:at]
        for idx, hit in enumerate(hits):
            search_scores[idx] = relevant_scores.get(
                str(hit["fields"][id_field]), default_score
            )

        for idx, score in enumerate(
            sorted(list(relevant_scores.values()), reverse=True)[:at]
        ):
            ideal_scores[idx] = score

        ideal_dcg = self._compute_dcg(ideal_scores)
        dcg = self._compute_dcg(search_scores)

        ndcg = 0
        if ideal_dcg > 0:
            ndcg = dcg / ideal_dcg

        metrics = {
            str(self.name): ndcg,
        }
        if detailed_metrics:
            metrics.update(
                {
                    str(self.name) + "_ideal_dcg": ideal_dcg,
                    str(self.name) + "_dcg": dcg,
                }
            )
        return metrics
