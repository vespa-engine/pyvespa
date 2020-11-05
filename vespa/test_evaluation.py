# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
import math

from vespa.query import VespaResult
from vespa.evaluation import (
    MatchRatio,
    Recall,
    ReciprocalRank,
    NormalizedDiscountedCumulativeGain,
)


class TestEvalMetric(unittest.TestCase):
    def setUp(self) -> None:
        self.labeled_data = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": "def", "score": 1}, {"id": "abc", "score": 1}],
            },
        ]

        self.query_results = {
            "root": {
                "id": "toplevel",
                "relevance": 1.0,
                "fields": {"totalCount": 1083},
                "coverage": {
                    "coverage": 100,
                    "documents": 62529,
                    "full": True,
                    "nodes": 2,
                    "results": 1,
                    "resultsFull": 1,
                },
                "children": [
                    {
                        "id": "id:covid-19:doc::40216",
                        "relevance": 10,
                        "source": "content",
                        "fields": {
                            "vespa_id_field": "ghi",
                            "sddocname": "doc",
                            "body_text": "this is a body 2",
                            "title": "this is a title 2",
                            "rankfeatures": {"a": 3, "b": 4},
                        },
                    },
                    {
                        "id": "id:covid-19:doc::40217",
                        "relevance": 8,
                        "source": "content",
                        "fields": {
                            "vespa_id_field": "def",
                            "sddocname": "doc",
                            "body_text": "this is a body 3",
                            "title": "this is a title 3",
                            "rankfeatures": {"a": 5, "b": 6},
                        },
                    },
                ],
            }
        }

        self.labeled_data2 = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": "ghi", "score": 1}, {"id": "abc", "score": 2}],
            },
        ]

        self.query_results2 = {
            "root": {
                "id": "toplevel",
                "relevance": 1.0,
                "fields": {"totalCount": 1083},
                "coverage": {
                    "coverage": 100,
                    "documents": 62529,
                    "full": True,
                    "nodes": 2,
                    "results": 1,
                    "resultsFull": 1,
                },
                "children": [
                    {
                        "id": "id:covid-19:doc::40216",
                        "relevance": 10,
                        "source": "content",
                        "fields": {
                            "vespa_id_field": "ghi",
                            "sddocname": "doc",
                            "body_text": "this is a body 2",
                            "title": "this is a title 2",
                            "rankfeatures": {"a": 3, "b": 4},
                        },
                    },
                    {
                        "id": "id:covid-19:doc::40217",
                        "relevance": 8,
                        "source": "content",
                        "fields": {
                            "vespa_id_field": "def",
                            "sddocname": "doc",
                            "body_text": "this is a body 3",
                            "title": "this is a title 3",
                            "rankfeatures": {"a": 5, "b": 6},
                        },
                    },
                    {
                        "id": "id:covid-19:doc::40218",
                        "relevance": 6,
                        "source": "content",
                        "fields": {
                            "vespa_id_field": "abc",
                            "sddocname": "doc",
                            "body_text": "this is a body 4",
                            "title": "this is a title 4",
                            "rankfeatures": {"a": 7, "b": 8},
                        },
                    },
                ],
            }
        }

    def test_match_ratio(self):
        metric = MatchRatio()

        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio_retrieved_docs": 1083,
                "match_ratio_docs_available": 62529,
                "match_ratio_value": 1083 / 62529,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaResult(
                {
                    "root": {
                        "id": "toplevel",
                        "relevance": 1.0,
                        "coverage": {
                            "coverage": 100,
                            "documents": 62529,
                            "full": True,
                            "nodes": 2,
                            "results": 1,
                            "resultsFull": 1,
                        },
                    }
                }
            ),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio_retrieved_docs": 0,
                "match_ratio_docs_available": 62529,
                "match_ratio_value": 0 / 62529,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaResult(
                {
                    "root": {
                        "id": "toplevel",
                        "relevance": 1.0,
                        "fields": {"totalCount": 1083},
                        "coverage": {
                            "coverage": 100,
                            "full": True,
                            "nodes": 2,
                            "results": 1,
                            "resultsFull": 1,
                        },
                    }
                }
            ),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio_retrieved_docs": 1083,
                "match_ratio_docs_available": 0,
                "match_ratio_value": 0,
            },
        )

    def test_recall(self):
        metric = Recall(at=2)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_2_value": 0.5,
            },
        )

        metric = Recall(at=1)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_1_value": 0.0,
            },
        )

        metric = Recall(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results2),
            relevant_docs=self.labeled_data2[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_3_value": 1,
            },
        )

    def test_reciprocal_rank(self):
        metric = ReciprocalRank(at=2)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_2_value": 0.5,
            },
        )

        metric = ReciprocalRank(at=1)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_1_value": 0.0,
            },
        )

        metric = ReciprocalRank(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results2),
            relevant_docs=self.labeled_data2[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_3_value": 1.0,
            },
        )

    def test_normalized_discounted_cumulative_gain(self):

        metric = NormalizedDiscountedCumulativeGain(at=2)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        expected_dcg = 0 / math.log2(2) + 1 / math.log2(3)
        expected_ideal_dcg = 1 / math.log2(2) + 0 / math.log2(3)
        expected_ndcg = expected_dcg / expected_ideal_dcg
        self.assertDictEqual(
            evaluation,
            {
                "ndcg_2_ideal_dcg": expected_ideal_dcg,
                "ndcg_2_dcg": expected_dcg,
                "ndcg_2_value": expected_ndcg,
            },
        )

        metric = NormalizedDiscountedCumulativeGain(at=1)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        expected_dcg = 0 / math.log2(2)
        expected_ideal_dcg = 0 / math.log2(3)
        expected_ndcg = 0
        self.assertDictEqual(
            evaluation,
            {
                "ndcg_1_ideal_dcg": expected_ideal_dcg,
                "ndcg_1_dcg": expected_dcg,
                "ndcg_1_value": expected_ndcg,
            },
        )

        metric = NormalizedDiscountedCumulativeGain(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaResult(self.query_results2),
            relevant_docs=self.labeled_data2[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        expected_dcg = 1 / math.log2(2) + 0 / math.log2(3) + 2 / math.log2(4)
        expected_ideal_dcg = 2 / math.log2(2) + 1 / math.log2(3) + 0 / math.log2(4)
        expected_ndcg = expected_dcg / expected_ideal_dcg
        self.assertDictEqual(
            evaluation,
            {
                "ndcg_3_ideal_dcg": expected_ideal_dcg,
                "ndcg_3_dcg": expected_dcg,
                "ndcg_3_value": expected_ndcg,
            },
        )
