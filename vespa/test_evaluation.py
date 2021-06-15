# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
import math

from vespa.io import VespaQueryResponse
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

        self.labeled_data2_with_zero_score = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": "ghi", "score": 0}, {"id": "abc", "score": 2}],
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

        self.labeled_data_int_id = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": 1, "score": 1}, {"id": 3, "score": 2}],
            },
        ]

        self.query_results_int_id = {
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
                            "vespa_id_field": 1,
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
                            "vespa_id_field": 2,
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
                            "vespa_id_field": 3,
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
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio": 1083 / 62529,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
            detailed_metrics=True,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio_retrieved_docs": 1083,
                "match_ratio_docs_available": 62529,
                "match_ratio": 1083 / 62529,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse({
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
            }, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio": 0 / 62529,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse({
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
            }, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
            detailed_metrics=True,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio_retrieved_docs": 0,
                "match_ratio_docs_available": 62529,
                "match_ratio": 0 / 62529,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse({
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
            }, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio": 0,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse({
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
            }, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
            detailed_metrics=True,
        )

        self.assertDictEqual(
            evaluation,
            {
                "match_ratio_retrieved_docs": 1083,
                "match_ratio_docs_available": 0,
                "match_ratio": 0,
            },
        )

    def test_recall(self):
        metric = Recall(at=2)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_2": 0.5,
            },
        )

        metric = Recall(at=1)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_1": 0.0,
            },
        )

        metric = Recall(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_3": 1,
            },
        )

        metric = Recall(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results_int_id, status_code=None, url=None),
            relevant_docs=self.labeled_data_int_id[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_3": 1,
            },
        )

    def test_recall_with_zero_score(self):

        metric = Recall(at=1)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2_with_zero_score[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_1": 0,
            },
        )

        metric = Recall(at=2)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2_with_zero_score[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_2": 0,
            },
        )

        metric = Recall(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2_with_zero_score[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_3": 1,
            },
        )

        metric = Recall(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=[{"id": "ghi", "score": 0}, {"id": "abc", "score": 0}],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "recall_3": 0,
            },
        )

    def test_reciprocal_rank(self):
        metric = ReciprocalRank(at=2)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_2": 0.5,
            },
        )

        metric = ReciprocalRank(at=1)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_1": 0.0,
            },
        )

        metric = ReciprocalRank(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_3": 1.0,
            },
        )

        metric = ReciprocalRank(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results_int_id, status_code=None, url=None),
            relevant_docs=self.labeled_data_int_id[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_3": 1.0,
            },
        )

    def test_reciprocal_rank_with_zero_score(self):

        metric = ReciprocalRank(at=1)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2_with_zero_score[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_1": 0.0,
            },
        )

        metric = ReciprocalRank(at=2)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2_with_zero_score[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_2": 0.0,
            },
        )

        metric = ReciprocalRank(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2_with_zero_score[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        self.assertDictEqual(
            evaluation,
            {
                "reciprocal_rank_3": 1 / 3,
            },
        )

    def test_normalized_discounted_cumulative_gain(self):

        metric = NormalizedDiscountedCumulativeGain(at=2)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        expected_dcg = 0 / math.log2(2) + 1 / math.log2(3)
        expected_ideal_dcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected_ndcg = expected_dcg / expected_ideal_dcg

        self.assertDictEqual(
            evaluation,
            {
                "ndcg_2": expected_ndcg,
            },
        )
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
            detailed_metrics=True,
        )

        self.assertDictEqual(
            evaluation,
            {
                "ndcg_2_ideal_dcg": expected_ideal_dcg,
                "ndcg_2_dcg": expected_dcg,
                "ndcg_2": expected_ndcg,
            },
        )

        metric = NormalizedDiscountedCumulativeGain(at=1)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        expected_dcg = 0 / math.log2(2)
        expected_ideal_dcg = 1 / math.log2(2)
        expected_ndcg = 0
        self.assertDictEqual(
            evaluation,
            {
                "ndcg_1": expected_ndcg,
            },
        )
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results, status_code=None, url=None),
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
            detailed_metrics=True,
        )

        self.assertDictEqual(
            evaluation,
            {
                "ndcg_1_ideal_dcg": expected_ideal_dcg,
                "ndcg_1_dcg": expected_dcg,
                "ndcg_1": expected_ndcg,
            },
        )

        metric = NormalizedDiscountedCumulativeGain(at=3)
        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
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
                "ndcg_3": expected_ndcg,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results2, status_code=None, url=None),
            relevant_docs=self.labeled_data2[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
            detailed_metrics=True,
        )

        self.assertDictEqual(
            evaluation,
            {
                "ndcg_3_ideal_dcg": expected_ideal_dcg,
                "ndcg_3_dcg": expected_dcg,
                "ndcg_3": expected_ndcg,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results_int_id, status_code=None, url=None),
            relevant_docs=self.labeled_data_int_id[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
        )
        expected_dcg = 1 / math.log2(2) + 0 / math.log2(3) + 2 / math.log2(4)
        expected_ideal_dcg = 2 / math.log2(2) + 1 / math.log2(3) + 0 / math.log2(4)
        expected_ndcg = expected_dcg / expected_ideal_dcg
        self.assertDictEqual(
            evaluation,
            {
                "ndcg_3": expected_ndcg,
            },
        )

        evaluation = metric.evaluate_query(
            query_results=VespaQueryResponse(self.query_results_int_id, status_code=None, url=None),
            relevant_docs=self.labeled_data_int_id[0]["relevant_docs"],
            id_field="vespa_id_field",
            default_score=0,
            detailed_metrics=True,
        )

        self.assertDictEqual(
            evaluation,
            {
                "ndcg_3_ideal_dcg": expected_ideal_dcg,
                "ndcg_3_dcg": expected_dcg,
                "ndcg_3": expected_ndcg,
            },
        )
