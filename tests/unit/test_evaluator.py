import unittest
from vespa.evaluation import (
    VespaEvaluator,
    VespaMatchEvaluator,
    VespaCollectorBase,
    VespaFeatureCollector,
)
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import tempfile
import os
import csv


@dataclass
class MockVespaResponse:
    """Mock Vespa query response"""

    hits: List[Dict[str, Any]]
    _total_count: Optional[int] = None  # Added for totalCount
    _timing: Optional[Dict[str, float]] = (
        None  # Added for timing e.g. {"searchtime": 0.1}
    )
    _status_code: int = 200  # Added for status code control

    def add_namespace_to_hit_ids(
        self, hits_list
    ) -> List[Dict[str, Any]]:  # Renamed hits to hits_list
        new_hits = []
        for hit_item in hits_list:  # Renamed hit to hit_item
            # Ensure id is a string before trying to check substring
            # And ensure 'id' key exists
            hit_id = hit_item.get("id")
            if isinstance(hit_id, str) and "id:mynamespace:mydoctype::" not in hit_id:
                hit_item["id"] = f"id:mynamespace:mydoctype::{hit_id}"
            elif (
                not isinstance(hit_id, str)
                and "fields" in hit_item
                and isinstance(hit_item["fields"].get("id"), str)
            ):
                # Fallback for id in fields, if top-level id is not a string or missing
                field_id = hit_item["fields"]["id"]
                if "id:mynamespace:mydoctype::" not in field_id:
                    hit_item["id"] = f"id:mynamespace:mydoctype::{field_id}"
                else:
                    hit_item["id"] = field_id  # Use it directly if already namespaced
            elif (
                not hit_id
                and "fields" in hit_item
                and isinstance(hit_item["fields"].get("id"), str)
            ):
                # if hit_item["id"] was None or empty string
                field_id = hit_item["fields"]["id"]
                hit_item["id"] = f"id:mynamespace:mydoctype::{field_id}"

            new_hits.append(hit_item)
        return new_hits

    def get_json(self):
        json_data = {"root": {}}
        # children should only be present if there are hits
        if self.hits:
            # Ensure hits are processed for id namespacing before being added to json
            processed_hits = self.add_namespace_to_hit_ids(self.hits)
            json_data["root"]["children"] = processed_hits
        else:
            json_data["root"]["children"] = []

        if self._total_count is not None:
            if "fields" not in json_data["root"]:  # Ensure fields key exists
                json_data["root"]["fields"] = {}
            json_data["root"]["fields"]["totalCount"] = self._total_count

        if self._timing:
            json_data["timing"] = self._timing

        return json_data

    @property
    def status_code(self):
        return self._status_code


class QueryBodyCapturingApp:
    """Mock Vespa app that captures query bodies passed to query_many."""

    def __init__(self, responses):
        self.responses = responses
        self.captured_query_bodies = None

    def query_many(self, query_bodies):
        self.captured_query_bodies = query_bodies
        return self.responses


class TestVespaEvaluator(unittest.TestCase):
    def setUp(self):
        # Sample queries
        self.queries = {
            "q1": "what is machine learning",
            "q2": "how to code python",
            "q3": "what is the capital of France",
        }

        # Sample relevant docs
        self.relevant_docs = {
            "q1": {"doc1", "doc2", "doc3"},
            "q2": {"doc4", "doc5"},
            "q3": {"doc6"},
        }

        self.relevant_docs_single = {
            "q1": "doc1",
            "q2": "doc4",
            "q3": "doc6",
        }

        self.relevant_docs_relevance = {
            "q1": {"doc1": 1.0, "doc2": 0.5, "doc3": 0.2},
            "q2": {"doc4": 0.8, "doc5": 0.6},
            "q3": {"doc6": 1.0},
        }

        # Mock Vespa responses
        # For q1: doc1 at rank 1, doc2 at rank 3, doc3 at rank 5
        q1_response = MockVespaResponse(
            [
                {"id": "doc1", "relevance": 0.9},
                {"id": "doc10", "relevance": 0.8},
                {"id": "doc2", "relevance": 0.7},
                {"id": "doc11", "relevance": 0.6},
                {"id": "doc3", "relevance": 0.5},
            ]
        )

        # For q2: doc4 at rank 2, doc5 at rank 4
        q2_response = MockVespaResponse(
            [
                {"id": "doc12", "relevance": 0.95},
                {"id": "doc4", "relevance": 0.85},
                {"id": "doc13", "relevance": 0.75},
                {"id": "doc5", "relevance": 0.65},
                {"id": "doc14", "relevance": 0.55},
            ]
        )
        # For q3: doc6 at rank 1
        q3_response = MockVespaResponse(
            [
                {"id": "doc6", "relevance": 0.9},
                {"id": "doc16", "relevance": 0.8},
                {"id": "doc17", "relevance": 0.7},
                {"id": "doc18", "relevance": 0.6},
                {"id": "doc19", "relevance": 0.5},
            ]
        )

        class MockVespaApp:
            def __init__(self, mock_responses):
                self.mock_responses = mock_responses
                self.current_query = 0

            def query_many(self, queries):
                return self.mock_responses

        self.mock_app = MockVespaApp([q1_response, q2_response, q3_response])

        def mock_vespa_query_fn(query_text: str, top_k: int) -> dict:
            return {
                "yql": f'select * from sources * where text contains "{query_text}";',
                "hits": top_k,
            }

        self.vespa_query_fn = mock_vespa_query_fn

    def test_basic_initialization(self):
        """Test basic initialization with default parameters"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )

        self.assertEqual(len(evaluator.queries_ids), 3)
        self.assertEqual(set(evaluator.queries_ids), {"q1", "q2", "q3"})
        self.assertEqual(evaluator.accuracy_at_k, [1, 3, 5, 10])
        self.assertEqual(evaluator.precision_recall_at_k, [1, 3, 5, 10])
        self.assertEqual(evaluator.mrr_at_k, [10])
        self.assertEqual(evaluator.ndcg_at_k, [10])
        self.assertEqual(evaluator.map_at_k, [100])

    def test_init_single_relevant_docs(self):
        """Test initialization with single relevant doc per query"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs_single,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )
        relevant_docs_to_set = {  # Convert to the same format as self.relevant_docs
            q_id: {doc_id} for q_id, doc_id in self.relevant_docs_single.items()
        }
        self.assertEqual(evaluator.relevant_docs, relevant_docs_to_set)

    def test_init_relevant_docs_with_relevance(self):
        """Test initialization with relevant docs having relevance scores"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs_relevance,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )
        self.assertEqual(evaluator.relevant_docs, self.relevant_docs_relevance)

    def test_custom_k_values(self):
        """Test initialization with custom k values"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
            accuracy_at_k=[1, 2],
            precision_recall_at_k=[1, 2, 3],
            mrr_at_k=[5],
            ndcg_at_k=[5],
            map_at_k=[5],
        )

        self.assertEqual(evaluator.accuracy_at_k, [1, 2])
        self.assertEqual(evaluator.precision_recall_at_k, [1, 2, 3])
        self.assertEqual(evaluator.mrr_at_k, [5])
        self.assertEqual(evaluator.ndcg_at_k, [5])
        self.assertEqual(evaluator.map_at_k, [5])

    def test_accuracy_metrics(self):
        """Test accuracy@k calculations"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
            accuracy_at_k=[1, 3, 5],
            precision_recall_at_k=[],
            mrr_at_k=[],
            ndcg_at_k=[],
            map_at_k=[],
        )

        results = evaluator.run()

        self.assertAlmostEqual(results["accuracy@1"], 2 / 3)  # q1 and q3 hit at 1
        self.assertAlmostEqual(results["accuracy@3"], 1.0)
        self.assertAlmostEqual(results["accuracy@5"], 1.0)

    def test_precision_recall_metrics(self):
        """Test precision@k and recall@k calculations"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
            accuracy_at_k=[],
            precision_recall_at_k=[3, 5],
            mrr_at_k=[],
            ndcg_at_k=[],
            map_at_k=[],
        )

        results = evaluator.run()

        self.assertAlmostEqual(results["precision@3"], (2 / 3 + 1 / 3 + 1 / 3) / 3)
        self.assertAlmostEqual(results["recall@3"], (2 / 3 + 1 / 2 + 1 / 1) / 3)
        self.assertAlmostEqual(results["precision@5"], (3 / 5 + 2 / 5 + 1 / 5) / 3)
        self.assertAlmostEqual(results["recall@5"], (3 / 3 + 2 / 2 + 1 / 1) / 3)

    def test_mrr_metric(self):
        """Test MRR@k calculations"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
            accuracy_at_k=[],
            precision_recall_at_k=[],
            mrr_at_k=[5],
            ndcg_at_k=[],
            map_at_k=[],
        )

        results = evaluator.run()
        expected_mrr = (
            1 + (1 / 2) + 1
        ) / 3  # q1 first at 1, q2 first at 2, q3 first at 1
        self.assertAlmostEqual(results["mrr@5"], expected_mrr)

    def test_ndcg_metric(self):
        """Test NDCG@k calculations"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
            accuracy_at_k=[],
            precision_recall_at_k=[],
            mrr_at_k=[],
            ndcg_at_k=[5],
            map_at_k=[],
        )

        results = evaluator.run()
        # NDCG@5 calculation:
        # q1: (1/log2(2) + 1/log2(4) + 1/log2(6)) / (1/log2(2) + 1/log2(3) + 1/log2(4))
        # q2: (1/log2(3) + 1/log2(5)) / (1/log2(2) + 1/log2(3))
        # q3: (1/log2(2)) / (1/log2(2))
        # Average of all three queries
        expected_ndcg = 0.8455  # Approximate value
        self.assertAlmostEqual(results["ndcg@5"], expected_ndcg, places=4)

    def test_map_metric(self):
        """Test MAP@k calculations"""
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
            accuracy_at_k=[],
            precision_recall_at_k=[],
            mrr_at_k=[],
            ndcg_at_k=[],
            map_at_k=[5],
        )

        results = evaluator.run()
        # MAP@5 calculation:
        # q1: (1/1 + 2/3 + 3/5) / 3 -> (1 + 0.6667 + 0.6) / 3 -> 2.2667 / 3 -> 0.7556
        # q2: (1/2 + 2/4) / 2 -> (0.5 + 0.5) / 2 -> 1 / 2 -> 0.5
        # q3: (1/1) / 1 -> 1 / 1 -> 1
        # Average of all three queries: (0.7556 + 0.5 + 1) / 3 -> 2.2556 / 3 -> 0.7519
        expected_map = 0.7519  # Approximate value
        self.assertAlmostEqual(results["map@5"], expected_map, places=4)

    def test_graded_ndcg_metric(self):
        """Test graded NDCG@k calculations"""
        queries = {"535": "06 bmw 325i radio oem not navigation system"}
        relevant_docs = {
            "535": {
                "B08VSJGP1N": 0.01,
                "B08VJ66CNL": 0.01,
                "B08SHMLP5S": 0.0,
                "B08QGZMCYQ": 0.0,
                "B08PB9TTKT": 1.0,
                "B08NVQ8MZX": 0.01,
                "B084TV3C1B": 0.01,
                "B0742BZXC2": 1.0,
                "B00DHUA9VA": 0.0,
                "B00B4PJC9K": 0.0,
                "B0072LFB68": 0.01,
                "B0051GN8JI": 0.01,
                "B000J1HDWI": 0.0,
                "B0007KPS3C": 0.0,
                "B01M0SFMIH": 1.0,
                "B0007KPRIS": 0.0,
            }
        }
        # B08PB9TTKT 1 0.463
        # B00B4PJC9K 2 0.431
        # B0051GN8JI 3 0.419
        # B084TV3C1B 4 0.417
        # B08NVQ8MZX 5 0.41
        # B00DHUA9VA 6 0.415
        # B08SHMLP5S 7 0.415
        # B08VSJGP1N 8 0.41
        # B08QGZMCYQ 9 0.411
        # B0007KPRIS 10 0.40
        # B08VJ66CNL 11 0.40
        # B000J1HDWI 12 0.40
        # B0007KPS3C 13 0.39
        # B0072LFB68 14 0.39
        # B01M0SFMIH 15 0.39
        # B0742BZXC2 16 0.37

        # Mock Vespa responses - must match doc_ids in relevant_docs
        q1_response = MockVespaResponse(
            [
                {"id": "B08PB9TTKT", "relevance": 0.463},
                {"id": "B00B4PJC9K", "relevance": 0.431},
                {"id": "B0051GN8JI", "relevance": 0.419},
                {"id": "B084TV3C1B", "relevance": 0.417},
                {"id": "B08NVQ8MZX", "relevance": 0.41},
                {"id": "B00DHUA9VA", "relevance": 0.415},
                {"id": "B08SHMLP5S", "relevance": 0.415},
                {"id": "B08VSJGP1N", "relevance": 0.41},
                {"id": "B08QGZMCYQ", "relevance": 0.411},
                {"id": "B0007KPRIS", "relevance": 0.40},
                {"id": "B08VJ66CNL", "relevance": 0.40},
                {"id": "B000J1HDWI", "relevance": 0.40},
                {"id": "B0007KPS3C", "relevance": 0.39},
                {"id": "B0072LFB68", "relevance": 0.39},
                {"id": "B01M0SFMIH", "relevance": 0.39},
                {"id": "B0742BZXC2", "relevance": 0.37},
            ]
        )

        class MockVespaApp:
            def __init__(self, mock_responses):
                self.mock_responses = mock_responses
                self.current_query = 0

            def query_many(self, queries):
                return self.mock_responses

        mock_app = MockVespaApp([q1_response])

        def mock_vespa_query_fn(query_text: str, top_k: int) -> dict:
            return {
                "yql": f'select * from sources * where text contains "{query_text}";',
                "hits": top_k,
            }

        evaluator = VespaEvaluator(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=mock_vespa_query_fn,
            app=mock_app,
            ndcg_at_k=[16],
        )

        results = evaluator.run()
        print(results)
        self.assertAlmostEqual(results["ndcg@16"], 0.7046, places=4)

    def test_vespa_query_fn_validation(self):
        """Test validation of vespa_query_fn with valid functions"""

        # Valid function with type hints
        def fn1(query: str, k: int) -> dict:
            return {"yql": query, "hits": k}

        # Valid function without type hints
        def fn2(query, k):
            return {"yql": query, "hits": k}

        # Valid function with default args
        def fn3(query: str, k: int = 10) -> dict:
            return {"yql": query, "hits": k}

        # All should work without raising exceptions
        for fn in [fn1, fn2, fn3]:
            evaluator = VespaEvaluator(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=fn,
                app=self.mock_app,
            )
            self.assertIsInstance(evaluator, VespaEvaluator)

    def test_vespa_query_fn_validation_errors(self):
        """Test validation of vespa_query_fn with invalid functions"""

        # Not a callable
        with self.assertRaisesRegex(ValueError, "must be callable"):
            VespaEvaluator(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn="not_a_function",
                app=self.mock_app,
            )

        # Wrong number of params
        def fn1(query: str) -> dict:
            return {"yql": query}

        with self.assertRaisesRegex(TypeError, "must take 2 or 3 parameters"):
            VespaEvaluator(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=fn1,
                app=self.mock_app,
            )

        # Wrong param types
        def fn2(query: int, k: str) -> dict:
            return {"yql": str(query), "hits": int(k)}

        with self.assertRaisesRegex(TypeError, "must be of type"):
            VespaEvaluator(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=fn2,
                app=self.mock_app,
            )

        # No type hints
        def fn3(query, k):
            return {"yql": query, "hits": k}

    def test_validate_qrels(self):
        """Test validation of qrels with valid qrels"""
        # Valid qrels
        qrels1 = {
            "q1": {"doc1", "doc2", "doc3"},
            "q2": {"doc4", "doc5"},
            "q3": {"doc6"},
        }
        qrels2 = {
            "q1": "doc1",
            "q2": "doc4",
            "q3": "doc6",
        }
        qrels3 = {
            "q1": {"doc1": 1.0, "doc2": 0.5, "doc3": 0.2},
            "q2": {"doc4": 0.8, "doc5": 0.6},
            "q3": {"doc6": 1.0},
        }

        # All should work without raising exceptions
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=qrels1,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )
        self.assertIsInstance(evaluator, VespaEvaluator)

        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=qrels2,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )
        self.assertIsInstance(evaluator, VespaEvaluator)

        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=qrels3,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )
        self.assertIsInstance(evaluator, VespaEvaluator)

    def test_validate_qrels_errors(self):
        """Test validation of qrels with invalid qrels"""

        # Not a dict
        with self.assertRaisesRegex(ValueError, "qrels must be a dict"):
            VespaEvaluator(
                queries=self.queries,
                relevant_docs="not_a_dict",
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
            )

        # Relevant docs not a set, string, or dict
        with self.assertRaisesRegex(ValueError, "must be a set, string, or dict"):
            VespaEvaluator(
                queries=self.queries,
                relevant_docs={"q1": 1},
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
            )

        # Relevance scores not numeric
        with self.assertRaisesRegex(
            ValueError, "must be a dict of string doc_id => numeric relevance"
        ):
            VespaEvaluator(
                queries=self.queries,
                relevant_docs={"q1": {"doc1": "not_numeric"}},
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
            )

        # Relevance scores not between 0 and 1
        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            VespaEvaluator(
                queries=self.queries,
                relevant_docs={"q1": {"doc1": 1.1}},
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
            )

        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            VespaEvaluator(
                queries=self.queries,
                relevant_docs={"q1": {"doc1": -0.1}},
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
            )

    def test_filter_queries(self):
        """Test filter_queries method"""
        queries = {
            "q1": "what is machine learning",
            "q2": "how to code python",
            "q3": "what is the capital of France",
            "q4": "irrelevant query",
        }

        relevant_docs = {
            "q1": {"doc1", "doc2", "doc3"},
            "q2": {"doc4", "doc5"},
            "q3": {"doc6"},
        }

        evaluator = VespaEvaluator(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )

        # Test that queries with no relevant docs are filtered out
        self.assertEqual(len(evaluator.queries_ids), 3)
        self.assertNotIn("q4", evaluator.queries_ids)

        # Test that queries with empty relevant docs are filtered out
        relevant_docs["q4"] = set()
        evaluator = VespaEvaluator(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )
        self.assertEqual(len(evaluator.queries_ids), 3)
        self.assertNotIn("q4", evaluator.queries_ids)

        # Test that queries with relevant docs are not filtered out
        relevant_docs["q4"] = {"doc7"}
        evaluator = VespaEvaluator(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )
        self.assertEqual(len(evaluator.queries_ids), 4)
        self.assertIn("q4", evaluator.queries_ids)

    def test_vespa_query_fn_with_query_id(self):
        """Test that vespa_query_fn accepting query_id receives it as the third argument."""

        def fn(query_text: str, top_k: int, query_id: str) -> dict:
            return {
                "yql": f'select * from sources * where text contains "{query_text}" and id="{query_id}";',
                "hits": top_k,
                "query_id": query_id,  # Not for passing to Vespa, but for testing
            }

        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=fn,
            app=self.mock_app,
        )
        self.assertTrue(evaluator._vespa_query_fn_takes_query_id)
        # Build query bodies and check that query_id is passed correctly.
        query_bodies = []
        max_k = evaluator._find_max_k()
        for qid, query_text in zip(evaluator.queries_ids, evaluator.queries):
            query_body = evaluator.vespa_query_fn(query_text, max_k, qid)
            query_bodies.append(query_body)

        for qid, qb in zip(evaluator.queries_ids, query_bodies):
            self.assertIn("query_id", qb)
            self.assertEqual(qb["query_id"], qid)

    def test_vespa_query_fn_without_query_id(self):
        """Test that a vespa_query_fn accepting only 2 parameters does not receive a query_id."""

        def fn(query_text: str, top_k: int) -> dict:
            # Return a basic query body.
            return {"yql": query_text, "hits": top_k}

        # Create a dummy response (the content is not used for these tests).
        dummy_response = MockVespaResponse([{"id": "doc1", "relevance": 1.0}])
        capturing_app = QueryBodyCapturingApp([dummy_response] * len(self.queries))

        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=fn,
            app=capturing_app,
        )
        # Since fn accepts only 2 params, the evaluator should mark it as NOT taking a query_id.
        self.assertFalse(evaluator._vespa_query_fn_takes_query_id)

        # Run the evaluator to trigger query body generation.
        evaluator.run()

        # Verify that none of the query bodies include a "query_id" key and that default_body keys were added.
        for qb in capturing_app.captured_query_bodies:
            self.assertNotIn("query_id", qb)
            self.assertIn("timeout", qb)
            self.assertEqual(qb["timeout"], "5s")
            self.assertIn("presentation.timing", qb)
            self.assertEqual(qb["presentation.timing"], True)

    def test_vespa_query_fn_no_type_hints(self):
        """Test that a vespa_query_fn without type hints is handled correctly."""

        def fn(query_text, top_k):
            # Return a basic query body.
            return {"yql": query_text, "hits": top_k}

        # Create a dummy response (the content is not used for these tests).
        dummy_response = MockVespaResponse([{"id": "doc1", "relevance": 1.0}])
        capturing_app = QueryBodyCapturingApp([dummy_response] * len(self.queries))

        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=fn,
            app=capturing_app,
        )

        # Run the evaluator to trigger query body generation.
        evaluator.run()

        # Verify that none of the query bodies include a "query_id" key and that default_body keys were added.
        for qb in capturing_app.captured_query_bodies:
            self.assertNotIn("query_id", qb)
            self.assertIn("timeout", qb)
            self.assertEqual(qb["timeout"], "5s")
            self.assertIn("presentation.timing", qb)
            self.assertEqual(qb["presentation.timing"], True)

    def test_vespa_query_fn_preserves_extra_keys(self):
        """Test that extra keys returned by vespa_query_fn are preserved after merging with default_body."""

        def fn_extra(query_text: str, top_k: int) -> dict:
            # Return a query body that includes an extra key.
            return {"yql": query_text, "hits": top_k, "extra": "value"}

        dummy_response = MockVespaResponse([{"id": "doc1", "relevance": 1.0}])
        capturing_app = QueryBodyCapturingApp([dummy_response] * len(self.queries))

        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=fn_extra,
            app=capturing_app,
        )
        evaluator.run()

        # Verify that the extra key is still present in each query body.
        for qb in capturing_app.captured_query_bodies:
            self.assertIn("extra", qb)
            self.assertEqual(qb["extra"], "value")

    def test_vespa_query_fn_respects_user_params(self):
        """Test that user-provided parameters in vespa_query_fn are not overridden by default_body."""

        def fn_with_timeout(query_text: str, top_k: int) -> dict:
            # Return a query body with a custom timeout
            return {
                "yql": query_text,
                "hits": top_k,
                "timeout": "10s",  # Different from default "5s"
            }

        dummy_response = MockVespaResponse([{"id": "doc1", "relevance": 1.0}])
        capturing_app = QueryBodyCapturingApp([dummy_response] * len(self.queries))

        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=fn_with_timeout,
            app=capturing_app,
        )
        evaluator.run()

        # After evaluator.run(), the user-provided timeout should be preserved
        for qb in capturing_app.captured_query_bodies:
            self.assertEqual(qb["timeout"], "10s")  # User's value preserved
            self.assertEqual(qb["presentation.timing"], True)  # Default added


class MockAppForMatchEvaluator:
    """Mock Vespa app for VespaMatchEvaluator tests."""

    def __init__(
        self,
        limit_responses: List[MockVespaResponse],
        recall_responses: List[MockVespaResponse],
    ):
        self.limit_responses = limit_responses
        self.recall_responses = recall_responses
        self.call_count = 0
        self.captured_query_bodies_limit: List[Dict] = []
        self.captured_query_bodies_recall: List[Dict] = []

    def query_many(self, query_bodies: List[Dict]):
        if self.call_count == 0:  # First call is for limit queries
            self.call_count += 1
            self.captured_query_bodies_limit = query_bodies
            return self.limit_responses[: len(query_bodies)]
        elif self.call_count == 1:  # Second call is for recall queries
            self.call_count += 1
            self.captured_query_bodies_recall = query_bodies
            return self.recall_responses[: len(query_bodies)]
        raise AssertionError(f"query_many called too many times: {self.call_count}")


class TestVespaMatchEvaluator(unittest.TestCase):
    def setUp(self):
        self.queries = {
            "q1": "query text one",
            "q2": "query text two",
            "q3": "query text three",  # Will be filtered if not in relevant_docs
        }
        self.relevant_docs_set = {
            "q1": {"doc1", "doc2"},
            "q2": {"doc4"},
        }

        def mock_vespa_query_fn(
            query_text: str, top_k: int, query_id: Optional[str] = None
        ) -> dict:
            yql = f'select * from sources * where userInput("{query_text}");'
            return {"yql": yql, "hits": top_k}

        self.vespa_query_fn = mock_vespa_query_fn

        # Default mock app for 2 queries (q1, q2)
        self.mock_app = MockAppForMatchEvaluator(
            limit_responses=[
                MockVespaResponse(
                    hits=[], _total_count=10, _timing={"searchtime": 0.01}
                ),  # q1
                MockVespaResponse(
                    hits=[], _total_count=5, _timing={"searchtime": 0.02}
                ),  # q2
            ],
            recall_responses=[
                MockVespaResponse(
                    hits=[{"id": "doc1"}, {"id": "doc_other"}],
                    _timing={"searchtime": 0.03},
                ),  # q1 (1 of 2 relevant)
                MockVespaResponse(
                    hits=[{"id": "doc4"}], _timing={"searchtime": 0.04}
                ),  # q2 (1 of 1 relevant)
            ],
        )

    def test_basic_initialization(self):
        evaluator = VespaMatchEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs_set,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
        )
        self.assertIsInstance(evaluator, VespaMatchEvaluator)
        self.assertEqual(evaluator.name, "")
        self.assertEqual(evaluator.id_field, "")
        self.assertEqual(evaluator.write_csv, False)
        self.assertEqual(evaluator.write_verbose, False)
        self.assertEqual(set(evaluator.queries_ids), {"q1", "q2"})  # q3 filtered

    def test_run_basic_scenario(self):
        evaluator = VespaMatchEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs_set,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,
            name="basic_match_run",
        )
        results = evaluator.run()

        # Metrics
        # q1: 1 matched / 2 relevant. q2: 1 matched / 1 relevant.
        # total_relevant_docs = 2 + 1 = 3
        # total_matched_relevant = 1 + 1 = 2
        # match_recall = 2 / 3
        # avg_recall_per_query = (0.5 + 1.0) / 2 = 0.75
        # avg_matched_per_query (totalCount from limit) = (10 + 5) / 2 = 7.5
        self.assertAlmostEqual(results["match_recall"], 2 / 3)
        self.assertAlmostEqual(results["avg_recall_per_query"], 0.75)
        self.assertEqual(results["total_relevant_docs"], 3)
        self.assertEqual(results["total_matched_relevant"], 2)
        self.assertAlmostEqual(results["avg_matched_per_query"], 7.5)
        self.assertEqual(results["total_queries"], 2)
        self.assertEqual(evaluator.primary_metric, "match_recall")
        self.assertAlmostEqual(
            results["searchtime_avg"], (0.01 + 0.02 + 0.03 + 0.04) / 4
        )

        # Check captured query bodies
        self.assertEqual(len(self.mock_app.captured_query_bodies_limit), 2)
        q1_limit_body = self.mock_app.captured_query_bodies_limit[0]
        self.assertIn("limit 0", q1_limit_body["yql"])
        self.assertNotIn("hits", q1_limit_body)  # 'hits' key popped for limit query

        self.assertEqual(len(self.mock_app.captured_query_bodies_recall), 2)
        q1_recall_body = self.mock_app.captured_query_bodies_recall[0]
        self.assertIn("recall", q1_recall_body)
        self.assertTrue(
            "id:doc1" in q1_recall_body["recall"]
            and "id:doc2" in q1_recall_body["recall"]
        )
        self.assertEqual(q1_recall_body["hits"], len(self.relevant_docs_set["q1"]))
        self.assertEqual(q1_recall_body["ranking"], "unranked")  # Default

    def test_run_all_relevant_docs_matched(self):
        app = MockAppForMatchEvaluator(
            limit_responses=[MockVespaResponse(hits=[], _total_count=3)],
            recall_responses=[MockVespaResponse(hits=[{"id": "doc1"}, {"id": "doc2"}])],
        )
        evaluator = VespaMatchEvaluator(
            queries={"q1": "q"},
            relevant_docs={"q1": {"doc1", "doc2"}},
            vespa_query_fn=self.vespa_query_fn,
            app=app,
        )
        results = evaluator.run()
        self.assertAlmostEqual(results["match_recall"], 1.0)
        self.assertAlmostEqual(results["avg_recall_per_query"], 1.0)
        self.assertEqual(results["total_matched_relevant"], 2)

    def test_run_no_relevant_docs_matched(self):
        app = MockAppForMatchEvaluator(
            limit_responses=[MockVespaResponse(hits=[], _total_count=3)],
            recall_responses=[MockVespaResponse(hits=[{"id": "other_doc"}])],
        )
        evaluator = VespaMatchEvaluator(
            queries={"q1": "q"},
            relevant_docs={"q1": {"doc1", "doc2"}},
            vespa_query_fn=self.vespa_query_fn,
            app=app,
        )
        results = evaluator.run()
        self.assertAlmostEqual(results["match_recall"], 0.0)
        self.assertAlmostEqual(results["avg_recall_per_query"], 0.0)
        self.assertEqual(results["total_matched_relevant"], 0)

    def test_error_on_graded_relevance(self):
        graded_relevant_docs = {"q1": {"doc1": 1.0}}  # Graded relevance
        evaluator = VespaMatchEvaluator(
            queries=self.queries,
            relevant_docs=graded_relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=self.mock_app,  # App setup doesn't matter as error is before query
        )
        with self.assertRaisesRegex(
            ValueError, "Graded relevance is not supported in VespaMatchEvaluator"
        ):
            evaluator.run()

    def test_id_field_usage(self):
        # Hits should contain the id_field in 'fields'
        app = MockAppForMatchEvaluator(
            limit_responses=[MockVespaResponse(hits=[], _total_count=1)],
            recall_responses=[
                MockVespaResponse(
                    hits=[{"fields": {"custom_doc_id": "doc123"}, "relevance": 1.0}]
                )
            ],
        )
        evaluator = VespaMatchEvaluator(
            queries={"q1": "q"},
            relevant_docs={"q1": {"doc123"}},
            vespa_query_fn=self.vespa_query_fn,
            app=app,
            id_field="custom_doc_id",
        )
        results = evaluator.run()
        self.assertAlmostEqual(
            results["match_recall"], 1.0
        )  # doc123 should be extracted and matched

    def test_vespa_query_failure(self):
        app = MockAppForMatchEvaluator(
            limit_responses=[
                MockVespaResponse(hits=[], _status_code=500, _total_count=0)
            ],  # Error on limit query
            recall_responses=[],
        )
        evaluator = VespaMatchEvaluator(
            queries={"q1": "q"},
            relevant_docs={"q1": {"d1"}},
            vespa_query_fn=self.vespa_query_fn,
            app=app,
        )
        with self.assertRaisesRegex(
            ValueError, "Vespa query failed with status code 500"
        ):
            evaluator.run()

        app_recall_fail = MockAppForMatchEvaluator(
            limit_responses=[MockVespaResponse(hits=[], _total_count=1)],
            recall_responses=[
                MockVespaResponse(hits=[], _status_code=503)
            ],  # Error on recall query
        )
        evaluator_recall_fail = VespaMatchEvaluator(
            queries={"q1": "q"},
            relevant_docs={"q1": {"d1"}},
            vespa_query_fn=self.vespa_query_fn,
            app=app_recall_fail,
        )
        with self.assertRaisesRegex(
            ValueError, "Vespa query failed with status code 503"
        ):
            evaluator_recall_fail.run()

    def test_custom_rank_profile_in_query_fn(self):
        def query_fn_custom_ranking(
            query_text: str, top_k: int, query_id: Optional[str] = None
        ) -> dict:
            return {
                "yql": f"select * from sources * where userInput('{query_text}');",
                "hits": top_k,
                "ranking": "my_custom_profile",  # Explicitly set
            }

        app = MockAppForMatchEvaluator(
            limit_responses=[MockVespaResponse(hits=[], _total_count=1)],
            recall_responses=[MockVespaResponse(hits=[{"id": "doc1"}])],
        )
        evaluator = VespaMatchEvaluator(
            queries={"q1": "q"},
            relevant_docs={"q1": {"doc1"}},
            vespa_query_fn=query_fn_custom_ranking,
            app=app,
        )
        evaluator.run()
        # Recall query should use the ranking from the function
        self.assertEqual(
            app.captured_query_bodies_recall[0]["ranking"], "my_custom_profile"
        )
        # Limit query body also contains it, though it might not be used by Vespa for limit 0
        self.assertEqual(
            app.captured_query_bodies_limit[0]["ranking"], "my_custom_profile"
        )


class TestUtilityFunctions(unittest.TestCase):
    """Test the module-level utility functions extracted from VespaEvaluatorBase."""

    def test_validate_queries_valid_inputs(self):
        """Test validate_queries with valid inputs."""
        from vespa.evaluation import validate_queries

        # String keys
        queries1 = {"q1": "query text 1", "q2": "query text 2"}
        result1 = validate_queries(queries1)
        self.assertEqual(result1, {"q1": "query text 1", "q2": "query text 2"})

        # Integer keys (should be converted to strings)
        queries2 = {1: "query text 1", 2: "query text 2"}
        result2 = validate_queries(queries2)
        self.assertEqual(result2, {"1": "query text 1", "2": "query text 2"})

        # Mixed keys
        queries3 = {"q1": "query text 1", 2: "query text 2"}
        result3 = validate_queries(queries3)
        self.assertEqual(result3, {"q1": "query text 1", "2": "query text 2"})

    def test_validate_queries_invalid_inputs(self):
        """Test validate_queries with invalid inputs."""
        from vespa.evaluation import validate_queries

        # Not a dict
        with self.assertRaisesRegex(ValueError, "queries must be a dict"):
            validate_queries("not a dict")

        # Invalid query ID type
        with self.assertRaisesRegex(ValueError, "Query ID must be a string or an int"):
            validate_queries({None: "query text"})

        # Invalid query text type
        with self.assertRaisesRegex(ValueError, "Query text must be a string"):
            validate_queries({"q1": 123})

    def test_validate_qrels_valid_inputs(self):
        """Test validate_qrels with valid inputs."""
        from vespa.evaluation import validate_qrels

        # Set of relevant docs
        qrels1 = {"q1": {"doc1", "doc2"}, "q2": {"doc3"}}
        result1 = validate_qrels(qrels1)
        self.assertEqual(result1, {"q1": {"doc1", "doc2"}, "q2": {"doc3"}})

        # Single relevant doc (string)
        qrels2 = {"q1": "doc1", "q2": "doc2"}
        result2 = validate_qrels(qrels2)
        self.assertEqual(result2, {"q1": {"doc1"}, "q2": {"doc2"}})

        # Graded relevance (dict)
        qrels3 = {"q1": {"doc1": 1.0, "doc2": 0.5}, "q2": {"doc3": 0.8}}
        result3 = validate_qrels(qrels3)
        self.assertEqual(
            result3, {"q1": {"doc1": 1.0, "doc2": 0.5}, "q2": {"doc3": 0.8}}
        )

        # Integer query IDs (should be converted to strings)
        qrels4 = {1: {"doc1"}, 2: {"doc2"}}
        result4 = validate_qrels(qrels4)
        self.assertEqual(result4, {"1": {"doc1"}, "2": {"doc2"}})

    def test_validate_qrels_invalid_inputs(self):
        """Test validate_qrels with invalid inputs."""
        from vespa.evaluation import validate_qrels

        # Not a dict
        with self.assertRaisesRegex(ValueError, "qrels must be a dict"):
            validate_qrels("not a dict")

        # Invalid query ID type
        with self.assertRaisesRegex(
            ValueError, "Query ID in qrels must be a string or an int"
        ):
            validate_qrels({None: {"doc1"}})

        # Invalid relevant docs type
        with self.assertRaisesRegex(ValueError, "must be a set, string, or dict"):
            validate_qrels({"q1": 123})

        # Invalid relevance score type
        with self.assertRaisesRegex(
            ValueError, "must be a dict of string doc_id => numeric relevance"
        ):
            validate_qrels({"q1": {"doc1": "not_numeric"}})

        # Relevance score out of range
        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            validate_qrels({"q1": {"doc1": 1.5}})

        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            validate_qrels({"q1": {"doc1": -0.1}})

    def test_validate_vespa_query_fn_valid_functions(self):
        """Test validate_vespa_query_fn with valid functions."""
        from vespa.evaluation import validate_vespa_query_fn

        # Function with 2 parameters
        def fn1(query: str, k: int) -> dict:
            return {"yql": query, "hits": k}

        result1 = validate_vespa_query_fn(fn1)
        self.assertFalse(result1)  # Should return False (no query_id param)

        # Function with 3 parameters
        def fn2(query: str, k: int, query_id: Optional[str]) -> dict:
            return {"yql": query, "hits": k}

        result2 = validate_vespa_query_fn(fn2)
        self.assertTrue(result2)  # Should return True (has query_id param)

        # Function without type hints
        def fn3(query, k):
            return {"yql": query, "hits": k}

        result3 = validate_vespa_query_fn(fn3)
        self.assertFalse(result3)

    def test_validate_vespa_query_fn_invalid_functions(self):
        """Test validate_vespa_query_fn with invalid functions."""
        from vespa.evaluation import validate_vespa_query_fn

        # Not callable
        with self.assertRaisesRegex(ValueError, "must be callable"):
            validate_vespa_query_fn("not a function")

        # Wrong number of parameters
        def fn1(query: str) -> dict:
            return {"yql": query}

        with self.assertRaisesRegex(TypeError, "must take 2 or 3 parameters"):
            validate_vespa_query_fn(fn1)

        # Wrong parameter types
        def fn2(query: int, k: str) -> dict:
            return {"yql": str(query), "hits": int(k)}

        with self.assertRaisesRegex(TypeError, "must be of type"):
            validate_vespa_query_fn(fn2)

    def test_filter_queries(self):
        """Test filter_queries function."""
        from vespa.evaluation import filter_queries

        queries = {"q1": "query 1", "q2": "query 2", "q3": "query 3", "q4": "query 4"}

        # Normal case
        relevant_docs = {"q1": {"doc1"}, "q2": {"doc2"}, "q3": {"doc3"}}
        result = filter_queries(queries, relevant_docs)
        self.assertEqual(set(result), {"q1", "q2", "q3"})

        # Some queries have no relevant docs
        relevant_docs2 = {"q1": {"doc1"}, "q3": {"doc3"}}
        result2 = filter_queries(queries, relevant_docs2)
        self.assertEqual(set(result2), {"q1", "q3"})

        # Some queries have empty relevant docs
        relevant_docs3 = {"q1": {"doc1"}, "q2": set(), "q3": {"doc3"}}
        result3 = filter_queries(queries, relevant_docs3)
        self.assertEqual(set(result3), {"q1", "q3"})

    def test_extract_doc_id_from_hit(self):
        """Test extract_doc_id_from_hit function."""
        from vespa.evaluation import extract_doc_id_from_hit

        # No id_field specified, use default logic
        hit1 = {"id": "id:namespace:doctype::doc123", "relevance": 0.9}
        result1 = extract_doc_id_from_hit(hit1, "")
        self.assertEqual(result1, "doc123")

        # No namespace in id, fallback to fields.id
        hit2 = {"id": "simple_id", "fields": {"id": "doc456"}, "relevance": 0.8}
        result2 = extract_doc_id_from_hit(hit2, "")
        self.assertEqual(result2, "doc456")

        # Custom id_field
        hit3 = {"fields": {"custom_id": "doc789"}, "relevance": 0.7}
        result3 = extract_doc_id_from_hit(hit3, "custom_id")
        self.assertEqual(result3, "doc789")

        # Error case: no extractable doc_id
        hit4 = {"relevance": 0.6}
        with self.assertRaisesRegex(ValueError, "Could not extract doc_id from hit"):
            extract_doc_id_from_hit(hit4, "")

    def test_calculate_searchtime_stats(self):
        """Test calculate_searchtime_stats function."""
        from vespa.evaluation import calculate_searchtime_stats

        # Normal case
        searchtimes = [0.1, 0.2, 0.15, 0.3, 0.25]
        result = calculate_searchtime_stats(searchtimes)

        self.assertAlmostEqual(result["searchtime_avg"], 0.2)
        self.assertAlmostEqual(result["searchtime_q50"], 0.2)  # median
        self.assertIn("searchtime_q90", result)
        self.assertIn("searchtime_q95", result)

        # Empty list
        result_empty = calculate_searchtime_stats([])
        self.assertEqual(result_empty, {})

    def test_execute_queries_success(self):
        """Test execute_queries function with successful responses."""
        from vespa.evaluation import execute_queries

        # Mock responses
        mock_responses = [
            MockVespaResponse(hits=[], _timing={"searchtime": 0.1}),
            MockVespaResponse(hits=[], _timing={"searchtime": 0.2}),
        ]

        # Mock app
        class MockApp:
            def query_many(self, query_bodies):
                return mock_responses

        app = MockApp()
        query_bodies = [{"yql": "query1"}, {"yql": "query2"}]

        responses, searchtimes = execute_queries(app, query_bodies)

        self.assertEqual(len(responses), 2)
        self.assertEqual(searchtimes, [0.1, 0.2])

    def test_execute_queries_failure(self):
        """Test execute_queries function with failed responses."""
        from vespa.evaluation import execute_queries

        # Mock failed response
        mock_response = MockVespaResponse(hits=[], _status_code=500)

        class MockApp:
            def query_many(self, query_bodies):
                return [mock_response]

        app = MockApp()
        query_bodies = [{"yql": "query1"}]

        with self.assertRaisesRegex(
            ValueError, "Vespa query failed with status code 500"
        ):
            execute_queries(app, query_bodies)

    def test_write_csv(self):
        """Test write_csv function."""
        from vespa.evaluation import write_csv
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = {"accuracy@1": 0.75, "precision@5": 0.6}
            searchtime_stats = {"searchtime_avg": 0.1}
            csv_file = "test_results.csv"
            name = "test_run"

            # Write CSV
            write_csv(metrics, searchtime_stats, csv_file, temp_dir, name)

            # Check file was created
            csv_path = os.path.join(temp_dir, csv_file)
            self.assertTrue(os.path.exists(csv_path))

            # Check content
            with open(csv_path, "r") as f:
                content = f.read()
                self.assertIn("name", content)
                self.assertIn("accuracy@1", content)
                self.assertIn("test_run", content)
                self.assertIn("0.75", content)

    def test_log_metrics(self):
        """Test log_metrics function."""
        from vespa.evaluation import log_metrics
        import logging
        from io import StringIO

        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("vespa.evaluation")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            metrics = {
                "accuracy@1": 0.75,
                "precision@5": 0.6,
                "ndcg@10": 0.8,
                "searchtime_avg": 0.123,
            }

            log_metrics("test_run", metrics)

            log_output = log_stream.getvalue()

            # Check that percentage metrics are formatted correctly
            self.assertIn("75.00%", log_output)  # accuracy
            self.assertIn("60.00%", log_output)  # precision

            # Check that non-percentage metrics are formatted correctly
            self.assertIn("0.8000", log_output)  # ndcg
            self.assertIn("0.1230", log_output)  # searchtime

        finally:
            logger.removeHandler(handler)

    def test_mean_function(self):
        """Test the mean utility function."""
        from vespa.evaluation import mean

        # Normal case
        self.assertAlmostEqual(mean([1, 2, 3, 4, 5]), 3.0)

        # Single value
        self.assertAlmostEqual(mean([5]), 5.0)

        # Empty list
        self.assertAlmostEqual(mean([]), 0.0)

        # Floating point values
        self.assertAlmostEqual(mean([0.1, 0.2, 0.3]), 0.2)

    def test_percentile_function(self):
        """Test the percentile utility function."""
        from vespa.evaluation import percentile

        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Test various percentiles
        self.assertAlmostEqual(percentile(values, 0), 1.0)
        self.assertAlmostEqual(percentile(values, 50), 5.5)
        self.assertAlmostEqual(percentile(values, 100), 10.0)

        # Test edge cases
        self.assertAlmostEqual(percentile([], 50), 0.0)
        self.assertAlmostEqual(percentile([5], 50), 5.0)

        # Test out of range percentiles
        self.assertAlmostEqual(percentile(values, -10), percentile(values, 0))
        self.assertAlmostEqual(percentile(values, 110), percentile(values, 100))


class MockAppForDataCollector:
    """Mock Vespa app for VespaCollectorBase tests."""

    def __init__(self, responses: List[MockVespaResponse]):
        self.responses = responses
        self.captured_query_bodies: List[Dict] = []

    def query_many(self, query_bodies: List[Dict]):
        self.captured_query_bodies = query_bodies
        # Convert MockVespaResponse to objects that behave like VespaQueryResponse
        mock_responses = []
        for mock_resp in self.responses[: len(query_bodies)]:
            # Create a response object that has the same interface as VespaQueryResponse
            mock_response = type(
                "MockResponse",
                (),
                {
                    "hits": mock_resp.hits,
                    "status_code": mock_resp.status_code,
                    "get_json": mock_resp.get_json,
                },
            )()
            mock_responses.append(mock_response)
        return mock_responses


class TestVespaCollectorBase(unittest.TestCase):
    """Test the abstract VespaCollectorBase base class."""

    def setUp(self):
        self.queries = {
            "q1": "what is machine learning",
            "q2": "how to code python",
            "q3": "what is the capital of France",
        }

        self.relevant_docs = {
            "q1": {"doc1", "doc2"},
            "q2": {"doc4"},
            "q3": {"doc6"},
        }

        def mock_vespa_query_fn(query_text: str, top_k: int) -> dict:
            return {
                "yql": f'select * from sources * where text contains "{query_text}";',
                "hits": top_k,
            }

        self.vespa_query_fn = mock_vespa_query_fn
        self.mock_app = MockAppForDataCollector([])

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that VespaCollectorBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            VespaCollectorBase(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
            )

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement collect method."""

        class IncompleteCollector(VespaCollectorBase):
            pass  # Missing collect method

        with self.assertRaises(TypeError):
            IncompleteCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
            )


class TestVespaFeatureCollector(unittest.TestCase):
    """Test the concrete VespaFeatureCollector implementation."""

    def setUp(self):
        self.queries = {
            "q1": "what is machine learning",
            "q2": "how to code python",
            "q3": "what is the capital of France",
        }

        self.relevant_docs = {
            "q1": {"doc1", "doc2"},
            "q2": {"doc4"},
            "q3": {"doc6"},
        }

        # Mock responses with some relevant and non-relevant documents
        self.mock_responses = [
            # Response for q1
            MockVespaResponse(
                [
                    {"id": "doc1", "relevance": 0.9},  # Relevant
                    {"id": "doc10", "relevance": 0.8},  # Not relevant
                    {"id": "doc2", "relevance": 0.7},  # Relevant
                    {"id": "doc11", "relevance": 0.6},  # Not relevant
                ]
            ),
            # Response for q2
            MockVespaResponse(
                [
                    {"id": "doc12", "relevance": 0.95},  # Not relevant
                    {"id": "doc4", "relevance": 0.85},  # Relevant
                    {"id": "doc13", "relevance": 0.75},  # Not relevant
                ]
            ),
            # Response for q3
            MockVespaResponse(
                [
                    {"id": "doc6", "relevance": 0.9},  # Relevant
                    {"id": "doc16", "relevance": 0.8},  # Not relevant
                ]
            ),
        ]

        self.mock_app = MockAppForDataCollector(self.mock_responses)

        def mock_vespa_query_fn(query_text: str, top_k: int) -> dict:
            return {
                "yql": f'select * from sources * where text contains "{query_text}";',
                "hits": top_k,
            }

        self.vespa_query_fn = mock_vespa_query_fn

    def test_basic_initialization(self):
        """Test basic initialization of VespaFeatureCollector."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                name="test_collector",
                csv_dir=temp_dir,
            )

            self.assertEqual(collector.name, "test_collector")
            self.assertEqual(collector.csv_dir, temp_dir)
            self.assertEqual(len(collector.queries_ids), 3)
            self.assertEqual(set(collector.queries_ids), {"q1", "q2", "q3"})

    def test_collect_creates_csv_file(self):
        """Test that collect() creates a CSV file with training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                name="test_run",
                csv_dir=temp_dir,
            )

            # Execute collection
            collector.collect()

            # Check that CSV file was created
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            self.assertEqual(len(csv_files), 1)

            csv_path = os.path.join(temp_dir, csv_files[0])
            self.assertTrue(os.path.exists(csv_path))

    def test_collect_csv_content_structure(self):
        """Test that the CSV file has the correct structure and content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                name="test_run",
                csv_dir=temp_dir,
            )

            collector.collect()

            # Read and verify CSV content
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            csv_path = os.path.join(temp_dir, csv_files[0])

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            # Check header
            self.assertEqual(
                header,
                [
                    "query_id",
                    "query_text",
                    "doc_id",
                    "relevance_label",
                    "relevance_score",
                ],
            )

            # Check that we have data for all queries
            query_ids_in_csv = set(row[0] for row in rows)
            self.assertEqual(query_ids_in_csv, {"q1", "q2", "q3"})

            # Check relevance labels are correct
            for row in rows:
                query_id, query_text, doc_id, relevance_label, relevance_score = row
                relevance_label_float = float(relevance_label)

                if doc_id in self.relevant_docs[query_id]:
                    self.assertEqual(relevance_label_float, 1.0)
                else:
                    self.assertEqual(relevance_label_float, 0.0)

                # relevance_score should be the Vespa relevance score from the hit
                self.assertIsInstance(float(relevance_score), float)

    def test_collect_with_single_relevant_doc(self):
        """Test collection with single relevant doc per query"""
        relevant_docs_single = {
            "q1": "doc1",
            "q2": "doc4",
            "q3": "doc6",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=relevant_docs_single,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                csv_dir=temp_dir,
            )

            collector.collect()

            # Verify CSV was created and has correct content
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            self.assertEqual(len(csv_files), 1)

    def test_collect_with_custom_id_field(self):
        """Test collection with custom id_field."""
        # Mock responses with custom id field
        mock_responses_custom = [
            MockVespaResponse(
                [
                    {"fields": {"custom_id": "doc1"}, "relevance": 0.9},
                    {"fields": {"custom_id": "doc10"}, "relevance": 0.8},
                ]
            ),
            MockVespaResponse(
                [
                    {"fields": {"custom_id": "doc4"}, "relevance": 0.85},
                ]
            ),
            MockVespaResponse(
                [
                    {"fields": {"custom_id": "doc6"}, "relevance": 0.9},
                ]
            ),
        ]

        mock_app = MockAppForDataCollector(mock_responses_custom)

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=mock_app,
                id_field="custom_id",
                csv_dir=temp_dir,
            )

            collector.collect()

            # Verify collection worked with custom id field
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            self.assertEqual(len(csv_files), 1)

    def test_collect_callable_interface(self):
        """Test that collector can be called as a function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                csv_dir=temp_dir,
            )

            # Should be callable
            collector()

            # Verify file was created
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            self.assertEqual(len(csv_files), 1)

    def test_collect_queries_without_relevant_docs_filtered(self):
        """Test that queries without relevant docs are filtered out."""
        queries_with_extra = {
            "q1": "what is machine learning",
            "q2": "how to code python",
            "q4": "query without relevant docs",  # This should be filtered out
        }

        relevant_docs_partial = {
            "q1": {"doc1"},
            "q2": {"doc4"},
            # q4 has no relevant docs
        }

        mock_responses_partial = [
            MockVespaResponse([{"id": "doc1", "relevance": 0.9}]),
            MockVespaResponse([{"id": "doc4", "relevance": 0.85}]),
        ]

        mock_app = MockAppForDataCollector(mock_responses_partial)

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=queries_with_extra,
                relevant_docs=relevant_docs_partial,
                vespa_query_fn=self.vespa_query_fn,
                app=mock_app,
                csv_dir=temp_dir,
            )

            # Verify q4 was filtered out
            self.assertEqual(len(collector.queries_ids), 2)
            self.assertNotIn("q4", collector.queries_ids)

            collector.collect()

            # Verify only q1 and q2 data in CSV
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            csv_path = os.path.join(temp_dir, csv_files[0])

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)

            query_ids_in_csv = set(row[0] for row in rows)
            self.assertEqual(query_ids_in_csv, {"q1", "q2"})
            self.assertNotIn("q4", query_ids_in_csv)

    def test_collect_vespa_query_fn_with_query_id(self):
        """Test collection with vespa_query_fn that accepts query_id."""
        captured_query_ids = []

        def query_fn_with_id(
            query_text: str, top_k: int, query_id: Optional[str] = None
        ) -> dict:
            if query_id:
                captured_query_ids.append(query_id)
            return {
                "yql": f'select * from sources * where text contains "{query_text}";',
                "hits": top_k,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=query_fn_with_id,
                app=self.mock_app,
                csv_dir=temp_dir,
            )

            collector.collect()

            # Verify query_ids were passed to the function
            self.assertEqual(set(captured_query_ids), {"q1", "q2", "q3"})

    def test_collect_default_body_parameters_added(self):
        """Test that default body parameters are added to query bodies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                csv_dir=temp_dir,
            )

            collector.collect()

            # Check captured query bodies have default parameters
            for query_body in self.mock_app.captured_query_bodies:
                self.assertIn("timeout", query_body)
                self.assertEqual(query_body["timeout"], "5s")
                self.assertIn("presentation.timing", query_body)
                self.assertEqual(query_body["presentation.timing"], True)

    def test_collect_preserves_user_query_parameters(self):
        """Test that user-provided query parameters are preserved."""

        def query_fn_with_custom_timeout(query_text: str, top_k: int) -> dict:
            return {
                "yql": f'select * from sources * where text contains "{query_text}";',
                "hits": top_k,
                "timeout": "10s",  # Custom timeout
                "custom_param": "custom_value",
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=query_fn_with_custom_timeout,
                app=self.mock_app,
                csv_dir=temp_dir,
            )

            collector.collect()

            # Check that user parameters are preserved
            for query_body in self.mock_app.captured_query_bodies:
                self.assertEqual(query_body["timeout"], "10s")  # User's value preserved
                self.assertEqual(query_body["custom_param"], "custom_value")
                self.assertEqual(
                    query_body["presentation.timing"], True
                )  # Default added

    def test_feature_collection_parameters_initialization(self):
        """Test initialization with feature collection parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test default parameters
            collector_default = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                name="test_default",
                csv_dir=temp_dir,
            )

            self.assertTrue(collector_default.collect_matchfeatures)

            self.assertFalse(collector_default.collect_rankfeatures)
            self.assertFalse(collector_default.collect_summaryfeatures)

            # Test custom parameters
            collector_custom = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                name="test_custom",
                csv_dir=temp_dir,
                collect_matchfeatures=False,
                collect_rankfeatures=True,
                collect_summaryfeatures=True,
            )

            self.assertFalse(collector_custom.collect_matchfeatures)
            self.assertTrue(collector_custom.collect_rankfeatures)
            self.assertTrue(collector_custom.collect_summaryfeatures)

    def test_collect_with_rankfeatures_enabled(self):
        """Test that rankfeatures parameter adds listFeatures to query body."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                name="test_rankfeatures",
                csv_dir=temp_dir,
                collect_rankfeatures=True,
            )

            collector.collect()

            # Check that ranking.listFeatures was added to query bodies
            captured_bodies = self.mock_app.captured_query_bodies
            for body in captured_bodies:
                self.assertIn("ranking", body)
                self.assertEqual(body["ranking"]["listFeatures"], "true")

    def test_collect_with_string_ranking_profile(self):
        """Test handling of string ranking profile when rankfeatures is enabled."""

        def query_fn_with_string_ranking(query_text: str, top_k: int) -> dict:
            return {
                "yql": f'select * from sources * where text contains "{query_text}";',
                "hits": top_k,
                "ranking": "my_profile",
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=query_fn_with_string_ranking,
                app=self.mock_app,
                name="test_string_ranking",
                csv_dir=temp_dir,
                collect_rankfeatures=True,
            )

            collector.collect()

            # Check that string ranking profile was converted to dict with listFeatures
            captured_bodies = self.mock_app.captured_query_bodies
            for body in captured_bodies:
                self.assertIn("ranking", body)
                self.assertIsInstance(body["ranking"], dict)
                self.assertEqual(body["ranking"]["profile"], "my_profile")
                self.assertEqual(body["ranking"]["listFeatures"], "true")

    def test_collect_with_features_in_hits(self):
        """Test collection with various feature types in mock responses."""
        # Mock responses with different feature types
        mock_responses_with_features = [
            # Response for q1 with match features and rank features
            MockVespaResponse(
                [
                    {
                        "id": "doc1",
                        "relevance": 0.9,
                        "matchfeatures": {
                            "bm25(title)": 1.5,
                            "bm25(body)": 2.3,
                            "fieldMatch(title)": 0.8,
                        },
                        "rankfeatures": {
                            "nativeRank(title)": 0.7,
                            "nativeRank(body)": 0.6,
                            "attributeMatch(id).totalWeight": 0.0,
                        },
                        "summaryfeatures": {
                            "summary_score": 0.95,
                            "custom_feature": 1.2,
                        },
                    },
                    {
                        "id": "doc10",
                        "relevance": 0.8,
                        "matchfeatures": {
                            "bm25(title)": 1.2,
                            "bm25(body)": 1.8,
                            "fieldMatch(title)": 0.6,
                        },
                        "rankfeatures": {
                            "nativeRank(title)": 0.5,
                            "nativeRank(body)": 0.4,
                            "attributeMatch(id).totalWeight": 0.1,
                        },
                        "summaryfeatures": {
                            "summary_score": 0.85,
                            "custom_feature": 0.9,
                        },
                    },
                ]
            ),
            # Response for q2
            MockVespaResponse(
                [
                    {
                        "id": "doc4",
                        "relevance": 0.85,
                        "matchfeatures": {
                            "bm25(title)": 2.1,
                            "bm25(body)": 2.8,
                            "fieldMatch(title)": 0.9,
                        },
                        "rankfeatures": {
                            "nativeRank(title)": 0.8,
                            "nativeRank(body)": 0.7,
                            "attributeMatch(id).totalWeight": 0.0,
                        },
                        "summaryfeatures": {
                            "summary_score": 0.92,
                            "custom_feature": 1.1,
                        },
                    }
                ]
            ),
            # Response for q3
            MockVespaResponse(
                [
                    {
                        "id": "doc6",
                        "relevance": 0.9,
                        "matchfeatures": {
                            "bm25(title)": 1.8,
                            "bm25(body)": 2.5,
                            "fieldMatch(title)": 0.7,
                        },
                        "rankfeatures": {
                            "nativeRank(title)": 0.6,
                            "nativeRank(body)": 0.8,
                            "attributeMatch(id).totalWeight": 0.2,
                        },
                        "summaryfeatures": {
                            "summary_score": 0.88,
                            "custom_feature": 1.0,
                        },
                    }
                ]
            ),
        ]

        mock_app_with_features = MockAppForDataCollector(mock_responses_with_features)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test collecting all feature types
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=mock_app_with_features,
                name="test_all_features",
                csv_dir=temp_dir,
                collect_matchfeatures=True,
                collect_rankfeatures=True,
                collect_summaryfeatures=True,
            )

            collector.collect()

            # Read and verify CSV content includes features
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            csv_path = os.path.join(temp_dir, csv_files[0])

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            # Check that feature columns are present
            expected_base_columns = [
                "query_id",
                "query_text",
                "doc_id",
                "relevance_label",
                "relevance_score",
            ]
            expected_match_features = [
                "match_bm25(body)",
                "match_bm25(title)",
                "match_fieldMatch(title)",
            ]
            expected_rank_features = [
                "rank_attributeMatch(id).totalWeight",
                "rank_nativeRank(body)",
                "rank_nativeRank(title)",
            ]
            expected_summary_features = [
                "summary_custom_feature",
                "summary_summary_score",
            ]

            # Verify all expected columns are in header
            for col in expected_base_columns:
                self.assertIn(col, header)
            for col in expected_match_features:
                self.assertIn(col, header)
            for col in expected_rank_features:
                self.assertIn(col, header)
            for col in expected_summary_features:
                self.assertIn(col, header)

            # Verify feature data is present and correctly formatted
            for row in rows:
                if row[2] == "doc1":  # Check specific document
                    row_dict = dict(zip(header, row))
                    self.assertEqual(float(row_dict["match_bm25(title)"]), 1.5)
                    self.assertEqual(float(row_dict["rank_nativeRank(title)"]), 0.7)
                    self.assertEqual(float(row_dict["summary_summary_score"]), 0.95)

    def test_collect_with_only_matchfeatures(self):
        """Test collection with only match features enabled."""
        mock_responses_match_only = [
            MockVespaResponse(
                [
                    {
                        "id": "doc1",
                        "relevance": 0.9,
                        "matchfeatures": {
                            "bm25(title)": 1.5,
                            "bm25(body)": 2.3,
                        },
                        "rankfeatures": {
                            "nativeRank(title)": 0.7,  # Should be ignored
                        },
                    }
                ]
            ),
        ]

        mock_app_match_only = MockAppForDataCollector(mock_responses_match_only)

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries={"q1": "test query"},
                relevant_docs={"q1": {"doc1"}},
                vespa_query_fn=self.vespa_query_fn,
                app=mock_app_match_only,
                name="test_match_only",
                csv_dir=temp_dir,
                collect_matchfeatures=True,
                collect_rankfeatures=False,
                collect_summaryfeatures=False,
            )

            collector.collect()

            # Read CSV and verify only match features are included
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            csv_path = os.path.join(temp_dir, csv_files[0])

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                _rows = list(reader)

            # Should have match features but not rank or summary features
            match_feature_columns = [col for col in header if col.startswith("match_")]
            rank_feature_columns = [col for col in header if col.startswith("rank_")]
            summary_feature_columns = [
                col for col in header if col.startswith("summary_")
            ]

            self.assertGreater(len(match_feature_columns), 0)
            self.assertEqual(len(rank_feature_columns), 0)
            self.assertEqual(len(summary_feature_columns), 0)

    def test_collect_with_no_features(self):
        """Test collection with all feature types disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries=self.queries,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=self.vespa_query_fn,
                app=self.mock_app,
                name="test_no_features",
                csv_dir=temp_dir,
                collect_matchfeatures=False,
                collect_rankfeatures=False,
                collect_summaryfeatures=False,
            )

            collector.collect()

            # Read CSV and verify only basic columns are present
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            csv_path = os.path.join(temp_dir, csv_files[0])

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)

            # Should only have the basic columns
            expected_columns = [
                "query_id",
                "query_text",
                "doc_id",
                "relevance_label",
                "relevance_score",
            ]
            self.assertEqual(header, expected_columns)

    def test_collect_with_missing_features_in_hits(self):
        """Test collection when some hits are missing feature data."""
        mock_responses_partial_features = [
            MockVespaResponse(
                [
                    {
                        "id": "doc1",
                        "relevance": 0.9,
                        "matchfeatures": {
                            "bm25(title)": 1.5,
                        },
                        # Missing rankfeatures and summaryfeatures
                    },
                    {
                        "id": "doc2",
                        "relevance": 0.8,
                        # Missing all feature types
                    },
                ]
            ),
        ]

        mock_app_partial = MockAppForDataCollector(mock_responses_partial_features)

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries={"q1": "test query"},
                relevant_docs={"q1": {"doc1", "doc2"}},
                vespa_query_fn=self.vespa_query_fn,
                app=mock_app_partial,
                name="test_partial_features",
                csv_dir=temp_dir,
                collect_matchfeatures=True,
                collect_rankfeatures=True,
                collect_summaryfeatures=True,
            )

            collector.collect()

            # Read CSV and verify missing features are handled as 0.0
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            csv_path = os.path.join(temp_dir, csv_files[0])

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            # Find the row for doc2 (which has no features)
            doc2_row = None
            for row in rows:
                if row[2] == "doc2":
                    doc2_row = dict(zip(header, row))
                    break

            self.assertIsNotNone(doc2_row)

            # Check that missing features default to empty strings (CSV best practice)
            feature_columns = [
                col for col in header if col.startswith(("match_", "rank_", "summary_"))
            ]
            for col in feature_columns:
                if col in doc2_row:
                    self.assertEqual(doc2_row[col], "")

    def test_csv_missing_values_best_practice(self):
        """Test that missing feature values follow CSV best practices."""
        # Create a response where some hits have features and others don't
        mock_responses_mixed = [
            MockVespaResponse(
                [
                    {
                        "id": "doc_with_features",
                        "relevance": 0.9,
                        "matchfeatures": {
                            "bm25(title)": 1.5,
                        },
                    },
                    {
                        "id": "doc_without_features",
                        "relevance": 0.8,
                        # No feature sections
                    },
                ]
            ),
        ]

        mock_app_mixed = MockAppForDataCollector(mock_responses_mixed)

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = VespaFeatureCollector(
                queries={"q1": "test query"},
                relevant_docs={"q1": {"doc_with_features"}},
                vespa_query_fn=self.vespa_query_fn,
                app=mock_app_mixed,
                name="test_csv_best_practice",
                csv_dir=temp_dir,
                collect_matchfeatures=True,
            )

            collector.collect()

            # Read CSV and verify missing values are empty strings
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            csv_path = os.path.join(temp_dir, csv_files[0])

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            # Find rows for both documents
            doc_with_features_row = None
            doc_without_features_row = None

            for row in rows:
                if row[2] == "doc_with_features":
                    doc_with_features_row = dict(zip(header, row))
                elif row[2] == "doc_without_features":
                    doc_without_features_row = dict(zip(header, row))

            self.assertIsNotNone(doc_with_features_row)
            self.assertIsNotNone(doc_without_features_row)

            # Check that document with features has actual values
            match_bm25_title_col = "match_bm25(title)"
            self.assertIn(match_bm25_title_col, header)
            self.assertEqual(doc_with_features_row[match_bm25_title_col], "1.5")

            # Check that document without features has empty string (not "0.0")
            self.assertEqual(doc_without_features_row[match_bm25_title_col], "")

            # Verify this can be properly handled by pandas (empty strings become NaN)
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)

                # Empty strings should become NaN in pandas
                self.assertTrue(
                    pd.isna(
                        df[df["doc_id"] == "doc_without_features"][
                            match_bm25_title_col
                        ].iloc[0]
                    )
                )

                # Actual values should be preserved
                self.assertEqual(
                    df[df["doc_id"] == "doc_with_features"][match_bm25_title_col].iloc[
                        0
                    ],
                    1.5,
                )

            except ImportError:
                # Skip pandas test if not available, but the empty string test above is sufficient
                pass
