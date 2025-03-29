import unittest
from vespa.evaluation import VespaEvaluator
from dataclasses import dataclass
from typing import List, Dict, Any
import httpx


@dataclass
class MockVespaResponse:
    """Mock Vespa query response"""

    hits: List[Dict[str, Any]]

    def add_namespace_to_hit_ids(self, hits) -> str:
        new_hits = []
        for hit in hits:
            hit["id"] = f"id:mynamespace:mydoctype::{hit['id']}"
            new_hits.append(hit)
        return new_hits

    def get_json(self):
        return {"root": {"children": self.add_namespace_to_hit_ids(self.hits)}}

    @property
    def status_code(self):
        return 200


class MockVespaApp:
    """Universal mock Vespa app for all tests"""

    def __init__(self, responses):
        self.responses = responses
        # For testing purposes - capture what was passed
        self.captured_query_bodies = None
        self.captured_client_kwargs = None

    def query_many(self, query_bodies, client_kwargs=None, **kwargs):
        # Store captured values for test validation
        self.captured_query_bodies = query_bodies
        self.captured_client_kwargs = client_kwargs
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
        capturing_app = MockVespaApp([dummy_response] * len(self.queries))

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
        capturing_app = MockVespaApp([dummy_response] * len(self.queries))

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
        capturing_app = MockVespaApp([dummy_response] * len(self.queries))

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
        capturing_app = MockVespaApp([dummy_response] * len(self.queries))

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

    def test_client_kwargs_passed_to_query_many(self):
        """Test that client_kwargs are correctly passed to query_many."""
        # Create a custom timeout
        custom_timeout = httpx.Timeout(connect=2.0, read=10.0, write=3.0, pool=15.0)

        # Set up client_kwargs with the custom timeout
        client_kwargs = {
            "timeout": custom_timeout,
            "limits": httpx.Limits(max_keepalive_connections=10),
            "follow_redirects": False,
        }

        # Create a dummy response
        dummy_response = MockVespaResponse([{"id": "doc1", "relevance": 1.0}])
        capturing_app = MockVespaApp([dummy_response] * len(self.queries))

        # Create evaluator with client_kwargs
        evaluator = VespaEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=self.vespa_query_fn,
            app=capturing_app,
            client_kwargs=client_kwargs,
        )

        # Run the evaluator
        evaluator.run()

        # Verify that client_kwargs were passed correctly to query_many
        self.assertEqual(capturing_app.captured_client_kwargs, client_kwargs)

        # Verify the individual values
        self.assertEqual(
            capturing_app.captured_client_kwargs["timeout"], custom_timeout
        )
        self.assertEqual(
            capturing_app.captured_client_kwargs["limits"].max_keepalive_connections, 10
        )
        self.assertEqual(
            capturing_app.captured_client_kwargs["follow_redirects"], False
        )


if __name__ == "__main__":
    unittest.main()
