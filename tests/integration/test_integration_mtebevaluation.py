# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license.
# See LICENSE in the project root.

import unittest
import tempfile
import shutil
from pathlib import Path

from vespa.evaluation import VespaMTEBEvaluator
from vespa.models import ModelConfig


class TestVespaMTEBEvaluatorIntegration(unittest.TestCase):
    """
    Integration tests for VespaMTEBEvaluator.

    Purpose:
        Verify that VespaMTEBEvaluator correctly deploys a Vespa application,
        indexes documents, and evaluates retrieval quality using MTEB benchmarks.

    Test Coverage:
        - Construct VespaMTEBEvaluator with model configs
        - Run evaluation on NanoMSMARCO benchmark
        - Verify results structure and metrics
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        cls.results_dir = tempfile.mkdtemp(prefix="mteb_test_")
        cls.model_configs = [
            ModelConfig(
                model_id="alibaba-gte-modernbert-int8",
                embedding_dim=768,
                binarized=True,
                embedding_field_type="int8",
                distance_metric="hamming",
                model_url="https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/onnx/model_int8.onnx",
                tokenizer_url="https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/tokenizer.json",
                max_tokens=8192,
                pooling_strategy="cls",
            ),
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, "results_dir") and Path(cls.results_dir).exists():
            shutil.rmtree(cls.results_dir)

    def test_mteb_evaluator_nanomsmarco(self):
        """
        Test VespaMTEBEvaluator on NanoMSMARCO benchmark.

        This test verifies that:
        1. The evaluator can deploy a Vespa application
        2. Index documents from the benchmark corpus
        3. Execute queries and compute retrieval metrics
        4. Return properly structured results
        """
        evaluator = VespaMTEBEvaluator(
            model_configs=self.model_configs,
            task_name="NanoMSMARCORetrieval",
            results_dir=self.results_dir,
            overwrite=True,
        )

        try:
            results = evaluator.evaluate()

            # Verify results structure
            self.assertIsInstance(results, dict)
            self.assertIn("metadata", results)
            self.assertIn("results", results)

            # Verify metadata
            metadata = results["metadata"]
            self.assertEqual(metadata["benchmark_name"], "NanoMSMARCORetrieval")
            self.assertIsNotNone(metadata["benchmark_finished_at"])

            # Verify we have results for the task
            task_results = results["results"]
            self.assertIn("NanoMSMARCORetrieval", task_results)

            # Verify at least one query function was evaluated
            task_data = task_results["NanoMSMARCORetrieval"]
            self.assertGreater(len(task_data), 0)

            # Check that we have scores
            for query_fn_name, query_fn_results in task_data.items():
                self.assertIn("scores", query_fn_results)
                self.assertIn("finished_at", query_fn_results)
                print(
                    f"Query function '{query_fn_name}' scores: {query_fn_results['scores']}"
                )

            # Expected NDCG@10 scores for each query function
            expected_scores = {
                "semantic": 0.63944,
                "bm25": 0.52641,
                "fusion": 0.6173,
                "atan_norm": 0.54054,
                "norm_linear": 0.64803,
            }

            # Verify all query functions have expected NDCG@10 scores
            for query_fn_name, expected_ndcg in expected_scores.items():
                self.assertIn(
                    query_fn_name, task_data, f"Missing query function: {query_fn_name}"
                )
                fn_scores = task_data[query_fn_name]["scores"]
                # Check train split scores (MTEB uses 'train' split for this task)
                self.assertIn("train", fn_scores)
                train_scores = fn_scores["train"][0]
                self.assertIn("ndcg_at_10", train_scores)
                # Assert NDCG@10 is approximately the expected value (with tolerance for minor variations)
                self.assertAlmostEqual(
                    train_scores["ndcg_at_10"],
                    expected_ndcg,
                    places=4,
                    msg=f"NDCG@10 mismatch for '{query_fn_name}': expected {expected_ndcg}, got {train_scores['ndcg_at_10']}",
                )

        finally:
            # Ensure cleanup even if test fails
            evaluator.cleanup()


if __name__ == "__main__":
    unittest.main()
