# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import unittest

from vespa.evaluation import VespaMTEBEvaluator
from vespa.models import ModelConfig


class TestVespaMTEBEvaluatorCloud(unittest.TestCase):
    """Test VespaMTEBEvaluator with Vespa Cloud deployment."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_config = ModelConfig(
            model_id="e5-small-v2",
            embedding_dim=384,
            binarized=False,
            max_tokens=512,
            query_prepend="query: ",
            document_prepend="passage: ",
        )
        cls.vespa_tenant = "vespa-team"
        cls.vespa_api_key = os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n")

    def test_evaluate_nanomsarco_retrieval_cloud(self):
        """Test running small evaluation on Vespa Cloud."""
        evaluator = VespaMTEBEvaluator(
            model_configs=self.model_config,
            task_name="NanoMSMARCORetrieval",  # Small task for quick testing
            results_dir="results",
            overwrite=True,
            deployment_target="cloud",
            tenant=self.vespa_tenant,
            application="mteb-test",
            key_content=self.vespa_api_key,
            auto_cleanup=True,
        )
        results = evaluator.evaluate()
        # Verify results structure
        self.assertIn("metadata", results)
        self.assertIn("results", results)
        self.assertIn("NanoMSMARCORetrieval", results["results"])
        self.assertIsNotNone(results["metadata"]["benchmark_finished_at"])
        # Finished evaluation for 'NanoMSMARCORetrieval' with 'semantic':
        # NDCG@10: 0.62092
        self.assertAlmostEqual(
            results["results"]["NanoMSMARCORetrieval"]["semantic"]["scores"]["train"][
                0
            ]["ndcg_at_10"],
            0.62092,
            places=5,
        )
        # Finished evaluation for 'NanoMSMARCORetrieval' with 'bm25':
        # NDCG@10: 0.52063
        self.assertAlmostEqual(
            results["results"]["NanoMSMARCORetrieval"]["bm25"]["scores"]["train"][0][
                "ndcg_at_10"
            ],
            0.52063,
            places=5,
        )
        # Finished evaluation for 'NanoMSMARCORetrieval' with 'fusion':
        # NDCG@10: 0.60338
        self.assertAlmostEqual(
            results["results"]["NanoMSMARCORetrieval"]["fusion"]["scores"]["train"][0][
                "ndcg_at_10"
            ],
            0.60338,
            places=5,
        )
        # Finished evaluation for 'NanoMSMARCORetrieval' with 'atan_norm':
        # NDCG@10: 0.62851
        self.assertAlmostEqual(
            results["results"]["NanoMSMARCORetrieval"]["atan_norm"]["scores"]["train"][
                0
            ]["ndcg_at_10"],
            0.62851,
            places=5,
        )
        # Finished evaluation for 'NanoMSMARCORetrieval' with 'norm_linear':
        # NDCG@10: 0.64687
        self.assertAlmostEqual(
            results["results"]["NanoMSMARCORetrieval"]["norm_linear"]["scores"][
                "train"
            ][0]["ndcg_at_10"],
            0.64687,
            places=5,
        )
