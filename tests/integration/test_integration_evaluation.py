# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license.
# See LICENSE in the project root.

import unittest

from typing import Dict, Any

from datasets import load_dataset
from vespa.package import (
    ApplicationPackage,
    Field,
    Document,
    HNSW,
    Schema,
    RankProfile,
    Component,
    Parameter,
    Function,
    FieldSet,
    SecondPhaseRanking,
)
from vespa.deployment import VespaDocker
from vespa.evaluation import (
    VespaEvaluator,
    VespaMatchEvaluator,
    VespaFeatureCollector,
)
from vespa.io import VespaResponse
import vespa.querybuilder as qb
from pathlib import Path
import os

# Reference metrics from your Sentence Transformers (semantic) runs:
# Code used to produce these metrics:

## /// script
## requires-python = ">=3.10"
## dependencies = [
##     "sentence-transformers==3.3.1",
## ]
## ///
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.evaluation import NanoBEIREvaluator

# model = SentenceTransformer('intfloat/e5-small-v2')

# datasets = ["MSMARCO"]

# evaluator = NanoBEIREvaluator(
#     dataset_names=datasets,
# )

# results = evaluator(model)
# print(results)
SENTENCE_TRANSFORMERS_REF = {
    "accuracy@1": 0.38,
    "accuracy@3": 0.64,
    "accuracy@5": 0.72,
    "accuracy@10": 0.82,
    "precision@1": 0.38,
    "precision@3": 0.21333333333333332,
    "precision@5": 0.14400000000000002,
    "precision@10": 0.08199999999999999,
    "recall@1": 0.38,
    "recall@3": 0.64,
    "recall@5": 0.72,
    "recall@10": 0.82,
    "ndcg@10": 0.6007397354752749,
    "mrr@10": 0.5308571428571428,
    "map@100": 0.5393493336728631,
}


def create_app_package() -> ApplicationPackage:
    """
    Single 'doc' schema with an 'embedding' field that uses hugging-face-embedder (E5).
    Enhanced with additional rank profiles containing match features for training data collection.
    """
    return ApplicationPackage(
        name="localevaluation",
        schema=[
            Schema(
                name="doc",
                document=Document(
                    fields=[
                        Field(
                            name="id", type="string", indexing=["attribute", "summary"]
                        ),
                        Field(
                            name="text",
                            type="string",
                            indexing=["index", "summary"],
                            index="enable-bm25",
                            bolding=True,
                        ),
                        Field(
                            name="embedding",
                            type="tensor<float>(x[384])",
                            indexing=[
                                "input text",
                                "embed",  # uses default hugging-face-embedder
                                "index",
                                "attribute",
                            ],
                            ann=HNSW(distance_metric="angular"),
                            is_document_field=False,
                        ),
                    ]
                ),
                fieldsets=[FieldSet(name="default", fields=["text"])],
                rank_profiles=[
                    RankProfile(
                        name="semantic",
                        inputs=[("query(q)", "tensor<float>(x[384])")],
                        first_phase="closeness(field, embedding)",
                    ),
                    RankProfile(
                        name="hybrid-match",
                        inputs=[("query(q)", "tensor<float>(x[384])")],
                        first_phase="",  # Temporary workaround, as pyvespa does not allow empty first_phase
                    ),
                    RankProfile(
                        name="feature-collection",
                        inputs=[("query(q)", "tensor<float>(x[384])")],
                        functions=[
                            Function(
                                name="cos_sim", expression="closeness(field, embedding)"
                            ),
                            Function(name="bm25text", expression="bm25(text)"),
                        ],
                        first_phase="cos_sim + bm25text",
                        second_phase=SecondPhaseRanking(expression="random"),
                        match_features=["cos_sim", "bm25text"],
                        summary_features=[
                            "cos_sim",
                            "bm25text",
                        ],
                    ),
                ],
            )
        ],
        components=[
            Component(
                id="e5",
                type="hugging-face-embedder",
                parameters=[
                    Parameter(
                        "transformer-model",
                        {
                            "url": "https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx"
                        },
                    ),
                    Parameter(
                        "tokenizer-model",
                        {
                            "url": "https://huggingface.co/intfloat/e5-small-v2/resolve/main/tokenizer.json"
                        },
                    ),
                ],
            ),
        ],
    )


def semantic_query_fn(query_text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Convert plain text into a JSON body for Vespa query with 'semantic' rank profile.
    """
    return {
        "yql": "select * from sources * where ({targetHits:1000}nearestNeighbor(embedding,q));",
        "query": query_text,
        "ranking": "semantic",
        "input.query(q)": f"embed({query_text})",
        "hits": top_k,
        "timeout": "2s",
    }


def hybrid_match_query_fn(query_text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Convert plain text into a JSON body for Vespa query with 'hybrid-match' rank profile.
    """
    return {
        "yql": str(
            qb.select("*")
            .from_("sources *")
            .where(
                qb.nearestNeighbor(
                    field="embedding",
                    query_vector="q",
                    annotations={"targetHits": 100},
                )
                | qb.userQuery(query_text)
            )
        ),
        "query": query_text,
        "ranking": "hybrid-match",
        "input.query(q)": f"embed({query_text})",
    }


def feature_collection_query_fn(
    query_text: str, top_k: int = 10, query_id: str = None
) -> Dict[str, Any]:
    """
    Convert plain text into a JSON body for Vespa query with 'feature-collection' rank profile.
    Includes both semantic similarity and BM25 matching with match features.
    """
    return {
        "yql": str(
            qb.select("*")
            .from_("sources *")
            .where(
                qb.nearestNeighbor(
                    field="embedding",
                    query_vector="q",
                    annotations={"targetHits": 100},
                )
                | qb.userQuery(query_text)
            )
        ),
        "query": query_text,
        "ranking": "feature-collection",
        "input.query(q)": f"embed({query_text})",
        "hits": top_k,
        "timeout": "5s",
        "presentation.timing": True,
    }


class TestEvaluatorsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1) Build and deploy the application package
        cls.package = create_app_package()
        cls.vespa_docker = VespaDocker(port=8089)

        cls.app = cls.vespa_docker.deploy(application_package=cls.package)
        # 2) Load a portion of the NanoMSMARCO corpus
        dataset_id = "zeta-alpha-ai/NanoMSMARCO"
        corpus = load_dataset(dataset_id, "corpus", split="train", streaming=True)
        # Convert each dataset item to feed format
        vespa_feed = corpus.map(
            lambda x: {
                "id": x["_id"],
                "fields": {"text": x["text"], "id": x["_id"]},
            }
        )

        # 3) Feed the documents
        def feed_callback(response: VespaResponse, doc_id: str):
            if not response.is_successful():
                print(f"Error feeding doc {doc_id}: {response.json}")

        cls.app.feed_iterable(vespa_feed, schema="doc", callback=feed_callback)

        # 4) Prepare queries and qrels
        #    The dataset provides separate 'queries' and 'qrels' subsets
        cls.query_ds = load_dataset(dataset_id, "queries", split="train")
        cls.qrels = load_dataset(dataset_id, "qrels", split="train")

        # Convert them to dictionaries
        q_ids = cls.query_ds["_id"]
        q_texts = cls.query_ds["text"]
        cls.ids_to_query = dict(zip(q_ids, q_texts))

        rel_q_ids = cls.qrels["query-id"]
        rel_doc_ids = cls.qrels["corpus-id"]
        cls.relevant_docs = dict(zip(rel_q_ids, rel_doc_ids))

    # @classmethod
    # def tearDownClass(cls):
    #     # Clean up container
    #     cls.vespa_docker.container.stop(timeout=10)
    #     cls.vespa_docker.container.remove()

    def test_semantic_metrics_close_to_sentence_transformers(self):
        """
        Use VespaEvaluator on the 'semantic' ranking profile with the
        queries & relevant docs from the NanoMSMARCO subset,
        then compare the results to the reference values from ST.
        """

        evaluator = VespaEvaluator(
            queries=self.ids_to_query,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=semantic_query_fn,
            app=self.app,
            name="semantic",
        )

        # Evaluate
        results = evaluator.run()
        print("Got results: ", results)

        # Compare to your reference values
        for metric, st_value in SENTENCE_TRANSFORMERS_REF.items():
            vespa_val = results[metric]
            with self.subTest(metric=metric):
                # Tolerance of 0.01
                self.assertAlmostEqual(vespa_val, st_value, delta=0.0001)

    def test_hybrid_match_metrics(self):
        """
        Use VespaMatchEvaluator on the 'hybrid-match' ranking profile with the
        queries & relevant docs from the NanoMSMARCO subset,
        then compare the results to the reference values from ST.
        """

        evaluator = VespaMatchEvaluator(
            queries=self.ids_to_query,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=hybrid_match_query_fn,
            app=self.app,
            name="hybrid-match",
            write_csv=True,
            write_verbose=True,
        )

        # Evaluate
        results = evaluator.run()
        self.assertEqual(results["match_recall"], 1.0)
        self.assertEqual(results["avg_recall_per_query"], 1.0)
        print("Got results: ", results)

        # Assert file is written
        self.assertTrue(Path(evaluator.csv_file).exists())
        self.assertTrue(Path(evaluator.verbose_csv_file).exists())
        # Read the csv and check recall column
        import csv

        with open(evaluator.verbose_csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # assert that recall column is 1.0 for all rows
            for row in rows:
                self.assertEqual(float(row["recall"]), 1.0)

    def test_vespa_feature_collector_integration(self):
        """
        Test VespaFeatureCollector with different rank profiles to ensure
        it correctly collects match features and summary features.
        """
        # Test with feature-collection rank profile
        feature_collector = VespaFeatureCollector(
            queries=self.ids_to_query,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=feature_collection_query_fn,
            app=self.app,
            name="feature-collection-test",
            id_field="id",
            collect_matchfeatures=True,
            collect_summaryfeatures=True,
            collect_rankfeatures=False,
        )

        # Collect features
        results = feature_collector.collect()

        # Verify structure
        self.assertIsInstance(results, dict)
        # Expected dict:
        # {"results": [{'query_id': '721409', 'doc_id': '7301814', 'relevance_label': 0.0, 'relevance_score': 0.6344519422768364, 'match_bm25(text)': 0.0, 'match_closeness(field,embedding)': 0.6344519422768364, 'summary_bm25(text)': 0.0, 'summary_closeness(field,embedding)': 0.6344519422768364, 'summary_vespa.summaryFeatures.cached': 0.0}, ...],
        # print(results)
        rows = results["results"]

        # Check that we have data
        self.assertGreater(len(rows), 0)
        expected_match_features = [
            "match_cos_sim",
            "match_bm25text",
        ]
        expected_summary_features = [
            "summary_cos_sim",
            "summary_bm25text",
        ]
        expected_base_features = [
            "query_id",
            "doc_id",
            "relevance_label",
            "relevance_score",
        ]
        for row in rows:
            # Check match features
            for feature in expected_match_features:
                self.assertIn(
                    feature, row, f"Match feature {feature} missing in row {row}"
                )
            # Check summary features
            for feature in expected_summary_features:
                self.assertIn(
                    feature, row, f"Summary feature {feature} missing in row {row}"
                )
            # Check base features
            for feature in expected_base_features:
                self.assertIn(
                    feature, row, f"Base feature {feature} missing in row {row}"
                )

    def test_vespa_feature_collector_csv_output(self):
        """
        Test that VespaFeatureCollector correctly writes CSV files when enabled.
        """
        # Create a temporary directory for CSV output
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            feature_collector = VespaFeatureCollector(
                queries=self.ids_to_query,
                relevant_docs=self.relevant_docs,
                vespa_query_fn=feature_collection_query_fn,
                app=self.app,
                name="csv-test",
                id_field="id",
                collect_matchfeatures=True,
                collect_summaryfeatures=False,
                collect_rankfeatures=False,
                write_csv=True,
                csv_dir=temp_dir,
            )

            results = feature_collector.collect()

            # Check that CSV file was created (single file with all data)
            csv_file = feature_collector.csv_file

            self.assertTrue(
                os.path.exists(csv_file),
                f"Training data CSV not created at {csv_file}",
            )

            # Verify CSV content by reading and comparing basic structure
            import csv

            # Read the single CSV file
            with open(csv_file, "r") as f:
                csv_reader = csv.DictReader(f)
                rows_from_csv = list(csv_reader)

            # Basic checks: we should have some rows
            self.assertGreater(len(rows_from_csv), 0)

            # Verify that the CSV has the same number of total samples as the return data
            total_samples = len(results["results"])
            self.assertEqual(len(rows_from_csv), total_samples)

            # Check that expected columns are present
            if rows_from_csv:
                csv_columns = set(rows_from_csv[0].keys())
                expected_base_columns = {
                    "query_id",
                    "doc_id",
                    "relevance_label",
                    "relevance_score",
                }
                for col in expected_base_columns:
                    self.assertIn(
                        col, csv_columns, f"Expected column {col} missing from CSV"
                    )

            print("CSV output test completed:")
            print(f"  - Training data CSV: {csv_file} ({len(rows_from_csv)} rows)")
            print(
                f"  - Columns: {sorted(list(csv_columns)) if rows_from_csv else 'No data'}"
            )

    def test_vespa_feature_collector_ratio_random_hits_strategy(self):
        """
        Test VespaFeatureCollector with RATIO random hits sampling strategy.
        This tests the integration with a real Vespa application.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a smaller subset for faster testing
            limited_queries = dict(list(self.ids_to_query.items())[:5])
            limited_relevant_docs = {
                qid: doc_id
                for qid, doc_id in self.relevant_docs.items()
                if qid in limited_queries
            }

            feature_collector = VespaFeatureCollector(
                queries=limited_queries,
                relevant_docs=limited_relevant_docs,
                vespa_query_fn=feature_collection_query_fn,
                app=self.app,
                name="ratio-strategy-test",
                id_field="id",
                collect_matchfeatures=True,
                collect_summaryfeatures=False,
                collect_rankfeatures=False,
                write_csv=True,
                csv_dir=temp_dir,
                random_hits_strategy="ratio",
                random_hits_value=2.0,  # 2x ratio
                max_random_hits_per_query=50,
            )

            results = feature_collector.collect()

            # Verify that random hits were collected
            self.assertIn("results", results)
            features = results["results"]
            self.assertGreater(len(features), 0)

            # Count relevant vs random hits
            relevant_count = sum(1 for f in features if f["relevance_label"] == 1.0)
            random_count = sum(1 for f in features if f["relevance_label"] == 0.0)

            # With ratio strategy, we should have approximately 2x random hits
            if relevant_count > 0:
                ratio = random_count / relevant_count
                self.assertGreater(ratio, 1.5)  # Allow some tolerance
                self.assertLess(ratio, 3.0)  # Upper bound with tolerance

            # Verify CSV was created
            self.assertTrue(os.path.exists(feature_collector.csv_file))

            print(
                f"Ratio strategy test - Relevant: {relevant_count}, Random: {random_count}, Ratio: {random_count/max(relevant_count, 1):.2f}"
            )

    def test_vespa_feature_collector_fixed_random_hits_strategy(self):
        """
        Test VespaFeatureCollector with FIXED random hits sampling strategy.
        This tests the integration with a real Vespa application.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a smaller subset for faster testing
            limited_queries = dict(list(self.ids_to_query.items())[:3])
            limited_relevant_docs = {
                qid: doc_id
                for qid, doc_id in self.relevant_docs.items()
                if qid in limited_queries
            }

            fixed_random_hits = 5
            feature_collector = VespaFeatureCollector(
                queries=limited_queries,
                relevant_docs=limited_relevant_docs,
                vespa_query_fn=feature_collection_query_fn,
                app=self.app,
                name="fixed-strategy-test",
                id_field="id",
                collect_matchfeatures=True,
                collect_summaryfeatures=False,
                collect_rankfeatures=False,
                write_csv=True,
                csv_dir=temp_dir,
                random_hits_strategy="fixed",
                random_hits_value=fixed_random_hits,
            )

            results = feature_collector.collect()

            # Verify that random hits were collected
            self.assertIn("results", results)
            features = results["results"]
            self.assertGreater(len(features), 0)

            # Count relevant vs random hits per query
            query_counts = {}
            for feature in features:
                qid = feature["query_id"]
                if qid not in query_counts:
                    query_counts[qid] = {"relevant": 0, "random": 0}

                if feature["relevance_label"] == 1.0:
                    query_counts[qid]["relevant"] += 1
                else:
                    query_counts[qid]["random"] += 1

            # With fixed strategy, each query should have exactly fixed_random_hits random hits
            for qid, counts in query_counts.items():
                self.assertEqual(
                    counts["random"],
                    fixed_random_hits,
                    f"Query {qid} should have {fixed_random_hits} random hits, got {counts['random']}",
                )

            # Verify CSV was created
            self.assertTrue(os.path.exists(feature_collector.csv_file))

            print(f"Fixed strategy test - Query counts: {query_counts}")

    def test_vespa_feature_collector_no_random_hits(self):
        """
        Test VespaFeatureCollector with random hits disabled (value=0).
        This should only collect relevant hits.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a smaller subset for faster testing
            limited_queries = dict(list(self.ids_to_query.items())[:3])
            limited_relevant_docs = {
                qid: doc_id
                for qid, doc_id in self.relevant_docs.items()
                if qid in limited_queries
            }

            feature_collector = VespaFeatureCollector(
                queries=limited_queries,
                relevant_docs=limited_relevant_docs,
                vespa_query_fn=feature_collection_query_fn,
                app=self.app,
                name="no-random-hits-test",
                id_field="id",
                collect_matchfeatures=True,
                collect_summaryfeatures=False,
                collect_rankfeatures=False,
                write_csv=True,
                csv_dir=temp_dir,
                random_hits_strategy="fixed",
                random_hits_value=0,  # No random hits
            )

            results = feature_collector.collect()

            # Verify that only relevant hits were collected
            self.assertIn("results", results)
            features = results["results"]
            self.assertGreater(len(features), 0)

            # All hits should be relevant (relevance_label == 1.0)
            for feature in features:
                self.assertEqual(
                    feature["relevance_label"],
                    1.0,
                    "All hits should be relevant when random hits are disabled",
                )

            # Verify CSV was created
            self.assertTrue(os.path.exists(feature_collector.csv_file))

            print(
                f"No random hits test - Total features: {len(features)}, all should be relevant"
            )

    def test_vespa_feature_collector_max_random_hits_limit(self):
        """
        Test VespaFeatureCollector with max_random_hits_per_query limit.
        This tests that the limit is respected even with high ratio values.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a smaller subset for faster testing
            limited_queries = dict(list(self.ids_to_query.items())[:3])
            limited_relevant_docs = {
                qid: doc_id
                for qid, doc_id in self.relevant_docs.items()
                if qid in limited_queries
            }

            max_random_hits = 3
            feature_collector = VespaFeatureCollector(
                queries=limited_queries,
                relevant_docs=limited_relevant_docs,
                vespa_query_fn=feature_collection_query_fn,
                app=self.app,
                name="max-limit-test",
                id_field="id",
                collect_matchfeatures=True,
                collect_summaryfeatures=False,
                collect_rankfeatures=False,
                write_csv=True,
                csv_dir=temp_dir,
                random_hits_strategy="ratio",
                random_hits_value=10.0,  # high ratio
                max_random_hits_per_query=max_random_hits,  # But limited by this
            )

            results = feature_collector.collect()

            # Verify that random hits were collected but limited
            self.assertIn("results", results)
            features = results["results"]
            self.assertGreater(len(features), 0)

            # Count random hits per query
            query_random_counts = {}
            for feature in features:
                qid = feature["query_id"]
                if qid not in query_random_counts:
                    query_random_counts[qid] = 0

                if feature["relevance_label"] == 0.0:
                    query_random_counts[qid] += 1

            # Each query should have at most max_random_hits random hits
            for qid, random_count in query_random_counts.items():
                self.assertLessEqual(
                    random_count,
                    max_random_hits,
                    f"Query {qid} should have at most {max_random_hits} random hits, got {random_count}",
                )

            # Verify CSV was created
            self.assertTrue(os.path.exists(feature_collector.csv_file))

            print(f"Max limit test - Random hits per query: {query_random_counts}")
