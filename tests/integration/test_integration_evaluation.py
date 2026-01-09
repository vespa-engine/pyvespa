# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license.
# See LICENSE in the project root.

import unittest

from typing import Dict, Any

from datasets import load_dataset
from vespa.package import (
    ApplicationPackage,
    Field,
    Document,
    DocumentSummary,
    HNSW,
    Schema,
    RankProfile,
    Component,
    Parameter,
    Function,
    FieldSet,
    SecondPhaseRanking,
    Summary,
)
from vespa.configuration.query_profiles import query_profile, query_profile_type, field
from vespa.deployment import VespaDocker
from vespa.evaluation import (
    VespaEvaluator,
    VespaMatchEvaluator,
    VespaFeatureCollector,
    VespaNNParameterOptimizer,
)
from vespa.io import VespaResponse
import vespa.querybuilder as qb
from pathlib import Path
import json
import os
import pandas as pd
import requests
import tempfile

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


def small_targethits_query_fn(query_text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Convert plain text into a JSON body for Vespa query with 'semantic' rank profile and targetHits=10.
    """
    return {
        "yql": str(
            qb.select("*")
            .from_("sources *")
            .where(
                qb.nearestNeighbor(
                    field="embedding",
                    query_vector="q",
                    annotations={"targetHits": 10},
                )
            )
        ),
        "query": query_text,
        "ranking": "semantic",
        "input.query(q)": f"embed({query_text})",
        "timeout": "10s",
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
        "timeout": "10s",
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

    @classmethod
    def tearDownClass(cls):
        # Clean up container
        cls.vespa_docker.container.stop(timeout=10)
        cls.vespa_docker.container.remove()

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
            id_field="id",
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

    def test_small_targethits_metrics(self):
        """
        Test VespaMatchEvaluator with a smaller targetHits=10 in the query,
        to verify that match_recall and avg_recall_per_query are less than 1.0.
        """

        evaluator = VespaMatchEvaluator(
            queries=self.ids_to_query,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=small_targethits_query_fn,
            app=self.app,
            name="small-targethits",
            id_field="id",
            write_csv=True,
            write_verbose=True,
        )

        # Evaluate
        results = evaluator.run()
        # Assert avg_matched_per_query is 10
        self.assertEqual(results["avg_matched_per_query"], 10.0)
        # This should be less than 1.0 due to smaller targetHits=10
        self.assertLess(results["match_recall"], 1.0)
        self.assertLess(results["avg_recall_per_query"], 1.0)
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
                self.assertLessEqual(float(row["recall"]), 1.0)
                self.assertGreaterEqual(float(row["recall"]), 0.0)

    def test_extremely_many_relevant_docs(self):
        """
        Test that VespaMatchEvaluator can handle queries with a very large number of relevant docs.
        """
        # Create a fake relevant_docs mapping with 1000 relevant docs for a single query
        many_rels = {
            qid: {f"NOT_A_DOCID-{i}" for i in range(1000)} for qid in self.ids_to_query
        }
        evaluator = VespaMatchEvaluator(
            queries=self.ids_to_query,
            relevant_docs=many_rels,
            vespa_query_fn=small_targethits_query_fn,
            app=self.app,
            name="many-relevant-docs",
            id_field="id",
            write_csv=False,
            write_verbose=False,
        )

        # Evaluate
        results = evaluator.run()
        print("Got results: ", results)

        # Assert avg_matched_per_query is 10 (due to targetHits=10)
        self.assertEqual(results["avg_matched_per_query"], 10.0)
        # match_recall should be = 0.0
        self.assertEqual(results["match_recall"], 0.0)
        self.assertEqual(results["avg_recall_per_query"], 0.0)

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
                f"Ratio strategy test - Relevant: {relevant_count}, Random: {random_count}, Ratio: {random_count / max(relevant_count, 1):.2f}"
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

    def test_vespa_feature_collector_list_features(self):
        """
        Test VespaFeatureCollector with collect_rankfeatures.
        This tests that the collector can retrieve rank features correctly.
        """

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
            name="rankfeatures-test",
            id_field="id",
            collect_matchfeatures=False,
            collect_summaryfeatures=False,
            collect_rankfeatures=True,
            write_csv=True,
            random_hits_strategy="ratio",
            random_hits_value=1,
        )

        results = feature_collector.collect()
        print("Rank features results:", results)
        # Verify that random hits were collected but limited
        self.assertIn("results", results)
        features = results["results"]
        self.assertGreater(len(features), 0)
        # Load as dataframe from csv
        results_df = pd.read_csv(feature_collector.csv_file)

        expected_columns = [
            "query_id",
            "doc_id",
            "relevance_label",
            "relevance_score",
            "rank_attributeMatch(id)",
            "rank_attributeMatch(id).averageWeight",
            "rank_attributeMatch(id).completeness",
            "rank_attributeMatch(id).fieldCompleteness",
            "rank_attributeMatch(id).importance",
            "rank_attributeMatch(id).matches",
            "rank_attributeMatch(id).maxWeight",
            "rank_attributeMatch(id).normalizedWeight",
            "rank_attributeMatch(id).normalizedWeightedWeight",
            "rank_attributeMatch(id).queryCompleteness",
            "rank_attributeMatch(id).significance",
            "rank_attributeMatch(id).totalWeight",
            "rank_attributeMatch(id).weight",
            "rank_bm25(text)",
            "rank_elementCompleteness(text).completeness",
            "rank_elementCompleteness(text).elementWeight",
            "rank_elementCompleteness(text).fieldCompleteness",
            "rank_elementCompleteness(text).queryCompleteness",
            "rank_fieldMatch(text)",
            "rank_fieldMatch(text).absoluteOccurrence",
            "rank_fieldMatch(text).absoluteProximity",
            "rank_fieldMatch(text).completeness",
            "rank_fieldMatch(text).degradedMatches",
            "rank_fieldMatch(text).earliness",
            "rank_fieldMatch(text).fieldCompleteness",
            "rank_fieldMatch(text).gapLength",
            "rank_fieldMatch(text).gaps",
            "rank_fieldMatch(text).head",
            "rank_fieldMatch(text).importance",
            "rank_fieldMatch(text).longestSequence",
            "rank_fieldMatch(text).longestSequenceRatio",
            "rank_fieldMatch(text).matches",
            "rank_fieldMatch(text).occurrence",
            "rank_fieldMatch(text).orderness",
            "rank_fieldMatch(text).outOfOrder",
            "rank_fieldMatch(text).proximity",
            "rank_fieldMatch(text).queryCompleteness",
            "rank_fieldMatch(text).relatedness",
            "rank_fieldMatch(text).segmentDistance",
            "rank_fieldMatch(text).segmentProximity",
            "rank_fieldMatch(text).segments",
            "rank_fieldMatch(text).significance",
            "rank_fieldMatch(text).significantOccurrence",
            "rank_fieldMatch(text).tail",
            "rank_fieldMatch(text).unweightedProximity",
            "rank_fieldMatch(text).weight",
            "rank_fieldMatch(text).weightedAbsoluteOccurrence",
            "rank_fieldMatch(text).weightedOccurrence",
            "rank_fieldTermMatch(text,0).firstPosition",
            "rank_fieldTermMatch(text,0).occurrences",
            "rank_fieldTermMatch(text,0).weight",
            "rank_fieldTermMatch(text,1).firstPosition",
            "rank_fieldTermMatch(text,1).occurrences",
            "rank_fieldTermMatch(text,1).weight",
            "rank_fieldTermMatch(text,2).firstPosition",
            "rank_fieldTermMatch(text,2).occurrences",
            "rank_fieldTermMatch(text,2).weight",
            "rank_fieldTermMatch(text,3).firstPosition",
            "rank_fieldTermMatch(text,3).occurrences",
            "rank_fieldTermMatch(text,3).weight",
            "rank_fieldTermMatch(text,4).firstPosition",
            "rank_fieldTermMatch(text,4).occurrences",
            "rank_fieldTermMatch(text,4).weight",
            "rank_firstPhase",
            "rank_matches(embedding)",
            "rank_matches(id)",
            "rank_matches(text)",
            "rank_nativeAttributeMatch",
            "rank_nativeFieldMatch",
            "rank_nativeProximity",
            "rank_nativeRank",
            "rank_queryTermCount",
            "rank_term(0).connectedness",
            "rank_term(0).significance",
            "rank_term(0).weight",
            "rank_term(1).connectedness",
            "rank_term(1).significance",
            "rank_term(1).weight",
            "rank_term(2).connectedness",
            "rank_term(2).significance",
            "rank_term(2).weight",
            "rank_term(3).connectedness",
            "rank_term(3).significance",
            "rank_term(3).weight",
            "rank_term(4).connectedness",
            "rank_term(4).significance",
            "rank_term(4).weight",
            "rank_textSimilarity(text).fieldCoverage",
            "rank_textSimilarity(text).order",
            "rank_textSimilarity(text).proximity",
            "rank_textSimilarity(text).queryCoverage",
            "rank_textSimilarity(text).score",
        ]
        self.assertListEqual(results_df.columns.tolist(), expected_columns)


class TestVespaMatchEvaluatorWithURLs(unittest.TestCase):
    """
    Integration tests for VespaMatchEvaluator with URL-based document IDs.

    Purpose:
        Verify that VespaMatchEvaluator correctly handles document IDs containing special characters,
        such as those commonly found in URLs (e.g., ., ?, +, [, ], *, etc.).

    Test Coverage:
        - Feeding documents with URL-based IDs containing special characters.
        - Evaluating retrieval and matching functionality for these documents.
        - Ensuring robustness against edge cases in document ID parsing and matc
    """

    @classmethod
    def setUpClass(cls):
        # Create an application package with URL as the id field
        app_name = "urlevaluation"
        cls.package = ApplicationPackage(
            name=app_name,
            schema=[
                Schema(
                    name="urldoc",
                    document=Document(
                        fields=[
                            Field(
                                name="url",
                                type="string",
                                indexing=["attribute", "summary"],
                            ),
                            Field(
                                name="title",
                                type="string",
                                indexing=["index", "summary"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="content",
                                type="string",
                                indexing=["index", "summary"],
                                index="enable-bm25",
                            ),
                        ]
                    ),
                    fieldsets=[FieldSet(name="default", fields=["title", "content"])],
                    rank_profiles=[
                        RankProfile(
                            name="bm25",
                            first_phase="bm25(title) + bm25(content)",
                        ),
                        RankProfile(
                            name="unranked",
                            first_phase="",
                        ),
                    ],
                )
            ],
        )

        # Deploy to Docker
        cls.vespa_docker = VespaDocker(port=8090)
        cls.app = cls.vespa_docker.deploy(application_package=cls.package)

        # Feed documents with URL-based IDs containing special characters
        cls.test_docs = [
            {
                "id": "doc1",
                "fields": {
                    "url": "http://example.com/doc1",
                    "title": "GPU Gaming Guide",
                    "content": "The best GPU for gaming in 2024 is the RTX 4090.",
                },
            },
            {
                "id": "doc2",
                "fields": {
                    "url": "https://example.com/doc2",
                    "title": "Sourdough Bread Recipe",
                    "content": "How to bake sourdough bread at home with natural yeast.",
                },
            },
            {
                "id": "doc3",
                "fields": {
                    "url": "http://example.com/doc?query=1",
                    "title": "Gaming Tips",
                    "content": "Advanced gaming tips for competitive players.",
                },
            },
            {
                "id": "doc4",
                "fields": {
                    "url": "http://example.com/doc+plus",
                    "title": "GPU Benchmark",
                    "content": "GPU benchmark results for gaming performance.",
                },
            },
            {
                "id": "doc5",
                "fields": {
                    "url": "http://example.com/doc[brackets]",
                    "title": "Bread Making",
                    "content": "Professional bread making techniques.",
                },
            },
            {
                "id": "doc6",
                "fields": {
                    "url": "http://example.com/doc*star",
                    "title": "Baking Guide",
                    "content": "Complete guide to baking various types of bread.",
                },
            },
        ]

        def feed_callback(response: VespaResponse, doc_id: str):
            if not response.is_successful():
                print(f"Error feeding doc {doc_id}: {response.json}")

        cls.app.feed_iterable(cls.test_docs, schema="urldoc", callback=feed_callback)

        # Define test queries and relevant docs with URLs
        cls.queries = {
            "q1": "best GPU for gaming",
            "q2": "how to bake sourdough bread",
        }

        # Relevant docs with URL-based IDs
        cls.relevant_docs = {
            "q1": {
                "http://example.com/doc1",
                "http://example.com/doc?query=1",
                "http://example.com/doc+plus",
            },
            "q2": {
                "https://example.com/doc2",
                "http://example.com/doc[brackets]",
                "http://example.com/doc*star",
            },
        }

    @classmethod
    def tearDownClass(cls):
        # Clean up container
        cls.vespa_docker.container.stop(timeout=10)
        cls.vespa_docker.container.remove()

    def test_match_evaluator_with_url_ids(self):
        """
        Test VespaMatchEvaluator with URL-based document IDs.
        This verifies that the evaluator can correctly match documents
        when IDs contain special regex characters.
        """

        def url_query_fn(query_text: str, top_k: int) -> Dict[str, Any]:
            return {
                "yql": f'select * from sources * where userInput("{query_text}");',
                "ranking": "unranked",
                "hits": top_k,
                "timeout": "5s",
            }

        evaluator = VespaMatchEvaluator(
            queries=self.queries,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=url_query_fn,
            app=self.app,
            name="url-test",
            id_field="url",
            write_csv=True,
            write_verbose=True,
        )

        # Run evaluation
        results = evaluator.run()

        # Assertions
        self.assertIn("match_recall", results)
        self.assertIn("avg_recall_per_query", results)
        self.assertIn("total_relevant_docs", results)
        self.assertIn("total_matched_relevant", results)

        # Total relevant docs should be 6 (3 per query)
        self.assertEqual(results["total_relevant_docs"], 6)

        # We expect to match all relevant docs for both queries
        self.assertEqual(results["total_matched_relevant"], 6)

        self.assertEqual(results["match_recall"], 1.0)
        self.assertEqual(results["avg_recall_per_query"], 1.0)


class TestVespaNNParameterOptimizer(unittest.TestCase):
    """
    Integration tests for VespaNNParameterOptimizer.

    Purpose:
        Verify that VespaNNParameterOptimizer correctly interacts with a Vespa application
        and yields somewhat stable results.

    Test Coverage:
        - Construct VespaNNParameterOptimizer object with queries.
        - Call run() method to get a suggestion for all possible parameters.
    """

    @classmethod
    def setUpClass(cls):
        # Download files
        base_url = (
            "https://data.vespa-cloud.com/tests/performance/nearest-neighbor/gist-data/"
        )
        temp_dir = tempfile.TemporaryDirectory()

        def download_file(url: str):
            local_filename = os.path.join(temp_dir.name, url.split("/")[-1])
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return local_filename

        docs_path = download_file(base_url + "docs.1k.json")
        query_path = download_file(base_url + "query_vectors.10.txt")
        print(f'Downloaded document file "{docs_path}" and query file "{query_path}"')

        # Vespa application
        doc = Document(
            fields=[
                Field(
                    name="id",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="filter",
                    type="array<int>",
                    indexing=["attribute", "summary"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="vec_m16",
                    type="tensor<float>(x[960])",
                    indexing=["attribute", "index", "summary"],
                    ann=HNSW(
                        distance_metric="euclidean",
                        max_links_per_node=16,
                        neighbors_to_explore_at_insert=500,
                    ),
                ),
            ]
        )

        rank_profile = RankProfile(
            name="default",
            inputs=[
                ("query(q_vec)", "tensor<float>(x[960])"),
            ],
            first_phase="closeness(label,nns)",
        )

        minimal_summary = DocumentSummary(
            name="minimal", summary_fields=[Summary(name="id")]
        )

        schema = Schema(
            name="test",
            document=doc,
            rank_profiles=[rank_profile],
            document_summaries=[minimal_summary],
        )

        qp = query_profile(
            id="default",
            type="root",
        )

        qpt = query_profile_type(
            field(
                name="ranking.features.query(q_vec)",
                type="tensor<float>(x[960])",
            ),
            id="root",
            inherits="native",
        )

        # Create the application package
        cls.package = ApplicationPackage(
            name="test", schema=[schema], query_profile_config=[qp, qpt]
        )

        # Deploy to Docker
        print("Deploying to docker")
        cls.vespa_docker = VespaDocker(port=8090)
        cls.app = cls.vespa_docker.deploy(application_package=cls.package)

        with open(docs_path, "r") as f:
            docs = json.load(f)

        docs_formatted = [
            {
                "id": str(doc["fields"]["id"]),
                "fields": doc["fields"],
            }
            for doc in docs
        ]

        def callback(response: VespaResponse, id: str):
            if not response.is_successful():
                print(f"Error feeding doc {id}: {response.json}")

        # Feed documents
        print(f"Feeding {len(docs_formatted)} documents")
        cls.app.feed_iterable(docs_formatted, callback=callback)
        print("Finished feeding")

        # Read vectors
        with open(query_path, "r") as f:
            cls.query_vectors = f.readlines()

    @classmethod
    def tearDownClass(cls):
        # Clean up container
        cls.vespa_docker.container.stop(timeout=10)
        cls.vespa_docker.container.remove()

    def test_suggestions(self):
        def vector_to_query(vec_str: str, filter_value: int) -> dict:
            return {
                "yql": str(
                    qb.select("*")
                    .from_("test")
                    .where(
                        qb.nearestNeighbor(
                            "vec_m16",
                            "q_vec",
                            annotations={
                                "targetHits": 100,
                                "approximate": True,
                                "label": "nns",
                            },
                        )
                        & (qb.QueryField("filter") == filter_value),
                    )
                ),
                "presentation.summary": "minimal",
                "timeout": "20s",
                "ranking.features.query(q_vec)": vec_str.strip(),
            }

        print("Building queries")
        filter_percentage = [1, 10, 50, 90, 95, 99]

        queries = []
        for filter_value in filter_percentage:
            for vec in self.query_vectors:
                queries.append(vector_to_query(vec, filter_value))

        print(f"Built {len(queries)} queries")

        print("Constructing optimizer object")
        optimizer = VespaNNParameterOptimizer(
            self.app,
            queries,
            100,
            print_progress=True,
            benchmark_time_limit=1000,
            recall_query_limit=10,
        )

        print("Running optimizer")
        report = optimizer.run()
        print(json.dumps(report, sort_keys=True, indent=4))

        # With so few documents, the values we obtain do not really make sense.
        # We just check if we get numbers in [0.0,1.0]
        self.assertIn("approximateThreshold", report)
        self.assertIn("suggestion", report["approximateThreshold"])
        approximate_threshold = report["approximateThreshold"]["suggestion"]
        self.assertGreaterEqual(approximate_threshold, 0.0)
        self.assertLessEqual(approximate_threshold, 1.0)

        self.assertIn("filterFirstThreshold", report)
        self.assertIn("suggestion", report["filterFirstThreshold"])
        filter_first_threshold = report["filterFirstThreshold"]["suggestion"]
        self.assertGreaterEqual(filter_first_threshold, 0.0)
        self.assertLessEqual(filter_first_threshold, 1.0)

        self.assertIn("filterFirstExploration", report)
        self.assertIn("suggestion", report["filterFirstExploration"])
        filter_first_exploration = report["filterFirstExploration"]["suggestion"]
        self.assertGreaterEqual(filter_first_exploration, 0.0)
        self.assertLessEqual(filter_first_exploration, 1.0)

        self.assertIn("postFilterThreshold", report)
        self.assertIn("suggestion", report["postFilterThreshold"])
        post_filter_threshold = report["postFilterThreshold"]["suggestion"]
        self.assertGreaterEqual(post_filter_threshold, 0.0)
        self.assertLessEqual(post_filter_threshold, 1.0)
