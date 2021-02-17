import unittest
import os
import shutil
from vespa.package import (
    HNSW,
    Document,
    Field,
    Schema,
    FieldSet,
    SecondPhaseRanking,
    RankProfile,
    ApplicationPackage,
    VespaCloud,
)
from vespa.ml import BertModelConfig
from vespa.query import QueryModel, RankProfile as Ranking, OR, QueryRankingFeature


class TestCloudDeployment(unittest.TestCase):
    def setUp(self) -> None:
        #
        # Create application package
        #
        document = Document(
            fields=[
                Field(name="id", type="string", indexing=["attribute", "summary"]),
                Field(
                    name="title",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="body",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="metadata",
                    type="string",
                    indexing=["attribute", "summary"],
                    attribute=["fast-search", "fast-access"],
                ),
                Field(
                    name="tensor_field",
                    type="tensor<float>(x[128])",
                    indexing=["attribute"],
                    ann=HNSW(
                        distance_metric="euclidean",
                        max_links_per_node=16,
                        neighbors_to_explore_at_insert=200,
                    ),
                ),
            ]
        )
        msmarco_schema = Schema(
            name="msmarco",
            document=document,
            fieldsets=[FieldSet(name="default", fields=["title", "body"])],
            rank_profiles=[
                RankProfile(name="default", first_phase="nativeRank(title, body)")
            ],
        )
        app_package = ApplicationPackage(name="msmarco", schema=msmarco_schema)
        #
        # Deploy on Vespa Cloud
        #
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
            application_package=app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "test"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )

    def test_data_operation(self):
        #
        # Make sure we start with no data with id = 1
        #
        _ = self.app.delete_data(schema="msmarco", data_id="1")
        #
        # Get data that does not exist
        #
        self.assertEqual(
            self.app.get_data(schema="msmarco", data_id="1").status_code, 404
        )
        #
        # Feed a data point
        #
        response = self.app.feed_data_point(
            schema="msmarco",
            data_id="1",
            fields={
                "id": "1",
                "title": "this is my first title",
                "body": "this is my first body",
            },
        )
        self.assertEqual(response.json()["id"], "id:msmarco:msmarco::1")
        #
        # Get data that exist
        #
        response = self.app.get_data(schema="msmarco", data_id="1")
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(
            response.json(),
            {
                "fields": {
                    "id": "1",
                    "title": "this is my first title",
                    "body": "this is my first body",
                },
                "id": "id:msmarco:msmarco::1",
                "pathId": "/document/v1/msmarco/msmarco/docid/1",
            },
        )
        #
        # Update data
        #
        response = self.app.update_data(
            schema="msmarco", data_id="1", fields={"title": "this is my updated title"}
        )
        self.assertEqual(response.json()["id"], "id:msmarco:msmarco::1")
        #
        # Get the updated data point
        #
        response = self.app.get_data(schema="msmarco", data_id="1")
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(
            response.json(),
            {
                "fields": {
                    "id": "1",
                    "title": "this is my updated title",
                    "body": "this is my first body",
                },
                "id": "id:msmarco:msmarco::1",
                "pathId": "/document/v1/msmarco/msmarco/docid/1",
            },
        )
        #
        # Delete a data point
        #
        response = self.app.delete_data(schema="msmarco", data_id="1")
        self.assertEqual(response.json()["id"], "id:msmarco:msmarco::1")
        #
        # Deleted data should be gone
        #
        self.assertEqual(
            self.app.get_data(schema="msmarco", data_id="1").status_code, 404
        )
        #
        # Update a non-existent data point
        #
        response = self.app.update_data(
            schema="msmarco",
            data_id="1",
            fields={"title": "this is my updated title"},
            create=True,
        )
        self.assertEqual(response.json()["id"], "id:msmarco:msmarco::1")
        #
        # Get the updated data point
        #
        response = self.app.get_data(schema="msmarco", data_id="1")
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(
            response.json(),
            {
                "fields": {
                    "title": "this is my updated title",
                },
                "id": "id:msmarco:msmarco::1",
                "pathId": "/document/v1/msmarco/msmarco/docid/1",
            },
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)


class TestOnnxModelCloudDeployment(unittest.TestCase):
    def setUp(self) -> None:
        #
        # Create application package
        #
        self.app_package = ApplicationPackage(name="cord19")
        self.app_package.schema.add_fields(
            Field(name="cord_uid", type="string", indexing=["attribute", "summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
        )
        self.app_package.schema.add_field_set(
            FieldSet(name="default", fields=["title"])
        )
        self.app_package.schema.add_rank_profile(
            RankProfile(name="bm25", first_phase="bm25(title)")
        )
        self.bert_config = BertModelConfig(
            model_id="pretrained_bert_tiny",
            tokenizer="google/bert_uncased_L-2_H-128_A-2",
            model="google/bert_uncased_L-2_H-128_A-2",
            query_input_size=5,
            doc_input_size=10,
        )
        self.app_package.add_model_ranking(
            model_config=self.bert_config,
            include_model_summary_features=True,
            inherits="default",
            first_phase="bm25(title)",
            second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
        )
        #
        # Deploy on Vespa Cloud
        #
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "test"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )

    def test_data_operation(self):
        #
        # Get data that does not exist
        #
        self.assertEqual(
            self.app.get_data(schema="cord19", data_id="1").status_code, 404
        )
        #
        # Feed a data point
        #
        fields = {
            "cord_uid": "1",
            "title": "this is my first title",
        }
        fields.update(self.bert_config.doc_fields(text=str(fields["title"])))
        response = self.app.feed_data_point(
            schema="cord19",
            data_id="1",
            fields=fields,
        )
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Get data that exist
        #
        response = self.app.get_data(schema="cord19", data_id="1")
        self.assertEqual(response.status_code, 200)
        embedding_values = fields["pretrained_bert_tiny_doc_token_ids"]["values"]
        self.assertDictEqual(
            response.json(),
            {
                "fields": {
                    "cord_uid": "1",
                    "title": "this is my first title",
                    "pretrained_bert_tiny_doc_token_ids": {
                        "cells": [
                            {
                                "address": {"d0": str(x)},
                                "value": float(embedding_values[x]),
                            }
                            for x in range(len(embedding_values))
                        ]
                    },
                },
                "id": "id:cord19:cord19::1",
                "pathId": "/document/v1/cord19/cord19/docid/1",
            },
        )
        #
        # Update data
        #
        fields = {"title": "this is my updated title"}
        fields.update(self.bert_config.doc_fields(text=str(fields["title"])))
        response = self.app.update_data(schema="cord19", data_id="1", fields=fields)
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Get the updated data point
        #
        response = self.app.get_data(schema="cord19", data_id="1")
        self.assertEqual(response.status_code, 200)
        embedding_values = fields["pretrained_bert_tiny_doc_token_ids"]["values"]
        self.assertDictEqual(
            response.json(),
            {
                "fields": {
                    "cord_uid": "1",
                    "title": "this is my updated title",
                    "pretrained_bert_tiny_doc_token_ids": {
                        "cells": [
                            {
                                "address": {"d0": str(x)},
                                "value": float(embedding_values[x]),
                            }
                            for x in range(len(embedding_values))
                        ]
                    },
                },
                "id": "id:cord19:cord19::1",
                "pathId": "/document/v1/cord19/cord19/docid/1",
            },
        )
        #
        # Delete a data point
        #
        response = self.app.delete_data(schema="cord19", data_id="1")
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Deleted data should be gone
        #
        self.assertEqual(
            self.app.get_data(schema="cord19", data_id="1").status_code, 404
        )

    def _parse_vespa_tensor(self, hit, feature):
        return [x["value"] for x in hit["fields"]["summaryfeatures"][feature]["cells"]]

    def test_rank_input_output(self):
        #
        # Feed a data point
        #
        fields = {
            "cord_uid": "1",
            "title": "this is my first title",
        }
        fields.update(self.bert_config.doc_fields(text=str(fields["title"])))
        response = self.app.feed_data_point(
            schema="cord19",
            data_id="1",
            fields=fields,
        )
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Run a test query
        #
        result = self.app.query(
            query="this is a test",
            query_model=QueryModel(
                query_properties=[
                    QueryRankingFeature(
                        name=self.bert_config.query_token_ids_name,
                        mapping=self.bert_config.query_tensor_mapping,
                    )
                ],
                match_phase=OR(),
                rank_profile=Ranking(name="pretrained_bert_tiny"),
            ),
        )
        vespa_input_ids = self._parse_vespa_tensor(
            result.hits[0], "rankingExpression(input_ids)"
        )
        vespa_attention_mask = self._parse_vespa_tensor(
            result.hits[0], "rankingExpression(attention_mask)"
        )
        vespa_token_type_ids = self._parse_vespa_tensor(
            result.hits[0], "rankingExpression(token_type_ids)"
        )

        expected_inputs = self.bert_config.create_encodings(
            queries=["this is a test"], docs=["this is my first title"]
        )
        self.assertEqual(vespa_input_ids, expected_inputs["input_ids"][0])
        self.assertEqual(vespa_attention_mask, expected_inputs["attention_mask"][0])
        self.assertEqual(vespa_token_type_ids, expected_inputs["token_type_ids"][0])

        expected_logits = self.bert_config.predict(
            queries=["this is a test"], docs=["this is my first title"]
        )
        self.assertAlmostEqual(
            result.hits[0]["fields"]["summaryfeatures"]["rankingExpression(logit0)"],
            expected_logits[0][0],
            5,
        )
        self.assertAlmostEqual(
            result.hits[0]["fields"]["summaryfeatures"]["rankingExpression(logit1)"],
            expected_logits[0][1],
            5,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
