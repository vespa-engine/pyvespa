import unittest
import os
import shutil
from random import random, seed
from vespa.package import (
    Document,
    Field,
    Schema,
    FieldSet,
    QueryTypeField,
    SecondPhaseRanking,
    RankProfile,
    ApplicationPackage,
    VespaCloud,
)
from vespa.ml import BertModelConfig
from vespa.query import QueryModel, RankProfile as Ranking, OR, QueryRankingFeature, ANN


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

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)


class TestVectorSearchDeployment(unittest.TestCase):
    def setUp(self) -> None:
        #
        # Create application package
        #
        self.app_package = ApplicationPackage(name="productsearch")
        self.app_package.schema.add_fields(
            Field(name="asin", type="string", indexing=["attribute", "summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="description",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(name="price", type="float", indexing=["attribute", "summary"]),
            Field(
                name="image_vector",
                type="tensor<float>(x[4096])",
                indexing=["attribute"],
            ),
            Field(
                name="reduced_image_vector",
                type="tensor<float>(x[256])",
                indexing=["attribute"],
            ),
        )
        self.app_package.query_profile_type.add_fields(
            QueryTypeField(
                name="ranking.features.query(reduced_image_vector)",
                type="tensor<float>(x[256])",
            )
        )
        self.app_package.schema.add_field_set(
            FieldSet(name="default", fields=["title", "description"])
        )
        self.app_package.schema.add_rank_profile(
            RankProfile(name="bm25", first_phase="bm25(title) + bm25(description)")
        )
        self.app_package.schema.add_rank_profile(
            RankProfile(
                name="dot_product",
                first_phase="sum(query(reduced_image_vector)*attribute(reduced_image_vector))",
            )
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")

    def test_deploy_feed_query(self):
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
        self.assertEqual(self.app.get_application_status().status_code, 200)

        seed(345)
        product_data = [
            {
                "asin": "1",
                "title": "watch",
                "description": "to check the time",
                "price": 120.35,
                "image_vector": {"values": [random() for x in range(4096)]},
                "reduced_image_vector": {"values": [random() for x in range(256)]},
            },
            {
                "asin": "2",
                "title": "chair",
                "description": "object to seat",
                "price": 39.90,
                "image_vector": {"values": [random() for x in range(4096)]},
                "reduced_image_vector": {"values": [random() for x in range(256)]},
            },
            {
                "asin": "3",
                "title": "table",
                "description": "to eat dinner on",
                "price": 52.85,
                "image_vector": {"values": [random() for x in range(4096)]},
                "reduced_image_vector": {"values": [random() for x in range(256)]},
            },
        ]
        for data in product_data:
            self.app.feed_data_point(
                schema="productsearch", data_id=data["asin"], fields=data
            )

        or_model = QueryModel(match_phase=OR(), rank_profile=Ranking(name="bm25"))
        res = self.app.query(query="men's watch", query_model=or_model, tracelevel=3)
        self.assertEqual(res.hits[0]["id"], "id:productsearch:productsearch::1")

        nn_model = QueryModel(
            match_phase=ANN(
                doc_vector="reduced_image_vector",
                query_vector="reduced_image_vector",
                hits=1000,
                label="nn",
            ),
            rank_profile=Ranking(name="dot_product"),
        )
        vector_to_search = product_data[0]["reduced_image_vector"]["values"]

        res = self.app.query(
            query_properties=[
                QueryRankingFeature(name="reduced_image_vector", value=vector_to_search)
            ],
            query_model=nn_model,
        )
        self.assertEqual(len(res.hits), 3)

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
