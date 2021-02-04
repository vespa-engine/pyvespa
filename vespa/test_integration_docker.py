import unittest
import os
import re
import shutil
from vespa.package import (
    Document,
    Field,
    Schema,
    FieldSet,
    SecondPhaseRanking,
    RankProfile,
    ApplicationPackage,
    VespaDocker,
)
from vespa.ml import BertModelConfig


class TestDockerDeployment(unittest.TestCase):
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
        self.app_package = ApplicationPackage(name="msmarco", schema=msmarco_schema)
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")

    def test_deploy(self):
        self.vespa_docker = VespaDocker(port=8089)
        app = self.vespa_docker.deploy(
            application_package=self.app_package, disk_folder=self.disk_folder
        )
        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )
        self.assertEqual(app.get_application_status().status_code, 200)

    def test_data_operation(self):
        self.vespa_docker = VespaDocker(port=8089)
        app = self.vespa_docker.deploy(
            application_package=self.app_package, disk_folder=self.disk_folder
        )
        #
        # Get data that does not exist
        #
        self.assertEqual(app.get_data(schema="msmarco", data_id="1").status_code, 404)
        #
        # Feed a data point
        #
        response = app.feed_data_point(
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
        response = app.get_data(schema="msmarco", data_id="1")
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
        response = app.update_data(
            schema="msmarco", data_id="1", fields={"title": "this is my updated title"}
        )
        self.assertEqual(response.json()["id"], "id:msmarco:msmarco::1")
        #
        # Get the updated data point
        #
        response = app.get_data(schema="msmarco", data_id="1")
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
        response = app.delete_data(schema="msmarco", data_id="1")
        self.assertEqual(response.json()["id"], "id:msmarco:msmarco::1")
        #
        # Deleted data should be gone
        #
        self.assertEqual(app.get_data(schema="msmarco", data_id="1").status_code, 404)

    def test_deploy_from_disk(self):
        self.vespa_docker = VespaDocker(port=8089)
        self.vespa_docker.export_application_package(
            dir_path=self.disk_folder, application_package=self.app_package
        )
        app = self.vespa_docker.deploy_from_disk(
            application_name=self.app_package.name, disk_folder=self.disk_folder
        )

        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()


class TestOnnxModelDockerDeployment(unittest.TestCase):
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
            query_input_size=32,
            doc_input_size=96,
        )
        self.app_package.add_model_ranking(
            model_config=self.bert_config,
            inherits="default",
            first_phase="bm25(title)",
            second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")

    def test_deploy(self):
        self.vespa_docker = VespaDocker(port=8089)
        app = self.vespa_docker.deploy(
            application_package=self.app_package, disk_folder=self.disk_folder
        )
        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )
        self.assertEqual(app.get_application_status().status_code, 200)

    def test_data_operation(self):
        self.vespa_docker = VespaDocker(port=8089)
        app = self.vespa_docker.deploy(
            application_package=self.app_package, disk_folder=self.disk_folder
        )
        #
        # Get data that does not exist
        #
        self.assertEqual(app.get_data(schema="cord19", data_id="1").status_code, 404)
        #
        # Feed a data point
        #
        fields = {
            "cord_uid": "1",
            "title": "this is my first title",
        }
        fields.update(self.bert_config.doc_fields(text=str(fields["title"])))
        response = app.feed_data_point(
            schema="cord19",
            data_id="1",
            fields=fields,
        )
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Get data that exist
        #
        response = app.get_data(schema="cord19", data_id="1")
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
        response = app.update_data(schema="cord19", data_id="1", fields=fields)
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Get the updated data point
        #
        response = app.get_data(schema="cord19", data_id="1")
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
        response = app.delete_data(schema="cord19", data_id="1")
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Deleted data should be gone
        #
        self.assertEqual(app.get_data(schema="cord19", data_id="1").status_code, 404)

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()
