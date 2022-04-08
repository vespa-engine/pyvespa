import os
import asyncio
import shutil
import json
import unittest
from pandas import DataFrame
from cryptography.hazmat.primitives import serialization
from vespa.application import Vespa
from vespa.query import QueryModel, OR
from vespa.gallery import TextSearch
from vespa.deployment import VespaCloud
from vespa.test_integration_docker import (
    TestApplicationCommon,
    create_msmarco_application_package,
    create_cord19_application_package,
    create_qa_application_package,
    create_sequence_classification_task,
)

CFG_SERVER_START_TIMEOUT = 300
APP_INIT_TIMEOUT = 300

class TestVespaKeyAndCertificate(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "msmarco"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )

    def test_key_cert_arguments(self):
        #
        # Write key and cert to different files
        #
        with open(os.path.join(self.disk_folder, "key_file.txt"), "w+") as file:
            file.write(
                self.vespa_cloud.data_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption(),
                ).decode("UTF-8")
            )
        with open(os.path.join(self.disk_folder, "cert_file.txt"), "w+") as file:
            file.write(
                self.vespa_cloud.data_certificate.public_bytes(
                    serialization.Encoding.PEM
                ).decode("UTF-8")
            )
        self.app = Vespa(
            url=self.app.url,
            key=os.path.join(self.disk_folder, "key_file.txt"),
            cert=os.path.join(self.disk_folder, "cert_file.txt"),
            application_package=self.app.application_package,
        )
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)
        self.assertEqual(200, self.app.get_application_status().status_code)
        self.assertDictEqual(
            {
                "pathId": "/document/v1/msmarco/msmarco/docid/1",
                "id": "id:msmarco:msmarco::1",
            },
            self.app.get_batch(batch=[{"id": 1}])[0].json,
        )

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="msmarco_content", schema="msmarco"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)


class TestMsmarcoApplication(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "msmarco"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        self.fields_to_send = [
            {
                "id": f"{i}",
                "title": f"this is title {i}",
                "body": f"this is body {i}",
            }
            for i in range(10)
        ]
        self.fields_to_update = [
            {
                "id": f"{i}",
                "title": "this is my updated title number {}".format(i),
            }
            for i in range(10)
        ]

    def test_model_endpoints_when_no_model_is_available(self):
        # The port should not be 4443, see https://jira.vzbuilders.com/browse/VESPA-21365
        self.get_model_endpoints_when_no_model_is_available(
            app=self.app,
            expected_model_endpoint="https://{}-container.{}.pyvespa-integration.vespa-team.aws-us-east-1c.dev.z.vespa-app.cloud:4443/model-evaluation/v1/".format(
                self.app_package.name, self.instance_name
            ),
        )

    def test_prediction_when_model_not_defined(self):
        self.get_stateless_prediction_when_model_not_defined(
            app=self.app, application_package=self.app_package
        )

    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.fields_to_send[0],
        )

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_data_operations(
                app=self.app,
                schema_name=self.app_package.name,
                fields_to_send=self.fields_to_send,
                field_to_update=self.fields_to_update[0],
                expected_fields_from_get_operation=self.fields_to_send,
            )
        )

    def test_batch_operations_synchronous_mode(self):
        self.batch_operations_synchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.fields_to_send,
            fields_to_update=self.fields_to_update,
        )

    def test_batch_operations_asynchronous_mode(self):
        self.batch_operations_asynchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.fields_to_send,
            fields_to_update=self.fields_to_update,
        )

    def test_batch_operations_default_mode_with_one_schema(self):
        self.batch_operations_default_mode_with_one_schema(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.fields_to_send,
            fields_to_update=self.fields_to_update,
        )

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="msmarco_content", schema="msmarco"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)


class TestCord19Application(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_cord19_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "cord19"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        self.model_config = self.app_package.model_configs["pretrained_bert_tiny"]
        self.fields_to_send = []
        self.expected_fields_from_get_operation = []
        for i in range(10):
            fields = {
                "id": f"{i}",
                "title": f"this is title {i}",
            }
            tensor_field_dict = self.model_config.doc_fields(text=str(fields["title"]))
            fields.update(tensor_field_dict)
            self.fields_to_send.append(fields)

            expected_fields = {
                "id": f"{i}",
                "title": f"this is title {i}",
            }
            tensor_field_values = tensor_field_dict[
                "pretrained_bert_tiny_doc_token_ids"
            ]["values"]
            expected_fields.update(
                {
                    "pretrained_bert_tiny_doc_token_ids": {
                        "cells": [
                            {
                                "address": {"d0": str(x)},
                                "value": float(tensor_field_values[x]),
                            }
                            for x in range(len(tensor_field_values))
                        ]
                    }
                }
            )
            self.expected_fields_from_get_operation.append(expected_fields)
        self.fields_to_update = [
            {
                "id": f"{i}",
                "title": "this is my updated title number {}".format(i),
            }
            for i in range(10)
        ]

    def test_model_endpoints_when_no_model_is_available(self):
        # The port should not be 4443, see https://jira.vzbuilders.com/browse/VESPA-21365
        self.get_model_endpoints_when_no_model_is_available(
            app=self.app,
            expected_model_endpoint="https://{}-container.{}.pyvespa-integration.vespa-team.aws-us-east-1c.dev.z.vespa-app.cloud:4443/model-evaluation/v1/".format(
                self.app_package.name, self.instance_name
            ),
        )

    def test_prediction_when_model_not_defined(self):
        self.get_stateless_prediction_when_model_not_defined(
            app=self.app, application_package=self.app_package
        )

    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.expected_fields_from_get_operation[
                0
            ],
        )

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_data_operations(
                app=self.app,
                schema_name=self.app_package.name,
                fields_to_send=self.fields_to_send,
                field_to_update=self.fields_to_update[0],
                expected_fields_from_get_operation=self.expected_fields_from_get_operation,
            )
        )

    def test_batch_operations_synchronous_mode(self):
        self.batch_operations_synchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.expected_fields_from_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def test_batch_operations_asynchronous_mode(self):
        self.batch_operations_asynchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.expected_fields_from_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def test_batch_operations_default_mode_with_one_schema(self):
        self.batch_operations_default_mode_with_one_schema(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.expected_fields_from_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def test_bert_model_input_and_output(self):
        self.bert_model_input_and_output(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            model_config=self.model_config,
        )

    def tearDown(self) -> None:
        self.app.delete_all_docs(content_cluster_name="cord19_content", schema="cord19")
        shutil.rmtree(self.disk_folder, ignore_errors=True)


class TestQaApplication(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_qa_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "qa"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        with open(
            os.path.join(os.environ["RESOURCES_DIR"], "qa_sample_sentence_data.json"),
            "r",
        ) as f:
            sample_sentence_data = json.load(f)
        self.fields_to_send_sentence = sample_sentence_data
        self.expected_fields_from_sentence_get_operation = []
        for d in sample_sentence_data:
            expected_d = {
                "id": d["id"],
                "text": d["text"],
                "dataset": d["dataset"],
                "context_id": d["context_id"],
                "sentence_embedding": {
                    "cells": [
                        {"address": {"x": str(idx)}, "value": value}
                        for idx, value in enumerate(d["sentence_embedding"]["values"])
                    ]
                },
            }
            if len(d["questions"]) > 0:
                expected_d.update({"questions": d["questions"]})
            self.expected_fields_from_sentence_get_operation.append(expected_d)
        with open(
            os.path.join(os.environ["RESOURCES_DIR"], "qa_sample_context_data.json"),
            "r",
        ) as f:
            sample_context_data = json.load(f)
        self.fields_to_send_context = sample_context_data
        self.fields_to_update = [
            {"id": d["id"], "text": "this is my updated text number {}".format(d["id"])}
            for d in self.fields_to_send_sentence
        ]

    def test_model_endpoints_when_no_model_is_available(self):
        # The port should not be 4443, see https://jira.vzbuilders.com/browse/VESPA-21365
        self.get_model_endpoints_when_no_model_is_available(
            app=self.app,
            expected_model_endpoint="https://{}-container.{}.pyvespa-integration.vespa-team.aws-us-east-1c.dev.z.vespa-app.cloud:4443/model-evaluation/v1/".format(
                self.app_package.name, self.instance_name
            ),
        )

    def test_prediction_when_model_not_defined(self):
        self.get_stateless_prediction_when_model_not_defined(
            app=self.app, application_package=self.app_package
        )

    def test_execute_data_operations_sentence_schema(self):
        self.execute_data_operations(
            app=self.app,
            schema_name="sentence",
            fields_to_send=self.fields_to_send_sentence[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.expected_fields_from_sentence_get_operation[
                0
            ],
        )

    def test_execute_data_operations_context_schema(self):
        self.execute_data_operations(
            app=self.app,
            schema_name="context",
            fields_to_send=self.fields_to_send_context[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.fields_to_send_context[0],
        )

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_data_operations(
                app=self.app,
                schema_name="sentence",
                fields_to_send=self.fields_to_send_sentence,
                field_to_update=self.fields_to_update[0],
                expected_fields_from_get_operation=self.expected_fields_from_sentence_get_operation,
            )
        )

    def test_batch_operations_synchronous_mode(self):
        self.batch_operations_synchronous_mode(
            app=self.app,
            schema_name="sentence",
            fields_to_send=self.fields_to_send_sentence,
            expected_fields_from_get_operation=self.expected_fields_from_sentence_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def test_batch_operations_asynchronous_mode(self):
        self.batch_operations_asynchronous_mode(
            app=self.app,
            schema_name="sentence",
            fields_to_send=self.fields_to_send_sentence,
            expected_fields_from_get_operation=self.expected_fields_from_sentence_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def tearDown(self) -> None:
        self.app.delete_all_docs(content_cluster_name="qa_content", schema="sentence")
        self.app.delete_all_docs(content_cluster_name="qa_content", schema="context")
        shutil.rmtree(self.disk_folder, ignore_errors=True)


class TestGalleryTextSearch(unittest.TestCase):
    def setUp(self) -> None:
        #
        # Create application
        #
        self.app_package = TextSearch(id_field="id", text_fields=["title", "body"])
        #
        # Deploy application
        #
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "text-search"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        #
        # Create a sample data frame
        #
        records = [
            {
                "id": idx,
                "title": "This doc is about {}".format(x),
                "body": "There is so much to learn about {}".format(x),
            }
            for idx, x in enumerate(
                ["finance", "sports", "celebrity", "weather", "politics"]
            )
        ]
        df = DataFrame.from_records(records)
        #
        # Feed application
        #
        self.app.feed_df(df)

    def test_query(self):
        result = self.app.query(
            query="what is finance?", query_model=QueryModel(match_phase=OR())
        )
        for hit in result.hits:
            self.assertIn("fields", hit)

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="text_search_content", schema="text_search"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)


class TestSequenceClassification(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_sequence_classification_task()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "classification-task"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )

    def test_model_endpoints(self):
        self.get_model_endpoints(
            app=self.app,
            expected_model_endpoint="https://{}-container.{}.pyvespa-integration.vespa-team.aws-us-east-1c.dev.z.vespa-app.cloud/model-evaluation/v1/".format(
                self.app_package.name, self.instance_name
            ),
        )

    def test_prediction(self):
        self.get_stateless_prediction(
            app=self.app, application_package=self.app_package
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
