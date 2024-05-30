# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import asyncio
import shutil
import pytest
from requests import HTTPError
import unittest
from cryptography.hazmat.primitives import serialization
from vespa.application import Vespa
from vespa.deployment import VespaCloud
from vespa.package import ApplicationPackage, Schema, Document, Field
import random
from vespa.io import VespaResponse
from test_integration_docker import (
    TestApplicationCommon,
    create_msmarco_application_package,
)

APP_INIT_TIMEOUT = 900


class TestVespaKeyAndCertificate(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getcwd(), "sample_application")
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
            self.app.get_data(schema="msmarco", data_id="1").json,
        )
        self.assertEqual(
            self.app.get_data(schema="msmarco", data_id="1").is_successful(), False
        )
        with pytest.raises(HTTPError):
            self.app.get_data(schema="msmarco", data_id="1", raise_on_not_found=True)

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="msmarco_content", schema="msmarco"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_cloud.delete(instance=self.instance_name)


class TestMsmarcoApplication(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getcwd(), "sample_application")
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

    def test_prediction_when_model_not_defined(self):
        self.get_stateless_prediction_when_model_not_defined(
            app=self.app, application_package=self.app_package
        )

    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            cluster_name=f"{self.app_package.name}_content",
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

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="msmarco_content", schema="msmarco"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_cloud.delete(instance=self.instance_name)


class TestRetryApplication(unittest.TestCase):
    """
    Test class that simulates slow docprocessing which causes 429 errors while feeding documents to Vespa.

    Through the CustomHTTPAdapter in `application.py`, these 429 errors will be retried with an exponential backoff strategy,
    and hence not be returned to the user.

    The slow docprocessing is simulated by setting the `sleep` indexing option on the `latency` field, which then will sleep
    according to the value of the field when processing the document.
    """

    def setUp(self) -> None:
        document = Document(
            fields=[
                Field(name="id", type="string", indexing=["attribute", "summary"]),
                Field(
                    name="latency",
                    type="double",
                    indexing=["attribute", "summary", "sleep"],
                ),
            ]
        )
        schema = Schema(
            name="retryapplication",
            document=document,
        )
        self.app_package = ApplicationPackage(name="retryapplication", schema=[schema])
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getcwd(), "sample_application")
        self.instance_name = "retryapplication"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )

    def doc_generator(self, num_docs: int):
        for i in range(num_docs):
            yield {
                "id": str(i),
                "fields": {
                    "id": str(i),
                    "latency": random.uniform(3, 4),
                },
            }

    def test_retry(self):
        num_docs = 10
        num_429 = 0

        def callback(response: VespaResponse, id: str):
            nonlocal num_429
            if response.status_code == 429:
                num_429 += 1

        self.assertEqual(num_429, 0)
        self.app.feed_iterable(
            self.doc_generator(num_docs),
            schema="retryapplication",
            callback=callback,
        )
        print(f"Number of 429 responses: {num_429}")
        total_docs = []
        for doc_slice in self.app.visit(
            content_cluster_name="retryapplication_content",
            schema="retryapplication",
            namespace="retryapplication",
            selection="true",
        ):
            for response in doc_slice:
                total_docs.extend(response.documents)
        self.assertEqual(len(total_docs), num_docs)

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="retryapplication_content", schema="retryapplication"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_cloud.delete(instance=self.instance_name)
