# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import asyncio
import shutil
import unittest
from cryptography.hazmat.primitives import serialization
from vespa.application import Vespa
from vespa.package import AuthClient, Parameter
from vespa.deployment import VespaCloud
from vespa.test_integration_docker import (
    TestApplicationCommon,
    create_msmarco_application_package,
)

APP_INIT_TIMEOUT = 900

class TestTokenBasedAuth(unittest.TestCase):
    def setUp(self) -> None:
        self.clients = [
            AuthClient(id="mtls",
                permissions=["read", "write"],
                parameters=[
                Parameter("certificate", {"file": "security/clients.pem"})
            ]),
            AuthClient(id="token",
                permissions=["read", "write"],
                parameters=[
                Parameter("token", {"id": "pyvespa_integration_msmarco"})
            ])
        ]
        self.app_package = create_msmarco_application_package(auth_clients=self.clients)
        
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "token"
        self.app: Vespa = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        print("Endpoint used " + self.app.url) 

    def test_right_endpoint_used_with_token(self):
        # The secrect token is set in env variable.
        # The token is used to access the application status endpoint.  
        print("Endpoint used " + self.app.url)      
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


class TestMsmarcoApplicationWithTokenAuth(TestApplicationCommon):
    def setUp(self) -> None:
        
        self.clients = [
            AuthClient(id="mtls",
                permissions=["read"],
                parameters=[
                Parameter("certificate", {"file": "security/clients.pem"})
            ]),
            AuthClient(id="token",
                permissions=["read", "write"],
                parameters=[
                Parameter("token", {"id": "pyvespa_integration_msmarco"})
            ])
        ]
        
        self.app_package = create_msmarco_application_package(auth_clients=self.clients)
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "token"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        print("Endpoint used " + self.app.url)
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
