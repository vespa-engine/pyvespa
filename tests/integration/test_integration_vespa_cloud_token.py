# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import asyncio
import shutil
import unittest
import pytest
from requests import HTTPError
from vespa.application import Vespa
from datetime import datetime, timedelta

from vespa.package import (
    AuthClient,
    Parameter,
    DeploymentConfiguration,
    ContentCluster,
    ContainerCluster,
    Nodes,
    EmptyDeploymentConfiguration,
    Validation,
    ValidationID,
)
from vespa.deployment import VespaCloud
from test_integration_docker import (
    TestApplicationCommon,
    create_msmarco_application_package,
)

APP_INIT_TIMEOUT = 900
CLIENT_TOKEN_ID = os.environ.get("VESPA_CLIENT_TOKEN_ID", "pyvespa_integration_msmarco")


class TestTokenBasedAuth(unittest.TestCase):
    def setUp(self) -> None:
        self.clients = [
            AuthClient(
                id="mtls",
                permissions=["read", "write"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token",
                permissions=["read", "write"],
                parameters=[Parameter("token", {"id": CLIENT_TOKEN_ID})],
            ),
        ]
        self.app_package = create_msmarco_application_package(auth_clients=self.clients)

        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
            auth_client_token_id=CLIENT_TOKEN_ID,
        )
        self.disk_folder = os.path.join(os.getcwd(), "sample_application")
        self.instance_name = "token"
        self.mtls_app: Vespa = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        self.app = self.vespa_cloud.get_application(
            instance=self.instance_name,
            environment="dev",
            endpoint_type="token",
            vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN"),
        )
        print("Endpoint used " + self.app.url)

    def test_right_endpoint_used_with_token(self):
        # The secrect token is set in env variable.
        # The token is used to access the application status endpoint.
        print("Endpoint used " + self.app.url)
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)
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


class TestMsmarcoApplicationWithTokenAuth(TestApplicationCommon):
    def setUp(self) -> None:
        self.clients = [
            AuthClient(
                id="mtls",
                permissions=["read"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token",
                permissions=["read", "write"],
                parameters=[Parameter("token", {"id": CLIENT_TOKEN_ID})],
            ),
        ]

        self.app_package = create_msmarco_application_package(auth_clients=self.clients)
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
            auth_client_token_id=CLIENT_TOKEN_ID,
        )
        self.disk_folder = os.path.join(os.getcwd(), "sample_application")
        self.instance_name = "token"
        self.mtls_app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        self.app = self.vespa_cloud.get_application(
            instance=self.instance_name,
            environment="dev",
            endpoint_type="token",
            vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN"),
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

    def test_using_token_endpoint(self):
        endpoint = self.app.url
        auth_method = self.vespa_cloud.get_endpoint_auth_method(
            instance=self.instance_name, url=endpoint, environment="dev"
        )
        self.assertEqual(auth_method, "token")

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


class TestMsmarcoProdApplicationWithTokenAuth(TestApplicationCommon):
    def setUp(self) -> None:
        schema_name = "msmarco"
        self.auth_clients = [
            AuthClient(
                id="mtls",
                permissions=["read"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token",
                permissions=["read", "write"],
                parameters=[Parameter("token", {"id": CLIENT_TOKEN_ID})],
            ),
        ]
        self.app_package = create_msmarco_application_package()
        # Add prod deployment config
        prod_region = "aws-us-east-1c"
        self.app_package.clusters = [
            ContentCluster(
                id=f"{schema_name}_content",
                nodes=Nodes(count="2"),
                document_name=schema_name,
                min_redundancy="2",
            ),
            ContainerCluster(
                id=f"{schema_name}_container",
                nodes=Nodes(count="2"),
                auth_clients=self.auth_clients,
            ),
        ]
        self.app_package.deployment_config = DeploymentConfiguration(
            environment="prod", regions=[prod_region]
        )
        # Deploy to Vespa Cloud
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-int-prod",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
            auth_client_token_id=CLIENT_TOKEN_ID,
        )
        self.application_root = os.path.join(os.getcwd(), "sample_application")
        self.instance_name = "default"
        self.build_no = self.vespa_cloud.deploy_to_prod(
            instance=self.instance_name,
            source_url="https://github.com/vespa-engine/pyvespa",
        )
        # Define fields to send and update
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

    @unittest.skip(
        "This test is too slow for normal testing. Can be used for manual testing if related code is changed."
    )
    def test_application_status(self):
        # Wait for deployment to be ready
        success = self.vespa_cloud.wait_for_prod_deployment(
            build_no=self.build_no, max_wait=3600
        )
        if not success:
            self.fail("Deployment failed")
        self.app = self.vespa_cloud.get_application(
            instance=self.instance_name,
            environment="prod",
            endpoint_type="token",
            vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN"),
        )
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)

    @unittest.skip(
        "Can't do this without reference to the app. Save for manual testing. Consider using fixed token endpoint URL."
    )
    def test_using_token_endpoint(self):
        endpoint = self.app.url
        auth_method = self.vespa_cloud.get_endpoint_auth_method(
            url=endpoint, environment="prod"
        )
        self.assertEqual(auth_method, "token")

    @unittest.skip("See above")
    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            cluster_name=f"{self.app_package.name}_content",
            fields_to_send=self.fields_to_send[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.fields_to_send[0],
        )

    @unittest.skip("See above")
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

    @unittest.skip("See above")
    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="msmarco_content", schema="msmarco"
        )

    @unittest.skip("Not relevant when not waiting.")
    def test_delete_deployment(self):
        # Deployment is deleted by deploying with an empty deployment.xml file.
        self.app_package.deployment_config = EmptyDeploymentConfiguration()

        # Vespa won't push the deleted deployment.xml file unless we add a validation override
        tomorrow = datetime.now() + timedelta(days=1)
        formatted_date = tomorrow.strftime("%Y-%m-%d")
        self.app_package.validations = [
            Validation(ValidationID("deployment-removal"), formatted_date)
        ]
        self.app_package.to_files(self.application_root)
        # This will delete the deployment
        self.vespa_cloud._start_prod_deployment(self.application_root)
        shutil.rmtree(self.application_root, ignore_errors=True)
