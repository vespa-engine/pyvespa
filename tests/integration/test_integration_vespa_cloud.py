# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import httpx
import asyncio
import shutil
import pytest
from requests import HTTPError
import unittest
from cryptography.hazmat.primitives import serialization
from tempfile import TemporaryDirectory
from vespa.application import Vespa
from vespa.deployment import VespaCloud
from vespa.package import (
    ApplicationPackage,
    Schema,
    Document,
    Field,
)
import vespa
import random
import time
from vespa.io import VespaResponse
from test_integration_docker import (
    TestApplicationCommon,
    create_msmarco_application_package,
)
import pathlib
from datetime import datetime, timedelta

from vespa.package import (
    EmptyDeploymentConfiguration,
    Validation,
    ValidationID,
    sample_package,
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

    def test_control_plane_useragent(self):
        response: httpx.Response = self.vespa_cloud._request_with_api_key(
            "GET",
            f"/application/v4/tenant/{self.vespa_cloud.tenant}/application/{self.vespa_cloud.application}/",
            return_raw_response=True,
        )
        self.assertEqual(
            response.request.headers["User-Agent"],
            f"pyvespa/{vespa.__version__}",
        )

    def test_data_plane_useragent_sync(self):
        with self.app.syncio() as session:
            response = session.http_session.get(
                self.app.end_point + "/ApplicationStatus"
            )
        self.assertEqual(
            response.request.headers["User-Agent"],
            f"pyvespa/{vespa.__version__}",
        )

    def test_data_plane_useragent_async(self):
        async def get_resp():
            async with self.app.asyncio() as session:
                response = await session.httpx_client.get(
                    self.app.end_point + "/ApplicationStatus"
                )
            return response

        response = asyncio.run(get_resp())
        self.assertEqual(
            response.request.headers["User-Agent"],
            f"pyvespa/{vespa.__version__}",
        )

    def test_is_using_http2_client(self):
        asyncio.run(self.async_is_http2_client(app=self.app))

    def test_handle_longlived_connection(self):
        asyncio.run(self.handle_longlived_connection(app=self.app))

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

    def test_get_app_package_contents(self):
        contents = self.vespa_cloud.get_app_package_contents(
            instance="msmarco", region="aws-us-east-1c", environment="dev"
        )
        self.assertIsInstance(contents, list)
        self.assertTrue(
            any(url.endswith(".sd") for url in contents), "No .sd files found."
        )
        self.assertTrue(
            any(url.endswith(".xml") for url in contents), "No .xml files found."
        )

    def test_get_schemas(self):
        schemas = self.vespa_cloud.get_schemas(
            instance="msmarco", region="aws-us-east-1c", environment="dev"
        )
        self.assertIsInstance(schemas, dict)
        self.assertTrue(
            all(isinstance(schema_content, str) for schema_content in schemas.values())
        )
        self.assertTrue(
            all(isinstance(schema_name, str) for schema_name in schemas.keys())
        )

    def test_download_app_package_content(self):
        with TemporaryDirectory() as temp_dir:
            self.vespa_cloud.download_app_package_content(
                temp_dir, instance="msmarco", region="aws-us-east-1c", environment="dev"
            )
            expected_file = os.path.join(temp_dir, "schemas", "msmarco.sd")
            self.assertTrue(
                os.path.exists(expected_file),
                f"msmarco.sd not downloaded. Only downloaded: {os.listdir(temp_dir)}",
            )


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


class TestDeployProdWithTests(unittest.TestCase):
    def setUp(self) -> None:
        # Set root to parent directory/testapps/production-deployment-with-tests
        self.application_root = (
            pathlib.Path(__file__).parent.parent
            / "testapps"
            / "production-deployment-with-tests"
        )
        # Vespa won't deploy without validation override for certificate-removal
        tomorrow = datetime.now() + timedelta(days=1)
        formatted_date = tomorrow.strftime("%Y-%m-%d")
        app_package = ApplicationPackage(name="empty")
        app_package.validations = [
            Validation(ValidationID("certificate-removal"), until=formatted_date)
        ]
        # Write validations_to_text to "validation-overrides.xml"
        with open(self.application_root / "validation-overrides.xml", "w") as f:
            f.write(app_package.validations_to_text)
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_root=self.application_root,
        )

        self.build_no = self.vespa_cloud.deploy_to_prod(
            source_url="https://github.com/vespa-engine/pyvespa",
        )

    @unittest.skip(
        "This test is too slow for normal testing. Can be used for manual testing if related code is changed."
    )
    def test_application_status(self):
        # Wait for deployment to be ready
        success = self.vespa_cloud.wait_for_prod_deployment(
            build_no=self.build_no, max_wait=3600 * 4
        )
        if not success:
            self.fail("Deployment failed")
        self.app = self.vespa_cloud.get_application(environment="prod")

    @unittest.skip("Can not tearDown when not waiting for application")
    def tearDown(self) -> None:
        # Deployment is deleted by deploying with an empty deployment.xml file
        # Creating a dummy ApplicationPackage just to use the validation_to_text method
        app_package = ApplicationPackage(name="empty")
        # Vespa won't push the deleted deployment.xml file unless we add a validation override
        tomorrow = datetime.now() + timedelta(days=1)
        formatted_date = tomorrow.strftime("%Y-%m-%d")
        app_package.validations = [
            Validation(ValidationID("deployment-removal"), formatted_date)
        ]
        # Write validations_to_text to "validation-overrides.xml"
        with open(self.application_root / "validation-overrides.xml", "w") as f:
            f.write(app_package.validations_to_text)
        # Create an empty deployment.xml file
        app_package.deployment_config = EmptyDeploymentConfiguration()
        with open(self.application_root / "deployment.xml", "w") as f:
            f.write(app_package.deployment_config.to_xml_string())
        # This will delete the deployment
        self.vespa_cloud._start_prod_deployment(self.application_root)


class TestGetClusterEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        tenant_name = "vespa-team"
        application = "vespacloud-docsearch"

        self.vespa_cloud = VespaCloud(
            tenant=tenant_name,
            application_root=application,
            key_content=os.getenv("VESPA_TEAM_API_KEY", None),
            application=application,
            cluster="playground",
        )

    def test_cluster_endpoints(self):
        all_endpoints = self.vespa_cloud.get_all_endpoints(environment="prod")
        self.assertGreater(len(all_endpoints), 2)
        playground_endpoint = self.vespa_cloud.get_mtls_endpoint(
            cluster="playground", environment="prod"
        )
        self.assertIn("vespa-app.cloud/", playground_endpoint)
        default_endpoint = self.vespa_cloud.get_mtls_endpoint(
            cluster="default", environment="prod"
        )
        self.assertIn("vespa-app.cloud/", default_endpoint)


class TestDeployApplicationRoot(unittest.TestCase):
    def setUp(self) -> None:
        self.application_root = (
            pathlib.Path(__file__).parent.parent / "testapps" / "rag-blueprint"
        )
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_root=self.application_root,
        )
        # Delete clients.pem if it exists, to test creation on first deployment
        if (self.application_root / "security" / "clients.pem").exists():
            os.remove(self.application_root / "security" / "clients.pem")
        _app = self.vespa_cloud.deploy(instance="rag-blueprint", max_wait=300)

    def test_deploy_application_root_with_clients_pem(self):
        """After first deployment, the clients.pem file is created in the application root."""
        self.assertTrue(
            (self.application_root / "security" / "clients.pem").exists(),
            "clients.pem file was not created.",
        )
        pem_file_modified_time = (
            (self.application_root / "security" / "clients.pem").stat().st_mtime
        )
        _app = self.vespa_cloud.deploy(instance="rag-blueprint", max_wait=300)
        # Verify modified time is not changed after redeployment
        self.assertEqual(
            pem_file_modified_time,
            (self.application_root / "security" / "clients.pem").stat().st_mtime,
        )

    def tearDown(self) -> None:
        self.vespa_cloud.delete(instance="rag-blueprint")


class TestDeployKeyLocation(unittest.TestCase):
    def test_deploy_key_location(self) -> None:
        self.application_root = (
            pathlib.Path(__file__).parent.parent / "testapps" / "rag-blueprint"
        )
        self.instance = "key-location"
        # Read key from environment variable and write to file
        key = os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n")
        self.key_location = pathlib.Path.home() / ".vespa" / "my-key.pem"
        self.key_location.parent.mkdir(parents=True, exist_ok=True)
        with open(self.key_location, "w") as f:
            f.write(key)
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=None,  # Not needed, but here for clarity
            key_location=self.key_location,
            application_root=self.application_root,
        )
        # Delete clients.pem if it exists, to test creation on first deployment
        if (self.application_root / "security" / "clients.pem").exists():
            os.remove(self.application_root / "security" / "clients.pem")
        self.app = self.vespa_cloud.deploy(instance=self.instance, max_wait=300)
        self.assertEqual(200, self.app.get_application_status().status_code)

    def tearDown(self) -> None:
        # Remove key file
        if self.key_location.exists():
            os.remove(self.key_location)
        self.vespa_cloud.delete(instance=self.instance)


class TestDeployPerf(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = sample_package
        self.instance = "perf-deployment"
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.environment = "perf"

    def test_deploy(self):
        self.app = self.vespa_cloud.deploy(
            instance=self.instance, environment=self.environment, max_wait=600
        )
        endpoints = self.vespa_cloud.get_all_endpoints(
            instance=self.instance, environment=self.environment
        )
        self.assertGreater(len(endpoints), 0)
        contents = self.vespa_cloud.get_app_package_contents(
            instance=self.instance, environment=self.environment
        )
        self.assertGreater(len(contents), 0)

    def tearDown(self) -> None:
        # Wait a little bit to make sure the deployment is finished
        time.sleep(10)
        self.vespa_cloud.delete(instance=self.instance, environment=self.environment)
