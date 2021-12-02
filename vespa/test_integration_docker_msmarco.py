import asyncio
import os
import shutil

from vespa.deployment import VespaDocker
from vespa.package import Document, Field, HNSW, Schema, FieldSet, RankProfile, ApplicationPackage
from vespa.test_integration_docker import TestDockerCommon, TestApplicationCommon


def create_msmarco_application_package():
    #
    # Application package
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
                indexing=["attribute", "index"],
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
    app_package = ApplicationPackage(name="msmarco", schema=[msmarco_schema])
    return app_package


class TestMsmarcoDockerDeployment(TestDockerCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")

    def test_deploy(self):
        self.deploy(application_package=self.app_package, disk_folder=self.disk_folder)

    def test_deploy_from_disk_with_disk_folder(self):
        self.deploy_from_disk_with_disk_folder(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_deploy_from_disk_with_application_folder(self):
        self.deploy_from_disk_with_application_folder(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_instantiate_vespa_docker_from_container_name_or_id(self):
        self.create_vespa_docker_from_container_name_or_id(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_redeploy_with_container_stopped(self):
        self.redeploy_with_container_stopped(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_redeploy_with_application_package_changes(self):
        self.redeploy_with_application_package_changes(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_trigger_start_stop_and_restart_services(self):
        self.trigger_start_stop_and_restart_services(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()


class TestMsmarcoApplication(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.vespa_docker = VespaDocker(port=8089, disk_folder=self.disk_folder)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)
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
        # The port should be 8089 instead of 8080, see https://jira.vzbuilders.com/browse/VESPA-21365
        self.get_model_endpoints_when_no_model_is_available(
            app=self.app,
            expected_model_endpoint="http://localhost:8080/model-evaluation/v1/",
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
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()