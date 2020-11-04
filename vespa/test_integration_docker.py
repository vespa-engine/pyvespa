import unittest
import os
import re
import shutil
from vespa.package import Document, Field
from vespa.package import Schema, FieldSet, RankProfile
from vespa.package import ApplicationPackage
from vespa.package import VespaDocker


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

    def test_data_operations(self):
        self.vespa_docker = VespaDocker()
        app = self.vespa_docker.deploy(
            application_package=self.app_package, disk_folder=self.disk_folder
        )
        print(app.get_data(schema="test", data_id="1"))

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()
