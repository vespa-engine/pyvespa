import unittest
import os
import shutil
from vespa.application import Vespa
from vespa.package import (
    Document,
    Field,
    Schema,
    FieldSet,
    RankProfile,
    ApplicationPackage,
    VespaCloud,
)


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
        self.assertEqual(self.app.get_data(schema="msmarco", data_id="1").status_code, 404)
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
        self.assertEqual(self.app.get_data(schema="msmarco", data_id="1").status_code, 404)

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
