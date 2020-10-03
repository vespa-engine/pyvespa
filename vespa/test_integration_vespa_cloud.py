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

    def test_deployment_message(self):
        self.assertIsInstance(self.app, Vespa)

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
