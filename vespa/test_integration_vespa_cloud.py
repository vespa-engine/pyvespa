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
        self.app_package = ApplicationPackage(name="msmarco", schema=msmarco_schema)
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")

    def test_deploy(self):
        vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
        )
        app = vespa_cloud.deploy(
            application_package=self.app_package,
            instance="test",
            disk_folder=self.disk_folder,
        )
        self.assertIsInstance(app, Vespa)

    def test_deploy_from_disk(self):
        vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_CLOUD_USER_KEY").replace(r"\n", "\n"),
        )
        vespa_cloud.export_application_package(
            dir_path=self.disk_folder, application_package=self.app_package
        )
        app = vespa_cloud.deploy_from_disk(
            application_name=self.app_package.name, disk_folder=self.disk_folder
        )
        self.assertIsInstance(app, Vespa)

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
