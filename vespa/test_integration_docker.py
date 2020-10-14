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
        self.vespa_docker = VespaDocker()
        app = self.vespa_docker.deploy(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )

    def test_deploy_from_disk(self):
        self.vespa_docker = VespaDocker()
        self.vespa_docker.export_application_package(
            dir_path=self.disk_folder, application_package=self.app_package
        )
        app = self.vespa_docker.deploy_from_disk(
            application_name=self.app_package.name, disk_folder=self.disk_folder
        )

        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()
