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
        app_package = ApplicationPackage(name="msmarco", schema=msmarco_schema)
        #
        # Deploy in a Docker container
        #
        vespa_docker = VespaDocker()
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.app = vespa_docker.deploy(
            application_package=app_package, disk_folder=self.disk_folder
        )

    def test_deployment_message(self):
        self.assertTrue(
            any(
                re.match("Generation: [0-9]+", line)
                for line in self.app.deployment_message
            )
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
