import unittest
import os
import re
import shutil

from random import random

from vespa.application import Vespa
from vespa.query import Union, WeakAnd, ANN, Query, RankProfile as Ranking
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
        self.vespa_docker = None

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

    def test_workflow(self):
        #
        # Connect to a running Vespa Application
        #
        app = Vespa(url="https://api.cord19.vespa.ai")
        #
        # Define a query model
        #
        match_phase = Union(
            WeakAnd(hits=10),
            ANN(
                doc_vector="title_embedding",
                query_vector="title_vector",
                embedding_model=lambda x: [random() for x in range(768)],
                hits=10,
                label="title",
            ),
        )
        rank_profile = Ranking(name="bm25", list_features=True)
        query_model = Query(match_phase=match_phase, rank_profile=rank_profile)
        #
        # Query Vespa app
        #
        query_result = app.query(
            query="Is remdesivir an effective treatment for COVID-19?",
            query_model=query_model,
        )
        self.assertTrue(query_result.number_documents_retrieved > 0)
        self.assertEqual(len(query_result.hits), 10)
        #
        # Define labelled data
        #
        labelled_data = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": 0, "score": 1}, {"id": 3, "score": 1}],
            },
            {
                "query_id": 1,
                "query": "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus",
                "relevant_docs": [{"id": 1, "score": 1}, {"id": 5, "score": 1}],
            },
        ]
        #
        # Collect training data
        #
        training_data_batch = app.collect_training_data(
            labelled_data=labelled_data,
            id_field="id",
            query_model=query_model,
            number_additional_docs=2,
            fields=["rankfeatures"],
        )
        self.assertTrue(training_data_batch.shape[0] > 0)
        self.assertEqual(
            len(
                {"document_id", "query_id", "label"}.intersection(
                    set(training_data_batch.columns)
                )
            ),
            3,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        if self.vespa_docker:
            self.vespa_docker.container.stop()
            self.vespa_docker.container.remove()
