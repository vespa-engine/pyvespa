# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import shutil
import unittest
import time
from vespa.application import Vespa, ApplicationPackage
from vespa.package import (
    Schema,
    Document,
    Field,
    HNSW,
    RankProfile,
    AuthClient,
    Parameter,
)
from vespa.deployment import VespaCloud
from vespa.io import VespaResponse, VespaQueryResponse
from vespa.package import (
    ContentCluster,
    ContainerCluster,
    Nodes,
    DeploymentConfiguration,
    Validation,
    ValidationID,
)
from datetime import datetime, timedelta

APP_INIT_TIMEOUT = 900
CLIENT_TOKEN_ID = os.environ.get("VESPA_CLIENT_TOKEN_ID", "pyvespa_integration_msmarco")


def create_vector_ada_application_package() -> ApplicationPackage:
    return ApplicationPackage(
        name="vector",
        schema=[
            Schema(
                name="vector",
                document=Document(
                    fields=[
                        Field(
                            name="id", type="string", indexing=["attribute", "summary"]
                        ),
                        Field(
                            name="meta",
                            type="string",
                            indexing=["attribute", "summary"],
                        ),
                        Field(
                            name="embedding",
                            type="tensor<bfloat16>(x[1536])",
                            indexing=["attribute", "summary", "index"],
                            ann=HNSW(
                                distance_metric="innerproduct",
                                max_links_per_node=16,
                                neighbors_to_explore_at_insert=128,
                            ),
                        ),
                    ]
                ),
                rank_profiles=[
                    RankProfile(
                        name="default",
                        inputs=[("query(q)", "tensor<float>(x[1536])")],
                        first_phase="closeness(field, embedding)",
                    )
                ],
            )
        ],
    )


class TestVectorSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = create_vector_ada_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-vsearch-dev",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getcwd(), "sample_application")
        self.instance_name = "default"
        self.app: Vespa = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )

    def test_vector_indexing_and_query(self):
        print("Waiting for endpoint " + self.app.url)
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)
        self.assertEqual(200, self.app.get_application_status().status_code)

        from datasets import load_dataset

        sample_size = 100
        # streaming=True pages the data from S3. This is needed to avoid memory issues when loading the dataset.
        dataset = load_dataset(
            "KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True
        ).take(sample_size)
        # Map does not page, this allows chaining of maps where the lambda is yielding the next document.
        pyvespa_feed_format = dataset.map(
            lambda x: {
                "id": x["_id"],
                "fields": {"id": x["_id"], "embedding": x["openai"]},
            }
        )

        docs = list(
            pyvespa_feed_format
        )  # we have enough memory to page everything into memory with list()
        # seems like we sometimes can get more than sample_size docs
        if len(docs) > sample_size:
            docs = docs[:sample_size]
        self.assertEqual(len(docs), sample_size)
        ok = 0
        callbacks = 0
        start_time = time.time()

        def callback(response: VespaResponse, id: str):
            nonlocal ok
            nonlocal start_time
            nonlocal callbacks
            if response.is_successful():
                ok += 1
            callbacks += 1

        start = time.time()
        self.app.feed_iterable(
            iter=docs,
            schema="vector",
            namespace="benchmark",
            callback=callback,
        )
        self.assertEqual(ok, sample_size)
        duration = time.time() - start
        docs_per_second = sample_size / duration
        print(
            "Sync Feed time: "
            + str(duration)
            + " seconds,  docs per second: "
            + str(docs_per_second)
        )

        with self.app.syncio() as sync_session:
            response: VespaQueryResponse = sync_session.query(
                {
                    "yql": "select id from sources * where {targetHits:10}nearestNeighbor(embedding,q)",
                    "input.query(q)": docs[0]["openai"],
                    "hits": 10,
                }
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 10)

            response: VespaQueryResponse = sync_session.query(
                yql="select id from sources * where {targetHits:10}nearestNeighbor(embedding,q)",
                hits=5,
                body={"input.query(q)": docs[0]["openai"]},
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 5)

        # check error callbacks
        ok = 0
        callbacks = 0
        start_time = time.time()

        feed_with_wrong_field = dataset.map(
            lambda x: {
                "id": x["_id"],
                "fields": {"id": x["_id"], "vector": x["openai"]},
            }
        )
        faulty_docs = list(feed_with_wrong_field)
        if len(faulty_docs) > sample_size:
            faulty_docs = faulty_docs[:sample_size]
        self.assertEqual(len(faulty_docs), sample_size)
        self.app.feed_iterable(
            iter=faulty_docs,
            schema="vector",
            namespace="benchmark",
            callback=callback,
        )
        self.assertEqual(ok, 0)
        self.assertEqual(callbacks, 100)

        ok = 0

        # Run update - assign all docs with a meta field

        updates = dataset.map(lambda x: {"id": x["_id"], "fields": {"meta": "stuff"}})
        start_time = time.time()
        self.app.feed_iterable(
            iter=updates,
            schema="vector",
            namespace="benchmark",
            callback=callback,
            operation_type="update",
        )
        self.assertEqual(ok, sample_size)
        duration = time.time() - start_time
        docs_per_second = sample_size / duration
        print(
            "Sync Update time: "
            + str(duration)
            + " seconds,  docs per second: "
            + str(docs_per_second)
        )

        with self.app.syncio() as sync_session:
            response: VespaQueryResponse = sync_session.query(
                yql='select id from sources * where meta contains "stuff"',
                hits=5,
                timeout="15s",
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 5)
            self.assertEqual(response.number_documents_retrieved, sample_size)

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="vector_content",
            schema="vector",
            namespace="benchmark",
        )
        time.sleep(5)
        with self.app.syncio() as sync_session:
            response: VespaResponse = sync_session.query(
                {"yql": "select id from sources * where true", "hits": 10}
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 0)
            print(response.get_json())
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_cloud.delete()


class TestProdDeploymentFromDisk(unittest.TestCase):
    def test_setup(self) -> None:
        self.app_package = create_vector_ada_application_package()
        prod_region = "aws-us-east-1c"
        self.app_package.clusters = [
            ContentCluster(
                id="vector_content",
                nodes=Nodes(count="2"),
                document_name="vector",
                min_redundancy="2",
            ),
            ContainerCluster(
                id="vector_container",
                nodes=Nodes(count="2"),
            ),
        ]
        self.app_package.deployment_config = DeploymentConfiguration(
            environment="prod", regions=[prod_region]
        )
        self.app_package.auth_clients = [
            AuthClient(
                id="mtls",
                permissions=["read,write"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            )
        ]

        # Validation override is needed to be able to be able to swith between triggering from local or CI.
        tomorrow = datetime.now() + timedelta(days=1)
        formatted_date = tomorrow.strftime("%Y-%m-%d")
        self.app_package.validations = [
            Validation(ValidationID("certificate-removal"), formatted_date)
        ]

        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-int-vsearch",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.application_root = os.path.join(os.getcwd(), "sample_application")
        self.vespa_cloud.application_package.to_files(self.application_root)
        self.instance_name = "default"
        self.build_no = self.vespa_cloud.deploy_to_prod(
            instance=self.instance_name,
            application_root=self.application_root,
        )

    @unittest.skip(
        "This test is too slow for normal testing. Can be used for manual testing if related code is changed."
    )
    def test_application_status(self):
        success = self.vespa_cloud.wait_for_prod_deployment(
            build_no=self.build_no, max_wait=3600
        )
        if not success:
            self.fail("Deployment failed")
        self.app: Vespa = self.vespa_cloud.get_application(environment="prod")
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)

    @unittest.skip("Do not run when not waiting for deployment.")
    def test_vector_indexing_and_query(self):
        super().test_vector_indexing_and_query()

    # DO NOT skip tearDown-method, as test will not exit.
    # @unittest.skip("Do not run when not waiting for deployment.")
    # def tearDown(self) -> None:
    #     self.app.delete_all_docs(
    #         content_cluster_name="vector_content",
    #         schema="vector",
    #         namespace="benchmark",
    #     )
    #     time.sleep(5)
    #     with self.app.syncio() as sync_session:
    #         response: VespaResponse = sync_session.query(
    #             {"yql": "select id from sources * where true", "hits": 10}
    #         )
    #         self.assertEqual(response.get_status_code(), 200)
    #         self.assertEqual(len(response.hits), 0)
    #         print(response.get_json())

    #     # Deployment is deleted by deploying with an empty deployment.xml file.
    #     self.app_package.deployment_config = EmptyDeploymentConfiguration()

    #     # Vespa won't push the deleted deployment.xml file unless we add a validation override
    #     tomorrow = datetime.now() + timedelta(days=1)
    #     formatted_date = tomorrow.strftime("%Y-%m-%d")
    #     self.app_package.validations = [
    #         Validation(ValidationID("deployment-removal"), formatted_date)
    #     ]
    #     self.app_package.to_files(self.application_root)
    #     # This will delete the deployment
    #     self.vespa_cloud._start_prod_deployment(self.application_root)
    #     shutil.rmtree(self.application_root, ignore_errors=True)
