# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import shutil
import unittest
from vespa.application import Vespa, ApplicationPackage
from vespa.package import Schema, Document, Field, HNSW, RankProfile
from vespa.deployment import VespaCloud
from vespa.io import VespaResponse

APP_INIT_TIMEOUT = 900

def create_vector_ada_application_package() -> ApplicationPackage:
    return ApplicationPackage(
        name="vector",
        schema=Schema(
            name="vector",
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["attribute", "summary"]),
                    Field(
                        name="embedding",
                        type="tensor<bfloat16>(x[1536])",
                        indexing=["attribute", "summary", "index"],
                        ann=HNSW(
                            distance_metric="innerproduct",
                            max_links_per_node=16,
                            neighbors_to_explore_at_insert=128,
                        ),
                    )
                ],
                rank_profile=RankProfile(
                    name="default", 
                    inputs=[("query(q)", "tensor<float>(x[1536])")],
                    first_phase="closeness(field, embedding))")
            )
    )
)

class TestVectorSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = create_vector_ada_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration-vector-search",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
            auth_client_token_id="pyvespa_integration_msmarco"
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "default"
        self.app: Vespa = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        print("Endpoint used " + self.app.url) 

    def test_right_endpoint_used_with_token(self):
        # The secrect token is set in env variable. 
        print("Endpoint used " + self.app.url)      
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)
        self.assertEqual(200, self.app.get_application_status().status_code)

    def test_vector_indexing_and_query(self):
        from datasets import load_dataset
        print("Endpoint used " + self.app.url)      
        sample_size = 2000

        dataset = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True).take(sample_size)
        docs = list(dataset)
        ok = 0
        with self.app.syncio() as sync_session:
            for doc in docs:
                response:VespaResponse = sync_session.feed_data_point(
                    schema="vector",
                    data_id=doc["_id"],
                    fields={
                        "id": doc["_id"],
                        "embedding": doc["openai"]
                    }
                )
                self.assertEqual(response.get_status_code(), 200)
                ok +=1

        self.assertEqual(ok, sample_size)
        ok = 0
        
        with self.app.asyncio() as async_session:
            for doc in docs:
                response:VespaResponse = async_session.feed_data_point(
                    schema="vector",
                    data_id=doc["_id"],
                    fields={
                        "id": doc["_id"],
                        "embedding": doc["openai"]
                    }
                )
                self.assertEqual(response.get_status_code(), 200)
                ok +=1
        self.assertEqual(ok, sample_size)

        with self.app.syncio() as sync_session:
            response:VespaResponse = sync_session.query(
                {
                    "yql": "select id from sources * where {targetHits:10}nearestNeighbor(embedding,q)",
                    "input.query(q)": docs[0]["openai"],
                    'hits' :10
                }
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 10)
        
        with self.app.asyncio() as async_session:
            response:VespaResponse = async_session.query(
                {
                    "yql": "select id from sources * where {targetHits:10}nearestNeighbor(embedding,q)",
                    "input.query(q)": docs[0]["openai"],
                    'hits' :10
                }
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 10)
          
    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="vector_content", schema="vector"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)


