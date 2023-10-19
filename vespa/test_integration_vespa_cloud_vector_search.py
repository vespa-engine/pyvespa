# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import sys
import shutil
import asyncio
from typing import Iterable
import unittest
from vespa.application import Vespa, ApplicationPackage
from vespa.package import Schema, Document, Field, HNSW, RankProfile
from vespa.deployment import VespaCloud
from vespa.io import VespaResponse
import time

APP_INIT_TIMEOUT = 900

def create_vector_ada_application_package() -> ApplicationPackage:
    return ApplicationPackage(
        name="vector",
        schema=[Schema(
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
                ]
            ),
            rank_profiles=[
                RankProfile(
                    name="default", 
                    inputs=[("query(q)", "tensor<float>(x[1536])")],
                    first_phase="closeness(field, embedding)"
                )
            ])
        ]
    ) 

async def execute_async(app: Vespa, docs: Iterable[dict]) -> int: 
    async with app.asyncio() as async_session:
        ok = 0
        for doc in docs:
            response:VespaResponse = await async_session.feed_data_point(
                schema="vector",
                data_id=doc["id"],
                fields=doc["fields"]
            )
            if response.status_code == 200:
                ok +=1
        return ok
            

class TestVectorSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = create_vector_ada_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-int-vsearch",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
            auth_client_token_id="pyvespa_integration_msmarco"
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "default"
        self.app: Vespa = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )


    def test_vector_indexing_and_query(self):
        print("Waiting for endpoint " + self.app.url)      
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)
        self.assertEqual(200, self.app.get_application_status().status_code)
       
        from datasets import load_dataset
        sample_size = 100000
        # streaming=True pages the data from S3. This is needed to avoid memory issues when loading the dataset.
        dataset = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True).take(sample_size)
        # Map does not page, this allows chaining of maps where the lambda is yielding the next document.
        pyvespa_feed_format = dataset.map(lambda x: {"id": x["_id"], "fields": {"id": x["_id"], "vector":x["openai"]}})

        docs = list(pyvespa_feed_format) # we have enough memory to page everything into memory with list()
        ok = 0
        def callback(response:VespaResponse, id:str):
            nonlocal ok
            if response.get_status_code() != 200:
                print("Error for doc " + id, sys.stderr)
                print(response.get_json())
            else:
                ok +=1
        start = time.time()
        self.app.feed_iterable(iter=docs, schema="vector", namespace="benchmark", callback=callback, max_workers=48, max_connections=48)
        self.assertEqual(ok, sample_size)
        duration = time.time() - start
        docs_per_second = sample_size / duration
        print("Sync Feed time: " + str(duration) + " docs per second: " + str(docs_per_second))
        
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
        
        # Async test
        ok = 0
        start = time.time()
        ok = asyncio.run(execute_async(self.app, docs))
        self.assertEqual(ok, sample_size)
        duration = time.time() - start
        docs_per_second = sample_size / duration
        print("Async Feed time: " + str(duration) + " docs per second: " + str(docs_per_second))

          
    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="vector_content", schema="vector"
        )
        with self.app.syncio() as sync_session:
            response:VespaResponse = sync_session.query(   
                {
                    "yql": "select id from sources * where true",
                    'hits' :10
                }
            )
            self.assertEqual(response.get_status_code(), 200)
            print(response.get_json())
        shutil.rmtree(self.disk_folder, ignore_errors=True)


