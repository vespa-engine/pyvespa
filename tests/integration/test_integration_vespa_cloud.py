# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import asyncio
import shutil
import pytest
from requests import HTTPError
import unittest
from cryptography.hazmat.primitives import serialization
from vespa.package import ContentCluster, ContainerCluster, Nodes, DeploymentConfiguration
from vespa.application import Vespa
from vespa.deployment import VespaCloud
from vespa.io import VespaResponse, VespaQueryResponse
from test_integration_docker import (
    TestApplicationCommon,
    create_msmarco_application_package,
)
from test_integration_vespa_cloud_vector_search import create_vector_ada_application_package
from pathlib import Path
import time

APP_INIT_TIMEOUT = 900


@pytest.mark.skip(reason="Temporarily disabled")
class TestVespaKeyAndCertificate(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "msmarco"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )

    def test_key_cert_arguments(self):
        #
        # Write key and cert to different files
        #
        with open(os.path.join(self.disk_folder, "key_file.txt"), "w+") as file:
            file.write(
                self.vespa_cloud.data_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption(),
                ).decode("UTF-8")
            )
        with open(os.path.join(self.disk_folder, "cert_file.txt"), "w+") as file:
            file.write(
                self.vespa_cloud.data_certificate.public_bytes(
                    serialization.Encoding.PEM
                ).decode("UTF-8")
            )
        self.app = Vespa(
            url=self.app.url,
            key=os.path.join(self.disk_folder, "key_file.txt"),
            cert=os.path.join(self.disk_folder, "cert_file.txt"),
            application_package=self.app.application_package,
        )
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)
        self.assertEqual(200, self.app.get_application_status().status_code)
        self.assertDictEqual(
            {
                "pathId": "/document/v1/msmarco/msmarco/docid/1",
                "id": "id:msmarco:msmarco::1",
            },
            self.app.get_data(schema="msmarco", data_id="1").json,
        )
        self.assertEqual(
            self.app.get_data(schema="msmarco", data_id="1").is_successful(), False
        )
        with pytest.raises(HTTPError):
            self.app.get_data(schema="msmarco", data_id="1", raise_on_not_found=True)

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="msmarco_content", schema="msmarco"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_cloud.delete(instance=self.instance_name)


@pytest.mark.skip(reason="Temporarily disabled")
class TestMsmarcoApplication(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "msmarco"
        self.app = self.vespa_cloud.deploy(
            instance=self.instance_name, disk_folder=self.disk_folder
        )
        self.fields_to_send = [
            {
                "id": f"{i}",
                "title": f"this is title {i}",
                "body": f"this is body {i}",
            }
            for i in range(10)
        ]
        self.fields_to_update = [
            {
                "id": f"{i}",
                "title": "this is my updated title number {}".format(i),
            }
            for i in range(10)
        ]

    def test_prediction_when_model_not_defined(self):
        self.get_stateless_prediction_when_model_not_defined(
            app=self.app, application_package=self.app_package
        )

    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.fields_to_send[0],
        )

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_data_operations(
                app=self.app,
                schema_name=self.app_package.name,
                fields_to_send=self.fields_to_send,
                field_to_update=self.fields_to_update[0],
                expected_fields_from_get_operation=self.fields_to_send,
            )
        )

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="msmarco_content", schema="msmarco"
        )
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_cloud.delete(instance=self.instance_name)


class TestProdDeployment(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = create_vector_ada_application_package()
        self.app_package.clusters  = [
            ContentCluster(
                id="vector_content",
                nodes=Nodes(count="2"),
                document_name="vector",
                min_redundancy="2"
            ),
            ContainerCluster(
                id="vector_container",
                nodes=Nodes(count="2"),
            )
        ]
        self.app_package.deployment_config = DeploymentConfiguration(
            environment="prod", regions=["aws-us-east-1c"]
        )

        self.vespa_cloud = VespaCloud(
            #tenant="vespa-team",
            tenant="torstein",
            application="vector",
            #key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            key_location = Path.home() / ".vespa" / "torstein.api-key.pem",
            application_package=self.app_package,
        )
        #self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.instance_name = "default"
        self.app = self.vespa_cloud.deploy_to_prod(instance=self.instance_name)  # TODO add disk_folder

    def test_indexing_and_query(self):
        print("Waiting for endpoint " + self.app.url)      
        self.app.wait_for_application_up(max_wait=APP_INIT_TIMEOUT)
        self.assertEqual(200, self.app.get_application_status().status_code)
       
        from datasets import load_dataset
        sample_size = 10000
        # streaming=True pages the data from S3. This is needed to avoid memory issues when loading the dataset.
        dataset = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True).take(sample_size)
        # Map does not page, this allows chaining of maps where the lambda is yielding the next document.
        pyvespa_feed_format = dataset.map(lambda x: {"id": x["_id"], "fields": {"id": x["_id"], "embedding":x["openai"]}})

        docs = list(pyvespa_feed_format) # we have enough memory to page everything into memory with list()
        ok = 0
        callbacks = 0
        start_time = time.time()
        def callback(response:VespaResponse, id:str):
            nonlocal ok
            nonlocal start_time
            nonlocal callbacks
            if response.is_successful():
                ok +=1
            callbacks +=1

        start = time.time()
        self.app.feed_iterable(iter=docs, schema="vector", namespace="benchmark", callback=callback, max_workers=48, max_connections=48, max_queue_size=4000)
        self.assertEqual(ok, sample_size)
        duration = time.time() - start
        docs_per_second = sample_size / duration
        print("Sync Feed time: " + str(duration) + " seconds,  docs per second: " + str(docs_per_second))
        
        with self.app.syncio() as sync_session:
            response:VespaQueryResponse = sync_session.query(   
                {
                    "yql": "select id from sources * where {targetHits:10}nearestNeighbor(embedding,q)",
                    "input.query(q)": docs[0]["openai"],
                    'hits' :10
                }
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 10)

            response:VespaQueryResponse = sync_session.query(
                yql="select id from sources * where {targetHits:10}nearestNeighbor(embedding,q)",
                hits=5,
                body={
                    "input.query(q)": docs[0]["openai"]
                }
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 5)
       
        #check error callbacks 
        ok = 0
        callbacks = 0
        start_time = time.time()
        dataset = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True).take(100)
        feed_with_wrong_field = dataset.map(lambda x: {"id": x["_id"], "fields": {"id": x["_id"], "vector":x["openai"]}})
        faulty_docs = list(feed_with_wrong_field) 
        self.app.feed_iterable(iter=faulty_docs, schema="vector", namespace="benchmark", callback=callback, max_workers=48, max_connections=48)
        self.assertEqual(ok, 0)
        self.assertEqual(callbacks, 100)

        ok = 0
        dataset = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True).take(sample_size)
        # Run update - assign all docs with a meta field
        
        updates = dataset.map(lambda x: {"id": x["_id"], "fields": {"meta":"stuff"}})
        start_time = time.time()
        self.app.feed_iterable(iter=updates, schema="vector", namespace="benchmark", callback=callback, operation_type="update")
        self.assertEqual(ok, sample_size)
        duration = time.time() - start_time
        docs_per_second = sample_size / duration
        print("Sync Update time: " + str(duration) + " seconds,  docs per second: " + str(docs_per_second))

        with self.app.syncio() as sync_session:
            response:VespaQueryResponse = sync_session.query(
                yql="select id from sources * where meta contains \"stuff\"",
                hits=5,
                timeout="15s"
            )
            self.assertEqual(response.get_status_code(), 200)
            self.assertEqual(len(response.hits), 5)
            self.assertEqual(response.number_documents_retrieved, sample_size)

    def tearDown(self) -> None:
        ...
