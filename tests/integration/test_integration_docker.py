# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
import pytest
import os
import asyncio
import json
import time
import requests
import random

from typing import List, Dict, Optional
from vespa.io import VespaResponse, VespaQueryResponse
from vespa.resources import get_resource_path
from vespa.package import (
    HNSW,
    Document,
    Field,
    Schema,
    FieldSet,
    RankProfile,
    ApplicationPackage,
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
    AuthClient,
    Struct,
)
from vespa.deployment import VespaDocker
from vespa.application import VespaSync
from vespa.exceptions import VespaError

CONTAINER_STOP_TIMEOUT = 10
RESOURCES_DIR = get_resource_path()


def create_msmarco_application_package(auth_clients: List[AuthClient] = None):
    #
    # Application package
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
            Field(
                name="metadata",
                type="string",
                indexing=["attribute", "summary"],
                attribute=["fast-search", "fast-access"],
            ),
            Field(
                name="tensor_field",
                type="tensor<float>(x[128])",
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="euclidean",
                    max_links_per_node=16,
                    neighbors_to_explore_at_insert=200,
                ),
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
    app_package = ApplicationPackage(
        name="msmarco", schema=[msmarco_schema], auth_clients=auth_clients
    )
    return app_package


class QuestionAnswering(ApplicationPackage):
    def __init__(self, name: str = "qa"):
        context_document = Document(
            fields=[
                Field(
                    name="questions",
                    type="array<int>",
                    indexing=["summary", "attribute"],
                ),
                Field(name="dataset", type="string", indexing=["summary", "attribute"]),
                Field(name="context_id", type="int", indexing=["summary", "attribute"]),
                Field(
                    name="text",
                    type="string",
                    indexing=["summary", "index"],
                    index="enable-bm25",
                ),
            ]
        )
        context_schema = Schema(
            name="context",
            document=context_document,
            fieldsets=[FieldSet(name="default", fields=["text"])],
            rank_profiles=[
                RankProfile(name="bm25", inherits="default", first_phase="bm25(text)"),
                RankProfile(
                    name="nativeRank",
                    inherits="default",
                    first_phase="nativeRank(text)",
                ),
            ],
        )
        sentence_document = Document(
            inherits="context",
            fields=[
                Field(
                    name="sentence_embedding",
                    type="tensor<float>(x[512])",
                    indexing=["attribute", "index"],
                    ann=HNSW(
                        distance_metric="euclidean",
                        max_links_per_node=16,
                        neighbors_to_explore_at_insert=500,
                    ),
                )
            ],
        )
        sentence_schema = Schema(
            name="sentence",
            document=sentence_document,
            fieldsets=[FieldSet(name="default", fields=["text"])],
            rank_profiles=[
                RankProfile(
                    name="semantic-similarity",
                    inherits="default",
                    first_phase="closeness(sentence_embedding)",
                ),
                RankProfile(name="bm25", inherits="default", first_phase="bm25(text)"),
                RankProfile(
                    name="bm25-semantic-similarity",
                    inherits="default",
                    first_phase="bm25(text) + closeness(sentence_embedding)",
                ),
            ],
        )
        super().__init__(
            name=name,
            schema=[context_schema, sentence_schema],
            query_profile=QueryProfile(),
            query_profile_type=QueryProfileType(
                fields=[
                    QueryTypeField(
                        name="ranking.features.query(query_embedding)",
                        type="tensor<float>(x[512])",
                    )
                ]
            ),
        )


def create_qa_application_package():
    app_package = QuestionAnswering()
    #
    # Our test suite requires that each schema has a 'id' field
    #
    app_package.get_schema("sentence").add_fields(
        Field(name="id", type="string", indexing=["attribute", "summary"])
    )
    app_package.get_schema("context").add_fields(
        Field(name="id", type="string", indexing=["attribute", "summary"])
    )
    return app_package


class TestDockerCommon(unittest.TestCase):
    def deploy(self, application_package, container_image=None):
        if container_image:
            self.vespa_docker = VespaDocker(port=8089, container_image=container_image)
        else:
            self.vespa_docker = VespaDocker(port=8089)

        try:
            self.vespa_docker.deploy(application_package=application_package)
        except RuntimeError as e:
            assert False, "Deployment error: {}".format(e)

    def create_vespa_docker_from_container_name_or_id(self, application_package):
        #
        # Raises ValueError if container does not exist
        #
        with self.assertRaises(ValueError):
            _ = VespaDocker.from_container_name_or_id(application_package.name)
        #
        # Test VespaDocker instance created from container
        #
        self.vespa_docker = VespaDocker(port=8089)
        _ = self.vespa_docker.deploy(application_package=application_package)
        vespa_docker_from_container = VespaDocker.from_container_name_or_id(
            application_package.name
        )
        self.assertEqual(self.vespa_docker, vespa_docker_from_container)

    def redeploy_with_container_stopped(self, application_package):
        self.vespa_docker = VespaDocker(port=8089)
        self.vespa_docker.deploy(application_package=application_package)
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        app = self.vespa_docker.deploy(application_package=application_package)
        self.assertEqual(app.get_application_status().status_code, 200)

    def redeploy_with_application_package_changes(self, application_package):
        self.vespa_docker = VespaDocker(port=8089)
        app = self.vespa_docker.deploy(application_package=application_package)
        with pytest.raises(VespaError):
            app.query(
                body={
                    "yql": "select * from sources * where default contains 'music'",
                    "ranking": "new-rank-profile",
                }
            )

        application_package.schema.add_rank_profile(
            RankProfile(
                name="new-rank-profile", inherits="default", first_phase="bm25(title)"
            )
        )
        app = self.vespa_docker.deploy(application_package=application_package)
        res = app.query(
            body={
                "yql": "select * from sources * where default contains 'music'",
                "ranking": "new-rank-profile",
            }
        ).json
        self.assertTrue("errors" not in res["root"])

    def trigger_start_stop_and_restart_services(self, application_package):
        self.vespa_docker = VespaDocker(port=8089)

        with self.assertRaises(RuntimeError):
            self.vespa_docker.stop_services()
        with self.assertRaises(RuntimeError):
            self.vespa_docker.start_services()

        app = self.vespa_docker.deploy(application_package=application_package)
        self._wait_for_service_start()

        self.assertTrue(self.vespa_docker._check_configuration_server())
        self.assertEqual(app.get_application_status().status_code, 200)

        self.vespa_docker.stop_services()
        self._wait_for_service_stop()

        self.assertFalse(self.vespa_docker._check_configuration_server())
        self.assertIsNone(self._safe_get_application_status(app))

        self.vespa_docker.start_services()
        self._wait_for_service_start()

        self.assertTrue(self.vespa_docker._check_configuration_server())
        self.assertEqual(app.get_application_status().status_code, 200)

        self.vespa_docker.restart_services()
        self._wait_for_service_start()

        self.assertTrue(self.vespa_docker._check_configuration_server())
        self.assertEqual(app.get_application_status().status_code, 200)

    def _wait_for_service_start(self, timeout=30, interval=1):
        """Wait for the Vespa service to start."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.vespa_docker._check_configuration_server():
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(interval)
        raise RuntimeError("Vespa service did not start within the timeout period.")

    def _wait_for_service_stop(self, timeout=30, interval=1):
        """Wait for the Vespa service to stop."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.vespa_docker._check_configuration_server():
                return
            time.sleep(interval)
        raise RuntimeError("Vespa service did not stop within the timeout period.")

    def _safe_get_application_status(self, app, retries=5, interval=1):
        """Try to get the application status, returning None if it fails."""
        for _ in range(retries):
            try:
                return app.get_application_status()
            except requests.exceptions.ConnectionError:
                time.sleep(interval)
        return None


class TestApplicationCommon(unittest.TestCase):
    # Set maxDiff to None to see full diff
    maxDiff = None

    async def handle_longlived_connection(self, app, n_seconds=10):
        # Test that the connection can live for at least n_seconds
        async with app.asyncio(connections=1) as async_app:
            response = await async_app.httpx_client.get(
                app.end_point + "/ApplicationStatus"
            )
            self.assertEqual(response.status_code, 200)
            await asyncio.sleep(n_seconds)
            response = await async_app.httpx_client.get(
                app.end_point + "/ApplicationStatus"
            )
            self.assertEqual(response.status_code, 200)

    async def async_is_http2_client(self, app):
        async with app.asyncio() as async_app:
            response = await async_app.httpx_client.get(
                app.end_point + "/ApplicationStatus"
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.http_version, "HTTP/2")

    def sync_client_accept_encoding_gzip(self, app):
        data = {
            "yql": "select * from sources * where true",
            "hits": 10,
        }
        with app.syncio() as sync_app:
            response = sync_app.http_session.post(app.search_end_point, json=data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-encoding"], "gzip")
            # Check that gzip is in request headers
            self.assertIn("gzip", response.request.headers["Accept-Encoding"])

    async def async_client_accept_encoding_gzip(self, app):
        data = {
            "yql": "select * from sources * where true",
            "hits": 10,
        }
        async with app.asyncio() as async_app:
            response = await async_app.httpx_client.post(
                app.search_end_point, json=data
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-encoding"], "gzip")
            # Check that gzip is in request headers
            self.assertIn("gzip", response.request.headers["Accept-Encoding"])

    def execute_data_operations(
        self,
        app,
        schema_name,
        cluster_name,
        fields_to_send,
        field_to_update,
        expected_fields_from_get_operation,
        expected_fields_after_update: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Feed, get, update and delete data to/from the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: Dict where keys are field names and values are field values. Must contain 'id' field
        :param field_to_update: Dict where keys are field names and values are field values.
        :param expected_fields_from_get_operation: Dict containing fields as returned by Vespa get operation.
            There are cases where fields returned from Vespa are different from inputs, e.g. when dealing with Tensors.
        :param expected_fields_after_update: Dict containing fields as returned by Vespa get operation after update. If None, will be inferred by performing `expected_fields_from_get_operation.update(field_to_update)`
        :param kwargs: Additional parameters to be passed to the get/update/delete operations
        :return:
        """
        assert "id" in fields_to_send, "fields_to_send must contain 'id' field."
        #
        # Get data that does not exist
        #
        response: VespaResponse = app.get_data(
            schema=schema_name, data_id=fields_to_send["id"], **kwargs
        )
        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.is_successful())

        #
        # Feed a data point
        #
        response = app.feed_data_point(
            schema=schema_name,
            data_id=fields_to_send["id"],
            fields=fields_to_send,
            **kwargs,
        )

        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get data that exist
        #
        response = app.get_data(
            schema=schema_name, data_id=fields_to_send["id"], **kwargs
        )
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(
            response.json,
            {
                "fields": expected_fields_from_get_operation,
                "id": "id:{}:{}::{}".format(
                    schema_name, schema_name, fields_to_send["id"]
                ),
                "pathId": "/document/v1/{}/{}/docid/{}".format(
                    schema_name, schema_name, fields_to_send["id"]
                ),
            },
        )

        #
        # Visit data
        visit_results = []
        for slice in app.visit(
            schema=schema_name,
            content_cluster_name=cluster_name,
            timeout="200s",
        ):
            for response in slice:
                visit_results.append(response)

        self.assertDictEqual(
            visit_results[0].json,
            {
                "pathId": "/document/v1/{}/{}/docid/".format(schema_name, schema_name),
                "documents": [
                    {
                        "id": "id:{}:{}::{}".format(
                            schema_name, schema_name, fields_to_send["id"]
                        ),
                        "fields": expected_fields_from_get_operation,
                    }
                ],
                "documentCount": 1,
            },
        )

        #
        # Update data
        #
        response = app.update_data(
            schema=schema_name,
            data_id=field_to_update["id"],
            fields=field_to_update,
            **kwargs,
        )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get the updated data point
        #
        response = app.get_data(
            schema=schema_name, data_id=field_to_update["id"], **kwargs
        )
        self.assertEqual(response.status_code, 200)
        if expected_fields_after_update is None:
            expected_result = {
                k: v for k, v in expected_fields_from_get_operation.items()
            }
            expected_result.update(field_to_update)
        else:
            expected_result = expected_fields_after_update

        self.assertDictEqual(
            response.json,
            {
                "fields": expected_result,
                "id": "id:{}:{}::{}".format(
                    schema_name, schema_name, fields_to_send["id"]
                ),
                "pathId": "/document/v1/{}/{}/docid/{}".format(
                    schema_name, schema_name, fields_to_send["id"]
                ),
            },
        )
        #
        # Delete a data point
        #
        response = app.delete_data(
            schema=schema_name, data_id=fields_to_send["id"], **kwargs
        )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Deleted data should be gone
        response: VespaResponse = app.get_data(
            schema=schema_name, data_id=fields_to_send["id"], **kwargs
        )
        self.assertFalse(response.is_successful())
        # Check if auto_assign is in kwargs and return if it is False
        # The remainding tests does not make sense (and will not work) for partial updates.
        if "auto_assign" in kwargs and not kwargs["auto_assign"]:
            return
        #
        # Update a non-existent data point
        #
        response = app.update_data(
            schema=schema_name,
            data_id=field_to_update["id"],
            fields=field_to_update,
            create=True,
            **kwargs,
        )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get the updated data point
        #
        response = app.get_data(
            schema=schema_name, data_id=fields_to_send["id"], **kwargs
        )
        self.assertEqual(response.status_code, 200)
        if expected_fields_after_update is None:
            expected_fields = field_to_update
        else:
            expected_fields = {
                k: v
                for k, v in expected_fields_after_update.items()
                if k in field_to_update
            }
        self.assertDictEqual(
            response.json,
            {
                "fields": expected_fields,
                "id": "id:{}:{}::{}".format(
                    schema_name, schema_name, field_to_update["id"]
                ),
                "pathId": "/document/v1/{}/{}/docid/{}".format(
                    schema_name, schema_name, field_to_update["id"]
                ),
            },
        )
        #
        # Use VespaSync - delete data point
        #
        with VespaSync(app=app) as sync_app:
            response = sync_app.delete_data(
                schema=schema_name, data_id=field_to_update["id"], **kwargs
            )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, field_to_update["id"]),
        )
        #
        # Use VespaSync via http attribute - feed data point
        #
        with app.http(pool_maxsize=20) as sync_app:
            response = sync_app.feed_data_point(
                schema=schema_name,
                data_id=fields_to_send["id"],
                fields=fields_to_send,
                tracelevel=9,
                **kwargs,
            )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        self.assertTrue(response.is_successful())
        self.assertTrue("trace" in response.json)

    async def execute_async_data_operations(
        self,
        app,
        schema_name,
        fields_to_send,
        field_to_update,
        expected_fields_from_get_operation,
        expected_fields_after_update: Optional[Dict] = None,
    ):
        """
        Async feed, get, update and delete data to/from the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :param field_to_update: Dict where keys are field names and values are field values.
        :param expected_fields_from_get_operation: Dict containing fields as returned by Vespa get operation.
            There are cases where fields returned from Vespa are different from inputs, e.g. when dealing with Tensors.
        :return:
        """
        async with app.asyncio(connections=12, total_timeout=50) as async_app:
            #
            # Get data that does not exist and test additional request params
            #
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"], tracelevel=9
            )
            self.assertEqual(response.status_code, 404)
            self.assertTrue("trace" in response.json)

            # Feed some data points
            feed = []
            for fields in fields_to_send:
                feed.append(
                    asyncio.create_task(
                        async_app.feed_data_point(
                            schema=schema_name,
                            data_id=fields["id"],
                            fields=fields,
                            timeout=10,
                        )
                    )
                )
            await asyncio.wait(feed, return_when=asyncio.ALL_COMPLETED)
            result = feed[0].result().json
            self.assertEqual(
                result["id"],
                "id:{}:{}::{}".format(
                    schema_name, schema_name, fields_to_send[0]["id"]
                ),
            )

            self.assertEqual(
                await async_app.feed_data_point(
                    schema=schema_name,
                    data_id="1",
                    fields=fields,
                ),
                app.feed_data_point(
                    schema=schema_name,
                    data_id="1",
                    fields=fields,
                ),
            )

            # Get data that exists
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"], timeout=180
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.is_successful(), True)
            result = response.json
            self.assertDictEqual(
                result,
                {
                    "fields": expected_fields_from_get_operation[0],
                    "id": "id:{}:{}::{}".format(
                        schema_name, schema_name, fields_to_send[0]["id"]
                    ),
                    "pathId": "/document/v1/{}/{}/docid/{}".format(
                        schema_name, schema_name, fields_to_send[0]["id"]
                    ),
                },
            )
            #
            # date data
            #
            response = await async_app.update_data(
                schema=schema_name,
                data_id=field_to_update["id"],
                fields=field_to_update,
                tracelevel=9,
            )
            result = response.json
            self.assertTrue("trace" in result)
            self.assertEqual(
                result["id"],
                "id:{}:{}::{}".format(schema_name, schema_name, field_to_update["id"]),
            )

            #
            # Get the updated data point
            #
            response = await async_app.get_data(
                schema=schema_name, data_id=field_to_update["id"]
            )
            self.assertEqual(response.status_code, 200)
            result = response.json
            if expected_fields_after_update is None:
                expected_result = {
                    k: v for k, v in expected_fields_from_get_operation[0].items()
                }
                expected_result.update(field_to_update)
            else:
                expected_result = {
                    k: v
                    for k, v in expected_fields_after_update.items()
                    if k in field_to_update
                }
            self.assertDictEqual(
                result,
                {
                    "fields": expected_result,
                    "id": "id:{}:{}::{}".format(
                        schema_name, schema_name, fields_to_send[0]["id"]
                    ),
                    "pathId": "/document/v1/{}/{}/docid/{}".format(
                        schema_name, schema_name, fields_to_send[0]["id"]
                    ),
                },
            )

            # Delete a data point
            response = await async_app.delete_data(
                schema=schema_name, data_id=fields_to_send[0]["id"], tracelevel=9
            )
            result = response.json
            self.assertTrue("trace" in result)
            self.assertEqual(
                result["id"],
                "id:{}:{}::{}".format(
                    schema_name, schema_name, fields_to_send[0]["id"]
                ),
            )

            # Deleted data should be gone
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"], tracelevel=9
            )
            self.assertEqual(response.status_code, 404)
            self.assertTrue("trace" in response.json)

            # Issue a bunch of queries in parallel
            queries = []
            for i in range(10):
                queries.append(
                    asyncio.create_task(
                        async_app.query(
                            yql="select * from sources * where true",
                            body={
                                "ranking": {
                                    "profile": "default",
                                    "listFeatures": "false",
                                },
                                "timeout": 5,
                            },
                        )
                    )
                )
            await asyncio.wait(queries, return_when=asyncio.ALL_COMPLETED)
            self.assertEqual(
                queries[0].result().number_documents_indexed, len(fields_to_send) - 1
            )
            for query in queries:
                self.assertEqual(query.result().status_code, 200)
                self.assertEqual(query.result().is_successful(), True)

    def execute_sync_partial_updates(self, app, schema_name):
        """
        Sync feed, get, update and delete data to/from the application.
        """
        with app.syncio(connections=8) as sync_app:
            # Feed data points
            for data in self.fields_to_send:
                response = sync_app.feed_data_point(
                    schema=schema_name, data_id=data["id"], fields=data
                )
                assert (
                    response.status_code == 200
                )  # Assuming you want to verify each operation immediately

            # Get and check initial data
            responses = []
            for data in self.fields_to_send:
                response = sync_app.get_data(schema=schema_name, data_id=data["id"])
                responses.append(response)
            for response, expected in zip(
                responses, self.expected_fields_from_get_operation
            ):
                assert response.status_code == 200
                assert response.json["fields"] == expected

            # Update data points
            update_responses = []
            for update in self.fields_to_update:
                response = sync_app.update_data(
                    schema=schema_name,
                    data_id=update["id"],
                    fields={k: v for k, v in update.items() if k != "auto_assign"},
                    auto_assign=update.get("auto_assign", True),
                )
                update_responses.append(response)
            for response in update_responses:
                assert response.status_code == 200

            # Verify updated data
            updated_responses = []
            for update in self.fields_to_update:
                response = sync_app.get_data(schema=schema_name, data_id=update["id"])
                updated_responses.append(response)
            for response, expected in zip(
                updated_responses, self.expected_fields_after_update
            ):
                assert response.status_code == 200
                assert response.json["fields"] == expected

            # Delete data points
            delete_responses = []
            for data in self.fields_to_send:
                response = sync_app.delete_data(schema=schema_name, data_id=data["id"])
                delete_responses.append(response)
            for response in delete_responses:
                assert (
                    response.status_code == 200
                )  # Check specific expected response code for deletion

            # Check deletion
            deletion_checks = []
            for data in self.fields_to_send:
                response = sync_app.get_data(schema=schema_name, data_id=data["id"])
                deletion_checks.append(response)
            for check in deletion_checks:
                assert (
                    check.status_code == 404
                )  # Verify that the data is indeed deleted

    async def execute_async_partial_updates(self, app, schema_name):
        """
        Async feed, get, update and delete data to/from the application.

        """

        async with app.asyncio(connections=12, total_timeout=50) as async_app:
            # Feed data points
            feed_tasks = [
                asyncio.create_task(
                    async_app.feed_data_point(
                        schema=schema_name, data_id=data["id"], fields=data
                    )
                )
                for data in self.fields_to_send
            ]
            await asyncio.gather(*feed_tasks)

            # Get and check initial data
            get_tasks = [
                asyncio.create_task(
                    async_app.get_data(schema=schema_name, data_id=data["id"])
                )
                for data in self.fields_to_send
            ]
            responses = await asyncio.gather(*get_tasks)
            for response, expected in zip(
                responses, self.expected_fields_from_get_operation
            ):
                assert response.status_code == 200
                assert response.json["fields"] == expected

            # Update data points
            update_tasks = [
                asyncio.create_task(
                    async_app.update_data(
                        schema=schema_name,
                        data_id=update["id"],
                        fields={k: v for k, v in update.items() if k != "auto_assign"},
                        auto_assign=update.get("auto_assign", True),
                    )
                )
                for update in self.fields_to_update
            ]
            await asyncio.gather(*update_tasks)
            # Check update responses
            update_responses = await asyncio.gather(*update_tasks)
            for response in update_responses:
                assert response.status_code == 200

            # Verify updated data
            check_updated_tasks = [
                asyncio.create_task(
                    async_app.get_data(schema=schema_name, data_id=update["id"])
                )
                for update in self.fields_to_update
            ]
            updated_responses = await asyncio.gather(*check_updated_tasks)
            for response, expected in zip(
                updated_responses, self.expected_fields_after_update
            ):
                assert response.status_code == 200
                assert response.json["fields"] == expected

            # Delete data points
            delete_tasks = [
                asyncio.create_task(
                    async_app.delete_data(schema=schema_name, data_id=data["id"])
                )
                for data in self.fields_to_send
            ]
            delete_responses = await asyncio.gather(*delete_tasks)
            for response in delete_responses:
                assert (
                    response.status_code == 200
                )  # Check specific expected response code for deletion

            # Check deletion
            check_deletion_tasks = [
                asyncio.create_task(
                    async_app.get_data(schema=schema_name, data_id=data["id"])
                )
                for data in self.fields_to_send
            ]
            deletion_checks = await asyncio.gather(*check_deletion_tasks)
            for check in deletion_checks:
                assert (
                    check.status_code == 404
                )  # Verify that the data is indeed deleted

    def get_model_endpoints_when_no_model_is_available(
        self, app, expected_model_endpoint
    ):
        self.assertEqual(
            app.get_model_endpoint(),
            {
                "status_code": 404,
                "message": "No binding for URI '{}'.".format(expected_model_endpoint),
            },
        )
        self.assertEqual(
            app.get_model_endpoint(model_id="bert_tiny"),
            {
                "status_code": 404,
                "message": "No binding for URI '{}bert_tiny'.".format(
                    expected_model_endpoint
                ),
            },
        )

    def get_stateless_prediction_when_model_not_defined(self, app, application_package):
        with self.assertRaisesRegex(
            ValueError, "Model named bert_tiny not defined in the application package"
        ):
            _ = app.predict("this is a test", model_id="bert_tiny")


class TestMsmarcoDockerDeployment(TestDockerCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()

    def test_deploy(self):
        self.deploy(application_package=self.app_package)

    def test_instantiate_vespa_docker_from_container_name_or_id(self):
        self.create_vespa_docker_from_container_name_or_id(
            application_package=self.app_package
        )

    @pytest.mark.skip(reason="Works locally but fails on Screwdriver")
    def test_redeploy_with_container_stopped(self):
        self.redeploy_with_container_stopped(application_package=self.app_package)

    def test_redeploy_with_application_package_changes(self):
        self.redeploy_with_application_package_changes(
            application_package=self.app_package
        )

    def test_trigger_start_stop_and_restart_services(self):
        self.trigger_start_stop_and_restart_services(
            application_package=self.app_package
        )

    def tearDown(self) -> None:
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()


class TestQaDockerDeployment(TestDockerCommon):
    def setUp(self) -> None:
        self.app_package = create_qa_application_package()

    def test_deploy(self):
        self.deploy(application_package=self.app_package)
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()

    def test_deploy_image(self):
        self.deploy(
            application_package=self.app_package,
            container_image="vespaengine/vespa",
        )
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()


class TestMsmarcoApplication(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.vespa_docker = VespaDocker(port=8089)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)
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
        self.queries_first_hit = ["this is title 1", "this is title 2"]
        self.compress_args = [True, False, "auto", None]

    def test_is_using_http2_client(self):
        asyncio.run(self.async_is_http2_client(app=self.app))

    def test_sync_client_accept_encoding(self):
        self.sync_client_accept_encoding_gzip(app=self.app)

    def test_async_client_accept_encoding(self):
        asyncio.run(self.async_client_accept_encoding_gzip(app=self.app))

    def test_handle_longlived_connection(self):
        asyncio.run(self.handle_longlived_connection(app=self.app))

    def test_model_endpoints_when_no_model_is_available(self):
        self.get_model_endpoints_when_no_model_is_available(
            app=self.app,
            expected_model_endpoint="http://localhost:8080/model-evaluation/v1/",
        )

    def test_prediction_when_model_not_defined(self):
        self.get_stateless_prediction_when_model_not_defined(
            app=self.app, application_package=self.app_package
        )

    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            cluster_name=f"{self.app_package.name}_content",
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

    def test_compress_large_feed_auto(self):
        for compress_arg in self.compress_args:
            with self.app.syncio(compress=compress_arg) as sync_app:
                response = sync_app.feed_data_point(
                    schema=self.app_package.schema.name,
                    data_id="1",
                    fields={
                        "title": "this is a title",
                        "body": "this is a body" * 1000,
                    },
                )
            self.assertEqual(response.status_code, 200)

    def test_compress_large_query_auto(self):
        for compress_arg in self.compress_args:
            with self.app.syncio(compress=compress_arg) as sync_app:
                response = sync_app.query(
                    body={
                        "yql": "select * from msmarco where userQuery();",
                        "hits": 10,
                        "query": "asdf" * 1000,
                    }
                )
            self.assertEqual(response.status_code, 200)

    def tearDown(self) -> None:
        self.app.delete_all_docs(
            content_cluster_name="content_msmarco", schema=self.app_package.name
        )
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()


class TestQaApplication(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_qa_application_package()
        self.app_package.get_schema("sentence").add_fields(
            Field(name="id", type="string", indexing=["attribute", "summary"])
        )
        self.app_package.get_schema("context").add_fields(
            Field(name="id", type="string", indexing=["attribute", "summary"])
        )
        self.vespa_docker = VespaDocker(port=8089)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)
        with open(
            os.path.join(RESOURCES_DIR, "qa_sample_sentence_data.json"),
            "r",
        ) as f:
            sample_sentence_data = json.load(f)
        self.fields_to_send_sentence = sample_sentence_data
        self.expected_fields_from_sentence_get_operation = []
        for d in sample_sentence_data:
            expected_d = {
                "id": d["id"],
                "text": d["text"],
                "dataset": d["dataset"],
                "context_id": d["context_id"],
                "sentence_embedding": {
                    "type": f"tensor<float>(x[{len(d['sentence_embedding']['values'])}])",
                    "values": d["sentence_embedding"]["values"],
                },
            }
            if len(d["questions"]) > 0:
                expected_d.update({"questions": d["questions"]})
            self.expected_fields_from_sentence_get_operation.append(expected_d)
        with open(
            os.path.join(RESOURCES_DIR, "qa_sample_context_data.json"),
            "r",
        ) as f:
            sample_context_data = json.load(f)
        self.fields_to_send_context = sample_context_data
        self.fields_to_update = [
            {"id": d["id"], "text": "this is my updated text number {}".format(d["id"])}
            for d in self.fields_to_send_sentence
        ]

    def test_model_endpoints_when_no_model_is_available(self):
        self.get_model_endpoints_when_no_model_is_available(
            app=self.app,
            expected_model_endpoint="http://localhost:8080/model-evaluation/v1/",
        )

    def test_prediction_when_model_not_defined(self):
        self.get_stateless_prediction_when_model_not_defined(
            app=self.app, application_package=self.app_package
        )

    def test_execute_data_operations_sentence_schema(self):
        self.execute_data_operations(
            app=self.app,
            schema_name="sentence",
            cluster_name="qa_content",
            fields_to_send=self.fields_to_send_sentence[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.expected_fields_from_sentence_get_operation[
                0
            ],
        )

    def test_execute_data_operations_context_schema(self):
        self.execute_data_operations(
            app=self.app,
            schema_name="context",
            cluster_name="qa_content",
            fields_to_send=self.fields_to_send_context[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.fields_to_send_context[0],
        )

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_data_operations(
                app=self.app,
                schema_name="sentence",
                fields_to_send=self.fields_to_send_sentence,
                field_to_update=self.fields_to_update[0],
                expected_fields_from_get_operation=self.expected_fields_from_sentence_get_operation,
            )
        )

    def test_feed_async_iterable(self):
        def sentence_to_doc(sentences):
            for sentence in sentences:
                yield {
                    "id": sentence["id"],
                    "fields": {k: v for k, v in sentence.items() if k != "id"},
                }

        self.app.feed_async_iterable(
            sentence_to_doc(self.fields_to_send_sentence),
            schema="sentence",
            operation_type="feed",
        )
        # check doc count
        total_docs = []
        for doc_slice in self.app.visit(
            schema="sentence", content_cluster_name="qa_content", selection="true"
        ):
            for response in doc_slice:
                total_docs.extend(response.documents)
        self.assertEqual(
            len(total_docs),
            len(self.fields_to_send_sentence),
        )
        self.app.delete_all_docs(content_cluster_name="qa_content", schema="sentence")

    def test_sync_client_accept_encoding(self):
        self.sync_client_accept_encoding_gzip(app=self.app)

    def test_async_client_accept_encoding(self):
        asyncio.run(self.async_client_accept_encoding_gzip(app=self.app))

    def tearDown(self) -> None:
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()


class TestStreamingApplication(unittest.TestCase):
    def setUp(self) -> None:
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
        mail_schema = Schema(
            name="mail",
            mode="streaming",
            document=document,
            fieldsets=[FieldSet(name="default", fields=["title", "body"])],
            rank_profiles=[
                RankProfile(name="default", first_phase="nativeRank(title, body)")
            ],
        )
        self.app_package = ApplicationPackage(name="mail", schema=[mail_schema])

        self.vespa_docker = VespaDocker(port=8089)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)

    def test_streaming(self):
        docs = [
            {
                "id": 1,
                "groupname": "a@hotmail.com",
                "fields": {"title": "this is a title", "body": "this is a body"},
            },
            {
                "id": 1,
                "groupname": "b@hotmail.com",
                "fields": {"title": "this is a title", "body": "this is a body"},
            },
            {
                "id": 2,
                "groupname": "b@hotmail.com",
                "fields": {
                    "title": "this is another title",
                    "body": "this is another body",
                },
            },
        ]
        self.app.wait_for_application_up(300)

        def callback(response: VespaResponse, id: str):
            if not response.is_successful():
                print("Id " + id + " + failed : " + response.json)

        self.app.feed_iterable(docs, schema="mail", namespace="test", callback=callback)

        response: VespaQueryResponse = self.app.query(
            yql="select * from sources * where title contains 'title'",
            groupname="a@hotmail.com",
        )
        self.assertTrue(response.is_successful())
        self.assertEqual(response.number_documents_retrieved, 1)

        response: VespaQueryResponse = self.app.query(
            yql="select * from sources * where title contains 'title'",
            groupname="b@hotmail.com",
        )
        self.assertTrue(response.is_successful())
        self.assertEqual(response.number_documents_retrieved, 2)

        with pytest.raises(Exception):
            response: VespaQueryResponse = self.app.query(
                yql="select * from sources * where title contains 'title'"
            )

        self.app.delete_data(
            schema="mail", namespace="test", data_id=2, groupname="b@hotmail.com"
        )

        response: VespaQueryResponse = self.app.query(
            yql="select * from sources * where title contains 'title'",
            groupname="b@hotmail.com",
        )
        self.assertTrue(response.is_successful())
        self.assertEqual(response.number_documents_retrieved, 1)

        self.app.update_data(
            schema="mail",
            namespace="test",
            data_id=1,
            groupname="b@hotmail.com",
            fields={"title": "this is a new foo"},
        )
        response: VespaQueryResponse = self.app.query(
            yql="select * from sources * where title contains 'foo'",
            groupname="b@hotmail.com",
        )
        self.assertTrue(response.is_successful())
        self.assertEqual(response.number_documents_retrieved, 1)

        response = self.app.get_data(
            schema="mail", namespace="test", data_id=1, groupname="b@hotmail.com"
        )
        self.assertDictEqual(
            response.json,
            {
                "pathId": "/document/v1/test/mail/group/b@hotmail.com/1",
                "id": "id:test:mail:g=b@hotmail.com:1",
                "fields": {"body": "this is a body", "title": "this is a new foo"},
            },
        )

    def tearDown(self) -> None:
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()


def create_update_application_package() -> ApplicationPackage:
    document = Document(
        structs=[
            Struct(
                name="person",
                fields=[
                    Field(name="first_name", type="string"),
                    Field(name="last_name", type="string"),
                ],
            )
        ],
        fields=[
            Field(name="id", type="string", indexing=["attribute", "summary"]),
            Field(name="title", type="string", indexing=["index", "summary"]),
            Field(name="price", type="int", indexing=["summary", "attribute"]),
            Field(name="tensorfield", type="tensor<int8>(x[10])", indexing=["summary"]),
            Field(name="contact", type="person", indexing=["summary"]),
        ],
    )
    schema = Schema(
        name="testupdates",
        document=document,
        fieldsets=[
            FieldSet(
                name="default", fields=["title", "tensorfield", "contact", "price"]
            )
        ],
        rank_profiles=[RankProfile(name="default", first_phase="nativeRank(title)")],
    )
    return ApplicationPackage(name="testupdates", schema=[schema])


class TestUpdateApplication(TestApplicationCommon):
    """Tests"""

    def setUp(self) -> None:
        self.app_package = create_update_application_package()
        self.schema_name = self.app_package.name
        self.vespa_docker = VespaDocker(port=8089)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)
        self.fields_to_send = [
            {
                "id": "1",
                "title": "this is a title",
                "tensorfield": {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                "contact": {"first_name": "John", "last_name": "Doe"},
                "price": 100,
            },
            {
                "id": "2",
                "title": "this is another title",
                "tensorfield": {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                "contact": {"first_name": "Jane", "last_name": "Doe"},
                "price": 200,
            },
            {
                "id": "3",
                "title": "this is the third title",
                "tensorfield": {
                    "x": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                },
                "contact": {"first_name": "Paul", "last_name": "Doe"},
                "price": 300,
            },
        ]
        self.expected_fields_from_get_operation = [
            {
                "id": "1",
                "title": "this is a title",
                "tensorfield": {
                    "type": "tensor<int8>(x[10])",
                    "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                "contact": {"first_name": "John", "last_name": "Doe"},
                "price": 100,
            },
            {
                "id": "2",
                "title": "this is another title",
                "tensorfield": {
                    "type": "tensor<int8>(x[10])",
                    "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                "contact": {"first_name": "Jane", "last_name": "Doe"},
                "price": 200,
            },
            {
                "id": "3",
                "title": "this is the third title",
                "tensorfield": {
                    "type": "tensor<int8>(x[10])",
                    "values": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                },
                "contact": {"first_name": "Paul", "last_name": "Doe"},
                "price": 300,
            },
        ]

        self.fields_to_update = [
            {
                "id": "1",
                "title": "this is an updated title",
            },
            {
                "id": "2",
                "tensorfield": {
                    "cells": [
                        {"address": {"x": 0}, "value": 42},
                        {"address": {"x": 9}, "value": 42},
                    ]
                },
            },
            {
                "id": "3",
                "auto_assign": False,
                "price": {
                    "increment": 1000,
                },
            },
        ]

        self.expected_fields_after_update = [
            {
                "id": "1",
                "title": "this is an updated title",
                "tensorfield": {
                    "type": "tensor<int8>(x[10])",
                    "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                "contact": {"first_name": "John", "last_name": "Doe"},
                "price": 100,
            },
            {
                "id": "2",
                "title": "this is another title",
                "tensorfield": {
                    "type": "tensor<int8>(x[10])",
                    "values": [42, 0, 0, 0, 0, 0, 0, 0, 0, 42],
                },
                "contact": {"first_name": "Jane", "last_name": "Doe"},
                "price": 200,
            },
            {
                "id": "3",
                "title": "this is the third title",
                "tensorfield": {
                    "type": "tensor<int8>(x[10])",
                    "values": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                },
                "contact": {"first_name": "Paul", "last_name": "Doe"},
                "price": 1300,
            },
        ]

        self.vespa_docker = VespaDocker(port=8089)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)

    def test_execute_sync_data_operations(self):
        self.execute_sync_partial_updates(app=self.app, schema_name=self.schema_name)

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_partial_updates(
                app=self.app, schema_name=self.schema_name
            )
        )

    def tearDown(self) -> None:
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()


class TestRetryApplication(unittest.TestCase):
    """
    Test class that simulates slow docprocessing which causes 429 errors while feeding documents to Vespa.

    Through the CustomHTTPAdapter in `application.py`, these 429 errors will be retried with an exponential backoff strategy,
    and hence not be returned to the user.

    The slow docprocessing is simulated by setting the `sleep` indexing option on the `latency` field, which then will sleep
    according to the value of the field when processing the document.
    """

    def setUp(self) -> None:
        document = Document(
            fields=[
                Field(name="id", type="string", indexing=["attribute", "summary"]),
                Field(
                    name="latency",
                    type="double",
                    indexing=["attribute", "summary", "sleep"],
                ),
            ]
        )
        schema = Schema(
            name="retryapplication",
            document=document,
        )
        self.app_package = ApplicationPackage(name="retryapplication", schema=[schema])

        self.vespa_docker = VespaDocker(port=8089)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)

    def doc_generator(self, num_docs: int):
        for i in range(num_docs):
            yield {
                "id": str(i),
                "fields": {
                    "id": str(i),
                    "latency": random.uniform(3, 4),
                },
            }

    def test_retries_sync(self):
        num_docs = 10
        num_429 = 0

        def callback(response: VespaResponse, id: str):
            nonlocal num_429
            if response.status_code == 429:
                print(f"429 response for id {id}")
                num_429 += 1

        self.app.feed_iterable(
            self.doc_generator(num_docs),
            schema="retryapplication",
            callback=callback,
        )
        self.assertEqual(num_429, 0)
        total_docs = []
        for doc_slice in self.app.visit(
            content_cluster_name="retryapplication_content",
            schema="retryapplication",
            namespace="retryapplication",
            selection="true",
        ):
            for response in doc_slice:
                total_docs.extend(response.documents)
        self.assertEqual(len(total_docs), num_docs)
        self.app.delete_all_docs(
            content_cluster_name="retryapplication_content",
            schema="retryapplication",
            namespace="retryapplication",
        )

    def test_retries_async(self):
        num_docs = 10
        num_429 = 0

        def callback(response: VespaResponse, id: str):
            nonlocal num_429
            if response.status_code == 429:
                print(f"429 response for id {id}")
                num_429 += 1

        self.app.feed_async_iterable(
            self.doc_generator(num_docs),
            schema="retryapplication",
            callback=callback,
        )
        self.assertEqual(num_429, 0)
        total_docs = []
        for doc_slice in self.app.visit(
            content_cluster_name="retryapplication_content",
            schema="retryapplication",
            namespace="retryapplication",
            selection="true",
        ):
            for response in doc_slice:
                total_docs.extend(response.documents)
        self.assertEqual(len(total_docs), num_docs)
        self.app.delete_all_docs(
            content_cluster_name="retryapplication_content",
            schema="retryapplication",
            namespace="retryapplication",
        )

    def tearDown(self) -> None:
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()
