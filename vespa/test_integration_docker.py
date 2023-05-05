# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
import pytest
import os
import re
import asyncio
import json

from requests import HTTPError

from vespa.package import (
    HNSW,
    Document,
    Field,
    Schema,
    FieldSet,
    SecondPhaseRanking,
    RankProfile,
    ApplicationPackage,
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
)
from vespa.deployment import VespaDocker
from vespa.application import VespaSync
from vespa.exceptions import VespaError


CONTAINER_STOP_TIMEOUT = 10


def create_msmarco_application_package():
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
    app_package = ApplicationPackage(name="msmarco", schema=[msmarco_schema])
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
                })

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
        self.assertTrue(self.vespa_docker._check_configuration_server())
        self.assertEqual(app.get_application_status().status_code, 200)
        self.vespa_docker.stop_services()
        self.assertFalse(self.vespa_docker._check_configuration_server())
        self.assertIsNone(app.get_application_status())
        self.vespa_docker.start_services()
        self.assertTrue(self.vespa_docker._check_configuration_server())
        self.assertEqual(app.get_application_status().status_code, 200)
        self.vespa_docker.restart_services()
        self.assertTrue(self.vespa_docker._check_configuration_server())
        self.assertEqual(app.get_application_status().status_code, 200)


class TestApplicationCommon(unittest.TestCase):
    def execute_data_operations(
        self,
        app,
        schema_name,
        fields_to_send,
        field_to_update,
        expected_fields_from_get_operation,
    ):
        """
        Feed, get, update and delete data to/from the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: Dict where keys are field names and values are field values. Must contain 'id' field
        :param field_to_update: Dict where keys are field names and values are field values.
        :param expected_fields_from_get_operation: Dict containing fields as returned by Vespa get operation.
            There are cases where fields returned from Vespa are different from inputs, e.g. when dealing with Tensors.
        :return:
        """
        assert "id" in fields_to_send, "fields_to_send must contain 'id' field."
        #
        # Get data that does not exist
        #
        with pytest.raises(HTTPError):
            app.get_data(schema=schema_name, data_id=fields_to_send["id"])

        #
        # Feed a data point
        #
        response = app.feed_data_point(
            schema=schema_name,
            data_id=fields_to_send["id"],
            fields=fields_to_send,
        )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get data that exist
        #
        response = app.get_data(schema=schema_name, data_id=fields_to_send["id"])
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
        # Update data
        #
        response = app.update_data(
            schema=schema_name,
            data_id=field_to_update["id"],
            fields=field_to_update,
        )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get the updated data point
        #
        response = app.get_data(schema=schema_name, data_id=field_to_update["id"])
        self.assertEqual(response.status_code, 200)
        expected_result = {k: v for k, v in expected_fields_from_get_operation.items()}
        expected_result.update(field_to_update)
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
        response = app.delete_data(schema=schema_name, data_id=fields_to_send["id"])
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Deleted data should be gone
        #
        with pytest.raises(HTTPError):
            app.get_data(schema=schema_name, data_id=fields_to_send["id"])

        #
        # Update a non-existent data point
        #
        response = app.update_data(
            schema=schema_name,
            data_id=field_to_update["id"],
            fields=field_to_update,
            create=True,
        )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get the updated data point
        #
        response = app.get_data(schema=schema_name, data_id=fields_to_send["id"])
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(
            response.json,
            {
                "fields": field_to_update,
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
                schema=schema_name, data_id=field_to_update["id"]
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
            )
        self.assertEqual(
            response.json["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )

    async def execute_async_data_operations(
        self,
        app,
        schema_name,
        fields_to_send,
        field_to_update,
        expected_fields_from_get_operation,
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
        async with app.asyncio(connections=120, total_timeout=50) as async_app:
            #
            # Get data that does not exist
            #
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"]
            )
            self.assertEqual(response.status_code, 404)

            #
            # Feed some data points
            #
            feed = []
            for fields in fields_to_send:
                feed.append(
                    asyncio.create_task(
                        async_app.feed_data_point(
                            schema=schema_name,
                            data_id=fields["id"],
                            fields=fields,
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

            #
            # Get data that exists
            #
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"]
            )
            self.assertEqual(response.status_code, 200)
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
            # Update data
            #
            response = await async_app.update_data(
                schema=schema_name,
                data_id=field_to_update["id"],
                fields=field_to_update,
            )
            result = response.json
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
            expected_result = {
                k: v for k, v in expected_fields_from_get_operation[0].items()
            }
            expected_result.update(field_to_update)

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
            #
            # Delete a data point
            #
            response = await async_app.delete_data(
                schema=schema_name, data_id=fields_to_send[0]["id"]
            )
            result = response.json
            self.assertEqual(
                result["id"],
                "id:{}:{}::{}".format(
                    schema_name, schema_name, fields_to_send[0]["id"]
                ),
            )
            #
            # Deleted data should be gone
            #
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"]
            )
            self.assertEqual(response.status_code, 404)
            #
            # Issue a bunch of queries in parallel
            #
            queries = []
            for i in range(10):
                queries.append(
                    asyncio.create_task(
                        async_app.query(
                            body={
                                "yql": 'select * from sources * where (userInput("sddocname:{}"))'.format(
                                    schema_name
                                ),
                                "ranking": {
                                    "profile": "default",
                                    "listFeatures": "false",
                                },
                                "timeout": 5000,
                            }
                        )
                    )
                )
            await asyncio.wait(queries, return_when=asyncio.ALL_COMPLETED)
            self.assertEqual(
                queries[0].result().number_documents_indexed, len(fields_to_send) - 1
            )

    def batch_operations_synchronous_mode(
        self,
        app,
        schema_name,
        fields_to_send,
        expected_fields_from_get_operation,
        fields_to_update,
        body_batch=None,
        hit_field_to_check=None,
        queries_first_hit=None,
    ):
        """
        Sync feed a batch of data to the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :param expected_fields_from_get_operation: Dict containing fields as returned by Vespa get operation.
            There are cases where fields returned from Vespa are different from inputs, e.g. when dealing with Tensors.
        :param fields_to_update: Dict where keys are field names and values are field values.
        :param body_batch: Optional list of query body requests.
        :param hit_field_to_check: Which field of the query response should be checked.
        :param queries_first_hit: The expected field of the first hit of each query sent
        :return:
        """

        #
        # Create and feed documents
        #
        num_docs = len(fields_to_send)
        schema = schema_name
        docs = [{"id": fields["id"], "fields": fields} for fields in fields_to_send]
        update_docs = [
            {"id": fields["id"], "fields": fields} for fields in fields_to_update
        ]

        app.feed_batch(schema=schema, batch=docs, asynchronous=False)

        #
        # Verify that all documents are fed
        #
        result = app.query(
            body={
                "yql": 'select * from sources * where (userInput("sddocname:{}"))'.format(
                    schema_name
                ),
                "ranking": {"profile": "default", "listFeatures": "false"},
            }
        )
        self.assertEqual(result.number_documents_indexed, num_docs)

        #
        # Query data
        #
        if body_batch:
            result = app.query_batch(body_batch=body_batch, asynchronous=False)
            for idx, first_hit in enumerate(queries_first_hit):
                self.assertEqual(
                    first_hit, result[idx].hits[0]["fields"][hit_field_to_check]
                )

        #
        # get batch data
        #
        result = app.get_batch(schema=schema, batch=docs, asynchronous=False)
        for idx, response in enumerate(result):
            self.assertDictEqual(
                response.json["fields"], expected_fields_from_get_operation[idx]
            )

        #
        # Update data
        #
        result = app.update_batch(schema=schema, batch=update_docs, asynchronous=False)
        for idx, response in enumerate(result):
            self.assertEqual(
                response.json["id"],
                "id:{}:{}::{}".format(schema, schema, fields_to_update[idx]["id"]),
            )

        #
        # Get updated data
        #
        result = app.get_batch(schema=schema, batch=docs, asynchronous=False)
        for idx, response in enumerate(result):
            expected_updated_fields = {
                k: v for k, v in expected_fields_from_get_operation[idx].items()
            }
            expected_updated_fields.update(fields_to_update[idx])
            self.assertDictEqual(response.json["fields"], expected_updated_fields)

        #
        # Delete data
        #
        result = app.delete_batch(schema=schema, batch=docs, asynchronous=False)
        for idx, response in enumerate(result):
            self.assertEqual(
                response.json["id"],
                "id:{}:{}::{}".format(schema, schema, docs[idx]["id"]),
            )

        #
        # get batch deleted data
        #
        with pytest.raises(HTTPError):
            app.get_batch(schema=schema, batch=docs, asynchronous=False)

    def batch_operations_asynchronous_mode(
        self,
        app,
        schema_name,
        fields_to_send,
        expected_fields_from_get_operation,
        fields_to_update,
        body_batch=None,
        hit_field_to_check=None,
        queries_first_hit=None,
    ):
        """
        Async feed a batch of data to the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :param expected_fields_from_get_operation: Dict containing fields as returned by Vespa get operation.
            There are cases where fields returned from Vespa are different from inputs, e.g. when dealing with Tensors.
        :param fields_to_update: Dict where keys are field names and values are field values.
        :param body_batch: Optional list of query body.
        :param hit_field_to_check: Which field of the query response should be checked.
        :param queries_first_hit: The expected field of the first hit of each query sent
        :return:
        """
        #
        # Create and feed documents
        #
        num_docs = len(fields_to_send)
        schema = schema_name
        docs = [{"id": fields["id"], "fields": fields} for fields in fields_to_send]
        update_docs = [
            {"id": fields["id"], "fields": fields} for fields in fields_to_update
        ]

        app.feed_batch(
            schema=schema,
            batch=docs,
            asynchronous=True,
            connections=120,
            total_timeout=50,
        )

        #
        # Verify that all documents are fed
        #
        result = app.query(
            body={
                "yql": 'select * from sources * where (userInput("sddocname:{}"))'.format(
                    schema_name
                ),
                "ranking": {"profile": "default", "listFeatures": "false"},
            }
        )
        self.assertEqual(result.number_documents_indexed, num_docs)

        #
        # Query data
        #
        if body_batch:
            result = app.query_batch(body_batch=body_batch)
            for idx, first_hit in enumerate(queries_first_hit):
                self.assertEqual(
                    first_hit, result[idx].hits[0]["fields"][hit_field_to_check]
                )

        #
        # get batch data
        #
        result = app.get_batch(schema=schema, batch=docs, asynchronous=True)
        for idx, response in enumerate(result):
            self.assertDictEqual(
                response.json["fields"], expected_fields_from_get_operation[idx]
            )

        #
        # Update data
        #
        result = app.update_batch(schema=schema, batch=update_docs, asynchronous=True)
        for idx, response in enumerate(result):
            self.assertEqual(
                response.json["id"],
                "id:{}:{}::{}".format(schema, schema, fields_to_update[idx]["id"]),
            )

        #
        # Get updated data
        #
        result = app.get_batch(schema=schema, batch=docs, asynchronous=True)
        for idx, response in enumerate(result):
            expected_updated_fields = {
                k: v for k, v in expected_fields_from_get_operation[idx].items()
            }
            expected_updated_fields.update(fields_to_update[idx])
            self.assertDictEqual(response.json["fields"], expected_updated_fields)

        #
        # Delete data
        #
        result = app.delete_batch(schema=schema, batch=docs, asynchronous=True)
        for idx, response in enumerate(result):
            self.assertEqual(
                response.json["id"],
                "id:{}:{}::{}".format(schema, schema, docs[idx]["id"]),
            )

        #
        # get batch deleted data
        #
        result = app.get_batch(schema=schema, batch=docs, asynchronous=True)
        for idx, response in enumerate(result):
            self.assertEqual(response.status_code, 404)

    def batch_operations_default_mode_with_one_schema(
        self,
        app,
        schema_name,
        fields_to_send,
        expected_fields_from_get_operation,
        fields_to_update,
    ):
        """
        Document batch operations for applications with one schema

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :param expected_fields_from_get_operation: Dict containing fields as returned by Vespa get operation.
            There are cases where fields returned from Vespa are different from inputs, e.g. when dealing with Tensors.
        :param fields_to_update: Dict where keys are field names and values are field values.
        :return:
        """
        #
        # Create and feed documents
        #
        num_docs = len(fields_to_send)
        schema = schema_name
        docs = [{"id": fields["id"], "fields": fields} for fields in fields_to_send]
        update_docs = [
            {"id": fields["id"], "fields": fields} for fields in fields_to_update
        ]

        app.feed_batch(batch=docs)

        #
        # Verify that all documents are fed
        #
        result = app.query(
            body={
                "yql": 'select * from sources * where (userInput("sddocname:{}"))'.format(
                    schema_name
                ),
                "ranking": {"profile": "default", "listFeatures": "false"},
            }
        )
        self.assertEqual(result.number_documents_indexed, num_docs)

        #
        # get batch data
        #
        result = app.get_batch(batch=docs)
        for idx, response in enumerate(result):
            self.assertDictEqual(
                response.json["fields"], expected_fields_from_get_operation[idx]
            )

        #
        # Update data
        #
        result = app.update_batch(batch=update_docs)
        for idx, response in enumerate(result):
            self.assertEqual(
                response.json["id"],
                "id:{}:{}::{}".format(schema, schema, fields_to_update[idx]["id"]),
            )

        #
        # Get updated data
        #
        result = app.get_batch(batch=docs)
        for idx, response in enumerate(result):
            expected_updated_fields = {
                k: v for k, v in expected_fields_from_get_operation[idx].items()
            }
            expected_updated_fields.update(fields_to_update[idx])
            self.assertDictEqual(response.json["fields"], expected_updated_fields)

        #
        # Delete data
        #
        result = app.delete_batch(batch=docs)
        for idx, response in enumerate(result):
            self.assertEqual(
                response.json["id"],
                "id:{}:{}::{}".format(schema, schema, docs[idx]["id"]),
            )

        #
        # get batch deleted data
        #
        result = app.get_batch(batch=docs)
        for idx, response in enumerate(result):
            self.assertEqual(response.status_code, 404)

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
        ) as exc:
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
        self.body_batch = [
            {
                "yql": 'select * from sources * where ({grammar: "any"}userInput("Give me title 1"))',
                "ranking": {"profile": "default", "listFeatures": "false"},
            },
            {
                "yql": 'select * from sources * where ({grammar: "any"}userInput("Give me title 2"))',
                "ranking": {"profile": "default", "listFeatures": "false"},
            },
        ]
        self.queries_first_hit = ["this is title 1", "this is title 2"]

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

    def test_batch_operations_synchronous_mode(self):
        self.batch_operations_synchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.fields_to_send,
            fields_to_update=self.fields_to_update,
            body_batch=self.body_batch,
            hit_field_to_check="title",
            queries_first_hit=self.queries_first_hit,
        )

    def test_batch_operations_asynchronous_mode(self):
        self.batch_operations_asynchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.fields_to_send,
            fields_to_update=self.fields_to_update,
            body_batch=self.body_batch,
            hit_field_to_check="title",
            queries_first_hit=self.queries_first_hit,
        )

    def test_batch_operations_default_mode_with_one_schema(self):
        self.batch_operations_default_mode_with_one_schema(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.fields_to_send,
            fields_to_update=self.fields_to_update,
        )

    def tearDown(self) -> None:
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
            os.path.join(os.environ["RESOURCES_DIR"], "qa_sample_sentence_data.json"),
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
            os.path.join(os.environ["RESOURCES_DIR"], "qa_sample_context_data.json"),
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

    def test_batch_operations_synchronous_mode(self):
        self.batch_operations_synchronous_mode(
            app=self.app,
            schema_name="sentence",
            fields_to_send=self.fields_to_send_sentence,
            expected_fields_from_get_operation=self.expected_fields_from_sentence_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def test_batch_operations_asynchronous_mode(self):
        self.batch_operations_asynchronous_mode(
            app=self.app,
            schema_name="sentence",
            fields_to_send=self.fields_to_send_sentence,
            expected_fields_from_get_operation=self.expected_fields_from_sentence_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def tearDown(self) -> None:
        self.vespa_docker.container.stop(timeout=CONTAINER_STOP_TIMEOUT)
        self.vespa_docker.container.remove()
