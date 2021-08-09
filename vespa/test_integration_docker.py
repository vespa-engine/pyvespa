import unittest
import os
import re
import shutil
import asyncio
import json
from vespa.package import (
    HNSW,
    Document,
    Field,
    Schema,
    FieldSet,
    SecondPhaseRanking,
    RankProfile,
    ApplicationPackage,
    VespaDocker,
)
from vespa.ml import BertModelConfig
from vespa.query import QueryModel, RankProfile as Ranking, OR, QueryRankingFeature
from vespa.gallery import QuestionAnswering
from vespa.application import VespaSync


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


def create_cord19_application_package():
    app_package = ApplicationPackage(name="cord19")
    app_package.schema.add_fields(
        Field(name="id", type="string", indexing=["attribute", "summary"]),
        Field(
            name="title",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
    )
    app_package.schema.add_field_set(FieldSet(name="default", fields=["title"]))
    app_package.schema.add_rank_profile(
        RankProfile(name="bm25", first_phase="bm25(title)")
    )
    bert_config = BertModelConfig(
        model_id="pretrained_bert_tiny",
        tokenizer="google/bert_uncased_L-2_H-128_A-2",
        model="google/bert_uncased_L-2_H-128_A-2",
        query_input_size=5,
        doc_input_size=10,
    )
    app_package.add_model_ranking(
        model_config=bert_config,
        include_model_summary_features=True,
        inherits="default",
        first_phase="bm25(title)",
        second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
    )
    return app_package


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
    def deploy(self, application_package, disk_folder):
        self.vespa_docker = VespaDocker(port=8089, disk_folder=disk_folder)
        app = self.vespa_docker.deploy(application_package=application_package)
        #
        # Test deployment
        #
        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )
        self.assertEqual(app.get_application_status().status_code, 200)
        #
        # Test VespaDocker serialization
        #
        self.assertEqual(
            self.vespa_docker, VespaDocker.from_dict(self.vespa_docker.to_dict)
        )

    def deploy_from_disk_with_disk_folder(self, application_package, disk_folder):
        self.vespa_docker = VespaDocker(port=8089, disk_folder=disk_folder)
        self.vespa_docker.export_application_package(
            application_package=application_package
        )
        #
        # Disk folder as the application folder
        #
        self.vespa_docker.disk_folder = os.path.join(disk_folder, "application")
        app = self.vespa_docker.deploy_from_disk(
            application_name=application_package.name,
        )
        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )

    def deploy_from_disk_with_application_folder(
        self, application_package, disk_folder
    ):
        self.vespa_docker = VespaDocker(port=8089, disk_folder=disk_folder)
        self.vespa_docker.export_application_package(
            application_package=application_package
        )
        #
        # Application folder inside disk folder
        #
        app = self.vespa_docker.deploy_from_disk(
            application_name=application_package.name,
            application_folder="application",
        )
        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )

    def create_vespa_docker_from_container_name_or_id(
        self, application_package, disk_folder
    ):
        #
        # Raises ValueError if container does not exist
        #
        with self.assertRaises(ValueError):
            _ = VespaDocker.from_container_name_or_id(application_package.name)
        #
        # Test VespaDocker instance created from container
        #
        self.vespa_docker = VespaDocker(port=8089, disk_folder=disk_folder)
        _ = self.vespa_docker.deploy(application_package=application_package)
        vespa_docker_from_container = VespaDocker.from_container_name_or_id(
            application_package.name
        )
        self.assertEqual(self.vespa_docker, vespa_docker_from_container)

    def redeploy_with_container_stopped(self, application_package, disk_folder):
        self.vespa_docker = VespaDocker(port=8089, disk_folder=disk_folder)
        app = self.vespa_docker.deploy(application_package=application_package)
        self.assertTrue(
            any(re.match("Generation: [0-9]+", line) for line in app.deployment_message)
        )
        self.vespa_docker.container.stop()
        app = self.vespa_docker.deploy(application_package=application_package)
        self.assertEqual(app.get_application_status().status_code, 200)

    def redeploy_with_application_package_changes(
        self, application_package, disk_folder
    ):
        self.vespa_docker = VespaDocker(port=8089, disk_folder=disk_folder)
        app = self.vespa_docker.deploy(application_package=application_package)
        res = app.query(
            body={
                "yql": "select * from sources * where default contains 'music';",
                "ranking": "new-rank-profile",
            }
        ).json
        self.assertIsNotNone(
            re.search(
                "Requested rank profile 'new-rank-profile' is undefined for document type ",
                res["root"]["errors"][0]["message"],
            )
        )
        application_package.schema.add_rank_profile(
            RankProfile(
                name="new-rank-profile", inherits="default", first_phase="bm25(title)"
            )
        )
        app = self.vespa_docker.deploy(application_package=application_package)
        res = app.query(
            body={
                "yql": "select * from sources * where default contains 'music';",
                "ranking": "new-rank-profile",
            }
        ).json
        self.assertTrue("errors" not in res["root"])

    def trigger_start_stop_and_restart_services(self, application_package, disk_folder):
        self.vespa_docker = VespaDocker(port=8089, disk_folder=disk_folder)
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
            There are cases where fields returned from Vespa are different than inputs, e.g. when dealing with Tensors.
        :return:
        """
        assert "id" in fields_to_send, "fields_to_send must contain 'id' field."
        #
        # Get data that does not exist
        #
        self.assertEqual(
            app.get_data(schema=schema_name, data_id=fields_to_send["id"]).status_code,
            404,
        )
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
        self.assertEqual(
            app.get_data(schema=schema_name, data_id=fields_to_send["id"]).status_code,
            404,
        )
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
            There are cases where fields returned from Vespa are different than inputs, e.g. when dealing with Tensors.
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
                    schema="msmarco",
                    data_id="1",
                    fields={
                        "id": "1",
                        "title": "this is title 1",
                        "body": "this is body 1",
                    },
                ),
                app.feed_data_point(
                    schema="msmarco",
                    data_id="1",
                    fields={
                        "id": "1",
                        "title": "this is title 1",
                        "body": "this is body 1",
                    },
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
                            query="sddocname:{}".format(schema_name),
                            query_model=QueryModel(),
                            timeout=5000,
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
    ):
        """
        Sync feed a batch of data to the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :param expected_fields_from_get_operation: Dict containing fields as returned by Vespa get operation.
            There are cases where fields returned from Vespa are different than inputs, e.g. when dealing with Tensors.
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

        app.feed_batch(schema=schema, batch=docs, asynchronous=False)

        #
        # Verify that all documents are fed
        #
        result = app.query(
            query="sddocname:{}".format(schema_name), query_model=QueryModel()
        )
        self.assertEqual(result.number_documents_indexed, num_docs)

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
        result = app.get_batch(schema=schema, batch=docs, asynchronous=False)
        for idx, response in enumerate(result):
            self.assertEqual(response.status_code, 404)

    def batch_operations_asynchronous_mode(
        self,
        app,
        schema_name,
        fields_to_send,
        expected_fields_from_get_operation,
        fields_to_update,
    ):
        """
        Async feed a batch of data to the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :param expected_fields_from_get_operation: Dict containing fields as returned by Vespa get operation.
            There are cases where fields returned from Vespa are different than inputs, e.g. when dealing with Tensors.
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
            query="sddocname:{}".format(schema_name), query_model=QueryModel()
        )
        self.assertEqual(result.number_documents_indexed, num_docs)

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

    @staticmethod
    def _parse_vespa_tensor(hit, feature):
        return [x["value"] for x in hit["fields"]["summaryfeatures"][feature]["cells"]]

    def bert_model_input_and_output(
        self, app, schema_name, fields_to_send, model_config
    ):
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
        # Run a test query
        #
        result = app.query(
            query="this is a test",
            query_model=QueryModel(
                query_properties=[
                    QueryRankingFeature(
                        name=model_config.query_token_ids_name,
                        mapping=model_config.query_tensor_mapping,
                    )
                ],
                match_phase=OR(),
                rank_profile=Ranking(name="pretrained_bert_tiny"),
            ),
        )
        vespa_input_ids = self._parse_vespa_tensor(
            result.hits[0], "rankingExpression(input_ids)"
        )
        vespa_attention_mask = self._parse_vespa_tensor(
            result.hits[0], "rankingExpression(attention_mask)"
        )
        vespa_token_type_ids = self._parse_vespa_tensor(
            result.hits[0], "rankingExpression(token_type_ids)"
        )

        expected_inputs = model_config.create_encodings(
            queries=["this is a test"], docs=[fields_to_send["title"]]
        )
        self.assertEqual(vespa_input_ids, expected_inputs["input_ids"][0])
        self.assertEqual(vespa_attention_mask, expected_inputs["attention_mask"][0])
        self.assertEqual(vespa_token_type_ids, expected_inputs["token_type_ids"][0])

        expected_logits = model_config.predict(
            queries=["this is a test"], docs=[fields_to_send["title"]]
        )
        self.assertAlmostEqual(
            result.hits[0]["fields"]["summaryfeatures"]["rankingExpression(logit0)"],
            expected_logits[0][0],
            5,
        )
        self.assertAlmostEqual(
            result.hits[0]["fields"]["summaryfeatures"]["rankingExpression(logit1)"],
            expected_logits[0][1],
            5,
        )


class TestMsmarcoDockerDeployment(TestDockerCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")

    def test_deploy(self):
        self.deploy(application_package=self.app_package, disk_folder=self.disk_folder)

    def test_deploy_from_disk_with_disk_folder(self):
        self.deploy_from_disk_with_disk_folder(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_deploy_from_disk_with_application_folder(self):
        self.deploy_from_disk_with_application_folder(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_instantiate_vespa_docker_from_container_name_or_id(self):
        self.create_vespa_docker_from_container_name_or_id(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_redeploy_with_container_stopped(self):
        self.redeploy_with_container_stopped(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_redeploy_with_application_package_changes(self):
        self.redeploy_with_application_package_changes(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def test_trigger_start_stop_and_restart_services(self):
        self.trigger_start_stop_and_restart_services(
            application_package=self.app_package, disk_folder=self.disk_folder
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()


class TestCord19DockerDeployment(TestDockerCommon):
    def setUp(self) -> None:
        self.app_package = create_cord19_application_package()
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")

    def test_deploy(self):
        self.deploy(application_package=self.app_package, disk_folder=self.disk_folder)

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()


class TestQaDockerDeployment(TestDockerCommon):
    def setUp(self) -> None:
        self.app_package = create_qa_application_package()
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")

    def test_deploy(self):
        self.deploy(application_package=self.app_package, disk_folder=self.disk_folder)

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()


class TestMsmarcoApplication(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_msmarco_application_package()
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.vespa_docker = VespaDocker(port=8089, disk_folder=self.disk_folder)
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
        )

    def test_batch_operations_asynchronous_mode(self):
        self.batch_operations_asynchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.fields_to_send,
            fields_to_update=self.fields_to_update,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()


class TestCord19Application(TestApplicationCommon):
    def setUp(self) -> None:
        self.app_package = create_cord19_application_package()
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.vespa_docker = VespaDocker(port=8089, disk_folder=self.disk_folder)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)
        self.model_config = self.app_package.model_configs["pretrained_bert_tiny"]
        self.fields_to_send = []
        self.expected_fields_from_get_operation = []
        for i in range(10):
            fields = {
                "id": f"{i}",
                "title": f"this is title {i}",
            }
            tensor_field_dict = self.model_config.doc_fields(text=str(fields["title"]))
            fields.update(tensor_field_dict)
            self.fields_to_send.append(fields)

            expected_fields = {
                "id": f"{i}",
                "title": f"this is title {i}",
            }
            tensor_field_values = tensor_field_dict[
                "pretrained_bert_tiny_doc_token_ids"
            ]["values"]
            expected_fields.update(
                {
                    "pretrained_bert_tiny_doc_token_ids": {
                        "cells": [
                            {
                                "address": {"d0": str(x)},
                                "value": float(tensor_field_values[x]),
                            }
                            for x in range(len(tensor_field_values))
                        ]
                    }
                }
            )
            self.expected_fields_from_get_operation.append(expected_fields)
        self.fields_to_update = [
            {
                "id": f"{i}",
                "title": "this is my updated title number {}".format(i),
            }
            for i in range(10)
        ]

    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            field_to_update=self.fields_to_update[0],
            expected_fields_from_get_operation=self.expected_fields_from_get_operation[
                0
            ],
        )

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_data_operations(
                app=self.app,
                schema_name=self.app_package.name,
                fields_to_send=self.fields_to_send,
                field_to_update=self.fields_to_update[0],
                expected_fields_from_get_operation=self.expected_fields_from_get_operation,
            )
        )

    def test_batch_operations_synchronous_mode(self):
        self.batch_operations_synchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.expected_fields_from_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def test_batch_operations_asynchronous_mode(self):
        self.batch_operations_asynchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
            expected_fields_from_get_operation=self.expected_fields_from_get_operation,
            fields_to_update=self.fields_to_update,
        )

    def test_bert_model_input_and_output(self):
        self.bert_model_input_and_output(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            model_config=self.model_config,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
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
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.vespa_docker = VespaDocker(port=8089, disk_folder=self.disk_folder)
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
                    "cells": [
                        {"address": {"x": str(idx)}, "value": value}
                        for idx, value in enumerate(d["sentence_embedding"]["values"])
                    ]
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
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()
