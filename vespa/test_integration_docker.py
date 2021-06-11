import unittest
import os
import re
import shutil
import asyncio
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
        self, app, schema_name, fields_to_send, fields_to_update
    ):
        """
        Feed, get, update and delete data to/from the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: Dict where keys are field names and values are field values. Must contain 'id' field
        :param fields_to_update: Dict where keys are field names and values are field values.
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
            response.json()["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get data that exist
        #
        response = app.get_data(schema=schema_name, data_id=fields_to_send["id"])
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(
            response.json(),
            {
                "fields": fields_to_send,
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
            data_id=fields_to_send["id"],
            fields=fields_to_update,
        )
        self.assertEqual(
            response.json()["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get the updated data point
        #
        response = app.get_data(schema=schema_name, data_id=fields_to_send["id"])
        self.assertEqual(response.status_code, 200)
        expected_result = {k: v for k, v in fields_to_send.items()}
        expected_result.update(fields_to_update)
        self.assertDictEqual(
            response.json(),
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
            response.json()["id"],
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
            data_id=fields_to_send["id"],
            fields=fields_to_update,
            create=True,
        )
        self.assertEqual(
            response.json()["id"],
            "id:{}:{}::{}".format(schema_name, schema_name, fields_to_send["id"]),
        )
        #
        # Get the updated data point
        #
        response = app.get_data(schema=schema_name, data_id=fields_to_send["id"])
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(
            response.json(),
            {
                "fields": fields_to_update,
                "id": "id:{}:{}::{}".format(
                    schema_name, schema_name, fields_to_send["id"]
                ),
                "pathId": "/document/v1/{}/{}/docid/{}".format(
                    schema_name, schema_name, fields_to_send["id"]
                ),
            },
        )

    async def execute_async_data_operations(
        self, app, schema_name, fields_to_send, fields_to_update
    ):
        """
        Async feed, get, update and delete data to/from the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :param fields_to_update: Dict where keys are field names and values are field values.
        :return:
        """
        async with app.asyncio() as async_app:
            #
            # Get data that does not exist
            #
            # response = await async_app.delete_data(
            #     schema=schema_name, data_id=fields_to_send[0]["id"]
            # )
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"]
            )
            self.assertEqual(response.status, 404)

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
            result = await feed[0].result().json()
            self.assertEqual(
                result["id"],
                "id:{}:{}::{}".format(
                    schema_name, schema_name, fields_to_send[0]["id"]
                ),
            )

            #
            # Get data that exists
            #
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"]
            )
            self.assertEqual(response.status, 200)
            result = await response.json()
            self.assertDictEqual(
                result,
                {
                    "fields": fields_to_send[0],
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
                data_id=fields_to_send[0]["id"],
                fields=fields_to_update,
            )
            result = await response.json()
            self.assertEqual(
                result["id"],
                "id:{}:{}::{}".format(
                    schema_name, schema_name, fields_to_send[0]["id"]
                ),
            )

            #
            # Get the updated data point
            #
            response = await async_app.get_data(
                schema=schema_name, data_id=fields_to_send[0]["id"]
            )
            self.assertEqual(response.status, 200)
            result = await response.json()
            expected_result = {k: v for k, v in fields_to_send[0].items()}
            expected_result.update(fields_to_update)

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
            result = await response.json()
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
            self.assertEqual(response.status, 404)
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

    def feed_batch_synchronous_mode(self, app, schema_name, fields_to_send):
        """
        Sync feed a batch of data to the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :return:
        """

        #
        # Create and feed documents
        #
        num_docs = len(fields_to_send)
        docs = []
        schema = schema_name
        for fields in fields_to_send:
            docs.append({"id": fields["id"], "fields": fields})
        app.feed_batch(schema=schema, batch=docs, asynchronous=False)

        # Verify that all documents are fed
        result = app.query(
            query="sddocname:{}".format(schema_name), query_model=QueryModel()
        )
        self.assertEqual(result.number_documents_indexed, num_docs)

    def feed_batch_asynchronous_mode(self, app, schema_name, fields_to_send):
        """
        Async feed a batch of data to the application

        :param app: Vespa instance holding the connection to the application
        :param schema_name: Schema name containing the document we want to send and retrieve data
        :param fields_to_send: List of Dicts where keys are field names and values are field values. Must
            contain 'id' field.
        :return:
        """
        #
        # Create and feed documents
        #
        num_docs = len(fields_to_send)
        docs = []
        schema = schema_name
        for fields in fields_to_send:
            docs.append({"id": fields["id"], "fields": fields})
        app.feed_batch(schema=schema, batch=docs, asynchronous=True)

        # Verify that all documents are fed
        result = app.query(
            query="sddocname:{}".format(schema_name), query_model=QueryModel()
        )
        self.assertEqual(result.number_documents_indexed, num_docs)


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
        self.fields_to_update = {"title": "this is my updated title"}

    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            fields_to_update=self.fields_to_update,
        )

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_data_operations(
                app=self.app,
                schema_name=self.app_package.name,
                fields_to_send=self.fields_to_send,
                fields_to_update=self.fields_to_update,
            )
        )

    def test_feed_batch_synchronous_mode(self):
        self.feed_batch_synchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
        )

    def test_feed_batch_asynchronous_mode(self):
        self.feed_batch_asynchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
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
        self.fields_to_send = [
            {
                "id": f"{i}",
                "title": f"this is title {i}",
            }
            for i in range(10)
        ]
        self.fields_to_update = {"title": "this is my updated title"}

    def test_execute_data_operations(self):
        self.execute_data_operations(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send[0],
            fields_to_update=self.fields_to_update,
        )

    def test_execute_async_data_operations(self):
        asyncio.run(
            self.execute_async_data_operations(
                app=self.app,
                schema_name=self.app_package.name,
                fields_to_send=self.fields_to_send,
                fields_to_update=self.fields_to_update,
            )
        )

    def test_feed_batch_synchronous_mode(self):
        self.feed_batch_synchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
        )

    def test_feed_batch_asynchronous_mode(self):
        self.feed_batch_asynchronous_mode(
            app=self.app,
            schema_name=self.app_package.name,
            fields_to_send=self.fields_to_send,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()


class TestOnnxModelDockerDeployment(unittest.TestCase):
    def setUp(self) -> None:
        #
        # Create application package
        #
        self.app_package = ApplicationPackage(name="cord19")
        self.app_package.schema.add_fields(
            Field(name="cord_uid", type="string", indexing=["attribute", "summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
        )
        self.app_package.schema.add_field_set(
            FieldSet(name="default", fields=["title"])
        )
        self.app_package.schema.add_rank_profile(
            RankProfile(name="bm25", first_phase="bm25(title)")
        )
        self.bert_config = BertModelConfig(
            model_id="pretrained_bert_tiny",
            tokenizer="google/bert_uncased_L-2_H-128_A-2",
            model="google/bert_uncased_L-2_H-128_A-2",
            query_input_size=5,
            doc_input_size=10,
        )
        self.app_package.add_model_ranking(
            model_config=self.bert_config,
            include_model_summary_features=True,
            inherits="default",
            first_phase="bm25(title)",
            second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
        )
        self.disk_folder = os.path.join(os.getenv("WORK_DIR"), "sample_application")
        self.vespa_docker = VespaDocker(port=8089, disk_folder=self.disk_folder)
        self.app = self.vespa_docker.deploy(application_package=self.app_package)

    def test_data_operation(self):
        #
        # Get data that does not exist
        #
        self.assertEqual(
            self.app.get_data(schema="cord19", data_id="1").status_code, 404
        )
        #
        # Feed a data point
        #
        fields = {
            "cord_uid": "1",
            "title": "this is my first title",
        }
        fields.update(self.bert_config.doc_fields(text=str(fields["title"])))
        response = self.app.feed_data_point(
            schema="cord19",
            data_id="1",
            fields=fields,
        )
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Get data that exist
        #
        response = self.app.get_data(schema="cord19", data_id="1")
        self.assertEqual(response.status_code, 200)
        embedding_values = fields["pretrained_bert_tiny_doc_token_ids"]["values"]
        self.assertDictEqual(
            response.json(),
            {
                "fields": {
                    "cord_uid": "1",
                    "title": "this is my first title",
                    "pretrained_bert_tiny_doc_token_ids": {
                        "cells": [
                            {
                                "address": {"d0": str(x)},
                                "value": float(embedding_values[x]),
                            }
                            for x in range(len(embedding_values))
                        ]
                    },
                },
                "id": "id:cord19:cord19::1",
                "pathId": "/document/v1/cord19/cord19/docid/1",
            },
        )
        #
        # Update data
        #
        fields = {"title": "this is my updated title"}
        fields.update(self.bert_config.doc_fields(text=str(fields["title"])))
        response = self.app.update_data(schema="cord19", data_id="1", fields=fields)
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Get the updated data point
        #
        response = self.app.get_data(schema="cord19", data_id="1")
        self.assertEqual(response.status_code, 200)
        embedding_values = fields["pretrained_bert_tiny_doc_token_ids"]["values"]
        self.assertDictEqual(
            response.json(),
            {
                "fields": {
                    "cord_uid": "1",
                    "title": "this is my updated title",
                    "pretrained_bert_tiny_doc_token_ids": {
                        "cells": [
                            {
                                "address": {"d0": str(x)},
                                "value": float(embedding_values[x]),
                            }
                            for x in range(len(embedding_values))
                        ]
                    },
                },
                "id": "id:cord19:cord19::1",
                "pathId": "/document/v1/cord19/cord19/docid/1",
            },
        )
        #
        # Delete a data point
        #
        response = self.app.delete_data(schema="cord19", data_id="1")
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Deleted data should be gone
        #
        self.assertEqual(
            self.app.get_data(schema="cord19", data_id="1").status_code, 404
        )

    def _parse_vespa_tensor(self, hit, feature):
        return [x["value"] for x in hit["fields"]["summaryfeatures"][feature]["cells"]]

    def test_rank_input_output(self):
        #
        # Feed a data point
        #
        fields = {
            "cord_uid": "1",
            "title": "this is my first title",
        }
        fields.update(self.bert_config.doc_fields(text=str(fields["title"])))
        response = self.app.feed_data_point(
            schema="cord19",
            data_id="1",
            fields=fields,
        )
        self.assertEqual(response.json()["id"], "id:cord19:cord19::1")
        #
        # Run a test query
        #
        result = self.app.query(
            query="this is a test",
            query_model=QueryModel(
                query_properties=[
                    QueryRankingFeature(
                        name=self.bert_config.query_token_ids_name,
                        mapping=self.bert_config.query_tensor_mapping,
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

        expected_inputs = self.bert_config.create_encodings(
            queries=["this is a test"], docs=["this is my first title"]
        )
        self.assertEqual(vespa_input_ids, expected_inputs["input_ids"][0])
        self.assertEqual(vespa_attention_mask, expected_inputs["attention_mask"][0])
        self.assertEqual(vespa_token_type_ids, expected_inputs["token_type_ids"][0])

        expected_logits = self.bert_config.predict(
            queries=["this is a test"], docs=["this is my first title"]
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

    def tearDown(self) -> None:
        shutil.rmtree(self.disk_folder, ignore_errors=True)
        self.vespa_docker.container.stop()
        self.vespa_docker.container.remove()
