# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import unittest

import pytest
from unittest.mock import PropertyMock, patch
from requests.models import HTTPError, Response

from vespa.package import ApplicationPackage, Schema, Document
from vespa.application import Vespa, raise_for_status
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse, VespaResponse
import requests_mock
from unittest.mock import MagicMock
from vespa.application import VespaSync, VespaAsync
import sys


class TestVespaRequestsUsage(unittest.TestCase):
    def test_additional_query_params(self):
        app = Vespa(url="http://localhost", port=8080)
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8080/search/", status_code=200, text="{}")
            r: VespaQueryResponse = app.query(
                query="this is a test", hits=10, searchChain="default"
            )
            self.assertEqual(
                r.url,
                "http://localhost:8080/search/?query=this+is+a+test&hits=10&searchChain=default",
            )

    def test_additional_doc_params(self):
        app = Vespa(url="http://localhost", port=8080)
        with requests_mock.Mocker() as m:
            m.post(
                "http://localhost:8080/document/v1/foo/foo/docid/0",
                status_code=200,
                text="{}",
            )
            r: VespaResponse = app.feed_data_point(
                schema="foo",
                data_id="0",
                fields={"body": "this is a test"},
                route="default",
                timeout="10s",
            )
            self.assertEqual(
                r.url,
                "http://localhost:8080/document/v1/foo/foo/docid/0?route=default&timeout=10s",
            )

        with requests_mock.Mocker() as m:
            m.put(
                "http://localhost:8080/document/v1/foo/foo/docid/0",
                status_code=200,
                text="{}",
            )
            r: VespaResponse = app.update_data(
                schema="foo",
                data_id="0",
                fields={"body": "this is a test"},
                route="default",
                timeout="10s",
            )
            self.assertEqual(
                r.url,
                "http://localhost:8080/document/v1/foo/foo/docid/0?create=false&route=default&timeout=10s",
            )

        with requests_mock.Mocker() as m:
            m.delete(
                "http://localhost:8080/document/v1/foo/foo/docid/0",
                status_code=200,
                text="{}",
            )
            r: VespaResponse = app.delete_data(
                schema="foo", data_id="0", route="default", timeout="10s", dryRun=True
            )
            self.assertEqual(
                r.url,
                "http://localhost:8080/document/v1/foo/foo/docid/0?route=default&timeout=10s&dryRun=True",
            )

    def test_delete_all_docs(self):
        app = Vespa(url="http://localhost", port=8080)
        with requests_mock.Mocker() as m:
            m.delete(
                "http://localhost:8080/document/v1/foo/foo/docid/",
                status_code=200,
                text="{}",
            )
            app.delete_all_docs(
                schema="foo",
                namespace="foo",
                content_cluster_name="content",
                timeout="200s",
            )


class TestVespa(unittest.TestCase):
    def test_init(self):
        vespa = Vespa(url="http://localhost", port=8080)
        self.assertEqual(vespa.url, "http://localhost")
        self.assertEqual(vespa.port, 8080)
        self.assertIsNone(vespa.deployment_message)
        self.assertIsNone(vespa.cert)
        self.assertIsNone(vespa.key)
        self.assertIsNone(vespa.vespa_cloud_secret_token)
        self.assertIs(sys.stdout, vespa.output_file)
        self.assertIsNone(vespa._application_package)
        self.assertEqual(vespa.end_point, "http://localhost:8080")
        self.assertEqual(vespa.search_end_point, "http://localhost:8080/search/")

    def test_validate_operation_type(self):
        vespa = Vespa(url="http://localhost", port=8080)
        vespa.validate_operation_type("feed")
        vespa.validate_operation_type("update")
        vespa.validate_operation_type("delete")
        with self.assertRaises(ValueError):
            vespa.validate_operation_type("foo")

    def test_get_namespace(self):
        vespa = Vespa(url="http://localhost", port=8080)
        self.assertEqual(vespa.get_namespace(None, "schema"), "schema")
        self.assertEqual(vespa.get_namespace("namespace", "schema"), "namespace")

    def test_get_schema_name(self):
        app_package = ApplicationPackage(
            name="bar", schema=[Schema(name="foo", document=Document())]
        )
        vespa = Vespa(
            url="http://localhost", port=8080, application_package=app_package
        )
        self.assertEqual(
            vespa.get_schema_name(), "foo"
        )  # Assuming schema name is "foo"
        self.assertEqual(vespa.get_schema_name("bar"), "bar")

    def test_asyncio(self):
        vespa = Vespa(url="http://localhost", port=8080)
        async_obj = vespa.asyncio(connections=8, total_timeout=10)
        self.assertIsInstance(async_obj, VespaAsync)
        self.assertEqual(async_obj.app, vespa)
        self.assertEqual(async_obj.connections, 8)
        self.assertEqual(async_obj.total_timeout, 10)

    def test_syncio(self):
        vespa = Vespa(url="http://localhost", port=8080)
        with vespa.syncio(connections=8) as sync_obj:
            self.assertIsInstance(sync_obj, VespaSync)
            self.assertEqual(sync_obj.app, vespa)
            self.assertEqual(sync_obj.pool_connections, 8)

    def test_http(self):
        vespa = Vespa(url="http://localhost", port=8080)
        http_obj = vespa.http(pool_maxsize=10)
        self.assertIsInstance(http_obj, VespaSync)
        self.assertEqual(http_obj.app, vespa)
        self.assertEqual(http_obj.pool_maxsize, 10)
        self.assertEqual(http_obj.pool_connections, 10)

    def test_repr(self):
        vespa = Vespa(url="http://localhost", port=8080)
        self.assertEqual(repr(vespa), "Vespa(http://localhost, 8080)")
        vespa = Vespa(url="http://localhost")
        self.assertEqual(repr(vespa), "Vespa(http://localhost)")

    def test_infer_schema_name(self):
        vespa = Vespa(url="http://localhost", port=8080)
        vespa._application_package = ApplicationPackage(name="test")
        schema_name = vespa._infer_schema_name()
        self.assertEqual(schema_name, "test")

        vespa._application_package = ApplicationPackage(name="foo", schema=[])
        schema_name = vespa._infer_schema_name()
        self.assertEqual(schema_name, "foo")

        vespa._application_package = ApplicationPackage(
            name="test",
            schema=[
                Schema(name="x", document=Document()),
                Schema(name="y", document=Document()),
            ],
        )
        with self.assertRaises(ValueError):
            vespa._infer_schema_name()

        vespa._application_package = ApplicationPackage(
            name="test",
            schema=[
                Schema(name="x", document=Document()),
            ],
        )
        self.assertEqual(vespa._infer_schema_name(), "x")

    @patch("vespa.application.requests.get")
    def test_get_application_status(self, mock_get):
        vespa = Vespa(url="http://localhost", port=8080)
        response = MagicMock()
        response.status_code = 200
        mock_get.return_value = response

        result = vespa.get_application_status()
        self.assertEqual(result, response)
        mock_get.assert_called_once_with("http://localhost:8080/ApplicationStatus")

    def test_wait_for_application_up(self):
        vespa = Vespa(url="http://localhost", port=8080)
        response = MagicMock()
        response.status_code = 200
        vespa.get_application_status = MagicMock(return_value=response)

        with patch("time.sleep"):
            vespa.wait_for_application_up(max_wait=10)

        vespa.get_application_status.assert_called()
        self.assertEqual(vespa.get_application_status.call_count, 1)

    @patch("vespa.application.VespaSync.get_model_endpoint")
    def test_get_model_endpoint(self, mock_get_model_endpoint):
        vespa = Vespa(url="http://localhost", port=8080)
        response = MagicMock()
        mock_get_model_endpoint.return_value = response

        result = vespa.get_model_endpoint(model_id="model1")
        self.assertEqual(result, response)
        mock_get_model_endpoint.assert_called_once_with(model_id="model1")

    @patch("vespa.application.VespaSync.query")
    def test_query(self, mock_query):
        vespa = Vespa(url="http://localhost", port=8080)
        response = MagicMock()
        mock_query.return_value = response

        result = vespa.query(body={"query": "test"}, groupname="group1")
        self.assertEqual(result, response)
        mock_query.assert_called_once_with(body={"query": "test"}, groupname="group1")

    @patch("vespa.application.VespaSync.feed_data_point")
    def test_feed_data_point(self, mock_feed_data_point):
        vespa = Vespa(url="http://localhost", port=8080)
        response = MagicMock()
        mock_feed_data_point.return_value = response

        result = vespa.feed_data_point(
            schema="schema1",
            data_id="data1",
            fields={"field1": "value1"},
            namespace="namespace1",
            groupname="group1",
        )
        self.assertEqual(result, response)
        mock_feed_data_point.assert_called_once_with(
            schema="schema1",
            data_id="data1",
            fields={"field1": "value1"},
            namespace="namespace1",
            groupname="group1",
        )

    def test_validate_operation_type(self):
        app = Vespa(url="http://localhost", port=8080)
        with self.assertRaises(ValueError):
            app.validate_operation_type("foo")

        app.validate_operation_type("feed")
        app.validate_operation_type("update")
        app.validate_operation_type("delete")

    def test_get_schema(self):
        schema = Schema(name="foo", document=Document())
        app_package = ApplicationPackage(name="test", schema=[schema])
        app = Vespa(url="http://localhost", port=8080, application_package=app_package)
        assert app.get_schema_name() == "foo"
        assert app.get_schema_name("foo") == "foo"

    def test_end_point(self):
        self.assertEqual(
            Vespa(url="https://cord19.vespa.ai").end_point, "https://cord19.vespa.ai"
        )
        self.assertEqual(
            Vespa(url="http://localhost", port=8080).end_point, "http://localhost:8080"
        )
        self.assertEqual(
            Vespa(url="http://localhost/", port=8080).end_point, "http://localhost:8080"
        )
        self.assertEqual(
            Vespa(url="http://localhost:8080").end_point, "http://localhost:8080"
        )

    def test_document_v1_format(self):
        vespa = Vespa(url="http://localhost", port=8080)
        self.assertEqual(
            vespa.get_document_v1_path(id=0, schema="foo"),
            "/document/v1/foo/foo/docid/0",
        )
        self.assertEqual(
            vespa.get_document_v1_path(id="0", schema="foo"),
            "/document/v1/foo/foo/docid/0",
        )

        self.assertEqual(
            vespa.get_document_v1_path(id="0", schema="foo", namespace="bar"),
            "/document/v1/bar/foo/docid/0",
        )

        self.assertEqual(
            vespa.get_document_v1_path(
                id="0", schema="foo", namespace="bar", group="g0"
            ),
            "/document/v1/bar/foo/group/g0/0",
        )

        self.assertEqual(
            vespa.get_document_v1_path(
                id="0", schema="foo", namespace="bar", number="0"
            ),
            "/document/v1/bar/foo/number/0/0",
        )

        self.assertEqual(
            vespa.get_document_v1_path(
                id="#1", schema="foo", namespace="bar", group="ab"
            ),
            "/document/v1/bar/foo/group/ab/#1",
        )

    def test_query_token(self):
        self.assertEqual(
            Vespa(
                url="https://cord19.vespa.ai",
                vespa_cloud_secret_token="vespa_cloud_str_secret",
            ).vespa_cloud_secret_token,
            "vespa_cloud_str_secret",
        )

    def test_query_token_from_env(self):
        import os

        os.environ["VESPA_CLOUD_SECRET_TOKEN"] = "vespa_cloud_str_secret"
        self.assertEqual(
            Vespa(url="https://cord19.vespa.ai").vespa_cloud_secret_token,
            "vespa_cloud_str_secret",
        )

    def test_infer_schema(self):
        #
        # No application package
        #
        app = Vespa(url="http://localhost", port=8080)
        with self.assertRaisesRegex(
            ValueError,
            "Application Package not available. Not possible to infer schema name.",
        ):
            _ = app._infer_schema_name()
        #
        # No schema
        #
        app_package = ApplicationPackage(name="test")

        # app = Vespa(url="http://localhost", port=8080, application_package=app_package)
        # with self.assertRaisesRegex(
        #     ValueError,
        #     "Application has no schema. Not possible to infer schema name.",
        # ):
        #     _ = app._infer_schema_name()

        # More than one schema
        app_package = ApplicationPackage(
            name="test",
            schema=[
                Schema(name="x", document=Document()),
                Schema(name="y", document=Document()),
            ],
        )
        app = Vespa(url="http://localhost", port=8080, application_package=app_package)
        with self.assertRaisesRegex(
            ValueError,
            "Application has more than one schema. Not possible to infer schema name.",
        ):
            _ = app._infer_schema_name()

        # One schema
        app_package = ApplicationPackage(
            name="test",
            schema=[
                Schema(name="x", document=Document()),
            ],
        )
        app = Vespa(url="http://localhost", port=8080, application_package=app_package)
        schema_name = app._infer_schema_name()
        self.assertEqual("x", schema_name)


class TestRaiseForStatus(unittest.TestCase):
    def test_successful_response(self):
        response = Response()
        response.status_code = 200
        try:
            raise_for_status(response)
        except Exception as e:
            self.fail(
                f"No exceptions were expected to be raised but {type(e).__name__} occurred"
            )

    def test_successful_response_with_error_content(self):
        with patch(
            "requests.models.Response.content", new_callable=PropertyMock
        ) as mock_content:
            response_json = {
                "root": {
                    "errors": [
                        {"code": 1, "summary": "summary", "message": "message"},
                    ],
                },
            }
            mock_content.return_value = json.dumps(response_json).encode("utf-8")
            response = Response()
            response.status_code = 200
            try:
                raise_for_status(response)
            except Exception as e:
                self.fail(
                    f"No exceptions were expected to be raised but {type(e).__name__} occurred"
                )

    def test_failure_response_for_400(self):
        response = Response()
        response.status_code = 400
        response.reason = "reason"
        response.url = "http://localhost:8080"
        with pytest.raises(HTTPError) as e:
            raise_for_status(response)
        self.assertEqual(
            str(e.value), "400 Client Error: reason for url: http://localhost:8080"
        )

    def test_failure_response_for_500(self):
        response = Response()
        response.status_code = 500
        response.reason = "reason"
        response.url = "http://localhost:8080"
        with pytest.raises(HTTPError) as e:
            raise_for_status(response)
        self.assertEqual(
            str(e.value), "500 Server Error: reason for url: http://localhost:8080"
        )

    def test_failure_response_without_error_content(self):
        with patch(
            "requests.models.Response.content", new_callable=PropertyMock
        ) as mock_content:
            response_json = {
                "root": {
                    "errors": [],
                },
            }
            mock_content.return_value = json.dumps(response_json).encode("utf-8")
            response = Response()
            response.status_code = 400
            response.reason = "reason"
            response.url = "http://localhost:8080"
            with pytest.raises(HTTPError):
                raise_for_status(response)

    def test_failure_response_with_error_content(self):
        with patch(
            "requests.models.Response.content", new_callable=PropertyMock
        ) as mock_content:
            response_json = {
                "root": {
                    "errors": [
                        {"code": 1, "summary": "summary", "message": "message"},
                    ],
                },
            }
            mock_content.return_value = json.dumps(response_json).encode("utf-8")
            response = Response()
            response.status_code = 400
            response.reason = "reason"
            response.url = "http://localhost:8080"
            with pytest.raises(VespaError):
                raise_for_status(response)

    def test_failure_response_with_error_content_504(self):
        with patch(
            "requests.models.Response.content", new_callable=PropertyMock
        ) as mock_content:
            response_json = {
                "root": {
                    "errors": [
                        {
                            "code": 12,
                            "summary": "Timed out",
                            "message": "No time left after waiting for 1ms to execute query",
                        },
                    ],
                },
            }
            mock_content.return_value = json.dumps(response_json).encode("utf-8")
            response = Response()
            response.status_code = 504
            response.reason = "reason"
            response.url = "http://localhost:8080"
            with pytest.raises(VespaError) as e:
                raise_for_status(response)
            self.assertEqual(
                str(e.value),
                "[{'code': 12, 'summary': 'Timed out', 'message': 'No time left after waiting for 1ms to execute query'}]",
            )

    def test_doc_failure_response_with_error_content(self):
        with patch(
            "requests.models.Response.content", new_callable=PropertyMock
        ) as mock_content:
            response_json = {
                "pathId": "/document/v1/textsearch/textsearch/docid/00",
                "message": "No field 'foo' in the structure of type 'textsearch'",
            }
            mock_content.return_value = json.dumps(response_json).encode("utf-8")
            response = Response()
            response.status_code = 400
            response.reason = "Bad Request"
            response.url = (
                "http://localhost:8080/document/v1/textsearch/textsearch/docid/00"
            )
            with pytest.raises(VespaError) as e:
                raise_for_status(response)
            self.assertEqual(
                str(e.value), "No field 'foo' in the structure of type 'textsearch'"
            )


class TestVespaCollectData(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Vespa(url="http://localhost", port=8080)
        self.raw_vespa_result_recall = {
            "root": {
                "id": "toplevel",
                "relevance": 1.0,
                "fields": {"totalCount": 1083},
                "coverage": {
                    "coverage": 100,
                    "documents": 62529,
                    "full": True,
                    "nodes": 2,
                    "results": 1,
                    "resultsFull": 1,
                },
                "children": [
                    {
                        "id": "id:covid-19:doc::40215",
                        "relevance": 30.368213170494712,
                        "source": "content",
                        "fields": {
                            "vespa_id_field": "abc",
                            "sddocname": "doc",
                            "body_text": "this is a body",
                            "title": "this is a title",
                            "rankfeatures": {"a": 1, "b": 2},
                        },
                    }
                ],
            }
        }

        self.raw_vespa_result_additional = {
            "root": {
                "id": "toplevel",
                "relevance": 1.0,
                "fields": {"totalCount": 1083},
                "coverage": {
                    "coverage": 100,
                    "documents": 62529,
                    "full": True,
                    "nodes": 2,
                    "results": 1,
                    "resultsFull": 1,
                },
                "children": [
                    {
                        "id": "id:covid-19:doc::40216",
                        "relevance": 10,
                        "source": "content",
                        "fields": {
                            "vespa_id_field": "def",
                            "sddocname": "doc",
                            "body_text": "this is a body 2",
                            "title": "this is a title 2",
                            "rankfeatures": {"a": 3, "b": 4},
                        },
                    },
                    {
                        "id": "id:covid-19:doc::40217",
                        "relevance": 8,
                        "source": "content",
                        "fields": {
                            "vespa_id_field": "ghi",
                            "sddocname": "doc",
                            "body_text": "this is a body 3",
                            "title": "this is a title 3",
                            "rankfeatures": {"a": 5, "b": 6},
                        },
                    },
                ],
            }
        }


if __name__ == "__main__":
    unittest.main()
