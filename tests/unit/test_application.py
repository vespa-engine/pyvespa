# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import unittest

import pytest
from unittest.mock import PropertyMock, patch
from unittest.mock import MagicMock, AsyncMock

from requests.models import HTTPError, Response

from vespa.package import ApplicationPackage, Schema, Document
from vespa.application import Vespa, raise_for_status
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse, VespaResponse
import requests_mock


class TestVespaRequestsUsage(unittest.TestCase):
    def test_additional_query_params(self):
        app = Vespa(url="http://localhost", port=8080)
        with requests_mock.Mocker() as m:
            m.get(
                "http://localhost:8080/ApplicationStatus",
                status_code=200,
            )
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
            # Mock the ApplicationStatus endpoint
            m.get(
                "http://localhost:8080/ApplicationStatus",
                status_code=200,
            )
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
            m.get(
                "http://localhost:8080/ApplicationStatus",
                status_code=200,
            )
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

    def test_visit(self):
        app = Vespa(url="http://localhost", port=8080)
        with requests_mock.Mocker() as m:
            m.get(
                "http://localhost:8080/ApplicationStatus",
                status_code=200,
            )
            m.get(
                "http://localhost:8080/document/v1/foo/foo/docid/",
                [
                    {"json": {"continuation": "AAA"}, "status_code": 200},
                    {"json": {}, "status_code": 200},
                ],
            )

            results = []
            for slice in app.visit(
                schema="foo",
                namespace="foo",
                content_cluster_name="content",
                timeout="200s",
            ):
                for response in slice:
                    results.append(response)
            assert len(results) == 2

            urls = [response.url for response in results]
            assert (
                "http://localhost:8080/document/v1/foo/foo/docid/"
                "?cluster=content"
                "&selection=true"
                "&wantedDocumentCount=500"
                "&slices=1"
                "&sliceId=0"
                "&timeout=200s"
                "&continuation=AAA"
            ) in urls

            assert (
                "http://localhost:8080/document/v1/foo/foo/docid/"
                "?cluster=content"
                "&selection=true"
                "&wantedDocumentCount=500"
                "&slices=1"
                "&sliceId=0"
                "&timeout=200s"
            ) in urls


class TestVespa(unittest.TestCase):
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
                id="mydoc#1", schema="foo", namespace="bar", group="ab"
            ),
            "/document/v1/bar/foo/group/ab/mydoc%231",
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
            Vespa(
                url="https://cord19.vespa.ai",
                vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN"),
            ).vespa_cloud_secret_token,
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


class TestFeedAsyncIterable(unittest.TestCase):
    def setUp(self):
        self.mock_session = AsyncMock()
        self.mock_asyncio_patcher = patch("vespa.application.VespaAsync")
        self.mock_asyncio = self.mock_asyncio_patcher.start()
        self.mock_asyncio.return_value.__aenter__.return_value = self.mock_session

        self.vespa = Vespa(url="http://localhost", port=8080)

    def tearDown(self):
        self.mock_asyncio_patcher.stop()

    def test_feed_async_iterable_happy_path(self):
        # Arrange
        iter_data = [
            {"id": "doc1", "fields": {"title": "Document 1"}},
            {"id": "doc2", "fields": {"title": "Document 2"}},
        ]
        callback = MagicMock()

        # Act
        self.vespa.feed_async_iterable(
            iter=iter_data,
            schema="test_schema",
            namespace="test_namespace",
            callback=callback,
            max_queue_size=2,
            max_workers=2,
            max_connections=2,
        )

        # Assert
        self.mock_session.feed_data_point.assert_has_calls(
            [
                unittest.mock.call(
                    schema="test_schema",
                    namespace="test_namespace",
                    groupname=None,
                    data_id="doc1",
                    fields={"title": "Document 1"},
                ),
                unittest.mock.call(
                    schema="test_schema",
                    namespace="test_namespace",
                    groupname=None,
                    data_id="doc2",
                    fields={"title": "Document 2"},
                ),
            ],
            any_order=True,
        )
        self.assertEqual(callback.call_count, 2)

    def test_feed_async_iterable_missing_id(self):
        # Arrange
        iter_data = [
            {"fields": {"title": "Document 1"}},
        ]
        callback = MagicMock()

        # Act
        self.vespa.feed_async_iterable(
            iter=iter_data,
            schema="test_schema",
            namespace="test_namespace",
            callback=callback,
            max_queue_size=1,
            max_workers=1,
            max_connections=1,
        )

        # Assert
        self.mock_session.feed_data_point.assert_not_called()
        callback.assert_called_once_with(unittest.mock.ANY, None)
        self.assertEqual(callback.call_args[0][0].status_code, 499)
        self.assertEqual(
            callback.call_args[0][0].json["message"], "Missing id in input dict"
        )

    def test_feed_async_iterable_missing_fields(self):
        # Arrange
        iter_data = [
            {"id": "doc1"},
        ]
        callback = MagicMock()

        # Act
        self.vespa.feed_async_iterable(
            iter=iter_data,
            schema="test_schema",
            namespace="test_namespace",
            callback=callback,
            max_queue_size=1,
            max_workers=1,
            max_connections=1,
        )

        # Assert
        self.mock_session.feed_data_point.assert_not_called()
        callback.assert_called_once_with(unittest.mock.ANY, "doc1")
        self.assertEqual(callback.call_args[0][0].status_code, 499)
        self.assertEqual(
            callback.call_args[0][0].json["message"], "Missing fields in input dict"
        )
