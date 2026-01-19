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
from unittest.mock import Mock
from vespa.application import (
    VespaAsync,
)
import httpx
from typing import List


def create_mock_httpr_response(status_code=200, json_data=None, text="{}", url=""):
    """Helper to create mock httpr Response objects."""
    mock_response = Mock()
    # Use spec to prevent Mock from auto-creating nested mocks
    mock_response.status_code = status_code
    # Configure url as a property that returns the actual string value
    type(mock_response).url = PropertyMock(return_value=url)
    mock_response.text = text
    if json_data is not None:
        mock_response.json.return_value = json_data
    else:
        mock_response.json.return_value = json.loads(text) if text else {}
    return mock_response


class TestVespaRequestsUsage(unittest.TestCase):
    @patch("vespa.application.httpr.Client")
    def test_additional_query_params(self, MockClient):
        # Create a mock client instance with the methods we need
        mock_client_instance = Mock()

        # Mock ApplicationStatus GET
        status_response = create_mock_httpr_response(status_code=200)
        # Mock search POST
        search_response = create_mock_httpr_response(
            status_code=200,
            text="{}",
            url="http://localhost:8080/search/?query=this+is+a+test&hits=10&searchChain=default",
        )
        mock_client_instance.get.return_value = status_response
        mock_client_instance.post.return_value = search_response
        mock_client_instance.close = Mock()  # Mock close method

        # Make httpr.Client() return our mock instance
        MockClient.return_value = mock_client_instance

        app = Vespa(url="http://localhost", port=8080)
        r: VespaQueryResponse = app.query(
            query="this is a test", hits=10, searchChain="default"
        )
        self.assertEqual(
            r.url,
            "http://localhost:8080/search/?query=this+is+a+test&hits=10&searchChain=default",
        )

    @patch("vespa.application.httpr.Client")
    def test_additional_doc_params(self, MockClient):
        mock_client_instance = Mock()
        MockClient.return_value = mock_client_instance
        mock_client_instance.close = Mock()

        app = Vespa(url="http://localhost", port=8080)

        # Test feed_data_point (POST)
        status_response = create_mock_httpr_response(status_code=200)
        post_response = create_mock_httpr_response(
            status_code=200,
            text="{}",
            url="http://localhost:8080/document/v1/foo/foo/docid/0?route=default&timeout=10s",
        )
        mock_client_instance.get.return_value = status_response
        mock_client_instance.post.return_value = post_response

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

        # Test update_data (PUT)
        put_response = create_mock_httpr_response(
            status_code=200,
            text="{}",
            url="http://localhost:8080/document/v1/foo/foo/docid/0?create=false&route=default&timeout=10s",
        )
        mock_client_instance.put.return_value = put_response

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

        # Test delete_data (DELETE)
        delete_response = create_mock_httpr_response(
            status_code=200,
            text="{}",
            url="http://localhost:8080/document/v1/foo/foo/docid/0?route=default&timeout=10s&dryRun=True",
        )
        mock_client_instance.delete.return_value = delete_response

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

    @patch("vespa.application.httpr.Client")
    def test_visit(self, MockClient):
        mock_client_instance = Mock()
        MockClient.return_value = mock_client_instance
        mock_client_instance.close = Mock()

        # First visit request returns continuation token
        first_visit_response = create_mock_httpr_response(
            status_code=200,
            json_data={"continuation": "AAA"},
            url="http://localhost:8080/document/v1/foo/foo/docid/"
            "?cluster=content"
            "&selection=true"
            "&wantedDocumentCount=500"
            "&slices=1"
            "&sliceId=0"
            "&timeout=200s",
        )

        # Second visit request with continuation returns empty (no more results)
        second_visit_response = create_mock_httpr_response(
            status_code=200,
            json_data={},
            url="http://localhost:8080/document/v1/foo/foo/docid/"
            "?cluster=content"
            "&selection=true"
            "&wantedDocumentCount=500"
            "&slices=1"
            "&sliceId=0"
            "&timeout=200s"
            "&continuation=AAA",
        )

        # Mock get to return different responses on successive calls
        mock_client_instance.get.side_effect = [
            first_visit_response,
            second_visit_response,
        ]

        app = Vespa(url="http://localhost", port=8080)
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

    @patch("vespa.application.httpr.Client")
    def test_visit_slice_id(self, MockClient):
        mock_client_instance = Mock()
        MockClient.return_value = mock_client_instance
        mock_client_instance.close = Mock()

        # First visit request returns continuation token
        first_visit_response = create_mock_httpr_response(
            status_code=200,
            json_data={"continuation": "AAA"},
            url="http://localhost:8080/document/v1/foo/foo/docid/"
            "?cluster=content"
            "&selection=true"
            "&wantedDocumentCount=500"
            "&slices=10"
            "&sliceId=2"
            "&timeout=200s",
        )

        # Second visit request with continuation returns empty (no more results)
        second_visit_response = create_mock_httpr_response(
            status_code=200,
            json_data={},
            url="http://localhost:8080/document/v1/foo/foo/docid/"
            "?cluster=content"
            "&selection=true"
            "&wantedDocumentCount=500"
            "&slices=10"
            "&sliceId=2"
            "&timeout=200s"
            "&continuation=AAA",
        )

        # Mock get to return different responses on successive calls
        mock_client_instance.get.side_effect = [
            first_visit_response,
            second_visit_response,
        ]

        app = Vespa(url="http://localhost", port=8080)
        results = []
        for slice in app.visit(
            schema="foo",
            namespace="foo",
            content_cluster_name="content",
            timeout="200s",
            slices=10,
            slice_id=2,
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
            "&slices=10"
            "&sliceId=2"
            "&timeout=200s"
            "&continuation=AAA"
        ) in urls

        assert (
            "http://localhost:8080/document/v1/foo/foo/docid/"
            "?cluster=content"
            "&selection=true"
            "&wantedDocumentCount=500"
            "&slices=10"
            "&sliceId=2"
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

    def test_init_additional_headers(self):
        app = Vespa(
            url="http://localhost",
            additional_headers={"X-Custom-Header": "test"},
        )
        assert app.base_headers == {
            "User-Agent": f"pyvespa/{app.pyvespa_version}",
            "X-Custom-Header": "test",
        }


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


class TestQueryMany(unittest.TestCase):
    def setUp(self):
        self.mock_session = AsyncMock()
        self.mock_asyncio_patcher = patch("vespa.application.VespaAsync")
        self.mock_asyncio = self.mock_asyncio_patcher.start()
        self.mock_asyncio.return_value.__aenter__.return_value = self.mock_session
        self.vespa = Vespa(url="http://localhost", port=8080)

    def tearDown(self):
        self.mock_asyncio_patcher.stop()

    def test_query_many_happy_path(self):
        # Arrange
        query_data = [
            {"query": "this is a test", "hits": 10, "ranking": "default"},
            {"query": "this is another test", "hits": 20, "ranking": "default"},
        ]
        #
        _responses: List[VespaQueryResponse] = self.vespa.query_many(
            queries=query_data,
            num_connections=2,
            max_concurrent=100,
            adaptive=False,  # Disable adaptive throttling for this test
        )

        # Assert that app.query is called for each query
        self.mock_session.query.assert_has_calls(
            [unittest.mock.call(q) for q in query_data],
            any_order=True,
        )

    def test_query_many_client_kwargs(self):
        # Arrange
        query_data = [
            {"query": "this is a test", "hits": 10, "ranking": "default"},
            {"query": "this is another test", "hits": 20, "ranking": "default"},
        ]
        #
        _responses: List[VespaQueryResponse] = self.vespa.query_many(
            queries=query_data,
            num_connections=2,
            max_concurrent=100,
            adaptive=False,  # Disable adaptive throttling for this test
            client_kwargs={"timeout": 10},
        )

        # Assert that VespaAsync is initialized once with the client_kwargs
        self.mock_asyncio.assert_called_once_with(
            app=self.vespa,
            connections=2,
            total_timeout=None,
            timeout=10,
            client=None,
        )

    def test_query_many_query_kwargs(self):
        # Arrange
        query_data = [
            {"query": "this is a test", "hits": 10, "ranking": "default"},
            {"query": "this is another test", "hits": 20, "ranking": "default"},
        ]
        #
        _responses: List[VespaQueryResponse] = self.vespa.query_many(
            queries=query_data,
            num_connections=2,
            max_concurrent=100,
            adaptive=False,  # Disable adaptive throttling for this test
            query_param="custom",
        )

        # Assert that app.query is called for each query with the query_kwargs
        self.mock_session.query.assert_has_calls(
            [unittest.mock.call(q, query_param="custom") for q in query_data],
            any_order=True,
        )


class MockVespa:
    def __init__(
        self,
        base_headers=None,
        auth_method=None,
        vespa_cloud_secret_token=None,
        cert=None,
        key=None,
    ):
        self.base_headers = base_headers or {}
        self.auth_method = auth_method
        self.vespa_cloud_secret_token = vespa_cloud_secret_token
        self.cert = cert
        self.key = key


# Test class
class TestVespaAsync:
    def test_init_default(self):
        """Test VespaAsync initialization with defaults (using httpr)"""
        app = MockVespa()
        vespa_async = VespaAsync(app)
        assert vespa_async.app == app
        assert vespa_async.httpr_client is None  # Changed from httpx_client
        assert vespa_async.connections == 1
        assert vespa_async.total_timeout is None
        assert vespa_async.timeout == 30.0  # Now a float, not httpx.Timeout
        assert vespa_async.kwargs == {}
        assert vespa_async.headers == app.base_headers

    def test_init_total_timeout_warns(self):
        """Test that total_timeout parameter emits deprecation warning"""
        app = MockVespa()
        with pytest.warns(DeprecationWarning, match="total_timeout is deprecated"):
            vespa_async = VespaAsync(app, total_timeout=10)
        assert vespa_async.total_timeout == 10

    def test_init_timeout_int(self):
        """Test timeout as int converts to float"""
        app = MockVespa()
        vespa_async = VespaAsync(app, timeout=10)
        assert vespa_async.timeout == 10.0  # Converted to float

    def test_init_timeout_timeout(self):
        """Test backward compatibility with httpx.Timeout (extracts read timeout)"""
        app = MockVespa()
        timeout = httpx.Timeout(connect=5, read=10, write=15, pool=20)
        with pytest.warns(
            DeprecationWarning, match="Passing httpx.Timeout is deprecated"
        ):
            vespa_async = VespaAsync(app, timeout=timeout)
        assert vespa_async.timeout == 10.0  # Extracted read timeout

    def test_init_keepalive_expiry_warning(self):
        """Test that limits parameter emits deprecation warning (httpr manages pooling)"""
        app = MockVespa()
        limits = httpx.Limits(keepalive_expiry=31)
        with pytest.warns(
            DeprecationWarning, match="limits.*no longer used with httpr"
        ):
            _vespa_async = VespaAsync(app, limits=limits)

    def test_init_no_keepalive_expiry_warning(self):
        """Test that limits parameter still emits deprecation warning regardless of value"""
        app = MockVespa()
        limits = httpx.Limits(keepalive_expiry=1)
        with pytest.warns(
            DeprecationWarning, match="limits.*no longer used with httpr"
        ):
            _vespa_async = VespaAsync(app, limits=limits)


class TestVespaSyncStreaming(unittest.TestCase):
    @patch("vespa.application.httpr.Client")
    def test_query_streaming(self, MockClient):
        """Test the query method with streaming=True and mocked Server-Sent Events response."""
        # Mock SSE response data in the expected format
        sse_data = [
            "event: token",
            'data: {"token":""}\n\n',
            "event: token",
            'data: {"token":"V"}\n\n',
            "event: token",
            'data: {"token":"es"}\n\n',
            "event: token",
            'data: {"token":"pa"}\n\n',
            "event: token",
            'data: {"token":" is"}\n\n',
            "event: token",
            'data: {"token":" a"}\n\n',
            "event: token",
            'data: {"token":" scalable"}\n\n',
            "event: token",
            'data: {"token":" open"}\n\n',
            "event: token",
            'data: {"token":"-source"}\n\n',
            "event: token",
            'data: {"token":" serving"}\n\n',
            "event: token",
            'data: {"token":" engine"}\n\n',
            "event: token",
            'data: {"token":" designed"}\n\n',
            "event: token",
            'data: {"token":" to"}\n\n',
            "event: token",
            'data: {"token":" store"}\n\n',
            "event: token",
            'data: {"token":","}\n\n',
            "event: token",
            'data: {"token":" compute"}\n\n',
        ]

        mock_client_instance = Mock()
        MockClient.return_value = mock_client_instance
        mock_client_instance.close = Mock()

        # Mock ApplicationStatus GET
        status_response = create_mock_httpr_response(status_code=200)
        mock_client_instance.get.return_value = status_response

        # Mock streaming response using iter_lines
        # httpr's iter_lines returns strings, not bytes
        mock_stream = Mock()
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)
        mock_stream.iter_lines.return_value = sse_data

        mock_client_instance.stream.return_value = mock_stream

        # Create a Vespa app instance
        app = Vespa(url="http://localhost", port=8080)

        results = []
        result_string = ""
        # Use the simpler app.syncio() syntax
        with app.syncio() as sync_app:
            # Test the streaming query using the public query method
            query_body = {"query": "test streaming query"}
            result_generator = sync_app.query(
                body=query_body, streaming=True, timeout="30s"
            )
            for line in result_generator:
                results.append(line)
                print(f"Received line: {line}")
                print(line)
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    print(event)
                    token = event.get("token", "")
                    result_string += token

            # Verify we got the expected number of lines
            self.assertEqual(len(results), 32)
            # Verify the final result string
            self.assertEqual(
                result_string,
                "Vespa is a scalable open-source serving engine designed to store, compute",
            )

            # Verify the stream was called with correct parameters
            mock_client_instance.stream.assert_called_once_with(
                "POST",
                app.search_end_point,
                json=query_body,
                params={"timeout": "30s"},
            )


class TestConnectionReuse(unittest.TestCase):
    """Test connection reuse functionality for both sync and async clients"""

    def test_get_sync_session_reuse(self):
        """Test that an externally provided session is not closed by VespaSync context manager"""
        app = Vespa(url="http://localhost", port=8080)

        session = app.get_sync_session()

        # Mock the close method to track if it's called
        session.close = Mock()

        with app.syncio(session=session) as sync_app:
            self.assertIs(sync_app.http_client, session)  # Changed from http_session

        # Session should NOT be closed when exiting the context manager
        session.close.assert_not_called()

        # User is responsible for closing
        session.close()
        session.close.assert_called_once()

    def test_sync_session_ownership(self):
        """Test that VespaSync closes clients it creates, but not external ones"""
        app = Vespa(url="http://localhost", port=8080)

        # When VespaSync creates its own client, it should close it
        with patch("vespa.application.httpr.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with app.syncio() as sync_app:
                self.assertTrue(sync_app._owns_client)  # Changed from _owns_session

            # Should have been closed
            mock_client.close.assert_called_once()

        # When given an external client, it should NOT close it
        external_client = app.get_sync_session()
        external_client.close = Mock()

        with app.syncio(session=external_client) as sync_app:
            self.assertFalse(sync_app._owns_client)  # Changed from _owns_session
            self.assertIs(
                sync_app.http_client, external_client
            )  # Changed from http_session

        # Should NOT be closed
        external_client.close.assert_not_called()


# Separate test functions for async tests to avoid unittest.TestCase async warnings
@pytest.mark.asyncio
async def test_get_async_session_reuse():
    """Test that an externally provided client is not closed by VespaAsync context manager"""
    app = Vespa(url="http://localhost", port=8080)

    client = app.get_async_session()

    # Mock the aclose method to track if it's called (use AsyncMock for async methods)
    client.aclose = AsyncMock()

    async with app.asyncio(client=client) as async_app:
        assert async_app.httpr_client is client  # Changed from httpx_client

    # Client should NOT be closed when exiting the context manager
    client.aclose.assert_not_called()

    # User is responsible for closing
    await client.aclose()
    client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_async_client_ownership():
    """Test that VespaAsync closes clients it creates, but not external ones"""
    app = Vespa(url="http://localhost", port=8080)

    # When VespaAsync creates its own client, it should close it
    with patch(
        "vespa.application.httpr.AsyncClient"
    ) as mock_client_class:  # Changed from httpx.AsyncClient
        mock_client = Mock()
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with app.asyncio() as async_app:
            assert async_app._owns_client is True

        # Should have been closed
        mock_client.aclose.assert_called_once()

    # When given an external client, it should NOT close it
    external_client = app.get_async_session()
    external_client.aclose = AsyncMock()

    async with app.asyncio(client=external_client) as async_app:
        assert async_app._owns_client is False
        assert async_app.httpr_client is external_client  # Changed from httpx_client

    # Should NOT be closed
    external_client.aclose.assert_not_called()


class TestQueryProfiling(unittest.TestCase):
    @patch("vespa.application.httpr.Client")
    def test_query_with_profiling(self, MockClient):
        """Test that profile=True adds the correct profiling parameters"""
        mock_client_instance = Mock()
        MockClient.return_value = mock_client_instance
        mock_client_instance.close = Mock()

        # Mock ApplicationStatus GET
        status_response = create_mock_httpr_response(status_code=200)

        # Mock search POST with profiling params in URL
        search_url = (
            "http://localhost:8080/search/"
            "?trace.level=1"
            "&trace.explainLevel=1"
            "&trace.profileDepth=100"
            "&trace.timestamps=true"
            "&presentation.timing=true"
        )
        search_response = create_mock_httpr_response(
            status_code=200, text="{}", url=search_url
        )

        mock_client_instance.get.return_value = status_response
        mock_client_instance.post.return_value = search_response

        app = Vespa(url="http://localhost", port=8080)
        r: VespaQueryResponse = app.query(
            body={"yql": "select * from sources * where true"}, profile=True
        )
        print(r.url)
        # Verify profiling params are in the URL
        self.assertIn("trace.level=1", r.url)
        self.assertIn("trace.explainLevel=1", r.url)
        self.assertIn("trace.profileDepth=100", r.url)
        self.assertIn("trace.timestamps=true", r.url)
        self.assertIn("presentation.timing=true", r.url)


if __name__ == "__main__":
    unittest.main()
