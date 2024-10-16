# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import unittest

import pytest
from unittest.mock import PropertyMock, patch, ANY
from unittest.mock import MagicMock, AsyncMock

from requests.models import HTTPError, Response

from vespa.package import ApplicationPackage, Schema, Document
from vespa.application import Vespa, raise_for_status
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse, VespaResponse
import requests_mock
from unittest.mock import Mock
from requests import Request, Session
import gzip
from vespa.application import (
    CustomHTTPAdapter,
    VespaAsync,
)
import httpx
import tenacity


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


class TestCustomHTTPAdapterCompression(unittest.TestCase):
    def setUp(self):
        """Set up the CustomHTTPAdapter for testing."""
        self.adapter = CustomHTTPAdapter(compress="auto")

    def test_compression_auto_with_large_body(self):
        """Test auto compression with a large request body."""
        request = Request(method="POST", url="http://test.com", data=b"test_data" * 300)
        self.adapter.check_size = Mock(return_value=5000)  # Simulate large content
        prepared_request = request.prepare()

        self.adapter._maybe_compress_request(prepared_request)
        self.assertIn("Content-Encoding", prepared_request.headers)
        self.assertEqual(prepared_request.headers["Content-Encoding"], "gzip")

    def test_no_compression_auto_with_small_body(self):
        """Test no compression with a small request body."""
        request = Request(method="POST", url="http://test.com", data=b"test_data")
        self.adapter.check_size = Mock(return_value=10)  # Simulate small content
        prepared_request = request.prepare()

        self.adapter._maybe_compress_request(prepared_request)
        self.assertNotIn("Content-Encoding", prepared_request.headers)

    def test_force_compression(self):
        """Test forced compression when compress=True."""
        self.adapter = CustomHTTPAdapter(compress=True)
        request = Request(method="POST", url="http://test.com", data=b"test_data")
        prepared_request = request.prepare()

        self.adapter._maybe_compress_request(prepared_request)
        self.assertIn("Content-Encoding", prepared_request.headers)
        self.assertEqual(prepared_request.headers["Content-Encoding"], "gzip")

    def test_disable_compression(self):
        """Test no compression when compress=False."""
        self.adapter = CustomHTTPAdapter(compress=False)
        request = Request(method="POST", url="http://test.com", data=b"test_data")
        prepared_request = request.prepare()

        self.adapter._maybe_compress_request(prepared_request)
        self.assertNotIn("Content-Encoding", prepared_request.headers)

    def test_invalid_compression_value(self):
        """Test invalid compress value raises error."""
        with self.assertRaises(ValueError):
            CustomHTTPAdapter(compress="invalid_value")

    def test_compress_request_body(self):
        """Test if request body is compressed when compress=True."""
        adapter = CustomHTTPAdapter(compress=True)
        session = Session()
        session.mount("http://", adapter)

        request = Request(method="POST", url="http://test.com", data=b"test_data")
        prepared_request = session.prepare_request(request)
        # Mock sending the request
        with patch("requests.adapters.HTTPAdapter.send") as mock_send:
            adapter.send(prepared_request)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            self.assertEqual(args[0].body, gzip.compress(b"test_data"))

    def test_retry_on_429_status(self):
        """Test retry logic when response status is 429."""
        adapter = CustomHTTPAdapter(num_retries_429=2)
        session = Session()
        session.mount("http://", adapter)

        request = Request(method="POST", url="http://test.com", data=b"test_data")
        prepared_request = session.prepare_request(request)

        with patch.object(adapter, "_wait_with_backoff") as mock_backoff, patch(
            "requests.adapters.HTTPAdapter.send"
        ) as mock_send:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_send.side_effect = [mock_response, mock_response, mock_response]

            adapter.send(prepared_request)

            self.assertEqual(mock_send.call_count, 3)
            self.assertEqual(mock_backoff.call_count, mock_send.call_count)


class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_total_timeout(self):
        app = Vespa(url="http://localhost", port=8080)
        total_timeout = 1  # seconds
        vespa_async = VespaAsync(app=app, total_timeout=total_timeout)

        # Patch httpx.AsyncClient in the module where VespaAsync is defined
        with patch(
            "vespa.application.httpx.AsyncClient", autospec=True
        ) as MockAsyncClient:
            # Create an instance of the mock AsyncClient
            mock_client_instance = MockAsyncClient.return_value
            # Ensure that 'post' is an AsyncMock
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.ReadTimeout("Read timeout")
            )

            with pytest.raises(tenacity.RetryError) as exc_info:
                async with vespa_async:
                    await vespa_async.query(
                        body={
                            "yql": "select * from sources * where title contains 'music';"
                        }
                    )

            assert isinstance(
                exc_info.value.last_attempt.exception(), httpx.ReadTimeout
            )

    @pytest.mark.asyncio
    async def test_read_timeout(self):
        app = Vespa(url="http://localhost", port=8080)
        read_timeout = 1  # seconds
        vespa_async = VespaAsync(app=app, total_timeout=None, read_timeout=read_timeout)

        with patch(
            "vespa.application.httpx.AsyncClient", autospec=True
        ) as MockAsyncClient:
            mock_client_instance = MockAsyncClient.return_value
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.ReadTimeout("Read timeout")
            )

            with pytest.raises(tenacity.RetryError) as exc_info:
                async with vespa_async:
                    await vespa_async.query(
                        body={
                            "yql": "select * from sources * where title contains 'music';"
                        }
                    )

            assert isinstance(
                exc_info.value.last_attempt.exception(), httpx.ReadTimeout
            )

    @pytest.mark.asyncio
    async def test_write_timeout(self):
        app = Vespa(url="http://localhost", port=8080)
        write_timeout = 1  # seconds
        vespa_async = VespaAsync(
            app=app, total_timeout=None, write_timeout=write_timeout
        )

        with patch(
            "vespa.application.httpx.AsyncClient", autospec=True
        ) as MockAsyncClient:
            mock_client_instance = MockAsyncClient.return_value
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.WriteTimeout("Write timeout")
            )

            with pytest.raises(tenacity.RetryError) as exc_info:
                async with vespa_async:
                    await vespa_async.query(
                        body={
                            "yql": "select * from sources * where title contains 'music';"
                        }
                    )

            assert isinstance(
                exc_info.value.last_attempt.exception(), httpx.WriteTimeout
            )

    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        app = Vespa(url="http://10.255.255.1", port=8080)  # Non-routable IP address
        connect_timeout = 1  # seconds
        vespa_async = VespaAsync(
            app=app, total_timeout=None, connect_timeout=connect_timeout
        )

        with patch(
            "vespa.application.httpx.AsyncClient", autospec=True
        ) as MockAsyncClient:
            mock_client_instance = MockAsyncClient.return_value
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.ConnectTimeout("Connect timeout")
            )

            with pytest.raises(tenacity.RetryError) as exc_info:
                async with vespa_async:
                    await vespa_async.query(
                        body={
                            "yql": "select * from sources * where title contains 'music';"
                        }
                    )

            assert isinstance(
                exc_info.value.last_attempt.exception(), httpx.ConnectTimeout
            )

    def test_keepalive_expiry_warning(self):
        app = Vespa(url="http://localhost", port=8080)
        with pytest.warns(
            UserWarning,
            match="Setting keepalive_expiry higher than 30 seconds may cause the Vespa server to reset idle connection.",
        ):
            VespaAsync(app=app, keepalive_expiry=31)

    def test_client_initialization(self):
        app = Vespa(url="http://localhost", port=8080)
        with patch(
            "vespa.application.httpx.AsyncClient", autospec=True
        ) as MockAsyncClient:
            vespa_async = VespaAsync(
                app=app,
                connections=5,
                total_timeout=10,
                connect_timeout=3,
                read_timeout=4,
                write_timeout=2,
                pool_timeout=5,
                keepalive_expiry=15,
                proxies={
                    "http": "http://localhost:8000"
                },  # passing kwarg (must be valid)
            )
            vespa_async._open_httpx_client()

            expected_limits = httpx.Limits(
                max_keepalive_connections=5,
                max_connections=5,
                keepalive_expiry=15,
            )
            expected_timeout = httpx.Timeout(10)

            MockAsyncClient.assert_called_with(
                timeout=expected_timeout,
                headers=vespa_async.headers,
                verify=False,
                http2=True,
                http1=False,
                limits=expected_limits,
                proxies={"http": "http://localhost:8000"},
            )

    def test_connections(self):
        app = Vespa(url="http://localhost", port=8080)
        connections = 5
        with patch(
            "vespa.application.httpx.AsyncClient", autospec=True
        ) as MockAsyncClient:
            vespa_async = VespaAsync(app=app, connections=connections)
            vespa_async._open_httpx_client()

            MockAsyncClient.assert_called_with(
                timeout=ANY,
                headers=ANY,
                verify=ANY,
                http2=ANY,
                http1=ANY,
                limits=httpx.Limits(
                    max_keepalive_connections=connections,
                    max_connections=connections,
                    keepalive_expiry=vespa_async.keepalive_expiry,
                ),
            )

    def test_custom_kwargs(self):
        app = Vespa(url="http://localhost", port=8080)
        with patch(
            "vespa.application.httpx.AsyncClient", autospec=True
        ) as MockAsyncClient:
            vespa_async = VespaAsync(app=app, proxies={"http": "http://localhost:8000"})
            vespa_async._open_httpx_client()

            MockAsyncClient.assert_called_with(
                timeout=ANY,
                headers=ANY,
                verify=ANY,
                http2=ANY,
                http1=ANY,
                limits=ANY,
                proxies={"http": "http://localhost:8000"},
            )


if __name__ == "__main__":
    unittest.main()
