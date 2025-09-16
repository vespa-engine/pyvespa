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
from requests import Request, Session
import gzip
from vespa.application import (
    CustomHTTPAdapter,
    VespaAsync,
)
import httpx
from typing import List


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

    def test_visit_slice_id(self):
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

    def test_cert_content_and_key_content_accepted(self):
        """Test that cert_content and key_content are accepted together."""
        cert_content = "-----BEGIN CERTIFICATE-----\ntest cert content\n-----END CERTIFICATE-----"
        key_content = "-----BEGIN PRIVATE KEY-----\ntest key content\n-----END PRIVATE KEY-----"
        
        app = Vespa(
            url="http://localhost",
            cert_content=cert_content,
            key_content=key_content
        )
        
        self.assertEqual(app.cert_content, cert_content)
        self.assertEqual(app.key_content, key_content)
        self.assertIsNone(app.cert)
        self.assertIsNone(app.key)

    def test_cert_and_cert_content_mutual_exclusion(self):
        """Test that cert and cert_content are mutually exclusive."""
        with self.assertRaises(ValueError) as context:
            Vespa(
                url="http://localhost",
                cert="path/to/cert.pem",
                cert_content="cert content"
            )
        self.assertIn("cert and cert_content are mutually exclusive", str(context.exception))

    def test_key_and_key_content_mutual_exclusion(self):
        """Test that key and key_content are mutually exclusive."""
        with self.assertRaises(ValueError) as context:
            Vespa(
                url="http://localhost",
                key="path/to/key.pem",
                key_content="key content"
            )
        self.assertIn("key and key_content are mutually exclusive", str(context.exception))

    def test_cert_content_without_key_content_rejected(self):
        """Test that cert_content without key_content is rejected."""
        with self.assertRaises(ValueError) as context:
            Vespa(
                url="http://localhost",
                cert_content="cert content"
            )
        self.assertIn("cert_content and key_content must be provided together", str(context.exception))

    def test_key_content_without_cert_content_rejected(self):
        """Test that key_content without cert_content is rejected."""
        with self.assertRaises(ValueError) as context:
            Vespa(
                url="http://localhost",
                key_content="key content"
            )
        self.assertIn("cert_content and key_content must be provided together", str(context.exception))

    def test_key_without_cert_rejected(self):
        """Test that providing key without cert is rejected."""
        with self.assertRaises(ValueError) as context:
            Vespa(
                url="http://localhost",
                key="path/to/key.pem"
            )
        self.assertIn("If key is provided, cert must also be provided", str(context.exception))

    def test_cert_without_key_accepted(self):
        """Test that providing cert without key is accepted (combined cert+key file)."""
        app = Vespa(
            url="http://localhost",
            cert="path/to/combined.pem"
        )
        self.assertEqual(app.cert, "path/to/combined.pem")
        self.assertIsNone(app.key)

    def test_vespa_sync_with_cert_content(self):
        """Test that VespaSync creates temporary files for cert_content and key_content."""
        import tempfile
        import os
        from vespa.application import VespaSync
        from unittest.mock import patch
        
        cert_content = "-----BEGIN CERTIFICATE-----\ntest cert content\n-----END CERTIFICATE-----"
        key_content = "-----BEGIN PRIVATE KEY-----\ntest key content\n-----END PRIVATE KEY-----"
        
        app = Vespa(
            url="http://localhost",
            cert_content=cert_content,
            key_content=key_content
        )
        
        # Mock Session.cert to avoid actual SSL operations
        with patch('requests.Session.cert', create=True):
            with VespaSync(app) as sync_app:
                # Check that cert is a tuple (cert_path, key_path)
                self.assertIsInstance(sync_app.cert, tuple)
                cert_path, key_path = sync_app.cert
                
                # Verify that temporary files exist and contain the correct content
                self.assertTrue(os.path.exists(cert_path))
                self.assertTrue(os.path.exists(key_path))
                
                with open(cert_path, 'r') as f:
                    self.assertEqual(f.read(), cert_content)
                
                with open(key_path, 'r') as f:
                    self.assertEqual(f.read(), key_content)
            
            # After exiting context, temporary files should be cleaned up
            self.assertFalse(os.path.exists(cert_path))
            self.assertFalse(os.path.exists(key_path))

    def test_vespa_async_with_cert_content(self):
        """Test that VespaAsync creates temporary files for cert_content and key_content."""
        import tempfile
        import os
        from vespa.application import VespaAsync
        import asyncio
        from unittest.mock import patch
        
        cert_content = "-----BEGIN CERTIFICATE-----\ntest cert content\n-----END CERTIFICATE-----"
        key_content = "-----BEGIN PRIVATE KEY-----\ntest key content\n-----END PRIVATE KEY-----"
        
        app = Vespa(
            url="http://localhost",
            cert_content=cert_content,
            key_content=key_content
        )
        
        async def test_async():
            with patch('httpx.create_ssl_context') as mock_ssl_context:
                # Mock the SSL context creation to avoid SSL validation
                mock_ssl_context.return_value = None
                
                async with VespaAsync(app) as async_app:
                    # Check that temporary files are created
                    self.assertTrue(os.path.exists(async_app._cert_path))
                    self.assertTrue(os.path.exists(async_app._key_path))
                    
                    # Verify that temporary files contain the correct content
                    with open(async_app._cert_path, 'r') as f:
                        self.assertEqual(f.read(), cert_content)
                    
                    with open(async_app._key_path, 'r') as f:
                        self.assertEqual(f.read(), key_content)
                    
                    # Store paths for cleanup verification
                    cert_path = async_app._cert_path
                    key_path = async_app._key_path
                    return cert_path, key_path
                
                # After exiting context, temporary files should be cleaned up
                self.assertFalse(os.path.exists(cert_path))
                self.assertFalse(os.path.exists(key_path))
        
        # Run the async test
        asyncio.run(test_async())


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
            client_kwargs={"timeout": 10},
        )

        # Assert that VespaAsync is initialized once with the client_kwargs
        self.mock_asyncio.assert_called_once_with(
            app=self.vespa,
            connections=2,
            total_timeout=None,
            timeout=10,
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
            query_param="custom",
        )

        # Assert that app.query is called for each query with the query_kwargs
        self.mock_session.query.assert_has_calls(
            [unittest.mock.call(q, query_param="custom") for q in query_data],
            any_order=True,
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

        with (
            patch.object(adapter, "_wait_with_backoff") as mock_backoff,
            patch("requests.adapters.HTTPAdapter.send") as mock_send,
        ):
            mock_response = Mock()
            mock_response.status_code = 429
            mock_send.side_effect = [mock_response, mock_response, mock_response]

            adapter.send(prepared_request)

            self.assertEqual(mock_send.call_count, 3)
            self.assertEqual(mock_backoff.call_count, mock_send.call_count)


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
        app = MockVespa()
        vespa_async = VespaAsync(app)
        assert vespa_async.app == app
        assert vespa_async.httpx_client is None
        assert vespa_async.connections == 1
        assert vespa_async.total_timeout is None
        assert vespa_async.timeout == httpx.Timeout(5)
        assert vespa_async.kwargs == {}
        assert vespa_async.headers == app.base_headers
        assert vespa_async.limits == httpx.Limits(max_keepalive_connections=1)

    def test_init_total_timeout_warns(self):
        app = MockVespa()
        with pytest.warns(DeprecationWarning, match="total_timeout is deprecated"):
            vespa_async = VespaAsync(app, total_timeout=10)
        assert vespa_async.total_timeout == 10

    def test_init_timeout_int(self):
        app = MockVespa()
        vespa_async = VespaAsync(app, timeout=10)
        assert vespa_async.timeout == httpx.Timeout(10)

    def test_init_timeout_timeout(self):
        app = MockVespa()
        timeout = httpx.Timeout(connect=5, read=10, write=15, pool=20)
        vespa_async = VespaAsync(app, timeout=timeout)
        assert vespa_async.timeout == timeout

    def test_init_keepalive_expiry_warning(self):
        app = MockVespa()
        limits = httpx.Limits(keepalive_expiry=31)
        with pytest.warns(
            UserWarning, match="Keepalive expiry is set to more than 30 seconds"
        ):
            _vespa_async = VespaAsync(app, limits=limits)

    def test_init_no_keepalive_expiry_warning(self):
        app = MockVespa()
        limits = httpx.Limits(keepalive_expiry=1)
        _vespa_async = VespaAsync(app, limits=limits)


class TestVespaSyncStreaming(unittest.TestCase):
    def test_query_streaming(self):
        """Test the query method with streaming=True and mocked Server-Sent Events response."""
        # Create a Vespa app instance
        app = Vespa(url="http://localhost", port=8080)

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

        with requests_mock.Mocker() as m:
            # Mock the ApplicationStatus endpoint
            m.get(
                "http://localhost:8080/ApplicationStatus",
                status_code=200,
            )

            # Mock streaming response using iter_lines
            mock_response = Mock()
            mock_response.iter_lines.return_value = [
                line.encode("utf-8") for line in sse_data
            ]

            # Set up context manager behavior
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)
            results = []
            result_string = ""
            # Use the simpler app.syncio() syntax
            with app.syncio() as sync_app:
                # Manually patch the http_session.post to return our mock
                with patch.object(
                    sync_app.http_session, "post", return_value=mock_response
                ):
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

                    # Verify the HTTP session was called with correct parameters
                    sync_app.http_session.post.assert_called_once_with(
                        app.search_end_point,
                        json=query_body,
                        params={"timeout": "30s"},
                        stream=True,
                    )


if __name__ == "__main__":
    unittest.main()
