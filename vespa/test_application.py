# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import unittest

import pandas
import pytest
from unittest.mock import PropertyMock, patch
from pandas import DataFrame
from requests.models import HTTPError, Response

from vespa.package import ApplicationPackage, Schema, Document
from vespa.application import Vespa, parse_feed_df, df_to_vespafeed, raise_for_status
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse, VespaResponse
import requests_mock


def test_df_to_vespafeed():
    df = pandas.DataFrame({
        "id": [0, 1, 2],
        "body": ["body 1", "body 2", "body 3"]
    })
    feed = json.loads(df_to_vespafeed(df, "myschema", "id"))

    assert feed[1]["fields"]["body"] == "body 2"
    assert feed[0]["id"] == "id:myschema:myschema::0"

    feed = json.loads(df_to_vespafeed(df, "myschema", "id", "mynamespace"))
    assert feed[2]["id"] == "id:mynamespace:myschema::2"

class TestVespaRequestsUsage(unittest.TestCase):

    def test_additional_query_params(self):
        app = Vespa(url="http://localhost", port=8080)
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8080/search/", status_code=200, text="{}")
            r:VespaQueryResponse = app.query(
                query="this is a test",
                hits=10,
                searchChain="default"
            )
            self.assertEqual(r.url, "http://localhost:8080/search/?query=this+is+a+test&hits=10&searchChain=default")
        
    def test_additional_doc_params(self):
        app = Vespa(url="http://localhost", port=8080)
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8080/document/v1/foo/foo/docid/0", status_code=200, text="{}")
            r:VespaResponse = app.feed_data_point(
                schema="foo", data_id="0", fields={"body": "this is a test"}, route="default", timeout="10s")
            self.assertEqual(r.url, "http://localhost:8080/document/v1/foo/foo/docid/0?route=default&timeout=10s")

        with requests_mock.Mocker() as m:
            m.put("http://localhost:8080/document/v1/foo/foo/docid/0", status_code=200, text="{}")
            r:VespaResponse = app.update_data(
                schema="foo", data_id="0", fields={"body": "this is a test"}, route="default", timeout="10s")
            self.assertEqual(r.url, "http://localhost:8080/document/v1/foo/foo/docid/0?create=false&route=default&timeout=10s")
        
        with requests_mock.Mocker() as m:
            m.delete("http://localhost:8080/document/v1/foo/foo/docid/0", status_code=200, text="{}")
            r:VespaResponse = app.delete_data(
                schema="foo", data_id="0", route="default", timeout="10s", dryRun=True)
            self.assertEqual(r.url, "http://localhost:8080/document/v1/foo/foo/docid/0?route=default&timeout=10s&dryRun=True")

    def test_delete_all_docs(self):
        app = Vespa(url="http://localhost", port=8080)
        with requests_mock.Mocker() as m:
            m.delete("http://localhost:8080/document/v1/foo/foo/docid/", status_code=200, text="{}")
            r:VespaResponse = app.delete_all_docs(
                schema="foo", namespace="foo", content_cluster_name="content", timeout="200s")
            self.assertEqual(r.url, "http://localhost:8080/document/v1/foo/foo/docid/?cluster=content&selection=true&timeout=200s")
            
        
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
    
    def test_query_token(self):
        self.assertEqual(
            Vespa(url="https://cord19.vespa.ai", vespa_cloud_secret_token="vespa_cloud_str_secret").vespa_cloud_secret_token, "vespa_cloud_str_secret"
        )

    def test_query_token_from_env(self):
        import os
        os.environ['VESPA_CLOUD_SECRET_TOKEN'] = "vespa_cloud_str_secret"
        self.assertEqual(
            Vespa(url="https://cord19.vespa.ai").vespa_cloud_secret_token, "vespa_cloud_str_secret"
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
        # ToDo: re-enable this test later, maybe ...
        #app = Vespa(url="http://localhost", port=8080, application_package=app_package)
        #with self.assertRaisesRegex(
        #    ValueError,
        #    "Application has no schema. Not possible to infer schema name.",
        #):
        #    _ = app._infer_schema_name()
        #
        # More than one schema
        #
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
        #
        # One schema
        #
        app_package = ApplicationPackage(
            name="test",
            schema=[
                Schema(name="x", document=Document()),
            ],
        )
        app = Vespa(url="http://localhost", port=8080, application_package=app_package)
        schema_name = app._infer_schema_name()
        self.assertEqual("x", schema_name)


class TestParseFeedDataFrame(unittest.TestCase):
    def setUp(self) -> None:
        records = [
            {
                "id": idx,
                "title": "This doc is about {}".format(x),
                "body": "There is so much to learn about {}".format(x),
            }
            for idx, x in enumerate(
                ["finance", "sports", "celebrity", "weather", "politics"]
            )
        ]
        self.df = DataFrame.from_records(records)

    def test_parse_simplified_feed_batch(self):
        batch = parse_feed_df(self.df, include_id=True)
        expected_batch = [
            {
                "id": idx,
                "fields": {
                    "id": idx,
                    "title": "This doc is about {}".format(x),
                    "body": "There is so much to learn about {}".format(x),
                },
            }
            for idx, x in enumerate(
                ["finance", "sports", "celebrity", "weather", "politics"]
            )
        ]
        self.assertEqual(expected_batch, batch)

    def test_parse_simplified_feed_batch_not_including_id(self):
        batch = parse_feed_df(self.df, include_id=False)
        expected_batch = [
            {
                "id": idx,
                "fields": {
                    "title": "This doc is about {}".format(x),
                    "body": "There is so much to learn about {}".format(x),
                },
            }
            for idx, x in enumerate(
                ["finance", "sports", "celebrity", "weather", "politics"]
            )
        ]
        self.assertEqual(expected_batch, batch)

    def test_parse_simplified_feed_batch_with_wrong_columns(self):
        missing_id_df = self.df[["title", "body"]]
        with self.assertRaisesRegex(
            AssertionError,
            "DataFrame needs at least the following columns: \['id'\]",
        ):
            _ = parse_feed_df(df=missing_id_df, include_id=True)



class TestRaiseForStatus(unittest.TestCase):
    def test_successful_response(self):
        response = Response()
        response.status_code = 200
        try:
            raise_for_status(response)
        except Exception as e:
            self.fail(f"No exceptions were expected to be raised but {type(e).__name__} occurred")

    def test_successful_response_with_error_content(self):
        with patch("requests.models.Response.content", new_callable=PropertyMock) as mock_content:
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
                self.fail(f"No exceptions were expected to be raised but {type(e).__name__} occurred")

    def test_failure_response_for_400(self):
        response = Response()
        response.status_code = 400
        response.reason = "reason"
        response.url = "http://localhost:8080"
        with pytest.raises(HTTPError) as e:
            raise_for_status(response)
        self.assertEqual(str(e.value), "400 Client Error: reason for url: http://localhost:8080")

    def test_failure_response_for_500(self):
        response = Response()
        response.status_code = 500
        response.reason = "reason"
        response.url = "http://localhost:8080"
        with pytest.raises(HTTPError) as e:
            raise_for_status(response)
        self.assertEqual(str(e.value), "500 Server Error: reason for url: http://localhost:8080")

    def test_failure_response_without_error_content(self):
        with patch("requests.models.Response.content", new_callable=PropertyMock) as mock_content:
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
        with patch("requests.models.Response.content", new_callable=PropertyMock) as mock_content:
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
        with patch("requests.models.Response.content", new_callable=PropertyMock) as mock_content:
            response_json = {
                "root": {
                    "errors": [
                        {"code": 12, "summary": "Timed out", "message": "No time left after waiting for 1ms to execute query"},
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
            self.assertEqual(str(e.value), "[{'code': 12, 'summary': 'Timed out', 'message': 'No time left after waiting for 1ms to execute query'}]")

    def test_doc_failure_response_with_error_content(self):
        with patch("requests.models.Response.content", new_callable=PropertyMock) as mock_content:
            response_json = {
                "pathId": "/document/v1/textsearch/textsearch/docid/00",
                  "message": "No field 'foo' in the structure of type 'textsearch'"
            }
            mock_content.return_value = json.dumps(response_json).encode("utf-8")
            response = Response()
            response.status_code = 400
            response.reason = "Bad Request"
            response.url = "http://localhost:8080/document/v1/textsearch/textsearch/docid/00"
            with pytest.raises(VespaError) as e:
                raise_for_status(response)
            self.assertEqual(str(e.value), "No field 'foo' in the structure of type 'textsearch'")

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


class TestVespaEvaluate(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Vespa(url="http://localhost", port=8080)

        self.labeled_data = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": "def", "score": 1}, {"id": "abc", "score": 1}],
            },
        ]

        self.query_results = {
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
                            "vespa_id_field": "ghi",
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
                            "vespa_id_field": "def",
                            "sddocname": "doc",
                            "body_text": "this is a body 3",
                            "title": "this is a title 3",
                            "rankfeatures": {"a": 5, "b": 6},
                        },
                    },
                ],
            }
        }
