# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
from pandas import DataFrame

from vespa.package import ApplicationPackage, Schema, Document
from vespa.application import Vespa, parse_feed_df
from learntorank.ml import ModelServer


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
        app_package = ModelServer(name="test")
        app = Vespa(url="http://localhost", port=8080, application_package=app_package)
        with self.assertRaisesRegex(
            ValueError,
            "Application has no schema. Not possible to infer schema name.",
        ):
            _ = app._infer_schema_name()
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
