# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
from unittest.mock import Mock, call
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from vespa.application import Vespa, parse_labeled_data
from vespa.io import VespaQueryResponse
from vespa.query import QueryModel, OR, RankProfile


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


class TestVespaQuery(unittest.TestCase):
    def test_query(self):
        app = Vespa(url="http://localhost", port=8080)

        body = {"yql": "select * from sources * where test"}
        self.assertDictEqual(
            app.query(body=body, debug_request=True).request_body, body
        )

        self.assertDictEqual(
            app.query(
                query="this is a test",
                query_model=QueryModel(match_phase=OR(), rank_profile=RankProfile()),
                debug_request=True,
                hits=10,
            ).request_body,
            {
                "yql": 'select * from sources * where ([{"grammar": "any"}]userInput("this is a test"));',
                "ranking": {"profile": "default", "listFeatures": "false"},
                "hits": 10,
            },
        )

        self.assertDictEqual(
            app.query(
                query="this is a test",
                query_model=QueryModel(match_phase=OR(), rank_profile=RankProfile()),
                debug_request=True,
                hits=10,
                recall=("id", [1, 5]),
            ).request_body,
            {
                "yql": 'select * from sources * where ([{"grammar": "any"}]userInput("this is a test"));',
                "ranking": {"profile": "default", "listFeatures": "false"},
                "hits": 10,
                "recall": "+(id:1 id:5)",
            },
        )

    def test_query_with_body_function(self):
        app = Vespa(url="http://localhost", port=8080)

        def body_function(query):
            body = {
                "yql": "select * from sources * where userQuery();",
                "query": query,
                "type": "any",
                "ranking": {"profile": "bm25", "listFeatures": "true"},
            }
            return body

        query_model = QueryModel(body_function=body_function)

        self.assertDictEqual(
            app.query(
                query="this is a test",
                query_model=query_model,
                debug_request=True,
                hits=10,
                recall=("id", [1, 5]),
            ).request_body,
            {
                "yql": "select * from sources * where userQuery();",
                "query": "this is a test",
                "type": "any",
                "ranking": {"profile": "bm25", "listFeatures": "true"},
                "hits": 10,
                "recall": "+(id:1 id:5)",
            },
        )


class TestLabeledData(unittest.TestCase):
    def test_parse_labeled_data(self):
        labeled_data_df = DataFrame(
            data={
                "qid": [0, 0, 1, 1],
                "query": ["Intrauterine virus infections and congenital heart disease"]
                * 2
                + [
                    "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus"
                ]
                * 2,
                "doc_id": [0, 3, 1, 5],
                "relevance": [1, 1, 1, 1],
            }
        )
        labeled_data = parse_labeled_data(df=labeled_data_df)
        expected_labeled_data = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": 0, "score": 1}, {"id": 3, "score": 1}],
            },
            {
                "query_id": 1,
                "query": "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus",
                "relevant_docs": [{"id": 1, "score": 1}, {"id": 5, "score": 1}],
            },
        ]
        self.assertEqual(labeled_data, expected_labeled_data)

    def test_parse_labeled_data_with_wrong_columns(self):
        labeled_data_df = DataFrame(
            data={
                "qid": [0, 0, 1, 1],
                "doc_id": [0, 3, 1, 5],
                "relevance": [1, 1, 1, 1],
            }
        )
        with self.assertRaises(AssertionError):
            _ = parse_labeled_data(df=labeled_data_df)


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

    def test_collect_training_data_point(self):

        self.app.query = Mock(
            side_effect=[
                VespaQueryResponse(self.raw_vespa_result_recall, status_code=None, url=None),
                VespaQueryResponse(self.raw_vespa_result_additional, status_code=None, url=None),
            ]
        )
        query_model = QueryModel(rank_profile=RankProfile(list_features=True))
        data = self.app.collect_training_data_point(
            query="this is a query",
            query_id="123",
            relevant_id="abc",
            id_field="vespa_id_field",
            query_model=query_model,
            number_additional_docs=2,
            fields=["rankfeatures", "title"],
            timeout="15s",
        )

        self.assertEqual(self.app.query.call_count, 2)
        self.app.query.assert_has_calls(
            [
                call(
                    query="this is a query",
                    query_model=query_model,
                    recall=("vespa_id_field", ["abc"]),
                    timeout="15s",
                ),
                call(
                    query="this is a query",
                    query_model=query_model,
                    hits=2,
                    timeout="15s",
                ),
            ]
        )
        expected_data = [
            {
                "document_id": "abc",
                "query_id": "123",
                "label": 1,
                "a": 1,
                "b": 2,
                "title": "this is a title",
            },
            {
                "document_id": "def",
                "query_id": "123",
                "label": 0,
                "a": 3,
                "b": 4,
                "title": "this is a title 2",
            },
            {
                "document_id": "ghi",
                "query_id": "123",
                "label": 0,
                "a": 5,
                "b": 6,
                "title": "this is a title 3",
            },
        ]
        self.assertEqual(data, expected_data)

    def test_collect_training_data_point_absent_field(self):

        self.app.query = Mock(
            side_effect=[
                VespaQueryResponse(self.raw_vespa_result_recall, status_code=None, url=None),
                VespaQueryResponse(self.raw_vespa_result_additional, status_code=None, url=None),
            ]
        )
        query_model = QueryModel(rank_profile=RankProfile(list_features=True))
        data = self.app.collect_training_data_point(
            query="this is a query",
            query_id="123",
            relevant_id="abc",
            id_field="vespa_id_field",
            query_model=query_model,
            number_additional_docs=2,
            fields=["rankfeatures", "crazy_field"],
            timeout="15s",
        )

        self.assertEqual(self.app.query.call_count, 2)
        self.app.query.assert_has_calls(
            [
                call(
                    query="this is a query",
                    query_model=query_model,
                    recall=("vespa_id_field", ["abc"]),
                    timeout="15s",
                ),
                call(
                    query="this is a query",
                    query_model=query_model,
                    hits=2,
                    timeout="15s",
                ),
            ]
        )
        expected_data = [
            {
                "document_id": "abc",
                "query_id": "123",
                "label": 1,
                "a": 1,
                "b": 2,
            },
            {
                "document_id": "def",
                "query_id": "123",
                "label": 0,
                "a": 3,
                "b": 4,
            },
            {
                "document_id": "ghi",
                "query_id": "123",
                "label": 0,
                "a": 5,
                "b": 6,
            },
        ]
        self.assertEqual(data, expected_data)

    def test_collect_training_data_point_0_recall_hits(self):

        self.raw_vespa_result_recall = {
            "root": {
                "id": "toplevel",
                "relevance": 1.0,
                "fields": {"totalCount": 0},
                "coverage": {
                    "coverage": 100,
                    "documents": 62529,
                    "full": True,
                    "nodes": 2,
                    "results": 1,
                    "resultsFull": 1,
                },
            }
        }
        self.app.query = Mock(
            side_effect=[
                VespaQueryResponse(self.raw_vespa_result_recall, status_code=None, url=None),
                VespaQueryResponse(self.raw_vespa_result_additional, status_code=None, url=None),
            ]
        )
        query_model = QueryModel(rank_profile=RankProfile(list_features=True))
        data = self.app.collect_training_data_point(
            query="this is a query",
            query_id="123",
            relevant_id="abc",
            id_field="vespa_id_field",
            query_model=query_model,
            number_additional_docs=2,
            fields=["rankfeatures"],
            timeout="15s",
        )

        self.assertEqual(self.app.query.call_count, 1)
        self.app.query.assert_has_calls(
            [
                call(
                    query="this is a query",
                    query_model=query_model,
                    recall=("vespa_id_field", ["abc"]),
                    timeout="15s",
                ),
            ]
        )
        expected_data = []
        self.assertEqual(data, expected_data)

    def test_collect_training_data(self):

        mock_return_value = [
            {
                "document_id": "abc",
                "query_id": "123",
                "relevant": 1,
                "a": 1,
                "b": 2,
            },
            {
                "document_id": "def",
                "query_id": "123",
                "relevant": 0,
                "a": 3,
                "b": 4,
            },
            {
                "document_id": "ghi",
                "query_id": "123",
                "relevant": 0,
                "a": 5,
                "b": 6,
            },
        ]
        self.app.collect_training_data_point = Mock(return_value=mock_return_value)
        labeled_data = [
            {
                "query_id": 123,
                "query": "this is a query",
                "relevant_docs": [{"id": "abc", "score": 1}],
            }
        ]
        query_model = QueryModel(rank_profile=RankProfile(list_features=True))
        data = self.app.collect_training_data(
            labeled_data=labeled_data,
            id_field="vespa_id_field",
            query_model=query_model,
            number_additional_docs=2,
            timeout="15s",
        )
        self.app.collect_training_data_point.assert_has_calls(
            [
                call(
                    query="this is a query",
                    query_id=123,
                    relevant_id="abc",
                    id_field="vespa_id_field",
                    query_model=query_model,
                    number_additional_docs=2,
                    relevant_score=1,
                    default_score=0,
                    timeout="15s",
                )
            ]
        )
        assert_frame_equal(data, DataFrame.from_records(mock_return_value))


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

    def test_evaluate_query(self):
        self.app.query = Mock(return_value={})
        eval_metric = Mock()
        eval_metric.evaluate_query = Mock(return_value={"metric": 1})
        eval_metric2 = Mock()
        eval_metric2.evaluate_query = Mock(return_value={"metric_2": 2})
        query_model = QueryModel()
        evaluation = self.app.evaluate_query(
            eval_metrics=[eval_metric, eval_metric2],
            query_model=query_model,
            query_id="0",
            query="this is a test",
            id_field="vespa_id_field",
            relevant_docs=self.labeled_data[0]["relevant_docs"],
            default_score=0,
            hits=10,
        )
        self.assertEqual(self.app.query.call_count, 1)
        self.app.query.assert_has_calls(
            [
                call(query="this is a test", query_model=query_model, hits=10),
            ]
        )
        self.assertEqual(eval_metric.evaluate_query.call_count, 1)
        eval_metric.evaluate_query.assert_has_calls(
            [
                call(
                    {},
                    self.labeled_data[0]["relevant_docs"],
                    "vespa_id_field",
                    0,
                    False,
                ),
            ]
        )
        self.assertDictEqual(
            evaluation,
            {"model": "default_name", "query_id": "0", "metric": 1, "metric_2": 2},
        )

