# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from vespa.query import (
    QueryModel,
    QueryRankingFeature,
    OR,
    AND,
    WeakAnd,
    ANN,
    Union,
    RankProfile,
)
from vespa.io import VespaQueryResponse


class TestQueryProperty(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "this is  a test"

    def test_query_ranking_feature(self):
        query_property = QueryRankingFeature(
            name="query_vector", mapping=lambda x: [1, 2, 3]
        )
        self.assertDictEqual(
            query_property.get_query_properties(query=self.query),
            {"ranking.features.query(query_vector)": "[1, 2, 3]"},
        )


class TestMatchFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "this is  a test"

    def test_and(self):
        match_filter = AND()
        self.assertEqual(
            match_filter.create_match_filter(query=self.query),
            '(userInput("this is  a test"))',
        )
        self.assertDictEqual(match_filter.get_query_properties(query=self.query), {})

    def test_or(self):
        match_filter = OR()
        self.assertEqual(
            match_filter.create_match_filter(query=self.query),
            '([{"grammar": "any"}]userInput("this is  a test"))',
        )
        self.assertDictEqual(match_filter.get_query_properties(query=self.query), {})

    def test_weak_and(self):
        match_filter = WeakAnd(hits=10, field="field_name")
        self.assertEqual(
            match_filter.create_match_filter(query=self.query),
            '([{"targetNumHits": 10}]weakAnd(field_name contains "this", field_name contains "is", field_name contains "", '
            'field_name contains "a", field_name contains "test"))',
        )
        self.assertDictEqual(match_filter.get_query_properties(query=self.query), {})

    def test_ann(self):
        match_filter = ANN(
            doc_vector="doc_vector",
            query_vector="query_vector",
            hits=10,
            label="label",
        )
        self.assertEqual(
            match_filter.create_match_filter(query=self.query),
            '([{"targetNumHits": 10, "label": "label", "approximate": true}]nearestNeighbor(doc_vector, query_vector))',
        )
        self.assertDictEqual(
            match_filter.get_query_properties(query=self.query),
            {},
        )

        match_filter = ANN(
            doc_vector="doc_vector",
            query_vector="query_vector",
            hits=10,
            label="label",
            approximate=False,
        )
        self.assertEqual(
            match_filter.create_match_filter(query=self.query),
            '([{"targetNumHits": 10, "label": "label", "approximate": false}]nearestNeighbor(doc_vector, query_vector))',
        )
        self.assertDictEqual(
            match_filter.get_query_properties(query=self.query),
            {},
        )

    def test_union(self):
        match_filter = Union(
            WeakAnd(hits=10, field="field_name"),
            ANN(
                doc_vector="doc_vector",
                query_vector="query_vector",
                hits=10,
                label="label",
            ),
        )
        self.assertEqual(
            match_filter.create_match_filter(query=self.query),
            '([{"targetNumHits": 10}]weakAnd(field_name contains "this", field_name contains "is", '
            'field_name contains "", '
            'field_name contains "a", field_name contains "test")) or '
            '([{"targetNumHits": 10, "label": "label", "approximate": true}]nearestNeighbor(doc_vector, query_vector))',
        )
        self.assertDictEqual(
            match_filter.get_query_properties(query=self.query),
            {},
        )


class TestRankProfile(unittest.TestCase):
    def test_rank_profile(self):
        rank_profile = RankProfile(name="rank_profile", list_features=True)
        self.assertEqual(rank_profile.name, "rank_profile")
        self.assertEqual(rank_profile.list_features, "true")


class TestQuery(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "this is  a test"

    def test_default(self):
        query = QueryModel()
        self.assertDictEqual(
            query.create_body(query=self.query),
            {
                "yql": 'select * from sources * where (userInput("this is  a test"));',
                "ranking": {"profile": "default", "listFeatures": "false"},
            },
        )

    def test_body_function(self):
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
            query_model.create_body(query=self.query),
            {
                "yql": "select * from sources * where userQuery();",
                "query": "this is  a test",
                "type": "any",
                "ranking": {"profile": "bm25", "listFeatures": "true"},
            },
        )

    def test_query_properties_match_and_rank(self):

        query_model = QueryModel(
            query_properties=[
                QueryRankingFeature(name="query_vector", mapping=lambda x: [1, 2, 3])
            ],
            match_phase=OR(),
            rank_profile=RankProfile(name="bm25", list_features=True),
        )
        self.assertDictEqual(
            query_model.create_body(query=self.query),
            {
                "yql": 'select * from sources * where ([{"grammar": "any"}]userInput("this is  a test"));',
                "ranking": {"profile": "bm25", "listFeatures": "true"},
                "ranking.features.query(query_vector)": "[1, 2, 3]",
            },
        )

        query_model = QueryModel(
            query_properties=[
                QueryRankingFeature(name="query_vector", mapping=lambda x: [1, 2, 3])
            ],
            match_phase=ANN(
                doc_vector="doc_vector",
                query_vector="query_vector",
                hits=10,
                label="label",
            ),
            rank_profile=RankProfile(name="bm25", list_features=True),
        )
        self.assertDictEqual(
            query_model.create_body(query=self.query),
            {
                "yql": 'select * from sources * where ([{"targetNumHits": 10, "label": "label", "approximate": true}]nearestNeighbor(doc_vector, query_vector));',
                "ranking": {"profile": "bm25", "listFeatures": "true"},
                "ranking.features.query(query_vector)": "[1, 2, 3]",
            },
        )


class TestVespaResult(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_vespa_result_empty_hits = {
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

        self.raw_vespa_result = {
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
                            "sddocname": "doc",
                            "body_text": "this is a body",
                            "title": "this is a title",
                        },
                    }
                ],
            }
        }

        self.raw_vespa_result_multiple = {
            "root": {
                "id": "toplevel",
                "relevance": 1.0,
                "fields": {"totalCount": 236369},
                "coverage": {
                    "coverage": 100,
                    "documents": 309201,
                    "full": True,
                    "nodes": 2,
                    "results": 1,
                    "resultsFull": 1,
                },
                "children": [
                    {
                        "id": "id:covid-19:doc::142863",
                        "relevance": 11.893775281559614,
                        "source": "content",
                        "fields": {
                            "sddocname": "doc",
                            "title": "A workable strategy for COVID-19 <hi>testing</hi>: stratified periodic <hi>testing</hi> rather than universal random <hi>testing</hi>",
                            "cord_uid": "0p6vrujx",
                        },
                    },
                    {
                        "id": "id:covid-19:doc::31328",
                        "relevance": 11.891779491908013,
                        "source": "content",
                        "fields": {
                            "sddocname": "doc",
                            "title": "A workable strategy for COVID-19 <hi>testing</hi>: stratified periodic <hi>testing</hi> rather than universal random <hi>testing</hi>",
                            "cord_uid": "moy0u7n5",
                        },
                    },
                    {
                        "id": "id:covid-19:doc::187156",
                        "relevance": 11.887045666057702,
                        "source": "content",
                        "fields": {
                            "sddocname": "doc",
                            "title": "A comparison of group <hi>testing</hi> architectures for COVID-19 <hi>testing</hi>",
                            "cord_uid": "rhmywn8n",
                        },
                    },
                ],
            }
        }

    def test_json(self):
        vespa_result = VespaQueryResponse(json=self.raw_vespa_result, status_code=None, url=None)
        self.assertDictEqual(vespa_result.json, self.raw_vespa_result)

    def test_hits(self):
        empty_hits_vespa_result = VespaQueryResponse(json=self.raw_vespa_result_empty_hits, status_code=None, url=None)
        self.assertEqual(empty_hits_vespa_result.hits, [])
        vespa_result = VespaQueryResponse(json=self.raw_vespa_result, status_code=None, url=None)
        self.assertEqual(
            vespa_result.hits,
            [
                {
                    "id": "id:covid-19:doc::40215",
                    "relevance": 30.368213170494712,
                    "source": "content",
                    "fields": {
                        "sddocname": "doc",
                        "body_text": "this is a body",
                        "title": "this is a title",
                    },
                }
            ],
        )

    def test_get_hits_none(self):
        empty_hits_vespa_result = VespaQueryResponse(json=self.raw_vespa_result_empty_hits, status_code=None, url=None)
        self.assertEqual(
            empty_hits_vespa_result.get_hits(format_function=None),
            empty_hits_vespa_result.hits,
        )
        vespa_result = VespaQueryResponse(json=self.raw_vespa_result, status_code=None, url=None)
        self.assertEqual(
            vespa_result.get_hits(format_function=None),
            vespa_result.hits,
        )

    def test_get_hits_trec_format(self):
        empty_hits_vespa_result = VespaQueryResponse(json=self.raw_vespa_result_empty_hits, status_code=None, url=None)
        self.assertTrue(empty_hits_vespa_result.get_hits().empty)
        vespa_result = VespaQueryResponse(json=self.raw_vespa_result, status_code=None, url=None)
        assert_frame_equal(
            vespa_result.get_hits(),
            DataFrame(
                data={
                    "qid": [0],
                    "doc_id": ["id:covid-19:doc::40215"],
                    "score": [30.368213170494712],
                    "rank": [0],
                },
                columns=["qid", "doc_id", "score", "rank"],
            ),
        )
        vespa_result = VespaQueryResponse(json=self.raw_vespa_result_multiple, status_code=None, url=None)
        assert_frame_equal(
            vespa_result.get_hits(),
            DataFrame(
                data={
                    "qid": [0, 0, 0],
                    "doc_id": [
                        "id:covid-19:doc::142863",
                        "id:covid-19:doc::31328",
                        "id:covid-19:doc::187156",
                    ],
                    "score": [
                        11.893775281559614,
                        11.891779491908013,
                        11.887045666057702,
                    ],
                    "rank": [0, 1, 2],
                },
                columns=["qid", "doc_id", "score", "rank"],
            ),
        )
        assert_frame_equal(
            vespa_result.get_hits(id_field="cord_uid", qid=2),
            DataFrame(
                data={
                    "qid": [2, 2, 2],
                    "doc_id": [
                        "0p6vrujx",
                        "moy0u7n5",
                        "rhmywn8n",
                    ],
                    "score": [
                        11.893775281559614,
                        11.891779491908013,
                        11.887045666057702,
                    ],
                    "rank": [0, 1, 2],
                },
                columns=["qid", "doc_id", "score", "rank"],
            ),
        )
