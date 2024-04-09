# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
from vespa.io import VespaQueryResponse


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
        vespa_result = VespaQueryResponse(
            json=self.raw_vespa_result, status_code=None, url=None
        )
        self.assertDictEqual(vespa_result.json, self.raw_vespa_result)

    def test_hits(self):
        empty_hits_vespa_result = VespaQueryResponse(
            json=self.raw_vespa_result_empty_hits, status_code=None, url=None
        )
        self.assertEqual(empty_hits_vespa_result.hits, [])
        vespa_result = VespaQueryResponse(
            json=self.raw_vespa_result, status_code=None, url=None
        )
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
