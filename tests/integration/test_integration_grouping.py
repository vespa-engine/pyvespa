import unittest
import requests
from vespa.deployment import VespaDocker
from vespa.package import (
    ApplicationPackage,
    Schema,
    Document,
    Field,
    RankProfile,
    FirstPhaseRanking,
    FieldSet,
    DocumentSummary,
    Summary,
)
from tests.unit.test_grouping import GroupingQueries

qb = GroupingQueries()


class TestGroupingIntegration(unittest.TestCase):
    """Integration tests for grouping.
    Most of the tests and data are retrieved from https://github.com/vespa-engine/system-test/tree/master/tests/search/grouping_adv
    and https://github.com/vespa-engine/system-test/blob/master/tests/search/grouping/
    """

    @classmethod
    def setUpClass(cls):
        application_name = "grouping"
        cls.application_name = application_name
        schema_name = "purchase"
        cls.schema_name = schema_name
        # Define all fields used in the unit tests
        # Add purchase schema for grouping test
        purchase_schema = Schema(
            name=schema_name,
            document=Document(
                fields=[
                    Field(name="date", type="long", indexing=["summary", "attribute"]),
                    Field(name="price", type="int", indexing=["summary", "attribute"]),
                    Field(name="tax", type="double", indexing=["summary", "attribute"]),
                    Field(
                        name="item", type="string", indexing=["summary", "attribute"]
                    ),
                    Field(
                        name="customer",
                        type="string",
                        indexing=["summary", "attribute"],
                    ),
                ]
            ),
            rank_profiles=[
                RankProfile(name="pricerank", first_phase="attribute(price)")
            ],
        )
        # test schema:
        # Copyright Vespa.ai. All rights reserved.
        # schema test {
        #   document test {
        #     field n type int {
        #       indexing: summary | attribute
        #     }
        #     field fa type array<double> {
        #       indexing: attribute
        #     }
        #     field na type array<int> {
        #       indexing: attribute
        #     }
        #     field nb type array<byte> {
        #       indexing: attribute
        #     }
        #     field nw type weightedset<int> {
        #       indexing: attribute
        #     }
        #     field f type float {
        #       indexing: attribute
        #     }
        #     field d type double {
        #       indexing: attribute
        #     }
        #     field sf type string {
        #       indexing: attribute
        #     }
        #     field s type string {
        #       indexing: attribute | index
        #     }
        #     field a type string {
        #       indexing: attribute | index | summary
        #     }
        #     field b type string {
        #       indexing: attribute | index | summary
        #     }
        #     field c type string {
        #       indexing: attribute | index | summary
        #     }
        #     field from type int {
        #       indexing: attribute | summary
        #     }
        #     field to type long {
        #       indexing: attribute | summary
        #     }
        #     field lang type string {
        #       indexing: attribute
        #     }
        #     field body type string {
        #       indexing: index | summary
        #       rank-type: identity
        #     }
        #     field boool type bool {
        #       indexing: attribute | summary
        #     }
        #     field by type byte {
        #       indexing: attribute
        #     }
        #     field i type int {
        #       indexing: attribute
        #     }
        #   }
        #   fieldset default {
        #     fields: body
        #   }

        #   rank-profile default {
        #     first-phase {
        #       expression: attribute(f) * (attribute(from) / 1000000)
        #     }
        #   }

        #   rank-profile default-values {
        #     first-phase {
        #       expression: attribute(i)
        #     }
        #   }

        #   document-summary normal {
        #     summary a { source: a }
        #     summary b { source: b }
        #     summary c { source: c }
        #     summary documentid { source: documentid }
        #     summary from { source: from }
        #     summary to { source: to }
        #     summary body { source: body }
        #   }

        #   document-summary summary1 {
        #     summary a { source: a }
        #     summary n { source: n }
        #   }
        # }
        # Schema from https://github.com/vespa-engine/system-test/blob/master/tests/search/grouping_adv/test.sd
        test_schema = Schema(
            name="test",
            document=Document(
                fields=[
                    Field(name="n", type="int", indexing=["summary", "attribute"]),
                    Field(name="fa", type="array<double>", indexing=["attribute"]),
                    Field(name="na", type="array<int>", indexing=["attribute"]),
                    Field(name="nb", type="array<byte>", indexing=["attribute"]),
                    Field(name="nw", type="weightedset<int>", indexing=["attribute"]),
                    Field(name="f", type="float", indexing=["attribute"]),
                    Field(name="d", type="double", indexing=["attribute"]),
                    Field(name="sf", type="string", indexing=["attribute"]),
                    Field(name="s", type="string", indexing=["attribute", "index"]),
                    Field(
                        name="a",
                        type="string",
                        indexing=["attribute", "index", "summary"],
                    ),
                    Field(
                        name="b",
                        type="string",
                        indexing=["attribute", "index", "summary"],
                    ),
                    Field(
                        name="c",
                        type="string",
                        indexing=["attribute", "index", "summary"],
                    ),
                    Field(name="from", type="int", indexing=["attribute", "summary"]),
                    Field(name="to", type="long", indexing=["attribute", "summary"]),
                    Field(name="lang", type="string", indexing=["attribute"]),
                    Field(name="body", type="string", indexing=["index", "summary"]),
                    Field(name="boool", type="bool", indexing=["attribute", "summary"]),
                    Field(name="by", type="byte", indexing=["attribute"]),
                    Field(name="i", type="int", indexing=["attribute"]),
                ]
            ),
            fieldset=FieldSet(name="default", fields=["body"]),
            rank_profiles=[
                RankProfile(
                    name="default",
                    first_phase=FirstPhaseRanking(
                        expression="attribute(f) * (attribute(from) / 1000000)"
                    ),
                ),
                RankProfile(
                    name="default-values",
                    first_phase=FirstPhaseRanking(expression="attribute(i)"),
                ),
            ],
            document_summaries=[
                DocumentSummary(
                    name="normal",
                    summary_fields=[
                        Summary(name="a", fields=[("source", "a")]),
                        Summary(name="b", fields=[("source", "b")]),
                        Summary(name="c", fields=[("source", "c")]),
                        Summary(name="documentid", fields=[("source", "documentid")]),
                        Summary(name="from", fields=[("source", "from")]),
                        Summary(name="to", fields=[("source", "to")]),
                        Summary(name="body", fields=[("source", "body")]),
                    ],
                ),
                DocumentSummary(
                    name="summary1",
                    summary_fields=[
                        Summary(name="a", fields=[("source", "a")]),
                        Summary(name="n", fields=[("source", "n")]),
                    ],
                ),
            ],
        )
        # Create the application package
        application_package = ApplicationPackage(
            name=application_name, schema=[purchase_schema, test_schema]
        )
        print(application_package.get_schema("purchase").schema_to_text)
        print(application_package.get_schema("test").schema_to_text)
        # Deploy the application
        cls.vespa_docker = VespaDocker(port=8089)
        cls.app = cls.vespa_docker.deploy(application_package=application_package)
        cls.app.wait_for_application_up()

    @classmethod
    def tearDownClass(cls):
        cls.vespa_docker.container.stop(timeout=5)
        cls.vespa_docker.container.remove()

    @property
    def purchase_grouping_data(self):
        sample_data = [
            {
                "fields": {
                    "customer": "Smith",
                    "date": 1157526000,
                    "item": "Intake valve",
                    "price": "1000",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::0",
            },
            {
                "fields": {
                    "customer": "Smith",
                    "date": 1157616000,
                    "item": "Rocker arm",
                    "price": "1000",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::1",
            },
            {
                "fields": {
                    "customer": "Smith",
                    "date": 1157619600,
                    "item": "Spring",
                    "price": "2000",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::2",
            },
            {
                "fields": {
                    "customer": "Jones",
                    "date": 1157709600,
                    "item": "Valve cover",
                    "price": "3000",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::3",
            },
            {
                "fields": {
                    "customer": "Jones",
                    "date": 1157702400,
                    "item": "Intake port",
                    "price": "5000",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::4",
            },
            {
                "fields": {
                    "customer": "Brown",
                    "date": 1157706000,
                    "item": "Head",
                    "price": "8000",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::5",
            },
            {
                "fields": {
                    "customer": "Smith",
                    "date": 1157796000,
                    "item": "Coolant",
                    "price": "1300",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::6",
            },
            {
                "fields": {
                    "customer": "Jones",
                    "date": 1157788800,
                    "item": "Engine block",
                    "price": "2100",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::7",
            },
            {
                "fields": {
                    "customer": "Brown",
                    "date": 1157792400,
                    "item": "Oil pan",
                    "price": "3400",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::8",
            },
            {
                "fields": {
                    "customer": "Smith",
                    "date": 1157796000,
                    "item": "Oil sump",
                    "price": "5500",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::9",
            },
            {
                "fields": {
                    "customer": "Jones",
                    "date": 1157875200,
                    "item": "Camshaft",
                    "price": "8900",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::10",
            },
            {
                "fields": {
                    "customer": "Brown",
                    "date": 1157878800,
                    "item": "Exhaust valve",
                    "price": "1440",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::11",
            },
            {
                "fields": {
                    "customer": "Brown",
                    "date": 1157882400,
                    "item": "Rocker arm",
                    "price": "2330",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::12",
            },
            {
                "fields": {
                    "customer": "Brown",
                    "date": 1157875200,
                    "item": "Spring",
                    "price": "3770",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::13",
            },
            {
                "fields": {
                    "customer": "Smith",
                    "date": 1157878800,
                    "item": "Spark plug",
                    "price": "6100",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::14",
            },
            {
                "fields": {
                    "customer": "Jones",
                    "date": 1157968800,
                    "item": "Exhaust port",
                    "price": "9870",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::15",
            },
            {
                "fields": {
                    "customer": "Brown",
                    "date": 1157961600,
                    "item": "Piston",
                    "price": "1597",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::16",
            },
            {
                "fields": {
                    "customer": "Smith",
                    "date": 1157965200,
                    "item": "Connection rod",
                    "price": "2584",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::17",
            },
            {
                "fields": {
                    "customer": "Jones",
                    "date": 1157968800,
                    "item": "Rod bearing",
                    "price": "4181",
                    "tax": "0.24",
                },
                "put": "id:purchase:purchase::18",
            },
            {
                "fields": {
                    "customer": "Jones",
                    "date": 1157972400,
                    "item": "Crankshaft",
                    "price": "6765",
                    "tax": "0.12",
                },
                "put": "id:purchase:purchase::19",
            },
        ]
        docs = [
            {"fields": doc["fields"], "id": doc["put"].split("::")[-1]}
            for doc in sample_data
        ]
        return docs

    @property
    def test_grouping_data(self):
        docs = [
            {
                "put": "id:test:test:n=1234:01",
                "fields": {
                    "s": "a",
                    "sf": 1,
                    "f": 1,
                    "d": 1,
                    "n": 1,
                    "a": "a1",
                    "b": "b1",
                    "c": "c1",
                    "from": 1234517162,
                    "to": 1234517162000000000,
                    "boool": True,
                    "body": "test",
                    "na": [1, 2],
                    "nb": [7, 9],
                    "lang": "heissz",
                },
            },
            {
                "put": "id:test:test:n=1234:03",
                "fields": {
                    "s": "aaa",
                    "sf": 3.9,
                    "f": 3.9,
                    "d": 3.9,
                    "n": 3,
                    "a": "a1",
                    "b": "b1",
                    "c": "c3",
                    "from": 1234517151,
                    "to": 1234537162000000000,
                    "boool": False,
                    "body": "test",
                    "fa": [1.6, 2.9],
                    "lang": "heissz",
                },
            },
            {
                "put": "id:test:test:n=1234:04",
                "fields": {
                    "s": "aab",
                    "sf": 4.9,
                    "f": 4.9,
                    "d": 4.9,
                    "n": 4,
                    "a": "a1",
                    "b": "b2",
                    "c": "c1",
                    "from": 1234517051,
                    "to": 1234547162000000000,
                    "boool": True,
                    "body": "test",
                    "fa": [1.6],
                    "lang": "heißz",
                },
            },
            {
                "put": "id:test:test:n=1234:05",
                "fields": {
                    "s": "aac",
                    "sf": 5.9,
                    "f": 5.9,
                    "d": 5.9,
                    "n": 5,
                    "a": "a1",
                    "b": "b2",
                    "c": "c2",
                    "from": 1234516051,
                    "to": 1234557162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissz",
                },
            },
            {
                "put": "id:test:test:n=1234:06",
                "fields": {
                    "s": "ab",
                    "sf": 6.9,
                    "f": 6.9,
                    "d": 6.9,
                    "n": 6,
                    "a": "a1",
                    "b": "b2",
                    "c": "c3",
                    "from": 1234506051,
                    "to": 1234567162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissz",
                },
            },
            {
                "put": "id:test:test:n=1234:07",
                "fields": {
                    "s": "aba",
                    "sf": 7.9,
                    "f": 7.9,
                    "d": 7.9,
                    "n": 7,
                    "a": "a1",
                    "b": "b3",
                    "c": "c1",
                    "from": 1234406051,
                    "to": 1234577162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heissz",
                },
            },
            {
                "put": "id:test:test:n=1234:08",
                "fields": {
                    "s": "abb",
                    "sf": 8.9,
                    "f": 8.9,
                    "d": 8.9,
                    "n": 8,
                    "a": "a1",
                    "b": "b3",
                    "c": "c2",
                    "from": 1233406051,
                    "to": 1234587162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=1234:09",
                "fields": {
                    "s": "abc",
                    "sf": 9.9,
                    "f": 9.9,
                    "d": 9.9,
                    "n": 9,
                    "a": "a1",
                    "b": "b3",
                    "c": "c3",
                    "from": 1223406051,
                    "to": 1234607162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=1234:10",
                "fields": {
                    "s": "ac",
                    "sf": 1.9,
                    "f": 1.9,
                    "d": 1.9,
                    "n": 1,
                    "a": "a2",
                    "b": "b1",
                    "c": "c1",
                    "from": 1123406051,
                    "to": 1234617162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=1234:11",
                "fields": {
                    "s": "aca",
                    "sf": 2.9,
                    "f": 2.9,
                    "d": 2.9,
                    "n": 2,
                    "a": "a2",
                    "b": "b1",
                    "c": "c2",
                    "from": 123406051,
                    "to": 1234627162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=1234:12",
                "fields": {
                    "s": "acb",
                    "sf": 3.9,
                    "f": 3.9,
                    "d": 3.9,
                    "n": 3,
                    "a": "a2",
                    "b": "b1",
                    "c": "c3",
                    "from": 1123406050,
                    "to": 1234637162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=1234:13",
                "fields": {
                    "s": "acc",
                    "sf": 4.9,
                    "f": 4.9,
                    "d": 4.9,
                    "n": 4,
                    "a": "a2",
                    "b": "b2",
                    "c": "c1",
                    "from": 1123406040,
                    "to": 1234647162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:14",
                "fields": {
                    "s": "b",
                    "sf": 5.9,
                    "f": 5.9,
                    "d": 5.9,
                    "n": 5,
                    "a": "a2",
                    "b": "b2",
                    "c": "c2",
                    "from": 1123405940,
                    "to": 1234657162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heißz",
                },
            },
            {
                "put": "id:test:test:n=72331337:15",
                "fields": {
                    "s": "ba",
                    "sf": 6.9,
                    "f": 6.9,
                    "d": 6.9,
                    "n": 6,
                    "a": "a2",
                    "b": "b2",
                    "c": "c3",
                    "from": 1123404940,
                    "to": 1234667162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heißz",
                },
            },
            {
                "put": "id:test:test:n=72331337:16",
                "fields": {
                    "s": "baa",
                    "sf": 7.9,
                    "f": 7.9,
                    "d": 7.9,
                    "n": 7,
                    "a": "a2",
                    "b": "b3",
                    "c": "c1",
                    "from": 1123494940,
                    "to": 1234677162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:17",
                "fields": {
                    "s": "bab",
                    "sf": 8.9,
                    "f": 8.9,
                    "d": 8.9,
                    "n": 8,
                    "a": "a2",
                    "b": "b3",
                    "c": "c2",
                    "from": 1123394940,
                    "to": 1234687162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:18",
                "fields": {
                    "s": "bac",
                    "sf": 9.9,
                    "f": 9.9,
                    "d": 9.9,
                    "n": 9,
                    "a": "a2",
                    "b": "b3",
                    "c": "c3",
                    "from": 1122394940,
                    "to": 1234697162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:19",
                "fields": {
                    "s": "bba",
                    "sf": 1.9,
                    "f": 1.9,
                    "d": 1.9,
                    "n": 1,
                    "a": "a3",
                    "b": "b1",
                    "c": "c1",
                    "from": 1112394940,
                    "to": 1234707162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:20",
                "fields": {
                    "s": "bbb",
                    "sf": 2.9,
                    "f": 2.9,
                    "d": 2.9,
                    "n": 2,
                    "a": "a3",
                    "b": "b1",
                    "c": "c2",
                    "from": 1012394940,
                    "to": 1234717162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:21",
                "fields": {
                    "s": "bbc",
                    "sf": 3.9,
                    "f": 3.9,
                    "d": 3.9,
                    "n": 3,
                    "a": "a3",
                    "b": "b1",
                    "c": "c3",
                    "from": 1012394939,
                    "to": 1234727162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:22",
                "fields": {
                    "s": "bc",
                    "sf": 4.9,
                    "f": 4.9,
                    "d": 4.9,
                    "n": 4,
                    "a": "a3",
                    "b": "b2",
                    "c": "c1",
                    "from": 1012394929,
                    "to": 1234737162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:23",
                "fields": {
                    "s": "bca",
                    "sf": 5.9,
                    "f": 5.9,
                    "d": 5.9,
                    "n": 5,
                    "a": "a3",
                    "b": "b2",
                    "c": "c2",
                    "from": 1012394829,
                    "to": 1234747162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:24",
                "fields": {
                    "s": "bcb",
                    "sf": 6.9,
                    "f": 6.9,
                    "d": 6.9,
                    "n": 6,
                    "a": "a3",
                    "b": "b2",
                    "c": "c3",
                    "from": 1012393829,
                    "to": 1234757162000000000,
                    "boool": False,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:25",
                "fields": {
                    "s": "bcc",
                    "sf": 7.9,
                    "f": 7.9,
                    "d": 7.9,
                    "n": 7,
                    "a": "a3",
                    "b": "b3",
                    "c": "c1",
                    "from": 1012383829,
                    "to": 1234767162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:26",
                "fields": {
                    "s": "c",
                    "sf": 8.9,
                    "f": 8.9,
                    "d": 8.9,
                    "n": 8,
                    "a": "a3",
                    "b": "b3",
                    "c": "c2",
                    "from": 1012283829,
                    "to": 1234777162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heissö",
                },
            },
            {
                "put": "id:test:test:n=72331337:02",
                "fields": {
                    "s": "aa",
                    "sf": 2.9,
                    "f": 2.9,
                    "d": 2.9,
                    "n": 2,
                    "a": "a1",
                    "b": "b1",
                    "c": "c2",
                    "from": 1234517161,
                    "to": 1234527162000000000,
                    "boool": False,
                    "body": "test",
                    "nw": {"1": 1, "2": 1},
                    "lang": "heissz",
                },
            },
            {
                "put": "id:test:test:n=72331337:28",
                "fields": {
                    "s": "caa",
                    "sf": -2.9,
                    "f": -2.9,
                    "d": -2.9,
                    "n": -2,
                    "a": "a1",
                    "b": "b1",
                    "c": "c2",
                    "from": 1234517161,
                    "to": 1234527162000000000,
                    "boool": True,
                    "body": "test",
                    "nw": {"1": 1, "2": 1},
                    "lang": "heissz",
                },
            },
            {
                "put": "id:test:test:n=72331337:27",
                "fields": {
                    "s": "ca",
                    "sf": "no-number",
                    "f": 9.9,
                    "d": 9.9,
                    "n": 9,
                    "a": "a3",
                    "b": "b3",
                    "c": "c3",
                    "from": 1011283829,
                    "to": 1234787162000000000,
                    "boool": True,
                    "body": "test",
                    "lang": "heissö",
                },
            },
        ]
        docs = [
            {"fields": doc["fields"], "id": doc["put"].split("::")[-1]} for doc in docs
        ]
        return docs

    def feed_purchase_data(self) -> None:
        # Feed documents
        self.app.feed_iterable(iter=self.purchase_grouping_data, schema="purchase")
        return None

    def feed_test_grouping_data(self) -> None:
        # Feed documents
        self.app.feed_iterable(iter=self.test_grouping_data, schema="test")
        return None

    def test_grouping_with_condition(self):
        # "select * from purchase | all(group(customer) each(output(sum(price))))"
        # Feed test documents
        self.feed_purchase_data()
        # Execute query
        q = qb.test_grouping_with_condition()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        result_children = result.json["root"]["children"][0]["children"]
        # also get result from https://api.search.vespa.ai/search/?yql=select%20*%20from%20purchase%20where%20True%20%7C%20all(%20group(customer)%20each(output(sum(price)))%20)
        # to compare
        api_resp = requests.get(
            "https://api.search.vespa.ai/search/?yql=select%20*%20from%20purchase%20where%20True%20%7C%20all(%20group(customer)%20each(output(sum(price)))%20)",
        )
        api_resp = api_resp.json()
        api_children = api_resp["root"]["children"][0]["children"]
        self.assertEqual(result_children, api_children)
        # Verify the result
        group_results = result_children[0]["children"]
        self.assertEqual(group_results[0]["id"], "group:string:Brown")
        self.assertEqual(group_results[0]["value"], "Brown")
        self.assertEqual(group_results[0]["fields"]["sum(price)"], 20537)
        self.assertEqual(group_results[1]["id"], "group:string:Jones")
        self.assertEqual(group_results[1]["value"], "Jones")
        self.assertEqual(group_results[1]["fields"]["sum(price)"], 39816)
        self.assertEqual(group_results[2]["id"], "group:string:Smith")
        self.assertEqual(group_results[2]["value"], "Smith")
        self.assertEqual(group_results[2]["fields"]["sum(price)"], 19484)

    def test_grouping_with_ordering_and_limiting(self):
        # The two customers with most purchases, returning the sum for each:
        # "select * from purchase where True | all(group(customer) max(2) precision(12) order(-count()) each(output(sum(price))))"
        # Feed test documents
        self.feed_purchase_data()
        # Execute query
        q = qb.test_grouping_with_ordering_and_limiting()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        result_children = result.json["root"]["children"][0]["children"][0]["children"]
        print(result_children)
        # assert 2 groups
        self.assertEqual(len(result_children), 2)
        # assert the first group is Jones
        self.assertEqual(result_children[0]["id"], "group:string:Jones")
        self.assertEqual(result_children[0]["value"], "Jones")
        self.assertEqual(result_children[0]["fields"]["sum(price)"], 39816)
        # assert the second group is Brown
        self.assertEqual(result_children[1]["id"], "group:string:Smith")
        self.assertEqual(result_children[1]["value"], "Smith")
        self.assertEqual(result_children[1]["fields"]["sum(price)"], 19484)

    def test_grouping_hits_per_group(self):
        #  Example: Return the three most expensive parts per customer:
        # 'select * from purchase where True | all(group(customer) each(max(3) each(output(summary()))))&ranking=pricerank'
        self.feed_purchase_data()
        q = qb.test_grouping_hits_per_group()
        with self.app.syncio() as sess:
            result = sess.query(yql=q, ranking="pricerank")
        # Find the children of the child that has "id": "group:root: 0",
        for child in result.json["root"]["children"]:
            if child["id"] == "group:root:0":
                group_children = child["children"][0]["children"]
                break
        # Verify the result
        self.assertEqual(len(group_children), 3)
        # Expected:
        # ### Jones
        # | Date               | Price   | Tax   | Item           | Customer |
        # |--------------------|---------|-------|----------------|----------|
        # | 2006-09-11 12:00  | $9,870  | 0.12  | Exhaust port   | Jones    |
        # | 2006-09-10 10:00  | $8,900  | 0.24  | Camshaft       | Jones    |
        # | 2006-09-11 13:00  | $6,765  | 0.12  | Crankshaft     | Jones    |
        ### Brown
        # | Date               | Price   | Tax   | Item        | Customer |
        # |--------------------|---------|-------|-------------|----------|
        # | 2006-09-08 11:00  | $8,000  | 0.12  | Head        | Brown    |
        # | 2006-09-10 10:00  | $3,770  | 0.12  | Spring      | Brown    |
        # | 2006-09-09 11:00  | $3,400  | 0.24  | Oil pan     | Brown    |
        # ### Smith
        # | Date               | Price   | Tax   | Item             | Customer |
        # |--------------------|---------|-------|------------------|----------|
        # | 2006-09-10 11:00  | $6,100  | 0.24  | Spark plug       | Smith    |
        # | 2006-09-09 12:00  | $5,500  | 0.12  | Oil sump         | Smith    |
        # | 2006-09-11 11:00  | $2,584  | 0.12  | Connection rod   | Smith    |
        self.assertEqual(group_children[0]["value"], "Jones")
        self.assertEqual(group_children[0]["relevance"], 9870)
        self.assertEqual(group_children[1]["value"], "Brown")
        self.assertEqual(group_children[1]["relevance"], 8000)
        self.assertEqual(group_children[2]["value"], "Smith")
        self.assertEqual(group_children[2]["relevance"], 6100)

    def test_all(self):
        """This test runs all the test methods in the unit test class against a Docker Vespa instance that is fed with corresponding data.
        We do not inspect and compare all the results. This is already done in vespa system test, and as long as we are sure that the generated
        expressions are correct and does not give an error, we are satisfied.

        """
        self.feed_test_grouping_data()
        test_methods = [
            getattr(qb, method) for method in dir(qb) if method.startswith("test_")
        ]
        for method in test_methods:
            q = method()
            # purchase data is handled separately
            if "purchase" not in str(q):
                print(f"Executing query: {q}")
                try:
                    with self.app.syncio() as sess:
                        result = sess.query(yql=q)
                except Exception as e:
                    print(f"Failed query: {q}")
                    raise e
                    continue
                self.assertTrue(len(result.json["root"]["children"]) > 0, msg=q)
