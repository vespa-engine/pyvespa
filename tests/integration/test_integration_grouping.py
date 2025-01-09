import unittest
import requests
from vespa.deployment import VespaDocker
from vespa.package import (
    ApplicationPackage,
    Schema,
    Document,
    Field,
    RankProfile,
)
from tests.unit.test_grouping import TestQueryBuilderGrouping

qb = TestQueryBuilderGrouping()


class TestGroupingIntegration(unittest.TestCase):
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
        # Create the application package
        application_package = ApplicationPackage(
            name=application_name, schema=[purchase_schema]
        )
        print(application_package.get_schema("purchase").schema_to_text)
        # Deploy the application
        cls.vespa_docker = VespaDocker(port=8089)
        cls.app = cls.vespa_docker.deploy(application_package=application_package)
        cls.app.wait_for_application_up()

    @classmethod
    def tearDownClass(cls):
        cls.vespa_docker.container.stop(timeout=5)
        cls.vespa_docker.container.remove()

    @property
    def sample_grouping_data(self):
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

    def feed_grouping_data(self) -> None:
        # Feed documents
        self.app.feed_iterable(iter=self.sample_grouping_data, schema="purchase")
        return None

    def test_grouping_with_condition(self):
        # "select * from purchase | all(group(customer) each(output(sum(price))))"
        # Feed test documents
        self.feed_grouping_data()
        # Execute query
        q = qb.test_grouping_with_condition()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        result_children = result.json["root"]["children"][0]["children"]
        # also get result from https://api.search.vespa.ai/search/?yql=select%20*%20from%20purchase%20where%20true%20%7C%20all(%20group(customer)%20each(output(sum(price)))%20)
        # to compare
        api_resp = requests.get(
            "https://api.search.vespa.ai/search/?yql=select%20*%20from%20purchase%20where%20true%20%7C%20all(%20group(customer)%20each(output(sum(price)))%20)",
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
        # "select * from purchase where true | all(group(customer) max(2) precision(12) order(-count()) each(output(sum(price))))"
        # Feed test documents
        self.feed_grouping_data()
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
        # 'select * from purchase where true | all(group(customer) each(max(3) each(output(summary()))))&ranking=pricerank'
        self.feed_grouping_data()
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
