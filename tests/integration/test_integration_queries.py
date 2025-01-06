import unittest
import requests
from vespa.deployment import VespaDocker
from vespa.package import (
    ApplicationPackage,
    Schema,
    Document,
    Field,
    FieldSet,
    StructField,
    Struct,
    RankProfile,
)
from tests.unit.test_q import TestQueryBuilder

qb = TestQueryBuilder()


class TestQueriesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        application_name = "querybuilder"
        cls.application_name = application_name
        schema_name = "sd1"
        cls.schema_name = schema_name
        # Define all fields used in the unit tests
        # Schema 1
        fields = [
            Field(
                name="weightedset_field",
                type="weightedset<string>",
                indexing=["attribute"],
            ),
            Field(name="location_field", type="position", indexing=["attribute"]),
            Field(name="f1", type="string", indexing=["attribute", "summary"]),
            Field(name="f2", type="string", indexing=["attribute", "summary"]),
            Field(name="f3", type="string", indexing=["attribute", "summary"]),
            Field(name="f4", type="string", indexing=["attribute", "summary"]),
            Field(name="age", type="int", indexing=["attribute", "summary"]),
            Field(name="duration", type="int", indexing=["attribute", "summary"]),
            Field(name="id", type="string", indexing=["attribute", "summary"]),
            Field(name="text", type="string", indexing=["index", "summary"]),
            Field(name="title", type="string", indexing=["index", "summary"]),
            Field(name="description", type="string", indexing=["index", "summary"]),
            Field(name="date", type="string", indexing=["attribute", "summary"]),
            Field(name="status", type="string", indexing=["attribute", "summary"]),
            Field(name="comments", type="string", indexing=["attribute", "summary"]),
            Field(
                name="embedding",
                type="tensor<float>(x[128])",
                indexing=["attribute"],
            ),
            Field(name="tags", type="array<string>", indexing=["attribute", "summary"]),
            Field(
                name="timestamp",
                type="long",
                indexing=["attribute", "summary"],
            ),
            Field(name="integer_field", type="int", indexing=["attribute", "summary"]),
            Field(
                name="predicate_field",
                type="predicate",
                indexing=["attribute", "summary"],
                index="arity: 2",  # This is required for predicate fields
            ),
            Field(
                name="myStringAttribute", type="string", indexing=["index", "summary"]
            ),
            Field(name="myUrlField", type="string", indexing=["index", "summary"]),
            Field(name="fieldName", type="string", indexing=["index", "summary"]),
            Field(
                name="dense_rep",
                type="tensor<float>(x[128])",
                indexing=["attribute"],
            ),
            Field(name="artist", type="string", indexing=["attribute", "summary"]),
            Field(name="subject", type="string", indexing=["attribute", "summary"]),
            Field(
                name="display_date", type="string", indexing=["attribute", "summary"]
            ),
            Field(name="price", type="double", indexing=["attribute", "summary"]),
            Field(name="keywords", type="string", indexing=["index", "summary"]),
        ]
        email_struct = Struct(
            name="email",
            fields=[
                Field(name="sender", type="string"),
                Field(name="recipient", type="string"),
                Field(name="subject", type="string"),
                Field(name="content", type="string"),
            ],
        )
        emails_field = Field(
            name="emails",
            type="array<email>",
            indexing=["summary"],
            struct_fields=[
                StructField(
                    name="content", indexing=["attribute"], attribute=["fast-search"]
                )
            ],
        )
        person_struct = Struct(
            name="person",
            fields=[
                Field(name="first_name", type="string"),
                Field(name="last_name", type="string"),
                Field(name="year_of_birth", type="int"),
            ],
        )
        persons_field = Field(
            name="persons",
            type="array<person>",
            indexing=["summary"],
            struct_fields=[
                StructField(name="first_name", indexing=["attribute"]),
                StructField(name="last_name", indexing=["attribute"]),
                StructField(name="year_of_birth", indexing=["attribute"]),
            ],
        )
        rank_profiles = [
            RankProfile(
                name="dotproduct",
                first_phase="rawScore(weightedset_field)",
                summary_features=["rawScore(weightedset_field)"],
            ),
            RankProfile(
                name="geolocation",
                first_phase="distance(location_field)",
                summary_features=["distance(location_field).km"],
            ),
            RankProfile(
                name="bm25", first_phase="bm25(text)", summary_features=["bm25(text)"]
            ),
        ]
        fieldset = FieldSet(name="default", fields=["text", "title", "description"])
        document = Document(fields=fields, structs=[email_struct, person_struct])
        schema = Schema(
            name=schema_name,
            document=document,
            rank_profiles=rank_profiles,
            fieldsets=[fieldset],
        )
        schema.add_fields(emails_field, persons_field)
        # Add purchase schema for grouping test
        purchase_schema = Schema(
            name="purchase",
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
        )
        # Create the application package
        application_package = ApplicationPackage(
            name=application_name, schema=[schema, purchase_schema]
        )
        print(application_package.get_schema(schema_name).schema_to_text)
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

    def test_dotproduct_with_annotations(self):
        # Feed a document with 'weightedset_field'
        field = "weightedset_field"
        fields = {field: {"feature1": 2, "feature2": 4}}
        data_id = 1
        self.app.feed_data_point(
            schema=self.schema_name, data_id=data_id, fields=fields
        )
        q = qb.test_dotproduct_with_annotations()
        with self.app.syncio() as sess:
            result = sess.query(yql=q, ranking="dotproduct")
        print(result.json)
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(
            result.hits[0]["id"],
            f"id:{self.schema_name}:{self.schema_name}::{data_id}",
        )
        self.assertEqual(
            result.hits[0]["fields"]["summaryfeatures"]["rawScore(weightedset_field)"],
            10,
        )

    def test_geolocation_with_annotations(self):
        # Feed a document with 'location_field'
        field_name = "location_field"
        fields = {
            field_name: {
                "lat": 37.77491,
                "lng": -122.41941,
            },  # 0.00001 degrees more than the query
        }
        data_id = 2
        self.app.feed_data_point(
            schema=self.schema_name, data_id=data_id, fields=fields
        )
        # Build and send the query
        q = qb.test_geolocation_with_annotations()
        with self.app.syncio() as sess:
            result = sess.query(yql=q, ranking="geolocation")
        # Check the result
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(
            result.hits[0]["id"],
            f"id:{self.schema_name}:{self.schema_name}::{data_id}",
        )
        self.assertAlmostEqual(
            result.hits[0]["fields"]["summaryfeatures"]["distance(location_field).km"],
            0.001417364012462494,
        )
        print(result.json)

    def test_basic_and_andnot_or_offset_limit_param_order_by_and_contains(self):
        docs = [
            {  # Should not match - f3 doesn't contain "v3"
                "f1": "v1",
                "f2": "v2",
                "f3": "asdf",
                "f4": "d",
                "age": 10,
                "duration": 100,
            },
            {  # Should match
                "f1": "v1",
                "f2": "v2",
                "f3": "v3",
                "f4": "d",
                "age": 20,
                "duration": 200,
            },
            {  # Should match
                "f1": "v1",
                "f2": "v2",
                "f3": "v3",
                "f4": "d",
                "age": 30,
                "duration": 300,
            },
            {  # Should not match - contains f4="v4"
                "f1": "v1",
                "f2": "v2",
                "f3": "v3",
                "f4": "v4",
                "age": 40,
                "duration": 400,
            },
        ]

        # Feed documents
        docs = [{"id": data_id, "fields": doc} for data_id, doc in enumerate(docs, 1)]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)

        # Build and send query
        q = qb.test_basic_and_andnot_or_offset_limit_param_order_by_and_contains()
        print(f"Executing query: {q}")

        with self.app.syncio() as sess:
            result = sess.query(yql=q)

        # Verify results
        self.assertEqual(
            len(result.hits), 1
        )  # Should get 1 hit due to offset=1, limit=2

        # The query orders by age desc, duration asc with offset 1
        # So we should get doc ID 2 (since doc ID 3 is skipped due to offset)
        hit = result.hits[0]
        self.assertEqual(hit["id"], f"id:{self.schema_name}:{self.schema_name}::2")

        # Verify the matching document has expected field values
        self.assertEqual(hit["fields"]["age"], 20)
        self.assertEqual(hit["fields"]["duration"], 200)
        self.assertEqual(hit["fields"]["f1"], "v1")
        self.assertEqual(hit["fields"]["f2"], "v2")
        self.assertEqual(hit["fields"]["f3"], "v3")
        self.assertEqual(hit["fields"]["f4"], "d")

        print(result.json)

    def test_matches(self):
        # Matches is a regex (or substring) match
        # Feed test documents
        docs = [
            {  # Doc 1: Should match - satisfies (f1="v1" AND f2="v2") and f4!="v4"
                "f1": "v1",
                "f2": "v2",
                "f3": "other",
                "f4": "nothing",
            },
            {  # Doc 2: Should not match - fails f4!="v4" condition
                "f1": "v1",
                "f2": "v2",
                "f3": "v3",
                "f4": "v4",
            },
            {  # Doc 3: Should match - satisfies f3="v3" and f4!="v4"
                "f1": "other",
                "f2": "other",
                "f3": "v3",
                "f4": "nothing",
            },
            {  # Doc 4: Should not match - fails all conditions
                "f1": "other",
                "f2": "other",
                "f3": "other",
                "f4": "v4",
            },
        ]

        # Ensure fields are properly indexed for matching
        docs = [
            {
                "fields": doc,
                "id": str(data_id),
            }
            for data_id, doc in enumerate(docs, 1)
        ]

        # Feed documents
        self.app.feed_iterable(iter=docs, schema=self.schema_name)

        # Build and send query
        q = qb.test_matches()
        # select * from sd1 where ((f1 matches "v1" and f2 matches "v2") or f3 matches "v3") and !(f4 matches "v4")
        print(f"Executing query: {q}")

        with self.app.syncio() as sess:
            result = sess.query(yql=q)

        # Check result count
        self.assertEqual(len(result.hits), 2)

        # Verify specific matches
        ids = sorted([hit["id"] for hit in result.hits])
        expected_ids = sorted(
            [
                f"id:{self.schema_name}:{self.schema_name}::1",
                f"id:{self.schema_name}:{self.schema_name}::3",
            ]
        )

        self.assertEqual(ids, expected_ids)
        print(result.json)

    def test_nested_queries(self):
        # Contains is an exact match
        # q = 'select * from sd1 where f1 contains "1" and (!((f2 contains "2" and f3 contains "3") or (f2 contains "4" and !(f3 contains "5"))))'
        # Feed test documents
        docs = [
            {  # Doc 1: Should not match - satisfies f1 contains "1" but fails inner query
                "f1": "1",
                "f2": "2",
                "f3": "3",
            },
            {  # Doc 2: Should match
                "f1": "1",
                "f2": "4",
                "f3": "5",
            },
            {  # Doc 3: Should not match - fails f1 contains "1"
                "f1": "other",
                "f2": "2",
                "f3": "3",
            },
            {  # Doc 4: Should not match
                "f1": "1",
                "f2": "4",
                "f3": "other",
            },
        ]
        docs = [
            {
                "fields": doc,
                "id": str(data_id),
            }
            for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        q = qb.test_nested_queries()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        print(result.json)
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(
            result.hits[0]["id"],
            f"id:{self.schema_name}:{self.schema_name}::2",
        )

    def test_userquery_defaultindex(self):
        # 'select * from sd1 where ({"defaultIndex":"text"}userQuery())'
        # Feed test documents
        docs = [
            {  # Doc 1: Should match
                "description": "foo",
                "text": "foo",
            },
            {  # Doc 2: Should match
                "description": "foo",
                "text": "bar",
            },
            {  # Doc 3: Should not match
                "description": "bar",
                "text": "baz",
            },
        ]

        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)

        # Execute query
        q = qb.test_userquery()
        query = "foo"
        print(f"Executing query: {q}")
        body = {
            "yql": str(q),
            "query": query,
        }
        with self.app.syncio() as sess:
            result = sess.query(body=body)
        self.assertEqual(len(result.hits), 2)
        ids = sorted([hit["id"] for hit in result.hits])
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::1", ids)
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::2", ids)

    def test_userquery_customindex(self):
        # 'select * from sd1 where userQuery())'
        # Feed test documents
        docs = [
            {  # Doc 1: Should match
                "description": "foo",
                "text": "foo",
            },
            {  # Doc 2: Should not match
                "description": "foo",
                "text": "bar",
            },
            {  # Doc 3: Should not match
                "description": "bar",
                "text": "baz",
            },
        ]

        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)

        # Execute query
        q = qb.test_userquery()
        query = "foo"
        print(f"Executing query: {q}")
        body = {
            "yql": str(q),
            "query": query,
            "ranking": "bm25",
            "model.defaultIndex": "text",  # userQuery() needs this to set index, see https://docs.vespa.ai/en/query-api.html#using-a-fieldset
        }
        with self.app.syncio() as sess:
            result = sess.query(body=body)
        # Verify only one document matches both conditions
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(
            result.hits[0]["id"], f"id:{self.schema_name}:{self.schema_name}::1"
        )

        # Verify matching document has expected values
        hit = result.hits[0]
        self.assertEqual(hit["fields"]["description"], "foo")
        self.assertEqual(hit["fields"]["text"], "foo")

    def test_userinput(self):
        # 'select * from sd1 where userInput(@myvar)'
        # Feed test documents
        myvar = "panda"
        docs = [
            {  # Doc 1: Should match
                "description": "a panda is a cute",
                "text": "foo",
            },
            {  # Doc 2: Should match
                "description": "foo",
                "text": "you are a cool panda",
            },
            {  # Doc 3: Should not match
                "description": "bar",
                "text": "baz",
            },
        ]
        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        # Execute query
        q = qb.test_userinput()
        print(f"Executing query: {q}")
        body = {
            "yql": str(q),
            "ranking": "bm25",
            "myvar": myvar,
        }
        with self.app.syncio() as sess:
            result = sess.query(body=body)
        # Verify only two documents match
        self.assertEqual(len(result.hits), 2)
        # Verify matching documents have expected values
        ids = sorted([hit["id"] for hit in result.hits])
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::1", ids)
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::2", ids)

    def test_userinput_with_defaultindex(self):
        # 'select * from sd1 where {defaultIndex:"text"}userInput(@myvar)'
        # Feed test documents
        myvar = "panda"
        docs = [
            {  # Doc 1: Should not match
                "description": "a panda is a cute",
                "text": "foo",
            },
            {  # Doc 2: Should match
                "description": "foo",
                "text": "you are a cool panda",
            },
            {  # Doc 3: Should not match
                "description": "bar",
                "text": "baz",
            },
        ]
        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        # Execute query
        q = qb.test_userinput_with_defaultindex()
        print(f"Executing query: {q}")
        body = {
            "yql": str(q),
            "ranking": "bm25",
            "myvar": myvar,
        }
        with self.app.syncio() as sess:
            result = sess.query(body=body)
        print(result.json)
        # Verify only one document matches
        self.assertEqual(len(result.hits), 1)
        # Verify matching document has expected values
        hit = result.hits[0]
        self.assertEqual(hit["id"], f"id:{self.schema_name}:{self.schema_name}::2")

    def test_in_operator_intfield(self):
        # 'select * from * where integer_field in (10, 20, 30)'
        # We use age field for this test
        # Feed test documents
        docs = [
            {  # Doc 1: Should match
                "age": 10,
            },
            {  # Doc 2: Should match
                "age": 20,
            },
            {  # Doc 3: Should not match
                "age": 31,
            },
            {  # Doc 4: Should not match
                "age": 40,
            },
        ]
        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        # Execute query
        q = qb.test_in_operator_intfield()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        print(result.json)
        # Verify only two documents match
        self.assertEqual(len(result.hits), 2)
        # Verify matching documents have expected values
        ids = sorted([hit["id"] for hit in result.hits])
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::1", ids)
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::2", ids)

    def test_in_operator_stringfield(self):
        # 'select * from sd1 where status in ("active", "inactive")'
        # Feed test documents
        docs = [
            {  # Doc 1: Should match
                "status": "active",
            },
            {  # Doc 2: Should match
                "status": "inactive",
            },
            {  # Doc 3: Should not match
                "status": "foo",
            },
            {  # Doc 4: Should not match
                "status": "bar",
            },
        ]
        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        # Execute query
        q = qb.test_in_operator_stringfield()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        # Verify only two documents match
        self.assertEqual(len(result.hits), 2)
        # Verify matching documents have expected values
        ids = sorted([hit["id"] for hit in result.hits])
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::1", ids)
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::2", ids)

    def test_near(self):
        # 'select * from * where title contains near("madonna", "saint")'
        # Feed test documents
        docs = [
            {  # Doc 1: Should match
                "title": "madonna the saint",
            },
            {  # Doc 2: Should not match
                "title": "saint and sinner",
            },
            {  # Doc 3: Should not match (exceed default distance of 2)
                "title": "madonna has become a saint",
            },
        ]
        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        # Execute query
        q = qb.test_near()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        # Verify only one document matches
        self.assertEqual(len(result.hits), 1)
        # Verify matching id
        hit = result.hits[0]
        self.assertEqual(hit["id"], f"id:{self.schema_name}:{self.schema_name}::1")

    def test_predicate(self):
        #  'select * from sd1 where predicate(predicate_field,{"gender":"Female"},{"age":25L})'
        # Feed test documents with predicate_field
        docs = [
            {  # Doc 1: Should match - satisfies both predicates
                "predicate_field": 'gender in ["Female"] and age in [20..30]',
            },
            {  # Doc 2: Should not match - wrong gender
                "predicate_field": 'gender in ["Male"] and age in [20..30]',
            },
            {  # Doc 3: Should not match - too young
                "predicate_field": 'gender in ["Female"] and age in [30..40]',
            },
        ]

        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)

        # Execute query using predicate search
        q = qb.test_predicate()
        print(f"Executing query: {q}")

        with self.app.syncio() as sess:
            result = sess.query(yql=q)

        # Verify only one document matches both predicates
        self.assertEqual(len(result.hits), 1)

        # Verify matching document has expected id
        hit = result.hits[0]
        self.assertEqual(hit["id"], f"id:{self.schema_name}:{self.schema_name}::1")

    def test_fuzzy(self):
        # 'select * from sd1 where f1 contains ({prefixLength:1,maxEditDistance:2}fuzzy("parantesis"))'
        # Feed test documents
        docs = [
            {  # Doc 1: Should match
                "f1": "parantesis",
            },
            {  # Doc 2: Should match - edit distance 1
                "f1": "paranthesis",
            },
            {  # Doc 3: Should not match - edit distance 3
                "f1": "parrenthesis",
            },
        ]
        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        # Execute query
        q = qb.test_fuzzy()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        # Verify only two documents match
        self.assertEqual(len(result.hits), 2)
        # Verify matching documents have expected values
        ids = sorted([hit["id"] for hit in result.hits])
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::1", ids)
        self.assertIn(f"id:{self.schema_name}:{self.schema_name}::2", ids)

    def test_uri(self):
        # 'select * from sd1 where myUrlField contains uri("vespa.ai/foo")'
        # Feed test documents
        docs = [
            {  # Doc 1: Should match
                "myUrlField": "https://vespa.ai/foo",
            },
            {  # Doc 2: Should not match - wrong path
                "myUrlField": "https://vespa.ai/bar",
            },
            {  # Doc 3: Should not match - wrong domain
                "myUrlField": "https://google.com/foo",
            },
        ]
        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        # Execute query
        q = qb.test_uri()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        # Verify only one document matches
        self.assertEqual(len(result.hits), 1)
        # Verify matching document has expected values
        hit = result.hits[0]
        self.assertEqual(hit["id"], f"id:{self.schema_name}:{self.schema_name}::1")

    def test_same_element(self):
        # 'select * from sd1 where persons contains sameElement(first_name contains "Joe", last_name contains "Smith", year_of_birth < 1940)'
        # Feed test documents
        docs = [
            {  # Doc 1: Should match
                "persons": [
                    {"first_name": "Joe", "last_name": "Smith", "year_of_birth": 1930}
                ],
            },
            {  # Doc 2: Should not match - wrong last name
                "persons": [
                    {"first_name": "Joe", "last_name": "Johnson", "year_of_birth": 1930}
                ],
            },
            {  # Doc 3: Should not match - wrong year of birth
                "persons": [
                    {"first_name": "Joe", "last_name": "Smith", "year_of_birth": 1940}
                ],
            },
        ]
        # Format and feed documents
        docs = [
            {"fields": doc, "id": str(data_id)} for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.schema_name)
        # Execute query
        q = qb.test_same_element()
        print(f"Executing query: {q}")
        with self.app.syncio() as sess:
            result = sess.query(yql=q)
        # Verify only one document matches
        self.assertEqual(len(result.hits), 1)
        # Verify matching document has expected values
        hit = result.hits[0]
        self.assertEqual(hit["id"], f"id:{self.schema_name}:{self.schema_name}::1")

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
