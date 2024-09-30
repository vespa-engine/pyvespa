import unittest
from vespa.deployment import VespaDocker
from vespa.package import (
    ApplicationPackage,
    Schema,
    Document,
    Field,
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

        # Define all fields used in the unit tests
        fields = [
            Field(
                name="weightedset_field",
                type="weightedset<string>",
                indexing=["attribute"],
            ),
            Field(name="location_field", type="position", indexing=["attribute"]),
            Field(name="f1", type="string", indexing=["index", "summary"]),
            Field(name="f2", type="string", indexing=["index", "summary"]),
            Field(name="f3", type="string", indexing=["index", "summary"]),
            Field(name="f4", type="string", indexing=["index", "summary"]),
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
        ]
        document = Document(fields=fields, structs=[email_struct])
        schema = Schema(
            name=application_name, document=document, rank_profiles=rank_profiles
        )
        schema.add_fields(emails_field)
        application_package = ApplicationPackage(name=application_name, schema=[schema])
        print(application_package.schema.schema_to_text)
        # Deploy the application
        cls.vespa_docker = VespaDocker(port=8089)
        cls.app = cls.vespa_docker.deploy(application_package=application_package)
        cls.app.wait_for_application_up()

    @classmethod
    def tearDownClass(cls):
        cls.vespa_docker.container.stop(timeout=5)
        cls.vespa_docker.container.remove()

    # @unittest.skip("Skip until we have a better way to test this")
    def test_dotProduct_with_annotations(self):
        # Feed a document with 'weightedset_field'
        field = "weightedset_field"
        fields = {field: {"feature1": 2, "feature2": 4}}
        data_id = 1
        self.app.feed_data_point(
            schema=self.application_name, data_id=data_id, fields=fields
        )
        q = qb.test_dotProduct_with_annotations()
        with self.app.syncio() as sess:
            result = sess.query(yql=q, ranking="dotproduct")
        print(result.json)
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(
            result.hits[0]["id"],
            f"id:{self.application_name}:{self.application_name}::{data_id}",
        )
        self.assertEqual(
            result.hits[0]["fields"]["summaryfeatures"]["rawScore(weightedset_field)"],
            10,
        )

    def test_geoLocation_with_annotations(self):
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
            schema=self.application_name, data_id=data_id, fields=fields
        )
        # Build and send the query
        q = qb.test_geoLocation_with_annotations()
        with self.app.syncio() as sess:
            result = sess.query(yql=q, ranking="geolocation")
        # Check the result
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(
            result.hits[0]["id"],
            f"id:{self.application_name}:{self.application_name}::{data_id}",
        )
        self.assertAlmostEqual(
            result.hits[0]["fields"]["summaryfeatures"]["distance(location_field).km"],
            0.001417364012462494,
        )
        print(result.json)

    def test_basic_and_andnot_or_offset_limit_param_order_by_and_contains(self):
        docs = [
            {  # Should not match
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
            {  # Should not match
                "f1": "v1",
                "f2": "v2",
                "f3": "v3",
                "f4": "v4",
                "age": 30,
                "duration": 300,
            },
        ]
        id_to_match = 2
        docs = [
            {
                "fields": doc,
                "id": data_id,
            }
            for data_id, doc in enumerate(docs, 1)
        ]
        self.app.feed_iterable(iter=docs, schema=self.application_name)
        # Build and send the query
        q = qb.test_basic_and_andnot_or_offset_limit_param_order_by_and_contains()
        print(q)
        with self.app.syncio() as sess:
            result = sess.query(
                yql=q,
            )
        # Check the result
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(
            result.hits[0]["id"],
            f"id:{self.application_name}:{self.application_name}::{id_to_match}",
        )
        print(result.json)
