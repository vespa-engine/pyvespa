import unittest
from vespa.deployment import VespaDocker
from vespa.package import (
    ApplicationPackage,
    Schema,
    Document,
    Field,
    StructField,
    Struct,
)
from vespa.querybuilder import Query, Q


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
        document = Document(fields=fields, structs=[email_struct])
        schema = Schema(name=application_name, document=document)
        schema.add_fields(emails_field)
        application_package = ApplicationPackage(name=application_name, schema=[schema])
        print(application_package.schema.schema_to_text)
        # Deploy the application
        cls.vespa_docker = VespaDocker(port=8089)
        cls.app = cls.vespa_docker.deploy(application_package=application_package)

    @classmethod
    def tearDown(cls):
        cls.vespa_docker.container.stop(timeout=5)
        cls.vespa_docker.container.remove()

    def test_dotProduct_with_annotations(self):
        # Feed a document with 'weightedset_field'
        field = "weightedset_field"
        doc = {
            "id": f"id:{self.application_name}:{self.application_name}::1",
            "fields": {field: {"feature1": 0.5, "feature2": 1.0}},
        }
        self.app.feed_data_point(
            schema=self.application_name, data_id=doc["id"], fields=doc["fields"]
        )

        # Build and send the query
        condition = Q.dotProduct(
            field,
            {"feature1": 1, "feature2": 2},
            annotations={"label": "myDotProduct"},
        )
        q = (
            Query(select_fields=[field])
            .from_(self.application_name)
            .where(condition)
            .build(prepend_yql=False)
        )
        print(q)
        result = self.app.query(yql=q)

        # Check the result
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0]["id"], doc["id"])

    def test_geoLocation_with_annotations(self):
        # Feed a document with 'location_field'
        doc = {
            "id": f"id:{self.application_name}:{self.application_name}::2",
            "fields": {"location_field": "37.7749, -122.4194"},
        }
        self.app.feed_data_point(
            schema=self.application_name, data_id=doc["id"], fields=doc["fields"]
        )

        # Build and send the query
        condition = Q.geoLocation(
            "location_field",
            37.7749,
            -122.4194,
            "10km",
            annotations={"targetHits": 100},
        )
        q = (
            Query(select_fields="")
            .from_(self.application_name)
            .where(condition)
            .build(prepend_yql=False)
        )
        result = self.app.query(yql=q)

        # Check the result
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0]["id"], doc["id"])
