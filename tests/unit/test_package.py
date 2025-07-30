# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
import platform
import pytest

from vespa.package import (
    HNSW,
    Field,
    ImportedField,
    Document,
    FieldSet,
    Function,
    SecondPhaseRanking,
    GlobalPhaseRanking,
    MatchPhaseRanking,
    Mutate,
    RankProfile,
    OnnxModel,
    Schema,
    QueryTypeField,
    QueryProfileType,
    QueryField,
    QueryProfile,
    Component,
    Nodes,
    ContentCluster,
    ContainerCluster,
    Parameter,
    ApplicationPackage,
    AuthClient,
    DeploymentConfiguration,
    Struct,
    StructField,
    ServicesConfiguration,
    ApplicationConfiguration,
)
from vespa.configuration.vt import compare_xml
from vespa.configuration.services import *


class TestField(unittest.TestCase):
    def test_field_name_type(self):
        field = Field(name="test_name", type="string")
        self.assertEqual(field.name, "test_name")
        self.assertEqual(field.type, "string")
        self.assertEqual(field, Field(name="test_name", type="string"))
        self.assertIsNone(field.indexing_to_text)

    def test_field_name_type_indexing_index(self):
        field = Field(
            name="body",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        )
        self.assertEqual(field.name, "body")
        self.assertEqual(field.type, "string")
        self.assertEqual(field.indexing, ["index", "summary"])
        self.assertEqual(field.index, "enable-bm25")
        self.assertEqual(
            field,
            Field(
                name="body",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
        )
        self.assertEqual(field.indexing_to_text, "index | summary")


class TestImportField(unittest.TestCase):
    def test_import_field(self):
        imported_field = ImportedField(
            name="global_category_ctrs",
            reference_field="category_ctr_ref",
            field_to_import="ctrs",
        )
        self.assertEqual(imported_field.name, "global_category_ctrs")
        self.assertEqual(imported_field.reference_field, "category_ctr_ref")
        self.assertEqual(imported_field.field_to_import, "ctrs")


class TestDocument(unittest.TestCase):
    def test_empty_document(self):
        document = Document()
        self.assertEqual(document.fields, [])

    def test_document_one_field(self):
        document = Document(inherits="context")
        field = Field(name="test_name", type="string")
        document.add_fields(field)
        self.assertEqual(document.fields, [field])
        self.assertEqual(document, Document([field], "context"))

    def test_document_two_fields(self):
        document = Document()
        field_1 = Field(name="test_name", type="string")
        field_2 = Field(
            name="body",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        )
        document.add_fields(field_1, field_2)
        self.assertEqual(document.fields, [field_1, field_2])
        self.assertEqual(document, Document([field_1, field_2]))

    def test_update_field(self):
        document = Document()
        field_1 = Field(name="test_name", type="string")
        document.add_fields(field_1)
        self.assertEqual(document.fields, [field_1])
        field_1_updated = Field(
            name="test_name", type="string", indexing=["index", "summary"]
        )
        document.add_fields(field_1_updated)
        self.assertEqual(document.fields, [field_1_updated])


class TestFieldSet(unittest.TestCase):
    def test_fieldset(self):
        field_set = FieldSet(name="default", fields=["title", "body"])
        self.assertEqual(field_set.name, "default")
        self.assertEqual(field_set.fields, ["title", "body"])
        self.assertEqual(field_set.fields_to_text, "title, body")


class TestFunction(unittest.TestCase):
    def test_function_no_argument(self):
        function = Function(
            name="myfeature", expression="fieldMatch(title) + freshness(timestamp)"
        )
        self.assertEqual(function.name, "myfeature")
        self.assertEqual(
            function.expression, "fieldMatch(title) + freshness(timestamp)"
        )
        self.assertEqual(function.args_to_text, "")

    def test_function_one_argument(self):
        function = Function(
            name="myfeature",
            expression="fieldMatch(title) + freshness(foo)",
            args=["foo"],
        )
        self.assertEqual(function.name, "myfeature")
        self.assertEqual(function.expression, "fieldMatch(title) + freshness(foo)")
        self.assertEqual(function.args, ["foo"])
        self.assertEqual(function.args_to_text, "foo")

    def test_function_multiple_argument(self):
        function = Function(
            name="myfeature",
            expression="fieldMatch(bar) + freshness(foo)",
            args=["foo", "bar"],
        )
        self.assertEqual(function.name, "myfeature")
        self.assertEqual(function.expression, "fieldMatch(bar) + freshness(foo)")
        self.assertEqual(function.args, ["foo", "bar"])
        self.assertEqual(function.args_to_text, "foo, bar")

    def test_function_multiple_lines(self):
        function = Function(
            name="token_type_ids",
            expression="""
            tensor<float>(d0[1],d1[128])(
               if (d1 < question_length,
                 0,
               if (d1 < question_length + doc_length,
                 1,
                 TOKEN_NONE
               )))
            """,
        )
        self.assertEqual(function.name, "token_type_ids")
        self.assertEqual(
            function.expression,
            """
            tensor<float>(d0[1],d1[128])(
               if (d1 < question_length,
                 0,
               if (d1 < question_length + doc_length,
                 1,
                 TOKEN_NONE
               )))
            """,
        )
        self.assertEqual(function.args, None)
        self.assertEqual(function.args_to_text, "")


class TestRankProfile(unittest.TestCase):
    def test_rank_profile(self):
        rank_profile = RankProfile(name="bm25", first_phase="bm25(title) + bm25(body)")
        self.assertEqual(rank_profile.name, "bm25")
        self.assertEqual(rank_profile.first_phase, "bm25(title) + bm25(body)")

    def test_rank_profile_inherits(self):
        rank_profile = RankProfile(
            name="bm25", first_phase="bm25(title) + bm25(body)", inherits="default"
        )
        self.assertEqual(rank_profile.name, "bm25")
        self.assertEqual(rank_profile.first_phase, "bm25(title) + bm25(body)")

    def test_rank_profile_bert_second_phase(self):
        rank_profile = RankProfile(
            name="bert",
            first_phase="bm25(title) + bm25(body)",
            second_phase=SecondPhaseRanking(
                rerank_count=10, expression="sum(onnx(bert_tiny).logits{d0:0,d1:0})"
            ),
            inherits="default",
            constants={"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102},
            functions=[
                Function(
                    name="question_length",
                    expression="sum(map(query(query_token_ids), f(a)(a > 0)))",
                ),
                Function(
                    name="doc_length",
                    expression="sum(map(attribute(doc_token_ids), f(a)(a > 0)))",
                ),
                Function(
                    name="input_ids",
                    expression="tensor<float>(d0[1],d1[128])(\n"
                    "    if (d1 == 0,\n"
                    "        TOKEN_CLS,\n"
                    "    if (d1 < question_length + 1,\n"
                    "        query(query_token_ids){d0:(d1-1)},\n"
                    "    if (d1 == question_length + 1,\n"
                    "        TOKEN_SEP,\n"
                    "    if (d1 < question_length + doc_length + 2,\n"
                    "        attribute(doc_token_ids){d0:(d1-question_length-2)},\n"
                    "    if (d1 == question_length + doc_length + 2,\n"
                    "        TOKEN_SEP,\n"
                    "        TOKEN_NONE\n"
                    "    ))))))",
                ),
                Function(
                    name="attention_mask",
                    expression="map(input_ids, f(a)(a > 0))",
                ),
                Function(
                    name="token_type_ids",
                    expression="tensor<float>(d0[1],d1[128])(\n"
                    "    if (d1 < question_length,\n"
                    "        0,\n"
                    "    if (d1 < question_length + doc_length,\n"
                    "        1,\n"
                    "        TOKEN_NONE\n"
                    "    )))",
                ),
            ],
            summary_features=[
                "onnx(bert).logits",
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ],
        )
        self.assertEqual(rank_profile.name, "bert")
        self.assertEqual(rank_profile.first_phase, "bm25(title) + bm25(body)")
        self.assertDictEqual(
            rank_profile.constants,
            {"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102},
        )
        self.assertEqual(
            rank_profile.summary_features,
            ["onnx(bert).logits", "input_ids", "attention_mask", "token_type_ids"],
        )

    def test_rank_profile_inputs(self):
        rank_profile = RankProfile(
            name="bm25",
            first_phase="bm25(title) + bm25(body)",
            inputs=[("query(image_query_embedding)", "tensor<float>(d0[512])")],
        )
        self.assertEqual(rank_profile.inputs[0][0], "query(image_query_embedding)")
        self.assertEqual(rank_profile.inputs[0][1], "tensor<float>(d0[512])")

        rank_profile = RankProfile(
            name="bm25",
            first_phase="bm25(title) + bm25(body)",
            inputs=[("query(image_query_embedding)", "tensor<float>(d0[512])", "0")],
        )
        self.assertEqual(rank_profile.inputs[0][0], "query(image_query_embedding)")
        self.assertEqual(rank_profile.inputs[0][1], "tensor<float>(d0[512])")
        self.assertEqual(rank_profile.inputs[0][2], "0")

        rank_profile = RankProfile(
            name="bm25",
            first_phase="bm25(title) + bm25(body)",
            inputs=[
                ("query(image_query_embedding)", "tensor<float>(d0[512])"),
                ("query(image_query_embedding2)", "tensor<float>(d1[512])"),
            ],
        )
        self.assertEqual(rank_profile.inputs[0][0], "query(image_query_embedding)")
        self.assertEqual(rank_profile.inputs[1][0], "query(image_query_embedding2)")

    def test_rank_profile_mutate_definition(self):
        mutate: Mutate = Mutate(
            on_match={
                "attribute": "my_mutable_attribute",
                "operation_string": "+=",
                "operation_value": 5,
            },
            on_first_phase=None,
            on_second_phase={
                "attribute": "my_mutable_attribute",
                "operation_string": "-=",
                "operation_value": 3,
            },
            on_summary={
                "attribute": "my_mutable_attribute",
                "operation_string": "=",
                "operation_value": 42,
            },
        )
        rank_profile = RankProfile(
            name="track_some_attribute", first_phase="bm25(title)", mutate=mutate
        )
        self.assertTrue(rank_profile.mutate.on_match)
        self.assertFalse(rank_profile.mutate.on_first_phase)
        self.assertTrue(rank_profile.mutate.on_second_phase)
        self.assertTrue(rank_profile.mutate.on_summary)

        self.assertEqual(rank_profile.mutate.on_match_attribute, "my_mutable_attribute")
        self.assertEqual(
            rank_profile.mutate.on_second_phase_attribute, "my_mutable_attribute"
        )
        self.assertEqual(
            rank_profile.mutate.on_summary_attribute, "my_mutable_attribute"
        )

        self.assertEqual(rank_profile.mutate.on_match_operation_string, "+=")
        self.assertEqual(rank_profile.mutate.on_second_phase_operation_string, "-=")
        self.assertEqual(rank_profile.mutate.on_summary_operation_string, "=")

        self.assertEqual(rank_profile.mutate.on_match_operation_value, 5)
        self.assertEqual(rank_profile.mutate.on_second_phase_operation_value, 3)
        self.assertEqual(rank_profile.mutate.on_summary_operation_value, 42)


class TestSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = Schema(
            name="test_schema",
            document=Document(fields=[Field(name="test_name", type="string")]),
            fieldsets=[FieldSet(name="default", fields=["title", "body"])],
            rank_profiles=[
                RankProfile(name="bm25", first_phase="bm25(title) + bm25(body)")
            ],
            models=[
                OnnxModel(
                    model_name="bert",
                    model_file_path="bert.onnx",
                    inputs={
                        "input_ids": "input_ids",
                        "token_type_ids": "token_type_ids",
                        "attention_mask": "attention_mask",
                    },
                    outputs={"logits": "logits"},
                )
            ],
            global_document=True,
            imported_fields=[
                ImportedField(
                    name="global_category_ctrs",
                    reference_field="category_ctr_ref",
                    field_to_import="ctrs",
                )
            ],
        )

    def test_rank_profile(self):
        self.assertDictEqual(
            self.schema.rank_profiles,
            {"bm25": RankProfile(name="bm25", first_phase="bm25(title) + bm25(body)")},
        )
        self.schema.add_rank_profile(
            RankProfile(name="default", first_phase="NativeRank(title)")
        )
        self.assertDictEqual(
            self.schema.rank_profiles,
            {
                "bm25": RankProfile(
                    name="bm25", first_phase="bm25(title) + bm25(body)"
                ),
                "default": RankProfile(name="default", first_phase="NativeRank(title)"),
            },
        )

    def test_imported_field(self):
        self.assertDictEqual(
            self.schema.imported_fields,
            {
                "global_category_ctrs": ImportedField(
                    name="global_category_ctrs",
                    reference_field="category_ctr_ref",
                    field_to_import="ctrs",
                )
            },
        )
        self.schema.add_imported_field(
            ImportedField(
                name="new_imported_field",
                reference_field="category_ctr_ref",
                field_to_import="ctrs",
            )
        )
        self.assertDictEqual(
            self.schema.imported_fields,
            {
                "global_category_ctrs": ImportedField(
                    name="global_category_ctrs",
                    reference_field="category_ctr_ref",
                    field_to_import="ctrs",
                ),
                "new_imported_field": ImportedField(
                    name="new_imported_field",
                    reference_field="category_ctr_ref",
                    field_to_import="ctrs",
                ),
            },
        )


class TestQueryTypeField(unittest.TestCase):
    def test_field_name_type(self):
        field = QueryTypeField(name="test_name", type="string")
        self.assertEqual(field.name, "test_name")
        self.assertEqual(field.type, "string")
        self.assertEqual(field, QueryTypeField(name="test_name", type="string"))


class TestQueryProfileType(unittest.TestCase):
    def test_empty(self):
        query_profile_type = QueryProfileType()
        self.assertEqual(query_profile_type.fields, [])

    def test_one_field(self):
        query_profile_type = QueryProfileType()
        field = QueryTypeField(name="test_name", type="string")
        query_profile_type.add_fields(field)
        self.assertEqual(query_profile_type.fields, [field])
        self.assertEqual(query_profile_type, QueryProfileType([field]))

    def test_two_fields(self):
        query_profile_type = QueryProfileType()
        field_1 = QueryTypeField(name="test_name", type="string")
        field_2 = QueryTypeField(
            name="test_name_2",
            type="string",
        )
        query_profile_type.add_fields(field_1, field_2)
        self.assertEqual(query_profile_type.fields, [field_1, field_2])
        self.assertEqual(query_profile_type, QueryProfileType([field_1, field_2]))


class TestQueryField(unittest.TestCase):
    def test_field_name_type(self):
        field = QueryField(name="test_name", value=1)
        self.assertEqual(field.name, "test_name")
        self.assertEqual(field.value, 1)
        self.assertEqual(field, QueryField(name="test_name", value=1))


class TestQueryProfile(unittest.TestCase):
    def test_empty(self):
        query_profile = QueryProfile()
        self.assertEqual(query_profile.fields, [])

    def test_one_field(self):
        query_profile = QueryProfile()
        field = QueryField(name="test_name", value=2.0)
        query_profile.add_fields(field)
        self.assertEqual(query_profile.fields, [field])
        self.assertEqual(query_profile, QueryProfile([field]))

    def test_two_fields(self):
        query_profile = QueryProfile()
        field_1 = QueryField(name="test_name", value=2.0)
        field_2 = QueryField(
            name="test_name_2",
            value="string",
        )
        query_profile.add_fields(field_1, field_2)
        self.assertEqual(query_profile.fields, [field_1, field_2])
        self.assertEqual(query_profile, QueryProfile([field_1, field_2]))


class TestApplicationPackage(unittest.TestCase):
    def setUp(self) -> None:
        self.test_schema = Schema(
            name="msmarco",
            document=Document(
                inherits="context",
                fields=[
                    Field(name="id", type="string", indexing=["attribute", "summary"]),
                    Field(
                        name="title",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="body",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="embedding",
                        type="tensor<float>(x[128])",
                        indexing=["attribute", "summary"],
                        attribute=["fast-search", "fast-access"],
                    ),
                ],
            ),
            fieldsets=[FieldSet(name="default", fields=["title", "body"])],
            rank_profiles=[
                RankProfile(name="default", first_phase="nativeRank(title, body)"),
                RankProfile(
                    name="bm25",
                    first_phase="bm25(title) + bm25(body)",
                    inherits="default",
                    num_threads_per_search=4,
                ),
                RankProfile(
                    name="bert",
                    first_phase="bm25(title) + bm25(body)",
                    second_phase=SecondPhaseRanking(
                        rerank_count=100,
                        expression="bm25(title)",
                        rank_score_drop_limit=0,
                    ),
                    global_phase=GlobalPhaseRanking(
                        rerank_count=10,
                        expression="sum(onnx(bert).logits{d0:0,d1:0})",
                        rank_score_drop_limit=0,
                    ),
                    inherits="default",
                    constants={"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102},
                    functions=[
                        Function(
                            name="question_length",
                            expression="sum(map(query(query_token_ids), f(a)(a > 0)))",
                        ),
                        Function(
                            name="doc_length",
                            expression="sum(map(attribute(doc_token_ids), f(a)(a > 0)))",
                        ),
                        Function(
                            name="input_ids",
                            expression="tensor<float>(d0[1],d1[128])(\n"
                            "    if (d1 == 0,\n"
                            "        TOKEN_CLS,\n"
                            "    if (d1 < question_length + 1,\n"
                            "        query(query_token_ids){d0:(d1-1)},\n"
                            "    if (d1 == question_length + 1,\n"
                            "        TOKEN_SEP,\n"
                            "    if (d1 < question_length + doc_length + 2,\n"
                            "        attribute(doc_token_ids){d0:(d1-question_length-2)},\n"
                            "    if (d1 == question_length + doc_length + 2,\n"
                            "        TOKEN_SEP,\n"
                            "        TOKEN_NONE\n"
                            "    ))))))",
                        ),
                        Function(
                            name="attention_mask",
                            expression="map(input_ids, f(a)(a > 0))",
                        ),
                        Function(
                            name="token_type_ids",
                            expression="tensor<float>(d0[1],d1[128])(\n"
                            "    if (d1 < question_length,\n"
                            "        0,\n"
                            "    if (d1 < question_length + doc_length,\n"
                            "        1,\n"
                            "        TOKEN_NONE\n"
                            "    )))",
                        ),
                    ],
                    summary_features=[
                        "onnx(bert).logits",
                        "input_ids",
                        "attention_mask",
                        "token_type_ids",
                    ],
                ),
            ],
            models=[
                OnnxModel(
                    model_name="bert",
                    model_file_path="bert.onnx",
                    inputs={
                        "input_ids": "input_ids",
                        "token_type_ids": "token_type_ids",
                        "attention_mask": "attention_mask",
                    },
                    outputs={"logits": "logits"},
                )
            ],
        )
        test_query_profile_type = QueryProfileType(
            fields=[
                QueryTypeField(
                    name="ranking.features.query(query_bert)",
                    type="tensor<float>(x[768])",
                )
            ]
        )
        test_query_profile = QueryProfile(
            fields=[
                QueryField(name="maxHits", value=100),
                QueryField(name="anotherField", value="string_value"),
            ]
        )
        self.app_package = ApplicationPackage(
            name="testapp",
            schema=[self.test_schema],
            query_profile=test_query_profile,
            query_profile_type=test_query_profile_type,
        )

    def test_get_schema(self):
        self.assertEqual(self.app_package.schema, self.test_schema)
        self.assertEqual(self.app_package.schema, self.app_package.get_schema())

    @unittest.skipIf(
        platform.system() == "Windows", "Disabled on Windows due to path differences"
    )
    def test_schema_to_text(self):
        expected_result = """schema msmarco {
    document msmarco inherits context {
        field id type string {
            indexing: attribute | summary
        }
        field title type string {
            indexing: index | summary
            index: enable-bm25
        }
        field body type string {
            indexing: index | summary
            index: enable-bm25
        }
        field embedding type tensor<float>(x[128]) {
            indexing: attribute | summary
            attribute {
                fast-search
                fast-access
            }
        }
    }
    fieldset default {
        fields: title, body
    }
    onnx-model bert {
        file: files/bert.onnx
        input input_ids: input_ids
        input token_type_ids: token_type_ids
        input attention_mask: attention_mask
        output logits: logits
    }
    rank-profile default {
        first-phase {
            expression {
                nativeRank(title, body)
            }
        }
    }
    rank-profile bm25 inherits default {
        first-phase {
            expression {
                bm25(title) + bm25(body)
            }
        }
        num-threads-per-search: 4
    }
    rank-profile bert inherits default {
        constants {
            TOKEN_NONE: 0
            TOKEN_CLS: 101
            TOKEN_SEP: 102
        }
        function question_length() {
            expression {
                sum(map(query(query_token_ids), f(a)(a > 0)))
            }
        }
        function doc_length() {
            expression {
                sum(map(attribute(doc_token_ids), f(a)(a > 0)))
            }
        }
        function input_ids() {
            expression {
                tensor<float>(d0[1],d1[128])(
                    if (d1 == 0,
                        TOKEN_CLS,
                    if (d1 < question_length + 1,
                        query(query_token_ids){d0:(d1-1)},
                    if (d1 == question_length + 1,
                        TOKEN_SEP,
                    if (d1 < question_length + doc_length + 2,
                        attribute(doc_token_ids){d0:(d1-question_length-2)},
                    if (d1 == question_length + doc_length + 2,
                        TOKEN_SEP,
                        TOKEN_NONE
                    ))))))
            }
        }
        function attention_mask() {
            expression {
                map(input_ids, f(a)(a > 0))
            }
        }
        function token_type_ids() {
            expression {
                tensor<float>(d0[1],d1[128])(
                    if (d1 < question_length,
                        0,
                    if (d1 < question_length + doc_length,
                        1,
                        TOKEN_NONE
                    )))
            }
        }
        first-phase {
            expression {
                bm25(title) + bm25(body)
            }
        }
        second-phase {
            expression {
                bm25(title)
            }
            rerank-count: 100
            rank-score-drop-limit: 0
        }
        global-phase {
            expression {
                sum(onnx(bert).logits{d0:0,d1:0})
            }
            rerank-count: 10
            rank-score-drop-limit: 0
        }
        summary-features {
            onnx(bert).logits
            input_ids
            attention_mask
            token_type_ids
        }
    }
}"""
        self.assertEqual(self.app_package.schema.schema_to_text, expected_result)

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testapp_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            "        <redundancy>1</redundancy>\n"
            "        <documents>\n"
            '            <document type="msmarco" mode="index"></document>\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )

        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result)
        )

    def test_query_profile_to_text(self):
        expected_result = (
            '<query-profile id="default" type="root">\n'
            '    <field name="maxHits">100</field>\n'
            '    <field name="anotherField">string_value</field>\n'
            "</query-profile>"
        )

        self.assertEqual(self.app_package.query_profile_to_text, expected_result)

    def test_query_profile_type_to_text(self):
        expected_result = (
            '<query-profile-type id="root">\n'
            '    <field name="ranking.features.query(query_bert)" type="tensor&lt;float&gt;(x[768])" />\n'
            "</query-profile-type>"
        )
        self.assertEqual(self.app_package.query_profile_type_to_text, expected_result)


class TestApplicationPackageStreaming(unittest.TestCase):
    def setUp(self) -> None:
        self.mail = Schema(
            name="mail",
            mode="streaming",
            document=Document(
                fields=[
                    Field(
                        name="title", type="string", indexing=["attribute", "summary"]
                    )
                ]
            ),
        )
        self.calendar = Schema(
            name="calendar",
            mode="streaming",
            document=Document(
                fields=[
                    Field(
                        name="title", type="string", indexing=["attribute", "summary"]
                    )
                ]
            ),
        )
        self.event = Schema(
            name="event",
            mode="index",
            document=Document(
                fields=[
                    Field(
                        name="title", type="string", indexing=["attribute", "summary"]
                    )
                ]
            ),
        )
        self.app_package = ApplicationPackage(
            name="testapp", schema=[self.mail, self.calendar, self.event]
        )

    def test_generated_services_uses_mode_streaming(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testapp_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            "        <redundancy>1</redundancy>\n"
            "        <documents>\n"
            '            <document type="mail" mode="streaming"></document>\n'
            '            <document type="calendar" mode="streaming"></document>\n'
            '            <document type="event" mode="index"></document>\n'
            '            <document-processing chain="indexing" cluster="testapp_container" />\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )
        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result)
        )


class TestSchemaInheritance(unittest.TestCase):
    def setUp(self) -> None:
        self.news_schema = Schema(
            name="news",
            document=Document(
                fields=[
                    Field(
                        name="news_id", type="string", indexing=["attribute", "summary"]
                    )
                ]
            ),
        )
        self.mail = Schema(
            name="mail",
            inherits="news",
            document=Document(
                inherits="news",
                fields=[
                    Field(
                        name="mail_id", type="string", indexing=["attribute", "summary"]
                    ),
                ],
            ),
        )

        self.app_package = ApplicationPackage(
            name="testapp",
            schema=[self.news_schema, self.mail],
        )

    def test_schema_to_text(self):
        expected_mail_result = (
            "schema mail inherits news {\n"
            "    document mail inherits news {\n"
            "        field mail_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "}"
        )
        self.assertEqual(
            self.app_package.get_schema(name="mail").schema_to_text,
            expected_mail_result,
        )


class TestApplicationPackageMultipleSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.news_schema = Schema(
            name="news",
            document=Document(
                fields=[
                    Field(
                        name="news_id", type="string", indexing=["attribute", "summary"]
                    ),
                    Field(
                        name="category_ctr_ref",
                        type="reference<category_ctr>",
                        indexing=["attribute"],
                    ),
                ]
            ),
            imported_fields=[
                ImportedField(
                    name="global_category_ctrs",
                    reference_field="category_ctr_ref",
                    field_to_import="ctrs",
                )
            ],
        )
        self.user_schema = Schema(
            name="user",
            document=Document(
                fields=[
                    Field(
                        name="user_id", type="string", indexing=["attribute", "summary"]
                    ),
                ]
            ),
        )
        self.category_ctr_schema = Schema(
            name="category_ctr",
            global_document=True,
            document=Document(
                fields=[
                    Field(
                        name="ctrs",
                        type="tensor<float>(category{})",
                        indexing=["attribute"],
                        attribute=["fast-search"],
                    ),
                ]
            ),
        )
        self.app_package = ApplicationPackage(
            name="testapp",
            schema=[self.news_schema, self.user_schema, self.category_ctr_schema],
        )

    def test_get_schema(self):
        self.assertEqual(self.app_package.get_schema(name="news"), self.news_schema)
        self.assertEqual(self.app_package.get_schema(name="user"), self.user_schema)
        self.assertEqual(
            self.app_package.get_schema(name="category_ctr"), self.category_ctr_schema
        )
        with self.assertRaises(AssertionError):
            self.app_package.get_schema()

    def test_schema_to_text(self):
        expected_news_result = (
            "schema news {\n"
            "    document news {\n"
            "        field news_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "        field category_ctr_ref type reference<category_ctr> {\n"
            "            indexing: attribute\n"
            "        }\n"
            "    }\n"
            "    import field category_ctr_ref.ctrs as global_category_ctrs {}\n"
            "}"
        )
        expected_user_result = (
            "schema user {\n"
            "    document user {\n"
            "        field user_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "}"
        )
        expected_category_ctr_result = (
            "schema category_ctr {\n"
            "    document category_ctr {\n"
            "        field ctrs type tensor<float>(category{}) {\n"
            "            indexing: attribute\n"
            "            attribute {\n"
            "                fast-search\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}"
        )

        expected_results = {
            "news": expected_news_result,
            "user": expected_user_result,
            "category_ctr": expected_category_ctr_result,
        }

        for schema in self.app_package.schemas:
            self.assertEqual(schema.schema_to_text, expected_results[schema.name])

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testapp_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            "        <redundancy>1</redundancy>\n"
            "        <documents>\n"
            '            <document type="news" mode="index"></document>\n'
            '            <document type="user" mode="index"></document>\n'
            '            <document type="category_ctr" mode="index" global="true"></document>\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )

        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result),
        )


class TestSimplifiedApplicationPackage(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = ApplicationPackage(name="testapp")

        self.app_package.schema.add_fields(
            Field(name="id", type="string", indexing=["attribute", "summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="body",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="tensor_field",
                type="tensor<float>(x[128])",
                indexing=["attribute"],
                attribute=["fast-search", "fast-access"],
                ann=HNSW(
                    distance_metric="euclidean",
                    max_links_per_node=16,
                    neighbors_to_explore_at_insert=200,
                ),
            ),
            Field(
                name="embedding",
                type="tensor<bfloat16>(x[384])",
                is_document_field=False,
                indexing=[
                    '(input title || "") . " " . (input body || "")',
                    "embed embedder",
                    "attribute",
                    "index",
                ],
                index="hnsw",
            ),
        )
        self.app_package.schema.add_field_set(
            FieldSet(name="default", fields=["title", "body"])
        )
        self.app_package.schema.add_rank_profile(
            RankProfile(name="default", first_phase="nativeRank(title, body)")
        )
        self.app_package.schema.add_rank_profile(
            RankProfile(
                name="bm25",
                first_phase="bm25(title) + bm25(body)",
                inherits="default",
                global_phase=GlobalPhaseRanking(
                    rerank_count=10, expression="bm25(title)"
                ),
            )
        )
        self.app_package.query_profile_type.add_fields(
            QueryTypeField(
                name="ranking.features.query(query_bert)",
                type="tensor<float>(x[768])",
            )
        )
        self.app_package.query_profile.add_fields(
            QueryField(name="maxHits", value=100),
            QueryField(name="anotherField", value="string_value"),
        )

    def test_schema_to_text(self):
        self.maxDiff = None
        expected_result = """schema testapp {
    document testapp {
        field id type string {
            indexing: attribute | summary
        }
        field title type string {
            indexing: index | summary
            index: enable-bm25
        }
        field body type string {
            indexing: index | summary
            index: enable-bm25
        }
        field tensor_field type tensor<float>(x[128]) {
            indexing: attribute
            attribute {
                distance-metric: euclidean
                fast-search
                fast-access
            }
            index {
                hnsw {
                    max-links-per-node: 16
                    neighbors-to-explore-at-insert: 200
                }
            }
        }
    }
    field embedding type tensor<bfloat16>(x[384]) {
        indexing: (input title || "") . " " . (input body || "") | embed embedder | attribute | index
        index: hnsw
    }
    fieldset default {
        fields: title, body
    }
    rank-profile default {
        first-phase {
            expression {
                nativeRank(title, body)
            }
        }
    }
    rank-profile bm25 inherits default {
        first-phase {
            expression {
                bm25(title) + bm25(body)
            }
        }
        global-phase {
            expression {
                bm25(title)
            }
            rerank-count: 10
        }
    }
}"""
        self.assertEqual(self.app_package.schema.schema_to_text, expected_result)

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testapp_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            "        <redundancy>1</redundancy>\n"
            "        <documents>\n"
            '            <document type="testapp" mode="index"></document>\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )

        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result),
        )

    def test_query_profile_to_text(self):
        expected_result = (
            '<query-profile id="default" type="root">\n'
            '    <field name="maxHits">100</field>\n'
            '    <field name="anotherField">string_value</field>\n'
            "</query-profile>"
        )

        self.assertEqual(self.app_package.query_profile_to_text, expected_result)

    def test_query_profile_type_to_text(self):
        expected_result = (
            '<query-profile-type id="root">\n'
            '    <field name="ranking.features.query(query_bert)" type="tensor&lt;float&gt;(x[768])" />\n'
            "</query-profile-type>"
        )
        self.assertEqual(self.app_package.query_profile_type_to_text, expected_result)

    def test_rank_profile_match_phase(self):
        rank_profile = RankProfile(
            name="match_phase_test",
            first_phase="bm25(title) + bm25(body)",
            match_phase=MatchPhaseRanking(
                attribute="popularity", order="descending", max_hits=1000
            ),
        )
        self.assertEqual(rank_profile.name, "match_phase_test")
        self.assertEqual(rank_profile.first_phase, "bm25(title) + bm25(body)")
        self.assertEqual(rank_profile.match_phase.attribute, "popularity")
        self.assertEqual(rank_profile.match_phase.order, "descending")
        self.assertEqual(rank_profile.match_phase.max_hits, 1000)

    def test_schema_to_text_with_match_phase(self):
        schema = Schema(
            name="test_match_phase",
            document=Document(
                fields=[
                    Field(name="title", type="string", indexing=["index", "summary"]),
                    Field(name="body", type="string", indexing=["index", "summary"]),
                    Field(name="popularity", type="int", indexing=["attribute"]),
                ]
            ),
            rank_profiles=[
                RankProfile(name="default", first_phase="nativeRank(title, body)"),
                RankProfile(
                    name="match_phase_test",
                    first_phase="bm25(title) + bm25(body)",
                    match_phase=MatchPhaseRanking(
                        attribute="popularity", order="descending", max_hits=1000
                    ),
                ),
            ],
        )
        expected_schema = """schema test_match_phase {
    document test_match_phase {
        field title type string {
            indexing: index | summary
        }
        field body type string {
            indexing: index | summary
        }
        field popularity type int {
            indexing: attribute
        }
    }
    rank-profile default {
        first-phase {
            expression {
                nativeRank(title, body)
            }
        }
    }
    rank-profile match_phase_test {
        match-phase {
            attribute: popularity
            order: descending
            max-hits: 1000
        }
        first-phase {
            expression {
                bm25(title) + bm25(body)
            }
        }
    }
}"""

        self.assertEqual(schema.schema_to_text, expected_schema)


class TestSimplifiedApplicationPackageWithMultipleSchemas(unittest.TestCase):
    def setUp(self) -> None:
        self.app_package = ApplicationPackage(name="news")

        self.app_package.schema.add_fields(
            Field(name="news_id", type="string", indexing=["attribute", "summary"]),
        )
        self.app_package.add_schema(
            Schema(
                name="user",
                document=Document(
                    fields=[
                        Field(
                            name="user_id",
                            type="string",
                            indexing=["attribute", "summary"],
                        )
                    ]
                ),
            )
        )

    def test_schema_to_text(self):
        expected_news_result = (
            "schema news {\n"
            "    document news {\n"
            "        field news_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "}"
        )
        expected_user_result = (
            "schema user {\n"
            "    document user {\n"
            "        field user_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "}"
        )
        expected_results = {"news": expected_news_result, "user": expected_user_result}

        for schema in self.app_package.schemas:
            self.assertEqual(schema.schema_to_text, expected_results[schema.name])

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="news_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            "    </container>\n"
            '    <content id="news_content" version="1.0">\n'
            "        <redundancy>1</redundancy>\n"
            "        <documents>\n"
            '            <document type="news" mode="index"></document>\n'
            '            <document type="user" mode="index"></document>\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )
        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result),
        )


class TestComponentSetup(unittest.TestCase):
    def setUp(self) -> None:
        components = [
            Component(id="my-component", bundle="my-bundle"),
            Component(
                id="hf-embedder",
                type="hugging-face-embedder",
                parameters=[
                    Parameter("transformer-model", {"path": "my-models/model.onnx"}),
                    Parameter("tokenizer-model", {"path": "my-models/tokenizer.json"}),
                ],
            ),
            Component(
                id="my-custom-component",
                cls="com.example.MyCustomEmbedder",
                parameters=[
                    Parameter(
                        "config",
                        {"name": "com.example.my-embedder"},
                        [
                            Parameter("model", {"model-id": "minilm-l6-v2"}),
                            Parameter("vocab", {"path": "files/vocab.txt"}),
                            Parameter("myValue", {}, "foo"),
                        ],
                    ),
                ],
            ),
        ]
        self.app_package = ApplicationPackage(name="content", components=components)

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="content_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            '        <component id="my-component" bundle="my-bundle"/>\n'
            '        <component id="hf-embedder" type="hugging-face-embedder">\n'
            '            <transformer-model path="my-models/model.onnx"/>\n'
            '            <tokenizer-model path="my-models/tokenizer.json"/>\n'
            "        </component>\n"
            '        <component id="my-custom-component" class="com.example.MyCustomEmbedder">\n'
            '            <config name="com.example.my-embedder">\n'
            '                <model model-id="minilm-l6-v2"/>\n'
            '                <vocab path="files/vocab.txt"/>\n'
            "                <myValue>foo</myValue>\n"
            "            </config>\n"
            "        </component>\n"
            "    </container>\n"
            '    <content id="content_content" version="1.0">\n'
            "        <redundancy>1</redundancy>\n"
            "        <documents>\n"
            '            <document type="content" mode="index"></document>\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )
        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result),
        )


class TestClientTokenSetup(unittest.TestCase):
    def setUp(self) -> None:
        clients = [
            AuthClient(
                id="mtls",
                permissions=["read"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token",
                permissions=["read"],
                parameters=[Parameter("token", {"id": "accessToken"})],
            ),
        ]
        self.app_package = ApplicationPackage(name="content", auth_clients=clients)

    def test_services_to_text(self):
        self.maxDiff = None
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="content_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            "        <clients>\n"
            '            <client id="mtls" permissions="read">\n'
            '                <certificate file="security/clients.pem"/>\n'
            "            </client>\n"
            '            <client id="token" permissions="read">\n'
            '                <token id="accessToken"/>\n'
            "            </client>\n"
            "        </clients>\n"
            "    </container>\n"
            '    <content id="content_content" version="1.0">\n'
            "        <redundancy>1</redundancy>\n"
            "        <documents>\n"
            '            <document type="content" mode="index"></document>\n'
            "        </documents>\n"
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "    </content>\n"
            "</services>"
        )

        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result),
        )


class TestClientsWithCluster(unittest.TestCase):
    def setUp(self) -> None:
        schema_name = "test"
        clients = [
            AuthClient(
                id="mtls",
                permissions=["read", "write"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token",
                permissions=["read", "write"],
                parameters=[Parameter("token", {"id": "accessToken"})],
            ),
        ]
        clusters = [
            ContentCluster(
                id=f"{schema_name}_content",
                nodes=Nodes(count="2"),
                document_name=schema_name,
                min_redundancy="2",
            ),
            ContainerCluster(
                id=f"{schema_name}_container",
                nodes=Nodes(count="2"),
                auth_clients=clients,
            ),
        ]
        self.app_package = ApplicationPackage(
            name="testapp",
            clusters=clusters,
        )

    def test_services_to_text(self):
        self.maxDiff = None
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <content id="test_content" version="1.0">\n'
            '        <nodes count="2"/>\n'
            "        <min-redundancy>2</min-redundancy>\n"
            "        <documents>\n"
            '            <document type="test" mode="index"></document>\n'
            "        </documents>\n"
            "    </content>\n"
            '    <container id="test_container" version="1.0">\n'
            '        <nodes count="2"/>\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            "        <clients>\n"
            '            <client id="mtls" permissions="read,write">\n'
            '                <certificate file="security/clients.pem"/>\n'
            "            </client>\n"
            '            <client id="token" permissions="read,write">\n'
            '                <token id="accessToken"/>\n'
            "            </client>\n"
            "        </clients>\n"
            "    </container>\n"
            "</services>"
        )
        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result),
        )


class TestValidAppName(unittest.TestCase):
    def test_invalid_name(self):
        with pytest.raises(ValueError):
            ApplicationPackage(name="test_app")
        with pytest.raises(ValueError):
            ApplicationPackage(name="test-app")
        with pytest.raises(ValueError):
            ApplicationPackage(name="42testapp")
        with pytest.raises(ValueError):
            ApplicationPackage(name="testApp")
        with pytest.raises(ValueError):
            ApplicationPackage(name="testapp" + "x" * 20)


class TestFieldAlias(unittest.TestCase):
    def setUp(self) -> None:
        self.test_schema = Schema(
            name="alias_test_schema",
            document=Document(
                fields=[
                    Field(
                        name="single_aliased_field",
                        type="string",
                        alias=["single_aliased_field_alias"],
                    ),
                    Field(
                        name="single_component_aliased_field",
                        type="string",
                        alias=["component:component_alias"],
                    ),
                    Field(
                        name="multiple_aliased_field",
                        type="string",
                        alias=[
                            "first_alias",
                            "second_alias",
                            "third_alias",
                            "fourth_component: fourth_alias",
                        ],
                    ),
                ]
            ),
        )

        self.app_package = ApplicationPackage(
            name="testapp",
            schema=[self.test_schema],
        )

    def test_alias_to_schema(self) -> None:
        expected_result = (
            "schema alias_test_schema {\n"
            "    document alias_test_schema {\n"
            "        field single_aliased_field type string {\n"
            "            alias: single_aliased_field_alias\n"
            "        }\n"
            "        field single_component_aliased_field type string {\n"
            "            alias component: component_alias\n"
            "        }\n"
            "        field multiple_aliased_field type string {\n"
            "            alias: first_alias\n"
            "            alias: second_alias\n"
            "            alias: third_alias\n"
            "            alias fourth_component: fourth_alias\n"
            "        }\n"
            "    }\n"
            "}"
        )
        self.assertEqual(
            self.app_package.get_schema("alias_test_schema").schema_to_text,
            expected_result,
        )


class TestCluster(unittest.TestCase):
    def setUp(self) -> None:
        clusters = [
            ContainerCluster(
                id="test_container",
                nodes=Nodes(
                    count="1",
                    parameters=[
                        Parameter(
                            "resources",
                            {"vcpu": "4.0", "memory": "16Gb", "disk": "125Gb"},
                            [Parameter("gpu", {"count": "1", "memory": "16Gb"})],
                        ),
                    ],
                ),
                components=[
                    Component(
                        id="e5",
                        type="hugging-face-embedder",
                        parameters=[
                            Parameter(
                                "transformer-model", {"path": "model/model.onnx"}
                            ),
                            Parameter(
                                "tokenizer-model", {"path": "model/tokenizer.json"}
                            ),
                        ],
                    )
                ],
            ),
            ContentCluster(id="test_content", document_name="test"),
        ]

        self.app_package = ApplicationPackage(name="test", clusters=clusters)

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="test_container" version="1.0">\n'
            '        <nodes count="1">\n'
            '            <resources vcpu="4.0" memory="16Gb" disk="125Gb">\n'
            '                <gpu count="1" memory="16Gb"/>\n'
            "            </resources>\n"
            "        </nodes>\n"
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "        <document-processing></document-processing>\n"
            '        <component id="e5" type="hugging-face-embedder">\n'
            '            <transformer-model path="model/model.onnx"/>\n'
            '            <tokenizer-model path="model/tokenizer.json"/>\n'
            "        </component>\n"
            "    </container>\n"
            '    <content id="test_content" version="1.0">\n'
            "        <nodes>\n"
            '            <node distribution-key="0" hostalias="node1"></node>\n'
            "        </nodes>\n"
            "        <min-redundancy>1</min-redundancy>\n"
            "        <documents>\n"
            '            <document type="test" mode="index"></document>\n'
            "        </documents>\n"
            "    </content>\n"
            "</services>"
        )
        self.assertEqual(self.app_package.services_to_text, expected_result)
        self.assertTrue(
            compare_xml(self.app_package.services_to_text_vt, expected_result),
        )


class TestAuthClientEquality(unittest.TestCase):
    def setUp(self) -> None:
        self.clients_one = [
            AuthClient(
                id="mtls",
                permissions=["read", "write"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token",
                permissions=["read"],
                parameters=[Parameter("token", {"id": "accessToken"})],
            ),
        ]

        self.clients_two = [
            AuthClient(
                id="token",
                permissions=["read"],
                parameters=[Parameter("token", {"id": "accessToken"})],
            ),
            AuthClient(
                id="mtls",
                permissions=["read", "write"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
        ]

        self.clients_three = [
            AuthClient(
                id="mtls",
                permissions=["read"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="foo",
                permissions=["read"],
                parameters=[Parameter("token", {"id": "bar"})],
            ),
        ]

    def test_auth_client_equality(self):
        # Test equality between two lists with different order
        self.assertEqual(sorted(self.clients_one), sorted(self.clients_two))

        # Test inequality between different client lists
        self.assertNotEqual(self.clients_one, self.clients_three)


class TestDeploymentConfiguration(unittest.TestCase):
    def test_deployment_to_text(self):
        deploy_config = DeploymentConfiguration(
            environment="prod", regions=["aws-us-east-1c", "aws-us-west-2a"]
        )

        app_package = ApplicationPackage(name="test", deployment_config=deploy_config)

        expected_result = (
            '<deployment version="1.0">\n'
            "    <prod>\n"
            "        <region>aws-us-east-1c</region>\n"
            "        <region>aws-us-west-2a</region>\n"
            "    </prod>\n"
            "</deployment>"
        )

        self.assertEqual(expected_result, app_package.deployment_to_text)


class TestSchemaStructField(unittest.TestCase):
    def setUp(self):
        self.app_package = ApplicationPackage(name="struct")

        mystruct = Struct("mystruct", [Field("key", "string"), Field("value", "int")])

        my_array = Field(
            "my_array",
            "array<mystruct>",
            ["summary"],
            struct_fields=[
                StructField(
                    "key",
                    indexing=["attribute"],
                    attribute=["fast-search"],
                    rank="filter",
                )
            ],
        )

        self.app_package.schema.document = Document([my_array], None, [mystruct])

    def test_schema_to_text(self):
        expected_result = (
            "schema struct {\n"
            "    document struct {\n"
            "        field my_array type array<mystruct> {\n"
            "            indexing: summary\n"
            "            struct-field key {\n"
            "                indexing: attribute\n"
            "                attribute {\n"
            "                    fast-search\n"
            "                }\n"
            "                rank: filter\n"
            "            }\n"
            "        }\n"
            "        struct mystruct {\n"
            "            field key type string {\n"
            "            }\n"
            "            field value type int {\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}"
        )
        self.assertEqual(self.app_package.schema.schema_to_text, expected_result)


class TestVTequality(unittest.TestCase):
    def test_application_configuration(self):
        app_config = ApplicationConfiguration(
            name="container.handler.observability.application-userdata",
            value={"version": "my-version"},
        )
        app_config_vt = app_config.to_vt()
        vt_str = str(app_config_vt.to_xml())
        app_config_str = app_config.to_text
        self.assertTrue(compare_xml(app_config_str, vt_str))

    def test_cluster_configuration(self):
        clusters = [
            ContainerCluster(
                id="test_container",
                nodes=Nodes(
                    count="1",
                    parameters=[
                        Parameter(
                            "resources",
                            {"vcpu": "4.0", "memory": "16Gb", "disk": "125Gb"},
                            [Parameter("gpu", {"count": "1", "memory": "16Gb"})],
                        ),
                    ],
                ),
                components=[
                    Component(
                        id="e5",
                        type="hugging-face-embedder",
                        parameters=[
                            Parameter(
                                "transformer-model", {"path": "model/model.onnx"}
                            ),
                            Parameter(
                                "tokenizer-model", {"path": "model/tokenizer.json"}
                            ),
                        ],
                    )
                ],
                auth_clients=[
                    AuthClient(
                        id="mtls",
                        permissions=["read", "write"],
                        parameters=[
                            Parameter("certificate", {"file": "security/clients.pem"})
                        ],
                    ),
                    AuthClient(
                        id="token",
                        permissions=["read"],
                        parameters=[Parameter("token", {"id": "accessToken"})],
                    ),
                ],
            ),
            ContentCluster(id="test_content", document_name="test"),
        ]
        for cluster_config in clusters:
            vt_str = str(cluster_config.to_vt().to_xml())
            cluster_config_str = cluster_config.to_xml_string()
            self.assertTrue(compare_xml(cluster_config_str, vt_str))


class TestServiceConfig(unittest.TestCase):
    def test_default_service_config_to_text(self):
        self.maxDiff = None
        application_name = "test"
        service_config = ServicesConfiguration(application_name=application_name)
        app_package = ApplicationPackage(
            name=application_name, services_config=service_config
        )
        expected_result = '<?xml version="1.0" encoding="UTF-8" ?>\n<services version="1.0">\n  <container id="test_container" version="1.0"></container>\n</services>'
        self.assertEqual(expected_result, app_package.services_to_text)
        self.assertTrue(
            compare_xml(app_package.services_to_text_vt, expected_result),
        )

    def test_document_expiry(self):
        # Create a Schema with name music and a field with name artist, title and timestamp
        # Ref https://docs.vespa.ai/en/documents.html#document-expiry
        application_name = "music"
        music_schema = Schema(
            name=application_name,
            document=Document(
                fields=[
                    Field(
                        name="artist",
                        type="string",
                        indexing=["attribute", "summary"],
                    ),
                    Field(
                        name="title",
                        type="string",
                        indexing=["attribute", "summary"],
                    ),
                    Field(
                        name="timestamp",
                        type="long",
                        indexing=["attribute", "summary"],
                        attribute=["fast-access"],
                    ),
                ]
            ),
        )
        # Create a ServicesConfiguration with document-expiry set to 1 day (timestamp > now() - 86400)
        services_config = ServicesConfiguration(
            application_name=application_name,
            services_config=services(
                container(
                    search(),
                    document_api(),
                    document_processing(),
                    id=f"{application_name}_container",
                    version="1.0",
                ),
                content(
                    redundancy("1"),
                    documents(
                        document(
                            type=application_name,
                            mode="index",
                            selection="music.timestamp > now() - 86400",
                        ),
                        garbage_collection="true",
                    ),
                    nodes(node(distribution_key="0", hostalias="node1")),
                    id=f"{application_name}_content",
                    version="1.0",
                ),
            ),
        )
        application_package = ApplicationPackage(
            name=application_name,
            schema=[music_schema],
            services_config=services_config,
        )
        expected = """<?xml version="1.0" encoding="UTF-8" ?>
<services>
  <container id="music_container" version="1.0">
    <search></search>
    <document-api></document-api>
    <document-processing></document-processing>
  </container>
  <content id="music_content" version="1.0">
    <redundancy>1</redundancy>
    <documents garbage-collection="true">
      <document type="music" mode="index" selection="music.timestamp &gt; now() - 86400"></document>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"></node>
    </nodes>
  </content>
</services>"""
        self.assertEqual(expected, application_package.services_to_text)
        self.assertTrue(validate_services(application_package.services_to_text))


class TestPredicateField(unittest.TestCase):
    def setUp(self):
        self.app_package = ApplicationPackage(name="predicatetest")

        # Add a document with a predicate field
        self.app_package.schema.add_fields(
            Field(name="id", type="string", indexing=["attribute", "summary"]),
            Field(
                name="predicate_field",
                type="predicate",
                indexing=["attribute"],
                index={
                    "arity": 2,
                    "lower-bound": 3,
                    "upper-bound": 200,
                    "dense-posting-list-threshold": 0.25,
                },
            ),
        )

    def test_predicate_field_schema(self):
        expected_result = """schema predicatetest {
    document predicatetest {
        field id type string {
            indexing: attribute | summary
        }
        field predicate_field type predicate {
            indexing: attribute
            index {
                arity: 2
                lower-bound: 3
                upper-bound: 200
                dense-posting-list-threshold: 0.25
            }
        }
    }
}"""
        print()
        print(self.app_package.schema.schema_to_text)
        print()
        print(expected_result)
        self.assertEqual(self.app_package.schema.schema_to_text, expected_result)


class TestRankProfileCustomSettings(unittest.TestCase):
    def test_rank_profile_with_filter_and_weakand(self):
        # Create a minimal schema with a dummy document to allow rank profile rendering.
        dummy_document = Document(fields=[Field(name="dummy", type="string")])
        rank_profile = RankProfile(
            name="optimized",
            first_phase="nativeRank(dummy)",
            inherits="baseline",
            filter_threshold=0.05,
            weakand={"stopword-limit": 0.6, "adjust-target": 0.01},
        )
        schema = Schema(
            name="test_schema",
            document=dummy_document,
            rank_profiles=[rank_profile],
        )
        # Expected text for the rank profile block.
        expected_schema = """\
schema test_schema {
    document test_schema {
        field dummy type string {
        }
    }
    rank-profile optimized inherits baseline {
        filter-threshold: 0.05
        weakand {
            stopword-limit: 0.6
            adjust-target: 0.01
        }
        first-phase {
            expression {
                nativeRank(dummy)
            }
        }
    }
}"""
        # Compare the expected and actual schema text.
        actual_schema = schema.schema_to_text
        self.assertEqual(
            actual_schema,
            expected_schema,
        )


class TestFieldIndexConfigurations(unittest.TestCase):
    """Tests for the multiple index configurations feature in Field definitions."""

    def test_single_string_index_backward_compatibility(self):
        """Test that single string index configuration works as before."""
        field = Field(name="title", type="string", index="enable-bm25")
        self.assertEqual(field.index, "enable-bm25")
        self.assertEqual(field.index_configurations, ["enable-bm25"])

        # Test rendering uses simple syntax
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text

        expected_schema = """\
schema test {
    document test {
        field title type string {
            index: enable-bm25
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)

    def test_single_dict_index(self):
        """Test that single dict index configuration works"""
        index_config = {"arity": 2, "lower-bound": 3}
        field = Field(name="predicate_field", type="predicate", index=index_config)
        self.assertEqual(field.index, index_config)
        self.assertEqual(field.index_configurations, [index_config])

        # Test rendering uses block syntax (since it's a dict)
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text

        expected_schema = """\
schema test {
    document test {
        field predicate_field type predicate {
            index {
                arity: 2
                lower-bound: 3
            }
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)

    def test_no_index_configuration(self):
        """Test field with no index configuration."""
        field = Field(name="no_index", type="string")
        self.assertIsNone(field.index)
        self.assertEqual(field.index_configurations, [])

        # Test rendering has no index statements
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text

        expected_schema = """\
schema test {
    document test {
        field no_index type string {
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)

    def test_multiple_string_indices(self):
        """Test field with multiple string index configurations."""
        field = Field(
            name="multi_string",
            type="string",
            index=["enable-bm25", "another-setting"],
        )
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text
        expected_schema = """\
schema test {
    document test {
        field multi_string type string {
            index {
                enable-bm25
                another-setting
            }
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)
        self.assertEqual(field.index, ["enable-bm25", "another-setting"])
        self.assertEqual(field.index_configurations, ["enable-bm25", "another-setting"])

    def test_multiple_dict_indices(self):
        """Test field with multiple dict index configurations."""
        indices = [{"param1": "value1"}, {"param2": "value2"}]
        field = Field(name="multi_dict", type="string", index=indices)
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text
        expected_schema = """\
schema test {
    document test {
        field multi_dict type string {
            index {
                param1: value1
                param2: value2
            }
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)
        self.assertEqual(field.index, indices)
        self.assertEqual(field.index_configurations, indices)

    def test_mixed_string_and_dict_indices(self):
        """Test field with mixed string and dict index configurations."""
        indices = ["enable-bm25", {"arity": 2}, "another-setting"]
        field = Field(name="mixed", type="string", index=indices)
        self.assertEqual(field.index, indices)
        self.assertEqual(field.index_configurations, indices)

    def test_predicate_field_from_issue(self):
        """Test the exact predicate field example from issue #983."""
        field = Field(
            name="predicate_field",
            type="predicate",
            indexing=["attribute"],
            index={
                "arity": 2,
                "lower-bound": 3,
                "upper-bound": 200,
                "dense-posting-list-threshold": 0.25,
            },
        )

        # Create schema and render
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text

        expected_schema = """\
schema test {
    document test {
        field predicate_field type predicate {
            indexing: attribute
            index {
                arity: 2
                lower-bound: 3
                upper-bound: 200
                dense-posting-list-threshold: 0.25
            }
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)

    def test_multiple_index_configurations_rendering(self):
        """Test that multiple index configurations render correctly in schema."""
        field = Field(
            name="multi_index",
            type="string",
            indexing=["index", "summary"],
            index=["enable-bm25", {"arity": 2, "lower-bound": 3}, "another-setting"],
        )

        # Create schema and render
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text

        expected_schema = """\
schema test {
    document test {
        field multi_index type string {
            indexing: index | summary
            index {
                enable-bm25
                arity: 2
                lower-bound: 3
                another-setting
            }
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)

    def test_dict_with_none_values(self):
        """Test that dict index configurations with None values render without ': None'."""
        field = Field(
            name="parameterless",
            type="string",
            index={"enable-bm25": None, "param": "value"},
        )

        # Create schema and render
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text

        expected_schema = """\
schema test {
    document test {
        field parameterless type string {
            index {
                enable-bm25
                param: value
            }
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)

    def test_ann_field_with_additional_index_configs(self):
        """Test that ANN fields work correctly with additional index configurations."""
        field = Field(
            name="vector_field",
            type="tensor<float>(x[128])",
            indexing=["attribute"],
            ann=HNSW(
                distance_metric="euclidean",
                max_links_per_node=16,
                neighbors_to_explore_at_insert=200,
            ),
            index=["enable-bm25", {"custom-param": "value"}],
        )

        # Create schema and render
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text

        expected_schema = """\
schema test {
    document test {
        field vector_field type tensor<float>(x[128]) {
            indexing: attribute
            index {
                enable-bm25
                custom-param: value
            }
            attribute {
                distance-metric: euclidean
            }
            index {
                hnsw {
                    max-links-per-node: 16
                    neighbors-to-explore-at-insert: 200
                }
            }
        }
    }
}"""
        self.assertEqual(schema_text, expected_schema)

    def test_equality_with_multiple_indices(self):
        """Test that Field equality works correctly with multiple index configurations."""
        index_config = ["enable-bm25", {"arity": 2}]
        field1 = Field(name="test", type="string", index=index_config)
        field2 = Field(name="test", type="string", index=index_config.copy())
        field3 = Field(name="test", type="string", index="enable-bm25")

        self.assertEqual(field1, field2)
        self.assertNotEqual(field1, field3)
