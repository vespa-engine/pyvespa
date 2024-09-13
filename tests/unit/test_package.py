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
)


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
                ),
                RankProfile(
                    name="bert",
                    first_phase="bm25(title) + bm25(body)",
                    second_phase=SecondPhaseRanking(
                        rerank_count=100, expression="bm25(title)"
                    ),
                    global_phase=SecondPhaseRanking(
                        rerank_count=10, expression="sum(onnx(bert).logits{d0:0,d1:0})"
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
        expected_result = (
            "schema msmarco {\n"
            "    document msmarco inherits context {\n"
            "        field id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "        field title type string {\n"
            "            indexing: index | summary\n"
            "            index: enable-bm25\n"
            "        }\n"
            "        field body type string {\n"
            "            indexing: index | summary\n"
            "            index: enable-bm25\n"
            "        }\n"
            "        field embedding type tensor<float>(x[128]) {\n"
            "            indexing: attribute | summary\n"
            "            attribute {\n"
            "                fast-search\n"
            "                fast-access\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    fieldset default {\n"
            "        fields: title, body\n"
            "    }\n"
            "    onnx-model bert {\n"
            "        file: files/bert.onnx\n"
            "        input input_ids: input_ids\n"
            "        input token_type_ids: token_type_ids\n"
            "        input attention_mask: attention_mask\n"
            "        output logits: logits\n"
            "    }\n"
            "    rank-profile default {\n"
            "        first-phase {\n"
            "            expression {\n"
            "                nativeRank(title, body)\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    rank-profile bm25 inherits default {\n"
            "        first-phase {\n"
            "            expression {\n"
            "                bm25(title) + bm25(body)\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    rank-profile bert inherits default {\n"
            "        constants {\n"
            "            TOKEN_NONE: 0\n"
            "            TOKEN_CLS: 101\n"
            "            TOKEN_SEP: 102\n"
            "        }\n"
            "        function question_length() {\n"
            "            expression {\n"
            "                sum(map(query(query_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function doc_length() {\n"
            "            expression {\n"
            "                sum(map(attribute(doc_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function input_ids() {\n"
            "            expression {\n"
            "                tensor<float>(d0[1],d1[128])(\n"
            "                    if (d1 == 0,\n"
            "                        TOKEN_CLS,\n"
            "                    if (d1 < question_length + 1,\n"
            "                        query(query_token_ids){d0:(d1-1)},\n"
            "                    if (d1 == question_length + 1,\n"
            "                        TOKEN_SEP,\n"
            "                    if (d1 < question_length + doc_length + 2,\n"
            "                        attribute(doc_token_ids){d0:(d1-question_length-2)},\n"
            "                    if (d1 == question_length + doc_length + 2,\n"
            "                        TOKEN_SEP,\n"
            "                        TOKEN_NONE\n"
            "                    ))))))\n"
            "            }\n"
            "        }\n"
            "        function attention_mask() {\n"
            "            expression {\n"
            "                map(input_ids, f(a)(a > 0))\n"
            "            }\n"
            "        }\n"
            "        function token_type_ids() {\n"
            "            expression {\n"
            "                tensor<float>(d0[1],d1[128])(\n"
            "                    if (d1 < question_length,\n"
            "                        0,\n"
            "                    if (d1 < question_length + doc_length,\n"
            "                        1,\n"
            "                        TOKEN_NONE\n"
            "                    )))\n"
            "            }\n"
            "        }\n"
            "        first-phase {\n"
            "            expression {\n"
            "                bm25(title) + bm25(body)\n"
            "            }\n"
            "        }\n"
            "        second-phase {\n"
            "            rerank-count: 100\n"
            "            expression {\n"
            "                bm25(title)\n"
            "            }\n"
            "        }\n"
            "        global-phase {\n"
            "            rerank-count: 10\n"
            "            expression {\n"
            "                sum(onnx(bert).logits{d0:0,d1:0})\n"
            "            }\n"
            "        }\n"
            "        summary-features {\n"
            "            onnx(bert).logits\n"
            "            input_ids\n"
            "            attention_mask\n"
            "            token_type_ids\n"
            "        }\n"
            "    }\n"
            "}"
        )
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
        expected_result = (
            "schema testapp {\n"
            "    document testapp {\n"
            "        field id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "        field title type string {\n"
            "            indexing: index | summary\n"
            "            index: enable-bm25\n"
            "        }\n"
            "        field body type string {\n"
            "            indexing: index | summary\n"
            "            index: enable-bm25\n"
            "        }\n"
            "        field tensor_field type tensor<float>(x[128]) {\n"
            "            indexing: attribute\n"
            "            attribute {\n"
            "                distance-metric: euclidean\n"
            "                fast-search\n"
            "                fast-access\n"
            "            }\n"
            "            index {\n"
            "                hnsw {\n"
            "                    max-links-per-node: 16\n"
            "                    neighbors-to-explore-at-insert: 200\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    field embedding type tensor<bfloat16>(x[384]) {\n"
            '        indexing: (input title || "") . " " . (input body || "") | embed embedder | attribute | index\n'
            "        index: hnsw\n"
            "    }\n"
            "    fieldset default {\n"
            "        fields: title, body\n"
            "    }\n"
            "    rank-profile default {\n"
            "        first-phase {\n"
            "            expression {\n"
            "                nativeRank(title, body)\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    rank-profile bm25 inherits default {\n"
            "        first-phase {\n"
            "            expression {\n"
            "                bm25(title) + bm25(body)\n"
            "            }\n"
            "        }\n"
            "        global-phase {\n"
            "            rerank-count: 10\n"
            "            expression {\n"
            "                bm25(title)\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}"
        )
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
                        name="multiple_aliased_field",
                        type="string",
                        alias=[
                            "first_alias",
                            "second_alias",
                            "third_alias",
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
            "        field multiple_aliased_field type string {\n"
            "            alias: first_alias\n"
            "            alias: second_alias\n"
            "            alias: third_alias\n"
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
