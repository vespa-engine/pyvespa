import unittest
import os
from shutil import rmtree

import pytest

from vespa.package import (
    HNSW,
    Field,
    ImportedField,
    Document,
    FieldSet,
    Function,
    SecondPhaseRanking,
    RankProfile,
    OnnxModel,
    Schema,
    QueryTypeField,
    QueryProfileType,
    QueryField,
    QueryProfile,
    ApplicationPackage,
    ModelServer
)
from vespa.ml import BertModelConfig


class TestField(unittest.TestCase):
    def test_field_name_type(self):
        field = Field(name="test_name", type="string")
        self.assertEqual(field.name, "test_name")
        self.assertEqual(field.type, "string")
        self.assertEqual(field.to_dict, {"name": "test_name", "type": "string"})
        self.assertEqual(field, Field(name="test_name", type="string"))
        self.assertEqual(field, Field.from_dict(field.to_dict))
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
            field.to_dict,
            {
                "name": "body",
                "type": "string",
                "indexing": ["index", "summary"],
                "index": "enable-bm25",
            },
        )
        self.assertEqual(
            field,
            Field(
                name="body",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
        )
        self.assertEqual(field, Field.from_dict(field.to_dict))
        self.assertEqual(field.indexing_to_text, "index | summary")

    def test_tensor_with_hnsw(self):
        field = Field(
            name="tensor_field",
            type="tensor<float>(x[128])",
            indexing=["attribute"],
            attribute=["fast-search", "fast-access"],
            ann=HNSW(
                distance_metric="euclidean",
                max_links_per_node=16,
                neighbors_to_explore_at_insert=200,
            ),
        )
        self.assertEqual(field, Field.from_dict(field.to_dict))


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
        self.assertEqual(
            imported_field, ImportedField.from_dict(imported_field.to_dict)
        )


class TestDocument(unittest.TestCase):
    def test_empty_document(self):
        document = Document()
        self.assertEqual(document.fields, [])
        self.assertEqual(document.to_dict, {"fields": [], "inherits": None})
        self.assertEqual(document, Document.from_dict(document.to_dict))

    def test_document_one_field(self):
        document = Document(inherits="context")
        field = Field(name="test_name", type="string")
        document.add_fields(field)
        self.assertEqual(document.fields, [field])
        self.assertEqual(document, Document.from_dict(document.to_dict))
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
        self.assertEqual(document, Document.from_dict(document.to_dict))
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
        self.assertEqual(field_set, FieldSet.from_dict(field_set.to_dict))
        self.assertEqual(field_set.fields_to_text, "title, body")


class TestSecondPhaseRanking(unittest.TestCase):
    def test_second_phase_ranking(self):
        second_phase_ranking = SecondPhaseRanking(
            expression="sum(eval)", rerank_count=10
        )
        self.assertEqual(
            second_phase_ranking,
            SecondPhaseRanking.from_dict(second_phase_ranking.to_dict),
        )


class TestFunction(unittest.TestCase):
    def test_function_no_argument(self):
        function = Function(
            name="myfeature", expression="fieldMatch(title) + freshness(timestamp)"
        )
        self.assertEqual(function.name, "myfeature")
        self.assertEqual(
            function.expression, "fieldMatch(title) + freshness(timestamp)"
        )
        self.assertEqual(function, Function.from_dict(function.to_dict))
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
        self.assertEqual(function, Function.from_dict(function.to_dict))
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
        self.assertEqual(function, Function.from_dict(function.to_dict))
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
        self.assertEqual(function, Function.from_dict(function.to_dict))
        self.assertEqual(function.args_to_text, "")


class TestRankProfile(unittest.TestCase):
    def test_rank_profile(self):
        rank_profile = RankProfile(name="bm25", first_phase="bm25(title) + bm25(body)")
        self.assertEqual(rank_profile.name, "bm25")
        self.assertEqual(rank_profile.first_phase, "bm25(title) + bm25(body)")
        self.assertEqual(rank_profile, RankProfile.from_dict(rank_profile.to_dict))

    def test_rank_profile_inherits(self):
        rank_profile = RankProfile(
            name="bm25", first_phase="bm25(title) + bm25(body)", inherits="default"
        )
        self.assertEqual(rank_profile.name, "bm25")
        self.assertEqual(rank_profile.first_phase, "bm25(title) + bm25(body)")
        self.assertEqual(rank_profile, RankProfile.from_dict(rank_profile.to_dict))

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
        self.assertEqual(rank_profile, RankProfile.from_dict(rank_profile.to_dict))


class TestOnnxModel(unittest.TestCase):
    def test_onnx_model(self):
        onnx_model = OnnxModel(
            model_name="bert",
            model_file_path="bert.onnx",
            inputs={
                "input_ids": "input_ids",
                "token_type_ids": "token_type_ids",
                "attention_mask": "attention_mask",
            },
            outputs={"logits": "logits"},
        )
        self.assertEqual(
            onnx_model,
            OnnxModel.from_dict(onnx_model.to_dict),
        )


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

    def test_serialization(self):
        self.assertEqual(self.schema, Schema.from_dict(self.schema.to_dict))

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
        self.assertEqual(field.to_dict, {"name": "test_name", "type": "string"})
        self.assertEqual(field, QueryTypeField(name="test_name", type="string"))
        self.assertEqual(field, QueryTypeField.from_dict(field.to_dict))


class TestQueryProfileType(unittest.TestCase):
    def test_empty(self):
        query_profile_type = QueryProfileType()
        self.assertEqual(query_profile_type.fields, [])
        self.assertEqual(query_profile_type.to_dict, {"fields": []})
        self.assertEqual(
            query_profile_type, QueryProfileType.from_dict(query_profile_type.to_dict)
        )

    def test_one_field(self):
        query_profile_type = QueryProfileType()
        field = QueryTypeField(name="test_name", type="string")
        query_profile_type.add_fields(field)
        self.assertEqual(query_profile_type.fields, [field])
        self.assertEqual(
            query_profile_type, QueryProfileType.from_dict(query_profile_type.to_dict)
        )
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
        self.assertEqual(
            query_profile_type, QueryProfileType.from_dict(query_profile_type.to_dict)
        )
        self.assertEqual(query_profile_type, QueryProfileType([field_1, field_2]))


class TestQueryField(unittest.TestCase):
    def test_field_name_type(self):
        field = QueryField(name="test_name", value=1)
        self.assertEqual(field.name, "test_name")
        self.assertEqual(field.value, 1)
        self.assertEqual(field.to_dict, {"name": "test_name", "value": 1})
        self.assertEqual(field, QueryField(name="test_name", value=1))
        self.assertEqual(field, QueryField.from_dict(field.to_dict))


class TestQueryProfile(unittest.TestCase):
    def test_empty(self):
        query_profile = QueryProfile()
        self.assertEqual(query_profile.fields, [])
        self.assertEqual(query_profile.to_dict, {"fields": []})
        self.assertEqual(query_profile, QueryProfile.from_dict(query_profile.to_dict))

    def test_one_field(self):
        query_profile = QueryProfile()
        field = QueryField(name="test_name", value=2.0)
        query_profile.add_fields(field)
        self.assertEqual(query_profile.fields, [field])
        self.assertEqual(query_profile, QueryProfile.from_dict(query_profile.to_dict))
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
        self.assertEqual(query_profile, QueryProfile.from_dict(query_profile.to_dict))
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

    def test_application_package(self):
        self.assertEqual(
            self.app_package, ApplicationPackage.from_dict(self.app_package.to_dict)
        )

    def test_get_schema(self):
        self.assertEqual(self.app_package.schema, self.test_schema)
        self.assertEqual(self.app_package.schema, self.app_package.get_schema())

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
            "            expression: nativeRank(title, body)\n"
            "        }\n"
            "    }\n"
            "    rank-profile bm25 inherits default {\n"
            "        first-phase {\n"
            "            expression: bm25(title) + bm25(body)\n"
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
            "            expression: bm25(title) + bm25(body)\n"
            "        }\n"
            "        second-phase {\n"
            "            rerank-count: 10\n"
            "            expression: sum(onnx(bert).logits{d0:0,d1:0})\n"
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
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            '        <redundancy reply-after="1">1</redundancy>\n'
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

    def test_application_package(self):
        self.assertEqual(
            self.app_package, ApplicationPackage.from_dict(self.app_package.to_dict)
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
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            '        <redundancy reply-after="1">1</redundancy>\n'
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


class TestApplicationPackageAddBertRankingWithMultipleSchemas(unittest.TestCase):
    def setUp(self) -> None:
        news_schema = Schema(
            name="news",
            document=Document(
                fields=[
                    Field(
                        name="news_id", type="string", indexing=["attribute", "summary"]
                    ),
                ]
            ),
        )
        user_schema = Schema(
            name="user",
            document=Document(
                fields=[
                    Field(
                        name="user_id", type="string", indexing=["attribute", "summary"]
                    ),
                ]
            ),
        )
        self.app_package = ApplicationPackage(
            name="testapp",
            schema=[news_schema, user_schema],
        )
        bert_config = BertModelConfig(
            model_id="bert_tiny",
            query_input_size=4,
            doc_input_size=8,
            tokenizer=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_tokenizer"),
            model=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_model"),
        )
        self.app_package.add_model_ranking(
            model_config=bert_config,
            schema="news",
            include_model_summary_features=True,
            inherits="default",
            first_phase="bm25(title)",
            second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
        )

        self.disk_folder = "saved_app"

    def test_application_package(self):
        self.assertEqual(
            self.app_package, ApplicationPackage.from_dict(self.app_package.to_dict)
        )

    def test_news_schema_to_text(self):
        expected_result = (
            "schema news {\n"
            "    document news {\n"
            "        field news_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "        field bert_tiny_doc_token_ids type tensor<float>(d0[7]) {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "    onnx-model bert_tiny {\n"
            "        file: files/bert_tiny.onnx\n"
            "        input input_ids: input_ids\n"
            "        input token_type_ids: token_type_ids\n"
            "        input attention_mask: attention_mask\n"
            "        output output_0: logits\n"
            "    }\n"
            "    rank-profile bert_tiny inherits default {\n"
            "        constants {\n"
            "            TOKEN_NONE: 0\n"
            "            TOKEN_CLS: 101\n"
            "            TOKEN_SEP: 102\n"
            "        }\n"
            "        function question_length() {\n"
            "            expression {\n"
            "                sum(map(query(bert_tiny_query_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function doc_length() {\n"
            "            expression {\n"
            "                sum(map(attribute(bert_tiny_doc_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function input_ids() {\n"
            "            expression {\n"
            "                tokenInputIds(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function attention_mask() {\n"
            "            expression {\n"
            "                tokenAttentionMask(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function token_type_ids() {\n"
            "            expression {\n"
            "                tokenTypeIds(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function logit0() {\n"
            "            expression {\n"
            "                onnx(bert_tiny).logits{d0:0,d1:0}\n"
            "            }\n"
            "        }\n"
            "        function logit1() {\n"
            "            expression {\n"
            "                onnx(bert_tiny).logits{d0:0,d1:1}\n"
            "            }\n"
            "        }\n"
            "        first-phase {\n"
            "            expression: bm25(title)\n"
            "        }\n"
            "        second-phase {\n"
            "            rerank-count: 10\n"
            "            expression: logit1\n"
            "        }\n"
            "        summary-features {\n"
            "            logit0\n"
            "            logit1\n"
            "            input_ids\n"
            "            attention_mask\n"
            "            token_type_ids\n"
            "        }\n"
            "    }\n"
            "}"
        )
        self.assertEqual(
            self.app_package.get_schema("news").schema_to_text, expected_result
        )

    def test_user_schema_to_text(self):
        expected_user_result = (
            "schema user {\n"
            "    document user {\n"
            "        field user_id type string {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "}"
        )
        self.assertEqual(
            self.app_package.get_schema("user").schema_to_text, expected_user_result
        )

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testapp_container" version="1.0">\n'
            "        <search></search>\n"
            "        <document-api></document-api>\n"
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            '        <redundancy reply-after="1">1</redundancy>\n'
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

    def test_query_profile_to_text(self):
        expected_result = (
            '<query-profile id="default" type="root">\n' "</query-profile>"
        )

        self.assertEqual(self.app_package.query_profile_to_text, expected_result)

    def test_query_profile_type_to_text(self):
        expected_result = (
            '<query-profile-type id="root">\n'
            '    <field name="ranking.features.query(bert_tiny_query_token_ids)" type="tensor&lt;float&gt;(d0[2])" />\n'
            "</query-profile-type>"
        )
        self.assertEqual(self.app_package.query_profile_type_to_text, expected_result)

    def test_save_load(self):
        self.app_package.save(disk_folder=self.disk_folder)
        self.assertEqual(
            self.app_package, ApplicationPackage.load(disk_folder=self.disk_folder)
        )

    def tearDown(self) -> None:
        rmtree(self.disk_folder, ignore_errors=True)


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

    def test_application_package(self):
        self.assertEqual(
            self.app_package, ApplicationPackage.from_dict(self.app_package.to_dict)
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
            "    fieldset default {\n"
            "        fields: title, body\n"
            "    }\n"
            "    rank-profile default {\n"
            "        first-phase {\n"
            "            expression: nativeRank(title, body)\n"
            "        }\n"
            "    }\n"
            "    rank-profile bm25 inherits default {\n"
            "        first-phase {\n"
            "            expression: bm25(title) + bm25(body)\n"
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
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            '        <redundancy reply-after="1">1</redundancy>\n'
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

    def test_application_package(self):
        self.assertEqual(
            self.app_package, ApplicationPackage.from_dict(self.app_package.to_dict)
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
            "    </container>\n"
            '    <content id="news_content" version="1.0">\n'
            '        <redundancy reply-after="1">1</redundancy>\n'
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


class TestSimplifiedApplicationPackageAddBertRanking(unittest.TestCase):
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

        bert_config = BertModelConfig(
            model_id="bert_tiny",
            query_input_size=4,
            doc_input_size=8,
            tokenizer=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_tokenizer"),
            model=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_model"),
        )

        self.app_package.add_model_ranking(
            model_config=bert_config,
            include_model_summary_features=True,
            inherits="default",
            first_phase="bm25(title)",
            second_phase=SecondPhaseRanking(rerank_count=10, expression="logit1"),
        )
        self.disk_folder = "saved_app"

    def test_application_package(self):
        self.assertEqual(
            self.app_package, ApplicationPackage.from_dict(self.app_package.to_dict)
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
            "        field bert_tiny_doc_token_ids type tensor<float>(d0[7]) {\n"
            "            indexing: attribute | summary\n"
            "        }\n"
            "    }\n"
            "    fieldset default {\n"
            "        fields: title, body\n"
            "    }\n"
            "    onnx-model bert_tiny {\n"
            "        file: files/bert_tiny.onnx\n"
            "        input input_ids: input_ids\n"
            "        input token_type_ids: token_type_ids\n"
            "        input attention_mask: attention_mask\n"
            "        output output_0: logits\n"
            "    }\n"
            "    rank-profile default {\n"
            "        first-phase {\n"
            "            expression: nativeRank(title, body)\n"
            "        }\n"
            "    }\n"
            "    rank-profile bm25 inherits default {\n"
            "        first-phase {\n"
            "            expression: bm25(title) + bm25(body)\n"
            "        }\n"
            "    }\n"
            "    rank-profile bert_tiny inherits default {\n"
            "        constants {\n"
            "            TOKEN_NONE: 0\n"
            "            TOKEN_CLS: 101\n"
            "            TOKEN_SEP: 102\n"
            "        }\n"
            "        function question_length() {\n"
            "            expression {\n"
            "                sum(map(query(bert_tiny_query_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function doc_length() {\n"
            "            expression {\n"
            "                sum(map(attribute(bert_tiny_doc_token_ids), f(a)(a > 0)))\n"
            "            }\n"
            "        }\n"
            "        function input_ids() {\n"
            "            expression {\n"
            "                tokenInputIds(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function attention_mask() {\n"
            "            expression {\n"
            "                tokenAttentionMask(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function token_type_ids() {\n"
            "            expression {\n"
            "                tokenTypeIds(12, query(bert_tiny_query_token_ids), attribute(bert_tiny_doc_token_ids))\n"
            "            }\n"
            "        }\n"
            "        function logit0() {\n"
            "            expression {\n"
            "                onnx(bert_tiny).logits{d0:0,d1:0}\n"
            "            }\n"
            "        }\n"
            "        function logit1() {\n"
            "            expression {\n"
            "                onnx(bert_tiny).logits{d0:0,d1:1}\n"
            "            }\n"
            "        }\n"
            "        first-phase {\n"
            "            expression: bm25(title)\n"
            "        }\n"
            "        second-phase {\n"
            "            rerank-count: 10\n"
            "            expression: logit1\n"
            "        }\n"
            "        summary-features {\n"
            "            logit0\n"
            "            logit1\n"
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
            "    </container>\n"
            '    <content id="testapp_content" version="1.0">\n'
            '        <redundancy reply-after="1">1</redundancy>\n'
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
            '    <field name="ranking.features.query(bert_tiny_query_token_ids)" type="tensor&lt;float&gt;(d0[2])" />\n'
            "</query-profile-type>"
        )
        self.assertEqual(self.app_package.query_profile_type_to_text, expected_result)

    def test_save_load(self):
        self.app_package.save(disk_folder=self.disk_folder)
        self.assertEqual(
            self.app_package, ApplicationPackage.load(disk_folder=self.disk_folder)
        )

    def tearDown(self) -> None:
        rmtree(self.disk_folder, ignore_errors=True)


class TestValidAppName(unittest.TestCase):

    def test_invalid_name(self):
        with pytest.raises(ValueError):
            app_package = ApplicationPackage(name="test_app")
        with pytest.raises(ValueError):
            app_package = ApplicationPackage(name="test-app")


class TestModelServer(unittest.TestCase):
    def setUp(self) -> None:
        self.server_name = "testserver"
        self.model_server = ModelServer(
            name=self.server_name,
        )

    def test_model_server_serialization(self):
        self.assertEqual(
            self.model_server, ModelServer.from_dict(self.model_server.to_dict)
        )

    def test_get_schema(self):
        self.assertIsNone(self.model_server.schema)
        self.assertEqual(self.model_server.schema, self.model_server.get_schema())

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="testserver_container" version="1.0">\n'
            "        <model-evaluation/>\n"
            "    </container>\n"
            "</services>"
        )

        self.assertEqual(self.model_server.services_to_text, expected_result)

    def test_query_profile_to_text(self):
        expected_result = (
            '<query-profile id="default" type="root">\n' "</query-profile>"
        )
        self.assertEqual(self.model_server.query_profile_to_text, expected_result)

    def test_query_profile_type_to_text(self):
        expected_result = '<query-profile-type id="root">\n' "</query-profile-type>"
        self.assertEqual(self.model_server.query_profile_type_to_text, expected_result)
