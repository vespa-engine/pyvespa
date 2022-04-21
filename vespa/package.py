import os
import zipfile

from pathlib import Path
from shutil import copyfile
from typing import List, Mapping, Optional, Union, Dict
from collections import OrderedDict
from jinja2 import Environment, PackageLoader, select_autoescape
from io import BytesIO

from vespa.json_serialization import ToJson, FromJson
from vespa.query import QueryModel


class HNSW(ToJson, FromJson["HNSW"]):
    def __init__(
        self,
        distance_metric="euclidean",
        max_links_per_node=16,
        neighbors_to_explore_at_insert=200,
    ):
        """
        Configure Vespa HNSW indexes

        :param distance_metric: Distance metric to use when computing distance between vectors. Default is 'euclidean'.
        :param max_links_per_node: Specifies how many links per HNSW node to select when building the graph.
            Default is 16.
        :param neighbors_to_explore_at_insert: Specifies how many neighbors to explore when inserting a document in
            the HNSW graph. Default is 200.
        """
        self.distance_metric = distance_metric
        self.max_links_per_node = max_links_per_node
        self.neighbors_to_explore_at_insert = neighbors_to_explore_at_insert

    @staticmethod
    def from_dict(mapping: Mapping) -> "HNSW":
        return HNSW(
            distance_metric=mapping["distance_metric"],
            max_links_per_node=mapping["max_links_per_node"],
            neighbors_to_explore_at_insert=mapping["neighbors_to_explore_at_insert"],
        )

    @property
    def to_dict(self) -> Mapping:
        map = {
            "distance_metric": self.distance_metric,
            "max_links_per_node": self.max_links_per_node,
            "neighbors_to_explore_at_insert": self.neighbors_to_explore_at_insert,
        }
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.distance_metric == other.distance_metric
            and self.max_links_per_node == other.max_links_per_node
            and self.neighbors_to_explore_at_insert
            == other.neighbors_to_explore_at_insert
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.distance_metric),
            repr(self.max_links_per_node),
            repr(self.neighbors_to_explore_at_insert),
        )


class Field(ToJson, FromJson["Field"]):
    def __init__(
        self,
        name: str,
        type: str,
        indexing: Optional[List[str]] = None,
        index: Optional[str] = None,
        attribute: Optional[List[str]] = None,
        ann: Optional[HNSW] = None,
    ) -> None:
        """
        Create a Vespa field.

        Check the `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#field>`__
        for more detailed information about fields.

        Once we have an :class:`ApplicationPackage` instance containing a :class:`Schema` and a :class:`Document`,
        we usually want to add fields so that we can store our data in a structured manner.
        We can accomplish that by creating :class:`Field` instances
        and adding those to the :class:`ApplicationPackage` instance via :class:`Schema` and :class:`Document` methods.

        :param name: Field name.
        :param type: Field data type.
        :param indexing: Configures how to process data of a field during indexing.
        :param index: Sets index parameters. Content in fields with index are normalized and tokenized by default.
        :param attribute:  Specifies a property of an index structure attribute.
        :param ann: Add configuration for approximate nearest neighbor.

        >>> Field(name = "title", type = "string", indexing = ["index", "summary"], index = "enable-bm25")
        Field('title', 'string', ['index', 'summary'], 'enable-bm25', None, None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     indexing = ["attribute"],
        ...     attribute=["fast-search", "fast-access"]
        ... )
        Field('abstract', 'string', ['attribute'], None, ['fast-search', 'fast-access'], None)

        >>> Field(name="tensor_field",
        ...     type="tensor<float>(x[128])",
        ...     indexing=["attribute"],
        ...     ann=HNSW(
        ...         distance_metric="euclidean",
        ...         max_links_per_node=16,
        ...         neighbors_to_explore_at_insert=200,
        ...     ),
        ... )
        Field('tensor_field', 'tensor<float>(x[128])', ['attribute'], None, None, HNSW('euclidean', 16, 200))

        """
        self.name = name
        self.type = type
        self.indexing = indexing
        self.attribute = attribute
        self.index = index
        self.ann = ann

    @property
    def indexing_to_text(self) -> Optional[str]:
        if self.indexing is not None:
            return " | ".join(self.indexing)

    @staticmethod
    def from_dict(mapping: Mapping) -> "Field":
        ann = mapping.get("ann", None)
        return Field(
            name=mapping["name"],
            type=mapping["type"],
            indexing=mapping.get("indexing", None),
            index=mapping.get("index", None),
            attribute=mapping.get("attribute", None),
            ann=FromJson.map(ann) if ann is not None else None,
        )

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name, "type": self.type}
        if self.indexing is not None:
            map.update(indexing=self.indexing)
        if self.index is not None:
            map.update(index=self.index)
        if self.attribute is not None:
            map.update(attribute=self.attribute)
        if self.ann is not None:
            map.update(ann=self.ann.to_envelope)
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.type == other.type
            and self.indexing == other.indexing
            and self.index == other.index
            and self.attribute == other.attribute
            and self.ann == other.ann
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.type),
            repr(self.indexing),
            repr(self.index),
            repr(self.attribute),
            repr(self.ann),
        )


class ImportedField(ToJson, FromJson["ImportedField"]):
    def __init__(
        self,
        name: str,
        reference_field: str,
        field_to_import: str,
    ) -> None:
        """
        Imported field from a reference document.

        Useful to implement `parent/child relationships <https://docs.vespa.ai/en/parent-child.html>`.

        :param name: Field name.
        :param reference_field: field of type reference that points to the document that contains the field to be
            imported.
        :param field_to_import: Field name to be imported, as defined in the reference document.

        >>> ImportedField(
        ...     name="global_category_ctrs",
        ...     reference_field="category_ctr_ref",
        ...     field_to_import="ctrs",
        ... )
        ImportedField('global_category_ctrs', 'category_ctr_ref', 'ctrs')

        """
        self.name = name
        self.reference_field = reference_field
        self.field_to_import = field_to_import

    @staticmethod
    def from_dict(mapping: Mapping) -> "ImportedField":
        return ImportedField(
            name=mapping["name"],
            reference_field=mapping["reference_field"],
            field_to_import=mapping["field_to_import"],
        )

    @property
    def to_dict(self) -> Mapping:
        map = {
            "name": self.name,
            "reference_field": self.reference_field,
            "field_to_import": self.field_to_import,
        }
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.reference_field == other.reference_field
            and self.field_to_import == other.field_to_import
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.reference_field),
            repr(self.field_to_import),
        )


class Document(ToJson, FromJson["Document"]):
    def __init__(
        self, fields: Optional[List[Field]] = None, inherits: Optional[str] = None
    ) -> None:
        """
        Create a Vespa Document.

        Check the `Vespa documentation <https://docs.vespa.ai/en/documents.html>`__
        for more detailed information about documents.

        :param fields: A list of :class:`Field` to include in the document's schema.

        To create a Document:

        >>> Document()
        Document(None, None)

        >>> Document(fields=[Field(name="title", type="string")])
        Document([Field('title', 'string', None, None, None, None)], None)

        >>> Document(fields=[Field(name="title", type="string")], inherits="context")
        Document([Field('title', 'string', None, None, None, None)], context)
        """
        self.inherits = inherits
        self._fields = (
            OrderedDict()
            if not fields
            else OrderedDict([(field.name, field) for field in fields])
        )

    @property
    def fields(self):
        return [x for x in self._fields.values()]

    def add_fields(self, *fields: Field) -> None:
        """
        Add :class:`Field`'s to the document.

        :param fields: fields to be added
        :return:
        """
        for field in fields:
            self._fields.update({field.name: field})

    @staticmethod
    def from_dict(mapping: Mapping) -> "Document":
        return Document(
            fields=[FromJson.map(field) for field in mapping.get("fields")],
            inherits=mapping.get("inherits", None),
        )

    @property
    def to_dict(self) -> Mapping:
        map = {
            "fields": [field.to_envelope for field in self.fields],
            "inherits": self.inherits,
        }
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.fields == other.fields and self.inherits == other.inherits

    def __repr__(self):
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.fields) if self.fields else None,
            self.inherits,
        )


class FieldSet(ToJson, FromJson["FieldSet"]):
    def __init__(self, name: str, fields: List[str]) -> None:
        """
        Create a Vespa field set.

        A fieldset groups fields together for searching. Check the
        `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#fieldset>`__
        for more detailed information about field sets.

        :param name: Name of the fieldset
        :param fields: Field names to be included in the fieldset.

        >>> FieldSet(name="default", fields=["title", "body"])
        FieldSet('default', ['title', 'body'])
        """
        self.name = name
        self.fields = fields

    @property
    def fields_to_text(self):
        if self.fields is not None:
            return ", ".join(self.fields)

    @staticmethod
    def from_dict(mapping: Mapping) -> "FieldSet":
        return FieldSet(name=mapping["name"], fields=mapping["fields"])

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name, "fields": self.fields}
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.fields == other.fields

    def __repr__(self):
        return "{0}({1}, {2})".format(
            self.__class__.__name__, repr(self.name), repr(self.fields)
        )


class Function(ToJson, FromJson["Function"]):
    def __init__(
        self, name: str, expression: str, args: Optional[List[str]] = None
    ) -> None:
        r"""
        Create a Vespa rank function.

        Define a named function that can be referenced as a part of the ranking expression, or (if having no arguments)
        as a feature. Check the
        `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#function-rank>`__`
        for more detailed information about rank functions.

        :param name: Name of the function.
        :param expression: String representing a Vespa expression.
        :param args: Optional. List of arguments to be used in the function expression.

        >>> Function(
        ...     name="myfeature",
        ...     expression="fieldMatch(bar) + freshness(foo)",
        ...     args=["foo", "bar"]
        ... )
        Function('myfeature', 'fieldMatch(bar) + freshness(foo)', ['foo', 'bar'])

        It is possible to define functions with multi-line expressions:

        >>> Function(
        ...     name="token_type_ids",
        ...     expression="tensor<float>(d0[1],d1[128])(\n"
        ...                "    if (d1 < question_length,\n"
        ...                "        0,\n"
        ...                "    if (d1 < question_length + doc_length,\n"
        ...                "        1,\n"
        ...                "        TOKEN_NONE\n"
        ...                "    )))",
        ... )
        Function('token_type_ids', 'tensor<float>(d0[1],d1[128])(\n    if (d1 < question_length,\n        0,\n    if (d1 < question_length + doc_length,\n        1,\n        TOKEN_NONE\n    )))', None)
        """
        self.name = name
        self.args = args
        self.expression = expression

    @property
    def args_to_text(self) -> str:
        if self.args is not None:
            return ", ".join(self.args)
        else:
            return ""

    @staticmethod
    def from_dict(mapping: Mapping) -> "Function":
        return Function(
            name=mapping["name"], expression=mapping["expression"], args=mapping["args"]
        )

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name, "expression": self.expression, "args": self.args}
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.expression == other.expression
            and self.args == other.args
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.expression),
            repr(self.args),
        )


class SecondPhaseRanking(ToJson, FromJson["SecondPhaseRanking"]):
    def __init__(self, expression: str, rerank_count: int = 100) -> None:
        r"""
        Create a Vespa second phase ranking configuration.

        This is the optional reranking performed on the best hits from the first phase. Check the
        `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#secondphase-rank>`__`
        for more detailed information about second phase ranking configuration.

        :param expression: Specify the ranking expression to be used for second phase of ranking. Check also the
            `Vespa documentation <https://docs.vespa.ai/en/reference/ranking-expressions.html>`__`
            for ranking expression.
        :param rerank_count: Specifies the number of hits to be reranked in the second phase. Default value is 100.

        >>> SecondPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10)
        SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10)
        """
        self.expression = expression
        self.rerank_count = rerank_count

    @staticmethod
    def from_dict(mapping: Mapping) -> "SecondPhaseRanking":
        return SecondPhaseRanking(
            expression=mapping["expression"], rerank_count=mapping["rerank_count"]
        )

    @property
    def to_dict(self) -> Mapping:
        map = {"expression": self.expression, "rerank_count": self.rerank_count}
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.expression == other.expression
            and self.rerank_count == other.rerank_count
        )

    def __repr__(self):
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.expression),
            repr(self.rerank_count),
        )


class RankProfile(ToJson, FromJson["RankProfile"]):
    def __init__(
        self,
        name: str,
        first_phase: str,
        inherits: Optional[str] = None,
        constants: Optional[Dict] = None,
        functions: Optional[List[Function]] = None,
        summary_features: Optional[List] = None,
        second_phase: Optional[SecondPhaseRanking] = None,
    ) -> None:
        """
        Create a Vespa rank profile.

        Rank profiles are used to specify an alternative ranking of the same data for different purposes, and to
        experiment with new rank settings. Check the
        `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#rank-profile>`__
        for more detailed information about rank profiles.

        :param name: Rank profile name.
        :param first_phase: The config specifying the first phase of ranking.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#firstphase-rank>`__`
            about first phase ranking.
        :param inherits: The inherits attribute is optional. If defined, it contains the name of one other
            rank profile in the same schema. Values not defined in this rank profile will then be inherited.
        :param constants: Dict of constants available in ranking expressions, resolved and optimized at
            configuration time.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#constants>`__`
            about constants.
        :param functions: Optional list of :class:`Function` representing rank functions to be included in the rank
            profile.
        :param summary_features: List of rank features to be included with each hit.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#summary-features>`__`
            about summary features.
        :param second_phase: Optional config specifying the second phase of ranking.
            See :class:`SecondPhaseRanking`.

        >>> RankProfile(name = "default", first_phase = "nativeRank(title, body)")
        RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None)

        >>> RankProfile(name = "new", first_phase = "BM25(title)", inherits = "default")
        RankProfile('new', 'BM25(title)', 'default', None, None, None, None)

        >>> RankProfile(
        ...     name = "new",
        ...     first_phase = "BM25(title)",
        ...     inherits = "default",
        ...     constants={"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102},
        ...     summary_features=["BM25(title)"]
        ... )
        RankProfile('new', 'BM25(title)', 'default', {'TOKEN_NONE': 0, 'TOKEN_CLS': 101, 'TOKEN_SEP': 102}, None, ['BM25(title)'], None)

        >>> RankProfile(
        ...     name="bert",
        ...     first_phase="bm25(title) + bm25(body)",
        ...     second_phase=SecondPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10),
        ...     inherits="default",
        ...     constants={"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102},
        ...     functions=[
        ...         Function(
        ...             name="question_length",
        ...             expression="sum(map(query(query_token_ids), f(a)(a > 0)))"
        ...         ),
        ...         Function(
        ...             name="doc_length",
        ...             expression="sum(map(attribute(doc_token_ids), f(a)(a > 0)))"
        ...         )
        ...     ],
        ...     summary_features=["question_length", "doc_length"]
        ... )
        RankProfile('bert', 'bm25(title) + bm25(body)', 'default', {'TOKEN_NONE': 0, 'TOKEN_CLS': 101, 'TOKEN_SEP': 102}, [Function('question_length', 'sum(map(query(query_token_ids), f(a)(a > 0)))', None), Function('doc_length', 'sum(map(attribute(doc_token_ids), f(a)(a > 0)))', None)], ['question_length', 'doc_length'], SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10))
        """
        self.name = name
        self.first_phase = first_phase
        self.inherits = inherits
        self.constants = constants
        self.functions = functions
        self.summary_features = summary_features
        self.second_phase = second_phase

    @staticmethod
    def from_dict(mapping: Mapping) -> "RankProfile":
        functions = mapping.get("functions", None)
        if functions is not None:
            functions = [FromJson.map(f) for f in functions]
        second_phase = mapping.get("second_phase", None)
        if second_phase is not None:
            second_phase = FromJson.map(second_phase)

        return RankProfile(
            name=mapping["name"],
            first_phase=mapping["first_phase"],
            inherits=mapping.get("inherits", None),
            constants=mapping.get("constants", None),
            functions=functions,
            summary_features=mapping.get("summary_features", None),
            second_phase=second_phase,
        )

    @property
    def to_dict(self) -> Mapping:
        map = {
            "name": self.name,
            "first_phase": self.first_phase,
        }
        if self.inherits is not None:
            map.update({"inherits": self.inherits})
        if self.constants is not None:
            map.update({"constants": self.constants})
        if self.functions is not None:
            map.update({"functions": [f.to_envelope for f in self.functions]})
        if self.summary_features is not None:
            map.update({"summary_features": self.summary_features})
        if self.second_phase is not None:
            map.update({"second_phase": self.second_phase.to_envelope})

        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.first_phase == other.first_phase
            and self.inherits == other.inherits
            and self.constants == other.constants
            and self.functions == other.functions
            and self.summary_features == other.summary_features
            and self.second_phase == other.second_phase
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.first_phase),
            repr(self.inherits),
            repr(self.constants),
            repr(self.functions),
            repr(self.summary_features),
            repr(self.second_phase),
        )


class OnnxModel(ToJson, FromJson["OnnxModel"]):
    def __init__(
        self,
        model_name: str,
        model_file_path: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
    ) -> None:
        """
        Create a Vespa ONNX model config.

        Vespa has support for advanced ranking models through it’s tensor API. If you have your model in the ONNX
        format, Vespa can import the models and use them directly. Check the
        `Vespa documentation <https://docs.vespa.ai/en/onnx.html>`__`
        for more detailed information about field sets.

        :param model_name: Unique model name to use as id when referencing the model.
        :param model_file_path: ONNX model file path.
        :param inputs: Dict mapping the ONNX input names as specified in the ONNX file to valid Vespa inputs,
            which can be a document field (`attribute(field_name)`), a query parameter (`query(query_param)`),
            a constant (`constant(name)`) and a user-defined function (`function_name`).
        :param outputs: Dict mapping the ONNX output names as specified in the ONNX file to the name used in Vespa to
            specify the output. If this is omitted, the first output in the ONNX file will be used.

        >>> OnnxModel(
        ...     model_name="bert",
        ...     model_file_path="bert.onnx",
        ...     inputs={
        ...         "input_ids": "input_ids",
        ...         "token_type_ids": "token_type_ids",
        ...         "attention_mask": "attention_mask",
        ...     },
        ...     outputs={"logits": "logits"},
        ... )
        OnnxModel('bert', 'bert.onnx', {'input_ids': 'input_ids', 'token_type_ids': 'token_type_ids', 'attention_mask': 'attention_mask'}, {'logits': 'logits'})
        """
        self.model_name = model_name
        self.model_file_path = model_file_path
        self.inputs = inputs
        self.outputs = outputs

        self.model_file_name = self.model_name + ".onnx"
        self.file_path = os.path.join("files", self.model_file_name)

    @staticmethod
    def from_dict(mapping: Mapping) -> "OnnxModel":
        return OnnxModel(
            model_name=mapping["model_name"],
            model_file_path=mapping["model_file_path"],
            inputs=mapping["inputs"],
            outputs=mapping["outputs"],
        )

    @property
    def to_dict(self) -> Mapping:
        map = {
            "model_name": self.model_name,
            "model_file_path": self.model_file_path,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.model_name == other.model_name
            and self.model_file_path == other.model_file_path
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4})".format(
            self.__class__.__name__,
            repr(self.model_name),
            repr(self.model_file_path),
            repr(self.inputs),
            repr(self.outputs),
        )


class Schema(ToJson, FromJson["Schema"]):
    def __init__(
        self,
        name: str,
        document: Document,
        fieldsets: Optional[List[FieldSet]] = None,
        rank_profiles: Optional[List[RankProfile]] = None,
        models: Optional[List[OnnxModel]] = None,
        global_document: bool = False,
        imported_fields: Optional[List[ImportedField]] = None,
    ) -> None:
        """
        Create a Vespa Schema.

        Check the `Vespa documentation <https://docs.vespa.ai/en/schemas.html>`__
        for more detailed information about schemas.

        :param name: Schema name.
        :param document: Vespa :class:`Document` associated with the Schema.
        :param fieldsets: A list of :class:`FieldSet` associated with the Schema.
        :param rank_profiles: A list of :class:`RankProfile` associated with the Schema.
        :param models: A list of :class:`OnnxModel` associated with the Schema.
        :param global_document: Set to True to copy the documents to all content nodes. Default to False.
        :param imported_fields: A list of :class:`ImportedField` defining fields from global documents to be imported.

        To create a Schema:

        >>> Schema(name="schema_name", document=Document())
        Schema('schema_name', Document(None, None), None, None, [], False, None)
        """
        self.name = name
        self.document = document
        self.global_document = global_document

        self.fieldsets = {}
        if fieldsets is not None:
            self.fieldsets = {fieldset.name: fieldset for fieldset in fieldsets}

        self.imported_fields = {}
        if imported_fields is not None:
            self.imported_fields = {
                imported_field.name: imported_field
                for imported_field in imported_fields
            }

        self.rank_profiles = {}
        if rank_profiles is not None:
            self.rank_profiles = {
                rank_profile.name: rank_profile for rank_profile in rank_profiles
            }

        self.models = [] if models is None else list(models)

    def add_fields(self, *fields: Field) -> None:
        """
        Add :class:`Field` to the Schema's :class:`Document`.

        :param fields: fields to be added.
        """
        self.document.add_fields(*fields)

    def add_field_set(self, field_set: FieldSet) -> None:
        """
        Add a :class:`FieldSet` to the Schema.

        :param field_set: field sets to be added.
        """
        self.fieldsets[field_set.name] = field_set

    def add_rank_profile(self, rank_profile: RankProfile) -> None:
        """
        Add a :class:`RankProfile` to the Schema.

        :param rank_profile: rank profile to be added.
        :return: None.
        """
        self.rank_profiles[rank_profile.name] = rank_profile

    def add_model(self, model: OnnxModel) -> None:
        """
        Add a :class:`OnnxModel` to the Schema.
        :param model: model to be added.
        :return: None.
        """
        self.models.append(model)

    def add_imported_field(self, imported_field: ImportedField) -> None:
        """
        Add a :class:`ImportedField` to the Schema.

        :param imported_field: imported field to be added.
        """
        self.imported_fields[imported_field.name] = imported_field

    @property
    def schema_to_text(self):
        env = Environment(
            loader=PackageLoader("vespa", "templates"),
            autoescape=select_autoescape(
                disabled_extensions=("txt",),
                default_for_string=True,
                default=True,
            ),
        )
        env.trim_blocks = True
        env.lstrip_blocks = True
        schema_template = env.get_template("schema.txt")
        return schema_template.render(
            schema_name=self.name,
            document_name=self.name,
            document=self.document,
            fieldsets=self.fieldsets,
            rank_profiles=self.rank_profiles,
            models=self.models,
            imported_fields=self.imported_fields,
        )

    @staticmethod
    def from_dict(mapping: Mapping) -> "Schema":
        fieldsets = mapping.get("fieldsets", None)
        if fieldsets:
            fieldsets = [FromJson.map(fieldset) for fieldset in mapping["fieldsets"]]
        rank_profiles = mapping.get("rank_profiles", None)
        if rank_profiles:
            rank_profiles = [
                FromJson.map(rank_profile) for rank_profile in mapping["rank_profiles"]
            ]
        models = mapping.get("models", None)
        if models:
            models = [FromJson.map(model) for model in mapping["models"]]
        imported_fields = mapping.get("imported_fields", None)
        if imported_fields:
            imported_fields = [
                FromJson.map(imported_field) for imported_field in imported_fields
            ]

        return Schema(
            name=mapping["name"],
            document=FromJson.map(mapping["document"]),
            fieldsets=fieldsets,
            rank_profiles=rank_profiles,
            models=models,
            global_document=mapping["global_document"],
            imported_fields=imported_fields,
        )

    @property
    def to_dict(self) -> Mapping:
        map = {
            "name": self.name,
            "document": self.document.to_envelope,
            "global_document": self.global_document,
        }
        if self.fieldsets:
            map["fieldsets"] = [
                self.fieldsets[name].to_envelope for name in self.fieldsets.keys()
            ]
        if self.rank_profiles:
            map["rank_profiles"] = [
                self.rank_profiles[name].to_envelope
                for name in self.rank_profiles.keys()
            ]
        if self.models:
            map["models"] = [model.to_envelope for model in self.models]
        if self.imported_fields:
            map["imported_fields"] = [
                self.imported_fields[name].to_envelope
                for name in self.imported_fields.keys()
            ]
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.document == other.document
            and self.fieldsets == other.fieldsets
            and self.rank_profiles == other.rank_profiles
            and self.models == other.models
            and self.global_document == other.global_document
            and self.imported_fields == other.imported_fields
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.document),
            repr(
                [field for field in self.fieldsets.values()] if self.fieldsets else None
            ),
            repr(
                [rank_profile for rank_profile in self.rank_profiles.values()]
                if self.rank_profiles
                else None
            ),
            repr(self.models),
            repr(self.global_document),
            repr(
                [imported_field for imported_field in self.imported_fields.values()]
                if self.imported_fields
                else None
            ),
        )


class QueryTypeField(ToJson, FromJson["QueryTypeField"]):
    def __init__(
        self,
        name: str,
        type: str,
    ) -> None:
        """
        Create a field to be included in a :class:`QueryProfileType`.

        :param name: Field name.
        :param type: Field type.

        >>> QueryTypeField(
        ...     name="ranking.features.query(title_bert)",
        ...     type="tensor<float>(x[768])"
        ... )
        QueryTypeField('ranking.features.query(title_bert)', 'tensor<float>(x[768])')
        """
        self.name = name
        self.type = type

    @staticmethod
    def from_dict(mapping: Mapping) -> "QueryTypeField":
        return QueryTypeField(
            name=mapping["name"],
            type=mapping["type"],
        )

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name, "type": self.type}
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.type == other.type

    def __repr__(self):
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.type),
        )


class QueryProfileType(ToJson, FromJson["QueryProfileType"]):
    def __init__(self, fields: Optional[List[QueryTypeField]] = None) -> None:
        """
        Create a Vespa Query Profile Type.

        Check the `Vespa documentation <https://docs.vespa.ai/en/query-profiles.html#query-profile-types>`__
        for more detailed information about query profile types.

        An :class:`ApplicationPackage` instance comes with a default :class:`QueryProfile` named `default`
        that is associated with a :class:`QueryProfileType` named `root`,
        meaning that you usually do not need to create those yourself, only add fields to them when required.

        :param fields: A list of :class:`QueryTypeField`.

        >>> QueryProfileType(
        ...     fields = [
        ...         QueryTypeField(
        ...             name="ranking.features.query(tensor_bert)",
        ...             type="tensor<float>(x[768])"
        ...         )
        ...     ]
        ... )
        QueryProfileType([QueryTypeField('ranking.features.query(tensor_bert)', 'tensor<float>(x[768])')])
        """
        self.name = "root"
        self.fields = [] if not fields else fields

    def add_fields(self, *fields: QueryTypeField) -> None:
        """
        Add :class:`QueryTypeField`'s to the Query Profile Type.

        :param fields: fields to be added

        >>> query_profile_type = QueryProfileType()
        >>> query_profile_type.add_fields(
        ...     QueryTypeField(
        ...         name="age",
        ...         type="integer"
        ...     ),
        ...     QueryTypeField(
        ...         name="profession",
        ...         type="string"
        ...     )
        ... )
        """
        self.fields.extend(fields)

    @staticmethod
    def from_dict(mapping: Mapping) -> "QueryProfileType":
        return QueryProfileType(
            fields=[FromJson.map(field) for field in mapping.get("fields")]
        )

    @property
    def to_dict(self) -> Mapping:
        map = {"fields": [field.to_envelope for field in self.fields]}
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.fields == other.fields

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, repr(self.fields) if self.fields else None
        )


class QueryField(ToJson, FromJson["QueryField"]):
    def __init__(
        self,
        name: str,
        value: Union[str, int, float],
    ) -> None:
        """
        Create a field to be included in a :class:`QueryProfile`.

        :param name: Field name.
        :param value: Field value.

        >>> QueryField(name="maxHits", value=1000)
        QueryField('maxHits', 1000)
        """
        self.name = name
        self.value = value

    @staticmethod
    def from_dict(mapping: Mapping) -> "QueryField":
        return QueryField(
            name=mapping["name"],
            value=mapping["value"],
        )

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name, "value": self.value}
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.value == other.value

    def __repr__(self):
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.value),
        )


class QueryProfile(ToJson, FromJson["QueryProfile"]):
    def __init__(self, fields: Optional[List[QueryField]] = None) -> None:
        """
        Create a Vespa Query Profile.

        Check the `Vespa documentation <https://docs.vespa.ai/en/query-profiles.html>`__
        for more detailed information about query profiles.

        A :class:`QueryProfile` is a named collection of query request parameters given in the configuration.
        The query request can specify a query profile whose parameters will be used as parameters of that request.
        The query profiles may optionally be type checked.
        Type checking is turned on by referencing a :class:`QueryProfileType` from the query profile.

        :param fields: A list of :class:`QueryField`.

        >>> QueryProfile(fields=[QueryField(name="maxHits", value=1000)])
        QueryProfile([QueryField('maxHits', 1000)])
        """
        self.name = "default"
        self.type = "root"
        self.fields = [] if not fields else fields

    def add_fields(self, *fields: QueryField) -> None:
        """
        Add :class:`QueryField`'s to the Query Profile.

        :param fields: fields to be added

        >>> query_profile = QueryProfile()
        >>> query_profile.add_fields(QueryField(name="maxHits", value=1000))
        """
        self.fields.extend(fields)

    @staticmethod
    def from_dict(mapping: Mapping) -> "QueryProfile":
        return QueryProfile(
            fields=[FromJson.map(field) for field in mapping.get("fields")]
        )

    @property
    def to_dict(self) -> Mapping:
        map = {"fields": [field.to_envelope for field in self.fields]}
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.fields == other.fields

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, repr(self.fields) if self.fields else None
        )


class ModelConfig(object):
    def __init__(self, model_id) -> None:
        self.model_id = model_id

    def onnx_model(self):
        raise NotImplementedError

    def query_profile_type_fields(self):
        raise NotImplementedError

    def document_fields(self, document_field_indexing):
        raise NotImplementedError

    def rank_profile(self, include_model_summary_features, **kwargs):
        raise NotImplementedError


class Task(object):
    def __init__(
        self,
        model_id: str,
    ):
        """
        Base class for ML Tasks.

        :param model_id: Id used to identify the model on Vespa applications.
        """
        self.model_id = model_id


class ApplicationPackage(ToJson, FromJson["ApplicationPackage"]):
    def __init__(
        self,
        name: str,
        schema: Optional[List[Schema]] = None,
        query_profile: Optional[QueryProfile] = None,
        query_profile_type: Optional[QueryProfileType] = None,
        stateless_model_evaluation: bool = False,
        create_schema_by_default: bool = True,
        create_query_profile_by_default: bool = True,
        tasks: Optional[List[Task]] = None,
        default_query_model: Optional[QueryModel] = None
    ) -> None:
        """
        Create an `Application Package <https://docs.vespa.ai/en/cloudconfig/application-packages.html>`__.
        An :class:`ApplicationPackage` instance comes with a default :class:`Schema`
        that contains a default :class:`Document`

        :param name: Application name. Cannot contain '-' or '_'.
        :param schema: List of :class:`Schema`s of the application.
            If `None`, an empty :class:`Schema` with the same name of the application will be created by default.
        :param query_profile: :class:`QueryProfile` of the application.
            If `None`, a :class:`QueryProfile` named `default` with :class:`QueryProfileType` named `root`
            will be created by default.
        :param query_profile_type: :class:`QueryProfileType` of the application. If `None`, a empty
            :class:`QueryProfileType` named `root` will be created by default.
        :param stateless_model_evaluation: Enable stateless model evaluation. Default to False.
        :param create_schema_by_default: Include a :class:`Schema` with the same name as the application if no Schema
            is provided in the `schema` argument.
        :param create_query_profile_by_default: Include a default :class:`QueryProfile` and :class:`QueryProfileType`
            in case it is not explicitly defined by the user in the `query_profile` and `query_profile_type` parameters.
        :param tasks: List of tasks to be served.
        :param default_query_model: Optional QueryModel to be used as default for the application.

        The easiest way to get started is to create a default application package:

        >>> ApplicationPackage(name="testapp")
        ApplicationPackage('testapp', [Schema('testapp', Document(None, None), None, None, [], False, None)], QueryProfile(None), QueryProfileType(None))

        It will create a default :class:`Schema`, :class:`QueryProfile` and :class:`QueryProfileType` that you can then
        populate with specifics of your application.
        """
        if not name.isalnum():
            raise ValueError("Application package name can only contain [a-zA-Z0-9], was '{}'".format(name))
        self.name = name
        if not schema:
            schema = (
                [Schema(name=self.name, document=Document())]
                if create_schema_by_default
                else []
            )
        self._schema = OrderedDict([(x.name, x) for x in schema])
        if not query_profile and create_query_profile_by_default:
            query_profile = QueryProfile()
        self.query_profile = query_profile
        if not query_profile_type and create_query_profile_by_default:
            query_profile_type = QueryProfileType()
        self.query_profile_type = query_profile_type
        self.model_ids = []
        self.model_configs = {}
        self.stateless_model_evaluation = stateless_model_evaluation
        self.models = {} if not tasks else {model.model_id: model for model in tasks}
        self.default_query_model = default_query_model

    @property
    def schemas(self) -> List[Schema]:
        return [x for x in self._schema.values()]

    @property
    def schema(self):
        assert (
            len(self.schemas) <= 1
        ), "Your application has more than one Schema, use get_schema instead."
        return self.schemas[0] if self.schemas else None

    def get_schema(self, name: Optional[str] = None):
        if not name:
            assert (
                len(self.schemas) <= 1
            ), "Your application has more than one Schema, specify name argument."
            return self.schema
        return self._schema[name]

    def add_schema(self, *schemas: Schema) -> None:
        """
        Add :class:`Schema`'s to the application package.

        :param schemas: schemas to be added
        :return:
        """
        for schema in schemas:
            self._schema.update({schema.name: schema})

    def get_model(self, model_id: str):
        try:
            return self.models[model_id]
        except KeyError:
            raise ValueError(
                "Model named {} not defined in the application package.".format(
                    model_id
                )
            )

    def add_model_ranking(
        self,
        model_config: ModelConfig,
        schema=None,
        include_model_summary_features=False,
        document_field_indexing=None,
        **kwargs
    ) -> None:
        """
        Add ranking profile based on a specific model config.

        :param model_config: Model config instance specifying the model to be used on the RankProfile.
        :param schema: Name of the schema to add model ranking to.
        :param include_model_summary_features: True to include model specific summary features, such as
            inputs and outputs that are useful for debugging. Default to False as this requires an extra model
            evaluation when fetching summary features.
        :param document_field_indexing: List of indexing attributes for the document fields required by the ranking
            model.
        :param kwargs: Further arguments to be passed to RankProfile.
        :return: None
        """

        model_id = model_config.model_id
        #
        # Validate and persist config
        #
        if model_id in self.model_ids:
            raise ValueError("model_id must be unique: {}".format(model_id))
        self.model_ids.append(model_id)
        self.model_configs[model_id] = model_config
        #
        # Export ONNX model
        #
        self.get_schema(schema).add_model(model_config.onnx_model())
        #
        # Add query profile type fields
        #
        self.query_profile_type.add_fields(*model_config.query_profile_type_fields())
        #
        # Add field for doc token ids
        #
        self.get_schema(schema).add_fields(
            *model_config.document_fields(
                document_field_indexing=document_field_indexing
            )
        )
        #
        # Add rank profiles
        #
        self.get_schema(schema).add_rank_profile(
            model_config.rank_profile(
                include_model_summary_features=include_model_summary_features, **kwargs
            )
        )

    @property
    def query_profile_to_text(self):
        env = Environment(
            loader=PackageLoader("vespa", "templates"),
            autoescape=select_autoescape(
                disabled_extensions=("txt",),
                default_for_string=True,
                default=True,
            ),
        )
        env.trim_blocks = True
        env.lstrip_blocks = True
        query_profile_template = env.get_template("query_profile.xml")
        return query_profile_template.render(query_profile=self.query_profile)

    @property
    def query_profile_type_to_text(self):
        env = Environment(
            loader=PackageLoader("vespa", "templates"),
            autoescape=select_autoescape(
                disabled_extensions=("txt",),
                default_for_string=True,
                default=True,
            ),
        )
        env.trim_blocks = True
        env.lstrip_blocks = True
        query_profile_type_template = env.get_template("query_profile_type.xml")
        return query_profile_type_template.render(
            query_profile_type=self.query_profile_type
        )

    @property
    def services_to_text(self):
        env = Environment(
            loader=PackageLoader("vespa", "templates"),
            autoescape=select_autoescape(
                disabled_extensions=("txt",),
                default_for_string=True,
                default=True,
            ),
        )
        env.trim_blocks = True
        env.lstrip_blocks = True
        schema_template = env.get_template("services.xml")
        return schema_template.render(
            application_name=self.name,
            schemas=self.schemas,
            stateless_model_evaluation=self.stateless_model_evaluation,
        )

    @staticmethod
    def from_dict(mapping: Mapping) -> "ApplicationPackage":
        schema = mapping.get("schema", None)
        if schema is not None:
            schema = [FromJson.map(x) for x in schema]
        return ApplicationPackage(name=mapping["name"], schema=schema)

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name}
        if self._schema is not None:
            map.update({"schema": [x.to_envelope for x in self.schemas]})
        return map

    @staticmethod
    def _application_package_file_name(disk_folder):
        return os.path.join(disk_folder, "application_package.json")

    def save(self, disk_folder: str) -> None:
        Path(disk_folder).mkdir(parents=True, exist_ok=True)
        file_path = ApplicationPackage._application_package_file_name(disk_folder)
        with open(file_path, "w") as f:
            f.write(self.to_json)

    @staticmethod
    def load(disk_folder: str) -> "ApplicationPackage":
        file_path = ApplicationPackage._application_package_file_name(disk_folder)
        with open(file_path, "r") as f:
            return ApplicationPackage.from_json(f.read())

    def to_zip(self) -> BytesIO:
        """
        Return the application package as zipped bytes,
        to be used in a subsequent deploy
        :return: BytesIO buffer
        """
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "a") as zip_archive:
            zip_archive.writestr(
                "services.xml", self.services_to_text
            )

            for schema in self.schemas:
                zip_archive.writestr(
                    "schemas/{}.sd".format(schema.name),
                    schema.schema_to_text,
                )
                for model in schema.models:
                    zip_archive.write(
                        model.model_file_path,
                        os.path.join("files", model.model_file_name),
                    )

            if self.models:
                for model_id, model in self.models.items():
                    temp_model_file = "{}.onnx".format(model_id)
                    model.export_to_onnx(output_path=temp_model_file)
                    zip_archive.write(
                        temp_model_file,
                        "models/{}.onnx".format(model_id),
                    )
                    os.remove(temp_model_file)

            if self.query_profile:
                zip_archive.writestr(
                    "search/query-profiles/default.xml",
                    self.query_profile_to_text,
                )
                zip_archive.writestr(
                    "search/query-profiles/types/root.xml",
                    self.query_profile_type_to_text,
                )

        buffer.seek(0)
        return buffer

        # ToDo: use this for the Vespa Cloud app package
        #zip_archive.writestr(
        #    "application/security/clients.pem",
        #    app.public_bytes(serialization.Encoding.PEM),
        #)

    def to_zipfile(self, zipfile: Path) -> None:
        """
        Export the application package as a deployable zipfile.
        See `application packages <https://docs.vespa.ai/en/cloudconfig/application-packages.html>`__
        for deployment options.

        :param zipfile: Filename to export to
        :return:
        """
        with open(zipfile, "wb") as f:
            f.write(self.to_zip().getbuffer().tobytes())

    def to_files(self, root: Path) -> None:
        """
        Export the application package as a directory tree.

        :param root: Directory to export files to
        :return:
        """
        if not os.path.exists(root):
            raise ValueError("Invalid path for export: {}".format(root))

        Path(os.path.join(root, "schemas")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, "files")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, "models")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, "search/query-profiles/types")).mkdir(parents=True, exist_ok=True)

        for schema in self.schemas:
            with open(os.path.join(root, "schemas/{}.sd".format(schema.name)), "w") as f:
                f.write(schema.schema_to_text)
            for model in schema.models:
                copyfile(
                    model.model_file_path,
                    os.path.join(root, "files", model.model_file_name)
                )

        if self.query_profile:
            with open(os.path.join(root, "search/query-profiles/default.xml"), "w") as f:
                f.write(self.query_profile_to_text)
            with open(os.path.join(root, "search/query-profiles/types/root.xml"), "w") as f:
                f.write(self.query_profile_type_to_text)

        with open(os.path.join(root, "services.xml"), "w") as f:
            f.write(self.services_to_text)

        if self.models:
            for model_id, model in self.models.items():
                model.export_to_onnx(output_path=os.path.join(root, "models/{}.onnx".format(model_id)))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self._schema == other._schema

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.schemas),
            repr(self.query_profile),
            repr(self.query_profile_type),
        )


class ModelServer(ApplicationPackage):
    def __init__(
        self,
        name: str,
        tasks: Optional[List[Task]] = None,
    ):
        """
        Create a Vespa stateless model evaluation server.

        A Vespa stateless model evaluation server is a simplified Vespa application without content clusters.

        :param name: Application name.
        :param tasks: List of tasks to be served.
        """
        super().__init__(
            name=name,
            schema=None,
            query_profile=None,
            query_profile_type=None,
            stateless_model_evaluation=True,
            create_schema_by_default=False,
            create_query_profile_by_default=False,
            tasks=tasks,
        )

    @staticmethod
    def from_dict(mapping: Mapping) -> "ModelServer":
        return ModelServer(name=mapping["name"])

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name}
        return map
