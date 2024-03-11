# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from enum import Enum
import os
import sys
import zipfile

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from pathlib import Path
from shutil import copyfile
from typing import List, Literal, Optional, Tuple, TypedDict, Union, Dict
from collections import OrderedDict
from jinja2 import Environment, PackageLoader, select_autoescape
from io import BytesIO

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    # Older versions of Python have Unpack in typing_extensions
    from typing_extensions import Unpack


class Summary(object):
    def __init__(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
        fields: Optional[List[Union[str, Tuple[str, Union[List[str], str]]]]] = None,
    ) -> None:
        """
        Configures a summary Field.

        :param name: Name of the summary field, can be None if used inside a Field,
            which then uses the name of the Field.
        :param type: Type of the summary field, can be None if used inside a Field,
            which then uses the type of the Field.
        :param fields: List of properties used to configure the summary,
            can be single properties (like "summary: dynamic", common in Field),
            or composite values (like "source: another_field")

        >>> Summary(None, None, ["dynamic"])
        Summary(None, None, ['dynamic'])

        >>> Summary(
        ...     "title",
        ...     "string",
        ...     [("source", "title")]
        ... )
        Summary('title', 'string', [('source', 'title')])

        >>> Summary(
        ...     "title",
        ...     "string",
        ...     [("source", ["title", "abstract"])]
        ... )
        Summary('title', 'string', [('source', ['title', 'abstract'])])

        >>> Summary(
        ...     name = "artist",
        ...     type = "string",
        ... )
        Summary('artist', 'string', None)
        """
        self.name = name
        self.type = type
        self.fields = fields

    @property
    def as_lines(self) -> List[str]:
        """
        Returns the object as a List of str, each str representing a line
        of configuration that can be used during schema generation as such:

        ```
        {% for line in field.summary.as_lines %}
        {{ line }}
        {% endfor %}
        ```

        >>> Summary(None, None, ["dynamic"]).as_lines
        ['summary: dynamic']

        >>> Summary(
        ...     "artist",
        ...     "string",
        ... ).as_lines
        ['summary artist type string {}']

        >>> Summary(
        ...     "artist",
        ...     "string",
        ...     [("bolding", "on"), ("sources", "artist")],
        ... ).as_lines
        ['summary artist type string {', '    bolding: on', '    sources: artist', '}']
        """
        final_list = []

        # Special case of `summary: dynamic` and others.
        if (
            not self.name
            and not self.type
            and self.fields
            and len(self.fields) == 1
            and isinstance(self.fields[0], str)
        ):
            return [f"summary: {self.fields[0]}"]

        starting_string = "summary"
        if self.name:
            starting_string += f" {self.name}"
        if self.type:
            starting_string += f" type {self.type}"

        # Add newline as each field resides in a separate line
        if self.fields is None:
            starting_string += " {}"
            return [starting_string]

        starting_string += " {"
        final_list.append(starting_string)

        for field in self.fields:
            if isinstance(field, str):
                final_list.append(f"    {field}")
            # We could use else, but that does not narrow down
            # the type
            else:
                tmp_string = f"    {field[0]}: "
                if isinstance(field[1], str):
                    tmp_string += f"{field[1]}"
                else:
                    tmp_string += f'{", ".join(field[1])}'
                final_list.append(tmp_string)

        final_list.append("}")
        return final_list

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Summary):
            return NotImplemented
        return (
            self.name == other.name
            and self.type == other.type
            and self.fields == other.fields
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__, repr(self.name), repr(self.type), repr(self.fields)
        )


class HNSW(object):
    def __init__(
        self,
        distance_metric: Literal[
            "euclidean",
            "angular",
            "dotproduct",
            "prenormalized-angular",
            "hamming",
            "geodegrees",
        ] = "euclidean",
        max_links_per_node=16,
        neighbors_to_explore_at_insert=200,
    ):
        """
        Configure Vespa HNSW indexes
        Check the `Vespa documentation <https://docs.vespa.ai/en/approximate-nn-hnsw.html>`__

        :param distance_metric: Distance metric to use when computing distance between vectors.
            Default is 'euclidean'.
        :param max_links_per_node: Specifies how many links per HNSW node to select when building the graph.
            Default is 16.
        :param neighbors_to_explore_at_insert: Specifies how many neighbors to explore when inserting a document in
            the HNSW graph. Default is 200.
        """
        self.distance_metric = distance_metric
        self.max_links_per_node = max_links_per_node
        self.neighbors_to_explore_at_insert = neighbors_to_explore_at_insert

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.distance_metric == other.distance_metric
            and self.max_links_per_node == other.max_links_per_node
            and self.neighbors_to_explore_at_insert
            == other.neighbors_to_explore_at_insert
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.distance_metric),
            repr(self.max_links_per_node),
            repr(self.neighbors_to_explore_at_insert),
        )


class StructFieldConfiguration(TypedDict, total=False):
    indexing: List[str]
    attribute: List[str]
    match: List[Union[str, Tuple[str, str]]]
    query_command: List[str]
    summary: Summary


class StructField:
    def __init__(self, name: str, **kwargs: Unpack[StructFieldConfiguration]) -> None:
        """
        Create a Vespa struct-field.
        Check the `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#struct-field>`__
        for more detailed information about struct-fields.

        :param name: Struct-field name.
        :key indexing: Configures how to process data of a struct-field during indexing.
        :key attribute: Specifies a property of an index structure attribute.
        :key match: Set properties that decide how the matching method for this field operate.
        :key query_command: Add configuration for query-command of the field.
        :key summary: Add configuration for summary of the field.

        >>> StructField(
        ...     name = "first_name",
        ... )
        StructField('first_name', None, None, None, None, None)

        >>> StructField(
        ...     name = "first_name",
        ...     indexing = ["attribute"],
        ...     attribute = ["fast-search"],
        ... )
        StructField('first_name', ['attribute'], ['fast-search'], None, None, None)

        >>> StructField(
        ...     name = "last_name",
        ...     match = ["exact", ("exact-terminator", '"@%"')],
        ...     query_command = ['"exact %%"'],
        ...     summary = Summary(None, None, fields=["dynamic", ("bolding", "on")])
        ... )
        StructField('last_name', None, None, ['exact', ('exact-terminator', '"@%"')], ['"exact %%"'], Summary(None, None, ['dynamic', ('bolding', 'on')]))
        """
        self.name = name
        self.indexing = kwargs.get("indexing", None)
        self.attribute = kwargs.get("attribute", None)
        self.match = kwargs.get("match", None)
        self.query_command = kwargs.get("query_command", None)
        self.summary = kwargs.get("summary", None)

    @property
    def indexing_to_text(self) -> Optional[str]:
        if self.indexing is not None:
            return " | ".join(self.indexing)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.name,
            self.indexing,
            self.attribute,
            self.match,
            self.query_command,
            self.summary,
        ) == (
            other.name,
            other.indexing,
            other.attribute,
            other.match,
            other.query_command,
            other.summary,
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4}, {5}, {6})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.indexing),
            repr(self.attribute),
            repr(self.match),
            repr(self.query_command),
            repr(self.summary),
        )


class FieldConfiguration(TypedDict, total=False):
    indexing: List[str]
    attribute: List[str]
    index: str
    ann: HNSW
    match: List[Union[str, Tuple[str, str]]]
    weight: int
    bolding: Literal[True]
    summary: Summary
    stemming: str
    rank: str
    query_command: List[str]
    struct_fields: List[StructField]
    alias: List[str]


class Field(object):
    def __init__(
        self,
        name: str,
        type: str,
        indexing: Optional[List[str]] = None,
        index: Optional[str] = None,
        attribute: Optional[List[str]] = None,
        ann: Optional[HNSW] = None,
        match: Optional[List[Union[str, Tuple[str, str]]]] = None,
        weight: Optional[int] = None,
        bolding: Optional[Literal[True]] = None,
        summary: Optional[Summary] = None,
        is_document_field: Optional[bool] = True,
        **kwargs: Unpack[FieldConfiguration],
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
        :param match: Set properties that decide how the matching method for this field operate.
        :param weight: Sets the weight of the field, using when calculating Rank Scores.
        :param bolding: Whether to highlight matching query terms in the summary.
        :param summary: Add configuration for summary of the field.
        :param is_document_field: Whether the field is a document field or part of the schema. True by default.
        :key stemming: Add configuration for stemming of the field.
        :key rank: Add configuration for ranking calculations of the field.
        :key query_command: Add configuration for query-command of the field.
        :key struct_fields: Add struct-fields to the field.
        :key alias: Add alias to the field.

        >>> Field(name = "title", type = "string", indexing = ["index", "summary"], index = "enable-bm25")
        Field('title', 'string', ['index', 'summary'], 'enable-bm25', None, None, None, None, None, None, True, None, None, None, [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     indexing = ["attribute"],
        ...     attribute=["fast-search", "fast-access"]
        ... )
        Field('abstract', 'string', ['attribute'], None, ['fast-search', 'fast-access'], None, None, None, None, None, True, None, None, None, [], None)

        >>> Field(name="tensor_field",
        ...     type="tensor<float>(x[128])",
        ...     indexing=["attribute"],
        ...     ann=HNSW(
        ...         distance_metric="euclidean",
        ...         max_links_per_node=16,
        ...         neighbors_to_explore_at_insert=200,
        ...     ),
        ... )
        Field('tensor_field', 'tensor<float>(x[128])', ['attribute'], None, None, HNSW('euclidean', 16, 200), None, None, None, None, True, None, None, None, [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     match = ["exact", ("exact-terminator", '"@%"',)],
        ... )
        Field('abstract', 'string', None, None, None, None, ['exact', ('exact-terminator', '"@%"')], None, None, None, True, None, None, None, [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     weight = 200,
        ... )
        Field('abstract', 'string', None, None, None, None, None, 200, None, None, True, None, None, None, [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     bolding = True,
        ... )
        Field('abstract', 'string', None, None, None, None, None, None, True, None, True, None, None, None, [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     summary = Summary(None, None, ["dynamic", ["bolding", "on"]]),
        ... )
        Field('abstract', 'string', None, None, None, None, None, None, None, Summary(None, None, ['dynamic', ['bolding', 'on']]), True, None, None, None, [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     stemming = "shortest",
        ... )
        Field('abstract', 'string', None, None, None, None, None, None, None, None, True, 'shortest', None, None, [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     rank = "filter",
        ... )
        Field('abstract', 'string', None, None, None, None, None, None, None, None, True, None, 'filter', None, [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     query_command = ['"exact %%"'],
        ... )
        Field('abstract', 'string', None, None, None, None, None, None, None, None, True, None, None, ['"exact %%"'], [], None)

        >>> Field(
        ...     name = "abstract",
        ...     type = "string",
        ...     struct_fields = [
        ...         StructField(
        ...             name = "first_name",
        ...             indexing = ["attribute"],
        ...             attribute = ["fast-search"],
        ...         ),
        ...     ],
        ... )
        Field('abstract', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [StructField('first_name', ['attribute'], ['fast-search'], None, None, None)], None)

        >>> Field(
        ...     name = "artist",
        ...     type = "string",
        ...     alias = ["artist_name"],
        ... )
        Field('artist', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], ['artist_name'])
        """
        self.name = name
        self.type = type
        self.is_document_field = is_document_field
        self.indexing = kwargs.get("indexing", indexing)
        self.attribute = kwargs.get("attribute", attribute)
        self.index = kwargs.get("index", index)
        self.ann = kwargs.get("ann", ann)
        self.match = kwargs.get("match", match)
        self.weight = kwargs.get("weight", weight)
        self.bolding = kwargs.get("bolding", bolding)
        self.summary = kwargs.get("summary", summary)
        self.stemming = kwargs.get("stemming", None)
        self.rank = kwargs.get("rank", None)
        self.query_command = kwargs.get("query_command", None)
        self._struct_fields = (
            OrderedDict()
            if not kwargs.get("struct_fields", None)
            else OrderedDict(
                [
                    (struct_field.name, struct_field)
                    for struct_field in kwargs.get("struct_fields", [])
                ]
            )
        )
        self.alias = kwargs.get("alias", None)

    @property
    def indexing_to_text(self) -> Optional[str]:
        if self.indexing is not None:
            return " | ".join(self.indexing)

    @property
    def struct_fields(self) -> List[StructField]:
        return [x for x in self._struct_fields.values()]

    def add_struct_fields(self, *struct_fields: StructField) -> None:
        """
        Add :class:`StructField`'s to the Field.

        :param struct_fields: struct-fields to be added
        """
        for struct_field in struct_fields:
            self._struct_fields.update({struct_field.name: struct_field})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.name == other.name
            and self.type == other.type
            and self.indexing == other.indexing
            and self.index == other.index
            and self.attribute == other.attribute
            and self.ann == other.ann
            and self.match == other.match
            and self.weight == other.weight
            and self.bolding == other.bolding
            and self.summary == other.summary
            and self.is_document_field == other.is_document_field
            and self.stemming == other.stemming
            and self.rank == other.rank
            and self.query_command == other.query_command
            and self.struct_fields == other.struct_fields
            and self.alias == other.alias
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.type),
            repr(self.indexing),
            repr(self.index),
            repr(self.attribute),
            repr(self.ann),
            repr(self.match),
            repr(self.weight),
            repr(self.bolding),
            repr(self.summary),
            repr(self.is_document_field),
            repr(self.stemming),
            repr(self.rank),
            repr(self.query_command),
            repr(self.struct_fields),
            repr(self.alias),
        )


class ImportedField(object):
    def __init__(
        self,
        name: str,
        reference_field: str,
        field_to_import: str,
    ) -> None:
        """
        Imported field from a reference document.

        Useful to implement `parent/child relationships <https://docs.vespa.ai/en/parent-child.html>`__.

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.name == other.name
            and self.reference_field == other.reference_field
            and self.field_to_import == other.field_to_import
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.reference_field),
            repr(self.field_to_import),
        )


class Struct(object):
    def __init__(self, name: str, fields: Optional[List[Field]] = None):
        """
        Create a Vespa struct.
        A struct defines a composite type. Check the `Vespa documentation
        <https://docs.vespa.ai/en/reference/schema-reference.html#struct>`__
        for more detailed information about structs.

        :param name: Name of the struct
        :param fields: Field names to be included in the fieldset

        >>> Struct("person")
        Struct('person', None)

        >>> Struct(
        ...     "person",
        ...     [
        ...         Field("first_name", "string"),
        ...         Field("last_name", "string"),
        ...     ],
        ... )
        Struct('person', [Field('first_name', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None), Field('last_name', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None)])
        """
        self.name = name
        self.fields = fields

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.name, self.fields) == (other.name, other.fields)

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.fields),
        )


class DocumentSummary(object):
    def __init__(
        self,
        name: str,
        inherits: Optional[str] = None,
        summary_fields: Optional[List[Summary]] = None,
        from_disk: Optional[Literal[True]] = None,
        omit_summary_features: Optional[Literal[True]] = None,
    ) -> None:
        """
        Create a Document Summary.
        Check the `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#document-summary>`__
        for more detailed information about documment-summary.

        :param name: Name of the document-summary.
        :param inherits: Name of another document-summary from which this inherits from.
        :param summary_fields: List of summaries used in this document-summary.
        :param from_disk: Marks this document-summary as accessing fields on disk.
        :param omit_summary_features: Specifies that summary-features should be omitted from this document summary.

        >>> DocumentSummary(
        ...     name="document-summary",
        ... )
        DocumentSummary('document-summary', None, None, None, None)

        >>> DocumentSummary(
        ...     name="which-inherits",
        ...     inherits="base-document-summary",
        ... )
        DocumentSummary('which-inherits', 'base-document-summary', None, None, None)

        >>> DocumentSummary(
        ...     name="with-field",
        ...     summary_fields=[Summary("title", "string", [("source", "title")])]
        ... )
        DocumentSummary('with-field', None, [Summary('title', 'string', [('source', 'title')])], None, None)

        >>> DocumentSummary(
        ...     name="with-bools",
        ...     from_disk=True,
        ...     omit_summary_features=True,
        ... )
        DocumentSummary('with-bools', None, None, True, True)
        """
        self.name = name
        self.inherits = inherits
        self.summary_fields = summary_fields
        self.from_disk = from_disk
        self.omit_summary_features = omit_summary_features

    def __eq__(self, other: object):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.name == other.name
            and self.inherits == other.inherits
            and self.summary_fields == other.summary_fields
            and self.from_disk == other.from_disk
            and self.omit_summary_features == other.omit_summary_features
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4}, {5})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.inherits),
            repr(self.summary_fields),
            repr(self.from_disk),
            repr(self.omit_summary_features),
        )


class Document(object):
    def __init__(
        self,
        fields: Optional[List[Field]] = None,
        inherits: Optional[str] = None,
        structs: Optional[List[Struct]] = None,
    ) -> None:
        """
        Create a Vespa Document.

        Check the `Vespa documentation <https://docs.vespa.ai/en/documents.html>`__
        for more detailed information about documents.

        :param fields: A list of :class:`Field` to include in the document's schema.

        To create a Document:

        >>> Document()
        Document(None, None, None)

        >>> Document(fields=[Field(name="title", type="string")])
        Document([Field('title', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None)], None, None)

        >>> Document(fields=[Field(name="title", type="string")], inherits="context")
        Document([Field('title', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None)], context, None)
        """
        self.inherits = inherits
        self._fields = (
            OrderedDict()
            if not fields
            else OrderedDict([(field.name, field) for field in fields])
        )
        self._structs = (
            OrderedDict()
            if not structs
            else OrderedDict([(struct.name, struct) for struct in structs])
        )

    @property
    def fields(self):
        return [x for x in self._fields.values()]

    @property
    def structs(self):
        return [x for x in self._structs.values()]

    def add_fields(self, *fields: Field) -> None:
        """
        Add :class:`Field`'s to the document.

        :param fields: fields to be added
        :return:
        """
        for field in fields:
            self._fields.update({field.name: field})

    def add_structs(self, *structs: Struct) -> None:
        """
        Add :class:`Struct`'s to the document.

        :param structs: structs to be added
        :return:
        """
        for struct in structs:
            self._structs.update({struct.name: struct})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.fields, self.inherits, self.structs) == (
            other.fields,
            other.inherits,
            other.structs,
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.fields) if self.fields else None,
            self.inherits,
            repr(self.structs) if self.structs else None,
        )


class FieldSet(object):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name and self.fields == other.fields

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__, repr(self.name), repr(self.fields)
        )


class Function(object):
    def __init__(
        self, name: str, expression: str, args: Optional[List[str]] = None
    ) -> None:
        r"""
        Create a Vespa rank function.

        Define a named function that can be referenced as a part of the ranking expression,
        or (if having no arguments) as a feature.
        Check the `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#function-rank>`__
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.name == other.name
            and self.expression == other.expression
            and self.args == other.args
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.expression),
            repr(self.args),
        )

class FirstPhaseRanking:
    def __init__(
            self,
            expression: str,
            keep_rank_count: Optional[int] = None,
            rank_score_drop_limit: Optional[float] = None
        ) -> None:
        r"""
        Create a Vespa first phase ranking configuration.

        This is the initial ranking performed on all matching documents.
        Check the `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#firstphase-rank>`__
        for more detailed information about first phase ranking configuration.


        :param expression: Specify the ranking expression to be used for first phase of ranking. Check also the
            `Vespa documentation <https://docs.vespa.ai/en/reference/ranking-expressions.html>`__
            for ranking expression.

        :param keep_rank_count: How many documents to keep the first phase top rank values for. Default value is 10000.
        :param rank_score_drop_limit: Drop all hits with a first phase rank score less than or equal
            to this floating point number.

        >>> FirstPhaseRanking("myFeature * 10")
        FirstPhaseRanking('myFeature * 10', None, None)

        >>> FirstPhaseRanking(expression="myFeature * 10", keep_rank_count=50, rank_score_drop_limit=10)
        FirstPhaseRanking('myFeature * 10', 50, 10)
        """
        self.expression = expression
        self.keep_rank_count = keep_rank_count
        self.rank_score_drop_limit = rank_score_drop_limit

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.expression == other.expression
            and self.keep_rank_count == other.keep_rank_count
            and self.rank_score_drop_limit == other.rank_score_drop_limit
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.expression),
            repr(self.keep_rank_count),
            repr(self.rank_score_drop_limit)
        )

class SecondPhaseRanking(object):
    def __init__(self, expression: str, rerank_count: int = 100) -> None:
        r"""
        Create a Vespa second phase ranking configuration.

        This is the optional reranking performed on the best hits from the first phase. Check the
        `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#secondphase-rank>`__
        for more detailed information about second phase ranking configuration.

        :param expression: Specify the ranking expression to be used for second phase of ranking. Check also the
            `Vespa documentation <https://docs.vespa.ai/en/reference/ranking-expressions.html>`__
            for ranking expression.
        :param rerank_count: Specifies the number of hits to be reranked in the second phase. Default value is 100.

        >>> SecondPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10)
        SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10)
        """
        self.expression = expression
        self.rerank_count = rerank_count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.expression == other.expression
            and self.rerank_count == other.rerank_count
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.expression),
            repr(self.rerank_count),
        )

class GlobalPhaseRanking(object):
    def __init__(self, expression: str, rerank_count: int = 100) -> None:
        r"""
        Create a Vespa global phase ranking configuration.

        This is the optional reranking performed on the best hits from the content nodes phase(s). Check the
        `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#globalphase-rank>`__
        for more detailed information about global phase ranking configuration.

        :param expression: Specify the ranking expression to be used for second phase of ranking. Check also the
            `Vespa documentation <https://docs.vespa.ai/en/reference/ranking-expressions.html>`__
            for ranking expression.
        :param rerank_count: Specifies the number of hits to be reranked in the second phase. Default value is 100.

        >>> GlobalPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10)
        GlobalPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10)
        """
        self.expression = expression
        self.rerank_count = rerank_count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.expression == other.expression
            and self.rerank_count == other.rerank_count
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.expression),
            repr(self.rerank_count),
        )


class RankProfileFields(TypedDict, total=False):
    inherits: str
    constants: Dict
    functions: List[Function]
    summary_features: List
    match_features: List
    second_phase: SecondPhaseRanking
    global_phase: GlobalPhaseRanking
    weight: List[Tuple[str, int]]
    rank_type: List[Tuple[str, str]]
    rank_properties: List[Tuple[str, str]]
    inputs: List[Union[Tuple[str, str], Tuple[str, str, str]]]


class RankProfile(object):
    def __init__(
        self,
        name: str,
        # Allow a str object as expression for backwards compatibility
        first_phase: Union[str, FirstPhaseRanking],
        inherits: Optional[str] = None,
        constants: Optional[Dict] = None,
        functions: Optional[List[Function]] = None,
        summary_features: Optional[List] = None,
        match_features: Optional[List] = None,
        second_phase: Optional[SecondPhaseRanking] = None,
        global_phase: Optional[GlobalPhaseRanking] = None,
        **kwargs: Unpack[RankProfileFields],
    ) -> None:
        """
        Create a Vespa rank profile.

        Rank profiles are used to specify an alternative ranking of the same data for different purposes, and to
        experiment with new rank settings. Check the
        `Vespa documentation <https://docs.vespa.ai/en/reference/schema-reference.html#rank-profile>`__
        for more detailed information about rank profiles.

        :param name: Rank profile name.
        :param first_phase: The config specifying the first phase of ranking.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#firstphase-rank>`__
            about first phase ranking.
        :param inherits: The inherits attribute is optional. If defined, it contains the name of one other
            rank profile in the same schema. Values not defined in this rank profile will then be inherited.
        :param constants: Dict of constants available in ranking expressions, resolved and optimized at
            configuration time.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#constants>`__
            about constants.
        :param functions: Optional list of :class:`Function` representing rank functions to be included in the rank
            profile.
        :param summary_features: List of rank features to be included with each hit.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#summary-features>`__
            about summary features.
        :param match_features: List of rank features to be included with each hit.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#match-features>`__
            about match features.
        :param second_phase: Optional config specifying the second phase of ranking.
            See :class:`SecondPhaseRanking`.
        :param global_phase: Optional config specifying the global phase of ranking.
            See :class:`GlobalPhaseRanking`.
        :key weight: A list of tuples containing the field and their weight
        :key rank_type: A list of tuples containing a field and the rank-type-name.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#rank-type>`__ about rank-type.
        :key rank_properties: A list of tuples containing a field and its configuration.
            `More info <https://docs.vespa.ai/en/reference/schema-reference.html#rank-properties>`__ about rank-properties.

        >>> RankProfile(name = "default", first_phase = "nativeRank(title, body)")
        RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, None, None, None, None)

        >>> RankProfile(name = "new", first_phase = "BM25(title)", inherits = "default")
        RankProfile('new', 'BM25(title)', 'default', None, None, None, None, None, None, None, None, None, None)

        >>> RankProfile(
        ...     name = "new",
        ...     first_phase = "BM25(title)",
        ...     inherits = "default",
        ...     constants={"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102},
        ...     summary_features=["BM25(title)"]
        ... )
        RankProfile('new', 'BM25(title)', 'default', {'TOKEN_NONE': 0, 'TOKEN_CLS': 101, 'TOKEN_SEP': 102}, None, ['BM25(title)'], None, None, None, None, None, None, None)
       
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
        RankProfile('bert', 'bm25(title) + bm25(body)', 'default', {'TOKEN_NONE': 0, 'TOKEN_CLS': 101, 'TOKEN_SEP': 102}, [Function('question_length', 'sum(map(query(query_token_ids), f(a)(a > 0)))', None), Function('doc_length', 'sum(map(attribute(doc_token_ids), f(a)(a > 0)))', None)], ['question_length', 'doc_length'], None, SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10), None, None, None, None, None)

        >>> RankProfile(
        ...     name = "default",
        ...     first_phase = "nativeRank(title, body)",
        ...     weight = [("title", 200), ("body", 100)]
        ... )
        RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, [('title', 200), ('body', 100)], None, None, None)
        
        >>> RankProfile(
        ...     name = "default",
        ...     first_phase = "nativeRank(title, body)",
        ...     rank_type = [("body", "about")]
        ... )
        RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, None, [('body', 'about')], None, None)
        
        >>> RankProfile(
        ...     name = "default",
        ...     first_phase = "nativeRank(title, body)",
        ...     rank_properties = [("fieldMatch(title).maxAlternativeSegmentations", "10")]
        ... )
        RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, None, None, [('fieldMatch(title).maxAlternativeSegmentations', '10')], None)
        
        >>> RankProfile(
        ...    name = "default",
        ...    first_phase = FirstPhaseRanking(expression="nativeRank(title, body)", keep_rank_count=50)
        ... )
        RankProfile('default', FirstPhaseRanking('nativeRank(title, body)', 50, None), None, None, None, None, None, None, None, None, None, None, None)
        
        """
        self.name = name
        self.first_phase = first_phase
        self.inherits = kwargs.get("inherits", inherits)
        self.constants = kwargs.get("constants", constants)
        self.functions = kwargs.get("functions", functions)
        self.summary_features = kwargs.get("summary_features", summary_features)
        self.match_features = kwargs.get("match_features", match_features)
        self.second_phase = kwargs.get("second_phase", second_phase)
        self.global_phase = kwargs.get("global_phase", global_phase)
        self.weight = kwargs.get("weight", None)
        self.rank_type = kwargs.get("rank_type", None)
        self.rank_properties = kwargs.get("rank_properties", None)
        self.inputs = kwargs.get("inputs", None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.name == other.name
            and self.first_phase == other.first_phase
            and self.inherits == other.inherits
            and self.constants == other.constants
            and self.functions == other.functions
            and self.summary_features == other.summary_features
            and self.match_features == other.match_features
            and self.second_phase == other.second_phase
            and self.global_phase == other.global_phase
            and self.weight == other.weight
            and self.rank_type == other.rank_type
            and self.rank_properties == other.rank_properties
            and self.inputs == other.inputs
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.first_phase),
            repr(self.inherits),
            repr(self.constants),
            repr(self.functions),
            repr(self.summary_features),
            repr(self.match_features),
            repr(self.second_phase),
            repr(self.global_phase),
            repr(self.weight),
            repr(self.rank_type),
            repr(self.rank_properties),
            repr(self.inputs),
        )


class OnnxModel(object):
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
        `Vespa documentation <https://docs.vespa.ai/en/onnx.html>`__
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.model_name == other.model_name
            and self.model_file_path == other.model_file_path
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4})".format(
            self.__class__.__name__,
            repr(self.model_name),
            repr(self.model_file_path),
            repr(self.inputs),
            repr(self.outputs),
        )

class SchemaConfiguration(TypedDict, total=False):
    stemming: Optional[str]

class Schema(object):
    def __init__(
        self,
        name: str,
        document: Document,
        fieldsets: Optional[List[FieldSet]] = None,
        rank_profiles: Optional[List[RankProfile]] = None,
        models: Optional[List[OnnxModel]] = None,
        global_document: bool = False,
        imported_fields: Optional[List[ImportedField]] = None,
        document_summaries: Optional[List[DocumentSummary]] = None,
        mode: Optional[str] = "index",
        inherits: Optional[str] = None,
        **kwargs: Unpack[SchemaConfiguration],
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
        :param document_summaries: A list of :class:`DocumentSummary` associated with the schema.
        :param mode: Schema mode. Defaults to 'index'. Other options are 'store-only' and 'streaming'.
        :param inherits: Schema to inherit from.
        :key stemming: The default stemming setting. Defaults to 'best'.

        To create a Schema:

        >>> Schema(name="schema_name", document=Document())
        Schema('schema_name', Document(None, None, None), None, None, [], False, None, [], None)
        """
        self.name = name
        self.document = document
        self.global_document = global_document
        self.inherits = inherits

        if mode not in ["index", "store-only", "streaming"]:
            raise ValueError(
                "Invalid mode: {0}. Valid options are 'index', 'store-only' and 'streaming'.".format(
                    mode
                )
            )
        self.mode = mode

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

        self.document_summaries = (
            [] if document_summaries is None else list(document_summaries)
        )

        self.stemming = kwargs.get("stemming", None)

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

    def add_document_summary(self, document_summary: DocumentSummary) -> None:
        """
        Add a :class:`DocumentSummary` to the Schema.

        :param document_summary: document summary to be added.
        """
        self.document_summaries.append(document_summary)

    @property
    def schema_to_text(self) -> str:
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
            schema=self,
            document=self.document,
            fieldsets=self.fieldsets,
            rank_profiles=self.rank_profiles,
            models=self.models,
            imported_fields=self.imported_fields,
            document_summaries=self.document_summaries,
            stemming=self.stemming,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.name == other.name
            and self.document == other.document
            and self.fieldsets == other.fieldsets
            and self.rank_profiles == other.rank_profiles
            and self.models == other.models
            and self.global_document == other.global_document
            and self.imported_fields == other.imported_fields
            and self.document_summaries == other.document_summaries
            and self.stemming == other.stemming
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9})".format(
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
            repr(self.document_summaries),
            repr(self.stemming),
        )


class QueryTypeField(object):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name and self.type == other.type

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.type),
        )


class QueryProfileType(object):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.fields == other.fields

    def __repr__(self) -> str:
        return "{0}({1})".format(
            self.__class__.__name__, repr(self.fields) if self.fields else None
        )


class QueryField(object):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name and self.value == other.value

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.value),
        )


class QueryProfile(object):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.fields == other.fields

    def __repr__(self) -> str:
        return "{0}({1})".format(
            self.__class__.__name__, repr(self.fields) if self.fields else None
        )


class ApplicationConfiguration(object):
    def __init__(self, name: str, value: Union[str, Dict[str, Union[Dict, str]]]) -> str:
        """
        Create a Vespa Schema.

        Check the `Config documentation <https://docs.vespa.ai/en/reference/services.html#generic-config>`__
        for more detailed information about generic configuration.

        :param name: Configuration name.
        :param value: Either a string or a Dict (it may be a nested dict) of values.

        Example:

        >>> ApplicationConfiguration(
        ...     name="container.handler.observability.application-userdata",
        ...     value={"version": "my-version"}
        ... )
        ApplicationConfiguration(name="container.handler.observability.application-userdata")
        """
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name=\"{self.name}\")"

    def __get_tab(self, n: int = 1) -> str:
        return " " * 4 * n

    def __to_xml_string(self, xml_elements: Dict[str, Union[Dict, str]], level=0) -> str:
        string = "\n"

        for tag, value in xml_elements.items():
            tabs = self.__get_tab(level)

            if isinstance(value, dict):
                value = self.__to_xml_string(value, level + 1)
                string += f"{tabs}<{tag}>{value}{tabs}</{tag}>\n"
            else:
                string += f"{tabs}<{tag}>{value}</{tag}>\n"
        return string

    @property
    def to_text(self) -> str:
        value = self.__get_tab() + self.__to_xml_string(self.value, level=1) if isinstance(self.value, dict) else self.value
        return f"<config name=\"{self.name}\">{value}</config>"


class Parameter(object):
    def __init__(self,
        name: str,
        args: Optional[Dict[str, str]] = None,
        children: Optional[Union[str, List["Parameter"]]] = None,
        ) -> None:
        """
        Create a Vespa Component configuration parameter.

        :param name: Parameter name.
        :param args: Parameter arguments.
        :param children: Parameter children. Can be either a string or a list of :class:`Parameter` for nested configs.
        """
        self.name = name
        self.args = args
        self.children = children

    def to_xml(self, root) -> ET.Element:
        xml = ET.SubElement(root, self.name)
        [xml.set(k, v) for k,v in self.args.items()]
        if self.children:
            if isinstance(self.children, str):
                xml.text = self.children
            elif isinstance(self.children, List):
                for child in self.children:
                    child.to_xml(xml)
        return xml

class AuthClient(object):
    def __init__(self,
        id: str,
        permissions: List[str],
        parameters: Optional[List[Parameter]] = None,
        )-> None:
        self.id = id
        self.permissions = permissions
        self.parameters = parameters

    def to_xml_string(self, indent: int = 1) -> str:
        root = ET.Element("client")
        root.set("id", self.id)
        root.set("permissions", ",".join(self.permissions))
        if self.parameters:
            for param in self.parameters:
                param.to_xml(root)
        xml_lines = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" " * 4).strip().split("\n")
        return "\n".join([xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]])

    def __repr__(self) -> str:
        id = f"id=\"{self.id}\""
        permissions = f", permissions=\"{self.permissions}\"" if self.permissions else ""
        return f"{self.__class__.__name__}({id}{permissions})"

class Component(object):
    def __init__(self,
        id: str,
        cls: Optional[str] = None,
        bundle: Optional[str] = None,
        type: Optional[str] = None,
        parameters: Optional[List[Parameter]] = None,
        ) -> None:
        """
        Create a Vespa Component.

        Can be used both for embedders (https://docs.vespa.ai/en/reference/embedding-reference.html)
        and generic components (https://docs.vespa.ai/en/reference/services-container.html#component).

        Please see the Vespa documention for more information.

        :param id: The component id.
        :param cls: Component class.
        :param bundle: Component bundle.
        :param type: Component type.
        :param parameters: Component configuration parameters.

        Example:

        >>> Component(id="hf-embedder", type="hugging-face-embedder",
        ...           parameters=[
        ...               Parameter("transformer-model", {"path": "my-models/model.onnx"}),
        ...               Parameter("tokenizer-model", {"path": "my-models/tokenizer.onnx"}),
        ...           ])
        Component(id="hf-embedder", type="hugging-face-embedder")
        """
        self.id = id
        self.cls = cls
        self.bundle = bundle
        self.type = type
        self.parameters = parameters

    def __repr__(self) -> str:
        id = f"id=\"{self.id}\""
        cls = f", class=\"{self.cls}\"" if self.cls else ""
        bundle = f", bundle=\"{self.bundle}\"" if self.bundle else ""
        type = f", type=\"{self.type}\"" if self.type else ""
        return f"{self.__class__.__name__}({id}{cls}{bundle}{type})"
    def to_xml(self, root) -> ET.Element:
        xml = ET.SubElement(root, "component")
        xml.set("id", self.id)
        if self.cls:
            xml.set("class", self.cls)
        if self.bundle:
            xml.set("bundle", self.bundle)
        if self.type:
            xml.set("type", self.type)
        if self.parameters:
            for param in self.parameters:
                param.to_xml(xml)

        return root

    def to_xml_string(self, indent: int = 1) -> str:
        root = ET.Element("root") # Add temporary root (needed by to_xml())
        self.to_xml(root)
        root = root.find("component") # Strip away temporary root

        # Fix indentation, except for the first line (to fit in template), and filter out xml declaration
        xml_lines = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" " * 4).strip().split("\n")
        return "\n".join([xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]])


class Nodes(object):
    def __init__(self,
                 count: Optional[str] = "1",
                 parameters: Optional[List[Parameter]] = None,
                 ) -> None:
        """
        Specify node resources for a content or container cluster as part of a :class: `ContainerCluster` or :class: `ContentCluster`.

        :param count: Number of nodes in a cluster.
        :param parameters: List of :class: `Parameter`s defining the configuration of the cluster resources.

        Example:

        >>> ContainerCluster(id="example_container",
        ...    nodes=Nodes(
        ...        count="2",
        ...        parameters=[
        ...            Parameter("resources", {"vcpu": "4.0", "memory": "16Gb", "disk": "125Gb"},
        ...            [Parameter("gpu", {"count": "1", "memory": "16Gb"})]),
        ...            Parameter("node", {"hostalias": "node1", "distribution-key": "0"}),
        ...        ]
        ...    )
        ... )
        ContainerCluster(id="example_container", version="1.0", nodes="Nodes(count="2")")
        """
        self.count = count
        self.parameters = parameters

    def __repr__(self) -> str:
        count = f"count=\"{self.count}\""
        return f"{self.__class__.__name__}({count})"

    def to_xml(self, root) -> ET.Element:
        xml = ET.SubElement(root, "nodes")
        xml.set("count", self.count)

        if self.parameters:
            for param in self.parameters:
                param.to_xml(xml)

        return root


class Cluster(object):
    def __init__(self,
                 id: str,
                 version: str = "1.0",
                 nodes: Optional[Nodes] = None,
                 ) -> None:
        """
        Base class for a cluster configuration. Should not be instantiated directly.
        Use subclasses :class: `ContainerCluster` or :class: `ContentCluster` instead.

        :param id: Cluster id
        :param version: Cluster version.
        :param nodes: :class: `Nodes` that specifies node resources.
        """
        self.id = id
        self.version = version
        self.nodes = nodes

    def __repr__(self) -> str:
        id = f"id=\"{self.id}\""
        version = f", version=\"{self.version}\""
        nodes = f", nodes=\"{self.nodes}\"" if self.nodes else ""
        return f"{self.__class__.__name__}({id}{version}{nodes}"

    def to_xml(self, root):
        """Set up XML elements that are used in both container and content clusters."""
        root.set("id", self.id)
        root.set("version", self.version)

        if self.nodes:
            self.nodes.to_xml(root)


class ContainerCluster(Cluster):
    def __init__(self,
                 id: str,
                 version: str = "1.0",
                 nodes: Optional[Nodes] = None,
                 components: Optional[List[Component]] = None
                 ) -> None:
        """
        Defines the configuration of a container cluster.

        :param components: List of :class:`Component` that contains configurations for application components, e.g. embedders.

        If :class: `ContainerCluster` is used, any :class: `Component`s must be added to the :class: `ContainerCluster`,
        rather than to the :class: `ApplicationPackage`, in order to be included in the generated schema.

        Example:

        >>> ContainerCluster(id="example_container",
        ...    components=[Component(id="e5", type="hugging-face-embedder",
        ...        parameters=[
        ...            Parameter("transformer-model", {"url": "https://github.com/vespa-engine/sample-apps/raw/master/simple-semantic-search/model/e5-small-v2-int8.onnx"}),
        ...            Parameter("tokenizer-model", {"url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/tokenizer.json"})
        ...        ]
        ...    )]
        ... )
        ContainerCluster(id="example_container", version="1.0", components="[Component(id="e5", type="hugging-face-embedder")]")
        """
        super().__init__(id, version, nodes)
        self.components = components

    def __repr__(self) -> str:
        base_str = super().__repr__()
        components = f", components=\"{self.components}\"" if self.components else ""
        return f"{base_str}{components})"

    def to_xml_string(self, indent=1):
        root = ET.Element("container")
        super().to_xml(root)

        # Add default elements in container
        for child in ["search", "document-api", "document-processing"]:
            ET.SubElement(root, child)

        # Add potential components
        if self.components:
            for comp in self.components:
                comp.to_xml(root)

        # Temporary workaround to get ElementTree to print closing tags.
        # Otherwise it prints <search/>, etc.
        # TODO: Find a permanent solution
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" " * 4)
        for child in ["search", "document-api", "document-processing"]:
            xml_str = xml_str.replace(f'<{child}/>', f'<{child}></{child}>')

        # Indent XML and remove opening tag
        xml_lines = xml_str.strip().split("\n")
        return "\n".join([xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]])


class ContentCluster(Cluster):
    def __init__(self,
                 id: str,
                 document_name: str,
                 version: str = "1.0",
                 nodes: Optional[Nodes] = None,
                 min_redundancy: Optional[str] = "1"
                 ) -> None:
        """
        Defines the configuration of a content cluster.

        :param document_name: Name of document.
        :param min_redundancy: Minimum redundancy of the content cluster. Must be at least 2 for production deployments.

        Example:

        >>> ContentCluster(id="example_content", document_name="doc")
        ContentCluster(id="example_content", version="1.0", document_name="doc")
        """
        super().__init__(id, version, nodes)
        self.document_name = document_name
        self.min_redundancy = min_redundancy

    def __repr__(self) -> str:
        base_str = super().__repr__()
        document_name = f", document_name=\"{self.document_name}\"" if self.document_name else ""
        return f"{base_str}{document_name})"

    def to_xml_string(self, indent=1):
        root = ET.Element("content")
        super().to_xml(root)

        if not self.nodes:
            # Use some sensible defaults if the user doesn't pass a Nodes configuration.
            # The defaults are the ones generated if the Cluster classes are not used at all.
            nodes = ET.SubElement(root, "nodes")
            node = ET.SubElement(nodes, "node")
            node.set("distribution-key", "0")
            node.set("hostalias", "node1")

        ET.SubElement(root, "min-redundancy").text = self.min_redundancy

        documents = ET.SubElement(root, "documents")
        document = ET.SubElement(documents, "document")
        document.set("type", self.document_name)
        document.set("mode", "index")

        # Temporary workaround for expanding tags.
        # minidom's toprettyxml collapses empty tags, even if short_empty_elements is false in ET.tostring()
        # Probably need to pretty print the xml ourselves
        # TODO Find a more permanent solution
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" " * 4)
        xml_str = xml_str.replace('<document type="test" mode="index"/>', '<document type="test" mode="index"></document>')
        xml_str = xml_str.replace('<node distribution-key="0" hostalias="node1"/>', '<node distribution-key="0" hostalias="node1"></node>')

        # Indent XML and remove opening tag
        xml_lines = xml_str.strip().split("\n")
        return "\n".join([xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]])


class ValidationID(Enum):
    """Collection of IDs that can be used in validation-overrides.xml

    Taken from `ValidationId.java <https://github.com/vespa-engine/vespa/blob/master/config-model-api/src/main/java/com/yahoo/config/application/api/ValidationId.java>`__

    clusterSizeReduction was not added as it will be removed in Vespa 9
    """

    indexingChange = "indexing-change"
    """Changing what tokens are expected and stored in field indexes"""
    indexModeChange = "indexing-mode-change"
    """Changing the index mode (streaming, indexed, store-only) of documents"""
    fieldTypeChange = "field-type-change"
    """Field type changes"""
    tensorTypeChange = "tensor-type-change"
    """Tensor type change"""
    resourcesReduction = "resources-reduction"
    """Large reductions in node resources (> 50% of the current max total resources)"""
    contentTypeRemoval = "schema-removal"
    """Removal of a schema (causes deletion of all documents)"""
    contentClusterRemoval = "content-cluster-removal"
    """Removal (or id change) of content clusters"""
    deploymentRemoval = "deployment-removal"
    """Removal of production zones from deployment.xml"""
    globalDocumentChange = "global-document-change"
    """Changing global attribute for document types in content clusters"""
    configModelVersionMismatch = "config-model-version-mismatch"
    """Internal use"""
    skipOldConfigModels = "skip-old-config-models"
    """Internal use"""
    accessControl = "access-control"
    """Internal use, used in zones where there should be no access-control"""
    globalEndpointChange = "global-endpoint-change"
    """Changing global endpoints"""
    zoneEndpointChange = "zone-endpoint-change"
    """Changing zone (possibly private) endpoint settings"""
    redundancyIncrease = "redundancy-increase"
    """Increasing redundancy - may easily cause feed blocked"""
    redundancyOne = "redundancy-one"
    """redundancy=1 requires a validation override on first deployment"""
    pagedSettingRemoval = "paged-setting-removal"
    """May cause content nodes to run out of memory"""
    certificateRemoval = "certificate-removal"
    """Remove data plane certificates"""


class Validation(object):
    def __init__(
        self,
        validation_id: Union[ValidationID, str],
        until: str,
        comment: Optional[str] = None
    ):
        r"""
        Represents a validation to be overridden on application.

        Check the `Vespa documentation <https://docs.vespa.ai/en/reference/validation-overrides.html>`__
        for more detailed information about validations.

        :param validation_id: ID of the validation.
        :param until: The last day this change is allowed, as a ISO-8601-format date in UTC, e.g. 2016-01-30.
            Dates may at most be 30 days in the future, but should be as close to now as possible for safety,
            while allowing time for review and propagation to all deployed zones. allow-tags with dates in the past are
            ignored.
        :param comment: Optional text explaining the reason for the change to humans.
        """

        if isinstance(validation_id, ValidationID):
            self.id = validation_id.value
        else:
            self.id = validation_id
        self.until = until
        self.comment = comment


class DeploymentConfiguration(object):
    def __init__(self, environment: str, regions: List[str]):
        """
        Create a DeploymentConfiguration, which defines how to generate a deployment.xml file (for use in production deployments).

        :param environment: The environment to deploy to. Currently, only 'prod' is supported.
        :param regions: List of regions to deploy to, e.g. [us-east-1, us-west-1].
            See `Vespa documentation <https://cloud.vespa.ai/en/reference/zones.html>`__ for more information.
        """
        self.environment = environment
        self.regions = regions

    def to_xml_string(self, indent=1) -> str:
        root = ET.Element(self.environment)
        for region in self.regions:
            region_xml = ET.SubElement(root, "region")
            region_xml.text = region

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" " * 4)
        xml_lines = xml_str.strip().split("\n")
        return "\n".join([xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]])

class ApplicationPackage(object):
    def __init__(
        self,
        name: str,
        schema: Optional[List[Schema]] = None,
        query_profile: Optional[QueryProfile] = None,
        query_profile_type: Optional[QueryProfileType] = None,
        stateless_model_evaluation: bool = False,
        create_schema_by_default: bool = True,
        create_query_profile_by_default: bool = True,
        configurations: Optional[List[ApplicationConfiguration]] = None,
        validations: Optional[List[Validation]] = None,
        components: Optional[List[Component]] = None,
        auth_clients: Optional[List[AuthClient]] = None,
        clusters: Optional[List[Cluster]] = None,
        deployment_config: Optional[DeploymentConfiguration] = None,
    ) -> None:
        """
        Create an `Application Package <https://docs.vespa.ai/en/application-packages.html>`__.
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
        :param configurations: List of :class:`ApplicationConfiguration` that contains configurations for the application.
        :param validations: Optional list of :class:`Validation` to be overridden.
        :param components: List of :class:`Component` that contains configurations for application components.
        :param clusters: List of :class:`Cluster` that contains configurations for content or container clusters.
            If clusters is used, any :class: `Component`s must be configured as part of a cluster.
        :param clients: List of :class:`Client` that contains configurations for client authorization.
        :param deployment_config: DeploymentConfiguration` that contains configurations for production deployments.

        The easiest way to get started is to create a default application package:

        >>> ApplicationPackage(name="testapp")
        ApplicationPackage('testapp', [Schema('testapp', Document(None, None, None), None, None, [], False, None, [], None)], QueryProfile(None), QueryProfileType(None))

        It will create a default :class:`Schema`, :class:`QueryProfile` and :class:`QueryProfileType` that you can then
        populate with specifics of your application.
        """
        if not name.isalnum():
            raise ValueError(
                "Application package name can only contain [a-zA-Z0-9], was '{}'".format(
                    name
                )
            )
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
        self.models = {}
        self.configurations = configurations
        self.validations = validations
        self.components = components
        self.auth_clients = auth_clients
        self.clusters = clusters
        self.deployment_config = deployment_config

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
        services_template = env.get_template("services.xml")
        return services_template.render(
            application_name=self.name,
            schemas=self.schemas,
            configurations=self.configurations,
            stateless_model_evaluation=self.stateless_model_evaluation,
            components=self.components,
            auth_clients=self.auth_clients,
            clusters=self.clusters
        )

    @property
    def validations_to_text(self):
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
        validations_template = env.get_template("validation-overrides.xml")
        return validations_template.render(validations=self.validations)

    @property
    def deployment_to_text(self):
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
        deployment_template = env.get_template("deployment.xml")
        return deployment_template.render(deployment_config=self.deployment_config)

    @staticmethod
    def _application_package_file_name(disk_folder):
        return os.path.join(disk_folder, "application_package.json")

    def to_zip(self) -> BytesIO:
        """
        Return the application package as zipped bytes,
        to be used in a subsequent deploy
        :return: BytesIO buffer
        """
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "a") as zip_archive:
            zip_archive.writestr("services.xml", self.services_to_text)
            zip_archive.writestr("validation-overrides.xml", self.validations_to_text)

            for schema in self.schemas:
                zip_archive.writestr(
                    "schemas/{}.sd".format(schema.name),
                    schema.schema_to_text,
                )
                for model in schema.models:
                    zip_archive.write(
                        model.model_file_path,
                        "files/{}".format(model.model_file_name),
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

            if self.deployment_config:
                zip_archive.writestr("deployment.xml", self.deployment_to_text)

        buffer.seek(0)
        return buffer

        # ToDo: use this for the Vespa Cloud app package
        # zip_archive.writestr(
        #    "application/security/clients.pem",
        #    app.public_bytes(serialization.Encoding.PEM),
        # )

    def to_zipfile(self, zfile: Path) -> None:
        """
        Export the application package as a deployable zipfile.
        See `application packages <https://docs.vespa.ai/en/application-packages.html>`__
        for deployment options.

        :param zfile: Filename to export to
        :return:
        """
        with open(zfile, "wb") as f:
            f.write(self.to_zip().getbuffer().tobytes())

    def to_files(self, root: Path) -> None:
        """
        Export the application package as a directory tree.

        :param root: Directory to export files to
        :return:
        """
        if not os.path.exists(root):
            Path(root).mkdir(parents=True, exist_ok=True)

        Path(os.path.join(root, "schemas")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, "files")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, "models")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, "search/query-profiles/types")).mkdir(
            parents=True, exist_ok=True
        )

        for schema in self.schemas:
            with open(
                os.path.join(root, "schemas/{}.sd".format(schema.name)), "w"
            ) as f:
                f.write(schema.schema_to_text)
            for model in schema.models:
                copyfile(
                    model.model_file_path,
                    os.path.join(root, "files", model.model_file_name),
                )

        if self.query_profile:
            with open(
                os.path.join(root, "search/query-profiles/default.xml"), "w"
            ) as f:
                f.write(self.query_profile_to_text)
            with open(
                os.path.join(root, "search/query-profiles/types/root.xml"), "w"
            ) as f:
                f.write(self.query_profile_type_to_text)

        with open(os.path.join(root, "services.xml"), "w") as f:
            f.write(self.services_to_text)

        if self.models:
            for model_id, model in self.models.items():
                model.export_to_onnx(
                    output_path=os.path.join(root, "models/{}.onnx".format(model_id))
                )

        if self.validations:
            with open(os.path.join(root, "validation-overrides.xml"), "w") as f:
                f.write(self.validations_to_text)

        if self.deployment_config:
            with open(os.path.join(root, "deployment.xml"), "w") as f:
                f.write(self.deployment_to_text)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name and self._schema == other._schema

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.schemas),
            repr(self.query_profile),
            repr(self.query_profile_type),
        )
