# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import sys
import warnings
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
import zipfile
from collections import OrderedDict
from enum import Enum
from io import BytesIO
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

from jinja2 import Environment, PackageLoader, select_autoescape
from vespa.configuration.vt import Xml, vt
from vespa.configuration.services import services
from vespa.configuration.services import *

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
        Configures a summary field.

        Args:
            name (str, optional): The name of the summary field. Can be `None` if used inside a `Field`, which then uses the name of the `Field`.
            type (str, optional): The type of the summary field. Can be `None` if used inside a `Field`, which then uses the type of the `Field`.
            fields (list): A list of properties used to configure the summary. These can be single properties (like "summary: dynamic", common in `Field`), or composite values (like "source: another_field").

        Example:
            ```py
                Summary(None, None, ["dynamic"])
                Summary(None, None, ['dynamic'])

                Summary(
                    "title",
                    "string",
                    [("source", "title")]
                )
                Summary('title', 'string', [('source', 'title')])

                Summary(
                    "title",
                    "string",
                    [("source", ["title", "abstract"])]
                )
                Summary('title', 'string', [('source', ['title', 'abstract'])])

                Summary(
                    name="artist",
                    type="string",
                )
                Summary('artist', 'string', None)
            ```
        """

        self.name = name
        self.type = type
        self.fields = fields

    @property
    def as_lines(self) -> List[str]:
        """
        Returns the object as a list of strings, where each string represents a line
        of configuration that can be used during schema generation as shown below:

        Example usage:
            ```
                {% for line in field.summary.as_lines %}
                    {{ line }}
                {% endfor %}
            ```

        Example:
            ```python
            Summary(None, None, ["dynamic"]).as_lines
            ['summary: dynamic']
            ```

            ```python
            Summary(
                "artist",
                "string",
            ).as_lines
            ['summary artist type string {}']
            ```

            ```python
            Summary(
                "artist",
                "string",
                [("bolding", "on"), ("sources", "artist")],
            ).as_lines
            ['summary artist type string {', '    bolding: on', '    sources: artist', '}']
            ```
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
                    tmp_string += f"{', '.join(field[1])}"
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
        Configures Vespa HNSW indexes.

        For more information, check the [Vespa documentation](https://docs.vespa.ai/en/approximate-nn-hnsw.html).

        Args:
            distance_metric (str, optional): The distance metric to use when computing distance between vectors. Default is 'euclidean'.
            max_links_per_node (int, optional): Specifies how many links per HNSW node to select when building the graph. Default is 16.
            neighbors_to_explore_at_insert (int, optional): Specifies how many neighbors to explore when inserting a document in the HNSW graph. Default is 200.
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
    rank: str


class StructField:
    def __init__(self, name: str, **kwargs: Unpack[StructFieldConfiguration]) -> None:
        """
        Create a Vespa struct-field.

        For more detailed information about struct-fields, check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#struct-field).

        Args:
            name (str): The name of the struct-field.
            indexing (list, optional): Configures how to process data of a struct-field during indexing.
            attribute (list, optional): Specifies a property of an index structure attribute.
            match (list, optional): Set properties that decide how the matching method for this field operates.
            query_command (list, optional): Add configuration for the query-command of the field.
            summary (Summary, optional): Add configuration for the summary of the field.
            rank (str, optional): Specifies the property that defines ranking calculations done for a field.

        Example:
            ```python
            StructField(
                name = "first_name",
            )
            StructField('first_name', None, None, None, None, None, None)
            ```

            ```python
            StructField(
                name = "first_name",
                indexing = ["attribute"],
                attribute = ["fast-search"],
            )
            StructField('first_name', ['attribute'], ['fast-search'], None, None, None, None)
            ```

            ```python
            StructField(
                name = "last_name",
                match = ["exact", ("exact-terminator", '"@%"')],
                query_command = ['"exact %%"'],
                summary = Summary(None, None, fields=["dynamic", ("bolding", "on")])
            )
            StructField('last_name', None, None, ['exact', ('exact-terminator', '"@%"')], ['"exact %%"'], Summary(None, None, ['dynamic', ('bolding', 'on')]), None)
            ```

            ```python
            StructField(
                name = "first_name",
                indexing = ["attribute"],
                attribute = ["fast-search"],
                rank = "filter",
            )
            StructField('first_name', ['attribute'], ['fast-search'], None, None, None, 'filter')
            ```
        """

        self.name = name
        self.indexing = kwargs.get("indexing", None)
        self.attribute = kwargs.get("attribute", None)
        self.match = kwargs.get("match", None)
        self.query_command = kwargs.get("query_command", None)
        self.summary = kwargs.get("summary", None)
        self.rank = kwargs.get("rank", None)

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
            self.rank,
        ) == (
            other.name,
            other.indexing,
            other.attribute,
            other.match,
            other.query_command,
            other.summary,
            other.rank,
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7})".format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.indexing),
            repr(self.attribute),
            repr(self.match),
            repr(self.query_command),
            repr(self.summary),
            repr(self.rank),
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

        For more detailed information about fields, check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#field).

        Once we have an `ApplicationPackage` instance containing a `Schema` and a `Document`,
        we usually want to add fields so that we can store our data in a structured manner.
        We can accomplish that by creating `Field` instances and adding those to the `ApplicationPackage` instance via `Schema` and `Document` methods.

        Args:
            name (str): The name of the field.
            type (str): The data type of the field.
            indexing (list, optional): Configures how to process data of a field during indexing.
            index (str, optional): Sets index parameters. Fields with index are normalized and tokenized by default.
            attribute (list, optional): Specifies a property of an index structure attribute.
            ann (HNSW, optional): Add configuration for approximate nearest neighbor.
            match (list, optional): Set properties that decide how the matching method for this field operates.
            weight (int, optional): Sets the weight of the field, used when calculating rank scores.
            bolding (bool, optional): Whether to highlight matching query terms in the summary.
            summary (Summary, optional): Add configuration for the summary of the field.
            is_document_field (bool, optional): Whether the field is a document field or part of the schema. Default is True.
            stemming (str, optional): Add configuration for stemming of the field.
            rank (str, optional): Add configuration for ranking calculations of the field.
            query_command (list, optional): Add configuration for query-command of the field.
            struct_fields (list, optional): Add struct-fields to the field.
            alias (list, optional): Add alias to the field.

        Example:
            ```python
            Field(name = "title", type = "string", indexing = ["index", "summary"], index = "enable-bm25")
            Field('title', 'string', ['index', 'summary'], 'enable-bm25', None, None, None, None, None, None, True, None, None, None, [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                indexing = ["attribute"],
                attribute=["fast-search", "fast-access"]
            )
            Field('abstract', 'string', ['attribute'], None, ['fast-search', 'fast-access'], None, None, None, None, None, True, None, None, None, [], None)
            ```

            ```python
            Field(name="tensor_field",
                type="tensor<float>(x[128])",
                indexing=["attribute"],
                ann=HNSW(
                    distance_metric="euclidean",
                    max_links_per_node=16,
                    neighbors_to_explore_at_insert=200,
                ),
            )
            Field('tensor_field', 'tensor<float>(x[128])', ['attribute'], None, None, HNSW('euclidean', 16, 200), None, None, None, None, True, None, None, None, [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                match = ["exact", ("exact-terminator", '"@%"',)],
            )
            Field('abstract', 'string', None, None, None, None, ['exact', ('exact-terminator', '"@%"')], None, None, None, True, None, None, None, [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                weight = 200,
            )
            Field('abstract', 'string', None, None, None, None, None, 200, None, None, True, None, None, None, [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                bolding = True,
            )
            Field('abstract', 'string', None, None, None, None, None, None, True, None, True, None, None, None, [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                summary = Summary(None, None, ["dynamic", ["bolding", "on"]]),
            )
            Field('abstract', 'string', None, None, None, None, None, None, None, Summary(None, None, ['dynamic', ['bolding', 'on']]), True, None, None, None, [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                stemming = "shortest",
            )
            Field('abstract', 'string', None, None, None, None, None, None, None, None, True, 'shortest', None, None, [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                rank = "filter",
            )
            Field('abstract', 'string', None, None, None, None, None, None, None, None, True, None, 'filter', None, [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                query_command = ['"exact %%"'],
            )
            Field('abstract', 'string', None, None, None, None, None, None, None, None, True, None, None, ['"exact %%"'], [], None)
            ```

            ```python
            Field(
                name = "abstract",
                type = "string",
                struct_fields = [
                    StructField(
                        name = "first_name",
                        indexing = ["attribute"],
                        attribute = ["fast-search"],
                    ),
                ],
            )
            Field('abstract', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [StructField('first_name', ['attribute'], ['fast-search'], None, None, None, None)], None)
            ```

            ```python
            Field(
                name = "artist",
                type = "string",
                alias = ["artist_name"],
            )
            Field('artist', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], ['artist_name'])
            ```
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
        Add `StructField`'s to the `Field`.

        Args:
            struct_fields (list): A list of `StructField` objects to be added.
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

        Useful to implement [parent/child relationships](https://docs.vespa.ai/en/parent-child.html).

        Args:
            name (str): Field name.
            reference_field (str): A field of type reference that points to the document that contains the field to be imported.
            field_to_import (str): Field name to be imported, as defined in the reference document.

        Example:
            ```python
            ImportedField(
                name="global_category_ctrs",
                reference_field="category_ctr_ref",
                field_to_import="ctrs",
            )
            ImportedField('global_category_ctrs', 'category_ctr_ref', 'ctrs')
            ```
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
        A struct defines a composite type. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#struct)
        for more detailed information about structs.

        Args:
            name (str): Name of the struct.
            fields (list): List of `Field` objects to be included in the fieldset.

        Example:
            ```python
            Struct("person")
            Struct('person', None)

            Struct(
                "person",
                [
                    Field("first_name", "string"),
                    Field("last_name", "string"),
                ],
            )
            Struct('person', [Field('first_name', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None), Field('last_name', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None)])
            ```
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
        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#document-summary)
        for more detailed information about document-summary.

        Args:
            name (str): Name of the document-summary.
            inherits (str): Name of another document-summary from which this inherits.
            summary_fields (list): List of `Summary` objects used in this document-summary.
            from_disk (bool): Marks this document-summary as accessing fields on disk.
            omit_summary_features (bool): Specifies that summary-features should be omitted from this document summary.

        Example:
            ```python
            DocumentSummary(
                name="document-summary",
            )
            DocumentSummary('document-summary', None, None, None, None)

            DocumentSummary(
                name="which-inherits",
                inherits="base-document-summary",
            )
            DocumentSummary('which-inherits', 'base-document-summary', None, None, None)

            DocumentSummary(
                name="with-field",
                summary_fields=[Summary("title", "string", [("source", "title")])]
            )
            DocumentSummary('with-field', None, [Summary('title', 'string', [('source', 'title')])], None, None)

            DocumentSummary(
                name="with-bools",
                from_disk=True,
                omit_summary_features=True,
            )
            DocumentSummary('with-bools', None, None, True, True)
            ```
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

        Check the [Vespa documentation](https://docs.vespa.ai/en/documents.html)
        for more detailed information about documents.

        Args:
            fields (list): A list of `Field` objects to include in the document's schema.

        Example:
            ```python
            Document()
            Document(None, None, None)

            Document(fields=[Field(name="title", type="string")])
            Document([Field('title', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None)], None, None)

            Document(fields=[Field(name="title", type="string")], inherits="context")
            Document([Field('title', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None)], context, None)
            ```
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
        Add `Field` objects to the document.

        Args:
            fields (list): Fields to be added.

        Returns:
            None
        """
        for field in fields:
            self._fields.update({field.name: field})

    def add_structs(self, *structs: Struct) -> None:
        """
        Add `Struct` objects to the document.

        Args:
            structs (list): Structs to be added.

        Returns:
            None
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
        [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#fieldset)
        for more detailed information about field sets.

        Args:
            name (str): Name of the fieldset.
            fields (list): Field names to be included in the fieldset.

        Returns:
            FieldSet: A field set instance.

        Example:
            ```
            FieldSet(name="default", fields=["title", "body"])
            FieldSet('default', ['title', 'body'])
            ```
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
        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#function-rank)
        for more detailed information about rank functions.

        Args:
            name (str): Name of the function.
            expression (str): String representing a Vespa expression.
            args (list, optional): List of arguments to be used in the function expression. Defaults to None.

        Returns:
            Function: A rank function instance.

        Example:
            ```
                Function(
                    name="myfeature",
                    expression="fieldMatch(bar) + freshness(foo)",
                    args=["foo", "bar"]
                )
                Function('myfeature', 'fieldMatch(bar) + freshness(foo)', ['foo', 'bar'])
            ```

            It is possible to define functions with multi-line expressions:
            ```
                Function(
                    name="token_type_ids",
                    expression="tensor<float>(d0[1],d1[128])(\n"
                               "    if (d1 < question_length,\n"
                               "        0,\n"
                               "    if (d1 < question_length + doc_length,\n"
                               "        1,\n"
                               "        TOKEN_NONE\n"
                               "    )))",
                )
                Function('token_type_ids', 'tensor<float>(d0[1],d1[128])(\n    if (d1 < question_length,\n        0,\n    if (d1 < question_length + doc_length,\n        1,\n        TOKEN_NONE\n    )))', None)
            ```
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
        rank_score_drop_limit: Optional[float] = None,
    ) -> None:
        r"""
        Create a Vespa first phase ranking configuration.

        This is the initial ranking performed on all matching documents.
        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#firstphase-rank)
        for more detailed information about first phase ranking configuration.

        Args:
            expression (str): Specify the ranking expression to be used for the first phase of ranking.
                Check also the [Vespa documentation](https://docs.vespa.ai/en/reference/ranking-expressions.html)
                for ranking expressions.
            keep_rank_count (int, optional): How many documents to keep the first phase top rank values for.
                Default value is 10000.
            rank_score_drop_limit (float, optional): Drop all hits with a first phase rank score less than or equal
                to this floating point number.

        Returns:
            FirstPhaseRanking: A first phase ranking configuration instance.

        Example:
            ```
            FirstPhaseRanking("myFeature * 10")
            FirstPhaseRanking('myFeature * 10', None, None)

            FirstPhaseRanking(expression="myFeature * 10", keep_rank_count=50, rank_score_drop_limit=10)
            FirstPhaseRanking('myFeature * 10', 50, 10)
            ```
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
            repr(self.rank_score_drop_limit),
        )


class SecondPhaseRanking(object):
    def __init__(
        self,
        expression: str,
        rerank_count: int = 100,
        rank_score_drop_limit: Optional[float] = None,
    ) -> None:
        r"""
        Create a Vespa second phase ranking configuration.

        This is the optional reranking performed on the best hits from the first phase.
        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#secondphase-rank)
        for more detailed information about second phase ranking configuration.

        Args:
            expression (str): Specify the ranking expression to be used for the second phase of ranking.
                Check also the [Vespa documentation](https://docs.vespa.ai/en/reference/ranking-expressions.html)
                for ranking expressions.
            rerank_count (int, optional): Specifies the number of hits to be reranked in the second phase.
                Default value is 100.
            rank_score_drop_limit (float, optional): Drop all hits with a first phase rank score less than or equal
                to this floating point number.

        Returns:
            SecondPhaseRanking: A second phase ranking configuration instance.

        Example:
            ```
            SecondPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10)
            SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, None)

            SecondPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10, rank_score_drop_limit=5)
            SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, 5)
            ```
        """
        self.expression = expression
        self.rerank_count = rerank_count
        self.rank_score_drop_limit = rank_score_drop_limit

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.expression == other.expression
            and self.rerank_count == other.rerank_count
            and self.rank_score_drop_limit == other.rank_score_drop_limit
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.expression),
            repr(self.rerank_count),
            repr(self.rank_score_drop_limit),
        )


class GlobalPhaseRanking(object):
    def __init__(
        self,
        expression: str,
        rerank_count: int = 100,
        rank_score_drop_limit: Optional[float] = None,
    ) -> None:
        r"""
        Create a Vespa global phase ranking configuration.

        This is the optional reranking performed on the best hits from the content nodes phase(s).
        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#globalphase-rank)
        for more detailed information about global phase ranking configuration.

        Args:
            expression (str): Specify the ranking expression to be used for the global phase of ranking.
                Check also the [Vespa documentation](https://docs.vespa.ai/en/reference/ranking-expressions.html)
                for ranking expressions.
            rerank_count (int, optional): Specifies the number of hits to be reranked in the global phase.
                Default value is 100.
            rank_score_drop_limit (float, optional): Drop all hits with a first phase rank score less than or equal
                to this floating point number.

        Returns:
            GlobalPhaseRanking: A global phase ranking configuration instance.

        Example:
            ```
                GlobalPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10)
                GlobalPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, None)

                GlobalPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10, rank_score_drop_limit=5)
                GlobalPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, 5)
            ```
        """

        self.expression = expression
        self.rerank_count = rerank_count
        self.rank_score_drop_limit = rank_score_drop_limit

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.expression == other.expression
            and self.rerank_count == other.rerank_count
            and self.rank_score_drop_limit == other.rank_score_drop_limit
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.expression),
            repr(self.rerank_count),
            repr(self.rank_score_drop_limit),
        )


class Mutate(object):
    def __init__(
        self,
        on_match: Union[Dict, None],
        on_first_phase: Union[Dict, None],
        on_second_phase: Union[Dict, None],
        on_summary: Union[Dict, None],
    ) -> None:
        r"""
        Enable mutating operations in rank profiles.

        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#mutate)
        for more detailed information about mutable attributes.

        Args:
            on_match (dict, optional): Dictionary for the on-match phase containing 3 mandatory keys:
                - `attribute`: name of the mutable attribute to mutate.
                - `operation_string`: operation to perform on the mutable attribute.
                - `operation_value`: number to set, add, or subtract to/from the current value of the mutable attribute.
            on_first_phase (dict, optional): Dictionary for the on-first-phase phase containing 3 mandatory keys:
                - `attribute`: name of the mutable attribute to mutate.
                - `operation_string`: operation to perform on the mutable attribute.
                - `operation_value`: number to set, add, or subtract to/from the current value of the mutable attribute.
            on_second_phase (dict, optional): Dictionary for the on-second-phase phase containing 3 mandatory keys:
                - `attribute`: name of the mutable attribute to mutate.
                - `operation_string`: operation to perform on the mutable attribute.
                - `operation_value`: number to set, add, or subtract to/from the current value of the mutable attribute.
            on_summary (dict, optional): Dictionary for the on-summary phase containing 3 mandatory keys:
                - `attribute`: name of the mutable attribute to mutate.
                - `operation_string`: operation to perform on the mutable attribute.
                - `operation_value`: number to set, add, or subtract to/from the current value of the mutable attribute.

        Example:
            ```python
            enable_mutating_operations(
                on_match={
                    'attribute': 'popularity',
                    'operation_string': 'add',
                    'operation_value': 5
                },
                on_first_phase={
                    'attribute': 'score',
                    'operation_string': 'subtract',
                    'operation_value': 3
                }
            )
            enable_mutating_operations({'attribute': 'popularity', 'operation_string': 'add', 'operation_value': 5},
                                        {'attribute': 'score', 'operation_string': 'subtract', 'operation_value': 3})
            ```

        """
        if on_match:
            self.on_match = True
            self.on_match_attribute = on_match["attribute"]
            self.on_match_operation_string = on_match["operation_string"]
            self.on_match_operation_value = on_match["operation_value"]
        else:
            self.on_match = False
        if on_first_phase:
            self.on_first_phase = True
            self.on_first_phase_attribute = on_first_phase["attribute"]
            self.on_first_phase_operation_string = on_first_phase["operation_string"]
            self.on_first_phase_operation_value = on_first_phase["operation_value"]
        else:
            self.on_first_phase = False
        if on_second_phase:
            self.on_second_phase = True
            self.on_second_phase_attribute = on_second_phase["attribute"]
            self.on_second_phase_operation_string = on_second_phase["operation_string"]
            self.on_second_phase_operation_value = on_second_phase["operation_value"]
        else:
            self.on_second_phase = False
        if on_summary:
            self.on_summary = True
            self.on_summary_attribute = on_summary["attribute"]
            self.on_summary_operation_string = on_summary["operation_string"]
            self.on_summary_operation_value = on_summary["operation_value"]
        else:
            self.on_summary = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.on_match == other.on_match
            and self.on_first_phase == other.on_first_phase
            and self.on_second_phase == other.on_second_phase
            and self.on_summary == other.on_summary
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4})".format(
            self.__class__.__name__,
            repr(self.on_match),
            repr(self.on_first_phase),
            repr(self.on_second_phase),
            repr(self.on_summary),
        )


class MatchPhaseRanking(object):
    def __init__(
        self, attribute: str, order: Literal["ascending", "descending"], max_hits: int
    ) -> None:
        r"""
        Create a Vespa match phase ranking configuration.

        This is an optional phase that can be used to quickly select a subset of hits for further ranking.
        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#match-phase)
        for more detailed information about match phase ranking configuration.

        Args:
            attribute (str): The numeric attribute to use for filtering.
            order (str): The sort order, either "ascending" or "descending".
            max_hits (int): Maximum number of hits to pass to the next phase.

        Example:
            ```python
            MatchPhaseRanking(attribute="popularity", order="descending", max_hits=1000)
            MatchPhaseRanking('popularity', 'descending', 1000)
            ```
        """
        self.attribute = attribute
        self.order = order
        self.max_hits = max_hits

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.attribute == other.attribute
            and self.order == other.order
            and self.max_hits == other.max_hits
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            repr(self.attribute),
            repr(self.order),
            repr(self.max_hits),
        )


class RankProfileFields(TypedDict, total=False):
    inherits: str
    constants: Dict
    functions: List[Function]
    summary_features: List
    match_features: List
    match_phase: MatchPhaseRanking
    second_phase: SecondPhaseRanking
    global_phase: GlobalPhaseRanking
    weight: List[Tuple[str, int]]
    rank_type: List[Tuple[str, str]]
    rank_properties: List[Tuple[str, str]]
    inputs: List[Union[Tuple[str, str], Tuple[str, str, str]]]
    mutate: Mutate
    filter_threshold: float
    weakand: Dict[str, float]  # <-- NEW: weakand parameters


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
        match_phase: Optional[MatchPhaseRanking] = None,
        num_threads_per_search: Optional[int] = None,
        **kwargs: Unpack[RankProfileFields],
    ) -> None:
        r"""
        Create a Vespa rank profile.

        Rank profiles are used to specify an alternative ranking of the same data for different purposes, and to experiment with new rank settings.
        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#rank-profile)
        for more detailed information about rank profiles.

        Args:
            name (str): Rank profile name.
            first_phase (str): The config specifying the first phase of ranking.
                [More info](https://docs.vespa.ai/en/reference/schema-reference.html#firstphase-rank) about first phase ranking.
            inherits (str, optional): The inherits attribute is optional. If defined, it contains the name of another rank profile
                in the same schema. Values not defined in this rank profile will then be inherited.
            constants (dict, optional): Dict of constants available in ranking expressions, resolved and optimized at configuration time.
                [More info](https://docs.vespa.ai/en/reference/schema-reference.html#constants) about constants.
            functions (list, optional): List of `Function` objects representing rank functions to be included in the rank profile.
            summary_features (list, optional): List of rank features to be included with each hit.
                [More info](https://docs.vespa.ai/en/reference/schema-reference.html#summary-features) about summary features.
            match_features (list, optional): List of rank features to be included with each hit.
                [More info](https://docs.vespa.ai/en/reference/schema-reference.html#match-features) about match features.
            second_phase (SecondPhaseRanking, optional): Config specifying the second phase of ranking. See `SecondPhaseRanking`.
            global_phase (GlobalPhaseRanking, optional): Config specifying the global phase of ranking. See `GlobalPhaseRanking`.
            match_phase (MatchPhaseRanking, optional): Config specifying the match phase of ranking. See `MatchPhaseRanking`.
            num_threads_per_search (int, optional): Overrides the global `persearch` value for this rank profile to a lower value.
            weight (list, optional): A list of tuples containing the field and their weight.
            rank_type (list, optional): A list of tuples containing a field and the rank-type-name.
                [More info](https://docs.vespa.ai/en/reference/schema-reference.html#rank-type) about rank-type.
            rank_properties (list, optional): A list of tuples containing a field and its configuration.
                [More info](https://docs.vespa.ai/en/reference/schema-reference.html#rank-properties) about rank-properties.
            mutate (Mutate, optional): A `Mutate` object containing attributes to mutate on, mutation operation, and value.
                [More info](https://docs.vespa.ai/en/reference/schema-reference.html#mutate) about mutate operation.

        Example:
            ```python
            RankProfile(name = "default", first_phase = "nativeRank(title, body)")
            RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, None, None, None, None, None, None)

            RankProfile(name = "new", first_phase = "BM25(title)", inherits = "default")
            RankProfile('new', 'BM25(title)', 'default', None, None, None, None, None, None, None, None, None, None, None, None)

            RankProfile(
                name = "new",
                first_phase = "BM25(title)",
                inherits = "default",
                constants={"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102},
                summary_features=["BM25(title)"]
            )
            RankProfile('new', 'BM25(title)', 'default', {'TOKEN_NONE': 0, 'TOKEN_CLS': 101, 'TOKEN_SEP': 102}, None, ['BM25(title)'], None, None, None, None, None, None, None, None, None)

            RankProfile(
                name="bert",
                first_phase="bm25(title) + bm25(body)",
                second_phase=SecondPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10),
                inherits="default",
                constants={"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102},
                functions=[
                    Function(
                        name="question_length",
                        expression="sum(map(query(query_token_ids), f(a)(a > 0)))"
                    ),
                    Function(
                        name="doc_length",
                        expression="sum(map(attribute(doc_token_ids), f(a)(a > 0)))"
                    )
                ],
                summary_features=["question_length", "doc_length"]
            )
            RankProfile('bert', 'bm25(title) + bm25(body)', 'default', {'TOKEN_NONE': 0, 'TOKEN_CLS': 101, 'TOKEN_SEP': 102}, [Function('question_length', 'sum(map(query(query_token_ids), f(a)(a > 0)))', None), Function('doc_length', 'sum(map(attribute(doc_token_ids), f(a)(a > 0)))', None)], ['question_length', 'doc_length'], None, SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, None), None, None, None, None, None, None, None)

            RankProfile(
                name = "default",
                first_phase = "nativeRank(title, body)",
                weight = [("title", 200), ("body", 100)]
            )
            RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, None, None, [('title', 200), ('body', 100)], None, None, None)

            RankProfile(
                name = "default",
                first_phase = "nativeRank(title, body)",
                rank_type = [("body", "about")]
            )
            RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, None, None, None, [('body', 'about')], None, None)

            RankProfile(
                name = "default",
                first_phase = "nativeRank(title, body)",
                rank_properties = [("fieldMatch(title).maxAlternativeSegmentations", "10")]
            )
            RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, None, None, None, None, [('fieldMatch(title).maxAlternativeSegmentations', '10')], None)

            RankProfile(
               name = "default",
               first_phase = FirstPhaseRanking(expression="nativeRank(title, body)", keep_rank_count=50)
            )
            RankProfile('default', FirstPhaseRanking('nativeRank(title, body)', 50, None), None, None, None, None, None, None, None, None, None, None, None, None, None)

            RankProfile(
                name = "default",
                first_phase = "nativeRank(title, body)",
                num_threads_per_search = 2
            )
            RankProfile('default', 'nativeRank(title, body)', None, None, None, None, None, None, None, None, 2, None, None, None, None)
            ```
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
        self.match_phase = kwargs.get("match_phase", match_phase)
        self.num_threads_per_search = kwargs.get(
            "num_threads_per_search", num_threads_per_search
        )
        self.weight = kwargs.get("weight", None)
        self.rank_type = kwargs.get("rank_type", None)
        self.rank_properties = kwargs.get("rank_properties", None)
        self.inputs = kwargs.get("inputs", None)
        self.mutate = kwargs.get("mutate", None)
        self.filter_threshold = kwargs.get("filter_threshold", None)
        self.weakand = kwargs.get("weakand", None)  # <-- NEW: store weakand parameters

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
            and self.match_phase == other.match_phase
            and self.num_threads_per_search == other.num_threads_per_search
            and self.weight == other.weight
            and self.rank_type == other.rank_type
            and self.rank_properties == other.rank_properties
            and self.inputs == other.inputs
            and self.mutate == other.mutate
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15})".format(
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
            repr(self.match_phase),
            repr(self.num_threads_per_search),
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

        Vespa has support for advanced ranking models through its tensor API. If you have your model in the ONNX format, Vespa can import the models and use them directly. Check the [Vespa documentation](https://docs.vespa.ai/en/onnx.html) for more detailed information about field sets.

        Args:
            model_name (str): Unique model name to use as an ID when referencing the model.
            model_file_path (str): ONNX model file path.
            inputs (dict): Dict mapping the ONNX input names as specified in the ONNX file to valid Vespa inputs.
                These can be a document field (`attribute(field_name)`), a query parameter (`query(query_param)`), a constant (`constant(name)`), or a user-defined function (`function_name`).
            outputs (dict, optional): Dict mapping the ONNX output names as specified in the ONNX file to the name used in Vespa to specify the output.
                If omitted, the first output in the ONNX file will be used.

        Example:
            ```python
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
            OnnxModel('bert', 'bert.onnx', {'input_ids': 'input_ids', 'token_type_ids': 'token_type_ids', 'attention_mask': 'attention_mask'}, {'logits': 'logits'})
            ```
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

        Check the [Vespa documentation](https://docs.vespa.ai/en/schemas.html) for more detailed information about schemas.

        Args:
            name (str): Schema name.
            document (Document): Vespa `Document` associated with the Schema.
            fieldsets (list, optional): A list of `FieldSet` associated with the Schema.
            rank_profiles (list, optional): A list of `RankProfile` associated with the Schema.
            models (list, optional): A list of `OnnxModel` associated with the Schema.
            global_document (bool, optional): Set to True to copy the documents to all content nodes. Defaults to False.
            imported_fields (list, optional): A list of `ImportedField` defining fields from global documents to be imported.
            document_summaries (list, optional): A list of `DocumentSummary` associated with the schema.
            mode (str, optional): Schema mode. Defaults to 'index'. Other options are 'store-only' and 'streaming'.
            inherits (str, optional): Schema to inherit from.
            stemming (str, optional): The default stemming setting. Defaults to 'best'.

        Example:
            ```python
            Schema(name="schema_name", document=Document())
            Schema('schema_name', Document(None, None, None), None, None, [], False, None, [], None)
            ```
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
        Add `Field` to the Schema's `Document`.

        Args:
            fields (list): A list of `Field` objects to be added to the `Document`.

        Example:
            ```python
            schema.add_fields([Field(name="title", type="string"), Field(name="body", type="text")])
            schema.add_fields([Field('title', 'string'), Field('body', 'text')])
            ```
        """
        self.document.add_fields(*fields)

    def add_field_set(self, field_set: FieldSet) -> None:
        """
        Add a `FieldSet` to the Schema.

        Args:
            field_set (list): A list of `FieldSet` objects to be added to the Schema.
        """
        self.fieldsets[field_set.name] = field_set

    def add_rank_profile(self, rank_profile: RankProfile) -> None:
        """
        Add a `RankProfile` to the Schema.

        Args:
            rank_profile (RankProfile): The rank profile to be added to the Schema.

        Returns:
            None
        """
        self.rank_profiles[rank_profile.name] = rank_profile

    def add_model(self, model: OnnxModel) -> None:
        """
        Add an `OnnxModel` to the Schema.

        Args:
            model (OnnxModel): The ONNX model to be added to the Schema.

        Returns:
            None
        """
        self.models.append(model)

    def add_imported_field(self, imported_field: ImportedField) -> None:
        """
        Add an `ImportedField` to the Schema.

        Args:
            imported_field (ImportedField): The imported field to be added to the Schema.
        """
        self.imported_fields[imported_field.name] = imported_field

    def add_document_summary(self, document_summary: DocumentSummary) -> None:
        """
        Add a `DocumentSummary` to the Schema.

        Args:
            document_summary (DocumentSummary): The document summary to be added to the Schema.
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
        Create a field to be included in a `QueryProfileType`.

        Args:
            name (str): Field name.
            type (str): Field type.

        Example:
            ```python
            QueryTypeField(
                name="ranking.features.query(title_bert)",
                type="tensor<float>(x[768])"
            )
            QueryTypeField('ranking.features.query(title_bert)', 'tensor<float>(x[768])')
            ```
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

        Check the [Vespa documentation](https://docs.vespa.ai/en/query-profiles.html#query-profile-types)
        for more detailed information about query profile types.

        An `ApplicationPackage` instance comes with a default `QueryProfile` named `default`
        that is associated with a `QueryProfileType` named `root`,
        meaning that you usually do not need to create those yourself, only add fields to them when required.

        Args:
            fields (list[QueryTypeField]): A list of `QueryTypeField`.

        Example:
            ```python
            QueryProfileType(
                fields=[
                    QueryTypeField(
                        name="ranking.features.query(tensor_bert)",
                        type="tensor<float>(x[768])"
                    )
                ]
            )
            # Output: QueryProfileType([QueryTypeField('ranking.features.query(tensor_bert)', 'tensor<float>(x[768])')])
            ```
        """
        self.name = "root"
        self.fields = [] if not fields else fields

    def add_fields(self, *fields: QueryTypeField) -> None:
        """
        Add `QueryTypeField` objects to the Query Profile Type.

        Args:
            fields (QueryTypeField): Fields to be added.

        Example:
            ```python
            query_profile_type = QueryProfileType()
            query_profile_type.add_fields(
                QueryTypeField(
                    name="age",
                    type="integer"
                ),
                QueryTypeField(
                    name="profession",
                    type="string"
                )
            )
            ```
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
        Create a field to be included in a `QueryProfile`.

        Args:
            name (str): Field name.
            value (Any): Field value.

        Example:
            ```python
            QueryField(name="maxHits", value=1000)
            # Output: QueryField('maxHits', 1000)
            ```
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

        Check the [Vespa documentation](https://docs.vespa.ai/en/query-profiles.html)
        for more detailed information about query profiles.

        A `QueryProfile` is a named collection of query request parameters given in the configuration.
        The query request can specify a query profile whose parameters will be used as parameters of that request.
        The query profiles may optionally be type-checked.
        Type checking is turned on by referencing a `QueryProfileType` from the query profile.

        Args:
            fields (list[QueryField]): A list of `QueryField`.

        Example:
            ```python
            QueryProfile(fields=[QueryField(name="maxHits", value=1000)])
            # Output: QueryProfile([QueryField('maxHits', 1000)])
            ```
        """
        self.name = "default"
        self.type = "root"
        self.fields = [] if not fields else fields

    def add_fields(self, *fields: QueryField) -> None:
        """
        Add `QueryField` objects to the Query Profile.

        Args:
            fields (QueryField): Fields to be added.

        Example:
            ```python
            query_profile = QueryProfile()
            query_profile.add_fields(QueryField(name="maxHits", value=1000))
            ```
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
    def __init__(
        self, name: str, value: Union[str, Dict[str, Union[Dict, str]]]
    ) -> str:
        """
        Create a Vespa Schema.

        Check the [Config documentation](https://docs.vespa.ai/en/reference/services.html#generic-config)
        for more detailed information about generic configuration.

        Args:
            name (str): Configuration name.
            value (str | dict): Either a string or a dictionary (which may be nested) of values.

        Example:
            ```python
            ApplicationConfiguration(
                name="container.handler.observability.application-userdata",
                value={"version": "my-version"}
            )
            # Output: ApplicationConfiguration(name="container.handler.observability.application-userdata")
            ```
        """
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self.name}")'

    def __get_tab(self, n: int = 1) -> str:
        return " " * 4 * n

    def __to_xml_string(
        self, xml_elements: Dict[str, Union[Dict, str]], level=0
    ) -> str:
        string = "\n"

        for tag, value in xml_elements.items():
            tabs = self.__get_tab(level)

            if isinstance(value, dict):
                value = self.__to_xml_string(value, level + 1)
                string += f"{tabs}<{tag}>{value}{tabs}</{tag}>\n"
            else:
                string += f"{tabs}<{tag}>{value}</{tag}>\n"
        return string

    def to_vt(self) -> VT:
        return config((vt(k, v) for k, v in self.value.items()), name=self.name)

    @property
    def to_text(self) -> str:
        value = (
            self.__get_tab() + self.__to_xml_string(self.value, level=1)
            if isinstance(self.value, dict)
            else self.value
        )
        return f'<config name="{self.name}">{value}</config>'


class Parameter(object):
    def __init__(
        self,
        name: str,
        args: Optional[Dict[str, str]] = None,
        children: Optional[Union[str, List["Parameter"]]] = None,
    ) -> None:
        """
        Create a Vespa Component configuration parameter.

        Args:
            name (str): Parameter name.
            args (Any): Parameter arguments.
            children (str | list[Parameter]): Parameter children. Can be either a string or a list of `Parameter` objects for nested configs.
        """
        self.name = name
        self.args = args
        self.children = children

    def to_xml(self, root) -> ET.Element:
        xml = ET.SubElement(root, self.name)
        [xml.set(k, v) for k, v in self.args.items()]
        if self.children:
            if isinstance(self.children, str):
                xml.text = self.children
            elif isinstance(self.children, List):
                for child in self.children:
                    child.to_xml(xml)
        return xml

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.name == other.name
            and self.args == other.args
            and self.children == other.children
        )

    def to_vt(self) -> VT:
        vt_func = vt(
            self.name,
        )
        if self.args:
            vt_func = vt_func(**self.args)
        if self.children:
            if isinstance(self.children, str):
                vt_func = vt_func(self.children)
            elif isinstance(self.children, List):
                vt_func = vt_func(*[child.to_vt() for child in self.children])
        return vt_func


class AuthClient(object):
    def __init__(
        self,
        id: str,
        permissions: List[str],
        parameters: Optional[List[Parameter]] = None,
    ) -> None:
        """
        Create a Vespa AuthClient.

        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/services-container.html).

        Args:
            id (str): The auth client ID.
            permissions (list[str]): List of permissions.
            parameters (list[Parameter]): List of `Parameter` objects defining the configuration of the auth client.

        Example:
            ```python
            AuthClient(
                id="token",
                permissions=["read", "write"],
                parameters=[Parameter("token", {"id": "my-token-id"})],
            )
            # Output: AuthClient(id="token", permissions="['read', 'write']")
            ```
        """
        self.id = id
        self.permissions = permissions
        self.parameters = parameters

    def to_xml(self, root) -> ET.Element:
        xml = ET.SubElement(root, "client")
        xml.set("id", self.id)
        xml.set("permissions", ",".join(self.permissions))
        if self.parameters:
            for param in self.parameters:
                param.to_xml(xml)
        return root

    def to_xml_string(self, indent: int = 1) -> str:
        root = ET.Element("client")
        root.set("id", self.id)
        root.set("permissions", ",".join(self.permissions))
        if self.parameters:
            for param in self.parameters:
                param.to_xml(root)
        xml_lines = (
            minidom.parseString(ET.tostring(root))
            .toprettyxml(indent=" " * 4)
            .strip()
            .split("\n")
        )
        return "\n".join(
            [xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.id == other.id
            and self.permissions == other.permissions
            and self.parameters == other.parameters
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id < other.id

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id > other.id

    def __repr__(self) -> str:
        id = f'id="{self.id}"'
        permissions = f', permissions="{self.permissions}"' if self.permissions else ""
        return f"{self.__class__.__name__}({id}{permissions})"

    def to_vt(self) -> VT:
        return client(
            *[p.to_vt() for p in self.parameters or []],
            id=self.id,
            permissions=",".join(self.permissions),
        )


class Component(object):
    def __init__(
        self,
        id: str,
        cls: Optional[str] = None,
        bundle: Optional[str] = None,
        type: Optional[str] = None,
        parameters: Optional[List[Parameter]] = None,
    ) -> None:
        def create_component(id, cls, bundle, type, parameters):
            """
            Create a Vespa Component.

            Can be used both for [embedders](https://docs.vespa.ai/en/reference/embedding-reference.html)
            and [generic components](https://docs.vespa.ai/en/reference/services-container.html#component).

            Please see the Vespa documentation for more information.

            Args:
                id (str): The component ID.
                cls (str): Component class.
                bundle (str): Component bundle.
                type (str): Component type.
                parameters (list[Parameter]): Component configuration parameters.

            Example:
                ```python
                Component(
                    id="hf-embedder",
                    type="hugging-face-embedder",
                    parameters=[
                        Parameter("transformer-model", {"path": "my-models/model.onnx"}),
                        Parameter("tokenizer-model", {"path": "my-models/tokenizer.onnx"}),
                    ]
                )
                # Output: Component(id="hf-embedder", type="hugging-face-embedder")
                ```
            """

        self.id = id
        self.cls = cls
        self.bundle = bundle
        self.type = type
        self.parameters = parameters

    def __repr__(self) -> str:
        id = f'id="{self.id}"'
        cls = f', class="{self.cls}"' if self.cls else ""
        bundle = f', bundle="{self.bundle}"' if self.bundle else ""
        type = f', type="{self.type}"' if self.type else ""
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
        root = ET.Element("root")  # Add temporary root (needed by to_xml())
        self.to_xml(root)
        root = root.find("component")  # Strip away temporary root

        # Fix indentation, except for the first line (to fit in template), and filter out xml declaration
        xml_lines = (
            minidom.parseString(ET.tostring(root))
            .toprettyxml(indent=" " * 4)
            .strip()
            .split("\n")
        )
        return "\n".join(
            [xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]]
        )

    def to_vt(self) -> VT:
        return component(
            *[p.to_vt() for p in self.parameters or []],
            **{
                k: v
                for k, v in {
                    "id": self.id,
                    "class": self.cls,
                    "bundle": self.bundle,
                    "type": self.type,
                }.items()
                if v
            },
        )


class Nodes(object):
    def __init__(
        self,
        count: Optional[str] = "1",
        parameters: Optional[List[Parameter]] = None,
    ) -> None:
        """
        Specify node resources for a content or container cluster as part of a `ContainerCluster` or `ContentCluster`.

        Args:
            count (int): Number of nodes in a cluster.
            parameters (list[Parameter]): List of `Parameter` objects defining the configuration of the cluster resources.

        Example:
            ```python
            ContainerCluster(
                id="example_container",
                nodes=Nodes(
                    count="2",
                    parameters=[
                        Parameter(
                            "resources",
                            {"vcpu": "4.0", "memory": "16Gb", "disk": "125Gb"},
                            children=[Parameter("gpu", {"count": "1", "memory": "16Gb"})]
                        ),
                        Parameter("node", {"hostalias": "node1", "distribution-key": "0"}),
                    ]
                )
            )
            # Output: ContainerCluster(id="example_container", version="1.0", nodes="Nodes(count='2')")
            ```
        """
        self.count = count
        self.parameters = parameters

    def __repr__(self) -> str:
        count = f'count="{self.count}"'
        return f"{self.__class__.__name__}({count})"

    def to_xml(self, root) -> ET.Element:
        xml = ET.SubElement(root, "nodes")
        xml.set("count", self.count)

        if self.parameters:
            for param in self.parameters:
                param.to_xml(xml)

        return root

    def to_vt(self) -> VT:
        return [nodes(*[p.to_vt() for p in self.parameters or []], count=self.count)]


class Cluster(object):
    def __init__(
        self,
        id: str,
        version: str = "1.0",
        nodes: Optional[Nodes] = None,
    ) -> None:
        """
        Base class for a cluster configuration. Should not be instantiated directly.
        Use subclasses `ContainerCluster` or `ContentCluster` instead.

        Args:
            id (str): Cluster ID.
            version (str): Cluster version.
            nodes (Nodes): `Nodes` that specifies node resources.
        """
        self.id = id
        self.version = version
        self.nodes = nodes

    def __repr__(self) -> str:
        id = f'id="{self.id}"'
        version = f', version="{self.version}"'
        nodes = f', nodes="{self.nodes}"' if self.nodes else ""
        return f"{self.__class__.__name__}({id}{version}{nodes}"

    def to_xml(self, root):
        """Set up XML elements that are used in both container and content clusters."""
        root.set("id", self.id)
        root.set("version", self.version)

        if self.nodes:
            self.nodes.to_xml(root)


class ContainerCluster(Cluster):
    def __init__(
        self,
        id: str,
        version: str = "1.0",
        nodes: Optional[Nodes] = None,
        components: Optional[List[Component]] = None,
        auth_clients: Optional[List[AuthClient]] = None,
    ) -> None:
        """
        Defines the configuration of a container cluster.

        Args:
            components (list[Component]): List of `Component` that contains configurations for application components, e.g. embedders.
            auth_clients (list[AuthClient]): List of `AuthClient` that contains configurations for authentication clients (e.g., mTLS/token).
            nodes (Nodes): `Nodes` that specifies the resources of the cluster.

        If `ContainerCluster` is used, any `Component`s must be added to the `ContainerCluster`,
        rather than to the `ApplicationPackage`, in order to be included in the generated schema.

        Example:
            ```python
            ContainerCluster(
                id="example_container",
                components=[
                    Component(
                        id="e5",
                        type="hugging-face-embedder",
                        parameters=[
                            Parameter(
                                "transformer-model",
                                {"url": "https://github.com/vespa-engine/sample-apps/raw/master/examples/model-exporting/model/e5-small-v2-int8.onnx"}
                            ),
                            Parameter(
                                "tokenizer-model",
                                {"url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/examples/model-exporting/model/tokenizer.json"}
                            )
                        ]
                    )
                ],
                auth_clients=[AuthClient(id="mtls", permissions=["read", "write"])],
                nodes=Nodes(count="2", parameters=[Parameter("resources", {"vcpu": "4.0", "memory": "16Gb", "disk": "125Gb"})])
            )
            # Output: ContainerCluster(id="example_container", version="1.0", nodes="Nodes(count='2')", components="[Component(id='e5', type='hugging-face-embedder')]", auth_clients="[AuthClient(id='mtls', permissions=['read', 'write'])]")
            ```
        """
        super().__init__(id, version, nodes)
        self.components = components
        self.auth_clients = auth_clients

    def __repr__(self) -> str:
        base_str = super().__repr__()
        components = f', components="{self.components}"' if self.components else ""
        auth_clients = (
            f', auth_clients="{self.auth_clients}"' if self.auth_clients else ""
        )
        return f"{base_str}{components}{auth_clients})"

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

        if self.auth_clients:
            clients = ET.SubElement(root, "clients")
            for client in self.auth_clients:
                client.to_xml(clients)

        # Temporary workaround to get ElementTree to print closing tags.
        # Otherwise it prints <search/>, etc.
        # TODO: Find a permanent solution
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" " * 4)
        for child in ["search", "document-api", "document-processing"]:
            xml_str = xml_str.replace(f"<{child}/>", f"<{child}></{child}>")

        # Indent XML and remove opening tag
        xml_lines = xml_str.strip().split("\n")
        return "\n".join(
            [xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]]
        )

    def to_vt(self) -> VT:
        return container(
            id=self.id,
            version=self.version,
            *self.nodes.to_vt(),
            *[search(), document_api(), document_processing()],
            *[c.to_vt() for c in self.components or []],
            *[
                clients(a.to_vt() for a in self.auth_clients or [])
                if self.auth_clients
                else None
            ],
        )


class ContentCluster(Cluster):
    def __init__(
        self,
        id: str,
        document_name: str,
        version: str = "1.0",
        nodes: Optional[Nodes] = None,
        min_redundancy: Optional[str] = "1",
    ) -> None:
        """
        Defines the configuration of a content cluster.

        Args:
            document_name (str): Name of document.
            min_redundancy (int): Minimum redundancy of the content cluster. Must be at least 2 for production deployments.

        Example:
            ```python
            ContentCluster(id="example_content", document_name="doc")
            # Output: ContentCluster(id="example_content", version="1.0", document_name="doc")
            ```
        """
        super().__init__(id, version, nodes)
        self.document_name = document_name
        self.min_redundancy = min_redundancy

    def __repr__(self) -> str:
        base_str = super().__repr__()
        document_name = (
            f', document_name="{self.document_name}"' if self.document_name else ""
        )
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
        xml_str = xml_str.replace(
            '<document type="test" mode="index"/>',
            '<document type="test" mode="index"></document>',
        )
        xml_str = xml_str.replace(
            '<node distribution-key="0" hostalias="node1"/>',
            '<node distribution-key="0" hostalias="node1"></node>',
        )

        # Indent XML and remove opening tag
        xml_lines = xml_str.strip().split("\n")
        return "\n".join(
            [xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]]
        )

    def to_vt(self) -> VT:
        return content(
            id=self.id,
            version=self.version,
            *self.nodes.to_vt()
            if self.nodes
            else [nodes(node(distribution_key="0", hostalias="node1"))],
            *[min_redundancy(self.min_redundancy)],
            *[documents(document(type=self.document_name, mode="index"))],
        )


class ValidationID(Enum):
    """
    Collection of IDs that can be used in validation-overrides.xml.

    Taken from [ValidationId.java](https://github.com/vespa-engine/vespa/blob/master/config-model-api/src/main/java/com/yahoo/config/application/api/ValidationId.java).

    `clusterSizeReduction` was not added as it will be removed in Vespa 9.
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
        comment: Optional[str] = None,
    ):
        """
        Represents a validation to be overridden on application.

        Check the [Vespa documentation](https://docs.vespa.ai/en/reference/validation-overrides.html)
        for more detailed information about validations.

        Args:
            validation_id (str): ID of the validation.
            until (str): The last day this change is allowed, as an ISO-8601-format date in UTC, e.g. 2016-01-30.
                        Dates may at most be 30 days in the future, but should be as close to now as possible for safety,
                        while allowing time for review and propagation to all deployed zones. `allow-tags` with dates in the past are ignored.
            comment (str, optional): Optional text explaining the reason for the change to humans.

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

        Args:
            environment (str): The environment to deploy to. Currently, only 'prod' is supported.
            regions (list[str]): List of regions to deploy to, e.g. ["us-east-1", "us-west-1"].
                                See [Vespa documentation](https://cloud.vespa.ai/en/reference/zones.html) for more information.

        Example:
            ```python
            DeploymentConfiguration(environment="prod", regions=["us-east-1", "us-west-1"])
            # Output: DeploymentConfiguration(environment='prod', regions=['us-east-1', 'us-west-1'])
            ```
        """
        self.environment = environment
        self.regions = regions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(environment='{self.environment}', regions={self.regions})"

    def to_xml_string(self, indent=1) -> str:
        root = ET.Element(self.environment)
        for region in self.regions:
            region_xml = ET.SubElement(root, "region")
            region_xml.text = region

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" " * 4)
        xml_lines = xml_str.strip().split("\n")
        return "\n".join(
            [xml_lines[1]] + [(" " * 4 * indent) + line for line in xml_lines[2:]]
        )


class EmptyDeploymentConfiguration(DeploymentConfiguration):
    def __init__(self):
        """
        Create an EmptyDeploymentConfiguration, which creates an empty deployment.xml, used to delete production deployments.
        """
        super().__init__("", [])

    def to_xml_string(
        self, indent=1
    ) -> str:  # Indent is unused, but included for compatibility
        return ""


class ServicesConfiguration(object):
    def __init__(
        self,
        application_name: str,
        schemas: Optional[List[Schema]] = None,
        configurations: List[ApplicationConfiguration] = [],
        stateless_model_evaluation: Optional[bool] = False,
        components: List[Component] = [],
        auth_clients: List[AuthClient] = [],
        clusters: List[Cluster] = [],
        services_config: Optional[VT] = None,
    ) -> None:
        """
        Create a ServicesConfiguration, adopting the VespaTag (VT) approach, rather than Jinja templates.
        Intended to be used in ApplicationPackage, to generate services.xml, based on either:
        - A passed `services_config` (VT) object, or
        - A set of configurations, schemas, components, auth_clients, and clusters (equivalent to the old approach).

        The latter will be done in code by calling `build_services_vt()` to generate the VT object.

        Args:
            application_name (str): Application name.
            schemas (Optional[List[Schema]]): List of `Schema`s of the application.
            configurations (Optional[List[ApplicationConfiguration]]): List of `ApplicationConfiguration` that contains configurations for the application.
            stateless_model_evaluation (Optional[bool]): Enable stateless model evaluation. Default is False.
            components (Optional[List[Component]]): List of `Component` that contains configurations for application components.
            auth_clients (Optional[List[AuthClient]]): List of `AuthClient` that contains configurations for authentication clients.
            clusters (Optional[List[Cluster]]): List of `Cluster` that contains configurations for content or container clusters.
            services_config (Optional[VT]): `VT` object that contains the services configuration.

        Example:
            ```python
            config = ServicesConfiguration(
                application_name="myapp",
                schemas=[Schema(name="myschema", document=Document())],
                configurations=[ApplicationConfiguration(name="container.handler.observability.application-userdata", value={"version": "my-version"})],
                components=[Component(id="hf-embedder", type="huggingface-embedder")],
                stateless_model_evaluation=True,
            )
            print(str(config))
            # Output: <?xml version="1.0" encoding="UTF-8" ?>
            # <services version="1.0">...</services>

            services_config = ServicesConfiguration(
                application_name="myapp",
                services_config=services(
                    container(id="myapp_default", version="1.0")(
                        component(
                            model(url="https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/raw/main/tokenizer.json"),
                            id="tokenizer", type="hugging-face-tokenizer"
                        ),
                        document_api(),
                        search(),
                    ),
                    content(id="myapp", version="1.0")(
                        min_redundancy("1"),
                        documents(document(type="doc", mode="index")),
                        engine(proton(tuning(searchnode(requestthreads(persearch("4"))))))
                    ),
                    version="1.0", minimum_required_vespa_version="8.311.28",
                ),
            )
            print(str(services_config))
            # Output: <?xml version="1.0" encoding="UTF-8" ?>
            # <services version="1.0" minimum-required-vespa-version="8.311.28">...</services>
            ```

        """
        self.application_name = application_name
        self.schemas = schemas or []
        self.configurations = configurations
        self.stateless_model_evaluation = stateless_model_evaluation
        self.components = components or []
        self.auth_clients = auth_clients
        self.clusters = clusters
        self.services_config = services_config or self.build_services_vt()

    def build_services_vt(self):
        services_vt = services(version="1.0")

        # Handle configurations
        for config in self.configurations:
            services_vt += config.to_vt()

        # Handle clusters
        if self.clusters:
            for cluster in self.clusters:
                services_vt += cluster.to_vt()
        else:
            # Default container
            container_id = f"{self.application_name}_container"
            container_vt = container(id=container_id, version="1.0")

            if self.schemas:
                container_vt += search()
                container_vt += document_api()
                container_vt += document_processing()

            for comp in self.components:
                container_vt += comp.to_vt()

            if self.auth_clients:
                clients_vt = clients()
                for client in self.auth_clients:
                    clients_vt += client.to_vt()
                container_vt += clients_vt
            if self.stateless_model_evaluation:
                container_vt += model_evaluation()

            services_vt += container_vt

            # Content cluster
            if self.schemas:
                content_id = f"{self.application_name}_content"
                content_vt = content(id=content_id, version="1.0")
                content_vt += redundancy("1")

                documents_vt = documents()
                streaming_modes_total = 0
                for schema in self.schemas:
                    if getattr(schema, "global_document", False):
                        documents_vt += document(
                            type=schema.name, mode="index", _global="true"
                        )
                    else:
                        documents_vt += document(type=schema.name, mode=schema.mode)
                    if schema.mode == "streaming":
                        streaming_modes_total += 1
                if streaming_modes_total > 0:
                    documents_vt += document_processing(
                        chain="indexing", cluster=container_id
                    )

                content_vt += documents_vt

                nodes_vt = nodes()
                nodes_vt += node(distribution_key="0", hostalias="node1")
                content_vt += nodes_vt

                services_vt += content_vt

        return services_vt

    def __str__(self) -> str:
        return (str(Xml().to_xml()) + str(self.services_config.to_xml())).rstrip("\n")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(services_config={self.services_config})"

    def _repr_markdown_(self):
        return

    def validate(self):
        return validate_services(str(self.services_config.to_xml()))


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
        services_config: Optional[ServicesConfiguration] = None,
    ) -> None:
        """Create an application package.

        Args:
            name (str): Application name. Cannot contain '-' or '_'.
            schema (list, optional): List of Schema objects for the application. If None, a default Schema
                with the same name as the application will be created. Defaults to None.
            query_profile (QueryProfile, optional): QueryProfile of the application. If None, a default
                QueryProfile with QueryProfileType 'root' will be created. Defaults to None.
            query_profile_type (QueryProfileType, optional): QueryProfileType of the application. If None,
                a default QueryProfileType 'root' will be created. Defaults to None.
            stateless_model_evaluation (bool, optional): Enable stateless model evaluation. Defaults to False.
            create_schema_by_default (bool, optional): Include a default Schema if none is provided in the schema
                argument. Defaults to True.
            create_query_profile_by_default (bool, optional): Include a default QueryProfile and QueryProfileType
                if not explicitly defined by the user. Defaults to True.
            configurations (list, optional): List of ApplicationConfiguration for the application. Defaults to None.
            validations (list, optional): Optional list of Validation objects to be overridden. Defaults to None.
            components (list, optional): List of Component objects for application components. Defaults to None.
            clusters (list, optional): List of Cluster objects for content or container clusters. If clusters is provided,
                any Component must be part of a cluster. Defaults to None.
            auth_clients (list, optional): List of AuthClient objects for client authorization. If clusters is passed,
                pass the auth clients to the ContainerCluster instead. Defaults to None.
            deployment_config (DeploymentConfiguration, optional): Configuration for production deployments. Defaults to None.

        Example:
            To create a default application package:

            ```python
            ApplicationPackage(name="testapp")
            ApplicationPackage('testapp', [Schema('testapp', Document(None, None, None), None, None, [], False, None, [], None)],
                            QueryProfile(None), QueryProfileType(None))
            ```

        This creates a default Schema, QueryProfile, and QueryProfileType, which can be populated with your application's specifics.
        """
        if not (
            name[0].isalpha() and name.islower() and len(name) <= 20 and name.isalnum()
        ):
            raise ValueError(
                "Application package name must start with a letter, must be lowercase, can only contain [a-z0-9], and may contain no more than 20 characters, was '{}'".format(
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
        if self.auth_clients and self.clusters:
            for cluster in self.clusters:
                if isinstance(cluster, ContainerCluster):
                    if cluster.auth_clients:
                        # It is only meaningful to warn and override if the auth_clients differ.
                        # Works due to __eq__ and __gt/lt__ implementation in AuthClient
                        if not sorted(cluster.auth_clients) == sorted(
                            self.auth_clients
                        ):
                            warnings.warn(
                                "Auth clients are defined in the container cluster and in the application package. Overriding the container cluster auth clients. If this is not the intended behavior, remove the auth clients that are defined in the application package.",
                                UserWarning,
                            )
                            cluster.auth_clients = self.auth_clients

        self.deployment_config = deployment_config
        self.services_config = services_config

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
        Add Schema's to the application package.

        Args:
            schemas (list): Schemas to be added.

        Returns:
            None
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
    def services_to_text_vt(self):
        if self.services_config:
            return str(self.services_config)
        else:
            self.services_config = ServicesConfiguration(
                application_name=self.name,
                schemas=self.schemas or [],
                configurations=self.configurations or [],
                stateless_model_evaluation=self.stateless_model_evaluation,
                components=self.components or [],
                auth_clients=self.auth_clients or [],
                clusters=self.clusters or [],
            )
            return str(self.services_config)

    @property
    def services_to_text(self):
        """Intention is to only use services_config, but keeping this until 100% compatibility is achieved through tests."""
        if self.services_config:
            return str(self.services_config)
        else:
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
                clusters=self.clusters,
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
        """Return the application package as zipped bytes, to be used in a subsequent deploy.

        Returns:
            BytesIO: A buffer containing the zipped application package.
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
        """Export the application package as a deployable zipfile.
        See [application packages](https://docs.vespa.ai/en/application-packages.html) for deployment options.

        Args:
            zfile (str): Filename to export to.

        Returns:
            None
        """
        with open(zfile, "wb") as f:
            f.write(self.to_zip().getbuffer().tobytes())

    def to_files(self, root: Path) -> None:
        """Export the application package as a directory tree.

        Args:
            root (str): Directory to export files to.

        Returns:
            None
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


sample_package = ApplicationPackage(
    name="sample",
    schema=[
        Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary"]),
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
                        bolding=True,
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["title", "body"])],
            rank_profiles=[
                RankProfile(
                    name="bm25",
                    functions=[
                        Function(name="bm25sum", expression="bm25(title) + bm25(body)")
                    ],
                    first_phase="bm25sum",
                ),
            ],
        )
    ],
)
