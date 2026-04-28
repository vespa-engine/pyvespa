## `vespa.package`

### `VT(tag, cs, attrs=None, void_=False, replace_underscores=True, **kwargs)`

A 'Vespa Tag' structure, containing `tag`, `children`, and `attrs`

#### `sanitize_tag_name(tag)`

Convert invalid tag names (with '-') to valid Python identifiers (with '\_')

#### `restore_tag_name()`

Restore sanitized tag names back to the original names for XML generation

### `Summary(name=None, type=None, fields=None, select_elements_by=None)`

Bases: `object`

Configures a summary field.

Parameters:

| Name                 | Type   | Description                                                                                                                                                                        | Default |
| -------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `name`               | `str`  | The name of the summary field. Can be None if used inside a Field, which then uses the name of the Field.                                                                          | `None`  |
| `type`               | `str`  | The type of the summary field. Can be None if used inside a Field, which then uses the type of the Field.                                                                          | `None`  |
| `fields`             | `list` | A list of properties used to configure the summary. These can be single properties (like "summary: dynamic", common in Field), or composite values (like "source: another_field"). | `None`  |
| `select_elements_by` | `str`  | The name of a function that determines which elements to include in the summary.                                                                                                   | `None`  |

Example

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
    Summary(None, None, None, best_chunks)
```

#### `as_lines`

Returns the object as a list of strings, where each string represents a line of configuration that can be used during schema generation as shown below:

Example usage

```text
    {% for line in field.summary.as_lines %}
        {{ line }}
    {% endfor %}
```

Example

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

```python
Summary(None, None, None, "best_chunks").as_lines
['summary {', '    select-elements-by: best_chunks', '}']
```

### `HNSW(distance_metric='euclidean', max_links_per_node=16, neighbors_to_explore_at_insert=200)`

Bases: `object`

Configures Vespa HNSW indexes.

For more information, check the [Vespa documentation](https://docs.vespa.ai/en/approximate-nn-hnsw.html).

Parameters:

| Name                             | Type  | Description                                                                                          | Default       |
| -------------------------------- | ----- | ---------------------------------------------------------------------------------------------------- | ------------- |
| `distance_metric`                | `str` | The distance metric to use when computing distance between vectors. Default is 'euclidean'.          | `'euclidean'` |
| `max_links_per_node`             | `int` | Specifies how many links per HNSW node to select when building the graph. Default is 16.             | `16`          |
| `neighbors_to_explore_at_insert` | `int` | Specifies how many neighbors to explore when inserting a document in the HNSW graph. Default is 200. | `200`         |

### `StructField(name, **kwargs)`

Create a Vespa struct-field.

For more detailed information about struct-fields, check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#struct-field).

Parameters:

| Name            | Type                  | Description                                                                                                                                                                                                              | Default    |
| --------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `name`          | `str`                 | The name of the struct-field.                                                                                                                                                                                            | *required* |
| `indexing`      | `list, tuple, or str` | Configures how to process data of a struct-field during indexing. - Tuple: renders as indexing { value1; value2; ... } block with each item on a new line, and semicolon at the end. - List: renders as indexing: value1 | value2     |
| `attribute`     | `list`                | Specifies a property of an index structure attribute.                                                                                                                                                                    | *required* |
| `match`         | `list`                | Set properties that decide how the matching method for this field operates.                                                                                                                                              | *required* |
| `query_command` | `list`                | Add configuration for the query-command of the field.                                                                                                                                                                    | *required* |
| `summary`       | `Summary`             | Add configuration for the summary of the field.                                                                                                                                                                          | *required* |
| `rank`          | `str`                 | Specifies the property that defines ranking calculations done for a field.                                                                                                                                               | *required* |

Example

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

```python
StructField(
    name = "complex_field",
    indexing = ('"preprocessing"', ["attribute", "summary"]),
    attribute = ["fast-search"],
)
StructField('complex_field', ('"preprocessing"', ['attribute', 'summary']), ['fast-search'], None, None, None, None)
```

#### `indexing_as_multiline`

Generate multiline indexing statements for tuple-based indexing.

### `FieldConfiguration`

Bases: `TypedDict`

alias (list[str]): Add alias to the field. Use the format "component: component_alias" to add an alias to a field's component. See [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#uri) for an example.

### `Field(name, type, indexing=None, index=None, attribute=None, ann=None, match=None, weight=None, bolding=None, summary=None, is_document_field=True, **kwargs)`

Bases: `object`

Create a Vespa field.

For more detailed information about fields, check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#field).

Once we have an `ApplicationPackage` instance containing a `Schema` and a `Document`, we usually want to add fields so that we can store our data in a structured manner. We can accomplish that by creating `Field` instances and adding those to the `ApplicationPackage` instance via `Schema` and `Document` methods.

Index Configuration Behavior

- Single string configuration: uses `index: value` syntax
- Single dict or multiple configurations: uses `index { ... }` block syntax
- All configurations in a list are consolidated into a single index block

Parameters:

| Name                | Type                  | Description                                                                                                                                                                                                                                                                                                              | Default    |
| ------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `name`              | `str`                 | The name of the field.                                                                                                                                                                                                                                                                                                   | *required* |
| `type`              | `str`                 | The data type of the field.                                                                                                                                                                                                                                                                                              | *required* |
| `indexing`          | `list, tuple, or str` | Configures how to process data of a field during indexing. - Tuple: renders as indexing { value1; value2; ... } block with each item on a new line, and semicolon at the end. - List: renders as indexing: value1                                                                                                        | value2     |
| `index`             | `str, dict, or list`  | Sets index parameters. - Single string (e.g., "enable-bm25"): renders as index: enable-bm25 - Single dict (e.g., {"arity": 2}): renders as index { arity: 2 } - List with multiple items: renders as single index { ... } block containing all configurations Fields with index are normalized and tokenized by default. | `None`     |
| `attribute`         | `list`                | Specifies a property of an index structure attribute.                                                                                                                                                                                                                                                                    | `None`     |
| `ann`               | `HNSW`                | Add configuration for approximate nearest neighbor.                                                                                                                                                                                                                                                                      | `None`     |
| `match`             | `list`                | Set properties that decide how the matching method for this field operates.                                                                                                                                                                                                                                              | `None`     |
| `weight`            | `int`                 | Sets the weight of the field, used when calculating rank scores.                                                                                                                                                                                                                                                         | `None`     |
| `bolding`           | `bool`                | Whether to highlight matching query terms in the summary.                                                                                                                                                                                                                                                                | `None`     |
| `summary`           | `Summary`             | Add configuration for the summary of the field.                                                                                                                                                                                                                                                                          | `None`     |
| `is_document_field` | `bool`                | Whether the field is a document field or part of the schema. Default is True.                                                                                                                                                                                                                                            | `True`     |
| `stemming`          | `str`                 | Add configuration for stemming of the field.                                                                                                                                                                                                                                                                             | *required* |
| `rank`              | `str`                 | Add configuration for ranking calculations of the field.                                                                                                                                                                                                                                                                 | *required* |
| `query_command`     | `list`                | Add configuration for query-command of the field.                                                                                                                                                                                                                                                                        | *required* |
| `struct_fields`     | `list`                | Add struct-fields to the field.                                                                                                                                                                                                                                                                                          | *required* |
| `alias`             | `list`                | Add alias to the field. Use the format "component: component_alias" to add an alias to a field's component. See Vespa documentation for an example.                                                                                                                                                                      | *required* |

Example

```python
Field(name = "title", type = "string", indexing = ["index", "summary"], index = "enable-bm25")
Field('title', 'string', ['index', 'summary'], 'enable-bm25', None, None, None, None, None, None, True, None, None, None, [], None)
```

```python
Field(
    name = "title",
    type = "array<string>",
    indexing = ('"en"', ["index", "summary"]),
)
Field('title', 'array<string>', ('"en"', ['index', 'summary']), None, None, None, None, None, None, None, True, None, None, None, [], None)
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
    alias = ["artist_name", "component: component_alias"],
)
Field('artist', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], ['artist_name', 'component: component_alias'])
```

```python
# Single string index - uses simple syntax
Field(name = "title", type = "string", index = "enable-bm25")
# Renders as: index: enable-bm25
```

```python
# Single dict index - uses block syntax
Field(name = "predicate_field", type = "predicate", index = {"arity": 2})
# Renders as: index { arity: 2 }
```

```python
# Multiple string indices - uses block syntax
Field(name = "multi", type = "string", index = ["enable-bm25", "another-setting"])
# Renders as: index { enable-bm25; another-setting }
```

```python
# Complex index configurations with multiple parameters
Field(
    name = "predicate_field",
    type = "predicate",
    indexing = ["attribute"],
    index = {
        "arity": 2,
        "lower-bound": 3,
        "upper-bound": 200,
        "dense-posting-list-threshold": 0.25
    }
)
# Renders as: index { arity: 2; lower-bound: 3; upper-bound: 200; dense-posting-list-threshold: 0.25 }
```

```python
# Multiple index configurations with mixed types
Field(
    name = "complex_field",
    type = "string",
    indexing = ["index", "summary"],
    index = [
        "enable-bm25",  # Simple index setting
        {"arity": 2, "lower-bound": 3},  # Complex index block
        "another-setting"  # Another simple setting
    ]
)
# Renders as single block:
# index {
#     enable-bm25
#     arity: 2
#     lower-bound: 3
#     another-setting
# }
```

```python
# Parameterless index settings using None values
Field(
    name = "taxonomy",
    type = "array<string>",
    indexing = ["index", "summary"],
    match = ["text"],
    index = {"enable-bm25": None}
)
# Renders as: index { enable-bm25 } (without ": None")
```

#### `indexing_as_multiline`

Generate multiline indexing statements for tuple-based indexing.

#### `index_configurations`

Returns index configurations as a list, normalizing single values to lists. This allows the template to consistently iterate over index configurations.

#### `use_simple_index_syntax`

Returns True if we should use simple 'index: value' syntax. Simple syntax is used only when there's exactly one string configuration. Otherwise, we use the block syntax 'index { ... }'.

#### `add_struct_fields(*struct_fields)`

Add `StructField`'s to the `Field`.

Parameters:

| Name            | Type   | Description                                | Default |
| --------------- | ------ | ------------------------------------------ | ------- |
| `struct_fields` | `list` | A list of StructField objects to be added. | `()`    |

### `ImportedField(name, reference_field, field_to_import)`

Bases: `object`

Imported field from a reference document.

Useful to implement [parent/child relationships](https://docs.vespa.ai/en/parent-child.html).

Parameters:

| Name              | Type  | Description                                                                                   | Default    |
| ----------------- | ----- | --------------------------------------------------------------------------------------------- | ---------- |
| `name`            | `str` | Field name.                                                                                   | *required* |
| `reference_field` | `str` | A field of type reference that points to the document that contains the field to be imported. | *required* |
| `field_to_import` | `str` | Field name to be imported, as defined in the reference document.                              | *required* |

Example

```python
ImportedField(
    name="global_category_ctrs",
    reference_field="category_ctr_ref",
    field_to_import="ctrs",
)
ImportedField('global_category_ctrs', 'category_ctr_ref', 'ctrs')
```

### `Struct(name, fields=None)`

Bases: `object`

Create a Vespa struct. A struct defines a composite type. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#struct) for more detailed information about structs.

Parameters:

| Name     | Type   | Description                                           | Default    |
| -------- | ------ | ----------------------------------------------------- | ---------- |
| `name`   | `str`  | Name of the struct.                                   | *required* |
| `fields` | `list` | List of Field objects to be included in the fieldset. | `None`     |

Example

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

### `DocumentSummary(name, inherits=None, summary_fields=None, from_disk=None, omit_summary_features=None)`

Bases: `object`

Create a Document Summary. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#document-summary) for more detailed information about document-summary.

Parameters:

| Name                    | Type   | Description                                                                   | Default    |
| ----------------------- | ------ | ----------------------------------------------------------------------------- | ---------- |
| `name`                  | `str`  | Name of the document-summary.                                                 | *required* |
| `inherits`              | `str`  | Name of another document-summary from which this inherits.                    | `None`     |
| `summary_fields`        | `list` | List of Summary objects used in this document-summary.                        | `None`     |
| `from_disk`             | `bool` | Marks this document-summary as accessing fields on disk.                      | `None`     |
| `omit_summary_features` | `bool` | Specifies that summary-features should be omitted from this document summary. | `None`     |

Example

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

### `Document(fields=None, inherits=None, structs=None)`

Bases: `object`

Create a Vespa Document.

Check the [Vespa documentation](https://docs.vespa.ai/en/documents.html) for more detailed information about documents.

Parameters:

| Name     | Type   | Description                                                  | Default |
| -------- | ------ | ------------------------------------------------------------ | ------- |
| `fields` | `list` | A list of Field objects to include in the document's schema. | `None`  |

Example

```python
Document()
Document(None, None, None)

Document(fields=[Field(name="title", type="string")])
Document([Field('title', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None)], None, None)

Document(fields=[Field(name="title", type="string")], inherits="context")
Document([Field('title', 'string', None, None, None, None, None, None, None, None, True, None, None, None, [], None)], context, None)
```

#### `add_fields(*fields)`

Add `Field` objects to the document.

Parameters:

| Name     | Type   | Description         | Default |
| -------- | ------ | ------------------- | ------- |
| `fields` | `list` | Fields to be added. | `()`    |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `add_structs(*structs)`

Add `Struct` objects to the document.

Parameters:

| Name      | Type   | Description          | Default |
| --------- | ------ | -------------------- | ------- |
| `structs` | `list` | Structs to be added. | `()`    |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

### `FieldSet(name, fields)`

Bases: `object`

Create a Vespa field set.

A fieldset groups fields together for searching. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#fieldset) for more detailed information about field sets.

Parameters:

| Name     | Type   | Description                                 | Default    |
| -------- | ------ | ------------------------------------------- | ---------- |
| `name`   | `str`  | Name of the fieldset.                       | *required* |
| `fields` | `list` | Field names to be included in the fieldset. | *required* |

Returns:

| Name       | Type   | Description           |
| ---------- | ------ | --------------------- |
| `FieldSet` | `None` | A field set instance. |

Example

```text
FieldSet(name="default", fields=["title", "body"])
FieldSet('default', ['title', 'body'])
```

### `Function(name, expression, args=None)`

Bases: `object`

Create a Vespa rank function.

Define a named function that can be referenced as a part of the ranking expression, or (if having no arguments) as a feature. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#function-rank) for more detailed information about rank functions.

Parameters:

| Name         | Type   | Description                                                                | Default    |
| ------------ | ------ | -------------------------------------------------------------------------- | ---------- |
| `name`       | `str`  | Name of the function.                                                      | *required* |
| `expression` | `str`  | String representing a Vespa expression.                                    | *required* |
| `args`       | `list` | List of arguments to be used in the function expression. Defaults to None. | `None`     |

Returns:

| Name       | Type   | Description               |
| ---------- | ------ | ------------------------- |
| `Function` | `None` | A rank function instance. |

Example

```text
    Function(
        name="myfeature",
        expression="fieldMatch(bar) + freshness(foo)",
        args=["foo", "bar"]
    )
    Function('myfeature', 'fieldMatch(bar) + freshness(foo)', ['foo', 'bar'])
```

It is possible to define functions with multi-line expressions:

```text
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

### `FirstPhaseRanking(expression, keep_rank_count=None, rank_score_drop_limit=None)`

Create a Vespa first phase ranking configuration.

This is the initial ranking performed on all matching documents. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#firstphase-rank) for more detailed information about first phase ranking configuration.

Parameters:

| Name                    | Type    | Description                                                                                                                           | Default    |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `expression`            | `str`   | Specify the ranking expression to be used for the first phase of ranking. Check also the Vespa documentation for ranking expressions. | *required* |
| `keep_rank_count`       | `int`   | How many documents to keep the first phase top rank values for. Default value is 10000.                                               | `None`     |
| `rank_score_drop_limit` | `float` | Drop all hits with a first phase rank score less than or equal to this floating point number.                                         | `None`     |

Returns:

| Name                | Type   | Description                                   |
| ------------------- | ------ | --------------------------------------------- |
| `FirstPhaseRanking` | `None` | A first phase ranking configuration instance. |

Example

```text
FirstPhaseRanking("myFeature * 10")
FirstPhaseRanking('myFeature * 10', None, None)

FirstPhaseRanking(expression="myFeature * 10", keep_rank_count=50, rank_score_drop_limit=10)
FirstPhaseRanking('myFeature * 10', 50, 10)
```

### `SecondPhaseRanking(expression, rerank_count=100, rank_score_drop_limit=None)`

Bases: `object`

Create a Vespa second phase ranking configuration.

This is the optional reranking performed on the best hits from the first phase. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#secondphase-rank) for more detailed information about second phase ranking configuration.

Parameters:

| Name                    | Type    | Description                                                                                                                            | Default    |
| ----------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `expression`            | `str`   | Specify the ranking expression to be used for the second phase of ranking. Check also the Vespa documentation for ranking expressions. | *required* |
| `rerank_count`          | `int`   | Specifies the number of hits to be reranked in the second phase. Default value is 100.                                                 | `100`      |
| `rank_score_drop_limit` | `float` | Drop all hits with a first phase rank score less than or equal to this floating point number.                                          | `None`     |

Returns:

| Name                 | Type   | Description                                    |
| -------------------- | ------ | ---------------------------------------------- |
| `SecondPhaseRanking` | `None` | A second phase ranking configuration instance. |

Example

```text
SecondPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10)
SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, None)

SecondPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10, rank_score_drop_limit=5)
SecondPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, 5)
```

### `GlobalPhaseRanking(expression, rerank_count=100, rank_score_drop_limit=None)`

Bases: `object`

Create a Vespa global phase ranking configuration.

This is the optional reranking performed on the best hits from the content nodes phase(s). Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#globalphase-rank) for more detailed information about global phase ranking configuration.

Parameters:

| Name                    | Type    | Description                                                                                                                            | Default    |
| ----------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `expression`            | `str`   | Specify the ranking expression to be used for the global phase of ranking. Check also the Vespa documentation for ranking expressions. | *required* |
| `rerank_count`          | `int`   | Specifies the number of hits to be reranked in the global phase. Default value is 100.                                                 | `100`      |
| `rank_score_drop_limit` | `float` | Drop all hits with a first phase rank score less than or equal to this floating point number.                                          | `None`     |

Returns:

| Name                 | Type   | Description                                    |
| -------------------- | ------ | ---------------------------------------------- |
| `GlobalPhaseRanking` | `None` | A global phase ranking configuration instance. |

Example

```text
    GlobalPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10)
    GlobalPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, None)

    GlobalPhaseRanking(expression="1.25 * bm25(title) + 3.75 * bm25(body)", rerank_count=10, rank_score_drop_limit=5)
    GlobalPhaseRanking('1.25 * bm25(title) + 3.75 * bm25(body)', 10, 5)
```

### `Mutate(on_match, on_first_phase, on_second_phase, on_summary)`

Bases: `object`

Enable mutating operations in rank profiles.

Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#mutate) for more detailed information about mutable attributes.

Parameters:

| Name              | Type   | Description                                                                                                                                                                                                                                                                                           | Default    |
| ----------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `on_match`        | `dict` | Dictionary for the on-match phase containing 3 mandatory keys: - attribute: name of the mutable attribute to mutate. - operation_string: operation to perform on the mutable attribute. - operation_value: number to set, add, or subtract to/from the current value of the mutable attribute.        | *required* |
| `on_first_phase`  | `dict` | Dictionary for the on-first-phase phase containing 3 mandatory keys: - attribute: name of the mutable attribute to mutate. - operation_string: operation to perform on the mutable attribute. - operation_value: number to set, add, or subtract to/from the current value of the mutable attribute.  | *required* |
| `on_second_phase` | `dict` | Dictionary for the on-second-phase phase containing 3 mandatory keys: - attribute: name of the mutable attribute to mutate. - operation_string: operation to perform on the mutable attribute. - operation_value: number to set, add, or subtract to/from the current value of the mutable attribute. | *required* |
| `on_summary`      | `dict` | Dictionary for the on-summary phase containing 3 mandatory keys: - attribute: name of the mutable attribute to mutate. - operation_string: operation to perform on the mutable attribute. - operation_value: number to set, add, or subtract to/from the current value of the mutable attribute.      | *required* |

Example

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

### `Diversity(attribute, min_groups)`

Bases: `object`

Create a Vespa ranking diversity configuration.

This is an optional config that is used to guarantee diversity in the different query phases. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#diversity) for more detailed information about diversity configuration.

Parameters:

| Name         | Type  | Description                                                                                                              | Default    |
| ------------ | ----- | ------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `attribute`  | `str` | Which attribute to use when deciding diversity. The attribute must be a single-valued numeric, string or reference type. | *required* |
| `min_groups` | `int` | Specifies the minimum number of groups returned from the phase.                                                          | *required* |

Returns:

| Name        | Type   | Description                                 |
| ----------- | ------ | ------------------------------------------- |
| `Diversity` | `None` | A ranking diversity configuration instance. |

Example

```text
Diversity(attribute="popularity", min_groups=5)
Diversity('popularity', 5)
```

### `MatchPhaseRanking(attribute, order, max_hits)`

Bases: `object`

Create a Vespa match phase ranking configuration.

This is an optional phase that can be used to quickly select a subset of hits for further ranking. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#match-phase) for more detailed information about match phase ranking configuration.

Parameters:

| Name        | Type  | Description                                         | Default    |
| ----------- | ----- | --------------------------------------------------- | ---------- |
| `attribute` | `str` | The numeric attribute to use for filtering.         | *required* |
| `order`     | `str` | The sort order, either "ascending" or "descending". | *required* |
| `max_hits`  | `int` | Maximum number of hits to pass to the next phase.   | *required* |

Example

```python
MatchPhaseRanking(attribute="popularity", order="descending", max_hits=1000)
MatchPhaseRanking('popularity', 'descending', 1000)
```

### `RankProfile(name, first_phase=None, inherits=None, constants=None, functions=None, summary_features=None, match_features=None, second_phase=None, global_phase=None, match_phase=None, num_threads_per_search=None, diversity=None, **kwargs)`

Bases: `object`

Create a Vespa rank profile.

Rank profiles are used to specify an alternative ranking of the same data for different purposes, and to experiment with new rank settings. Check the [Vespa documentation](https://docs.vespa.ai/en/reference/schema-reference.html#rank-profile) for more detailed information about rank profiles.

Parameters:

| Name                     | Type                  | Description                                                                                                                                                                      | Default    |
| ------------------------ | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `name`                   | `str`                 | Rank profile name.                                                                                                                                                               | *required* |
| `first_phase`            | `str`                 | The config specifying the first phase of ranking. More info about first phase ranking.                                                                                           | `None`     |
| `inherits`               | `str`                 | The inherits attribute is optional. If defined, it contains the name of another rank profile in the same schema. Values not defined in this rank profile will then be inherited. | `None`     |
| `constants`              | `dict`                | Dict of constants available in ranking expressions, resolved and optimized at configuration time. More info about constants.                                                     | `None`     |
| `functions`              | `list`                | List of Function objects representing rank functions to be included in the rank profile.                                                                                         | `None`     |
| `summary_features`       | `list`                | List of rank features to be included with each hit. More info about summary features.                                                                                            | `None`     |
| `match_features`         | `list`                | List of rank features to be included with each hit. More info about match features.                                                                                              | `None`     |
| `second_phase`           | `SecondPhaseRanking`  | Config specifying the second phase of ranking. See SecondPhaseRanking.                                                                                                           | `None`     |
| `global_phase`           | `GlobalPhaseRanking`  | Config specifying the global phase of ranking. See GlobalPhaseRanking.                                                                                                           | `None`     |
| `match_phase`            | `MatchPhaseRanking`   | Config specifying the match phase of ranking. See MatchPhaseRanking.                                                                                                             | `None`     |
| `num_threads_per_search` | `int`                 | Overrides the global persearch value for this rank profile to a lower value.                                                                                                     | `None`     |
| `diversity`              | `Optional[Diversity]` | Optional config specifying the diversity of ranking.                                                                                                                             | `None`     |
| `weight`                 | `list`                | A list of tuples containing the field and their weight.                                                                                                                          | *required* |
| `rank_type`              | `list`                | A list of tuples containing a field and the rank-type-name. More info about rank-type.                                                                                           | *required* |
| `rank_properties`        | `list`                | A list of tuples containing a field and its configuration. More info about rank-properties.                                                                                      | *required* |
| `mutate`                 | `Mutate`              | A Mutate object containing attributes to mutate on, mutation operation, and value. More info about mutate operation.                                                             | *required* |

Example

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

### `OnnxModel(model_name, model_file_path, inputs, outputs)`

Bases: `object`

Create a Vespa ONNX model config.

Vespa has support for advanced ranking models through its tensor API. If you have your model in the ONNX format, Vespa can import the models and use them directly. Check the [Vespa documentation](https://docs.vespa.ai/en/onnx.html) for more detailed information about field sets.

Parameters:

| Name              | Type   | Description                                                                                                                                                                                                                                                    | Default    |
| ----------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `model_name`      | `str`  | Unique model name to use as an ID when referencing the model.                                                                                                                                                                                                  | *required* |
| `model_file_path` | `str`  | ONNX model file path.                                                                                                                                                                                                                                          | *required* |
| `inputs`          | `dict` | Dict mapping the ONNX input names as specified in the ONNX file to valid Vespa inputs. These can be a document field (attribute(field_name)), a query parameter (query(query_param)), a constant (constant(name)), or a user-defined function (function_name). | *required* |
| `outputs`         | `dict` | Dict mapping the ONNX output names as specified in the ONNX file to the name used in Vespa to specify the output. If omitted, the first output in the ONNX file will be used.                                                                                  | *required* |

Example

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

### `Schema(name, document, fieldsets=None, rank_profiles=None, models=None, global_document=False, imported_fields=None, document_summaries=None, mode='index', inherits=None, **kwargs)`

Bases: `object`

Create a Vespa Schema.

Check the [Vespa documentation](https://docs.vespa.ai/en/schemas.html) for more detailed information about schemas.

Parameters:

| Name                 | Type       | Description                                                                       | Default    |
| -------------------- | ---------- | --------------------------------------------------------------------------------- | ---------- |
| `name`               | `str`      | Schema name.                                                                      | *required* |
| `document`           | `Document` | Vespa Document associated with the Schema.                                        | *required* |
| `fieldsets`          | `list`     | A list of FieldSet associated with the Schema.                                    | `None`     |
| `rank_profiles`      | `list`     | A list of RankProfile associated with the Schema.                                 | `None`     |
| `models`             | `list`     | A list of OnnxModel associated with the Schema.                                   | `None`     |
| `global_document`    | `bool`     | Set to True to copy the documents to all content nodes. Defaults to False.        | `False`    |
| `imported_fields`    | `list`     | A list of ImportedField defining fields from global documents to be imported.     | `None`     |
| `document_summaries` | `list`     | A list of DocumentSummary associated with the schema.                             | `None`     |
| `mode`               | `str`      | Schema mode. Defaults to 'index'. Other options are 'store-only' and 'streaming'. | `'index'`  |
| `inherits`           | `str`      | Schema to inherit from.                                                           | `None`     |
| `stemming`           | `str`      | The default stemming setting. Defaults to 'best'.                                 | *required* |

Example

```python
Schema(name="schema_name", document=Document())
Schema('schema_name', Document(None, None, None), None, None, [], False, None, [], None)
```

#### `add_fields(*fields)`

Add `Field` to the Schema's `Document`.

Parameters:

| Name     | Type   | Description                                          | Default |
| -------- | ------ | ---------------------------------------------------- | ------- |
| `fields` | `list` | A list of Field objects to be added to the Document. | `()`    |

Example

```python
schema.add_fields([Field(name="title", type="string"), Field(name="body", type="text")])
schema.add_fields([Field('title', 'string'), Field('body', 'text')])
```

#### `add_field_set(field_set)`

Add a `FieldSet` to the Schema.

Parameters:

| Name        | Type   | Description                                           | Default    |
| ----------- | ------ | ----------------------------------------------------- | ---------- |
| `field_set` | `list` | A list of FieldSet objects to be added to the Schema. | *required* |

#### `add_rank_profile(rank_profile)`

Add a `RankProfile` to the Schema.

Parameters:

| Name           | Type          | Description                                 | Default    |
| -------------- | ------------- | ------------------------------------------- | ---------- |
| `rank_profile` | `RankProfile` | The rank profile to be added to the Schema. | *required* |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `add_model(model)`

Add an `OnnxModel` to the Schema.

Parameters:

| Name    | Type        | Description                               | Default    |
| ------- | ----------- | ----------------------------------------- | ---------- |
| `model` | `OnnxModel` | The ONNX model to be added to the Schema. | *required* |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `add_imported_field(imported_field)`

Add an `ImportedField` to the Schema.

Parameters:

| Name             | Type            | Description                                   | Default    |
| ---------------- | --------------- | --------------------------------------------- | ---------- |
| `imported_field` | `ImportedField` | The imported field to be added to the Schema. | *required* |

#### `add_document_summary(document_summary)`

Add a `DocumentSummary` to the Schema.

Parameters:

| Name               | Type              | Description                                     | Default    |
| ------------------ | ----------------- | ----------------------------------------------- | ---------- |
| `document_summary` | `DocumentSummary` | The document summary to be added to the Schema. | *required* |

### `QueryTypeField(name, type)`

Bases: `object`

Create a field to be included in a `QueryProfileType`.

Parameters:

| Name   | Type  | Description | Default    |
| ------ | ----- | ----------- | ---------- |
| `name` | `str` | Field name. | *required* |
| `type` | `str` | Field type. | *required* |

Example

```python
QueryTypeField(
    name="ranking.features.query(title_bert)",
    type="tensor<float>(x[768])"
)
QueryTypeField('ranking.features.query(title_bert)', 'tensor<float>(x[768])')
```

### `QueryProfileType(fields=None)`

Bases: `object`

Create a Vespa Query Profile Type.

Check the [Vespa documentation](https://docs.vespa.ai/en/query-profiles.html#query-profile-types) for more detailed information about query profile types.

An `ApplicationPackage` instance comes with a default `QueryProfile` named `default` that is associated with a `QueryProfileType` named `root`, meaning that you usually do not need to create those yourself, only add fields to them when required.

Parameters:

| Name     | Type                   | Description               | Default |
| -------- | ---------------------- | ------------------------- | ------- |
| `fields` | `list[QueryTypeField]` | A list of QueryTypeField. | `None`  |

Example

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

#### `add_fields(*fields)`

Add `QueryTypeField` objects to the Query Profile Type.

Parameters:

| Name     | Type             | Description         | Default |
| -------- | ---------------- | ------------------- | ------- |
| `fields` | `QueryTypeField` | Fields to be added. | `()`    |

Example

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

### `QueryField(name, value)`

Bases: `object`

Create a field to be included in a `QueryProfile`.

Parameters:

| Name    | Type  | Description  | Default    |
| ------- | ----- | ------------ | ---------- |
| `name`  | `str` | Field name.  | *required* |
| `value` | `Any` | Field value. | *required* |

Example

```python
QueryField(name="maxHits", value=1000)
# Output: QueryField('maxHits', 1000)
```

### `QueryProfile(fields=None)`

Bases: `object`

Create a Vespa Query Profile.

Check the [Vespa documentation](https://docs.vespa.ai/en/query-profiles.html) for more detailed information about query profiles.

A `QueryProfile` is a named collection of query request parameters given in the configuration. The query request can specify a query profile whose parameters will be used as parameters of that request. The query profiles may optionally be type-checked. Type checking is turned on by referencing a `QueryProfileType` from the query profile.

Parameters:

| Name     | Type               | Description           | Default |
| -------- | ------------------ | --------------------- | ------- |
| `fields` | `list[QueryField]` | A list of QueryField. | `None`  |

Example

```python
QueryProfile(fields=[QueryField(name="maxHits", value=1000)])
# Output: QueryProfile([QueryField('maxHits', 1000)])
```

#### `add_fields(*fields)`

Add `QueryField` objects to the Query Profile.

Parameters:

| Name     | Type         | Description         | Default |
| -------- | ------------ | ------------------- | ------- |
| `fields` | `QueryField` | Fields to be added. | `()`    |

Example

```python
query_profile = QueryProfile()
query_profile.add_fields(QueryField(name="maxHits", value=1000))
```

### `ApplicationConfiguration(name, value)`

Bases: `object`

Create a Vespa Schema.

Check the [Config documentation](https://docs.vespa.ai/en/reference/services.html#generic-config) for more detailed information about generic configuration.

Parameters:

| Name    | Type  | Description         | Default                                                          |
| ------- | ----- | ------------------- | ---------------------------------------------------------------- |
| `name`  | `str` | Configuration name. | *required*                                                       |
| `value` | \`str | dict\`              | Either a string or a dictionary (which may be nested) of values. |

Example

```python
ApplicationConfiguration(
    name="container.handler.observability.application-userdata",
    value={"version": "my-version"}
)
# Output: ApplicationConfiguration(name="container.handler.observability.application-userdata")
```

### `Parameter(name, args=None, children=None)`

Bases: `object`

Create a Vespa Component configuration parameter.

Parameters:

| Name       | Type  | Description          | Default                                                                                       |
| ---------- | ----- | -------------------- | --------------------------------------------------------------------------------------------- |
| `name`     | `str` | Parameter name.      | *required*                                                                                    |
| `args`     | `Any` | Parameter arguments. | `None`                                                                                        |
| `children` | \`str | list[Parameter]\`    | Parameter children. Can be either a string or a list of Parameter objects for nested configs. |

### `AuthClient(id, permissions, parameters=None)`

Bases: `object`

Create a Vespa AuthClient.

Check the [Vespa documentation](https://docs.vespa.ai/en/reference/services-container.html).

Parameters:

| Name          | Type              | Description                                                              | Default    |
| ------------- | ----------------- | ------------------------------------------------------------------------ | ---------- |
| `id`          | `str`             | The auth client ID.                                                      | *required* |
| `permissions` | `list[str]`       | List of permissions.                                                     | *required* |
| `parameters`  | `list[Parameter]` | List of Parameter objects defining the configuration of the auth client. | `None`     |

Example

```python
AuthClient(
    id="token",
    permissions=["read", "write"],
    parameters=[Parameter("token", {"id": "my-token-id"})],
)
# Output: AuthClient(id="token", permissions="['read', 'write']")
```

### `Component(id, cls=None, bundle=None, type=None, parameters=None)`

Bases: `object`

### `Nodes(count='1', parameters=None)`

Bases: `object`

Specify node resources for a content or container cluster as part of a `ContainerCluster` or `ContentCluster`.

Parameters:

| Name         | Type              | Description                                                                    | Default |
| ------------ | ----------------- | ------------------------------------------------------------------------------ | ------- |
| `count`      | `int`             | Number of nodes in a cluster.                                                  | `'1'`   |
| `parameters` | `list[Parameter]` | List of Parameter objects defining the configuration of the cluster resources. | `None`  |

Example

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

### `Cluster(id, version='1.0', nodes=None)`

Bases: `object`

Base class for a cluster configuration. Should not be instantiated directly. Use subclasses `ContainerCluster` or `ContentCluster` instead.

Parameters:

| Name      | Type    | Description                          | Default    |
| --------- | ------- | ------------------------------------ | ---------- |
| `id`      | `str`   | Cluster ID.                          | *required* |
| `version` | `str`   | Cluster version.                     | `'1.0'`    |
| `nodes`   | `Nodes` | Nodes that specifies node resources. | `None`     |

#### `to_xml(root)`

Set up XML elements that are used in both container and content clusters.

### `ContainerCluster(id, version='1.0', nodes=None, components=None, auth_clients=None)`

Bases: `Cluster`

Defines the configuration of a container cluster.

Parameters:

| Name           | Type               | Description                                                                                    | Default |
| -------------- | ------------------ | ---------------------------------------------------------------------------------------------- | ------- |
| `components`   | `list[Component]`  | List of Component that contains configurations for application components, e.g. embedders.     | `None`  |
| `auth_clients` | `list[AuthClient]` | List of AuthClient that contains configurations for authentication clients (e.g., mTLS/token). | `None`  |
| `nodes`        | `Nodes`            | Nodes that specifies the resources of the cluster.                                             | `None`  |

If `ContainerCluster` is used, any `Component`s must be added to the `ContainerCluster`, rather than to the `ApplicationPackage`, in order to be included in the generated schema.

Example

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

### `ContentCluster(id, document_name, version='1.0', nodes=None, min_redundancy='1')`

Bases: `Cluster`

Defines the configuration of a content cluster.

Parameters:

| Name             | Type  | Description                                                                               | Default    |
| ---------------- | ----- | ----------------------------------------------------------------------------------------- | ---------- |
| `document_name`  | `str` | Name of document.                                                                         | *required* |
| `min_redundancy` | `int` | Minimum redundancy of the content cluster. Must be at least 2 for production deployments. | `'1'`      |

Example

```python
ContentCluster(id="example_content", document_name="doc")
# Output: ContentCluster(id="example_content", version="1.0", document_name="doc")
```

### `ValidationID`

Bases: `Enum`

Collection of IDs that can be used in validation-overrides.xml.

Taken from [ValidationId.java](https://github.com/vespa-engine/vespa/blob/master/config-model-api/src/main/java/com/yahoo/config/application/api/ValidationId.java).

`clusterSizeReduction` was not added as it will be removed in Vespa 9.

#### `indexingChange = 'indexing-change'`

Changing what tokens are expected and stored in field indexes

#### `indexModeChange = 'indexing-mode-change'`

Changing the index mode (streaming, indexed, store-only) of documents

#### `fieldTypeChange = 'field-type-change'`

Field type changes

#### `tensorTypeChange = 'tensor-type-change'`

Tensor type change

#### `resourcesReduction = 'resources-reduction'`

Large reductions in node resources (> 50% of the current max total resources)

#### `contentTypeRemoval = 'schema-removal'`

Removal of a schema (causes deletion of all documents)

#### `contentClusterRemoval = 'content-cluster-removal'`

Removal (or id change) of content clusters

#### `deploymentRemoval = 'deployment-removal'`

Removal of production zones from deployment.xml

#### `globalDocumentChange = 'global-document-change'`

Changing global attribute for document types in content clusters

#### `configModelVersionMismatch = 'config-model-version-mismatch'`

Internal use

#### `skipOldConfigModels = 'skip-old-config-models'`

Internal use

#### `accessControl = 'access-control'`

Internal use, used in zones where there should be no access-control

#### `globalEndpointChange = 'global-endpoint-change'`

Changing global endpoints

#### `zoneEndpointChange = 'zone-endpoint-change'`

Changing zone (possibly private) endpoint settings

#### `redundancyIncrease = 'redundancy-increase'`

Increasing redundancy - may easily cause feed blocked

#### `redundancyOne = 'redundancy-one'`

redundancy=1 requires a validation override on first deployment

#### `pagedSettingRemoval = 'paged-setting-removal'`

May cause content nodes to run out of memory

#### `certificateRemoval = 'certificate-removal'`

Remove data plane certificates

### `Validation(validation_id, until, comment=None)`

Bases: `object`

Represents a validation to be overridden on application.

Check the [Vespa documentation](https://docs.vespa.ai/en/reference/validation-overrides.html) for more detailed information about validations.

Parameters:

| Name            | Type  | Description                                                                                                                                                                                                                                                                                                     | Default    |
| --------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `validation_id` | `str` | ID of the validation.                                                                                                                                                                                                                                                                                           | *required* |
| `until`         | `str` | The last day this change is allowed, as an ISO-8601-format date in UTC, e.g. 2016-01-30. Dates may at most be 30 days in the future, but should be as close to now as possible for safety, while allowing time for review and propagation to all deployed zones. allow-tags with dates in the past are ignored. | *required* |
| `comment`       | `str` | Optional text explaining the reason for the change to humans.                                                                                                                                                                                                                                                   | `None`     |

### `DeploymentConfiguration(environment, regions)`

Bases: `object`

Create a DeploymentConfiguration, which defines how to generate a deployment.xml file (for use in production deployments).

Parameters:

| Name          | Type        | Description                                                                                                  | Default    |
| ------------- | ----------- | ------------------------------------------------------------------------------------------------------------ | ---------- |
| `environment` | `str`       | The environment to deploy to. Currently, only 'prod' is supported.                                           | *required* |
| `regions`     | `list[str]` | List of regions to deploy to, e.g. ["us-east-1", "us-west-1"]. See Vespa documentation for more information. | *required* |

Example

```python
DeploymentConfiguration(environment="prod", regions=["us-east-1", "us-west-1"])
# Output: DeploymentConfiguration(environment='prod', regions=['us-east-1', 'us-west-1'])
```

### `EmptyDeploymentConfiguration()`

Bases: `DeploymentConfiguration`

Create an EmptyDeploymentConfiguration, which creates an empty deployment.xml, used to delete production deployments.

### `ServicesConfiguration(application_name, schemas=None, configurations=[], stateless_model_evaluation=False, components=[], auth_clients=[], clusters=[], services_config=None)`

Bases: `object`

Create a ServicesConfiguration, adopting the VespaTag (VT) approach, rather than Jinja templates. Intended to be used in ApplicationPackage, to generate services.xml, based on either:

- A passed `services_config` (VT) object, or
- A set of configurations, schemas, components, auth_clients, and clusters (equivalent to the old approach).

The latter will be done in code by calling `build_services_vt()` to generate the VT object.

Parameters:

| Name                         | Type                                       | Description                                                                        | Default    |
| ---------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------- | ---------- |
| `application_name`           | `str`                                      | Application name.                                                                  | *required* |
| `schemas`                    | `Optional[List[Schema]]`                   | List of Schemas of the application.                                                | `None`     |
| `configurations`             | `Optional[List[ApplicationConfiguration]]` | List of ApplicationConfiguration that contains configurations for the application. | `[]`       |
| `stateless_model_evaluation` | `Optional[bool]`                           | Enable stateless model evaluation. Default is False.                               | `False`    |
| `components`                 | `Optional[List[Component]]`                | List of Component that contains configurations for application components.         | `[]`       |
| `auth_clients`               | `Optional[List[AuthClient]]`               | List of AuthClient that contains configurations for authentication clients.        | `[]`       |
| `clusters`                   | `Optional[List[Cluster]]`                  | List of Cluster that contains configurations for content or container clusters.    | `[]`       |
| `services_config`            | `Optional[VT]`                             | VT object that contains the services configuration.                                | `None`     |

Example

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

### `ApplicationPackage(name, schema=None, query_profile=None, query_profile_type=None, stateless_model_evaluation=False, create_schema_by_default=True, create_query_profile_by_default=True, configurations=None, validations=None, components=None, auth_clients=None, clusters=None, deployment_config=None, services_config=None, query_profile_config=None, include_files=None)`

Bases: `object`

Create an application package.

Parameters:

| Name                              | Type                                              | Description                                                                                                                                                                                                                | Default    |
| --------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `name`                            | `str`                                             | Application name. Cannot contain '-' or '\_'.                                                                                                                                                                              | *required* |
| `schema`                          | `list`                                            | List of Schema objects for the application. If None, a default Schema with the same name as the application will be created. Defaults to None.                                                                             | `None`     |
| `query_profile`                   | `QueryProfile`                                    | QueryProfile of the application. If None, a default QueryProfile with QueryProfileType 'root' will be created. Defaults to None.                                                                                           | `None`     |
| `query_profile_type`              | `QueryProfileType`                                | QueryProfileType of the application. If None, a default QueryProfileType 'root' will be created. Defaults to None.                                                                                                         | `None`     |
| `stateless_model_evaluation`      | `bool`                                            | Enable stateless model evaluation. Defaults to False.                                                                                                                                                                      | `False`    |
| `create_schema_by_default`        | `bool`                                            | Include a default Schema if none is provided in the schema argument. Defaults to True.                                                                                                                                     | `True`     |
| `create_query_profile_by_default` | `bool`                                            | Include a default QueryProfile and QueryProfileType if not explicitly defined by the user. Defaults to True.                                                                                                               | `True`     |
| `configurations`                  | `list`                                            | List of ApplicationConfiguration for the application. Defaults to None.                                                                                                                                                    | `None`     |
| `validations`                     | `list`                                            | Optional list of Validation objects to be overridden. Defaults to None.                                                                                                                                                    | `None`     |
| `components`                      | `list`                                            | List of Component objects for application components. Defaults to None.                                                                                                                                                    | `None`     |
| `clusters`                        | `list`                                            | List of Cluster objects for content or container clusters. If clusters is provided, any Component must be part of a cluster. Defaults to None.                                                                             | `None`     |
| `auth_clients`                    | `list`                                            | List of AuthClient objects for client authorization. If clusters is passed, pass the auth clients to the ContainerCluster instead. Defaults to None.                                                                       | `None`     |
| `deployment_config`               | `Union[DeploymentConfiguration, VT]`              | Deployment configuration for the application. Must be either a DeploymentConfiguration object (legacy) or a VT (Vespa Tag) based deployment configuration whose top-level tag must be deployment. Defaults to None.        | `None`     |
| `services_config`                 | `ServicesConfiguration`                           | (Optional) Services configuration for the application. For advanced configuration. See https://vespa-engine.github.io/pyvespa/advanced-configuration.md                                                                    | `None`     |
| `query_profile_config`            | `Union[VT, List[VT]]`                             | Configuration for query profiles. If provided, will override the query_profile and query_profile_type arguments. Defaults to None. See See https://vespa-engine.github.io/pyvespa/advanced-configuration.md                | `None`     |
| `include_files`                   | `List[Tuple[Union[str, Path], Union[str, Path]]]` | Extra files to bundle into the application package. Each entry is a (source_path, dest_path) tuple where source_path is a local file and dest_path is its location inside the package (relative, no ..). Defaults to None. | `None`     |

Example

To create a default application package:

```python
ApplicationPackage(name="testapp")
ApplicationPackage('testapp', [Schema('testapp', Document(None, None, None), None, None, [], False, None, [], None)],
                QueryProfile(None), QueryProfileType(None))
```

This creates a default Schema, QueryProfile, and QueryProfileType, which can be populated with your application's specifics.

#### `services_to_text`

Intention is to only use services_config, but keeping this until 100% compatibility is achieved through tests.

#### `add_schema(*schemas)`

Add Schema's to the application package.

Parameters:

| Name      | Type   | Description          | Default |
| --------- | ------ | -------------------- | ------- |
| `schemas` | `list` | Schemas to be added. | `()`    |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `add_query_profile(query_profile_item)`

Add a query profile item (query-profile or query-profile-type) to the application package.

Parameters:

| Name                 | Type             | Description                        | Default    |
| -------------------- | ---------------- | ---------------------------------- | ---------- |
| `query_profile_item` | `VT or List[VT]` | Query profile item(s) to be added. | *required* |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

Example

```python
app_package = ApplicationPackage(name="testapp")
qp = query_profile(
    field(30, name="hits"),
    field(3, name="trace.level"),
)
app_package.add_query_profile(
    qp
)
# Query profile item is added to the application package.
# inspect with `app_package.query_profile_config`
```

#### `to_zip()`

Return the application package as zipped bytes, to be used in a subsequent deploy.

Returns:

| Name      | Type      | Description                                         |
| --------- | --------- | --------------------------------------------------- |
| `BytesIO` | `BytesIO` | A buffer containing the zipped application package. |

#### `to_zipfile(zfile)`

Export the application package as a deployable zipfile. See [application packages](https://docs.vespa.ai/en/application-packages.html) for deployment options.

Parameters:

| Name    | Type  | Description            | Default    |
| ------- | ----- | ---------------------- | ---------- |
| `zfile` | `str` | Filename to export to. | *required* |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `to_files(root)`

Export the application package as a directory tree.

Parameters:

| Name   | Type  | Description                   | Default    |
| ------ | ----- | ----------------------------- | ---------- |
| `root` | `str` | Directory to export files to. | *required* |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

### `validate_services(xml_input)`

Validate an XML input against the RelaxNG schema file for services.xml

Parameters:

| Name        | Type                     | Description                | Default    |
| ----------- | ------------------------ | -------------------------- | ---------- |
| `xml_input` | `Path or str or Element` | The XML input to validate. | *required* |

Returns: True if the XML input is valid according to the RelaxNG schema, False otherwise.
