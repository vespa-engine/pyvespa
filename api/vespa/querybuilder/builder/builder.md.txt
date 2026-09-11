## `vespa.querybuilder.builder.builder`

### `QueryField(name)`

### `Condition(expression)`

#### `all(*conditions)`

Combine multiple conditions using logical AND.

#### `any(*conditions)`

Combine multiple conditions using logical OR.

### `Query(select_fields, prepend_yql=False)`

#### `from_(*sources)`

Specify the source schema(s) to query.

Example

```python
import vespa.querybuilder as qb
from vespa.package import Schema, Document

query = qb.select("*").from_("schema1", "schema2")
str(query)
'select * from schema1, schema2'
query = qb.select("*").from_(Schema(name="schema1", document=Document()), Schema(name="schema2", document=Document()))
str(query)
'select * from schema1, schema2'
```

Parameters:

| Name      | Type                 | Description                    | Default |
| --------- | -------------------- | ------------------------------ | ------- |
| `sources` | `Union[str, Schema]` | The source schema(s) to query. | `()`    |

Returns:

| Name    | Type    | Description       |
| ------- | ------- | ----------------- |
| `Query` | `Query` | The Query object. |

#### `where(condition)`

Adds a where clause to filter query results.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#where>

Parameters:

| Name        | Type                     | Description                                                                                                                               | Default    |
| ----------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `condition` | `Union[Condition, bool]` | Filter condition that can be: - Condition object for complex queries - Boolean for simple true/false - QueryField for field-based filters | *required* |

Returns:

| Name    | Type    | Description              |
| ------- | ------- | ------------------------ |
| `Query` | `Query` | Self for method chaining |

Example

```python
import vespa.querybuilder as qb

# Using field conditions
f1 = qb.QueryField("f1")
query = qb.select("*").from_("sd1").where(f1.contains("v1"))
str(query)
'select * from sd1 where f1 contains "v1"'
```

```python
# Using boolean
query = qb.select("*").from_("sd1").where(True)
str(query)
'select * from sd1 where true'
```

```python
# Using complex conditions
condition = f1.contains("v1") & qb.QueryField("f2").contains("v2")
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where f1 contains "v1" and f2 contains "v2"'
```

#### `order_by(field, ascending=True, annotations=None)`

Orders results by specified fields.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#order-by>

Parameters:

| Name          | Type                       | Description                                                                                                                                      | Default    |
| ------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `fields`      |                            | Field names or QueryField objects to order by                                                                                                    | *required* |
| `annotations` | `Optional[Dict[str, Any]]` | Optional annotations like "locale", "strength", etc. See https://docs.vespa.ai/en/reference/sorting.html#special-sorting-attributes for details. | `None`     |

Returns:

| Name    | Type    | Description              |
| ------- | ------- | ------------------------ |
| `Query` | `Query` | Self for method chaining |

Example

```python
import vespa.querybuilder as qb

# Simple ordering
query = qb.select("*").from_("sd1").order_by("price")
str(query)
'select * from sd1 order by price asc'

# Multiple fields with annotation
query = qb.select("*").from_("sd1").order_by(
    "price", annotations={"locale": "en_US"}, ascending=False
).order_by("name", annotations={"locale": "no_NO"}, ascending=True)
str(query)
'select * from sd1 order by {"locale":"en_US"}price desc, {"locale":"no_NO"}name asc'
```

#### `orderByAsc(field, annotations=None)`

Convenience method for ordering results by a field in ascending order. See `order_by` for more information.

#### `orderByDesc(field, annotations=None)`

Convenience method for ordering results by a field in descending order. See `order_by` for more information.

#### `set_limit(limit)`

Sets maximum number of results to return.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#limit-offset>

Parameters:

| Name    | Type  | Description                      | Default    |
| ------- | ----- | -------------------------------- | ---------- |
| `limit` | `int` | Maximum number of hits to return | *required* |

Returns:

| Name    | Type    | Description              |
| ------- | ------- | ------------------------ |
| `Query` | `Query` | Self for method chaining |

Example

```python
import vespa.querybuilder as qb

f1 = qb.QueryField("f1")
query = qb.select("*").from_("sd1").where(f1.contains("v1")).set_limit(5)
str(query)
'select * from sd1 where f1 contains "v1" limit 5'
```

#### `set_offset(offset)`

Sets number of initial results to skip for pagination.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#limit-offset>

Parameters:

| Name     | Type  | Description               | Default    |
| -------- | ----- | ------------------------- | ---------- |
| `offset` | `int` | Number of results to skip | *required* |

Returns:

| Name    | Type    | Description              |
| ------- | ------- | ------------------------ |
| `Query` | `Query` | Self for method chaining |

Example

```python
import vespa.querybuilder as qb

f1 = qb.QueryField("f1")
query = qb.select("*").from_("sd1").where(f1.contains("v1")).set_offset(10)
str(query)
'select * from sd1 where f1 contains "v1" offset 10'
```

#### `set_timeout(timeout)`

Sets query timeout in milliseconds.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#timeout>

Parameters:

| Name      | Type  | Description             | Default    |
| --------- | ----- | ----------------------- | ---------- |
| `timeout` | `int` | Timeout in milliseconds | *required* |

Returns:

| Name    | Type    | Description              |
| ------- | ------- | ------------------------ |
| `Query` | `Query` | Self for method chaining |

Example

```python
import vespa.querybuilder as qb

f1 = qb.QueryField("f1")
query = qb.select("*").from_("sd1").where(f1.contains("v1")).set_timeout(500)
str(query)
'select * from sd1 where f1 contains "v1" timeout 500'
```

#### `add_parameter(key, value)`

Adds a query parameter.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#parameter-substitution>

Parameters:

| Name    | Type  | Description     | Default    |
| ------- | ----- | --------------- | ---------- |
| `key`   | `str` | Parameter name  | *required* |
| `value` | `Any` | Parameter value | *required* |

Returns:

| Name    | Type    | Description              |
| ------- | ------- | ------------------------ |
| `Query` | `Query` | Self for method chaining |

Example

```python
import vespa.querybuilder as qb

condition = qb.userInput("@myvar")
query = qb.select("*").from_("sd1").where(condition).add_parameter("myvar", "test")
str(query)
'select * from sd1 where userInput(@myvar)&myvar=test'
```

#### `param(key, value)`

Alias for add_parameter().

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#parameter-substitution>

Parameters:

| Name    | Type  | Description     | Default    |
| ------- | ----- | --------------- | ---------- |
| `key`   | `str` | Parameter name  | *required* |
| `value` | `Any` | Parameter value | *required* |

Returns:

| Name    | Type    | Description              |
| ------- | ------- | ------------------------ |
| `Query` | `Query` | Self for method chaining |

Example

```python
import vespa.querybuilder as qb

condition = qb.userInput("@animal")
query = qb.select("*").from_("sd1").where(condition).param("animal", "panda")
str(query)
'select * from sd1 where userInput(@animal)&animal=panda'
```

#### `groupby(group_expression, continuations=[])`

Groups results by specified expression.

For more information, see <https://docs.vespa.ai/en/grouping.html>

Also see for available methods to build group expressions.

Parameters:

| Name               | Type   | Description                                                                         | Default    |
| ------------------ | ------ | ----------------------------------------------------------------------------------- | ---------- |
| `group_expression` | `str`  | Grouping expression                                                                 | *required* |
| `continuations`    | `List` | List of continuation tokens (see https://docs.vespa.ai/en/grouping.html#pagination) | `[]`       |

Returns:

| Type    | Description                |
| ------- | -------------------------- |
| `Query` | : Self for method chaining |

Example

```python
import vespa.querybuilder as qb
from vespa.querybuilder import Grouping as G

# Group by customer with sum of price
grouping = G.all(
    G.group("customer"),
    G.each(G.output(G.sum("price"))),
)
str(grouping)
'all(group(customer) each(output(sum(price))))'
query = qb.select("*").from_("sd1").groupby(grouping)
str(query)
'select * from sd1 | all(group(customer) each(output(sum(price))))'

# Group by year with count
grouping = G.all(
    G.group("time.year(a)"),
    G.each(G.output(G.count())),
)
str(grouping)
'all(group(time.year(a)) each(output(count())))'
query = qb.select("*").from_("purchase").where(True).groupby(grouping)
str(query)
'select * from purchase where true | all(group(time.year(a)) each(output(count())))'
# With continuations
query = qb.select("*").from_("purchase").where(True).groupby(grouping, continuations=["foo", "bar"])
str(query)
"select * from purchase where true | { 'continuations':['foo', 'bar'] }all(group(time.year(a)) each(output(count())))"
```

### `Q`

Wrapper class for QueryBuilder static methods. Methods are exposed as module-level functions. To use:

```python
import vespa.querybuilder as qb

query = qb.select("*").from_("sd1") # or any of the other Q class methods
```

#### `select(fields)`

Creates a new query selecting specified fields.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#select>

Parameters:

| Name     | Type                                      | Description                                 | Default    |
| -------- | ----------------------------------------- | ------------------------------------------- | ---------- |
| `fields` | `Union[str, List[str], List[QueryField]]` | Field names or QueryField objects to select | *required* |

Returns:

| Name    | Type    | Description      |
| ------- | ------- | ---------------- |
| `Query` | `Query` | New query object |

Example

```python
import vespa.querybuilder as qb

query = qb.select("*").from_("sd1")
str(query)
'select * from sd1'

query = qb.select(["title", "url"])
str(query)
'select title, url from *'
```

#### `any(*conditions)`

"Combines multiple conditions with OR operator.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#or>

Parameters:

| Name         | Type        | Description                                             | Default |
| ------------ | ----------- | ------------------------------------------------------- | ------- |
| `conditions` | `Condition` | Variable number of Condition objects to combine with OR | `()`    |

Returns:

| Name        | Type        | Description                           |
| ----------- | ----------- | ------------------------------------- |
| `Condition` | `Condition` | Combined condition using OR operators |

Example

```python
import vespa.querybuilder as qb

f1, f2 = qb.QueryField("f1"), qb.QueryField("f2")
condition = qb.any(f1 > 10, f2 == "v2")
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where f1 > 10 or f2 = "v2"'
```

#### `all(*conditions)`

Combines multiple conditions with AND operator.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#and>

Parameters:

| Name          | Type        | Description                                              | Default |
| ------------- | ----------- | -------------------------------------------------------- | ------- |
| `*conditions` | `Condition` | Variable number of Condition objects to combine with AND | `()`    |

Returns:

| Name        | Type        | Description                            |
| ----------- | ----------- | -------------------------------------- |
| `Condition` | `Condition` | Combined condition using AND operators |

Example

```python
import vespa.querybuilder as qb

f1, f2 = qb.QueryField("f1"), qb.QueryField("f2")
condition = qb.all(f1 > 10, f2 == "v2")
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where f1 > 10 and f2 = "v2"'
```

#### `userQuery(value='')`

Creates a userQuery operator for text search.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#userquery>

Parameters:

| Name    | Type  | Description                                     | Default |
| ------- | ----- | ----------------------------------------------- | ------- |
| `value` | `str` | Optional query string. Default is empty string. | `''`    |

Returns:

| Name        | Type        | Description           |
| ----------- | ----------- | --------------------- |
| `Condition` | `Condition` | A userQuery condition |

Example

```python
import vespa.querybuilder as qb

# Basic userQuery
condition = qb.userQuery()
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where userQuery()'

# UserQuery with search terms
condition = qb.userQuery("search terms")
query = qb.select("*").from_("documents").where(condition)
str(query)
'select * from documents where userQuery("search terms")'
```

#### `dotProduct(field, weights, annotations=None)`

Creates a dot product calculation condition.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#dotproduct>.

Parameters:

| Name          | Type                                        | Description                                                                                                             | Default    |
| ------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------- |
| `field`       | `str`                                       | Field containing vectors                                                                                                | *required* |
| `weights`     | `Union[List[float], Dict[str, float], str]` | Either list of numeric weights or dict mapping elements to weights or a parameter substitution string starting with '@' | *required* |
| `annotations` | `Optional[Dict]`                            | Optional modifiers like label                                                                                           | `None`     |

Returns:

| Name        | Type        | Description                         |
| ----------- | ----------- | ----------------------------------- |
| `Condition` | `Condition` | A dot product calculation condition |

Example

```python
import vespa.querybuilder as qb

# Using dict weights with annotation
condition = qb.dotProduct(
    "weightedset_field",
    {"feature1": 1, "feature2": 2},
    annotations={"label": "myDotProduct"}
)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where ({label:"myDotProduct"}dotProduct(weightedset_field, {"feature1": 1, "feature2": 2}))'

# Using list weights
condition = qb.dotProduct("weightedset_field", [0.4, 0.6])
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where dotProduct(weightedset_field, [0.4, 0.6])'

# Using parameter substitution
condition = qb.dotProduct("weightedset_field", "@myweights")
query = qb.select("*").from_("sd1").where(condition).add_parameter("myweights", [0.4, 0.6])
str(query)
'select * from sd1 where dotProduct(weightedset_field, "@myweights")&myweights=[0.4, 0.6]'
```

#### `weightedSet(field, weights, annotations=None)`

Creates a weighted set condition.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#weightedset>.

Parameters:

| Name          | Type                                        | Description                                                                                                             | Default    |
| ------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------- |
| `field`       | `str`                                       | Field containing weighted set data                                                                                      | *required* |
| `weights`     | `Union[List[float], Dict[str, float], str]` | Either list of numeric weights or dict mapping elements to weights or a parameter substitution string starting with '@' | *required* |
| `annotations` | `Optional[Dict]`                            | Optional annotations like targetNumHits                                                                                 | `None`     |

Returns:

| Name        | Type        | Description              |
| ----------- | ----------- | ------------------------ |
| `Condition` | `Condition` | A weighted set condition |

Example

```python
import vespa.querybuilder as qb

# using map weights
condition = qb.weightedSet(
    "weightedset_field",
    {"element1": 1, "element2": 2},
    annotations={"targetNumHits": 10}
)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where ({targetNumHits:10}weightedSet(weightedset_field, {"element1": 1, "element2": 2}))'

# using list weights
condition = qb.weightedSet("weightedset_field", [0.4, 0.6])
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where weightedSet(weightedset_field, [0.4, 0.6])'

# using parameter substitution
condition = qb.weightedSet("weightedset_field", "@myweights")
query = qb.select("*").from_("sd1").where(condition).add_parameter("myweights", [0.4, 0.6])
str(query)
'select * from sd1 where weightedSet(weightedset_field, "@myweights")&myweights=[0.4, 0.6]'
```

#### `nonEmpty(condition)`

Creates a nonEmpty operator to check if a field has content.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#nonempty>.

Parameters:

| Name        | Type                           | Description                 | Default    |
| ----------- | ------------------------------ | --------------------------- | ---------- |
| `condition` | `Union[Condition, QueryField]` | Field or condition to check | *required* |

Returns:

| Name        | Type        | Description          |
| ----------- | ----------- | -------------------- |
| `Condition` | `Condition` | A nonEmpty condition |

Example

```python
import vespa.querybuilder as qb

field = qb.QueryField("title")
condition = qb.nonEmpty(field)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where nonEmpty(title)'
```

#### `wand(field, weights, annotations=None)`

Creates a Weighted AND (WAND) operator for efficient top-k retrieval.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#wand>.

Parameters:

| Name          | Type                                        | Description                                                                                                          | Default    |
| ------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------- |
| `field`       | `str`                                       | Field name to search                                                                                                 | *required* |
| `weights`     | `Union[List[float], Dict[str, float], str]` | Either list of numeric weights or dict mapping terms to weights or a parameter substitution string starting with '@' | *required* |
| `annotations` | `Optional[Dict[str, Any]]`                  | Optional annotations like targetHits                                                                                 | `None`     |

Returns:

| Name        | Type        | Description      |
| ----------- | ----------- | ---------------- |
| `Condition` | `Condition` | A WAND condition |

Example

```python
import vespa.querybuilder as qb

# Using list weights
condition = qb.wand("description", weights=[0.4, 0.6])
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where wand(description, [0.4, 0.6])'

# Using dict weights with annotation
weights = {"hello": 0.3, "world": 0.7}
condition = qb.wand(
    "title",
    weights,
    annotations={"targetHits": 100}
)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where ({targetHits: 100}wand(title, {"hello": 0.3, "world": 0.7}))'
```

#### `weakAnd(*conditions, annotations=None)`

Creates a weakAnd operator for less strict AND matching.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#weakand>.

Parameters:

| Name          | Type                       | Description                              | Default |
| ------------- | -------------------------- | ---------------------------------------- | ------- |
| `*conditions` | `Condition`                | Variable number of conditions to combine | `()`    |
| `annotations` | `Optional[Dict[str, Any]]` | Optional annotations like targetHits     | `None`  |

Returns:

| Name        | Type        | Description         |
| ----------- | ----------- | ------------------- |
| `Condition` | `Condition` | A weakAnd condition |

Example

```python
import vespa.querybuilder as qb

f1, f2 = qb.QueryField("f1"), qb.QueryField("f2")
condition = qb.weakAnd(f1 == "v1", f2 == "v2")
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where weakAnd(f1 = "v1", f2 = "v2")'

# With annotation
condition = qb.weakAnd(
    f1 == "v1",
    f2 == "v2",
    annotations={"targetHits": 100}
)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where ({"targetHits": 100}weakAnd(f1 = "v1", f2 = "v2"))'
```

#### `geoLocation(field, lat, lng, radius, annotations=None)`

Creates a geolocation search condition.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#geolocation>.

Parameters:

| Name          | Type             | Description                       | Default    |
| ------------- | ---------------- | --------------------------------- | ---------- |
| `field`       | `str`            | Field containing location data    | *required* |
| `lat`         | `float`          | Latitude coordinate               | *required* |
| `lon`         | `float`          | Longitude coordinate              | *required* |
| `radius`      | `str`            | Search radius (e.g. "10km")       | *required* |
| `annotations` | `Optional[Dict]` | Optional settings like targetHits | `None`     |

Returns:

| Name        | Type        | Description                    |
| ----------- | ----------- | ------------------------------ |
| `Condition` | `Condition` | A geolocation search condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.geoLocation(
    "location_field",
    37.7749,
    -122.4194,
    "10km",
    annotations={"targetHits": 100}
)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where ({targetHits:100}geoLocation(location_field, 37.7749, -122.4194, "10km"))'
```

#### `nearestNeighbor(field, query_vector, annotations={'targetHits': 100})`

Creates a nearest neighbor search condition.

See <https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor> for more information.

Parameters:

| Name           | Type             | Description                                                                                | Default               |
| -------------- | ---------------- | ------------------------------------------------------------------------------------------ | --------------------- |
| `field`        | `str`            | Vector field to search in                                                                  | *required*            |
| `query_vector` | `str`            | Query vector to compare against                                                            | *required*            |
| `annotations`  | `Dict[str, Any]` | Optional annotations to modify the behavior. Required annotation: targetHits (default: 10) | `{'targetHits': 100}` |

Returns:

| Name        | Type        | Description                         |
| ----------- | ----------- | ----------------------------------- |
| `Condition` | `Condition` | A nearest neighbor search condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.nearestNeighbor(
    field="dense_rep",
    query_vector="q_dense",
)
query = qb.select(["id, text"]).from_("m").where(condition)
str(query)
'select id, text from m where ({targetHits:100}nearestNeighbor(dense_rep, q_dense))'
```

#### `rank(*queries)`

Creates a rank condition for combining multiple queries.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#rank>

Parameters:

| Name       | Type | Description                                 | Default |
| ---------- | ---- | ------------------------------------------- | ------- |
| `*queries` |      | Variable number of Query objects to combine | `()`    |

Returns:

| Name        | Type        | Description      |
| ----------- | ----------- | ---------------- |
| `Condition` | `Condition` | A rank condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.rank(
    qb.nearestNeighbor("field", "queryVector"),
    qb.QueryField("a").contains("A"),
    qb.QueryField("b").contains("B"),
    qb.QueryField("c").contains("C"),
)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where rank(({targetHits:100}nearestNeighbor(field, queryVector)), a contains "A", b contains "B", c contains "C")'
```

#### `phrase(*terms, annotations=None)`

Creates a phrase search operator for exact phrase matching.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#phrase>

Parameters:

| Name          | Type                       | Description                   | Default |
| ------------- | -------------------------- | ----------------------------- | ------- |
| `*terms`      | `str`                      | Terms that make up the phrase | `()`    |
| `annotations` | `Optional[Dict[str, Any]]` | Optional annotations          | `None`  |

Returns:

| Name        | Type        | Description        |
| ----------- | ----------- | ------------------ |
| `Condition` | `Condition` | A phrase condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.phrase("new", "york", "city")
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where phrase("new", "york", "city")'
```

#### `near(*terms, distance=None, annotations=None, **kwargs)`

Creates a near search operator for finding terms within a specified distance.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#near>

Parameters:

| Name          | Type                       | Description                                                              | Default |
| ------------- | -------------------------- | ------------------------------------------------------------------------ | ------- |
| `*terms`      | `str`                      | Terms to search for                                                      | `()`    |
| `distance`    | `Optional[int]`            | Maximum word distance between terms. Will default to 2 if not specified. | `None`  |
| `annotations` | `Optional[Dict[str, Any]]` | Optional annotations                                                     | `None`  |
| `**kwargs`    |                            | Additional annotations                                                   | `{}`    |

Returns:

| Name        | Type        | Description      |
| ----------- | ----------- | ---------------- |
| `Condition` | `Condition` | A near condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.near("machine", "learning", distance=5)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where ({distance:5}near("machine", "learning"))'
```

#### `onear(*terms, distance=None, annotations=None, **kwargs)`

Creates an ordered near operator for ordered proximity search.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#onear>

Parameters:

| Name          | Type                       | Description                                                              | Default |
| ------------- | -------------------------- | ------------------------------------------------------------------------ | ------- |
| `*terms`      | `str`                      | Terms to search for in order                                             | `()`    |
| `distance`    | `Optional[int]`            | Maximum word distance between terms. Will default to 2 if not specified. | `None`  |
| `annotations` | `Optional[Dict[str, Any]]` | Optional annotations                                                     | `None`  |

Returns:

| Name        | Type        | Description        |
| ----------- | ----------- | ------------------ |
| `Condition` | `Condition` | An onear condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.onear("deep", "learning", distance=3)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where ({distance:3}onear("deep", "learning"))'
```

#### `sameElement(*conditions)`

Creates a sameElement operator to match conditions in same array element.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#sameelement>

Parameters:

| Name          | Type        | Description                                | Default |
| ------------- | ----------- | ------------------------------------------ | ------- |
| `*conditions` | `Condition` | Conditions that must match in same element | `()`    |

Returns:

| Name        | Type        | Description             |
| ----------- | ----------- | ----------------------- |
| `Condition` | `Condition` | A sameElement condition |

Example

```python
import vespa.querybuilder as qb

persons = qb.QueryField("persons")
first_name = qb.QueryField("first_name")
last_name = qb.QueryField("last_name")
year_of_birth = qb.QueryField("year_of_birth")
condition = persons.contains(
    qb.sameElement(
        first_name.contains("Joe"),
        last_name.contains("Smith"),
        year_of_birth < 1940,
    )
 )
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where persons contains sameElement(first_name contains "Joe", last_name contains "Smith", year_of_birth < 1940)'
```

#### `equiv(*terms)`

Creates an equiv operator for matching equivalent terms.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#equiv>

Parameters:

| Name          | Type                       | Description              | Default    |
| ------------- | -------------------------- | ------------------------ | ---------- |
| `terms`       | `List[str]`                | List of equivalent terms | `()`       |
| `annotations` | `Optional[Dict[str, Any]]` | Optional annotations     | *required* |

Returns:

| Name        | Type        | Description        |
| ----------- | ----------- | ------------------ |
| `Condition` | `Condition` | An equiv condition |

Example

```python
import vespa.querybuilder as qb

fieldName = qb.QueryField("fieldName")
condition = fieldName.contains(qb.equiv("Snoop Dogg", "Calvin Broadus"))
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where fieldName contains equiv("Snoop Dogg", "Calvin Broadus")'
```

#### `uri(value, annotations=None)`

Creates a uri operator for matching URIs.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#uri>

Parameters:

| Name    | Type  | Description               | Default    |
| ------- | ----- | ------------------------- | ---------- |
| `field` | `str` | Field name containing URI | *required* |
| `value` | `str` | URI value to match        | *required* |

Returns:

| Name        | Type        | Description     |
| ----------- | ----------- | --------------- |
| `Condition` | `Condition` | A uri condition |

Example

```python
import vespa.querybuilder as qb

url = "vespa.ai/foo"
condition = qb.uri(url)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where uri("vespa.ai/foo")'
```

#### `fuzzy(value, annotations=None, **kwargs)`

Creates a fuzzy operator for approximate string matching.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#fuzzy>

Parameters:

| Name          | Type                       | Description                                                  | Default    |
| ------------- | -------------------------- | ------------------------------------------------------------ | ---------- |
| `term`        | `str`                      | Term to fuzzy match                                          | *required* |
| `annotations` | `Optional[Dict[str, Any]]` | Optional annotations                                         | `None`     |
| `**kwargs`    |                            | Optional parameters like maxEditDistance, prefixLength, etc. | `{}`       |

Returns:

| Name        | Type        | Description       |
| ----------- | ----------- | ----------------- |
| `Condition` | `Condition` | A fuzzy condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.fuzzy("parantesis")
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where fuzzy("parantesis")'

# With annotation
condition = qb.fuzzy("parantesis", annotations={"prefixLength": 1, "maxEditDistance": 2})
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where ({prefixLength:1,maxEditDistance:2}fuzzy("parantesis"))'
```

#### `userInput(value=None, annotations=None)`

Creates a userInput operator for query evaluation.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#userinput>.

Parameters:

| Name          | Type             | Description                                 | Default |
| ------------- | ---------------- | ------------------------------------------- | ------- |
| `value`       | `Optional[str]`  | The input variable name, e.g. "@myvar"      | `None`  |
| `annotations` | `Optional[Dict]` | Optional annotations to modify the behavior | `None`  |

Returns:

| Name        | Type        | Description                                     |
| ----------- | ----------- | ----------------------------------------------- |
| `Condition` | `Condition` | A condition representing the userInput operator |

Example

```python
import vespa.querybuilder as qb

condition = qb.userInput("@myvar")
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where userInput(@myvar)'

# With defaultIndex annotation
condition = qb.userInput("@myvar").annotate({"defaultIndex": "text"})
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where {defaultIndex:"text"}userInput(@myvar)'

# With parameter
condition = qb.userInput("@animal")
query = qb.select("*").from_("sd1").where(condition).param("animal", "panda")
str(query)
'select * from sd1 where userInput(@animal)&animal=panda'
```

#### `predicate(field, attributes=None, range_attributes=None)`

Creates a predicate condition for filtering documents based on specific attributes.

For more information, see <https://docs.vespa.ai/en/reference/query-language-reference.html#predicate>.

Parameters:

| Name               | Type                       | Description                                   | Default    |
| ------------------ | -------------------------- | --------------------------------------------- | ---------- |
| `field`            | `str`                      | The predicate field name                      | *required* |
| `attributes`       | `Optional[Dict[str, Any]]` | Dictionary of attribute key-value pairs       | `None`     |
| `range_attributes` | `Optional[Dict[str, Any]]` | Dictionary of range attribute key-value pairs | `None`     |

Returns:

| Name        | Type        | Description                                      |
| ----------- | ----------- | ------------------------------------------------ |
| `Condition` | `Condition` | A condition representing the predicate operation |

Example

```python
import vespa.querybuilder as qb

condition = qb.predicate(
    "predicate_field",
    attributes={"gender": "Female"},
    range_attributes={"age": "20L"}
)
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where predicate(predicate_field,{"gender":"Female"},{"age":20L})'
```

#### `true()`

Creates a condition that is always true.

Returns:

| Name        | Type        | Description      |
| ----------- | ----------- | ---------------- |
| `Condition` | `Condition` | A true condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.true()
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where true'
```

#### `false()`

Creates a condition that is always false.

Returns:

| Name        | Type        | Description       |
| ----------- | ----------- | ----------------- |
| `Condition` | `Condition` | A false condition |

Example

```python
import vespa.querybuilder as qb

condition = qb.false()
query = qb.select("*").from_("sd1").where(condition)
str(query)
'select * from sd1 where false'
```
