## `vespa.querybuilder.grouping.grouping`

### `Expression`

Bases: `str`

#### `__invert__()`

~expr → 'not (expr)' for filter predicates. Always wraps in parens so Vespa's precedence (not > and > or) does not reinterpret compound expressions.

#### `__and__(other)`

expr & expr → 'expr and expr' for filter predicates.

#### `__or__(other)`

expr | expr → '(expr or expr)' for filter predicates. Always wraps in parens so precedence is correct when combined with &.

### `Grouping`

A Pythonic DSL for building Vespa grouping expressions programmatically.

This class provides a set of static methods that build grouping syntax strings which can be combined to form a valid Vespa “select=…” grouping expression.

For a guide to grouping in vespa, see <https://docs.vespa.ai/en/grouping.html>. For the reference docs, see <https://docs.vespa.ai/en/reference/grouping-syntax.html>.

Minimal Example

```python
from vespa.querybuilder import Grouping as G

# Build a simple grouping expression which groups on "my_attribute"
# and outputs the count of matching documents under each group:
expr = G.all(
    G.group("my_attribute"),
    G.each(
        G.output(G.count())
    )
)
print(expr)
all(group(my_attribute) each(output(count())))
```

In the above example, the “all(...)” wraps the grouping operations at the top level. We first group on “my_attribute”, then under “each(...)” we add an output aggregator “count()”. The “print” output is the exact grouping expression string you would pass to Vespa in the “select” query parameter.

For multi-level (nested) grouping, you can nest additional calls to “group(...)” or “each(...)” inside. For example:

```python
# Nested grouping:
# 1) Group by 'category'
# 2) Within each category, group by 'sub_category'
# 3) Output the count() under each sub-category
nested_expr = G.all(
    G.group("category"),
    G.each(
        G.group("sub_category"),
        G.each(
            G.output(G.count())
        )
    )
)
print(nested_expr)
all(group(category) each(group(sub_category) each(output(count()))))
```

You may use any of the static methods below to build more advanced groupings, aggregations, or arithmetic/string expressions for sorting, filtering, or bucket definitions. Refer to Vespa documentation for the complete details.

#### `all(*args)`

Corresponds to the “all(...)” grouping block in Vespa, which means “group all documents (no top-level grouping) and then do the enclosed operations”.

Parameters:

| Name    | Type  | Description                                           | Default |
| ------- | ----- | ----------------------------------------------------- | ------- |
| `*args` | `str` | Sub-expressions to include within the all(...) block. | `()`    |

Returns:

| Name  | Type         | Description                         |
| ----- | ------------ | ----------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.all(G.group("my_attribute"), G.each(G.output(G.count())))
print(expr)
all(group(my_attribute) each(output(count())))
```

#### `each(*args)`

Corresponds to the “each(...)” grouping block in Vespa, which means “create a group for each unique value and then do the enclosed operations”.

Parameters:

| Name    | Type  | Description                                            | Default |
| ------- | ----- | ------------------------------------------------------ | ------- |
| `*args` | `str` | Sub-expressions to include within the each(...) block. | `()`    |

Returns:

| Name  | Type         | Description                         |
| ----- | ------------ | ----------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.each("output(count())", "output(avg(price))")
print(expr)
each(output(count()) output(avg(price)))
```

#### `group(field)`

Defines a grouping step on a field or expression.

Parameters:

| Name    | Type  | Description                                | Default    |
| ------- | ----- | ------------------------------------------ | ---------- |
| `field` | `str` | The field or expression on which to group. | *required* |

Returns:

| Name  | Type         | Description                         |
| ----- | ------------ | ----------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.group("my_map.key")
print(expr)
group(my_map.key)
```

#### `filter_(predicate)`

Wraps a predicate in `filter(...)` for use inside a grouping expression.

Parameters:

| Name        | Type | Description                                                       | Default    |
| ----------- | ---- | ----------------------------------------------------------------- | ---------- |
| `predicate` |      | A filter predicate expression (e.g. G.regex(...), G.istrue(...)). | *required* |

Returns:

| Name         | Type         | Description         |
| ------------ | ------------ | ------------------- |
| `Expression` | `Expression` | filter(<predicate>) |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.all(
    G.group("customer"),
    G.filter_(G.regex("Bonn.*", 'attributes{"sales_rep"}') & ~G.range_(0, 1000, "price")),
    G.each(G.output(G.sum("price"))),
)
print(expr)
all(group(customer) filter(regex("Bonn.*", attributes{"sales_rep"}) and not (range(0, 1000, price))) each(output(sum(price))))
```

#### `regex(pattern, expr)`

Creates a `regex(...)` filter predicate.

Parameters:

| Name      | Type  | Description                                      | Default    |
| --------- | ----- | ------------------------------------------------ | ---------- |
| `pattern` | `str` | The regular expression pattern (will be quoted). | *required* |
| `expr`    | `str` | The field or expression to match against.        | *required* |

Returns:

| Name         | Type         | Description                |
| ------------ | ------------ | -------------------------- |
| `Expression` | `Expression` | regex("<pattern>", <expr>) |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.regex("foo.*", "my_field")
print(expr)
regex("foo.*", my_field)
```

#### `range_(min_val, max_val, expr, lower_inclusive=None, upper_inclusive=None)`

Creates a `range(...)` filter predicate.

Vespa defaults: lower bound is inclusive, upper bound is exclusive. If either `lower_inclusive` or `upper_inclusive` is provided, both are emitted using Vespa defaults (`true`/`false`) for any omitted value.

Parameters:

| Name              | Type  | Description                                                            | Default    |
| ----------------- | ----- | ---------------------------------------------------------------------- | ---------- |
| `min_val`         |       | Lower bound of the range.                                              | *required* |
| `max_val`         |       | Upper bound of the range.                                              | *required* |
| `expr`            | `str` | The field or expression to check.                                      | *required* |
| `lower_inclusive` |       | Whether the lower bound is inclusive (default: True, matching Vespa).  | `None`     |
| `upper_inclusive` |       | Whether the upper bound is inclusive (default: False, matching Vespa). | `None`     |

Returns:

| Name         | Type         | Description                                       |
| ------------ | ------------ | ------------------------------------------------- |
| `Expression` | `Expression` | range(<min>, <max>, <expr>\[, <lower>, <upper>\]) |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.range_(1990, 2012, "year")
print(expr)
range(1990, 2012, year)

expr = G.range_(1990, 2012, "year", True, True)
print(expr)
range(1990, 2012, year, true, true)
```

#### `istrue(expr)`

Creates an `istrue(...)` filter predicate.

Parameters:

| Name   | Type  | Description                                      | Default    |
| ------ | ----- | ------------------------------------------------ | ---------- |
| `expr` | `str` | The field or expression to check for truthiness. | *required* |

Returns:

| Name         | Type         | Description    |
| ------------ | ------------ | -------------- |
| `Expression` | `Expression` | istrue(<expr>) |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.istrue("my_bool")
print(expr)
istrue(my_bool)
```

#### `count()`

“count()” aggregator.

By default, returns a string 'count()'. Negative ordering or usage can be done by prefixing a minus, e.g.: order(-count()) in Vespa syntax.

Returns:

| Name  | Type         | Description                                                  |
| ----- | ------------ | ------------------------------------------------------------ |
| `str` | `Expression` | 'count()' or prefixed version if used with a minus operator. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.count()
print(expr)
count()

sort_expr = f"-{expr}"
print(sort_expr)
-count()
```

#### `sum(value)`

“sum(...)” aggregator. Sums the given expression or field over all documents in the group.

Parameters:

| Name    | Type                     | Description                             | Default    |
| ------- | ------------------------ | --------------------------------------- | ---------- |
| `value` | `Union[str, int, float]` | The field or numeric expression to sum. | *required* |

Returns:

| Name  | Type         | Description                                                |
| ----- | ------------ | ---------------------------------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string of the form 'sum(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.sum("my_numeric_field")
print(expr)
sum(my_numeric_field)
```

#### `avg(value)`

“avg(...)” aggregator. Computes the average of the given expression or field for all documents in the group.

Parameters:

| Name    | Type                     | Description                                 | Default    |
| ------- | ------------------------ | ------------------------------------------- | ---------- |
| `value` | `Union[str, int, float]` | The field or numeric expression to average. | *required* |

Returns:

| Name  | Type         | Description                                                |
| ----- | ------------ | ---------------------------------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string of the form 'avg(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.avg("my_numeric_field")
print(expr)
avg(my_numeric_field)
```

#### `min(value)`

“min(...)” aggregator. Keeps the minimum value of the expression or field among all documents in the group.

Parameters:

| Name    | Type                     | Description                                             | Default    |
| ------- | ------------------------ | ------------------------------------------------------- | ---------- |
| `value` | `Union[str, int, float]` | The field or numeric expression to find the minimum of. | *required* |

Returns:

| Name  | Type         | Description                                                |
| ----- | ------------ | ---------------------------------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string of the form 'min(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.min("some_field")
print(expr)
min(some_field)
```

#### `max(value)`

“max(...)” aggregator. Keeps the maximum value of the expression or field among all documents in the group.

Parameters:

| Name    | Type                     | Description                                             | Default    |
| ------- | ------------------------ | ------------------------------------------------------- | ---------- |
| `value` | `Union[str, int, float]` | The field or numeric expression to find the maximum of. | *required* |

Returns:

| Name  | Type         | Description                                                |
| ----- | ------------ | ---------------------------------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string of the form 'max(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.max("relevance()")
print(expr)
max(relevance())
```

#### `stddev(value)`

“stddev(...)” aggregator. Computes the population standard deviation for the expression or field among all documents in the group.

Parameters:

| Name    | Type                     | Description                      | Default    |
| ------- | ------------------------ | -------------------------------- | ---------- |
| `value` | `Union[str, int, float]` | The field or numeric expression. | *required* |

Returns:

| Name  | Type         | Description                                                   |
| ----- | ------------ | ------------------------------------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string of the form 'stddev(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.stddev("my_numeric_field")
print(expr)
stddev(my_numeric_field)
```

#### `xor(value)`

“xor(...)” aggregator. XORs all values of the expression or field together over the documents in the group.

Parameters:

| Name    | Type                     | Description                      | Default    |
| ------- | ------------------------ | -------------------------------- | ---------- |
| `value` | `Union[str, int, float]` | The field or numeric expression. | *required* |

Returns:

| Name  | Type         | Description                                                |
| ----- | ------------ | ---------------------------------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string of the form 'xor(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.xor("my_field")
print(expr)
xor(my_field)
```

#### `output(*args)`

Defines output aggregators to be collected for the grouping level.

Parameters:

| Name    | Type                     | Description                                                     | Default |
| ------- | ------------------------ | --------------------------------------------------------------- | ------- |
| `*args` | `Union[str, Expression]` | Multiple aggregator expressions, e.g., 'count()', 'sum(price)'. | `()`    |

Returns:

| Name         | Type         | Description                                                   |
| ------------ | ------------ | ------------------------------------------------------------- |
| `Expression` | `Expression` | A Vespa grouping expression string of the form 'output(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.output(G.count(), G.sum("price"))
print(expr)
output(count(),sum(price))
```

#### `order(*args)`

Defines an order(...) clause to sort groups by the given expressions or aggregators.

Parameters:

| Name    | Type                     | Description                                      | Default |
| ------- | ------------------------ | ------------------------------------------------ | ------- |
| `*args` | `Union[str, Expression]` | Multiple expressions or aggregators to order by. | `()`    |

Returns:

| Name         | Type         | Description                                                  |
| ------------ | ------------ | ------------------------------------------------------------ |
| `Expression` | `Expression` | A Vespa grouping expression string of the form 'order(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.order(G.sum(G.relevance()), -G.count())
print(expr)
order(sum(relevance()),-count())
```

#### `precision(value)`

Sets the “precision(...)” for the grouping step.

Parameters:

| Name    | Type  | Description      | Default    |
| ------- | ----- | ---------------- | ---------- |
| `value` | `int` | Precision value. | *required* |

Returns:

| Name  | Type         | Description                                                      |
| ----- | ------------ | ---------------------------------------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string of the form 'precision(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.precision(1000)
print(expr)
precision(1000)
```

#### `add(*expressions)`

“add(...)” expression. Adds all arguments together in order.

Parameters:

| Name           | Type  | Description                  | Default |
| -------------- | ----- | ---------------------------- | ------- |
| `*expressions` | `str` | The expressions to be added. | `()`    |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'add(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.add("my_field", "5", "10")
print(expr)
add(my_field, 5, 10)
```

#### `sub(*expressions)`

“sub(...)” expression. Subtracts each subsequent argument from the first.

Parameters:

| Name           | Type  | Description                              | Default |
| -------------- | ----- | ---------------------------------------- | ------- |
| `*expressions` | `str` | The expressions involved in subtraction. | `()`    |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'sub(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.sub("my_field", "2")
print(expr)
sub(my_field, 2)
```

#### `mul(*expressions)`

“mul(...)” expression. Multiplies all arguments in order.

Parameters:

| Name           | Type  | Description                  | Default |
| -------------- | ----- | ---------------------------- | ------- |
| `*expressions` | `str` | The expressions to multiply. | `()`    |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'mul(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.mul("my_field", "2", "3")
print(expr)
mul(my_field, 2, 3)
```

#### `div(*expressions)`

“div(...)” expression. Divides the first argument by the second, etc.

Parameters:

| Name           | Type  | Description                         | Default |
| -------------- | ----- | ----------------------------------- | ------- |
| `*expressions` | `str` | The expressions to divide in order. | `()`    |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'div(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.div("my_field", "2")
print(expr)
div(my_field, 2)
```

#### `mod(*expressions)`

“mod(...)” expression. Modulo the first argument by the second, result by the third, etc.

Parameters:

| Name           | Type  | Description                                  | Default |
| -------------- | ----- | -------------------------------------------- | ------- |
| `*expressions` | `str` | The expressions to apply modulo on in order. | `()`    |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'mod(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.mod("my_field", "100")
print(expr)
mod(my_field,100)
```

#### `and_(*expressions)`

“and(...)” expression. Bitwise AND of the arguments in order.

Parameters:

| Name           | Type  | Description                           | Default |
| -------------- | ----- | ------------------------------------- | ------- |
| `*expressions` | `str` | The expressions to apply bitwise AND. | `()`    |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'and(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.and_("fieldA", "fieldB")
print(expr)
and(fieldA, fieldB)
```

#### `or_(*expressions)`

“or(...)” expression. Bitwise OR of the arguments in order.

Parameters:

| Name           | Type  | Description                          | Default |
| -------------- | ----- | ------------------------------------ | ------- |
| `*expressions` | `str` | The expressions to apply bitwise OR. | `()`    |

Returns:

| Name  | Type         | Description                                                    |
| ----- | ------------ | -------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'or(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.or_("fieldA", "fieldB")
print(expr)
or(fieldA, fieldB)
```

#### `xor_expr(*expressions)`

“xor(...)” bitwise expression.

(Note: For aggregator use, see xor(...) aggregator method above.)

Parameters:

| Name           | Type  | Description                           | Default |
| -------------- | ----- | ------------------------------------- | ------- |
| `*expressions` | `str` | The expressions to apply bitwise XOR. | `()`    |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'xor(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.xor_expr("fieldA", "fieldB")
print(expr)
xor(fieldA, fieldB)
```

#### `strlen(expr)`

“strlen(...)” expression. Returns the number of bytes in the string.

Parameters:

| Name   | Type  | Description                     | Default    |
| ------ | ----- | ------------------------------- | ---------- |
| `expr` | `str` | The string field or expression. | *required* |

Returns:

| Name  | Type         | Description                                          |
| ----- | ------------ | ---------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'strlen(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.strlen("my_string_field")
print(expr)
strlen(my_string_field)
```

#### `strcat(*expressions)`

“strcat(...)” expression. Concatenate all string arguments in order.

Parameters:

| Name           | Type  | Description                            | Default |
| -------------- | ----- | -------------------------------------- | ------- |
| `*expressions` | `str` | The string expressions to concatenate. | `()`    |

Returns:

| Name  | Type         | Description                                                        |
| ----- | ------------ | ------------------------------------------------------------------ |
| `str` | `Expression` | A Vespa expression string of the form 'strcat(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.strcat("fieldA", "_", "fieldB")
print(expr)
strcat(fieldA,_,fieldB)
```

#### `todouble(expr)`

“todouble(...)” expression. Convert argument to double.

Parameters:

| Name   | Type  | Description                         | Default    |
| ------ | ----- | ----------------------------------- | ---------- |
| `expr` | `str` | The expression or field to convert. | *required* |

Returns:

| Name  | Type         | Description                                            |
| ----- | ------------ | ------------------------------------------------------ |
| `str` | `Expression` | A Vespa expression string of the form 'todouble(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.todouble("my_field")
print(expr)
todouble(my_field)
```

#### `tolong(expr)`

“tolong(...)” expression. Convert argument to long.

Parameters:

| Name   | Type  | Description                         | Default    |
| ------ | ----- | ----------------------------------- | ---------- |
| `expr` | `str` | The expression or field to convert. | *required* |

Returns:

| Name  | Type         | Description                                          |
| ----- | ------------ | ---------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'tolong(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.tolong("my_field")
print(expr)
tolong(my_field)
```

#### `tostring(expr)`

“tostring(...)” expression. Convert argument to string.

Parameters:

| Name   | Type  | Description                         | Default    |
| ------ | ----- | ----------------------------------- | ---------- |
| `expr` | `str` | The expression or field to convert. | *required* |

Returns:

| Name  | Type         | Description                                            |
| ----- | ------------ | ------------------------------------------------------ |
| `str` | `Expression` | A Vespa expression string of the form 'tostring(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.tostring("my_field")
print(expr)
tostring(my_field)
```

#### `toraw(expr)`

“toraw(...)” expression. Convert argument to raw data.

Parameters:

| Name   | Type  | Description                         | Default    |
| ------ | ----- | ----------------------------------- | ---------- |
| `expr` | `str` | The expression or field to convert. | *required* |

Returns:

| Name  | Type         | Description                                         |
| ----- | ------------ | --------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'toraw(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.toraw("my_field")
print(expr)
toraw(my_field)
```

#### `cat(*expressions)`

“cat(...)” expression. Concatenate the binary representation of arguments.

Parameters:

| Name           | Type  | Description                                      | Default |
| -------------- | ----- | ------------------------------------------------ | ------- |
| `*expressions` | `str` | The binary expressions or fields to concatenate. | `()`    |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'cat(expr1, expr2, ...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.cat("fieldA", "fieldB")
print(expr)
cat(fieldA,fieldB)
```

#### `md5(expr, width)`

“md5(...)” expression.

Does an MD5 over the binary representation of the argument, and keeps the lowest 'width' bits.

Parameters:

| Name    | Type  | Description                              | Default    |
| ------- | ----- | ---------------------------------------- | ---------- |
| `expr`  | `str` | The expression or field to apply MD5 on. | *required* |
| `width` | `int` | The number of bits to keep.              | *required* |

Returns:

| Name  | Type         | Description                                               |
| ----- | ------------ | --------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'md5(expr, width)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.md5("my_field", 16)
print(expr)
md5(my_field, 16)
```

#### `xorbit(expr, width)`

“xorbit(...)” expression.

Performs an XOR of 'width' bits over the binary representation of the argument. Width is rounded up to a multiple of 8.

Parameters:

| Name    | Type  | Description                                 | Default    |
| ------- | ----- | ------------------------------------------- | ---------- |
| `expr`  | `str` | The expression or field to apply xorbit on. | *required* |
| `width` | `int` | The number of bits for the XOR operation.   | *required* |

Returns:

| Name  | Type         | Description                                                  |
| ----- | ------------ | ------------------------------------------------------------ |
| `str` | `Expression` | A Vespa expression string of the form 'xorbit(expr, width)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.xorbit("my_field", 16)
print(expr)
xorbit(my_field, 16)
```

#### `relevance()`

“relevance()” expression. Returns the computed rank (relevance) of a document.

Returns:

| Name  | Type         | Description                                 |
| ----- | ------------ | ------------------------------------------- |
| `str` | `Expression` | 'relevance()' as a Vespa expression string. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.relevance()
print(expr)
relevance()
```

#### `array_at(array_name, index_expr)`

“array.at(...)” accessor expression. Returns a single element from the array at the given index.

Parameters:

| Name         | Type              | Description                                         | Default    |
| ------------ | ----------------- | --------------------------------------------------- | ---------- |
| `array_name` | `str`             | The name of the array.                              | *required* |
| `index_expr` | `Union[str, int]` | The index or expression that evaluates to an index. | *required* |

Returns:

| Name  | Type         | Description                                                          |
| ----- | ------------ | -------------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'array.at(array_name, index)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.array_at("my_array", 0)
print(expr)
array.at(my_array, 0)
```

#### `zcurve_x(expr)`

“zcurve.x(...)” expression. Returns the X component of the given zcurve-encoded 2D point.

Parameters:

| Name   | Type  | Description                             | Default    |
| ------ | ----- | --------------------------------------- | ---------- |
| `expr` | `str` | The zcurve-encoded field or expression. | *required* |

Returns:

| Name  | Type         | Description                                             |
| ----- | ------------ | ------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'zcurve.x(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.zcurve_x("location_zcurve")
print(expr)
zcurve.x(location_zcurve)
```

#### `zcurve_y(expr)`

“zcurve.y(...)” expression. Returns the Y component of the given zcurve-encoded 2D point.

Parameters:

| Name   | Type  | Description                             | Default    |
| ------ | ----- | --------------------------------------- | ---------- |
| `expr` | `str` | The zcurve-encoded field or expression. | *required* |

Returns:

| Name  | Type         | Description                                             |
| ----- | ------------ | ------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'zcurve.y(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.zcurve_y("location_zcurve")
print(expr)
zcurve.y(location_zcurve)
```

#### `time_dayofmonth(expr)`

“time.dayofmonth(...)” expression. Returns the day of month (1-31).

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                                    |
| ----- | ------------ | -------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'time.dayofmonth(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_dayofmonth("timestamp_field")
print(expr)
time.dayofmonth(timestamp_field)
```

#### `time_dayofweek(expr)`

“time.dayofweek(...)” expression. Returns the day of week (0-6), Monday = 0.

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                                   |
| ----- | ------------ | ------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'time.dayofweek(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_dayofweek("timestamp_field")
print(expr)
time.dayofweek(timestamp_field)
```

#### `time_dayofyear(expr)`

“time.dayofyear(...)” expression. Returns the day of year (0-365).

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                                   |
| ----- | ------------ | ------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'time.dayofyear(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_dayofyear("timestamp_field")
print(expr)
time.dayofyear(timestamp_field)
```

#### `time_hourofday(expr)`

“time.hourofday(...)” expression. Returns the hour of day (0-23).

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                                   |
| ----- | ------------ | ------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'time.hourofday(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_hourofday("timestamp_field")
print(expr)
time.hourofday(timestamp_field)
```

#### `time_minuteofhour(expr)`

“time.minuteofhour(...)” expression. Returns the minute of hour (0-59).

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                                      |
| ----- | ------------ | ---------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'time.minuteofhour(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_minuteofhour("timestamp_field")
print(expr)
time.minuteofhour(timestamp_field)
```

#### `time_monthofyear(expr)`

“time.monthofyear(...)” expression. Returns the month of year (1-12).

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'time.monthofyear(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_monthofyear("timestamp_field")
print(expr)
time.monthofyear(timestamp_field)
```

#### `time_secondofminute(expr)`

“time.secondofminute(...)” expression. Returns the second of minute (0-59).

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                                        |
| ----- | ------------ | ------------------------------------------------------------------ |
| `str` | `Expression` | A Vespa expression string of the form 'time.secondofminute(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_secondofminute("timestamp_field")
print(expr)
time.secondofminute(timestamp_field)
```

#### `time_year(expr)`

“time.year(...)” expression. Returns the full year (e.g. 2009).

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                              |
| ----- | ------------ | -------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'time.year(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_year("timestamp_field")
print(expr)
time.year(timestamp_field)
```

#### `time_date(expr)`

“time.date(...)” expression. Returns the date (e.g. 2009-01-10).

Parameters:

| Name   | Type  | Description                        | Default    |
| ------ | ----- | ---------------------------------- | ---------- |
| `expr` | `str` | The timestamp field or expression. | *required* |

Returns:

| Name  | Type         | Description                                              |
| ----- | ------------ | -------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'time.date(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.time_date("timestamp_field")
print(expr)
time.date(timestamp_field)
```

#### `math_exp(expr)`

“math.exp(...)” expression. Returns e^expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description                                             |
| ----- | ------------ | ------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'math.exp(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_exp("my_field")
print(expr)
math.exp(my_field)
```

#### `math_log(expr)`

“math.log(...)” expression. Returns the natural logarithm of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description       |
| ----- | ------------ | ----------------- |
| `str` | `Expression` | 'math.log(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_log("my_field")
print(expr)
math.log(my_field)
```

#### `math_log1p(expr)`

“math.log1p(...)” expression. Returns the natural logarithm of (1 + expr).

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description         |
| ----- | ------------ | ------------------- |
| `str` | `Expression` | 'math.log1p(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_log1p("my_field")
print(expr)
math.log1p(my_field)
```

#### `math_log10(expr)`

“math.log10(...)” expression. Returns the base-10 logarithm of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description         |
| ----- | ------------ | ------------------- |
| `str` | `Expression` | 'math.log10(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_log10("my_field")
print(expr)
math.log10(my_field)
```

#### `math_sqrt(expr)`

“math.sqrt(...)” expression. Returns the square root of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description        |
| ----- | ------------ | ------------------ |
| `str` | `Expression` | 'math.sqrt(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_sqrt("my_field")
print(expr)
math.sqrt(my_field)
```

#### `math_cbrt(expr)`

“math.cbrt(...)” expression. Returns the cube root of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description        |
| ----- | ------------ | ------------------ |
| `str` | `Expression` | 'math.cbrt(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_cbrt("my_field")
print(expr)
math.cbrt(my_field)
```

#### `math_sin(expr)`

“math.sin(...)” expression. Returns the sine of expr (argument in radians).

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description       |
| ----- | ------------ | ----------------- |
| `str` | `Expression` | 'math.sin(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_sin("my_field")
print(expr)
math.sin(my_field)
```

#### `math_cos(expr)`

“math.cos(...)” expression. Returns the cosine of expr (argument in radians).

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description       |
| ----- | ------------ | ----------------- |
| `str` | `Expression` | 'math.cos(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_cos("my_field")
print(expr)
math.cos(my_field)
```

#### `math_tan(expr)`

“math.tan(...)” expression. Returns the tangent of expr (argument in radians).

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description       |
| ----- | ------------ | ----------------- |
| `str` | `Expression` | 'math.tan(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_tan("my_field")
print(expr)
math.tan(my_field)
```

#### `math_asin(expr)`

“math.asin(...)” expression. Returns the arcsine of expr (in radians).

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description        |
| ----- | ------------ | ------------------ |
| `str` | `Expression` | 'math.asin(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_asin("my_field")
print(expr)
math.asin(my_field)
```

#### `math_acos(expr)`

“math.acos(...)” expression. Returns the arccosine of expr (in radians).

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description        |
| ----- | ------------ | ------------------ |
| `str` | `Expression` | 'math.acos(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_acos("my_field")
print(expr)
math.acos(my_field)
```

#### `math_atan(expr)`

“math.atan(...)” expression. Returns the arctangent of expr (in radians).

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description        |
| ----- | ------------ | ------------------ |
| `str` | `Expression` | 'math.atan(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_atan("my_field")
print(expr)
math.atan(my_field)
```

#### `math_sinh(expr)`

“math.sinh(...)” expression. Returns the hyperbolic sine of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description        |
| ----- | ------------ | ------------------ |
| `str` | `Expression` | 'math.sinh(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_sinh("my_field")
print(expr)
math.sinh(my_field)
```

#### `math_cosh(expr)`

“math.cosh(...)” expression. Returns the hyperbolic cosine of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description        |
| ----- | ------------ | ------------------ |
| `str` | `Expression` | 'math.cosh(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_cosh("my_field")
print(expr)
math.cosh(my_field)
```

#### `math_tanh(expr)`

“math.tanh(...)” expression. Returns the hyperbolic tangent of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description        |
| ----- | ------------ | ------------------ |
| `str` | `Expression` | 'math.tanh(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_tanh("my_field")
print(expr)
math.tanh(my_field)
```

#### `math_asinh(expr)`

“math.asinh(...)” expression. Returns the inverse hyperbolic sine of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description         |
| ----- | ------------ | ------------------- |
| `str` | `Expression` | 'math.asinh(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_asinh("my_field")
print(expr)
math.asinh(my_field)
```

#### `math_acosh(expr)`

“math.acosh(...)” expression. Returns the inverse hyperbolic cosine of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description         |
| ----- | ------------ | ------------------- |
| `str` | `Expression` | 'math.acosh(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_acosh("my_field")
print(expr)
math.acosh(my_field)
```

#### `math_atanh(expr)`

“math.atanh(...)” expression. Returns the inverse hyperbolic tangent of expr.

Parameters:

| Name   | Type  | Description              | Default    |
| ------ | ----- | ------------------------ | ---------- |
| `expr` | `str` | The expression or field. | *required* |

Returns:

| Name  | Type         | Description         |
| ----- | ------------ | ------------------- |
| `str` | `Expression` | 'math.atanh(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_atanh("my_field")
print(expr)
math.atanh(my_field)
```

#### `math_pow(expr_x, expr_y)`

“math.pow(...)” expression. Returns expr_x^expr_y.

Parameters:

| Name     | Type  | Description                               | Default    |
| -------- | ----- | ----------------------------------------- | ---------- |
| `expr_x` | `str` | The expression or field for the base.     | *required* |
| `expr_y` | `str` | The expression or field for the exponent. | *required* |

Returns:

| Name  | Type         | Description                 |
| ----- | ------------ | --------------------------- |
| `str` | `Expression` | 'math.pow(expr_x, expr_y)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_pow("my_field", "2")
print(expr)
math.pow(my_field,2)
```

#### `math_hypot(expr_x, expr_y)`

“math.hypot(...)” expression. Returns the length of the hypotenuse given expr_x and expr_y.

Parameters:

| Name     | Type  | Description                                                  | Default    |
| -------- | ----- | ------------------------------------------------------------ | ---------- |
| `expr_x` | `str` | The expression or field for the first side of the triangle.  | *required* |
| `expr_y` | `str` | The expression or field for the second side of the triangle. | *required* |

Returns:

| Name  | Type         | Description                   |
| ----- | ------------ | ----------------------------- |
| `str` | `Expression` | 'math.hypot(expr_x, expr_y)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.math_hypot("my_field_x", "my_field_y")
print(expr)
math.hypot(my_field_x, my_field_y)
```

#### `size(expr)`

“size(...)” expression. Returns the number of elements if expr is a list; otherwise returns 1.

Parameters:

| Name   | Type  | Description                   | Default    |
| ------ | ----- | ----------------------------- | ---------- |
| `expr` | `str` | The list expression or field. | *required* |

Returns:

| Name  | Type         | Description                                         |
| ----- | ------------ | --------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'size(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.size("my_array")
print(expr)
size(my_array)
```

#### `sort(expr)`

“sort(...)” expression. Sorts the elements of the list argument in ascending order.

Parameters:

| Name   | Type  | Description                           | Default    |
| ------ | ----- | ------------------------------------- | ---------- |
| `expr` | `str` | The list expression or field to sort. | *required* |

Returns:

| Name  | Type         | Description                                         |
| ----- | ------------ | --------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'sort(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.sort("my_array")
print(expr)
sort(my_array)
```

#### `reverse(expr)`

“reverse(...)” expression. Reverses the elements of the list argument.

Parameters:

| Name   | Type  | Description                              | Default    |
| ------ | ----- | ---------------------------------------- | ---------- |
| `expr` | `str` | The list expression or field to reverse. | *required* |

Returns:

| Name  | Type         | Description                                            |
| ----- | ------------ | ------------------------------------------------------ |
| `str` | `Expression` | A Vespa expression string of the form 'reverse(expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.reverse("my_array")
print(expr)
reverse(my_array)
```

#### `fixedwidth(value, bucket_width)`

“fixedwidth(...)” bucket expression. Maps the value of the first argument into consecutive buckets whose width is the second argument.

Parameters:

| Name           | Type                | Description                        | Default    |
| -------------- | ------------------- | ---------------------------------- | ---------- |
| `value`        | `str`               | The field or expression to bucket. | *required* |
| `bucket_width` | `Union[int, float]` | The width of each bucket.          | *required* |

Returns:

| Name  | Type         | Description                                                              |
| ----- | ------------ | ------------------------------------------------------------------------ |
| `str` | `Expression` | A Vespa expression string of the form 'fixedwidth(value, bucket_width)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.fixedwidth("my_field",10)
print(expr)
fixedwidth(my_field,10)
```

#### `predefined(value, buckets)`

“predefined(...)” bucket expression. Maps the value into the provided list of buckets.

Each 'bucket' must be a string representing the range, e.g.: 'bucket(-inf,0)', 'bucket\[0,10)', 'bucket\[10,inf)', etc.

Parameters:

| Name      | Type        | Description                        | Default    |
| --------- | ----------- | ---------------------------------- | ---------- |
| `value`   | `str`       | The field or expression to bucket. | *required* |
| `buckets` | `List[str]` | A list of bucket definitions.      | *required* |

Returns:

| Name  | Type         | Description                                                     |
| ----- | ------------ | --------------------------------------------------------------- |
| `str` | `Expression` | A Vespa expression string of the form 'predefined(value, ( ))'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.predefined("my_field", ["bucket(-inf,0)", "bucket[0,10)", "bucket[10,inf)"])
print(expr)
predefined(my_field,bucket(-inf,0),bucket[0,10),bucket[10,inf))
```

#### `interpolatedlookup(array_attr, lookup_expr)`

“interpolatedlookup(...)” expression. Counts elements in a sorted array that are less than an expression, with linear interpolation if the expression is between element values.

Parameters:

| Name          | Type  | Description                        | Default    |
| ------------- | ----- | ---------------------------------- | ---------- |
| `array_attr`  | `str` | The sorted array field name.       | *required* |
| `lookup_expr` | `str` | The expression or value to lookup. | *required* |

Returns:

| Name  | Type         | Description                                                              |
| ----- | ------------ | ------------------------------------------------------------------------ |
| `str` | `Expression` | A Vespa expression string of the form 'interpolatedlookup(array, expr)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.interpolatedlookup("my_sorted_array", "4.2")
print(expr)
interpolatedlookup(my_sorted_array, 4.2)
```

#### `summary(summary_class='')`

“summary(...)” hit aggregator. Produces a summary of the requested summary class.

If no summary class is specified, “summary()” is used.

Parameters:

| Name            | Type  | Description                                | Default |
| --------------- | ----- | ------------------------------------------ | ------- |
| `summary_class` | `str` | Name of the summary class. Defaults to "". | `''`    |

Returns:

| Name  | Type         | Description                                                    |
| ----- | ------------ | -------------------------------------------------------------- |
| `str` | `Expression` | A Vespa grouping expression string of the form 'summary(...)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.summary()
print(expr)
summary()
```

```python
expr = G.summary("my_summary_class")
print(expr)
summary(my_summary_class)
```

#### `as_(expression, label)`

Appends an ' as(label)' part to a grouping block expression.

Parameters:

| Name         | Type  | Description                   | Default    |
| ------------ | ----- | ----------------------------- | ---------- |
| `expression` | `str` | The expression to be labeled. | *required* |
| `label`      | `str` | The label to be used.         | *required* |

Returns:

| Name  | Type  | Description                                                           |
| ----- | ----- | --------------------------------------------------------------------- |
| `str` | `str` | A Vespa grouping expression string of the form 'expression as(label)' |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.as_(G.each(G.output(G.count())), "mylabel")
print(expr)
each(output(count())) as(mylabel)
```

#### `alias(alias_name, expression)`

Defines an alias(...) grouping syntax. This lets you name an expression, so you can reference it later by $alias_name.

Parameters:

| Name         | Type  | Description              | Default    |
| ------------ | ----- | ------------------------ | ---------- |
| `alias_name` | `str` | The alias name.          | *required* |
| `expression` | `str` | The expression to alias. | *required* |

Returns:

| Name  | Type  | Description                                                                     |
| ----- | ----- | ------------------------------------------------------------------------------- |
| `str` | `str` | A Vespa grouping expression string of the form 'alias(alias_name, expression)'. |

Example

```python
from vespa.querybuilder import Grouping as G

expr = G.alias("my_alias", G.add("fieldA", "fieldB"))
print(expr)
alias(my_alias,add(fieldA, fieldB))
```
