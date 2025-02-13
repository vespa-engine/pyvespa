from __future__ import annotations
from typing import Any, List, Union, Optional, Dict
from vespa.package import Schema
import json


class QueryField:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: Any) -> Condition:  # type: ignore[override]
        return Condition(f"{self.name} = {self._format_value(other)}")

    def __ne__(self, other: Any) -> Condition:  # type: ignore[override]
        return Condition(f"{self.name} != {self._format_value(other)}")

    def __lt__(self, other: Any) -> Condition:
        return Condition(f"{self.name} < {self._format_value(other)}")

    def __le__(self, other: Any) -> Condition:
        return Condition(f"{self.name} <= {self._format_value(other)}")

    def __gt__(self, other: Any) -> Condition:
        return Condition(f"{self.name} > {self._format_value(other)}")

    def __ge__(self, other: Any) -> Condition:
        return Condition(f"{self.name} >= {self._format_value(other)}")

    def __and__(self, other: Any) -> Condition:
        return Condition(f"{self.name} and {self._format_value(other)}")

    def __or__(self, other: Any) -> Condition:
        return Condition(f"{self.name} or {self._format_value(other)}")

    # repr as str
    def __repr__(self) -> str:
        return self.name

    def _build_annotated_expression(
        self,
        operation: str,
        value_str: str,
        annotations: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Helper method to build annotated expressions.

        Args:
            operation: The operation name (e.g. 'contains', 'matches')
            value_str: The formatted value string
            annotations: Optional annotations dictionary
            **kwargs: Additional keyword arguments to merge with annotations
        """
        if kwargs:
            annotations = annotations or {}
            annotations.update(kwargs)

        if annotations:
            annotations_str = ",".join(
                f"{k}:{self._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            return f"{self.name} {operation}({{{annotations_str}}}{value_str})"
        return f"{self.name} {operation} {value_str}"

    def contains(
        self, value: Any, annotations: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Condition:
        value_str = self._format_value(value)
        expr = self._build_annotated_expression(
            "contains", value_str, annotations, **kwargs
        )
        return Condition(expr)

    def matches(
        self, value: Any, annotations: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Condition:
        value_str = self._format_value(value)
        expr = self._build_annotated_expression(
            "matches", value_str, annotations, **kwargs
        )
        return Condition(expr)

    def in_(self, *values) -> Condition:
        values_str = ", ".join(
            f'"{v}"' if isinstance(v, str) else str(v) for v in values
        )
        return Condition(f"{self.name} in ({values_str})")

    def in_range(
        self,
        start: Any,
        end: Any,
        annotations: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Condition:
        if annotations:
            if kwargs:
                annotations.update(kwargs)
            annotations_str = ",".join(
                f"{k}:{self._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            return Condition(
                f"({{{annotations_str}}}range({self.name}, {start}, {end}))"
            )
        else:
            return Condition(f"range({self.name}, {start}, {end})")

    def le(self, value: Any) -> Condition:
        return self.__le__(value)

    def lt(self, value: Any) -> Condition:
        return self.__lt__(value)

    def ge(self, value: Any) -> Condition:
        return self.__ge__(value)

    def gt(self, value: Any) -> Condition:
        return self.__gt__(value)

    def eq(self, value: Any) -> Condition:
        return self.__eq__(value)

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            # Wrap strings in double quotes, but not for parameters
            if value.startswith("@"):
                return value
            else:
                return f'"{value}"'
        elif isinstance(value, Condition):
            return value.build()
        else:
            return str(value)

    @staticmethod
    def _format_annotation_value(value: Any) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, dict):
            return (
                "{"
                + ",".join(
                    f'"{k}":{QueryField._format_annotation_value(v)}'
                    for k, v in value.items()
                )
                + "}"
            )
        elif isinstance(value, list):
            return (
                "["
                + ",".join(f"{QueryField._format_annotation_value(v)}" for v in value)
                + "]"
            )
        else:
            return str(value)

    def annotate(self, annotations: Dict[str, Any]) -> Condition:
        annotations_str = ",".join(
            f"{k}:{self._format_annotation_value(v)}" for k, v in annotations.items()
        )
        return Condition(f"({{{annotations_str}}}){self.name}")


class Condition:
    def __init__(self, expression: str):
        self.expression = expression

    def __and__(self, other: Condition) -> Condition:
        left = self.expression
        right = other.expression

        # Adjust parentheses based on operator precedence
        left = f"({left})" if " or " in left else left
        right = f"({right})" if " or " in right else right

        return Condition(f"{left} and {right}")

    def __or__(self, other: Condition) -> Condition:
        left = self.expression
        right = other.expression

        # Always add parentheses if 'and' or 'or' is in the expressions
        left = f"({left})" if " and " in left or " or " in left else left
        right = f"({right})" if " and " in right or " or " in right else right

        return Condition(f"{left} or {right}")

    def __invert__(self) -> Condition:
        return Condition(f"!({self.expression})")

    def annotate(self, annotations: Dict[str, Any]) -> Condition:
        annotations_str = ",".join(
            f"{k}:{QueryField._format_annotation_value(v)}"
            for k, v in annotations.items()
        )
        return Condition(f"{{{annotations_str}}}{self.expression}")

    def build(self) -> str:
        return self.expression

    @classmethod
    def all(cls, *conditions: Condition) -> Condition:
        """Combine multiple conditions using logical AND."""
        expressions = []
        for cond in conditions:
            expr = cond.expression
            # Wrap expressions with 'or' in parentheses
            if " or " in expr:
                expr = f"({expr})"
            expressions.append(expr)
        combined_expression = " and ".join(expressions)
        return Condition(combined_expression)

    @classmethod
    def any(cls, *conditions: Condition) -> Condition:
        """Combine multiple conditions using logical OR."""
        expressions = []
        for cond in conditions:
            expr = cond.expression
            # Wrap expressions with 'and' or 'or' in parentheses
            if " and " in expr or " or " in expr:
                expr = f"({expr})"
            expressions.append(expr)
        combined_expression = " or ".join(expressions)
        return Condition(combined_expression)


class Query:
    def __init__(
        self, select_fields: Union[str, List[str], List[QueryField]], prepend_yql=False
    ):
        if isinstance(select_fields, str):
            self.select_fields = select_fields
        else:
            # Convert all elements to strings before joining
            self.select_fields = ", ".join(str(field) for field in select_fields)

        self.sources = "*"
        self.order_by_clauses: List[str] = []
        self.parameters: Dict[str, Any] = {}
        self.field: Optional[QueryField] = None
        self.condition: Optional[Condition] = None
        self.where_condition: Optional[Condition] = None
        self.offset_value: Optional[int] = None
        self.limit_value: Optional[int] = None
        self.timeout_value: Optional[int] = None
        self.grouping: Optional[str] = None
        self.prepend_yql = prepend_yql

    def __str__(self) -> str:
        return self.build(self.prepend_yql)

    def __eq__(self, other: Any) -> bool:
        return self.build() == other

    def __ne__(self, other: Any) -> bool:
        return self.build() != other

    def __repr__(self) -> str:
        return str(self)

    def from_(self, *sources: Union[str, Schema]) -> Query:
        """Specify the source schema(s) to query.

        Example:
            >>> import vespa.querybuilder as qb
            >>> from vespa.package import Schema, Document
            >>> query = qb.select("*").from_("schema1", "schema2")
            >>> str(query)
            'select * from schema1, schema2'
            >>> query = qb.select("*").from_(Schema(name="schema1", document=Document()), Schema(name="schema2", document=Document()))
            >>> str(query)
            'select * from schema1, schema2'

        Args:
            sources: The source schema(s) to query.

        Returns:
            Query: The Query object.
        """
        # Convert all elements to string (schema.name) if not already str before joining
        self.sources = ", ".join(
            str(schema.name) if isinstance(schema, Schema) else str(schema)
            for schema in sources
        )
        return self

    def where(self, condition: Union[Condition, bool]) -> Query:
        """Adds a where clause to filter query results.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#where

        Args:
            condition: Filter condition that can be:
                - Condition object for complex queries
                - Boolean for simple true/false
                - QueryField for field-based filters

        Returns:
            Query: Self for method chaining

        Examples:
            >>> import vespa.querybuilder as qb
            >>> # Using field conditions
            >>> f1 = qb.QueryField("f1")
            >>> query = qb.select("*").from_("sd1").where(f1.contains("v1"))
            >>> str(query)
            'select * from sd1 where f1 contains "v1"'

            >>> # Using boolean
            >>> query = qb.select("*").from_("sd1").where(True)
            >>> str(query)
            'select * from sd1 where true'

            >>> # Using complex conditions
            >>> condition = f1.contains("v1") & qb.QueryField("f2").contains("v2")
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where f1 contains "v1" and f2 contains "v2"'
        """
        if isinstance(condition, QueryField):
            self.condition = condition
        elif isinstance(condition, bool):
            self.condition = Condition("true") if condition else Condition("false")
        else:
            self.condition = condition
        return self

    def order_by(
        self,
        field: str,
        ascending: bool = True,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Query:
        """Orders results by specified fields.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#order-by

        Args:
            fields: Field names or QueryField objects to order by
            annotations: Optional annotations like "locale", "strength", etc. See https://docs.vespa.ai/en/reference/sorting.html#special-sorting-attributes for details.

        Returns:
            Query: Self for method chaining

        Examples:
            >>> import vespa.querybuilder as qb
            >>> # Simple ordering
            >>> query = qb.select("*").from_("sd1").order_by("price")
            >>> str(query)
            'select * from sd1 order by price asc'

            >>> # Multiple fields with annotation
            >>> query = qb.select("*").from_("sd1").order_by(
            ...     "price", annotations={"locale": "en_US"}, ascending=False
            ... ).order_by("name", annotations={"locale": "no_NO"}, ascending=True)
            >>> str(query)
            'select * from sd1 order by {"locale":"en_US"}price desc, {"locale":"no_NO"}name asc'
        """
        direction = "asc" if ascending else "desc"
        if annotations:
            annotations_str = ",".join(
                f'"{k}":{QueryField._format_annotation_value(v)}'
                for k, v in annotations.items()
            )
            self.order_by_clauses.append(f"{{{annotations_str}}}{field} {direction}")
        else:
            self.order_by_clauses.append(f"{field} {direction}")
        return self

    def orderByAsc(
        self, field: str, annotations: Optional[Dict[str, Any]] = None
    ) -> Query:
        """Convenience method for ordering results by a field in ascending order.
        See `order_by` for more information.
        """
        return self.order_by(field, True, annotations)

    def orderByDesc(
        self, field: str, annotations: Optional[Dict[str, Any]] = None
    ) -> Query:
        """Convenience method for ordering results by a field in descending order.
        See `order_by` for more information.
        """
        return self.order_by(field, False, annotations)

    def set_limit(self, limit: int) -> Query:
        """Sets maximum number of results to return.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#limit-offset

        Args:
            limit (int): Maximum number of hits to return

        Returns:
            Query: Self for method chaining

        Examples:
            >>> import vespa.querybuilder as qb
            >>> f1 = qb.QueryField("f1")
            >>> query = qb.select("*").from_("sd1").where(f1.contains("v1")).set_limit(5)
            >>> str(query)
            'select * from sd1 where f1 contains "v1" limit 5'
        """
        self.limit_value = limit
        return self

    def set_offset(self, offset: int) -> Query:
        """Sets number of initial results to skip for pagination.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#limit-offset

        Args:
            offset (int): Number of results to skip

        Returns:
            Query: Self for method chaining

        Examples:
            >>> import vespa.querybuilder as qb
            >>> f1 = qb.QueryField("f1")
            >>> query = qb.select("*").from_("sd1").where(f1.contains("v1")).set_offset(10)
            >>> str(query)
            'select * from sd1 where f1 contains "v1" offset 10'
        """
        self.offset_value = offset
        return self

    def set_timeout(self, timeout: int) -> Query:
        """Sets query timeout in milliseconds.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#timeout

        Args:
            timeout (int): Timeout in milliseconds

        Returns:
            Query: Self for method chaining

        Examples:
            >>> import vespa.querybuilder as qb
            >>> f1 = qb.QueryField("f1")
            >>> query = qb.select("*").from_("sd1").where(f1.contains("v1")).set_timeout(500)
            >>> str(query)
            'select * from sd1 where f1 contains "v1" timeout 500'
        """
        self.timeout_value = timeout
        return self

    def add_parameter(self, key: str, value: Any) -> Query:
        """Adds a query parameter.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#parameter-substitution

        Args:
            key (str): Parameter name
            value (Any): Parameter value

        Returns:
            Query: Self for method chaining

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.userInput("@myvar")
            >>> query = qb.select("*").from_("sd1").where(condition).add_parameter("myvar", "test")
            >>> str(query)
            'select * from sd1 where userInput(@myvar)&myvar=test'
        """
        self.parameters[key] = value
        return self

    def param(self, key: str, value: Any) -> Query:
        """Alias for add_parameter().

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#parameter-substitution

        Args:
            key (str): Parameter name
            value (Any): Parameter value

        Returns:
            Query: Self for method chaining

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.userInput("@animal")
            >>> query = qb.select("*").from_("sd1").where(condition).param("animal", "panda")
            >>> str(query)
            'select * from sd1 where userInput(@animal)&animal=panda'
        """
        return self.add_parameter(key, value)

    def groupby(self, group_expression: str, continuations: List = []) -> Query:
        """Groups results by specified expression.

        For more information, see https://docs.vespa.ai/en/grouping.html

        Also see :class:`vespa.querybuilder.Grouping` for available methods to build group expressions.

        Args:
            - group_expression (str): Grouping expression
            - continuations (List): List of continuation tokens (see https://docs.vespa.ai/en/grouping.html#pagination)

        Returns:
            :class:`vespa.querybuilder.Query`: Self for method chaining

        Examples:
            >>> import vespa.querybuilder as qb
            >>> from vespa.querybuilder import Grouping as G
            >>> # Group by customer with sum of price
            >>> grouping = G.all(
            ...     G.group("customer"),
            ...     G.each(G.output(G.sum("price"))),
            ... )
            >>> str(grouping)
            'all(group(customer) each(output(sum(price))))'
            >>> query = qb.select("*").from_("sd1").groupby(grouping)
            >>> str(query)
            'select * from sd1 | all(group(customer) each(output(sum(price))))'

            >>> # Group by year with count
            >>> grouping = G.all(
            ...     G.group("time.year(a)"),
            ...     G.each(G.output(G.count())),
            ... )
            >>> str(grouping)
            'all(group(time.year(a)) each(output(count())))'
            >>> query = qb.select("*").from_("purchase").where(True).groupby(grouping)
            >>> str(query)
            'select * from purchase where true | all(group(time.year(a)) each(output(count())))'
            >>> # With continuations
            >>> query = qb.select("*").from_("purchase").where(True).groupby(grouping, continuations=["foo", "bar"])
            >>> str(query)
            "select * from purchase where true | { 'continuations':['foo', 'bar'] }all(group(time.year(a)) each(output(count())))"
        """
        if continuations:
            cont_str = self._format_continuations(continuations)
        else:
            cont_str = ""
        self.grouping = f"{cont_str}{group_expression}"
        return self

    def _format_continuations(self, continuations: List) -> str:
        """Helper method to format continuations for groupby"""
        cont_str = f"{{ 'continuations':{repr(continuations)} }}"
        return cont_str

    def build(self, prepend_yql=False) -> str:
        query = f"select {self.select_fields} from {self.sources}"
        if prepend_yql:
            query = f"yql={query}"
        if self.condition:
            query += f" where {self.condition.build()}"
        if self.order_by_clauses:
            query += " order by " + ", ".join(self.order_by_clauses)
        if self.limit_value is not None:
            query += f" limit {self.limit_value}"
        if self.offset_value is not None:
            query += f" offset {self.offset_value}"
        if self.timeout_value is not None:
            query += f" timeout {self.timeout_value}"
        if self.grouping:
            query += f" | {self.grouping}"
        if self.parameters:
            params = "&" + "&".join(f"{k}={v}" for k, v in self.parameters.items())
            query += params
        return query


class Q:
    """Wrapper class for QueryBuilder static methods. Methods are exposed as module-level functions.
    To use:

    ```python
    import vespa.querybuilder as qb

    query = qb.select("*").from_("sd1") # or any of the other Q class methods
    ```
    """

    @staticmethod
    def select(fields: Union[str, List[str], List[QueryField]]) -> Query:
        """Creates a new query selecting specified fields.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#select

        Args:
            fields (Union[str, List[str], List[QueryField]): Field names or QueryField objects to select

        Returns:
            Query: New query object

        Examples:
            >>> import vespa.querybuilder as qb
            >>> query = qb.select("*").from_("sd1")
            >>> str(query)
            'select * from sd1'

            >>> query = qb.select(["title", "url"])
            >>> str(query)
            'select title, url from *'
        """
        return Query(select_fields=fields)

    @staticmethod
    def any(*conditions: Condition) -> Condition:
        """Combines multiple conditions with OR operator.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#or

        Args:
            *conditions (Condition): Variable number of Condition objects to combine with OR

        Returns:
            Condition: Combined condition using OR operators

        Examples:
            >>> import vespa.querybuilder as qb
            >>> f1, f2 = qb.QueryField("f1"), qb.QueryField("f2")
            >>> condition = qb.any(f1 > 10, f2 == "v2")
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where f1 > 10 or f2 = "v2"'
        """
        return Condition.any(*conditions)

    @staticmethod
    def all(*conditions: Condition) -> Condition:
        """Combines multiple conditions with AND operator.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#and

        Args:
            *conditions (Condition): Variable number of Condition objects to combine with AND

        Returns:
            Condition: Combined condition using AND operators

        Examples:
            >>> import vespa.querybuilder as qb
            >>> f1, f2 = qb.QueryField("f1"), qb.QueryField("f2")
            >>> condition = qb.all(f1 > 10, f2 == "v2")
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where f1 > 10 and f2 = "v2"'
        """
        return Condition.all(*conditions)

    @staticmethod
    def userQuery(value: str = "") -> Condition:
        """Creates a userQuery operator for text search.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#userquery

        Args:
            value (str): Optional query string. Default is empty string.

        Returns:
            Condition: A userQuery condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> # Basic userQuery
            >>> condition = qb.userQuery()
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where userQuery()'

            >>> # UserQuery with search terms
            >>> condition = qb.userQuery("search terms")
            >>> query = qb.select("*").from_("documents").where(condition)
            >>> str(query)
            'select * from documents where userQuery("search terms")'
        """
        return Condition(f'userQuery("{value}")') if value else Condition("userQuery()")

    @staticmethod
    def dotProduct(
        field: str,
        weights: Union[List[float], Dict[str, float], str],
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Condition:
        """Creates a dot product calculation condition.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#dotproduct.

        Args:
            field (str): Field containing vectors
            weights (Union[List[float], Dict[str, float], str]):
                Either list of numeric weights or dict mapping elements to weights or a parameter substitution string starting with '@'
            annotations (Optional[Dict]): Optional modifiers like label

        Returns:
            Condition: A dot product calculation condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> # Using dict weights with annotation
            >>> condition = qb.dotProduct(
            ...     "weightedset_field",
            ...     {"feature1": 1, "feature2": 2},
            ...     annotations={"label": "myDotProduct"}
            ... )
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where ({label:"myDotProduct"}dotProduct(weightedset_field, {"feature1": 1, "feature2": 2}))'
            >>> # Using list weights
            >>> condition = qb.dotProduct("weightedset_field", [0.4, 0.6])
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where dotProduct(weightedset_field, [0.4, 0.6])'
            >>> # Using parameter substitution
            >>> condition = qb.dotProduct("weightedset_field", "@myweights")
            >>> query = qb.select("*").from_("sd1").where(condition).add_parameter("myweights", [0.4, 0.6])
            >>> str(query)
            'select * from sd1 where dotProduct(weightedset_field, "@myweights")&myweights=[0.4, 0.6]'
        """
        weights_str = json.dumps(weights)
        expr = f"dotProduct({field}, {weights_str})"
        if annotations:
            annotations_str = ", ".join(
                f"{k}:{json.dumps(v)}" for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def weightedSet(
        field: str,
        weights: Union[List[float], Dict[str, float], str],
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Condition:
        """Creates a weighted set condition.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#weightedset.

        Args:
            field (str): Field containing weighted set data
            weights (Union[List[float], Dict[str, float], str]):
                Either list of numeric weights or dict mapping elements to weights or a parameter substitution string starting with
            annotations (Optional[Dict]): Optional annotations like targetNumHits

        Returns:
            Condition: A weighted set condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> # using map weights
            >>> condition = qb.weightedSet(
            ...     "weightedset_field",
            ...     {"element1": 1, "element2": 2},
            ...     annotations={"targetNumHits": 10}
            ... )
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where ({targetNumHits:10}weightedSet(weightedset_field, {"element1": 1, "element2": 2}))'
            >>> # using list weights
            >>> condition = qb.weightedSet("weightedset_field", [0.4, 0.6])
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where weightedSet(weightedset_field, [0.4, 0.6])'
            >>> # using parameter substitution
            >>> condition = qb.weightedSet("weightedset_field", "@myweights")
            >>> query = qb.select("*").from_("sd1").where(condition).add_parameter("myweights", [0.4, 0.6])
            >>> str(query)
            'select * from sd1 where weightedSet(weightedset_field, "@myweights")&myweights=[0.4, 0.6]'
        """
        weights_str = json.dumps(weights)
        expr = f"weightedSet({field}, {weights_str})"
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def nonEmpty(condition: Union[Condition, QueryField]) -> Condition:
        """Creates a nonEmpty operator to check if a field has content.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#nonempty

        Args:
            condition (Union[Condition, QueryField]): Field or condition to check

        Returns:
            Condition: A nonEmpty condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> field = qb.QueryField("title")
            >>> condition = qb.nonEmpty(field)
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where nonEmpty(title)'
        """
        if isinstance(condition, QueryField):
            expr = str(condition)
        else:
            expr = condition.build()
        return Condition(f"nonEmpty({expr})")

    @staticmethod
    def wand(
        field: str,
        weights: Union[List[float], Dict[str, float], str],
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Condition:
        """Creates a Weighted AND (WAND) operator for efficient top-k retrieval.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#wand

        Args:
            field (str): Field name to search
            weights (Union[List[float], Dict[str, float], str]):
                Either list of numeric weights or dict mapping terms to weights or a parameter substitution string starting with '@'
            annotations (Optional[Dict[str, Any]]): Optional annotations like targetHits

        Returns:
            Condition: A WAND condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> # Using list weights
            >>> condition = qb.wand("description", weights=[0.4, 0.6])
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where wand(description, [0.4, 0.6])'

            >>> # Using dict weights with annotation
            >>> weights = {"hello": 0.3, "world": 0.7}
            >>> condition = qb.wand(
            ...     "title",
            ...     weights,
            ...     annotations={"targetHits": 100}
            ... )
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where ({targetHits: 100}wand(title, {"hello": 0.3, "world": 0.7}))'
        """
        weights_str = json.dumps(weights)
        expr = f"wand({field}, {weights_str})"
        if annotations:
            annotations_str = ", ".join(
                f"{k}: {QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def weakAnd(*conditions, annotations: Optional[Dict[str, Any]] = None) -> Condition:
        """Creates a weakAnd operator for less strict AND matching.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#weakand

        Args:
            *conditions (Condition): Variable number of conditions to combine
            annotations (Optional[Dict[str, Any]]): Optional annotations like targetHits

        Returns:
            Condition: A weakAnd condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> f1, f2 = qb.QueryField("f1"), qb.QueryField("f2")
            >>> condition = qb.weakAnd(f1 == "v1", f2 == "v2")
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where weakAnd(f1 = "v1", f2 = "v2")'

            >>> # With annotation
            >>> condition = qb.weakAnd(
            ...     f1 == "v1",
            ...     f2 == "v2",
            ...     annotations={"targetHits": 100}
            ... )
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where ({"targetHits": 100}weakAnd(f1 = "v1", f2 = "v2"))'
        """
        conditions_str = ", ".join(cond.build() for cond in conditions)
        expr = f"weakAnd({conditions_str})"
        if annotations:
            annotations_str = ",".join(
                f'"{k}": {QueryField._format_annotation_value(v)}'
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def geoLocation(
        field: str,
        lat: float,
        lng: float,
        radius: str,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Condition:
        """Creates a geolocation search condition.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#geolocation.

        Args:
            field (str): Field containing location data
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            radius (str): Search radius (e.g. "10km")
            annotations (Optional[Dict]): Optional settings like targetHits

        Returns:
            Condition: A geolocation search condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.geoLocation(
            ...     "location_field",
            ...     37.7749,
            ...     -122.4194,
            ...     "10km",
            ...     annotations={"targetHits": 100}
            ... )
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where ({targetHits:100}geoLocation(location_field, 37.7749, -122.4194, "10km"))'
        """
        expr = f'geoLocation({field}, {lat}, {lng}, "{radius}")'
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def nearestNeighbor(
        field: str, query_vector: str, annotations: Dict[str, Any] = {"targetHits": 100}
    ) -> Condition:
        """Creates a nearest neighbor search condition.

        See https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor for more information.

        Args:
            field (str): Vector field to search in
            query_vector (str): Query vector to compare against
            annotations (Dict[str, Any]): Optional annotations to modify the behavior. Required annotation: targetHits (default: 10)

        Returns:
            Condition: A nearest neighbor search condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.nearestNeighbor(
            ...     field="dense_rep",
            ...     query_vector="q_dense",
            ... )
            >>> query = qb.select(["id, text"]).from_("m").where(condition)
            >>> str(query)
            'select id, text from m where ({targetHits:100}nearestNeighbor(dense_rep, q_dense))'
        """
        if "targetHits" not in annotations:
            raise ValueError("targetHits annotation is required")
        annotations_str = ",".join(
            f"{k}:{QueryField._format_annotation_value(v)}"
            for k, v in annotations.items()
        )
        return Condition(
            f"({{{annotations_str}}}nearestNeighbor({field}, {query_vector}))"
        )

    @staticmethod
    def rank(*queries) -> Condition:
        """Creates a rank condition for combining multiple queries.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#rank

        Args:
            *queries: Variable number of Query objects to combine

        Returns:
            Condition: A rank condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.rank(
            ...     qb.nearestNeighbor("field", "queryVector"),
            ...     qb.QueryField("a").contains("A"),
            ...     qb.QueryField("b").contains("B"),
            ...     qb.QueryField("c").contains("C"),
            ... )
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where rank(({targetHits:100}nearestNeighbor(field, queryVector)), a contains "A", b contains "B", c contains "C")'
        """
        queries_str = ", ".join(query.build() for query in queries)
        return Condition(f"rank({queries_str})")

    @staticmethod
    def phrase(*terms, annotations: Optional[Dict[str, Any]] = None) -> Condition:
        """Creates a phrase search operator for exact phrase matching.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#phrase

        Args:
            *terms (str): Terms that make up the phrase
            annotations (Optional[Dict[str, Any]]): Optional annotations

        Returns:
            Condition: A phrase condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.phrase("new", "york", "city")
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where phrase("new", "york", "city")'
        """
        terms_str = ", ".join(f'"{term}"' for term in terms)
        expr = f"phrase({terms_str})"
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def near(
        *terms,
        distance: Optional[int] = None,
        annotations: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Condition:
        """Creates a near operator for proximity search.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#near

        Args:
            *terms (str): Terms to search for
            distance (Optional[int]): Maximum word distance between terms. Will default to 2 if not specified.
            annotations (Optional[Dict[str, Any]]): Optional annotations
            **kwargs: Additional annotations

        Returns:
            Condition: A near condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.near("machine", "learning", distance=5)
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where ({distance:5}near("machine", "learning"))'
        """
        terms_str = ", ".join(f'"{term}"' for term in terms)
        expr = f"near({terms_str})"
        if annotations is None:
            annotations = {}
        if kwargs:
            annotations.update(kwargs)
        if distance is not None:
            annotations["distance"] = distance
        if annotations:
            annotations_str = ", ".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def onear(
        *terms,
        distance: Optional[int] = None,
        annotations: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Condition:
        """Creates an ordered near operator for ordered proximity search.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#onear

        Args:
            *terms (str): Terms to search for in order
            distance (Optional[int]): Maximum word distance between terms. Will default to 2 if not specified.
            annotations (Optional[Dict[str, Any]]): Optional annotations

        Returns:
            Condition: An onear condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.onear("deep", "learning", distance=3)
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where ({distance:3}onear("deep", "learning"))'
        """
        terms_str = ", ".join(f'"{term}"' for term in terms)
        expr = f"onear({terms_str})"
        if annotations is None:
            annotations = {}
        if kwargs:
            annotations.update(kwargs)
        if distance is not None:
            annotations["distance"] = distance
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def sameElement(*conditions) -> Condition:
        """Creates a sameElement operator to match conditions in same array element.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#sameelement

        Args:
            *conditions (Condition): Conditions that must match in same element

        Returns:
            Condition: A sameElement condition


        Examples:
            >>> import vespa.querybuilder as qb
            >>> persons = qb.QueryField("persons")
            >>> first_name = qb.QueryField("first_name")
            >>> last_name = qb.QueryField("last_name")
            >>> year_of_birth = qb.QueryField("year_of_birth")
            >>> condition = persons.contains(
            ...     qb.sameElement(
            ...         first_name.contains("Joe"),
            ...         last_name.contains("Smith"),
            ...         year_of_birth < 1940,
            ...     )
            ...  )
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where persons contains sameElement(first_name contains "Joe", last_name contains "Smith", year_of_birth < 1940)'
        """
        conditions_str = ", ".join(cond.build() for cond in conditions)
        expr = f"sameElement({conditions_str})"
        return Condition(expr)

    @staticmethod
    def equiv(*terms) -> Condition:
        """Creates an equiv operator for matching equivalent terms.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#equiv

        Args:
            terms (List[str]): List of equivalent terms
            annotations (Optional[Dict[str, Any]]): Optional annotations

        Returns:
            Condition: An equiv condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> fieldName = qb.QueryField("fieldName")
            >>> condition = fieldName.contains(qb.equiv("Snoop Dogg", "Calvin Broadus"))
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where fieldName contains equiv("Snoop Dogg", "Calvin Broadus")'
        """
        terms_str = ", ".join(f'"{term}"' for term in terms)
        expr = f"equiv({terms_str})"
        return Condition(expr)

    @staticmethod
    def uri(value: str, annotations: Optional[Dict[str, Any]] = None) -> Condition:
        """Creates a uri operator for matching URIs.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#uri

        Args:
            field (str): Field name containing URI
            value (str): URI value to match

        Returns:
            Condition: A uri condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> url = "vespa.ai/foo"
            >>> condition = qb.uri(url)
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where uri("vespa.ai/foo")'

        """
        expr = f'uri("{value}")'
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def fuzzy(
        value: str, annotations: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Condition:
        """Creates a fuzzy operator for approximate string matching.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#fuzzy

        Args:
            term (str): Term to fuzzy match
            annotations (Optional[Dict[str, Any]]): Optional annotations
            **kwargs: Optional parameters like maxEditDistance, prefixLength, etc.

        Returns:
            Condition: A fuzzy condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.fuzzy("parantesis")
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where fuzzy("parantesis")'

            >>> # With annotation
            >>> condition = qb.fuzzy("parantesis", annotations={"prefixLength": 1, "maxEditDistance": 2})
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where ({prefixLength:1,maxEditDistance:2}fuzzy("parantesis"))'

        """

        expr = f'fuzzy("{value}")'
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def userInput(
        value: Optional[str] = None, annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        """Creates a userInput operator for query evaluation.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#userinput.

        Args:
            value (Optional[str]): The input variable name, e.g. "@myvar"
            annotations (Optional[Dict]): Optional annotations to modify the behavior

        Returns:
            Condition: A condition representing the userInput operator

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.userInput("@myvar")
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where userInput(@myvar)'

            >>> # With defaultIndex annotation
            >>> condition = qb.userInput("@myvar").annotate({"defaultIndex": "text"})
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where {defaultIndex:"text"}userInput(@myvar)'

            >>> # With parameter
            >>> condition = qb.userInput("@animal")
            >>> query = qb.select("*").from_("sd1").where(condition).param("animal", "panda")
            >>> str(query)
            'select * from sd1 where userInput(@animal)&animal=panda'
        """
        if value is None:
            expr = "userInput()"
        elif value.startswith("@"):
            expr = f"userInput({value})"
        else:
            expr = f'userInput("{value}")'
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def predicate(
        field: str,
        attributes: Optional[Dict[str, Any]] = None,
        range_attributes: Optional[Dict[str, Any]] = None,
    ) -> Condition:
        """Creates a predicate condition for filtering documents based on specific attributes.

        For more information, see https://docs.vespa.ai/en/reference/query-language-reference.html#predicate.

        Args:
            field (str): The predicate field name
            attributes (Optional[Dict[str, Any]]): Dictionary of attribute key-value pairs
            range_attributes (Optional[Dict[str, Any]]): Dictionary of range attribute key-value pairs

        Returns:
            Condition: A condition representing the predicate operation

        Example:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.predicate(
            ...     "predicate_field",
            ...     attributes={"gender": "Female"},
            ...     range_attributes={"age": "20L"}
            ... )
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where predicate(predicate_field,{"gender":"Female"},{"age":20L})'
        """
        if attributes is None:
            attributes_str = "0"
        else:
            attributes_str = (
                "{" + ",".join(f'"{k}":"{v}"' for k, v in attributes.items()) + "}"
            )
        if range_attributes is None:
            range_attributes_str = "0"
        else:
            range_attributes_str = (
                "{" + ",".join(f'"{k}":{v}' for k, v in range_attributes.items()) + "}"
            )
        expr = f"predicate({field},{attributes_str},{range_attributes_str})"
        return Condition(expr)

    @staticmethod
    def true() -> Condition:
        """Creates a condition that is always true.

        Returns:
            Condition: A true condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.true()
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where true'
        """
        return Condition("true")

    @staticmethod
    def false() -> Condition:
        """Creates a condition that is always false.

        Returns:
            Condition: A false condition

        Examples:
            >>> import vespa.querybuilder as qb
            >>> condition = qb.false()
            >>> query = qb.select("*").from_("sd1").where(condition)
            >>> str(query)
            'select * from sd1 where false'
        """
        return Condition("false")
