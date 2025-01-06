from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Union, Optional, Dict


@dataclass
class QueryField:
    name: str

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

    def contains(
        self, value: Any, annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        value_str = self._format_value(value)
        if annotations:
            annotations_str = ",".join(
                f"{k}:{self._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            return Condition(f"{self.name} contains({{{annotations_str}}}{value_str})")
        else:
            return Condition(f"{self.name} contains {value_str}")

    def matches(
        self, value: Any, annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        value_str = self._format_value(value)
        if annotations:
            annotations_str = ",".join(
                f"{k}:{self._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            return Condition(f"{self.name} matches({{{annotations_str}}}{value_str})")
        else:
            return Condition(f"{self.name} matches {value_str}")

    def in_(self, *values) -> Condition:
        values_str = ", ".join(
            f'"{v}"' if isinstance(v, str) else str(v) for v in values
        )
        return Condition(f"{self.name} in ({values_str})")

    def in_range(
        self, start: Any, end: Any, annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        if annotations:
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


@dataclass
class Condition:
    expression: str

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

    def from_(self, *sources: Union[str, QueryField]) -> Query:
        self.sources = ", ".join(str(source) for source in sources)
        return self

    def where(self, condition: Union[Condition, bool]) -> Query:
        if isinstance(condition, QueryField):
            self.condition = condition
        elif isinstance(condition, bool):
            self.condition = Condition("true") if condition else Condition("false")
        else:
            self.condition = condition
        return self

    def order_by_field(
        self,
        field: str,
        ascending: bool = True,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Query:
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
        return self.order_by_field(field, True, annotations)

    def orderByDesc(
        self, field: str, annotations: Optional[Dict[str, Any]] = None
    ) -> Query:
        return self.order_by_field(field, False, annotations)

    def set_limit(self, limit: int) -> Query:
        self.limit_value = limit
        return self

    def set_offset(self, offset: int) -> Query:
        self.offset_value = offset
        return self

    def set_timeout(self, timeout: int) -> Query:
        self.timeout_value = timeout
        return self

    def add_parameter(self, key: str, value: Any) -> Query:
        self.parameters[key] = value
        return self

    def param(self, key: str, value: Any) -> Query:
        return self.add_parameter(key, value)

    def groupby(self, group_expression: str) -> Query:
        self.grouping = group_expression
        return self

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
    @staticmethod
    def select(fields):
        return Query(select_fields=fields)

    @staticmethod
    def any(*conditions):
        return Condition.any(*conditions)

    @staticmethod
    def all(*conditions):
        return Condition.all(*conditions)

    @staticmethod
    def userQuery(value: str = "") -> Condition:
        return Condition(f'userQuery("{value}")') if value else Condition("userQuery()")

    @staticmethod
    def dotProduct(
        field: str, vector: Dict[str, int], annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        vector_str = "{" + ",".join(f'"{k}":{v}' for k, v in vector.items()) + "}"
        expr = f"dotProduct({field}, {vector_str})"
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def weightedSet(
        field: str, vector: Dict[str, int], annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        vector_str = "{" + ",".join(f'"{k}":{v}' for k, v in vector.items()) + "}"
        expr = f"weightedSet({field}, {vector_str})"
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def nonEmpty(condition: Union[Condition, QueryField]) -> Condition:
        if isinstance(condition, QueryField):
            expr = str(condition)
        else:
            expr = condition.build()
        return Condition(f"nonEmpty({expr})")

    @staticmethod
    def wand(
        field: str, weights, annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        if isinstance(weights, list):
            weights_str = "[" + ", ".join(str(item) for item in weights) + "]"
        elif isinstance(weights, dict):
            weights_str = (
                "{" + ", ".join(f'"{k}":{v}' for k, v in weights.items()) + "}"
            )
        else:
            raise ValueError("Invalid weights for wand")
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
        field: str, query_vector: str, annotations: Dict[str, Any] = {"targetHits": 10}
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
            'select id, text from m where ({targetHits:10}nearestNeighbor(dense_rep, q_dense))'
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
        queries_str = ", ".join(query.build() for query in queries)
        return Condition(f"rank({queries_str})")

    @staticmethod
    def phrase(*terms, annotations: Optional[Dict[str, Any]] = None) -> Condition:
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
        *terms, annotations: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Condition:
        terms_str = ", ".join(f'"{term}"' for term in terms)
        expr = f"near({terms_str})"
        # if kwargs - add to annotations
        if kwargs:
            if not annotations:
                annotations = {}
            annotations.update(kwargs)
        if annotations:
            annotations_str = ", ".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def onear(
        *terms, annotations: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Condition:
        terms_str = ", ".join(f'"{term}"' for term in terms)
        expr = f"onear({terms_str})"
        # if kwargs - add to annotations
        if kwargs:
            if not annotations:
                annotations = {}
            annotations.update(kwargs)
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def sameElement(*conditions) -> Condition:
        conditions_str = ", ".join(cond.build() for cond in conditions)
        expr = f"sameElement({conditions_str})"
        return Condition(expr)

    @staticmethod
    def equiv(*terms) -> Condition:
        terms_str = ", ".join(f'"{term}"' for term in terms)
        expr = f"equiv({terms_str})"
        return Condition(expr)

    @staticmethod
    def uri(value: str, annotations: Optional[Dict[str, Any]] = None) -> Condition:
        expr = f'uri("{value}")'
        if annotations:
            annotations_str = ",".join(
                f"{k}:{QueryField._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def fuzzy(value: str, annotations: Optional[Dict[str, Any]] = None) -> Condition:
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
        return Condition("true")

    @staticmethod
    def false() -> Condition:
        return Condition("false")
