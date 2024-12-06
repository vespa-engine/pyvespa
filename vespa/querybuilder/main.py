from dataclasses import dataclass
from typing import Any, List, Union, Optional, Dict


@dataclass
class Queryfield:
    name: str

    def __eq__(self, other: Any) -> "Condition":
        return Condition(f"{self.name} = {self._format_value(other)}")

    def __ne__(self, other: Any) -> "Condition":
        return Condition(f"{self.name} != {self._format_value(other)}")

    def __lt__(self, other: Any) -> "Condition":
        return Condition(f"{self.name} < {self._format_value(other)}")

    def __le__(self, other: Any) -> "Condition":
        return Condition(f"{self.name} <= {self._format_value(other)}")

    def __gt__(self, other: Any) -> "Condition":
        return Condition(f"{self.name} > {self._format_value(other)}")

    def __ge__(self, other: Any) -> "Condition":
        return Condition(f"{self.name} >= {self._format_value(other)}")

    def __and__(self, other: Any) -> "Condition":
        return Condition(f"{self.name} and {self._format_value(other)}")

    def __or__(self, other: Any) -> "Condition":
        return Condition(f"{self.name} or {self._format_value(other)}")

    def contains(
        self, value: Any, annotations: Optional[Dict[str, Any]] = None
    ) -> "Condition":
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
    ) -> "Condition":
        value_str = self._format_value(value)
        if annotations:
            annotations_str = ",".join(
                f"{k}:{self._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            return Condition(f"{self.name} matches({{{annotations_str}}}{value_str})")
        else:
            return Condition(f"{self.name} matches {value_str}")

    def in_(self, *values) -> "Condition":
        values_str = ", ".join(
            f'"{v}"' if isinstance(v, str) else str(v) for v in values
        )
        return Condition(f"{self.name} in ({values_str})")

    def in_range(
        self, start: Any, end: Any, annotations: Optional[Dict[str, Any]] = None
    ) -> "Condition":
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

    def le(self, value: Any) -> "Condition":
        return self.__le__(value)

    def lt(self, value: Any) -> "Condition":
        return self.__lt__(value)

    def ge(self, value: Any) -> "Condition":
        return self.__ge__(value)

    def gt(self, value: Any) -> "Condition":
        return self.__gt__(value)

    def eq(self, value: Any) -> "Condition":
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
                    f'"{k}":{Queryfield._format_annotation_value(v)}'
                    for k, v in value.items()
                )
                + "}"
            )
        elif isinstance(value, list):
            return (
                "["
                + ",".join(f"{Queryfield._format_annotation_value(v)}" for v in value)
                + "]"
            )
        else:
            return str(value)

    def annotate(self, annotations: Dict[str, Any]) -> "Condition":
        annotations_str = ",".join(
            f"{k}:{self._format_annotation_value(v)}" for k, v in annotations.items()
        )
        return Condition(f"({{{annotations_str}}}){self.name}")


@dataclass
class Condition:
    expression: str

    def __and__(self, other: "Condition") -> "Condition":
        left = self.expression
        right = other.expression

        # Adjust parentheses based on operator precedence
        left = f"({left})" if " or " in left else left
        right = f"({right})" if " or " in right else right

        return Condition(f"{left} and {right}")

    def __or__(self, other: "Condition") -> "Condition":
        left = self.expression
        right = other.expression

        # Always add parentheses if 'and' or 'or' is in the expressions
        left = f"({left})" if " and " in left or " or " in left else left
        right = f"({right})" if " and " in right or " or " in right else right

        return Condition(f"{left} or {right}")

    def __invert__(self) -> "Condition":
        return Condition(f"!({self.expression})")

    def annotate(self, annotations: Dict[str, Any]) -> "Condition":
        annotations_str = ",".join(
            f"{k}:{Queryfield._format_annotation_value(v)}"
            for k, v in annotations.items()
        )
        return Condition(f"{{{annotations_str}}}{self.expression}")

    def build(self) -> str:
        return self.expression

    @classmethod
    def all(cls, *conditions: "Condition") -> "Condition":
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
    def any(cls, *conditions: "Condition") -> "Condition":
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
        self, select_fields: Union[str, List[str], List[Queryfield]], prepend_yql=False
    ):
        self.select_fields = (
            ", ".join(select_fields)
            if isinstance(select_fields, List)
            and all(isinstance(f, str) for f in select_fields)
            else ", ".join(str(f) for f in select_fields)
        )
        self.sources = "*"
        self.condition = None
        self.order_by_clauses = []
        self.limit_value = None
        self.offset_value = None
        self.timeout_value = None
        self.parameters = {}
        self.grouping = None
        self.prepend_yql = prepend_yql

    def __str__(self) -> str:
        return self.build(self.prepend_yql)

    def __eq__(self, other: Any) -> bool:
        return self.build() == other

    def __ne__(self, other: Any) -> bool:
        return self.build() != other

    def __repr__(self) -> str:
        return str(self)

    def from_(self, *sources: str) -> "Query":
        self.sources = ", ".join(sources)
        return self

    def where(self, condition: Union[Condition, Queryfield, bool]) -> "Query":
        if isinstance(condition, Queryfield):
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
    ) -> "Query":
        direction = "asc" if ascending else "desc"
        if annotations:
            annotations_str = ",".join(
                f'"{k}":{Queryfield._format_annotation_value(v)}'
                for k, v in annotations.items()
            )
            self.order_by_clauses.append(f"{{{annotations_str}}}{field} {direction}")
        else:
            self.order_by_clauses.append(f"{field} {direction}")
        return self

    def orderByAsc(
        self, field: str, annotations: Optional[Dict[str, Any]] = None
    ) -> "Query":
        return self.order_by_field(field, True, annotations)

    def orderByDesc(
        self, field: str, annotations: Optional[Dict[str, Any]] = None
    ) -> "Query":
        return self.order_by_field(field, False, annotations)

    def set_limit(self, limit: int) -> "Query":
        self.limit_value = limit
        return self

    def set_offset(self, offset: int) -> "Query":
        self.offset_value = offset
        return self

    def set_timeout(self, timeout: int) -> "Query":
        self.timeout_value = timeout
        return self

    def add_parameter(self, key: str, value: Any) -> "Query":
        self.parameters[key] = value
        return self

    def param(self, key: str, value: Any) -> "Query":
        return self.add_parameter(key, value)

    def groupby(self, group_expression: str) -> "Query":
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
    def select(*fields):
        return Query(select_fields=list(fields))

    @staticmethod
    def p(*args):
        if not args:
            return Condition("")
        else:
            condition = args[0]
            for arg in args[1:]:
                condition = condition & arg
            return condition

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
                f"{k}:{Queryfield._format_annotation_value(v)}"
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
                f"{k}:{Queryfield._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def nonEmpty(condition: Union[Condition, Queryfield]) -> Condition:
        if isinstance(condition, Queryfield):
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
                f"{k}: {Queryfield._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def weakAnd(*conditions, annotations: Dict[str, Any] = None) -> Condition:
        conditions_str = ", ".join(cond.build() for cond in conditions)
        expr = f"weakAnd({conditions_str})"
        if annotations:
            annotations_str = ",".join(
                f'"{k}": {Queryfield._format_annotation_value(v)}'
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
        expr = f'geoLocation({field}, {lat}, {lng}, "{radius}")'
        if annotations:
            annotations_str = ",".join(
                f"{k}:{Queryfield._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def nearestNeighbor(
        field: str, query_vector: str, annotations: Dict[str, Any]
    ) -> Condition:
        if "targetHits" not in annotations:
            raise ValueError("targetHits annotation is required")
        annotations_str = ",".join(
            f"{k}:{Queryfield._format_annotation_value(v)}"
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
                f"{k}:{Queryfield._format_annotation_value(v)}"
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
                f"{k}:{Queryfield._format_annotation_value(v)}"
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
                f"{k}:{Queryfield._format_annotation_value(v)}"
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
                f"{k}:{Queryfield._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def fuzzy(value: str, annotations: Optional[Dict[str, Any]] = None) -> Condition:
        expr = f'fuzzy("{value}")'
        if annotations:
            annotations_str = ",".join(
                f"{k}:{Queryfield._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def userInput(
        value: Optional[str] = None, annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        if value is None:
            expr = "userInput()"
        elif value.startswith("@"):
            expr = f"userInput({value})"
        else:
            expr = f'userInput("{value}")'
        if annotations:
            annotations_str = ",".join(
                f"{k}:{Queryfield._format_annotation_value(v)}"
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
