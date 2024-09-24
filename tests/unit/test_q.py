# test_querybuilder.py

import unittest
from dataclasses import dataclass
from typing import Any, List, Union, Optional, Dict
import json


@dataclass
class Field:
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

    def contains(self, value: Any) -> "Condition":
        return Condition(f"{self.name} contains {self._format_value(value)}")

    def matches(self, value: Any) -> "Condition":
        return Condition(f"{self.name} matches {self._format_value(value)}")

    def in_range(self, start: Any, end: Any) -> "Condition":
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
        else:
            return str(value)

    def annotate(self, annotations: Dict[str, Any]) -> "Condition":
        annotations_str = ",".join(
            f'"{k}":{self._format_annotation_value(v)}' for k, v in annotations.items()
        )
        return Condition(f"({{{{{annotations_str}}}}})({self.name})")

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
                    f'"{k}":{Field._format_annotation_value(v)}'
                    for k, v in value.items()
                )
                + "}"
            )
        elif isinstance(value, list):
            return (
                "["
                + ",".join(f"{Field._format_annotation_value(v)}" for v in value)
                + "]"
            )
        else:
            return str(value)


@dataclass
class Condition:
    expression: str

    def __and__(self, other: "Condition") -> "Condition":
        left = self.expression
        right = other.expression

        if " and " in left or " or " in left:
            left = f"({left})"
        if " and " in right or " or " in right:
            right = f"({right})"

        return Condition(f"{left} and {right}")

    def __or__(self, other: "Condition") -> "Condition":
        left = self.expression
        right = other.expression

        if " and " in left or " or " in left:
            left = f"({left})"
        if " and " in right or " or " in right:
            right = f"({right})"

        return Condition(f"{left} or {right}")

    def __invert__(self) -> "Condition":
        return Condition(f"!({self.expression})")

    def annotate(self, annotations: Dict[str, Any]) -> "Condition":
        annotations_str = ",".join(
            f'"{k}":{Field._format_annotation_value(v)}' for k, v in annotations.items()
        )
        return Condition(f"([{annotations_str}]({self.expression}))")

    def build(self) -> str:
        return self.expression


class Query:
    def __init__(self, select_fields: Union[str, List[str], List[Field]]):
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

    def from_(self, *sources: str) -> "Query":
        self.sources = ", ".join(sources)
        return self

    def where(self, condition: Union[Condition, Field]) -> "Query":
        if isinstance(condition, Field):
            self.condition = condition
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
                f'"{k}":{Field._format_annotation_value(v)}'
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

    def group(self, group_expression: str) -> "Query":
        self.grouping = group_expression
        return self

    def build(self) -> str:
        query = f"yql=select {self.select_fields} from {self.sources}"
        if self.condition:
            query += f" where {self.condition.build()}"
        if self.grouping:
            query += f" | {self.grouping}"
        if self.order_by_clauses:
            query += " order by " + ", ".join(self.order_by_clauses)
        if self.limit_value is not None:
            query += f" limit {self.limit_value}"
        if self.offset_value is not None:
            query += f" offset {self.offset_value}"
        if self.timeout_value is not None:
            query += f" timeout {self.timeout_value}"
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
    def userQuery(value: str = "", index: Optional[str] = None) -> Condition:
        if index is None:
            # Only value provided
            return (
                Condition(f'userQuery("{value}")')
                if value
                else Condition("userQuery()")
            )
        else:
            # Both index and value provided
            default_index_json = json.dumps({"defaultIndex": index})
            return Condition(f'({default_index_json})userQuery("{value}")')

    @staticmethod
    def dotPdt(field: str, vector: Dict[str, int]) -> Condition:
        vector_str = "{" + ",".join(f'"{k}":{v}' for k, v in vector.items()) + "}"
        return Condition(f"dotProduct({field}, {vector_str})")

    @staticmethod
    def wtdSet(field: str, vector: Dict[str, int]) -> Condition:
        vector_str = "{" + ",".join(f'"{k}":{v}' for k, v in vector.items()) + "}"
        return Condition(f"weightedSet({field}, {vector_str})")

    @staticmethod
    def nonEmpty(condition: Condition) -> Condition:
        return Condition(f"nonEmpty({condition.build()})")

    @staticmethod
    def wand(field: str, weights, annotations: Dict[str, Any] = None) -> Condition:
        if isinstance(weights, list):
            weights_str = "[" + ",".join(str(item) for item in weights) + "]"
        elif isinstance(weights, dict):
            weights_str = "{" + ",".join(f'"{k}":{v}' for k, v in weights.items()) + "}"
        else:
            raise ValueError("Invalid weights for wand")
        expr = f"wand({field}, {weights_str})"
        if annotations:
            annotations_str = ",".join(
                f'"{k}":{Field._format_annotation_value(v)}'
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def weakand(*conditions, annotations: Dict[str, Any] = None) -> Condition:
        conditions_str = ", ".join(cond.build() for cond in conditions)
        expr = f"weakAnd({conditions_str})"
        if annotations:
            annotations_str = ",".join(
                f'"{k}":{Field._format_annotation_value(v)}'
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def geoLocation(field: str, lat: float, lng: float, radius: str) -> Condition:
        return Condition(f'geoLocation({field}, {lat}, {lng}, "{radius}")')

    @staticmethod
    def nearestNeighbor(
        field: str, query_vector: str, annotations: Dict[str, Any] = None
    ) -> Condition:
        if annotations:
            if "targetHits" not in annotations:
                raise ValueError("targetHits annotation is required")
            annotations_str = ",".join(
                f"{k}:{Field._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            return Condition(
                f"({{{annotations_str}}}nearestNeighbor({field}, {query_vector}))"
            )
        else:
            raise ValueError("Annotations are required for nearestNeighbor")

    @staticmethod
    def rank(*queries) -> Condition:
        queries_str = ", ".join(query.build() for query in queries)
        return Condition(f"rank({queries_str})")


class G:
    @staticmethod
    def all(*args) -> str:
        return "all(" + " ".join(args) + ")"

    @staticmethod
    def group(field: str) -> str:
        return f"group({field})"

    @staticmethod
    def maxRtn(value: int) -> str:
        return f"max({value})"

    @staticmethod
    def each(*args) -> str:
        return "each(" + " ".join(args) + ")"

    @staticmethod
    def output(output_func: str) -> str:
        return f"output({output_func})"

    @staticmethod
    def count() -> str:
        return "count()"

    @staticmethod
    def summary() -> str:
        return "summary()"


class A:
    @staticmethod
    def a(*args, **kwargs) -> Dict[str, Any]:
        if args and isinstance(args[0], dict):
            return args[0]
        else:
            annotations = {}
            for i in range(0, len(args), 2):
                annotations[args[i]] = args[i + 1]
            annotations.update(kwargs)
            return annotations

    @staticmethod
    def filter() -> Dict[str, Any]:
        return {"filter": True}

    @staticmethod
    def defaultIndex(index: str) -> Dict[str, Any]:
        return {"defaultIndex": index}

    @staticmethod
    def append(annotations: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        annotations.update(other)
        return annotations


class QTest(unittest.TestCase):
    def test_select_specific_fields(self):
        f1 = Field("f1")
        condition = f1.contains("v1")
        q = Query(select_fields=["f1", "f2"]).from_("sd1").where(condition).build()

        self.assertEqual(q, 'yql=select f1, f2 from sd1 where f1 contains "v1"')

    def test_select_from_specific_sources(self):
        f1 = Field("f1")
        condition = f1.contains("v1")
        q = Query(select_fields="*").from_("sd1").where(condition).build()

        self.assertEqual(q, 'yql=select * from sd1 where f1 contains "v1"')

    def test_select_from_multiples_sources(self):
        f1 = Field("f1")
        condition = f1.contains("v1")
        q = Query(select_fields="*").from_("sd1", "sd2").where(condition).build()

        self.assertEqual(q, 'yql=select * from sd1, sd2 where f1 contains "v1"')

    def test_basic_and_andnot_or_offset_limit_param_order_by_and_contains(self):
        f1 = Field("f1")
        f2 = Field("f2")
        f3 = Field("f3")
        f4 = Field("f4")
        condition = ((f1.contains("v1") & f2.contains("v2")) | f3.contains("v3")) & (
            ~f4.contains("v4")
        )
        q = (
            Query(select_fields="*")
            .from_("sd1")
            .where(condition)
            .set_offset(1)
            .set_limit(2)
            .set_timeout(3)
            .orderByDesc("f1")
            .orderByAsc("f2")
            .param("paramk1", "paramv1")
            .build()
        )

        expected = 'yql=select * from sd1 where ((f1 contains "v1" and f2 contains "v2") or f3 contains "v3") and !(f4 contains "v4") order by f1 desc, f2 asc limit 2 offset 1 timeout 3&paramk1=paramv1'
        self.assertEqual(q, expected)

    def test_matches(self):
        condition = (
            (Field("f1").matches("v1") & Field("f2").matches("v2"))
            | Field("f3").matches("v3")
        ) & ~Field("f4").matches("v4")
        q = Query(select_fields="*").from_("sd1").where(condition).build()
        expected = 'yql=select * from sd1 where ((f1 matches "v1" and f2 matches "v2") or f3 matches "v3") and !(f4 matches "v4")'
        self.assertEqual(q, expected)

    def test_nested_queries(self):
        nested_query = (Field("f2").contains("2") & Field("f3").contains("3")) | (
            Field("f2").contains("4") & ~Field("f3").contains("5")
        )
        condition = Field("f1").contains("1") & ~nested_query
        q = Query(select_fields="*").from_("sd1").where(condition).build()
        expected = 'yql=select * from sd1 where f1 contains "1" and (!((f2 contains "2" and f3 contains "3") or (f2 contains "4" and !(f3 contains "5"))))'
        self.assertEqual(q, expected)

    def test_userInput_with_and_without_defaultIndex(self):
        condition = Q.userQuery(value="value1") & Q.userQuery(
            index="index", value="value2"
        )
        q = Query(select_fields="*").from_("sd1").where(condition).build()
        expected = 'yql=select * from sd1 where userQuery("value1") and ({"defaultIndex": "index"})userQuery("value2")'
        self.assertEqual(q, expected)

    def test_fields_duration(self):
        f1 = Field("subject")
        f2 = Field("display_date")
        f3 = Field("duration")
        condition = (
            Query(select_fields=[f1, f2]).from_("calendar").where(f3 > 0).build()
        )
        expected = "yql=select subject, display_date from calendar where duration > 0"
        self.assertEqual(condition, expected)

    def test_nearest_neighbor(self):
        condition_uq = Q.userQuery()
        condition_nn = Q.nearestNeighbor(
            field="dense_rep", query_vector="q_dense", annotations={"targetHits": 10}
        )
        q = (
            Query(select_fields=["id, text"])
            .from_("m")
            .where(condition_uq | condition_nn)
            .build()
        )
        expected = "yql=select id, text from m where userQuery() or ({targetHits:10}nearestNeighbor(dense_rep, q_dense))"
        self.assertEqual(q, expected)

    def test_build_many_nn_operators(self):
        conditions = [
            Q.nearestNeighbor(
                field="colbert",
                query_vector=f"binary_vector_{i}",
                annotations={"targetHits": 100},
            )
            for i in range(32)
        ]
        q = (
            Query(select_fields="*")
            .from_("doc")
            .where(condition=Q.p(*conditions))
            .build()
        )
        expected = "yql=select * from doc where ({targetHits:100}nearestNeighbor(colbert, binary_vector_0)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_1)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_2)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_3)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_4)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_5)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_6)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_7)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_8)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_9)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_10)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_11)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_12)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_13)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_14)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_15)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_16)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_17)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_18)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_19)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_20)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_21)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_22)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_23)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_24)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_25)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_26)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_27)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_28)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_29)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_30)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_31))"
        self.assertEqual(q, expected)


if __name__ == "__main__":
    unittest.main()
