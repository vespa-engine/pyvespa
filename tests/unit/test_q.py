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
            f"{k}:{Field._format_annotation_value(v)}" for k, v in annotations.items()
        )
        return Condition(f"({{{annotations_str}}}){self.expression}")

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
            default_index_json = json.dumps(
                {"defaultIndex": index}, separators=(",", ":")
            )
            return Condition(f'({default_index_json})userQuery("{value}")')

    @staticmethod
    def dotProduct(
        field: str, vector: Dict[str, int], annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        vector_str = "{" + ",".join(f'"{k}":{v}' for k, v in vector.items()) + "}"
        expr = f"dotProduct({field}, {vector_str})"
        if annotations:
            annotations_str = ",".join(
                f"{k}:{Field._format_annotation_value(v)}"
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
                f"{k}:{Field._format_annotation_value(v)}"
                for k, v in annotations.items()
            )
            expr = f"({{{annotations_str}}}{expr})"
        return Condition(expr)

    @staticmethod
    def nonEmpty(condition: Union[Condition, Field]) -> Condition:
        if isinstance(condition, Field):
            expr = str(condition)
        else:
            expr = condition.build()
        return Condition(f"nonEmpty({expr})")

    @staticmethod
    def wand(
        field: str, weights, annotations: Optional[Dict[str, Any]] = None
    ) -> Condition:
        if isinstance(weights, list):
            weights_str = "[" + ",".join(str(item) for item in weights) + "]"
        elif isinstance(weights, dict):
            weights_str = "{" + ",".join(f'"{k}":{v}' for k, v in weights.items()) + "}"
        else:
            raise ValueError("Invalid weights for wand")
        expr = f"wand({field}, {weights_str})"
        if annotations:
            annotations_str = ",".join(
                f"{k}:{Field._format_annotation_value(v)}"
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
                f'"{k}":{Field._format_annotation_value(v)}'
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
                f"{k}:{Field._format_annotation_value(v)}"
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
            f"{k}:{Field._format_annotation_value(v)}" for k, v in annotations.items()
        )
        return Condition(
            f"({{{annotations_str}}}nearestNeighbor({field}, {query_vector}))"
        )

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


class QTest(unittest.TestCase):
    def test_dotProduct_with_annotations(self):
        condition = Q.dotProduct(
            "vector_field",
            {"feature1": 1, "feature2": 2},
            annotations={"label": "myDotProduct"},
        )
        q = Query(select_fields="*").from_("vectors").where(condition).build()
        expected = 'yql=select * from vectors where ({label:"myDotProduct"}dotProduct(vector_field, {"feature1":1,"feature2":2}))'
        self.assertEqual(q, expected)

    def test_geoLocation_with_annotations(self):
        condition = Q.geoLocation(
            "location_field",
            37.7749,
            -122.4194,
            "10km",
            annotations={"targetHits": 100},
        )
        q = Query(select_fields="*").from_("places").where(condition).build()
        expected = 'yql=select * from places where ({targetHits:100}geoLocation(location_field, 37.7749, -122.4194, "10km"))'
        self.assertEqual(q, expected)

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
        expected = 'yql=select * from sd1 where userQuery("value1") and ({"defaultIndex":"index"})userQuery("value2")'
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
        self.maxDiff = None
        conditions = [
            Q.nearestNeighbor(
                field="colbert",
                query_vector=f"binary_vector_{i}",
                annotations={"targetHits": 100},
            )
            for i in range(32)
        ]
        # Use Condition.any to combine conditions with OR
        q = (
            Query(select_fields="*")
            .from_("doc")
            .where(condition=Condition.any(*conditions))
            .build()
        )
        expected = "yql=select * from doc where " + " or ".join(
            [
                f"({{targetHits:100}}nearestNeighbor(colbert, binary_vector_{i}))"
                for i in range(32)
            ]
        )
        self.assertEqual(q, expected)

    def test_field_comparison_operators(self):
        f1 = Field("age")
        condition = (f1 > 30) & (f1 <= 50)
        q = Query(select_fields="*").from_("people").where(condition).build()
        expected = "yql=select * from people where age > 30 and age <= 50"
        self.assertEqual(q, expected)

    def test_field_in_range(self):
        f1 = Field("age")
        condition = f1.in_range(18, 65)
        q = Query(select_fields="*").from_("people").where(condition).build()
        expected = "yql=select * from people where range(age, 18, 65)"
        self.assertEqual(q, expected)

    def test_field_annotation(self):
        f1 = Field("title")
        annotations = {"highlight": True}
        annotated_field = f1.annotate(annotations)
        q = Query(select_fields="*").from_("articles").where(annotated_field).build()
        expected = "yql=select * from articles where ({highlight:true})title"
        self.assertEqual(q, expected)

    def test_condition_annotation(self):
        f1 = Field("title")
        condition = f1.contains("Python")
        annotated_condition = condition.annotate({"filter": True})
        q = (
            Query(select_fields="*")
            .from_("articles")
            .where(annotated_condition)
            .build()
        )
        expected = (
            'yql=select * from articles where ({filter:true})title contains "Python"'
        )
        self.assertEqual(q, expected)

    def test_grouping_aggregation(self):
        grouping = G.all(G.group("category"), G.output(G.count()))
        q = Query(select_fields="*").from_("products").group(grouping).build()
        expected = "yql=select * from products | all(group(category) output(count()))"
        self.assertEqual(q, expected)

    def test_add_parameter(self):
        f1 = Field("title")
        condition = f1.contains("Python")
        q = (
            Query(select_fields="*")
            .from_("articles")
            .where(condition)
            .add_parameter("tracelevel", 1)
            .build()
        )
        expected = (
            'yql=select * from articles where title contains "Python"&tracelevel=1'
        )
        self.assertEqual(q, expected)

    def test_custom_ranking_expression(self):
        condition = Q.rank(
            Q.userQuery(), Q.dotProduct("embedding", {"feature1": 1, "feature2": 2})
        )
        q = Query(select_fields="*").from_("documents").where(condition).build()
        expected = 'yql=select * from documents where rank(userQuery(), dotProduct(embedding, {"feature1":1,"feature2":2}))'
        self.assertEqual(q, expected)

    def test_wand(self):
        condition = Q.wand("keywords", {"apple": 10, "banana": 20})
        q = Query(select_fields="*").from_("fruits").where(condition).build()
        expected = (
            'yql=select * from fruits where wand(keywords, {"apple":10,"banana":20})'
        )
        self.assertEqual(q, expected)

    def test_weakand(self):
        condition1 = Field("title").contains("Python")
        condition2 = Field("description").contains("Programming")
        condition = Q.weakAnd(
            condition1, condition2, annotations={"targetNumHits": 100}
        )
        q = Query(select_fields="*").from_("articles").where(condition).build()
        expected = 'yql=select * from articles where ({"targetNumHits":100}weakAnd(title contains "Python", description contains "Programming"))'
        self.assertEqual(q, expected)

    def test_geoLocation(self):
        condition = Q.geoLocation("location_field", 37.7749, -122.4194, "10km")
        q = Query(select_fields="*").from_("places").where(condition).build()
        expected = 'yql=select * from places where geoLocation(location_field, 37.7749, -122.4194, "10km")'
        self.assertEqual(q, expected)

    def test_condition_all_any(self):
        c1 = Field("f1").contains("v1")
        c2 = Field("f2").contains("v2")
        c3 = Field("f3").contains("v3")
        condition = Condition.all(c1, c2, Condition.any(c3, ~c1))
        q = Query(select_fields="*").from_("sd1").where(condition).build()
        expected = 'yql=select * from sd1 where f1 contains "v1" and f2 contains "v2" and (f3 contains "v3" or !(f1 contains "v1"))'
        self.assertEqual(q, expected)

    def test_order_by_with_annotations(self):
        f1 = "relevance"
        f2 = "price"
        annotations = {"strength": 0.5}
        q = (
            Query(select_fields="*")
            .from_("products")
            .orderByDesc(f1, annotations)
            .orderByAsc(f2)
            .build()
        )
        expected = 'yql=select * from products order by {"strength":0.5}relevance desc, price asc'
        self.assertEqual(q, expected)

    def test_field_comparison_methods(self):
        f1 = Field("age")
        condition = f1.ge(18) & f1.lt(30)
        q = Query(select_fields="*").from_("users").where(condition).build()
        expected = "yql=select * from users where age >= 18 and age < 30"
        self.assertEqual(q, expected)

    def test_filter_annotation(self):
        f1 = Field("title")
        condition = f1.contains("Python").annotate({"filter": True})
        q = Query(select_fields="*").from_("articles").where(condition).build()
        expected = (
            'yql=select * from articles where ({filter:true})title contains "Python"'
        )
        self.assertEqual(q, expected)

    def test_nonEmpty(self):
        condition = Q.nonEmpty(Field("comments").eq("any_value"))
        q = Query(select_fields="*").from_("posts").where(condition).build()
        expected = 'yql=select * from posts where nonEmpty(comments = "any_value")'
        self.assertEqual(q, expected)

    def test_dotProduct(self):
        condition = Q.dotProduct("vector_field", {"feature1": 1, "feature2": 2})
        q = Query(select_fields="*").from_("vectors").where(condition).build()
        expected = 'yql=select * from vectors where dotProduct(vector_field, {"feature1":1,"feature2":2})'
        self.assertEqual(q, expected)

    def test_in_range_string_values(self):
        f1 = Field("date")
        condition = f1.in_range("2021-01-01", "2021-12-31")
        q = Query(select_fields="*").from_("events").where(condition).build()
        expected = "yql=select * from events where range(date, 2021-01-01, 2021-12-31)"
        self.assertEqual(q, expected)

    def test_condition_inversion(self):
        f1 = Field("status")
        condition = ~f1.eq("inactive")
        q = Query(select_fields="*").from_("users").where(condition).build()
        expected = 'yql=select * from users where !(status = "inactive")'
        self.assertEqual(q, expected)

    def test_multiple_parameters(self):
        f1 = Field("title")
        condition = f1.contains("Python")
        q = (
            Query(select_fields="*")
            .from_("articles")
            .where(condition)
            .add_parameter("tracelevel", 1)
            .add_parameter("language", "en")
            .build()
        )
        expected = 'yql=select * from articles where title contains "Python"&tracelevel=1&language=en'
        self.assertEqual(q, expected)

    def test_multiple_groupings(self):
        grouping = G.all(
            G.group("category"),
            G.maxRtn(10),
            G.output(G.count()),
            G.each(G.group("subcategory"), G.output(G.summary())),
        )
        q = Query(select_fields="*").from_("products").group(grouping).build()
        expected = "yql=select * from products | all(group(category) max(10) output(count()) each(group(subcategory) output(summary())))"
        self.assertEqual(q, expected)

    def test_default_index_annotation(self):
        condition = Q.userQuery("search terms", index="default_field")
        q = Query(select_fields="*").from_("documents").where(condition).build()
        expected = 'yql=select * from documents where ({"defaultIndex":"default_field"})userQuery("search terms")'
        self.assertEqual(q, expected)

    def test_Q_p_function(self):
        condition = Q.p(
            Field("f1").contains("v1"),
            Field("f2").contains("v2"),
            Field("f3").contains("v3"),
        )
        q = Query(select_fields="*").from_("sd1").where(condition).build()
        expected = 'yql=select * from sd1 where f1 contains "v1" and f2 contains "v2" and f3 contains "v3"'
        self.assertEqual(q, expected)

    def test_rank_multiple_conditions(self):
        condition = Q.rank(
            Q.userQuery(),
            Q.dotProduct("embedding", {"feature1": 1}),
            Q.weightedSet("tags", {"tag1": 2}),
        )
        q = Query(select_fields="*").from_("documents").where(condition).build()
        expected = 'yql=select * from documents where rank(userQuery(), dotProduct(embedding, {"feature1":1}), weightedSet(tags, {"tag1":2}))'
        self.assertEqual(q, expected)

    def test_nonEmpty_with_annotations(self):
        annotated_field = Field("comments").annotate({"filter": True})
        condition = Q.nonEmpty(annotated_field)
        q = Query(select_fields="*").from_("posts").where(condition).build()
        expected = "yql=select * from posts where nonEmpty(({filter:true})comments)"
        self.assertEqual(q, expected)

    def test_weight_annotation(self):
        condition = Field("title").contains("heads", annotations={"weight": 200})
        q = Query(select_fields="*").from_("s1").where(condition).build()
        expected = 'yql=select * from s1 where title contains({weight:200}"heads")'
        self.assertEqual(q, expected)

    def test_nearest_neighbor_annotations(self):
        condition = Q.nearestNeighbor(
            field="dense_rep", query_vector="q_dense", annotations={"targetHits": 10}
        )
        q = Query(select_fields=["id, text"]).from_("m").where(condition).build()
        expected = "yql=select id, text from m where ({targetHits:10}nearestNeighbor(dense_rep, q_dense))"
        self.assertEqual(q, expected)


if __name__ == "__main__":
    unittest.main()
