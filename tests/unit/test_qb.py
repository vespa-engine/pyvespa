import unittest
from vespa.querybuilder import Grouping as G
import vespa.querybuilder as qb
from vespa.package import Schema, Document


class TestQueryBuilder(unittest.TestCase):
    def test_dotproduct_with_annotations(self):
        condition = qb.dotProduct(
            "weightedset_field",
            {"feature1": 1, "feature2": 2},
            annotations={"label": "myDotProduct"},
        )
        q = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where ({label:"myDotProduct"}dotProduct(weightedset_field, {"feature1": 1, "feature2": 2}))'
        self.assertEqual(q, expected)
        return q

    def test_geolocation_with_annotations(self):
        condition = qb.geoLocation(
            "location_field",
            37.7749,
            -122.4194,
            "10km",
            annotations={"targetHits": 100},
        )
        q = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where ({targetHits:100}geoLocation(location_field, 37.7749, -122.4194, "10km"))'
        self.assertEqual(q, expected)
        return q

    def test_select_specific_fields(self):
        f1 = qb.QueryField("f1")
        condition = f1.contains("v1")
        q = qb.select(["f1", "f2"]).from_("sd1").where(condition)
        self.assertEqual(q, 'select f1, f2 from sd1 where f1 contains "v1"')
        return q

    def test_select_from_specific_sources(self):
        f1 = qb.QueryField("f1")
        condition = f1.contains("v1")
        q = qb.select("*").from_("sd1").where(condition)
        self.assertEqual(q, 'select * from sd1 where f1 contains "v1"')
        return q

    def test_select_from_pyvespa_schema(self):
        schema = Schema(name="schema_name", document=Document())
        f1 = qb.QueryField("f1")
        condition = f1.contains("v1")
        q = qb.select("*").from_(schema).where(condition)
        self.assertEqual(q, 'select * from schema_name where f1 contains "v1"')
        return q

    def test_select_from_multiples_sources(self):
        f1 = qb.QueryField("f1")
        condition = f1.contains("v1")
        q = qb.select("*").from_("sd1", "sd2").where(condition)
        self.assertEqual(q, 'select * from sd1, sd2 where f1 contains "v1"')
        return q

    def test_basic_and_andnot_or_offset_limit_param_order_by_and_contains(self):
        f1 = qb.QueryField("f1")
        f2 = qb.QueryField("f2")
        f3 = qb.QueryField("f3")
        f4 = qb.QueryField("f4")
        condition = ((f1.contains("v1") & f2.contains("v2")) | f3.contains("v3")) & (
            ~f4.contains("v4")
        )
        q = (
            qb.select("*")
            .from_("sd1")
            .where(condition)
            .set_offset(1)
            .set_limit(2)
            .set_timeout(3000)
            .orderByDesc("age")
            .orderByAsc("duration")
        )

        expected = 'select * from sd1 where ((f1 contains "v1" and f2 contains "v2") or f3 contains "v3") and !(f4 contains "v4") order by age desc, duration asc limit 2 offset 1 timeout 3000'
        self.assertEqual(q, expected)
        return q

    def test_timeout(self):
        f1 = qb.QueryField("title")
        condition = f1.contains("madonna")
        q = qb.select("*").from_("sd1").where(condition).set_timeout(70)
        expected = 'select * from sd1 where title contains "madonna" timeout 70'
        self.assertEqual(q, expected)
        return q

    def test_matches(self):
        condition = (
            (qb.QueryField("f1").matches("v1") & qb.QueryField("f2").matches("v2"))
            | qb.QueryField("f3").matches("v3")
        ) & ~qb.QueryField("f4").matches("v4")
        q = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where ((f1 matches "v1" and f2 matches "v2") or f3 matches "v3") and !(f4 matches "v4")'
        self.assertEqual(q, expected)
        return q

    def test_matches_with_regex(self):
        condition = qb.QueryField("f1").matches("^TestText$")
        q = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where f1 matches "^TestText$"'
        self.assertEqual(q, expected)
        return q

    def test_nested_queries(self):
        nested_query = (
            qb.QueryField("f2").contains("2") & qb.QueryField("f3").contains("3")
        ) | (qb.QueryField("f2").contains("4") & ~qb.QueryField("f3").contains("5"))
        condition = qb.QueryField("f1").contains("1") & ~nested_query
        q = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where f1 contains "1" and (!((f2 contains "2" and f3 contains "3") or (f2 contains "4" and !(f3 contains "5"))))'
        self.assertEqual(q, expected)
        return q

    def test_userquery(self):
        condition = qb.userQuery()
        q = qb.select("*").from_("sd1").where(condition)
        expected = "select * from sd1 where userQuery()"
        self.assertEqual(q, expected)
        return q

    def test_fields_duration(self):
        f1 = qb.QueryField("subject")
        f2 = qb.QueryField("display_date")
        f3 = qb.QueryField("duration")
        q = qb.select([f1, f2]).from_("calendar").where(f3 > 0)
        expected = "select subject, display_date from calendar where duration > 0"
        self.assertEqual(q, expected)
        return q

    def test_nearest_neighbor(self):
        condition_uq = qb.userQuery()
        condition_nn = qb.nearestNeighbor(
            field="dense_rep", query_vector="q_dense", annotations={"targetHits": 10}
        )
        q = qb.select(["id, text"]).from_("m").where(condition_uq | condition_nn)
        expected = "select id, text from m where userQuery() or ({targetHits:10}nearestNeighbor(dense_rep, q_dense))"
        self.assertEqual(q, expected)
        return q

    def test_build_many_nn_operators(self):
        conditions = [
            qb.nearestNeighbor(
                field="colbert",
                query_vector=f"binary_vector_{i}",
                annotations={"targetHits": 100},
            )
            for i in range(32)
        ]
        # Use Condition.any to combine conditions with OR
        q = qb.select("*").from_("doc").where(condition=qb.any(*conditions))
        expected = "select * from doc where " + " or ".join(
            [
                f"({{targetHits:100}}nearestNeighbor(colbert, binary_vector_{i}))"
                for i in range(32)
            ]
        )
        self.assertEqual(q, expected)
        return q

    def test_field_comparison_operators(self):
        f1 = qb.QueryField("age")
        condition = (f1 > 30) & (f1 <= 50)
        q = qb.select("*").from_("people").where(condition)
        expected = "select * from people where age > 30 and age <= 50"
        self.assertEqual(q, expected)
        return q

    def test_field_in_range(self):
        f1 = qb.QueryField("age")
        condition = f1.in_range(18, 65)
        q = qb.select("*").from_("people").where(condition)
        expected = "select * from people where range(age, 18, 65)"
        self.assertEqual(q, expected)
        return q

    def test_field_annotation(self):
        f1 = qb.QueryField("title")
        annotations = {"highlight": True}
        annotated_field = f1.annotate(annotations)
        q = qb.select("*").from_("articles").where(annotated_field)
        expected = "select * from articles where ({highlight:true})title"
        self.assertEqual(q, expected)
        return q

    def test_condition_annotation(self):
        f1 = qb.QueryField("title")
        condition = f1.contains("Python")
        annotated_condition = condition.annotate({"filter": True})
        q = qb.select("*").from_("articles").where(annotated_condition)
        expected = 'select * from articles where {filter:true}title contains "Python"'
        self.assertEqual(q, expected)
        return q

    def test_add_parameter(self):
        f1 = qb.QueryField("title")
        condition = f1.contains("Python")
        q = (
            qb.select("*")
            .from_("articles")
            .where(condition)
            .add_parameter("tracelevel", 1)
        )
        expected = 'select * from articles where title contains "Python"&tracelevel=1'
        self.assertEqual(q, expected)
        return q

    def test_rank_nn_contains(self):
        condition = qb.rank(
            qb.nearestNeighbor("field", "queryVector"),
            qb.QueryField("a").contains("A"),
            qb.QueryField("b").contains("B"),
            qb.QueryField("c").contains("C"),
        )
        q = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where rank(({targetHits:100}nearestNeighbor(field, queryVector)), a contains "A", b contains "B", c contains "C")'
        self.assertEqual(q, expected)

    def test_custom_ranking_expression(self):
        condition = qb.rank(
            qb.userQuery(), qb.dotProduct("embedding", {"feature1": 1, "feature2": 2})
        )
        q = qb.select("*").from_("documents").where(condition)
        expected = 'select * from documents where rank(userQuery(), dotProduct(embedding, {"feature1": 1, "feature2": 2}))'
        self.assertEqual(q, expected)
        return q

    def test_wand(self):
        condition = qb.wand("keywords", {"apple": 10, "banana": 20})
        q = qb.select("*").from_("fruits").where(condition)
        expected = (
            'select * from fruits where wand(keywords, {"apple": 10, "banana": 20})'
        )
        self.assertEqual(q, expected)
        return q

    def test_wand_numeric(self):
        condition = qb.wand("description", [[11, 1], [37, 2]])
        q = qb.select("*").from_("fruits").where(condition)
        expected = "select * from fruits where wand(description, [[11, 1], [37, 2]])"
        self.assertEqual(q, expected)
        return q

    def test_wand_annotations(self):
        condition = qb.wand(
            "description",
            weights={"a": 1, "b": 2},
            annotations={"scoreThreshold": 0.13, "targetHits": 7},
        )
        q = qb.select("*").from_("fruits").where(condition)
        expected = 'select * from fruits where ({scoreThreshold: 0.13, targetHits: 7}wand(description, {"a": 1, "b": 2}))'
        self.assertEqual(q, expected)
        return q

    def test_weakand(self):
        condition1 = qb.QueryField("title").contains("Python")
        condition2 = qb.QueryField("description").contains("Programming")
        condition = qb.weakAnd(condition1, condition2, annotations={"targetHits": 100})
        q = qb.select("*").from_("articles").where(condition)
        expected = 'select * from articles where ({"targetHits": 100}weakAnd(title contains "Python", description contains "Programming"))'
        self.assertEqual(q, expected)
        return q

    def test_geolocation(self):
        condition = qb.geoLocation("location_field", 37.7749, -122.4194, "10km")
        q = qb.select("*").from_("places").where(condition)
        expected = 'select * from places where geoLocation(location_field, 37.7749, -122.4194, "10km")'
        self.assertEqual(q, expected)
        return q

    def test_condition_all_any(self):
        c1 = qb.QueryField("f1").contains("v1")
        c2 = qb.QueryField("f2").contains("v2")
        c3 = qb.QueryField("f3").contains("v3")
        condition = qb.all(c1, c2, qb.any(c3, ~c1))
        q = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where f1 contains "v1" and f2 contains "v2" and (f3 contains "v3" or !(f1 contains "v1"))'
        self.assertEqual(q, expected)
        return q

    def test_order_by_with_annotations(self):
        f1 = "relevance"
        f2 = "price"
        annotations = {"function": "uca", "locale": "en_US", "strength": "IDENTICAL"}
        q = qb.select("*").from_("products").orderByDesc(f1, annotations).orderByAsc(f2)
        expected = 'select * from products order by {"function":"uca","locale":"en_US","strength":"IDENTICAL"}relevance desc, price asc'
        self.assertEqual(q, expected)
        return q

    def test_field_comparison_methods_builtins(self):
        f1 = qb.QueryField("age")
        condition = (f1 >= 18) & (f1 < 30)
        q = qb.select("*").from_("users").where(condition)
        expected = "select * from users where age >= 18 and age < 30"
        self.assertEqual(q, expected)
        return q

    def test_field_comparison_methods(self):
        f1 = qb.QueryField("age")
        condition = (f1.ge(18) & f1.lt(30)) | f1.eq(40)
        q = qb.select("*").from_("users").where(condition)
        expected = "select * from users where (age >= 18 and age < 30) or age = 40"
        self.assertEqual(q, expected)
        return q

    def test_filter_annotation(self):
        f1 = qb.QueryField("title")
        condition = f1.contains("Python").annotate({"filter": True})
        q = qb.select("*").from_("articles").where(condition)
        expected = 'select * from articles where {filter:true}title contains "Python"'
        self.assertEqual(q, expected)
        return q

    def test_non_empty(self):
        condition = qb.nonEmpty(qb.QueryField("comments").eq("any_value"))
        q = qb.select("*").from_("posts").where(condition)
        expected = 'select * from posts where nonEmpty(comments = "any_value")'
        self.assertEqual(q, expected)
        return q

    def test_dotproduct(self):
        condition = qb.dotProduct("vector_field", {"feature1": 1, "feature2": 2})
        q = qb.select("*").from_("vectors").where(condition)
        expected = 'select * from vectors where dotProduct(vector_field, {"feature1": 1, "feature2": 2})'
        self.assertEqual(q, expected)
        return q

    def test_in_range_string_values(self):
        f1 = qb.QueryField("date")
        condition = f1.in_range("2021-01-01", "2021-12-31")
        q = qb.select("*").from_("events").where(condition)
        expected = "select * from events where range(date, 2021-01-01, 2021-12-31)"
        self.assertEqual(q, expected)
        return q

    def test_condition_inversion(self):
        f1 = qb.QueryField("status")
        condition = ~f1.eq("inactive")
        q = qb.select("*").from_("users").where(condition)
        expected = 'select * from users where !(status = "inactive")'
        self.assertEqual(q, expected)
        return q

    def test_multiple_parameters(self):
        f1 = qb.QueryField("title")
        condition = f1.contains("Python")
        q = (
            qb.select("*")
            .from_("articles")
            .where(condition)
            .add_parameter("tracelevel", 1)
            .add_parameter("language", "en")
        )
        expected = 'select * from articles where title contains "Python"&tracelevel=1&language=en'
        self.assertEqual(q, expected)
        return q

    def test_multiple_groupings(self):
        grouping = G.all(
            G.group("category"),
            G.max(10),
            G.output(G.count()),
            G.each(G.group("subcategory"), G.output(G.summary())),
        )
        q = qb.select("*").from_("products").groupby(grouping)
        expected = "select * from products | all(group(category) max(10) output(count()) each(group(subcategory) output(summary())))"
        self.assertEqual(q, expected)
        return q

    def test_userquery_basic(self):
        condition = qb.userQuery("search terms")
        q = qb.select("*").from_("documents").where(condition)
        expected = 'select * from documents where userQuery("search terms")'
        self.assertEqual(q, expected)
        return q

    def test_rank_multiple_conditions(self):
        condition = qb.rank(
            qb.userQuery(),
            qb.dotProduct("embedding", {"feature1": 1}),
            qb.weightedSet("tags", {"tag1": 2}),
        )
        q = qb.select("*").from_("documents").where(condition)
        expected = 'select * from documents where rank(userQuery(), dotProduct(embedding, {"feature1": 1}), weightedSet(tags, {"tag1": 2}))'
        print(q)
        print(expected)
        self.assertEqual(q, expected)
        return q

    def test_non_empty_with_annotations(self):
        annotated_field = qb.QueryField("comments").annotate({"filter": True})
        condition = qb.nonEmpty(annotated_field)
        q = qb.select("*").from_("posts").where(condition)
        expected = "select * from posts where nonEmpty(({filter:true})comments)"
        self.assertEqual(q, expected)
        return q

    def test_weight_annotation(self):
        condition = qb.QueryField("title").contains(
            "heads", annotations={"weight": 200}
        )
        q = qb.select("*").from_("s1").where(condition)
        expected = 'select * from s1 where title contains({weight:200}"heads")'
        self.assertEqual(q, expected)
        return q

    def test_nearest_neighbor_annotations(self):
        condition = qb.nearestNeighbor(
            field="dense_rep", query_vector="q_dense", annotations={"targetHits": 10}
        )
        q = qb.select(["id, text"]).from_("m").where(condition)
        expected = "select id, text from m where ({targetHits:10}nearestNeighbor(dense_rep, q_dense))"
        self.assertEqual(q, expected)
        return q

    def test_phrase(self):
        text = qb.QueryField("text")
        condition = text.contains(qb.phrase("st", "louis", "blues"))
        query = qb.select("*").where(condition)
        expected = 'select * from * where text contains phrase("st", "louis", "blues")'
        self.assertEqual(query, expected)
        return query

    def test_near(self):
        title = qb.QueryField("title")
        condition = title.contains(qb.near("madonna", "saint"))
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where title contains near("madonna", "saint")'
        self.assertEqual(query, expected)
        return query

    def test_near_with_distance(self):
        title = qb.QueryField("title")
        condition = title.contains(qb.near("madonna", "saint", distance=10))
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where title contains ({distance:10}near("madonna", "saint"))'
        self.assertEqual(query, expected)
        return query

    def test_onear(self):
        title = qb.QueryField("title")
        condition = title.contains(qb.onear("madonna", "saint"))
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where title contains onear("madonna", "saint")'
        self.assertEqual(query, expected)
        return query

    def test_onear_with_distance(self):
        title = qb.QueryField("title")
        condition = title.contains(qb.onear("madonna", "saint", distance=5))
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where title contains ({distance:5}onear("madonna", "saint"))'
        self.assertEqual(query, expected)
        return query

    def test_same_element(self):
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
        expected = 'select * from sd1 where persons contains sameElement(first_name contains "Joe", last_name contains "Smith", year_of_birth < 1940)'
        self.assertEqual(query, expected)
        return query

    def test_equiv(self):
        fieldName = qb.QueryField("fieldName")
        condition = fieldName.contains(qb.equiv("Snoop Dogg", "Calvin Broadus"))
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where fieldName contains equiv("Snoop Dogg", "Calvin Broadus")'
        self.assertEqual(query, expected)
        return query

    def test_uri(self):
        myUrlField = qb.QueryField("myUrlField")
        condition = myUrlField.contains(qb.uri("vespa.ai/foo"))
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where myUrlField contains uri("vespa.ai/foo")'
        self.assertEqual(query, expected)
        return query

    def test_fuzzy(self):
        myStringAttribute = qb.QueryField("f1")
        annotations = {"prefixLength": 1, "maxEditDistance": 2}
        condition = myStringAttribute.contains(
            qb.fuzzy("parantesis", annotations=annotations)
        )
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where f1 contains ({prefixLength:1,maxEditDistance:2}fuzzy("parantesis"))'
        self.assertEqual(query, expected)
        return query

    def test_userinput(self):
        condition = qb.userInput("@myvar")
        query = qb.select("*").from_("sd1").where(condition)
        expected = "select * from sd1 where userInput(@myvar)"
        self.assertEqual(query, expected)
        return query

    def test_userinput_param(self):
        condition = qb.userInput("@animal")
        query = qb.select("*").from_("sd1").where(condition).param("animal", "panda")
        expected = "select * from sd1 where userInput(@animal)&animal=panda"
        self.assertEqual(query, expected)
        return query

    def test_userinput_with_defaultindex(self):
        condition = qb.userInput("@myvar").annotate({"defaultIndex": "text"})
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where {defaultIndex:"text"}userInput(@myvar)'
        self.assertEqual(query, expected)
        return query

    def test_in_operator_intfield(self):
        integer_field = qb.QueryField("age")
        condition = integer_field.in_(10, 20, 30)
        query = qb.select("*").from_("sd1").where(condition)
        expected = "select * from sd1 where age in (10, 20, 30)"
        self.assertEqual(query, expected)
        return query

    def test_in_operator_stringfield(self):
        string_field = qb.QueryField("status")
        condition = string_field.in_("active", "inactive")
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where status in ("active", "inactive")'
        self.assertEqual(query, expected)
        return query

    def test_predicate(self):
        condition = qb.predicate(
            "predicate_field",
            attributes={"gender": "Female"},
            range_attributes={"age": "20L"},
        )
        query = qb.select("*").from_("sd1").where(condition)
        expected = 'select * from sd1 where predicate(predicate_field,{"gender":"Female"},{"age":20L})'
        self.assertEqual(query, expected)
        return query

    def test_true(self):
        query = qb.select("*").from_("sd1").where(True)
        expected = "select * from sd1 where true"
        self.assertEqual(query, expected)
        return query

    def test_false(self):
        query = qb.select("*").from_("sd1").where(False)
        expected = "select * from sd1 where false"
        self.assertEqual(query, expected)
        return query


if __name__ == "__main__":
    unittest.main()
