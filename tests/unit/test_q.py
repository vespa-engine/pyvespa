import unittest
from vespa.querybuilder import Query, Q, Queryfield, G, Condition


class TestQueryBuilder(unittest.TestCase):
    def test_dotProduct_with_annotations(self):
        condition = Q.dotProduct(
            "weightedset_field",
            {"feature1": 1, "feature2": 2},
            annotations={"label": "myDotProduct"},
        )
        q = Query(select_fields="*").from_("querybuilder").where(condition)
        expected = 'select * from querybuilder where ({label:"myDotProduct"}dotProduct(weightedset_field, {"feature1":1,"feature2":2}))'
        self.assertEqual(q, expected)
        return q

    def test_geoLocation_with_annotations(self):
        condition = Q.geoLocation(
            "location_field",
            37.7749,
            -122.4194,
            "10km",
            annotations={"targetHits": 100},
        )
        q = Query(select_fields="*").from_("querybuilder").where(condition)
        expected = 'select * from querybuilder where ({targetHits:100}geoLocation(location_field, 37.7749, -122.4194, "10km"))'
        self.assertEqual(q, expected)
        return q

    def test_select_specific_fields(self):
        f1 = Queryfield("f1")
        condition = f1.contains("v1")
        q = Query(select_fields=["f1", "f2"]).from_("sd1").where(condition)

        self.assertEqual(q, 'select f1, f2 from sd1 where f1 contains "v1"')

    def test_select_from_specific_sources(self):
        f1 = Queryfield("f1")
        condition = f1.contains("v1")
        q = Query(select_fields="*").from_("sd1").where(condition)

        self.assertEqual(q, 'select * from sd1 where f1 contains "v1"')

    def test_select_from_multiples_sources(self):
        f1 = Queryfield("f1")
        condition = f1.contains("v1")
        q = Query(select_fields="*").from_("sd1", "sd2").where(condition)

        self.assertEqual(q, 'select * from sd1, sd2 where f1 contains "v1"')

    def test_basic_and_andnot_or_offset_limit_param_order_by_and_contains(self):
        f1 = Queryfield("f1")
        f2 = Queryfield("f2")
        f3 = Queryfield("f3")
        f4 = Queryfield("f4")
        condition = ((f1.contains("v1") & f2.contains("v2")) | f3.contains("v3")) & (
            ~f4.contains("v4")
        )
        q = (
            Query(select_fields="*")
            .from_("querybuilder")
            .where(condition)
            .set_offset(1)
            .set_limit(2)
            .set_timeout(3000)
            .orderByDesc("age")
            .orderByAsc("duration")
        )

        expected = 'select * from querybuilder where ((f1 contains "v1" and f2 contains "v2") or f3 contains "v3") and !(f4 contains "v4") order by age desc, duration asc limit 2 offset 1 timeout 3000'
        self.assertEqual(q, expected)
        return q

    def test_matches(self):
        condition = (
            (Queryfield("f1").matches("v1") & Queryfield("f2").matches("v2"))
            | Queryfield("f3").matches("v3")
        ) & ~Queryfield("f4").matches("v4")
        q = Query(select_fields="*").from_("sd1").where(condition)
        expected = 'select * from sd1 where ((f1 matches "v1" and f2 matches "v2") or f3 matches "v3") and !(f4 matches "v4")'
        self.assertEqual(q, expected)

    def test_nested_queries(self):
        nested_query = (
            Queryfield("f2").contains("2") & Queryfield("f3").contains("3")
        ) | (Queryfield("f2").contains("4") & ~Queryfield("f3").contains("5"))
        condition = Queryfield("f1").contains("1") & ~nested_query
        q = Query(select_fields="*").from_("sd1").where(condition)
        expected = 'select * from sd1 where f1 contains "1" and (!((f2 contains "2" and f3 contains "3") or (f2 contains "4" and !(f3 contains "5"))))'
        self.assertEqual(q, expected)

    def test_userInput_with_and_without_defaultIndex(self):
        condition = Q.userQuery(value="value1") & Q.userQuery(
            index="index", value="value2"
        )
        q = Query(select_fields="*").from_("sd1").where(condition)
        expected = 'select * from sd1 where userQuery("value1") and ({"defaultIndex":"index"})userQuery("value2")'
        self.assertEqual(q, expected)

    def test_fields_duration(self):
        f1 = Queryfield("subject")
        f2 = Queryfield("display_date")
        f3 = Queryfield("duration")
        condition = Query(select_fields=[f1, f2]).from_("calendar").where(f3 > 0)
        expected = "select subject, display_date from calendar where duration > 0"
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
        )
        expected = "select id, text from m where userQuery() or ({targetHits:10}nearestNeighbor(dense_rep, q_dense))"
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
        )
        expected = "select * from doc where " + " or ".join(
            [
                f"({{targetHits:100}}nearestNeighbor(colbert, binary_vector_{i}))"
                for i in range(32)
            ]
        )
        self.assertEqual(q, expected)

    def test_field_comparison_operators(self):
        f1 = Queryfield("age")
        condition = (f1 > 30) & (f1 <= 50)
        q = Query(select_fields="*").from_("people").where(condition)
        expected = "select * from people where age > 30 and age <= 50"
        self.assertEqual(q, expected)

    def test_field_in_range(self):
        f1 = Queryfield("age")
        condition = f1.in_range(18, 65)
        q = Query(select_fields="*").from_("people").where(condition)
        expected = "select * from people where range(age, 18, 65)"
        self.assertEqual(q, expected)

    def test_field_annotation(self):
        f1 = Queryfield("title")
        annotations = {"highlight": True}
        annotated_field = f1.annotate(annotations)
        q = Query(select_fields="*").from_("articles").where(annotated_field)
        expected = "select * from articles where ({highlight:true})title"
        self.assertEqual(q, expected)

    def test_condition_annotation(self):
        f1 = Queryfield("title")
        condition = f1.contains("Python")
        annotated_condition = condition.annotate({"filter": True})
        q = Query(select_fields="*").from_("articles").where(annotated_condition)
        expected = 'select * from articles where ({filter:true})title contains "Python"'
        self.assertEqual(q, expected)

    def test_grouping_aggregation(self):
        grouping = G.all(G.group("category"), G.output(G.count()))
        q = Query(select_fields="*").from_("products").group(grouping)
        expected = "select * from products | all(group(category) output(count()))"
        self.assertEqual(q, expected)

    def test_add_parameter(self):
        f1 = Queryfield("title")
        condition = f1.contains("Python")
        q = (
            Query(select_fields="*")
            .from_("articles")
            .where(condition)
            .add_parameter("tracelevel", 1)
        )
        expected = 'select * from articles where title contains "Python"&tracelevel=1'
        self.assertEqual(q, expected)

    def test_custom_ranking_expression(self):
        condition = Q.rank(
            Q.userQuery(), Q.dotProduct("embedding", {"feature1": 1, "feature2": 2})
        )
        q = Query(select_fields="*").from_("documents").where(condition)
        expected = 'select * from documents where rank(userQuery(), dotProduct(embedding, {"feature1":1,"feature2":2}))'
        self.assertEqual(q, expected)

    def test_wand(self):
        condition = Q.wand("keywords", {"apple": 10, "banana": 20})
        q = Query(select_fields="*").from_("fruits").where(condition)
        expected = 'select * from fruits where wand(keywords, {"apple":10,"banana":20})'
        self.assertEqual(q, expected)

    def test_weakand(self):
        condition1 = Queryfield("title").contains("Python")
        condition2 = Queryfield("description").contains("Programming")
        condition = Q.weakAnd(
            condition1, condition2, annotations={"targetNumHits": 100}
        )
        q = Query(select_fields="*").from_("articles").where(condition)
        expected = 'select * from articles where ({"targetNumHits":100}weakAnd(title contains "Python", description contains "Programming"))'
        self.assertEqual(q, expected)

    def test_geoLocation(self):
        condition = Q.geoLocation("location_field", 37.7749, -122.4194, "10km")
        q = Query(select_fields="*").from_("places").where(condition)
        expected = 'select * from places where geoLocation(location_field, 37.7749, -122.4194, "10km")'
        self.assertEqual(q, expected)

    def test_condition_all_any(self):
        c1 = Queryfield("f1").contains("v1")
        c2 = Queryfield("f2").contains("v2")
        c3 = Queryfield("f3").contains("v3")
        condition = Condition.all(c1, c2, Condition.any(c3, ~c1))
        q = Query(select_fields="*").from_("sd1").where(condition)
        expected = 'select * from sd1 where f1 contains "v1" and f2 contains "v2" and (f3 contains "v3" or !(f1 contains "v1"))'
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
        )
        expected = (
            'select * from products order by {"strength":0.5}relevance desc, price asc'
        )
        self.assertEqual(q, expected)

    def test_field_comparison_methods(self):
        f1 = Queryfield("age")
        condition = f1.ge(18) & f1.lt(30)
        q = Query(select_fields="*").from_("users").where(condition)
        expected = "select * from users where age >= 18 and age < 30"
        self.assertEqual(q, expected)

    def test_filter_annotation(self):
        f1 = Queryfield("title")
        condition = f1.contains("Python").annotate({"filter": True})
        q = Query(select_fields="*").from_("articles").where(condition)
        expected = 'select * from articles where ({filter:true})title contains "Python"'
        self.assertEqual(q, expected)

    def test_nonEmpty(self):
        condition = Q.nonEmpty(Queryfield("comments").eq("any_value"))
        q = Query(select_fields="*").from_("posts").where(condition)
        expected = 'select * from posts where nonEmpty(comments = "any_value")'
        self.assertEqual(q, expected)

    def test_dotProduct(self):
        condition = Q.dotProduct("vector_field", {"feature1": 1, "feature2": 2})
        q = Query(select_fields="*").from_("vectors").where(condition)
        expected = 'select * from vectors where dotProduct(vector_field, {"feature1":1,"feature2":2})'
        self.assertEqual(q, expected)

    def test_in_range_string_values(self):
        f1 = Queryfield("date")
        condition = f1.in_range("2021-01-01", "2021-12-31")
        q = Query(select_fields="*").from_("events").where(condition)
        expected = "select * from events where range(date, 2021-01-01, 2021-12-31)"
        self.assertEqual(q, expected)

    def test_condition_inversion(self):
        f1 = Queryfield("status")
        condition = ~f1.eq("inactive")
        q = Query(select_fields="*").from_("users").where(condition)
        expected = 'select * from users where !(status = "inactive")'
        self.assertEqual(q, expected)

    def test_multiple_parameters(self):
        f1 = Queryfield("title")
        condition = f1.contains("Python")
        q = (
            Query(select_fields="*")
            .from_("articles")
            .where(condition)
            .add_parameter("tracelevel", 1)
            .add_parameter("language", "en")
        )
        expected = 'select * from articles where title contains "Python"&tracelevel=1&language=en'
        self.assertEqual(q, expected)

    def test_multiple_groupings(self):
        grouping = G.all(
            G.group("category"),
            G.maxRtn(10),
            G.output(G.count()),
            G.each(G.group("subcategory"), G.output(G.summary())),
        )
        q = Query(select_fields="*").from_("products").group(grouping)
        expected = "select * from products | all(group(category) max(10) output(count()) each(group(subcategory) output(summary())))"
        self.assertEqual(q, expected)

    def test_default_index_annotation(self):
        condition = Q.userQuery("search terms", index="default_field")
        q = Query(select_fields="*").from_("documents").where(condition)
        expected = 'select * from documents where ({"defaultIndex":"default_field"})userQuery("search terms")'
        self.assertEqual(q, expected)

    def test_Q_p_function(self):
        condition = Q.p(
            Queryfield("f1").contains("v1"),
            Queryfield("f2").contains("v2"),
            Queryfield("f3").contains("v3"),
        )
        q = Query(select_fields="*").from_("sd1").where(condition)
        expected = 'select * from sd1 where f1 contains "v1" and f2 contains "v2" and f3 contains "v3"'
        self.assertEqual(q, expected)

    def test_rank_multiple_conditions(self):
        condition = Q.rank(
            Q.userQuery(),
            Q.dotProduct("embedding", {"feature1": 1}),
            Q.weightedSet("tags", {"tag1": 2}),
        )
        q = Query(select_fields="*").from_("documents").where(condition)
        expected = 'select * from documents where rank(userQuery(), dotProduct(embedding, {"feature1":1}), weightedSet(tags, {"tag1":2}))'
        self.assertEqual(q, expected)

    def test_nonEmpty_with_annotations(self):
        annotated_field = Queryfield("comments").annotate({"filter": True})
        condition = Q.nonEmpty(annotated_field)
        q = Query(select_fields="*").from_("posts").where(condition)
        expected = "select * from posts where nonEmpty(({filter:true})comments)"
        self.assertEqual(q, expected)

    def test_weight_annotation(self):
        condition = Queryfield("title").contains("heads", annotations={"weight": 200})
        q = Query(select_fields="*").from_("s1").where(condition)
        expected = 'select * from s1 where title contains({weight:200}"heads")'
        self.assertEqual(q, expected)

    def test_nearest_neighbor_annotations(self):
        condition = Q.nearestNeighbor(
            field="dense_rep", query_vector="q_dense", annotations={"targetHits": 10}
        )
        q = Query(select_fields=["id, text"]).from_("m").where(condition)
        expected = "select id, text from m where ({targetHits:10}nearestNeighbor(dense_rep, q_dense))"
        self.assertEqual(q, expected)

    def test_phrase(self):
        text = Queryfield("text")
        condition = text.contains(Q.phrase("st", "louis", "blues"))
        query = Q.select("*").where(condition)
        expected = 'select * from * where text contains phrase("st", "louis", "blues")'
        self.assertEqual(query, expected)

    def test_near(self):
        title = Queryfield("title")
        condition = title.contains(Q.near("madonna", "saint"))
        query = Q.select("*").where(condition)
        expected = 'select * from * where title contains near("madonna", "saint")'
        self.assertEqual(query, expected)

    def test_onear(self):
        title = Queryfield("title")
        condition = title.contains(Q.onear("madonna", "saint"))
        query = Q.select("*").where(condition)
        expected = 'select * from * where title contains onear("madonna", "saint")'
        self.assertEqual(query, expected)

    def test_sameElement(self):
        persons = Queryfield("persons")
        first_name = Queryfield("first_name")
        last_name = Queryfield("last_name")
        year_of_birth = Queryfield("year_of_birth")
        condition = persons.contains(
            Q.sameElement(
                first_name.contains("Joe"),
                last_name.contains("Smith"),
                year_of_birth < 1940,
            )
        )
        query = Q.select("*").where(condition)
        expected = 'select * from * where persons contains sameElement(first_name contains "Joe", last_name contains "Smith", year_of_birth < 1940)'
        self.assertEqual(query, expected)

    def test_equiv(self):
        fieldName = Queryfield("fieldName")
        condition = fieldName.contains(Q.equiv("A", "B"))
        query = Q.select("*").where(condition)
        expected = 'select * from * where fieldName contains equiv("A", "B")'
        self.assertEqual(query, expected)

    def test_uri(self):
        myUrlField = Queryfield("myUrlField")
        condition = myUrlField.contains(Q.uri("vespa.ai/foo"))
        query = Q.select("*").where(condition)
        expected = 'select * from * where myUrlField contains uri("vespa.ai/foo")'
        self.assertEqual(query, expected)

    def test_fuzzy(self):
        myStringAttribute = Queryfield("myStringAttribute")
        annotations = {"prefixLength": 1, "maxEditDistance": 2}
        condition = myStringAttribute.contains(
            Q.fuzzy("parantesis", annotations=annotations)
        )
        query = Q.select("*").where(condition)
        expected = 'select * from * where myStringAttribute contains ({prefixLength:1,maxEditDistance:2}fuzzy("parantesis"))'
        self.assertEqual(query, expected)

    def test_userInput(self):
        condition = Q.userInput("@animal")
        query = Q.select("*").where(condition).param("animal", "panda")
        expected = "select * from * where userInput(@animal)&animal=panda"
        self.assertEqual(query, expected)

    def test_in_operator(self):
        integer_field = Queryfield("integer_field")
        condition = integer_field.in_(10, 20, 30)
        query = Q.select("*").where(condition)
        expected = "select * from * where integer_field in (10, 20, 30)"
        self.assertEqual(query, expected)

    def test_predicate(self):
        condition = Q.predicate(
            "predicate_field",
            attributes={"gender": "Female"},
            range_attributes={"age": "20L"},
        )
        query = Q.select("*").where(condition)
        expected = 'select * from * where predicate(predicate_field,{"gender":"Female"},{"age":20L})'
        self.assertEqual(query, expected)

    def test_true(self):
        condition = Q.true()
        query = Q.select("*").where(condition)
        expected = "select * from * where true"
        self.assertEqual(query, expected)

    def test_false(self):
        condition = Q.false()
        query = Q.select("*").where(condition)
        expected = "select * from * where false"
        self.assertEqual(query, expected)


if __name__ == "__main__":
    unittest.main()
