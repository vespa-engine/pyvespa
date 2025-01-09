import vespa.querybuilder as qb
from vespa.querybuilder import Grouping as G
import unittest


class TestQueryBuilderGrouping(unittest.TestCase):
    def test_grouping_with_condition(self):
        grouping = G.all(
            G.group("customer"),
            G.each(G.output(G.sum("price"))),
        )
        q = qb.select("*").from_("purchase").where(True).set_limit(0).groupby(grouping)
        expected = "select * from purchase where true limit 0 | all(group(customer) each(output(sum(price))))"
        self.assertEqual(q, expected)
        return q

    def test_grouping_with_ordering_and_limiting(self):
        grouping = G.all(
            G.group("customer"),
            G.max(2),
            G.precision(12),
            G.order(-G.count()),
            G.each(G.output(G.sum("price"))),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = "select * from purchase where true | all(group(customer) max(2) precision(12) order(-count()) each(output(sum(price))))"
        self.assertEqual(q, expected)
        return q

    def test_grouping_with_map_keys(self):
        grouping = G.all(
            G.group("mymap.key"),
            G.each(
                G.group("mymap.value"),
                G.each(G.output(G.count())),
            ),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = "select * from purchase where true | all(group(mymap.key) each(group(mymap.value) each(output(count()))))"
        self.assertEqual(q, expected)
        return q

    def test_group_by_year(self):
        grouping = G.all(
            G.group("time.year(a)"),
            G.each(G.output(G.count())),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = "select * from purchase where true | all(group(time.year(a)) each(output(count())))"
        self.assertEqual(q, expected)
        return q

    def test_grouping_with_date_agg(self):
        grouping = G.all(
            G.group("time.year(a)"),
            G.each(
                G.output(G.count()),
                G.all(
                    G.group("time.monthofyear(a)"),
                    G.each(
                        G.output(G.count()),
                        G.all(
                            G.group("time.dayofmonth(a)"),
                            G.each(
                                G.output(G.count()),
                                G.all(
                                    G.group("time.hourofday(a)"),
                                    G.each(G.output(G.count())),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = "select * from purchase where true | all(group(time.year(a)) each(output(count()) all(group(time.monthofyear(a)) each(output(count()) all(group(time.dayofmonth(a)) each(output(count()) all(group(time.hourofday(a)) each(output(count())))))))))"
        self.assertEqual(q, expected)
        return q

    def test_grouping_hits_per_group(self):
        # Return the three most expensive parts per customer:
        # 'select * from purchase where true | all(group(customer) each(max(3) each(output(summary()))))'
        grouping = G.all(
            G.group("customer"),
            G.each(
                G.max(3),
                G.each(G.output(G.summary())),
            ),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = "select * from purchase where true | all(group(customer) each(max(3) each(output(summary()))))"
        self.assertEqual(q, expected)
        return q
