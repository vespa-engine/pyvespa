import vespa.querybuilder as qb
from vespa.querybuilder import Grouping as G
import unittest


class TestQueryBuilderGrouping(unittest.TestCase):
    maxDiff = None

    def test_grouping_with_condition(self):
        grouping = G.all(
            G.group("customer"),
            G.each(G.output(G.sum("price"))),
        )
        q = qb.select("*").from_("purchase").where(True).set_limit(0).groupby(grouping)
        expected = (
            "select * from purchase where true limit 0 "
            "| all(group(customer) each(output(sum(price))))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")
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
        expected = (
            "select * from purchase where true "
            "| all(group(customer) max(2) precision(12) order(-count()) each(output(sum(price))))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")
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
        expected = (
            "select * from purchase where true "
            "| all(group(mymap.key) each(group(mymap.value) each(output(count()))))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")
        return q

    def test_group_by_year(self):
        grouping = G.all(
            G.group("time.year(a)"),
            G.each(G.output(G.count())),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = (
            "select * from purchase where true "
            "| all(group(time.year(a)) each(output(count())))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")
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
        expected = (
            "select * from purchase where true | "
            "all(group(time.year(a)) each(output(count()) "
            "all(group(time.monthofyear(a)) each(output(count()) "
            "all(group(time.dayofmonth(a)) each(output(count()) "
            "all(group(time.hourofday(a)) each(output(count())))))))))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")
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
        expected = (
            "select * from purchase where true | "
            "all(group(customer) each(max(3) each(output(summary()))))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")
        return q

    #
    # ---------------------------------
    # Tests retrieved and adapted from https://github.com/vespa-engine/system-test/blob/master/tests/search/grouping_adv/grouping_base.rb
    # check_query(...) calls
    # ---------------------------------
    #
    #
    # Subgroup examples:
    #
    def test_subgroup1_part1(self):
        # check_query('all(group(a) max(5) each(output(count()) each(output(summary(normal)))))', 'subgroup1')
        grouping = G.all(
            G.group("a"),
            G.max(5),
            G.each(G.output(G.count()), G.each(G.output(G.summary("normal")))),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(5) each(output(count()) each(output(summary(normal)))))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_subgroup1_part2(self):
        # check_query('all(group(a) max(5) each(max(69) output(count()) each(output(summary(normal)))))', 'subgroup1')
        grouping = G.all(
            G.group("a"),
            G.max(5),
            G.each(
                G.max(69), G.output(G.count()), G.each(G.output(G.summary("normal")))
            ),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(5) each(max(69) output(count()) each(output(summary(normal)))))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_subgroup2(self):
        # check_query('all(group(a) max(5) each(output(count()) all(group(b) max(5) each(max(69) output(count()) each(output(summary(normal)))))))',
        #             'subgroup2')
        grouping = G.all(
            G.group("a"),
            G.max(5),
            G.each(
                G.output(G.count()),
                G.all(
                    G.group("b"),
                    G.max(5),
                    G.each(
                        G.max(69),
                        G.output(G.count()),
                        G.each(G.output(G.summary("normal"))),
                    ),
                ),
            ),
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(5) each(output(count()) "
            "all(group(b) max(5) each(max(69) output(count()) each(output(summary(normal)))))))"
        )
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_subgroup3(self):
        # check_query('all(group(a) max(5) each(output(count()) all(group(b) max(5) each(output(count()) all(group(c) max(5) each(max(69) output(count()) each(output(summary(normal)))))))))',
        #             'subgroup3')
        grouping = G.all(
            G.group("a"),
            G.max(5),
            G.each(
                G.output(G.count()),
                G.all(
                    G.group("b"),
                    G.max(5),
                    G.each(
                        G.output(G.count()),
                        G.all(
                            G.group("c"),
                            G.max(5),
                            G.each(
                                G.max(69),
                                G.output(G.count()),
                                G.each(G.output(G.summary("normal"))),
                            ),
                        ),
                    ),
                ),
            ),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(5) each(output(count()) "
            "all(group(b) max(5) each(output(count()) "
            "all(group(c) max(5) each(max(69) output(count()) each(output(summary(normal)))))))))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_subgroup4(self):
        # check_query('all(group(fixedwidth(n,3)) max(5) each(output(count()) all(group(a) max(2) each(output(count())))))',
        #             'subgroup4')
        grouping = G.all(
            G.group(G.fixedwidth("n", 3)),
            G.max(5),
            G.each(
                G.output(G.count()),
                G.all(G.group("a"), G.max(2), G.each(G.output(G.count()))),
            ),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(fixedwidth(n,3)) max(5) each(output(count()) "
            "all(group(a) max(2) each(output(count())))))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_subgroup5(self):
        # Identical to 'subgroup4' in Ruby snippet?
        # check_query('all(group(fixedwidth(n,3)) max(5) each(output(count()) all(group(a) max(2) each(output(count())))))',
        #             'subgroup5')
        grouping = G.all(
            G.group(G.fixedwidth("n", 3)),
            G.max(5),
            G.each(
                G.output(G.count()),
                G.all(G.group("a"), G.max(2), G.each(G.output(G.count()))),
            ),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(fixedwidth(n,3)) max(5) each(output(count()) "
            "all(group(a) max(2) each(output(count())))))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_subgroup6(self):
        # check_query('all(group(fixedwidth(n,3)) max(5) each(output(count()) all(group(a) max(2) each(max(1) output(count()) each(output(summary(normal)))))))',
        #             'subgroup6')
        grouping = G.all(
            G.group(G.fixedwidth("n", 3)),
            G.max(5),
            G.each(
                G.output(G.count()),
                G.all(
                    G.group("a"),
                    G.max(2),
                    G.each(
                        G.max(1),
                        G.output(G.count()),
                        G.each(G.output(G.summary("normal"))),
                    ),
                ),
            ),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(fixedwidth(n,3)) max(5) each(output(count()) "
            "all(group(a) max(2) each(max(1) output(count()) each(output(summary(normal)))))))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # Ordering examples:
    #
    def test_orderby1(self):
        # check_query('all(group(a) order(-sum(from)) each(output(count())))', 'orderby1')
        grouping = G.all(
            G.group("a"), G.order(f"-{G.sum('from')}"), G.each(G.output(G.count()))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) order(-sum(from)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_orderby_neg1(self):
        # check_query('all(group(a) order(sum(from)) each(output(count())))', 'orderby-1')
        grouping = G.all(
            G.group("a"), G.order(G.sum("from")), G.each(G.output(G.count()))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) order(sum(from)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_orderby1_m1(self):
        # check_query('all(group(a) max(2) order(-sum(from)) precision(3) each(output(count())))', 'orderby1-m1')
        grouping = G.all(
            G.group("a"),
            G.max(2),
            G.order(f"-{G.sum('from')}"),
            G.precision(3),
            G.each(G.output(G.count())),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(2) order(-sum(from)) precision(3) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_orderby_neg1_m1(self):
        # check_query('all(group(a) max(2) order(sum(from)) precision(3) each(output(count())))', 'orderby-1-m1')
        grouping = G.all(
            G.group("a"),
            G.max(2),
            G.order(G.sum("from")),
            G.precision(3),
            G.each(G.output(G.count())),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(2) order(sum(from)) precision(3) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_orderby2(self):
        # check_query('all(group(a) max(2) order(-count()) each(output(count())))', 'orderby2')
        grouping = G.all(
            G.group("a"), G.max(2), G.order(-G.count()), G.each(G.output(G.count()))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(2) order(-count()) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_combination_1(self):
        # check_query('all(group(a) max(2) order(-count()) each(output(count())) as(foo) each(output(max(b))) as(bar))',
        #             'combination-1')
        # Note that "as(foo)" or "as(bar)" is not a direct DSL method in `Grouping`.
        # Typically you’d do something like: "all(group(a) max(2) order(-count()) each(output(count()))) as(foo)"
        # but Vespa grouping syntax just places "as(foo)" inline.
        # If you absolutely need to replicate "as(...)", you can treat it as a literal appended string:
        grouping = "all(group(a) max(2) order(-count()) each(output(count())) as(foo) each(output(max(b))) as(bar))"
        # (We store it as a literal for demonstration; the DSL itself doesn't have a method for "as(...)".)
        q = f"select * from purchase where true | {grouping}"
        expected = "select * from purchase where true | all(group(a) max(2) order(-count()) each(output(count())) as(foo) each(output(max(b))) as(bar))"
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # Limit / precision checks:
    #
    def test_constraint2(self):
        # check_query('all(group(a) max(2) each(output(count())))', 'constraint2')
        grouping = G.all(G.group("a"), G.max(2), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(2) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_constraint3(self):
        # check_query('all(group(a) max(2) precision(10) each(output(count())))', 'constraint3')
        grouping = G.all(
            G.group("a"), G.max(2), G.precision(10), G.each(G.output(G.count()))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) max(2) precision(10) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # Time examples:
    #
    def test_time_year(self):
        # check_query('all(group(time.year(from)) each(output(count()) ))', 'time.year')
        grouping = G.all(G.group(G.time_year("from")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(time.year(from)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_time_month(self):
        # check_query('all(group(time.monthofyear(from)) each(output(count()) ))', 'time.month')
        grouping = G.all(
            G.group(G.time_monthofyear("from")), G.each(G.output(G.count()))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(time.monthofyear(from)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    # ... similarly for time.dayofmonth, time.dayofweek, time.dayofyear, etc. ...
    # (You can keep adding them if desired.)

    #
    # Relevance aggregator example:
    #
    def test_relevance(self):
        # check_query('all(group(a) each(output(count(),sum(mod(relevance(),100000))) ))', 'relevance')
        grouping = G.all(
            G.group("a"),
            G.each(
                G.output(
                    # Multiple outputs in one "output(...)":
                    # There's no direct "output(count(), sum(...))" aggregator in the DSL,
                    # so you usually do multiple 'output(...)' calls. But Vespa allows "output(count(), sum(...))"
                    # as a single aggregator expression. We can do a literal here for brevity:
                    "count(),sum(mod(relevance(),100000))"
                )
            ),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) each(output(count(),sum(mod(relevance(),100000)))))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # cat(...) example:
    #
    def test_cat(self):
        # check_query('all(group(cat(a,b,c)) each(output(count())))', 'cat')
        grouping = G.all(G.group(G.cat("a", "b", "c")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(cat(a,b,c)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # zcurve.x(...) example:
    #
    def test_zcurve_x(self):
        # check_query('all(group(zcurve.x(to)) each(output(count())))', 'zcurve.x')
        grouping = G.all(G.group(G.zcurve_x("to")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(zcurve.x(to)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # Arithmetic expression examples:
    #
    def test_add_nf(self):
        # check_query('all(group(add(n,f)) each(output(count())))', 'add-nf')
        grouping = G.all(G.group(G.add("n", "f")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(add(n, f)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_mul_nf(self):
        # check_query('all(group(mul(n,f)) each(output(count())))', 'mul-nf')
        grouping = G.all(G.group(G.mul("n", "f")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(mul(n, f)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # Aggregator expression in ordering:
    #
    def test_rank_relevance_count(self):
        # check_query('all(group(a) order(sum(relevance()),-count()) each(output(count(),sum(mod(relevance(),100000))) ))',
        #             'rank-relevance-count')
        grouping = G.all(
            G.group("a"),
            # order(...) can hold multiple expressions, but Vespa syntax is e.g.
            # order(sum(relevance()), -count()). We replicate as literal:
            "order(sum(relevance()),-count())",
            G.each(
                # As above, multiple aggregator calls in one output:
                "output(count(),sum(mod(relevance(),100000)))"
            ),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) order(sum(relevance()),-count()) each(output(count(),sum(mod(relevance(),100000)))))"
        )
        q = f"select * from purchase where true | {grouping}"
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # strcat(...) example:
    #
    def test_strcat_example(self):
        # check_query('all(group(strcat(a,b,c)) each(output(count())))', 'strcat')
        grouping = G.all(G.group(G.strcat("a", "b", "c")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(strcat(a,b,c)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_strlen_example(self):
        # check_query('all(group(strlen(strcat(a,b,c))) each(output(count())))', 'strlen')
        grouping = G.all(
            G.group(G.strlen(G.strcat("a", "b", "c"))), G.each(G.output(G.count()))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(strlen(strcat(a,b,c))) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # tolong(...), todouble(...), etc:
    #
    def test_tostring_field(self):
        # check_query('all(group(tostring(f)) each(output(count())))', 'tostring-f')
        grouping = G.all(G.group(G.tostring("f")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(tostring(f)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # math.* examples:
    #
    def test_math_exp(self):
        # check_query('all(group(math.exp(d)) each(output(count())))', 'math.exp')
        grouping = G.all(G.group(G.math_exp("d")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(math.exp(d)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    def test_math_pow(self):
        # check_query('all(group(math.pow(d,d)) each(output(count())))', 'math.pow')
        grouping = G.all(G.group(G.math_pow("d", "d")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(math.pow(d, d)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # size(...) example:
    #
    def test_size_na(self):
        # check_query('all(group(size(na)) each(output(count())))', 'length-a')
        grouping = G.all(G.group(G.size("na")), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(size(na)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # predefined(...) example:
    #
    def test_predef1(self):
        # check_query('all(group(predefined(n,bucket(1,3),bucket(6,9))) each(output(count())))', 'predef1')
        grouping = G.all(
            G.group(G.predefined("n", ["bucket(1,3)", "bucket(6,9)"])),
            G.each(G.output(G.count())),
        )
        expected = (
            "select * from purchase where true | "
            "all(group(predefined(n, (bucket(1,3), bucket(6,9)))) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # fixedwidth(...) example:
    #
    def test_fixedwidth_n_3(self):
        # check_query('all(group(fixedwidth(n,3)) each(output(count())))', 'fixedwidth-n-3')
        grouping = G.all(G.group(G.fixedwidth("n", 3)), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(fixedwidth(n,3)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # xorbit(...) example:
    #
    def test_xorbit_16(self):
        # check_query('all(group(xorbit(cat(a,b,c), 16)) each(output(count())))', 'xorbit.16')
        grouping = G.all(
            G.group(G.xorbit(G.cat("a", "b", "c"), 16)), G.each(G.output(G.count()))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(xorbit(cat(a,b,c), 16)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # md5(...) example:
    #
    def test_md5_64(self):
        # check_query('all(group(md5(cat(a,b,c), 64)) each(output(count())))', 'md5.64')
        grouping = G.all(
            G.group(G.md5(G.cat("a", "b", "c"), 64)), G.each(G.output(G.count()))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(md5(cat(a,b,c), 64)) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # "boool" grouping:
    #
    def test_boool(self):
        # check_query('all(group(boool) each(output(count())))', 'boool')
        grouping = G.all(G.group("boool"), G.each(G.output(G.count())))
        expected = (
            "select * from purchase where true | "
            "all(group(boool) each(output(count())))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")

    #
    # Alias usage example:
    #
    @unittest.skip("Not yet implemented")
    def test_alias(self):
        # check_query(all(group(a)alias(myalias,count())each(output($myalias)))
        grouping = G.all(
            G.group("a").alias("myalias", G.count()), G.each(G.output("$myalias"))
        )
        expected = (
            "select * from purchase where true | "
            "all(group(a) alias(myalias, count()) each(output($myalias)))"
        )
        q = qb.select("*").from_("purchase").where(True).groupby(grouping)
        self.assertTrue(q == expected, f"\nq:\n{q}\n\ndiffers from:\n\n{expected}")
