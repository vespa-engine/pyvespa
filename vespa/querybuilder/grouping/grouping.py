from __future__ import annotations
from typing import Union, List


class Expression(str):
    def alias(self, alias_name: str, expression: str) -> Expression:
        return Expression(f"{self} alias({alias_name},{expression})")

    def as_(self, label: str) -> Expression:
        return Expression(f"{self} as({label})")

    def __neg__(self) -> Expression:
        return Expression(f"-{self}")


class Grouping:
    """A Pythonic DSL for building Vespa grouping expressions programmatically.

    This class provides a set of static methods that build grouping syntax strings
    which can be combined to form a valid Vespa “select=…” grouping expression.

    For a guide to grouping in vespa, see https://docs.vespa.ai/en/grouping.html.
    For the reference docs, see https://docs.vespa.ai/en/reference/grouping-syntax.html.

    Minimal Example:

        >>> from vespa.querybuilder import Grouping as G
        >>> # Build a simple grouping expression which groups on "my_attribute"
        >>> # and outputs the count of matching documents under each group:
        >>> expr = G.all(
        ...     G.group("my_attribute"),
        ...     G.each(
        ...         G.output(G.count())
        ...     )
        ... )
        >>> print(expr)
        all(group(my_attribute) each(output(count())))

    In the above example, the “all(...)” wraps the grouping operations at the top
    level. We first group on “my_attribute”, then under “each(...)” we add an output
    aggregator “count()”. The “print” output is the exact grouping expression string
    you would pass to Vespa in the “select” query parameter.

    For multi-level (nested) grouping, you can nest additional calls to “group(...)”
    or “each(...)” inside. For example:

        >>> # Nested grouping:
        >>> # 1) Group by 'category'
        >>> # 2) Within each category, group by 'sub_category'
        >>> # 3) Output the count() under each sub-category
        >>> nested_expr = G.all(
        ...     G.group("category"),
        ...     G.each(
        ...         G.group("sub_category"),
        ...         G.each(
        ...             G.output(G.count())
        ...         )
        ...     )
        ... )
        >>> print(nested_expr)
        all(group(category) each(group(sub_category) each(output(count()))))

    You may use any of the static methods below to build more advanced groupings,
    aggregations, or arithmetic/string expressions for sorting, filtering, or
    bucket definitions. Refer to Vespa documentation for the complete details.
    """

    #
    # Core grouping block builders
    #

    @staticmethod
    def all(*args: Union[str, Expression]) -> Expression:
        """Corresponds to the “all(...)” grouping block in Vespa, which means
        “group all documents (no top-level grouping) and then do the enclosed operations”.

        Args:
            *args (str): Sub-expressions to include within the `all(...)` block.

        Returns:
            str: A Vespa grouping expression string.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.all(G.group("my_attribute"), G.each(G.output(G.count())))
            >>> print(expr)
            all(group(my_attribute) each(output(count())))
        """
        return Expression("all(" + " ".join(map(str, args)) + ")")

    @staticmethod
    def each(*args: Union[str, Expression]) -> Expression:
        """Corresponds to the “each(...)” grouping block in Vespa, which means
        “create a group for each unique value and then do the enclosed operations”.

        Args:
            *args (str): Sub-expressions to include within the `each(...)` block.

        Returns:
            str: A Vespa grouping expression string.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.each("output(count())", "output(avg(price))")
            >>> print(expr)
            each(output(count()) output(avg(price)))
        """
        return Expression("each(" + " ".join(map(str, args)) + ")")

    @staticmethod
    def group(field: str) -> Expression:
        """Defines a grouping step on a field or expression.

        Args:
            field (str): The field or expression on which to group.

        Returns:
            str: A Vespa grouping expression string.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.group("my_map.key")
            >>> print(expr)
            group(my_map.key)
        """
        return Expression(f"group({field})")

    #
    # Common aggregator wrappers
    #

    @staticmethod
    def count() -> Expression:
        """“count()” aggregator.

        By default, returns a string 'count()'. Negative ordering or usage can be
        done by prefixing a minus, e.g.: order(-count()) in Vespa syntax.

        Returns:
            str: 'count()' or prefixed version if used with a minus operator.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.count()
            >>> print(expr)
            count()

            >>> sort_expr = f"-{expr}"
            >>> print(sort_expr)
            -count()
        """

        class MaybenegativeCount(Expression):
            def __new__(cls):
                return super().__new__(cls, "count()")

            def __neg__(self):
                return Expression(f"-{self}")

        return MaybenegativeCount()

    @staticmethod
    def sum(value: Union[str, int, float]) -> Expression:
        """“sum(...)” aggregator. Sums the given expression or field over all documents in the group.

        Args:
            value (Union[str, int, float]): The field or numeric expression to sum.

        Returns:
            str: A Vespa grouping expression string of the form 'sum(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.sum("my_numeric_field")
            >>> print(expr)
            sum(my_numeric_field)
        """
        return Expression(f"sum({value})")

    @staticmethod
    def avg(value: Union[str, int, float]) -> Expression:
        """“avg(...)” aggregator. Computes the average of the given expression or field
        for all documents in the group.

        Args:
            value (Union[str, int, float]): The field or numeric expression to average.

        Returns:
            str: A Vespa grouping expression string of the form 'avg(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.avg("my_numeric_field")
            >>> print(expr)
            avg(my_numeric_field)
        """
        return Expression(f"avg({value})")

    @staticmethod
    def min(value: Union[str, int, float]) -> Expression:
        """“min(...)” aggregator. Keeps the minimum value of the expression or field
        among all documents in the group.

        Args:
            value (Union[str, int, float]): The field or numeric expression to find the minimum of.

        Returns:
            str: A Vespa grouping expression string of the form 'min(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.min("some_field")
            >>> print(expr)
            min(some_field)
        """
        return Expression(f"min({value})")

    @staticmethod
    def max(value: Union[str, int, float]) -> Expression:
        """“max(...)” aggregator. Keeps the maximum value of the expression or field
        among all documents in the group.

        Args:
            value (Union[str, int, float]): The field or numeric expression to find the maximum of.

        Returns:
            str: A Vespa grouping expression string of the form 'max(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.max("relevance()")
            >>> print(expr)
            max(relevance())
        """
        return Expression(f"max({value})")

    @staticmethod
    def stddev(value: Union[str, int, float]) -> Expression:
        """“stddev(...)” aggregator. Computes the population standard deviation
        for the expression or field among all documents in the group.

        Args:
            value (Union[str, int, float]): The field or numeric expression.

        Returns:
            str: A Vespa grouping expression string of the form 'stddev(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.stddev("my_numeric_field")
            >>> print(expr)
            stddev(my_numeric_field)
        """
        return Expression(f"stddev({value})")

    @staticmethod
    def xor(value: Union[str, int, float]) -> Expression:
        """“xor(...)” aggregator. XORs all values of the expression or field together
        over the documents in the group.

        Args:
            value (Union[str, int, float]): The field or numeric expression.

        Returns:
            str: A Vespa grouping expression string of the form 'xor(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.xor("my_field")
            >>> print(expr)
            xor(my_field)
        """
        return Expression(f"xor({value})")

    #
    # Grouping output and ordering
    #

    @staticmethod
    def output(*args: Union[str, Expression]) -> Expression:
        """Defines output aggregators to be collected for the grouping level.

        Args:
            *args: Multiple aggregator expressions, e.g., 'count()', 'sum(price)'.

        Returns:
            Expression: A Vespa grouping expression string of the form 'output(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.output(G.count(), G.sum("price"))
            >>> print(expr)
            output(count(),sum(price))
        """
        return Expression(f"output({','.join(str(x) for x in args)})")

    @staticmethod
    def order(*args: Union[str, Expression]) -> Expression:
        """Defines an order(...) clause to sort groups by the given expressions
        or aggregators.

        Args:
            *args: Multiple expressions or aggregators to order by.

        Returns:
            Expression: A Vespa grouping expression string of the form 'order(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.order(G.sum(G.relevance()), -G.count())
            >>> print(expr)
            order(sum(relevance()),-count())
        """
        return Expression(f"order({','.join(str(x) for x in args)})")

    @staticmethod
    def precision(value: int) -> Expression:
        """Sets the “precision(...)” for the grouping step.

        Args:
            value (int): Precision value.

        Returns:
            str: A Vespa grouping expression string of the form 'precision(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.precision(1000)
            >>> print(expr)
            precision(1000)
        """
        return Expression(f"precision({value})")

    #
    # Additional aggregator/alias syntax
    #
    # Alias is NOT CURRENTLY IMPLEMENTED
    # Would require a bit refactoring that is not obvious how to do in a simple way.
    # Thinking it is not so much used that it should be a blocker.
    # // @thomasht86
    # @staticmethod
    # def alias(alias_name: str, expression: str) -> str:
    #     """“alias(...)” syntax. Assign an alias to an expression so it can
    #     be reused without repeating the expression.

    #     Args:
    #         alias_name (str): The alias name to assign.
    #         expression (str): The expression to alias.

    #     Returns:
    #         str: A Vespa grouping expression string of the form 'alias(...)'.

    #     Examples:
    #         >>> from vespa.querybuilder import Grouping as G
    #         >>> expr = G.alias("myalias", "count()")
    #         >>> print(expr)
    #         alias(myalias,count())
    #     """
    #     return f"alias({alias_name},{expression})"

    #
    # Arithmetic expressions
    #

    @staticmethod
    def add(*expressions: Union[str, Expression]) -> Expression:
        """“add(...)” expression. Adds all arguments together in order.

        Args:
            *expressions (str): The expressions to be added.

        Returns:
            str: A Vespa expression string of the form 'add(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.add("my_field", "5", "10")
            >>> print(expr)
            add(my_field, 5, 10)
        """
        return Expression(f"add({', '.join(map(str, expressions))})")

    @staticmethod
    def sub(*expressions: Union[str, Expression]) -> Expression:
        """“sub(...)” expression. Subtracts each subsequent argument from the first.

        Args:
            *expressions (str): The expressions involved in subtraction.

        Returns:
            str: A Vespa expression string of the form 'sub(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.sub("my_field", "2")
            >>> print(expr)
            sub(my_field, 2)
        """
        return Expression(f"sub({', '.join(map(str, expressions))})")

    @staticmethod
    def mul(*expressions: Union[str, Expression]) -> Expression:
        """“mul(...)” expression. Multiplies all arguments in order.

        Args:
            *expressions (str): The expressions to multiply.

        Returns:
            str: A Vespa expression string of the form 'mul(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.mul("my_field", "2", "3")
            >>> print(expr)
            mul(my_field, 2, 3)
        """
        return Expression(f"mul({', '.join(map(str, expressions))})")

    @staticmethod
    def div(*expressions: Union[str, Expression]) -> Expression:
        """“div(...)” expression. Divides the first argument by the second, etc.

        Args:
            *expressions (str): The expressions to divide in order.

        Returns:
            str: A Vespa expression string of the form 'div(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.div("my_field", "2")
            >>> print(expr)
            div(my_field, 2)
        """
        return Expression(f"div({', '.join(map(str, expressions))})")

    @staticmethod
    def mod(*expressions: Union[str, Expression]) -> Expression:
        """“mod(...)” expression. Modulo the first argument by the second, result by the third, etc.

        Args:
            *expressions (str): The expressions to apply modulo on in order.

        Returns:
            str: A Vespa expression string of the form 'mod(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.mod("my_field", "100")
            >>> print(expr)
            mod(my_field,100)
        """
        return Expression(f"mod({','.join(map(str, expressions))})")

    #
    # Bitwise expressions
    #

    @staticmethod
    def and_(*expressions: Union[str, Expression]) -> Expression:
        """“and(...)” expression. Bitwise AND of the arguments in order.

        Args:
            *expressions (str): The expressions to apply bitwise AND.

        Returns:
            str: A Vespa expression string of the form 'and(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.and_("fieldA", "fieldB")
            >>> print(expr)
            and(fieldA, fieldB)
        """
        return Expression(f"and({', '.join(map(str, expressions))})")

    @staticmethod
    def or_(*expressions: Union[str, Expression]) -> Expression:
        """“or(...)” expression. Bitwise OR of the arguments in order.

        Args:
            *expressions (str): The expressions to apply bitwise OR.

        Returns:
            str: A Vespa expression string of the form 'or(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.or_("fieldA", "fieldB")
            >>> print(expr)
            or(fieldA, fieldB)
        """
        return Expression(f"or({', '.join(map(str, expressions))})")

    @staticmethod
    def xor_expr(*expressions: Union[str, Expression]) -> Expression:
        """“xor(...)” bitwise expression.

        (Note: For aggregator use, see xor(...) aggregator method above.)

        Args:
            *expressions (str): The expressions to apply bitwise XOR.

        Returns:
            str: A Vespa expression string of the form 'xor(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.xor_expr("fieldA", "fieldB")
            >>> print(expr)
            xor(fieldA, fieldB)
        """
        return Expression(f"xor({', '.join(map(str, expressions))})")

    #
    # String expressions
    #

    @staticmethod
    def strlen(expr: Union[str, Expression]) -> Expression:
        """“strlen(...)” expression. Returns the number of bytes in the string.

        Args:
            expr (str): The string field or expression.

        Returns:
            str: A Vespa expression string of the form 'strlen(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.strlen("my_string_field")
            >>> print(expr)
            strlen(my_string_field)
        """
        return Expression(f"strlen({expr})")

    @staticmethod
    def strcat(*expressions: Union[str, Expression]) -> Expression:
        """“strcat(...)” expression. Concatenate all string arguments in order.

        Args:
            *expressions (str): The string expressions to concatenate.

        Returns:
            str: A Vespa expression string of the form 'strcat(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.strcat("fieldA", "_", "fieldB")
            >>> print(expr)
            strcat(fieldA,_,fieldB)
        """
        return Expression(f"strcat({','.join(map(str, expressions))})")

    #
    # Type conversion expressions
    #

    @staticmethod
    def todouble(expr: Union[str, Expression]) -> Expression:
        """“todouble(...)” expression. Convert argument to double.

        Args:
            expr (str): The expression or field to convert.

        Returns:
            str: A Vespa expression string of the form 'todouble(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.todouble("my_field")
            >>> print(expr)
            todouble(my_field)
        """
        return Expression(f"todouble({expr})")

    @staticmethod
    def tolong(expr: Union[str, Expression]) -> Expression:
        """“tolong(...)” expression. Convert argument to long.

        Args:
            expr (str): The expression or field to convert.

        Returns:
            str: A Vespa expression string of the form 'tolong(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.tolong("my_field")
            >>> print(expr)
            tolong(my_field)
        """
        return Expression(f"tolong({expr})")

    @staticmethod
    def tostring(expr: Union[str, Expression]) -> Expression:
        """“tostring(...)” expression. Convert argument to string.

        Args:
            expr (str): The expression or field to convert.

        Returns:
            str: A Vespa expression string of the form 'tostring(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.tostring("my_field")
            >>> print(expr)
            tostring(my_field)
        """
        return Expression(f"tostring({expr})")

    @staticmethod
    def toraw(expr: Union[str, Expression]) -> Expression:
        """“toraw(...)” expression. Convert argument to raw data.

        Args:
            expr (str): The expression or field to convert.

        Returns:
            str: A Vespa expression string of the form 'toraw(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.toraw("my_field")
            >>> print(expr)
            toraw(my_field)
        """
        return Expression(f"toraw({expr})")

    #
    # Raw data expressions
    #

    @staticmethod
    def cat(*expressions: Union[str, Expression]) -> Expression:
        """“cat(...)” expression. Concatenate the binary representation of arguments.

        Args:
            *expressions (str): The binary expressions or fields to concatenate.

        Returns:
            str: A Vespa expression string of the form 'cat(expr1, expr2, ...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.cat("fieldA", "fieldB")
            >>> print(expr)
            cat(fieldA,fieldB)
        """
        return Expression(f"cat({','.join(map(str, expressions))})")

    @staticmethod
    def md5(expr: Union[str, Expression], width: int) -> Expression:
        """“md5(...)” expression.

        Does an MD5 over the binary representation of the argument,
        and keeps the lowest 'width' bits.

        Args:
            expr (str): The expression or field to apply MD5 on.
            width (int): The number of bits to keep.

        Returns:
            str: A Vespa expression string of the form 'md5(expr, width)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.md5("my_field", 16)
            >>> print(expr)
            md5(my_field, 16)
        """
        return Expression(f"md5({expr}, {width})")

    @staticmethod
    def xorbit(expr: Union[str, Expression], width: int) -> Expression:
        """“xorbit(...)” expression.

        Performs an XOR of 'width' bits over the binary representation of the argument.
        Width is rounded up to a multiple of 8.

        Args:
            expr (str): The expression or field to apply xorbit on.
            width (int): The number of bits for the XOR operation.

        Returns:
            str: A Vespa expression string of the form 'xorbit(expr, width)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.xorbit("my_field", 16)
            >>> print(expr)
            xorbit(my_field, 16)
        """
        return Expression(f"xorbit({expr}, {width})")

    #
    # Accessor expressions
    #

    @staticmethod
    def relevance() -> Expression:
        """“relevance()” expression. Returns the computed rank (relevance) of a document.

        Returns:
            str: 'relevance()' as a Vespa expression string.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.relevance()
            >>> print(expr)
            relevance()
        """
        return Expression("relevance()")

    @staticmethod
    def array_at(
        array_name: str, index_expr: Union[str, int, Expression]
    ) -> Expression:
        """“array.at(...)” accessor expression.
        Returns a single element from the array at the given index.

        Args:
            array_name (str): The name of the array.
            index_expr (Union[str, int]): The index or expression that evaluates to an index.

        Returns:
            str: A Vespa expression string of the form 'array.at(array_name, index)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.array_at("my_array", 0)
            >>> print(expr)
            array.at(my_array, 0)
        """
        return Expression(f"array.at({array_name}, {index_expr})")

    #
    # zcurve decoding expressions
    #

    @staticmethod
    def zcurve_x(expr: Union[str, Expression]) -> Expression:
        """“zcurve.x(...)” expression. Returns the X component of the given zcurve-encoded 2D point.

        Args:
            expr (str): The zcurve-encoded field or expression.

        Returns:
            str: A Vespa expression string of the form 'zcurve.x(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.zcurve_x("location_zcurve")
            >>> print(expr)
            zcurve.x(location_zcurve)
        """
        return Expression(f"zcurve.x({expr})")

    @staticmethod
    def zcurve_y(expr: Union[str, Expression]) -> Expression:
        """“zcurve.y(...)” expression. Returns the Y component of the given zcurve-encoded 2D point.

        Args:
            expr (str): The zcurve-encoded field or expression.

        Returns:
            str: A Vespa expression string of the form 'zcurve.y(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.zcurve_y("location_zcurve")
            >>> print(expr)
            zcurve.y(location_zcurve)
        """
        return Expression(f"zcurve.y({expr})")

    #
    # Time-based expressions
    #

    @staticmethod
    def time_dayofmonth(expr: Union[str, Expression]) -> Expression:
        """“time.dayofmonth(...)” expression. Returns the day of month (1-31).

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.dayofmonth(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_dayofmonth("timestamp_field")
            >>> print(expr)
            time.dayofmonth(timestamp_field)
        """
        return Expression(f"time.dayofmonth({expr})")

    @staticmethod
    def time_dayofweek(expr: Union[str, Expression]) -> Expression:
        """“time.dayofweek(...)” expression. Returns the day of week (0-6), Monday = 0.

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.dayofweek(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_dayofweek("timestamp_field")
            >>> print(expr)
            time.dayofweek(timestamp_field)
        """
        return Expression(f"time.dayofweek({expr})")

    @staticmethod
    def time_dayofyear(expr: Union[str, Expression]) -> Expression:
        """“time.dayofyear(...)” expression. Returns the day of year (0-365).

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.dayofyear(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_dayofyear("timestamp_field")
            >>> print(expr)
            time.dayofyear(timestamp_field)
        """
        return Expression(f"time.dayofyear({expr})")

    @staticmethod
    def time_hourofday(expr: Union[str, Expression]) -> Expression:
        """“time.hourofday(...)” expression. Returns the hour of day (0-23).

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.hourofday(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_hourofday("timestamp_field")
            >>> print(expr)
            time.hourofday(timestamp_field)
        """
        return Expression(f"time.hourofday({expr})")

    @staticmethod
    def time_minuteofhour(expr: Union[str, Expression]) -> Expression:
        """“time.minuteofhour(...)” expression. Returns the minute of hour (0-59).

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.minuteofhour(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_minuteofhour("timestamp_field")
            >>> print(expr)
            time.minuteofhour(timestamp_field)
        """
        return Expression(f"time.minuteofhour({expr})")

    @staticmethod
    def time_monthofyear(expr: Union[str, Expression]) -> Expression:
        """“time.monthofyear(...)” expression. Returns the month of year (1-12).

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.monthofyear(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_monthofyear("timestamp_field")
            >>> print(expr)
            time.monthofyear(timestamp_field)
        """
        return Expression(f"time.monthofyear({expr})")

    @staticmethod
    def time_secondofminute(expr: Union[str, Expression]) -> Expression:
        """“time.secondofminute(...)” expression. Returns the second of minute (0-59).

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.secondofminute(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_secondofminute("timestamp_field")
            >>> print(expr)
            time.secondofminute(timestamp_field)
        """
        return Expression(f"time.secondofminute({expr})")

    @staticmethod
    def time_year(expr: Union[str, Expression]) -> Expression:
        """“time.year(...)” expression. Returns the full year (e.g. 2009).

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.year(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_year("timestamp_field")
            >>> print(expr)
            time.year(timestamp_field)
        """
        return Expression(f"time.year({expr})")

    @staticmethod
    def time_date(expr: Union[str, Expression]) -> Expression:
        """“time.date(...)” expression. Returns the date (e.g. 2009-01-10).

        Args:
            expr (str): The timestamp field or expression.

        Returns:
            str: A Vespa expression string of the form 'time.date(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.time_date("timestamp_field")
            >>> print(expr)
            time.date(timestamp_field)
        """
        return Expression(f"time.date({expr})")

    #
    # Math expressions
    #
    @staticmethod
    def math_exp(expr: Union[str, Expression]) -> Expression:
        """“math.exp(...)” expression. Returns e^expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: A Vespa expression string of the form 'math.exp(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_exp("my_field")
            >>> print(expr)
            math.exp(my_field)
        """
        return Expression(f"math.exp({expr})")

    @staticmethod
    def math_log(expr: Union[str, Expression]) -> Expression:
        """“math.log(...)” expression. Returns the natural logarithm of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.log(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_log("my_field")
            >>> print(expr)
            math.log(my_field)
        """
        return Expression(f"math.log({expr})")

    @staticmethod
    def math_log1p(expr: Union[str, Expression]) -> Expression:
        """“math.log1p(...)” expression. Returns the natural logarithm of (1 + expr).

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.log1p(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_log1p("my_field")
            >>> print(expr)
            math.log1p(my_field)
        """
        return Expression(f"math.log1p({expr})")

    @staticmethod
    def math_log10(expr: Union[str, Expression]) -> Expression:
        """“math.log10(...)” expression. Returns the base-10 logarithm of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.log10(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_log10("my_field")
            >>> print(expr)
            math.log10(my_field)
        """
        return Expression(f"math.log10({expr})")

    @staticmethod
    def math_sqrt(expr: Union[str, Expression]) -> Expression:
        """“math.sqrt(...)” expression. Returns the square root of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.sqrt(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_sqrt("my_field")
            >>> print(expr)
            math.sqrt(my_field)
        """
        return Expression(f"math.sqrt({expr})")

    @staticmethod
    def math_cbrt(expr: Union[str, Expression]) -> Expression:
        """“math.cbrt(...)” expression. Returns the cube root of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.cbrt(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_cbrt("my_field")
            >>> print(expr)
            math.cbrt(my_field)
        """
        return Expression(f"math.cbrt({expr})")

    @staticmethod
    def math_sin(expr: Union[str, Expression]) -> Expression:
        """“math.sin(...)” expression. Returns the sine of expr (argument in radians).

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.sin(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_sin("my_field")
            >>> print(expr)
            math.sin(my_field)
        """
        return Expression(f"math.sin({expr})")

    @staticmethod
    def math_cos(expr: Union[str, Expression]) -> Expression:
        """“math.cos(...)” expression. Returns the cosine of expr (argument in radians).

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.cos(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_cos("my_field")
            >>> print(expr)
            math.cos(my_field)
        """
        return Expression(f"math.cos({expr})")

    @staticmethod
    def math_tan(expr: Union[str, Expression]) -> Expression:
        """“math.tan(...)” expression. Returns the tangent of expr (argument in radians).

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.tan(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_tan("my_field")
            >>> print(expr)
            math.tan(my_field)
        """
        return Expression(f"math.tan({expr})")

    @staticmethod
    def math_asin(expr: Union[str, Expression]) -> Expression:
        """“math.asin(...)” expression. Returns the arcsine of expr (in radians).

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.asin(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_asin("my_field")
            >>> print(expr)
            math.asin(my_field)
        """
        return Expression(f"math.asin({expr})")

    @staticmethod
    def math_acos(expr: Union[str, Expression]) -> Expression:
        """“math.acos(...)” expression. Returns the arccosine of expr (in radians).

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.acos(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_acos("my_field")
            >>> print(expr)
            math.acos(my_field)
        """
        return Expression(f"math.acos({expr})")

    @staticmethod
    def math_atan(expr: Union[str, Expression]) -> Expression:
        """“math.atan(...)” expression. Returns the arctangent of expr (in radians).

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.atan(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_atan("my_field")
            >>> print(expr)
            math.atan(my_field)
        """
        return Expression(f"math.atan({expr})")

    @staticmethod
    def math_sinh(expr: Union[str, Expression]) -> Expression:
        """“math.sinh(...)” expression. Returns the hyperbolic sine of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.sinh(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_sinh("my_field")
            >>> print(expr)
            math.sinh(my_field)
        """
        return Expression(f"math.sinh({expr})")

    @staticmethod
    def math_cosh(expr: Union[str, Expression]) -> Expression:
        """“math.cosh(...)” expression. Returns the hyperbolic cosine of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.cosh(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_cosh("my_field")
            >>> print(expr)
            math.cosh(my_field)
        """
        return Expression(f"math.cosh({expr})")

    @staticmethod
    def math_tanh(expr: Union[str, Expression]) -> Expression:
        """“math.tanh(...)” expression. Returns the hyperbolic tangent of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.tanh(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_tanh("my_field")
            >>> print(expr)
            math.tanh(my_field)
        """
        return Expression(f"math.tanh({expr})")

    @staticmethod
    def math_asinh(expr: Union[str, Expression]) -> Expression:
        """“math.asinh(...)” expression. Returns the inverse hyperbolic sine of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.asinh(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_asinh("my_field")
            >>> print(expr)
            math.asinh(my_field)
        """
        return Expression(f"math.asinh({expr})")

    @staticmethod
    def math_acosh(expr: Union[str, Expression]) -> Expression:
        """“math.acosh(...)” expression. Returns the inverse hyperbolic cosine of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.acosh(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_acosh("my_field")
            >>> print(expr)
            math.acosh(my_field)
        """
        return Expression(f"math.acosh({expr})")

    @staticmethod
    def math_atanh(expr: Union[str, Expression]) -> Expression:
        """“math.atanh(...)” expression. Returns the inverse hyperbolic tangent of expr.

        Args:
            expr (str): The expression or field.

        Returns:
            str: 'math.atanh(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_atanh("my_field")
            >>> print(expr)
            math.atanh(my_field)
        """
        return Expression(f"math.atanh({expr})")

    @staticmethod
    def math_pow(
        expr_x: Union[str, Expression], expr_y: Union[str, Expression]
    ) -> Expression:
        """“math.pow(...)” expression. Returns expr_x^expr_y.

        Args:
            expr_x (str): The expression or field for the base.
            expr_y (str): The expression or field for the exponent.

        Returns:
            str: 'math.pow(expr_x, expr_y)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_pow("my_field", "2")
            >>> print(expr)
            math.pow(my_field,2)
        """
        return Expression(f"math.pow({expr_x},{expr_y})")

    @staticmethod
    def math_hypot(
        expr_x: Union[str, Expression], expr_y: Union[str, Expression]
    ) -> Expression:
        """“math.hypot(...)” expression. Returns the length of the hypotenuse
        given expr_x and expr_y.

        Args:
            expr_x (str): The expression or field for the first side of the triangle.
            expr_y (str): The expression or field for the second side of the triangle.

        Returns:
            str: 'math.hypot(expr_x, expr_y)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.math_hypot("my_field_x", "my_field_y")
            >>> print(expr)
            math.hypot(my_field_x, my_field_y)
        """
        return Expression(f"math.hypot({expr_x}, {expr_y})")

    #
    # List expressions
    #
    @staticmethod
    def size(expr: Union[str, Expression]) -> Expression:
        """“size(...)” expression. Returns the number of elements if expr is a list;
        otherwise returns 1.

        Args:
            expr (str): The list expression or field.

        Returns:
            str: A Vespa expression string of the form 'size(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.size("my_array")
            >>> print(expr)
            size(my_array)
        """
        return Expression(f"size({expr})")

    @staticmethod
    def sort(expr: Union[str, Expression]) -> Expression:
        """“sort(...)” expression. Sorts the elements of the list argument in ascending order.

        Args:
            expr (str): The list expression or field to sort.

        Returns:
            str: A Vespa expression string of the form 'sort(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.sort("my_array")
            >>> print(expr)
            sort(my_array)
        """
        return Expression(f"sort({expr})")

    @staticmethod
    def reverse(expr: Union[str, Expression]) -> Expression:
        """“reverse(...)” expression. Reverses the elements of the list argument.

        Args:
            expr (str): The list expression or field to reverse.

        Returns:
            str: A Vespa expression string of the form 'reverse(expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.reverse("my_array")
            >>> print(expr)
            reverse(my_array)
        """
        return Expression(f"reverse({expr})")

    #
    # Bucket expressions
    #

    @staticmethod
    def fixedwidth(
        value: Union[str, Expression], bucket_width: Union[int, float]
    ) -> Expression:
        """“fixedwidth(...)” bucket expression. Maps the value of the first argument
        into consecutive buckets whose width is the second argument.

        Args:
            value (str): The field or expression to bucket.
            bucket_width (Union[int, float]): The width of each bucket.

        Returns:
            str: A Vespa expression string of the form 'fixedwidth(value, bucket_width)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.fixedwidth("my_field",10)
            >>> print(expr)
            fixedwidth(my_field,10)
        """
        return Expression(f"fixedwidth({value},{bucket_width})")

    @staticmethod
    def predefined(value: Union[str, Expression], buckets: List[str]) -> Expression:
        """“predefined(...)” bucket expression. Maps the value into the provided list of buckets.

        Each 'bucket' must be a string representing the range, e.g.:
        'bucket(-inf,0)', 'bucket[0,10)', 'bucket[10,inf)', etc.

        Args:
            value (str): The field or expression to bucket.
            buckets (List[str]): A list of bucket definitions.

        Returns:
            str: A Vespa expression string of the form 'predefined(value, ( ... ))'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.predefined("my_field", ["bucket(-inf,0)", "bucket[0,10)", "bucket[10,inf)"])
            >>> print(expr)
            predefined(my_field,bucket(-inf,0),bucket[0,10),bucket[10,inf))
        """
        joined_buckets = ",".join(buckets)
        return Expression(f"predefined({value},{joined_buckets})")

    @staticmethod
    def interpolatedlookup(
        array_attr: Union[str, Expression], lookup_expr: Union[str, Expression]
    ) -> Expression:
        """“interpolatedlookup(...)” expression.
        Counts elements in a sorted array that are less than an expression,
        with linear interpolation if the expression is between element values.

        Args:
            array_attr (str): The sorted array field name.
            lookup_expr (str): The expression or value to lookup.

        Returns:
            str: A Vespa expression string of the form 'interpolatedlookup(array, expr)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.interpolatedlookup("my_sorted_array", "4.2")
            >>> print(expr)
            interpolatedlookup(my_sorted_array, 4.2)
        """
        return Expression(f"interpolatedlookup({array_attr}, {lookup_expr})")

    #
    # Hit aggregator
    #

    @staticmethod
    def summary(summary_class: str = "") -> Expression:
        """“summary(...)” hit aggregator. Produces a summary of the requested summary class.

        If no summary class is specified, “summary()” is used.

        Args:
            summary_class (str, optional): Name of the summary class. Defaults to "".

        Returns:
            str: A Vespa grouping expression string of the form 'summary(...)'.

        Examples:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.summary()
            >>> print(expr)
            summary()

            >>> expr = G.summary("my_summary_class")
            >>> print(expr)
            summary(my_summary_class)
        """
        if summary_class:
            return Expression(f"summary({summary_class})")
        else:
            return Expression("summary()")

    @staticmethod
    def as_(expression: str, label: str) -> str:
        """
        Appends an ' as(label)' part to a grouping block expression.

        Args:
            expression (str): The expression to be labeled.
            label (str): The label to be used.

        Returns:
            str: A Vespa grouping expression string of the form 'expression as(label)'

        Example:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.as_(G.each(G.output(G.count())), "mylabel")
            >>> print(expr)
            each(output(count())) as(mylabel)
        """
        return f"{expression} as({label})"

    @staticmethod
    def alias(alias_name: str, expression: str) -> str:
        """
        Defines an alias(...) grouping syntax. This lets you name an
        expression, so you can reference it later by $alias_name.

        Args:
            alias_name (str): The alias name.
            expression (str): The expression to alias.

        Returns:
            str: A Vespa grouping expression string of the form 'alias(alias_name, expression)'.

        Example:
            >>> from vespa.querybuilder import Grouping as G
            >>> expr = G.alias("my_alias", G.add("fieldA", "fieldB"))
            >>> print(expr)
            alias(my_alias,add(fieldA, fieldB))

        """
        return f"alias({alias_name},{expression})"
