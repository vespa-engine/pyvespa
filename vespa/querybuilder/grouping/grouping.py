from typing import Union


class Grouping:
    @staticmethod
    def all(*args) -> str:
        return "all(" + " ".join(args) + ")"

    @staticmethod
    def group(field: str) -> str:
        return f"group({field})"

    @staticmethod
    def max(value: Union[int, float]) -> str:
        return f"max({value})"

    @staticmethod
    def precision(value: int) -> str:
        return f"precision({value})"

    @staticmethod
    def min(value: Union[int, float]) -> str:
        return f"min({value})"

    @staticmethod
    def sum(value: Union[int, float]) -> str:
        return f"sum({value})"

    @staticmethod
    def avg(value: Union[int, float]) -> str:
        return f"avg({value})"

    @staticmethod
    def stddev(value: Union[int, float]) -> str:
        return f"stddev({value})"

    @staticmethod
    def xor(value: str) -> str:
        return f"xor({value})"

    @staticmethod
    def each(*args) -> str:
        return "each(" + " ".join(args) + ")"

    @staticmethod
    def output(output_func: str) -> str:
        return f"output({output_func})"

    @staticmethod
    def count() -> str:
        # Also need to handle negative count
        class MaybenegativeCount(str):
            def __new__(cls, value):
                return super().__new__(cls, value)

            def __neg__(self):
                return f"-{self}"

        return MaybenegativeCount("count()")

    @staticmethod
    def order(value: str) -> str:
        return f"order({value})"

    @staticmethod
    def summary() -> str:
        return "summary()"
