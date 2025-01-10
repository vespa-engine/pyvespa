from .builder.builder import Q, Query, QueryField
from .grouping.grouping import Grouping
import inspect

# Automatically expose all static methods from Q
for cls in [Q]:  # do not expose G for now
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("_"):
            # Create function with same name and signature as the static method
            globals()[name] = method


def get_function_members(cls):
    return [
        name
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]


# Create __all__ list
__all__ = [
    # Classes
    "Query",
    "QueryField",
    "Grouping",
    # "Condition",
    # Add all exposed functions
    *get_function_members(Q),
]
