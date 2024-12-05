from .builder.builder import Query, Q, Queryfield, Condition
from .grouping.grouping import G
import inspect

# Import original classes
# ...existing code...

# Automatically expose all static methods from Q and G classes
for cls in [Q, G]:
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("_"):
            # Create function with same name and signature as the static method
            globals()[name] = method

# Create __all__ list dynamically
__all__ = [
    # Classes
    "Query",
    "Q",
    "Queryfield",
    "G",
    "Condition",
    # Add all exposed functions
    *(
        name
        for name, method in inspect.getmembers(Q, predicate=inspect.isfunction)
        if not name.startswith("_")
    ),
    *(
        name
        for name, method in inspect.getmembers(G, predicate=inspect.isfunction)
        if not name.startswith("_")
    ),
]
