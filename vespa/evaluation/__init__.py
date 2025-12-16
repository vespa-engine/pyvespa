# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""
Vespa evaluation module.

This module provides tools for evaluating and benchmarking Vespa applications.
"""

# Import all public symbols from _base module (original evaluation.py content)
from vespa.evaluation._base import *  # noqa: F401, F403

# Lazy import for MTEB-related classes to avoid requiring mteb as a dependency
# when not using MTEB functionality


def __getattr__(name):
    """Lazy import for optional MTEB dependencies."""
    if name in ("VespaMTEBApp", "VespaMTEBEvaluator"):
        from vespa.evaluation._mteb import VespaMTEBApp, VespaMTEBEvaluator

        if name == "VespaMTEBApp":
            return VespaMTEBApp
        return VespaMTEBEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
