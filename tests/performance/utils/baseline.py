# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""k6 floors from recent CI runs. Vespa Cloud moves the instance to newer
hardware on its own, which moves the k6 ceiling, so a fixed floor goes stale.
Each floor is a share of the median of the same record in the last runs, never
below the fixed minimum in the test file. The workflow downloads those runs'
reports into PERFORMANCE_BASELINE_DIR; without them the fixed minimum applies."""

import json
import os
import statistics
from pathlib import Path
from typing import List

BASELINE_SHARE = 0.75  # explained under "Thresholds" in tests/performance/README.md
MIN_BASELINE_RUNS = 3


def recent_rps(name: str, transport: str) -> List[float]:
    """rps of record `name` and `transport` in each downloaded run."""
    root = os.environ.get("PERFORMANCE_BASELINE_DIR")
    if not root:
        return []
    values = []
    for records_file in sorted(Path(root).glob(f"*/{name}_records.json")):
        try:
            records = json.loads(records_file.read_text())["records"]
        except (OSError, ValueError, KeyError):
            continue
        values += [r["rps"] for r in records if r.get("transport") == transport]
    return values


def floor(name: str, transport: str, minimum: float) -> float:
    values = recent_rps(name, transport)
    if len(values) < MIN_BASELINE_RUNS:
        print(
            f"Floor {name} {transport}: {minimum:.0f} rps (fixed, {len(values)} runs)"
        )
        return minimum
    median = statistics.median(values)
    value = max(minimum, BASELINE_SHARE * median)
    print(
        f"Floor {name} {transport}: {value:.0f} rps ({BASELINE_SHARE:.0%} of the "
        f"median {median:.0f} of {len(values)} runs, minimum {minimum:.0f})"
    )
    return value
