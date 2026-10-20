# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""k6 floors from the recent CI runs in PERFORMANCE_BASELINE_DIR, so they
follow the instance's hardware. See "Thresholds" in tests/performance/README.md."""

import json
import os
import statistics
from pathlib import Path
from typing import List

BASELINE_SHARE = 0.75
MIN_BASELINE_RUNS = 3


def recent_rps(name: str, transport: str) -> List[float]:
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
