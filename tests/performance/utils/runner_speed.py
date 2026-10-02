# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Times the runner's Python speed so the pyvespa floors can follow it."""

import json
import time
from typing import Optional

from utils.config import make_doc

_DOCS = 50000
_REPEATS = 5


def benchmark_s() -> float:
    best = float("inf")
    for _ in range(_REPEATS):
        started = time.perf_counter()
        for _ in range(_DOCS):
            doc_id, fields = make_doc("bench")
            body = json.dumps({"fields": fields})
            json.loads(body)
            json.loads(json.dumps({"pathId": f"/document/v1/{doc_id}", "id": doc_id}))
        best = min(best, time.perf_counter() - started)
    return best


def speed(measured_s: float, reference_s: Optional[float]) -> float:
    if not reference_s or measured_s <= 0:
        return 1.0
    return reference_s / measured_s
