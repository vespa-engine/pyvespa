# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import random
import string
from dataclasses import dataclass, replace
from typing import Dict, Tuple

from utils.metrics import ValidityLimits

TENANT = "vespa-team"
APPLICATION = "pyvespa-performance"
INSTANCE = "default"
ENVIRONMENT = "prod"
REGION = "aws-us-east-1c"
SCHEMA = "msmarco"
CONTENT_CLUSTER = "msmarco_content"
CONTAINER_CLUSTER = "msmarco_container"


def make_doc(prefix: str) -> Tuple[str, Dict]:
    doc_id = f"{prefix}-" + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=16)
    )
    return doc_id, {"id": doc_id, "title": "performance-doc", "body": "benchmark run"}


@dataclass(frozen=True)
class LoadProfile:
    # The values are explained under "Load settings" in tests/performance/README.md.
    concurrency: int = 400
    server_queue_target: int = 200
    max_concurrency: int = 800
    warmup_s: float = 30.0
    duration_s: float = 150.0
    k6_connections: int = 8
    iterable_docs: int = 400000
    iterable_warmup_docs: int = 2000

    def k6_env(self) -> dict:
        return {
            "MAX_VUS": str(self.concurrency),
            "RAMP_UP": f"{int(self.warmup_s)}s",
            "HOLD": f"{int(self.duration_s)}s",
            "CONNECTIONS": str(self.k6_connections),
        }

    def for_session(self, ceiling_rps: float, rtt_s: float) -> "LoadProfile":
        if ceiling_rps <= 0 or rtt_s <= 0:
            return self
        in_flight = self.server_queue_target + ceiling_rps * rtt_s
        step = self.k6_connections
        concurrency = max(step, round(in_flight / step) * step)
        return replace(self, concurrency=min(concurrency, self.max_concurrency))


# Every value below is explained under "Settings" in tests/performance/README.md.

# Session (conftest.py): wait for the instance to settle, clean up documents.
IDLE_CPU_UTIL = 0.30
CLEANUP_SLICES = 16

# k6 lane (test_k6_lane.py): the instance ceiling and the token hop.
PROFILE = LoadProfile()  # opening and closing runs, sized by for_session
WARMUP = replace(PROFILE, concurrency=400, warmup_s=15.0, duration_s=45.0)
LATENCY_PROBE = replace(
    PROFILE, concurrency=1, k6_connections=1, warmup_s=5.0, duration_s=30.0
)
# Asserts that a k6 number is about the instance at all: not overloaded (429s),
# queue kept full (in flight), runner not CPU-bound, instance saturated.
VALIDITY = ValidityLimits(
    max_rate_limited_rate=0.01,
    max_client_cpu_fraction=0.90,
    min_server_container_cpu_util=0.75,
    min_in_flight_fraction=0.85,
)

# pyvespa lane (test_pyvespa_lane.py): the batch APIs in one process.
PYVESPA_METHODS = ("feed_iterable", "feed_async_iterable")
FEED_ITERABLE_PARAMETERS = dict(max_workers=128)
FEED_ASYNC_ITERABLE_PARAMETERS = dict(max_workers=400, max_connections=4)
