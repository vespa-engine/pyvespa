# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import random
import string
from dataclasses import dataclass, replace
from typing import Dict, Tuple

from utils.metrics import Thresholds, ValidityLimits

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
    """k6 load for the one transport under test; token and mTLS run one after
    the other. The pyvespa lane takes its concurrency from the API knobs below."""

    concurrency: int = 400  # per transport; warmup/fallback, for_session adjusts
    server_queue_target: int = 200  # 250 sat at the 429 edge from a 56 ms runner
    max_concurrency: int = 800
    warmup_s: float = 30.0
    duration_s: float = 150.0
    k6_connections: int = 8  # HTTP/2 connections the k6 streams are spread over
    iterable_docs: int = 150000  # one batch; sized to take about duration_s
    iterable_warmup_docs: int = 2000

    def k6_env(self) -> dict:
        return {
            "MAX_VUS": str(self.concurrency),
            "RAMP_UP": f"{int(self.warmup_s)}s",
            "HOLD": f"{int(self.duration_s)}s",
            "CONNECTIONS": str(self.k6_connections),
        }

    def for_session(self, ceiling_rps: float, rtt_s: float) -> "LoadProfile":
        """In flight = queued requests + throughput * RTT; transports run one at
        a time, so this is the concurrency of the single active transport."""
        if ceiling_rps <= 0 or rtt_s <= 0:
            return self
        in_flight = self.server_queue_target + ceiling_rps * rtt_s
        step = self.k6_connections
        concurrency = max(step, round(in_flight / step) * step)
        return replace(self, concurrency=min(concurrency, self.max_concurrency))


PROFILE = LoadProfile()
# mTLS only, 60 s: warms the instance and gives a conservative ceiling estimate
# (400 in flight on one transport is under the 429 edge from us-east).
WARMUP = replace(PROFILE, concurrency=400, warmup_s=15.0, duration_s=45.0)

# The batch APIs as a user calls them: one process, one call, these knobs.
# Values are the library defaults until the local sweep (README) says otherwise.
FEED_ITERABLE_KNOBS = dict(max_workers=8, max_connections=16, max_queue_size=1000)
FEED_ASYNC_ITERABLE_KNOBS = dict(max_workers=64, max_connections=1, max_queue_size=1000)
PYVESPA_METHODS = ("feed_iterable", "feed_async_iterable")

# k6 only: reject overload, a client that did not keep the queue full, a
# CPU-bound runner, or an underloaded instance. In-flight (Little's law) is the
# direct client check; the CPU fraction is a backstop.
VALIDITY = ValidityLimits(
    max_rate_limited_rate=0.01,
    max_client_cpu_fraction=0.90,
    min_server_container_cpu_util=0.75,
    min_in_flight_fraction=0.85,
)
IDLE_CPU_UTIL = 0.30
CLEANUP_SLICES = 8

# About 30% below CI run #34 (2026-09-23): token 2019+, mTLS 2271+ rps.
# Ratio bounds are loose because the token path's extra latency varies with RTT.
THRESHOLDS = Thresholds(
    max_error_rate=0.02,
    min_token_rps=1400,
    min_mtls_rps=1600,
    min_token_rps_ratio=0.4,
    max_token_p95_ratio=4.0,
)
# pyvespa floors per method; set about 30% below the first CI run of this lane.
PYVESPA_THRESHOLDS = {
    "feed_iterable": replace(THRESHOLDS, min_token_rps=0, min_mtls_rps=0),
    "feed_async_iterable": replace(THRESHOLDS, min_token_rps=0, min_mtls_rps=0),
}
