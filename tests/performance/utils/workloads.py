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
    """In-flight load for the one transport under test; each process owns one
    HTTP/2 connection. Token and mTLS run one after the other."""

    concurrency: int = 400  # per transport; warmup/fallback, for_session adjusts
    server_queue_target: int = 200  # 250 sat at the 429 edge from a 56 ms runner
    max_concurrency: int = 800
    warmup_s: float = 30.0
    duration_s: float = 150.0
    processes: int = 8  # avoid a single Python process becoming the bottleneck
    iterable_docs: int = 300000  # roughly the same duration as the closed loops
    iterable_warmup_docs: int = 2000
    async_connections: int = 1

    def k6_env(self) -> dict:
        return {
            "MAX_VUS": str(self.concurrency),
            "RAMP_UP": f"{int(self.warmup_s)}s",
            "HOLD": f"{int(self.duration_s)}s",
            "STREAMS_PER_CONNECTION": str(self.streams_per_connection()),
        }

    def streams_per_connection(self) -> int:
        return max(1, self.concurrency // max(1, self.processes))

    def connections(self) -> int:
        return max(1, self.concurrency // self.streams_per_connection())

    def for_session(self, ceiling_rps: float, rtt_s: float) -> "LoadProfile":
        """In flight = queued requests + throughput * RTT; transports run one at
        a time, so this is the concurrency of the single active transport."""
        if ceiling_rps <= 0 or rtt_s <= 0:
            return self
        in_flight = self.server_queue_target + ceiling_rps * rtt_s
        step = max(1, self.processes)
        concurrency = max(step, round(in_flight / step) * step)
        return replace(self, concurrency=min(concurrency, self.max_concurrency))

    def per_process(self) -> "LoadProfile":
        n = max(1, self.processes)
        return replace(
            self,
            concurrency=max(1, self.concurrency // n),
            processes=1,
            iterable_docs=self.iterable_docs // n,
            iterable_warmup_docs=max(1, self.iterable_warmup_docs // n),
        )


PROFILE = LoadProfile()
# mTLS only, 60 s: warms the instance and gives a conservative ceiling estimate
# (400 in flight on one transport is under the 429 edge from us-east).
WARMUP = replace(PROFILE, concurrency=400, warmup_s=15.0, duration_s=45.0)
PYVESPA_METHODS = (
    "sync_feed_data_point",
    "async_feed_data_point",
    "feed_iterable",
    "feed_async_iterable",
)

# Reject overload, a CPU-bound runner, or an underloaded instance.
VALIDITY = ValidityLimits(
    max_rate_limited_rate=0.01,
    max_client_cpu_fraction=0.70,
    min_server_container_cpu_util=0.75,
)
MIN_PYVESPA_VS_K6_RATIO = 0.8
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

# httpr negotiates HTTP/2 in both modes; these labels record the API defaults.
SYNC_HTTP_MODE = "negotiate"
ASYNC_HTTP_MODE = "h2only"
