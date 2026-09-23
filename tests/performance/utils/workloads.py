# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import random
import string
from dataclasses import dataclass, replace
from typing import Dict, Tuple

from utils.metrics import Thresholds, ValidityLimits

# Persistent prod performance app, deployed once via
# test_deploy_performance_instance.py.
TENANT = "vespa-team"
APPLICATION = "pyvespa-performance"
INSTANCE = "default"
ENVIRONMENT = "prod"
REGION = "aws-us-east-1c"
SCHEMA = "msmarco"
CONTENT_CLUSTER = "msmarco_content"
CONTAINER_CLUSTER = "msmarco_container"


def make_doc(prefix: str) -> Tuple[str, Dict]:
    """Return a (doc_id, fields) pair identical in shape to the k6 payload."""
    doc_id = f"{prefix}-" + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=16)
    )
    return doc_id, {"id": doc_id, "title": "performance-doc", "body": "benchmark run"}


@dataclass(frozen=True)
class LoadProfile:
    """Closed-model load: N workers per transport feed as fast as the instance
    responds. The same `concurrency` drives k6 (VUs per transport) and pyvespa
    (in-flight requests per transport, spread over `processes` worker
    processes), so both lanes put identical load on the instance."""

    # In-flight requests per transport, same for k6 and pyvespa. This is the
    # default (used for the warmup and when the session cannot measure); the
    # session fixture derives the effective value with for_session() below.
    # k6 sweep of 2026-09-22 from Europe (RTT ~130 ms): 100/200/400/800 per
    # transport gave 1121 / 2284 / 3517 / 3383 total rps at 42 / 70 / 94 / 95 %
    # container CPU, so 400 sat just past the knee there.
    concurrency: int = 400
    # What the instance actually feels is the number of requests queued inside
    # it, not the client's in-flight count: at ceiling X and network RTT r,
    # N in flight means about N - X * r queued (the rest are on the wire). The
    # same N therefore overloads from a 50 ms runner (CI run #32: 429s at 400
    # per transport from us-east) and under-loads from 130 ms away. The session
    # picks N so that the queued count is this target: N = target + X * r.
    server_queue_target: int = 250
    max_concurrency: int = 800
    warmup_s: float = 30.0
    duration_s: float = 150.0
    # Worker processes per transport for the pyvespa lane, and therefore the
    # number of connections per transport in both lanes (see connections()).
    # Fixed rather than cpu_count so a 4-vCPU runner and a laptop generate the
    # same shape (8 connections x 50 streams); the client CPU guard catches a
    # runner that cannot afford it. One Python process (one GIL) tops out
    # around 2000 feed requests/s, below the instance ceiling.
    processes: int = 8
    # Sized so the batch methods run about as long as the closed loops (~160 s
    # at ~1900 rps per transport). Not longer: ~6 min of sustained ~3800 rps
    # filled this instance's document/v1 queue (content node persistence) and
    # produced 3% 429s and halved throughput, a different regime from the
    # 3-minute ceiling the closed loops measure.
    iterable_docs: int = 300000
    iterable_warmup_docs: int = 2000
    # `connections` passed to the async client per worker process. httpr
    # (reqwest) multiplexes every request to a host over one HTTP/2 connection
    # regardless (verified with lsof: 50 threads on one VespaSync = 1
    # connection, one VespaSync per thread = 50), so each pyvespa worker
    # process is one connection with concurrency / processes streams, and k6
    # is run in the same shape (STREAMS_PER_CONNECTION). Connection count
    # matters to the instance: 400 single-stream connections per transport
    # gave ~3400-3750 rps at 94% container CPU, 8 multiplexed ones ~4400.
    async_connections: int = 1

    def k6_env(self) -> dict:
        """Env vars for k6/token_vs_mtls.js so both lanes share one load shape."""
        return {
            "MAX_VUS": str(self.concurrency),
            "RAMP_UP": f"{int(self.warmup_s)}s",
            "HOLD": f"{int(self.duration_s)}s",
            # Same topology as the pyvespa lane: `processes` connections per
            # transport, each multiplexing concurrency / processes requests.
            "STREAMS_PER_CONNECTION": str(self.streams_per_connection()),
        }

    def streams_per_connection(self) -> int:
        return max(1, self.concurrency // max(1, self.processes))

    def connections(self) -> int:
        """Connections per transport both lanes open (verified: one shared
        httpr client multiplexes all its requests over one HTTP/2 connection)."""
        return max(1, self.concurrency // self.streams_per_connection())

    def for_session(self, ceiling_rps: float, rtt_s: float) -> "LoadProfile":
        """Concurrency that puts `server_queue_target` requests inside the
        instance given its measured ceiling and this client's network RTT,
        split over the two transports and rounded to whole processes."""
        if ceiling_rps <= 0 or rtt_s <= 0:
            return self
        total = self.server_queue_target + ceiling_rps * rtt_s
        per_transport = total / 2
        step = max(1, self.processes)
        per_transport = max(step, round(per_transport / step) * step)
        return replace(self, concurrency=min(per_transport, self.max_concurrency))

    def per_process(self) -> "LoadProfile":
        """The share of this profile one worker process runs."""
        n = max(1, self.processes)
        return replace(
            self,
            concurrency=max(1, self.concurrency // n),
            processes=1,
            iterable_docs=self.iterable_docs // n,
            iterable_warmup_docs=max(1, self.iterable_warmup_docs // n),
        )


PROFILE = LoadProfile()

# Session warmup before the first measured test: same concurrency, shorter.
WARMUP = replace(PROFILE, warmup_s=15.0, duration_s=45.0)

PYVESPA_METHODS = (
    "sync_feed_data_point",
    "async_feed_data_point",
    "feed_iterable",
    "feed_async_iterable",
)


# A run only counts as a measurement of the instance when the load generator
# had headroom, the instance answered (almost) no 429s, and, once the sweep has
# shown what saturation looks like here, the container was busy. Applied to
# both lanes identically (utils/metrics.py::assert_measurement_valid).
VALIDITY = ValidityLimits(
    max_rate_limited_rate=0.01,
    max_client_cpu_fraction=0.70,
    # Sweep of 2026-09-22: 94% container CPU at concurrency 400, 70% at 200
    # (where throughput was only 65% of the ceiling). The probe is the metrics
    # proxy's ~60 s snapshot, so a ~150 s run yields two or three samples and
    # saturated runs read anywhere from 83% to 96%; 0.75 separates saturated
    # from not without failing on probe granularity.
    min_server_container_cpu_util=0.75,
)

# Second, probe-independent saturation check for the pyvespa lane: its total
# rps must reach this share of the opening k6 run's total in the same session.
# A client-bound path shows up here (the per-thread-client sync worker gave
# 0.72) even when the CPU probe is inconclusive.
MIN_PYVESPA_VS_K6_RATIO = 0.8

# A test starts only when every node's cpu_util is at or below this (or after
# a bounded wait), so the previous test's tail or cleanup does not bleed in.
IDLE_CPU_UTIL = 0.30

# Slices for the setup/teardown delete_all_docs: ~2M documents are fed per run
# and a single slice deletes them at ~2800 docs/s (12 min); slices run in
# parallel against the content cluster.
CLEANUP_SLICES = 8

# Floors sit ~30% under the lowest values seen with the final, fair shape
# (2026-09-22, concurrency 400 over 8 connections per transport, warmup,
# container at 87-95%): k6 token/mTLS 2108/2222 (two runs, drift -2.9%);
# pyvespa sync 2181/2267, async 2159/2350, feed_iterable 1702/1924,
# feed_async_iterable 1512/1706 (the last two from a session whose closing k6
# run had already degraded after per-test deletes, since dropped). Token/mTLS
# rps ratio 0.88-0.96, token/mTLS p95 ratio 1.1-1.2. With the instance as the
# bottleneck these do not depend on the runner; a miss means the instance got
# slower or a client path regressed (then the validity checks say which).
# The token endpoint adds latency (an extra hop before the container; CI run
# #32 from us-east: token p50 209 ms vs mTLS 143 ms), so in a closed loop the
# token transport gets fewer requests through: rps ratio 0.5-0.6 and p95 ratio
# 2-3 from the US, 0.9 and 1.2 from Europe where the network dominates. The
# ratios are sanity bounds only; the per-transport floors carry the gate.
# Floors ~30% under CI run #34 (2026-09-23, derived concurrency 240 per
# transport from us-east): token 2019-2143, mTLS 2271-2388 rps on all six
# tests, k6 and pyvespa within 1.5% of each other, drift -1.6%.
K6_THRESHOLDS = Thresholds(
    max_error_rate=0.02,
    min_token_rps=1400,
    min_mtls_rps=1600,
    min_token_rps_ratio=0.4,
    max_token_p95_ratio=4.0,
)
_PYVESPA_FLOOR = Thresholds(
    max_error_rate=0.02,
    min_token_rps=1400,
    min_mtls_rps=1600,
    min_token_rps_ratio=0.4,
    max_token_p95_ratio=4.0,
)
PYVESPA_THRESHOLDS = {method: _PYVESPA_FLOOR for method in PYVESPA_METHODS}

# No HTTP-version axis: httpr cannot force HTTP/1.1 nor report the negotiated
# protocol. These labels record each code path's library default. Verified
# 2026-09-22: httpr negotiates HTTP/2 in both modes (ALPN) and k6 reports
# HTTP/2.0 against the endpoint, so both lanes run HTTP/2.
SYNC_HTTP_MODE = "negotiate"
ASYNC_HTTP_MODE = "h2only"
