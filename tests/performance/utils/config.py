# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import math
import random
import string
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

from utils.metrics import ValidityLimits

TENANT = "vespa-team"
APPLICATION = "pyvespa-performance"
INSTANCE = "default"
ENVIRONMENT = "prod"
REGION = "aws-us-east-1c"
SCHEMA = "msmarco"
CONTENT_CLUSTER = "msmarco_content"
CONTAINER_CLUSTER = "msmarco_container"


# The same text feeds both lanes, so a large document compresses identically in k6 and pyvespa.
DOCUMENTS = Path(__file__).parent.parent / "documents"
BODY_SMALL = (DOCUMENTS / "body_small.txt").read_text()
BODY_4K = (DOCUMENTS / "body_4k.txt").read_text()


def make_doc(prefix: str, body: str = BODY_SMALL) -> Tuple[str, Dict]:
    doc_id = f"{prefix}-" + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=16)
    )
    return doc_id, {"id": doc_id, "title": "performance-doc", "body": body}


@dataclass(frozen=True)
class FeedCase:
    """What each request carries: the small document, or a large one, gzipped or not."""

    body_bytes: int = 0  # 0 is the small benchmark document
    gzip: bool = False
    docs: Optional[int] = None  # pyvespa batch size; None is LoadProfile.iterable_docs

    @property
    def suffix(self) -> str:
        size = f"_{self.body_bytes // 1024}k" if self.body_bytes else ""
        return size + ("_gzip" if self.gzip else "")

    @property
    def body(self) -> str:
        return BODY_4K[: self.body_bytes] if self.body_bytes else BODY_SMALL


SMALL = FeedCase()
LARGE = FeedCase(body_bytes=4096, docs=100000)
LARGE_GZIP = replace(LARGE, gzip=True)


@dataclass(frozen=True)
class LoadProfile:
    # The values are explained under "Load settings" in tests/performance/README.md.
    concurrency: int = 400
    server_queue_target: int = 200
    max_concurrency: int = 800
    warmup_s: float = 30.0
    duration_s: float = 90.0
    k6_connections: int = 8
    token_streams_per_connection: int = 14
    pyvespa_workers: int = 128
    max_pyvespa_workers: int = 1024
    service_s: float = 0.02
    iterable_docs: int = 200000
    iterable_warmup_docs: int = 2000

    def connections(self, transport: str) -> int:
        # Token gains from more connections, mTLS loses (see README).
        if transport != "token":
            return self.k6_connections
        per = self.token_streams_per_connection
        return max(self.k6_connections, round(self.concurrency / per))

    def k6_env(self, transport: str) -> dict:
        return {
            "MAX_VUS": str(self.concurrency),
            "RAMP_UP": f"{int(self.warmup_s)}s",
            "HOLD": f"{int(self.duration_s)}s",
            "CONNECTIONS": str(self.connections(transport)),
        }

    def for_session(self, ceiling_rps: float, rtt_s: float) -> "LoadProfile":
        if ceiling_rps <= 0 or rtt_s <= 0:
            return self
        in_flight = self.server_queue_target + ceiling_rps * rtt_s
        step = self.k6_connections
        concurrency = max(step, round(in_flight / step) * step)
        workers = math.ceil(ceiling_rps * (rtt_s + self.service_s) / 16) * 16
        return replace(
            self,
            concurrency=min(concurrency, self.max_concurrency),
            pyvespa_workers=min(max(workers, 64), self.max_pyvespa_workers),
        )


# Every value below is explained under "Settings" in tests/performance/README.md.

# Session (conftest.py): wait for the instance to settle, clean up documents.
IDLE_CPU_UTIL = 0.30
CLEANUP_SLICES = 16

# k6 lane (test_k6_lane.py): the instance ceiling and the token hop.
PROFILE = LoadProfile()  # opening and closing runs, sized by for_session
WARMUP = replace(PROFILE, concurrency=400, warmup_s=15.0, duration_s=45.0)
# Sizes the 4 KB runs from the 4 KB ceiling (see README).
WARMUP_LARGE = replace(PROFILE, concurrency=256, warmup_s=10.0, duration_s=30.0)
PROFILE_LARGE = replace(PROFILE, service_s=0.05)
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

# pyvespa lane (test_pyvespa_lane.py): the batch APIs in one process, with
# max_workers from PROFILE.pyvespa_workers and everything else the library default.
PYVESPA_METHODS = ("feed_iterable", "feed_async_iterable")
# pyvespa rps relative to a near runner, by RTT in ms, measured with added
# latency on one runner; the floors are scaled by it (see README).
PYVESPA_RTT_FACTORS = {
    "mtls": ((24.0, 1.0), (52.0, 0.83), (84.0, 0.72)),
    "token": ((24.0, 1.0), (52.0, 0.74), (84.0, 0.54)),
}


def rtt_factor(transport: str, rtt_ms: float) -> float:
    points = PYVESPA_RTT_FACTORS[transport]
    if rtt_ms <= points[0][0]:
        return 1.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if rtt_ms <= x1:
            return y0 + (y1 - y0) * (rtt_ms - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return max(0.3, y1 + (y1 - y0) * (rtt_ms - x1) / (x1 - x0))
