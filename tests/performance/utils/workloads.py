# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import random
import string
from dataclasses import dataclass, replace
from typing import Dict, Tuple

from utils.metrics import Thresholds

# Persistent prod performance app, deployed once via
# test_deploy_performance_instance.py.
TENANT = "vespa-team"
APPLICATION = "pyvespa-performance"
INSTANCE = "default"
ENVIRONMENT = "prod"
REGION = "aws-us-east-1c"
SCHEMA = "msmarco"
CONTENT_CLUSTER = "msmarco_content"


def make_doc(prefix: str) -> Tuple[str, Dict]:
    """Return a (doc_id, fields) pair identical in shape to the k6 payload."""
    doc_id = f"{prefix}-" + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=16)
    )
    return doc_id, {"id": doc_id, "title": "performance-doc", "body": "benchmark run"}


@dataclass(frozen=True)
class LoadProfile:
    """Closed-model load: N workers feed as fast as the instance responds."""

    concurrency: int = 200
    warmup_s: float = 30.0
    duration_s: float = 150.0
    # Sized for a stable measurement window (~2 min at observed feed rates).
    iterable_docs: int = 200000
    iterable_warmup_docs: int = 2000
    iterable_max_workers: int = 200
    iterable_max_connections: int = 200
    # HTTP/2 connections for the async paths. pyvespa defaults to 1; a single
    # connection was observed to cap throughput, so let async use several.
    async_connections: int = 8

    def k6_env(self) -> dict:
        """Env vars for k6/token_vs_mtls.js so both lanes share one load shape."""
        return {
            "MAX_VUS": str(self.concurrency),
            "RAMP_UP": f"{int(self.warmup_s)}s",
            "HOLD": f"{int(self.duration_s)}s",
        }


PROFILE = LoadProfile()

# Educated-guess near-optimal concurrency per pyvespa method: sync threads
# degrade past ~64 on a 4-vcpu runner; async in-flight capped near typical
# HTTP/2 per-connection stream limits. k6 stays at PROFILE.concurrency as the
# raw-HTTP ceiling.
METHOD_CONCURRENCY = {
    "sync_feed_data_point": 64,
    "async_feed_data_point": 128,
    "feed_iterable": 64,
    "feed_async_iterable": 128,
}


def method_profile(method: str) -> LoadProfile:
    concurrency = METHOD_CONCURRENCY[method]
    return replace(
        PROFILE,
        concurrency=concurrency,
        iterable_max_workers=concurrency,
        iterable_max_connections=concurrency,
    )


# TODO: k6 floors are established CI baselines; pyvespa floors are deliberately
# loose until a few scheduled runs exist -- tighten them here.
K6_THRESHOLDS = Thresholds(
    max_error_rate=0.02,
    min_token_rps=500,
    min_mtls_rps=750,
    min_token_rps_ratio=0.5,
    max_token_p95_ratio=3.0,
)
PYVESPA_THRESHOLDS = Thresholds(
    max_error_rate=0.05,
    min_token_rps=25,
    min_mtls_rps=25,
    min_token_rps_ratio=0.3,
    max_token_p95_ratio=5.0,
)

# No HTTP-version axis: httpr cannot force HTTP/1.1 nor report the negotiated
# protocol. These labels record each code path's library default.
SYNC_HTTP_MODE = "negotiate"
ASYNC_HTTP_MODE = "h2only"
