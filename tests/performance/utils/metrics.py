# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LaneResult:
    """One lane's measurement of one transport. Every field is explained under
    "Metrics reference" in tests/performance/README.md; None means not measured."""

    lane: str
    method: str
    transport: str
    rps: float
    error_rate: float
    requests: int
    duration_s: float
    concurrency: int
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    mean_ms: Optional[float] = None
    achieved_in_flight: Optional[float] = None
    cpu_ms_per_request: Optional[float] = None
    rate_limited_rate: Optional[float] = None
    client_cpu_fraction: Optional[float] = None
    client_cpu_cores: Optional[float] = None
    server_container_cpu_util: Optional[float] = None
    server_content_cpu_util: Optional[float] = None
    status_counts: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class Thresholds:
    """Performance floors and bounds a test fails on."""

    max_error_rate: float
    min_token_rps: float
    min_mtls_rps: float
    min_token_rps_ratio: float
    max_token_p95_ratio: float


@dataclass(frozen=True)
class ValidityLimits:
    """When a measurement does not count as a measurement of the instance.
    0 disables a minimum."""

    max_rate_limited_rate: float
    max_client_cpu_fraction: float
    min_server_container_cpu_util: float
    min_in_flight_fraction: float = 0.0


def resolve_report_dir(fallback: Path) -> Path:
    report_dir = Path(os.environ.get("PERFORMANCE_REPORT_DIR") or fallback)
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def write_records(results: List[LaneResult], report_dir: Path, name: str) -> Path:
    """Write results as {name}_records.json for the Prometheus converter."""
    out = report_dir / f"{name}_records.json"
    out.write_text(json.dumps({"records": [asdict(r) for r in results]}, indent=2))
    return out
