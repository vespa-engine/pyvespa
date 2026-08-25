# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LaneResult:
    lane: str  # "k6" | "pyvespa"
    method: str  # e.g. "http_post", "sync_feed_data_point", "feed_iterable"
    transport: str  # "token" | "mtls"
    http: str  # "negotiate" | "h2only"
    rps: float
    error_rate: float
    requests: int
    duration_s: float
    concurrency: int
    # Per-request latency is only available where the harness times each
    # request itself; feed_iterable/feed_async_iterable own their loops, so
    # these stay None there.
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None


@dataclass(frozen=True)
class Thresholds:
    max_error_rate: float
    min_token_rps: float
    min_mtls_rps: float
    min_token_rps_ratio: float
    max_token_p95_ratio: float


def percentiles(latencies_ms: List[float]) -> Tuple[float, float, float]:
    """Return (p50, p95, p99) of the given latencies in milliseconds."""
    if not latencies_ms:
        raise ValueError("No latency samples collected.")
    ordered = sorted(latencies_ms)

    def pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
        return ordered[index]

    return pct(0.50), pct(0.95), pct(0.99)


def resolve_report_dir(fallback: Path) -> Path:
    report_dir = Path(os.environ.get("PERFORMANCE_REPORT_DIR") or fallback)
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def write_records(results: List[LaneResult], report_dir: Path, name: str) -> Path:
    """Write results as {name}_records.json for the Prometheus converter."""
    out = report_dir / f"{name}_records.json"
    out.write_text(json.dumps({"records": [asdict(r) for r in results]}, indent=2))
    return out


def _fmt_ms(value: Optional[float]) -> str:
    return f"{value:.2f}ms" if value is not None else "n/a"


def print_results(token: LaneResult, mtls: LaneResult) -> None:
    print(f"\n=== Results: {token.lane}/{token.method}/{token.http} ===")
    print(
        f"Token: {token.rps:.2f} req/s, p95={_fmt_ms(token.p95_ms)}, "
        f"error_rate={token.error_rate:.4f} ({token.requests} reqs)"
    )
    print(
        f"mTLS:  {mtls.rps:.2f} req/s, p95={_fmt_ms(mtls.p95_ms)}, "
        f"error_rate={mtls.error_rate:.4f} ({mtls.requests} reqs)"
    )
    print(f"Token/mTLS ratio: {token.rps / mtls.rps if mtls.rps > 0 else 0:.2f}")


def assert_token_vs_mtls(
    token: LaneResult, mtls: LaneResult, thresholds: Thresholds
) -> None:
    """The one assert set both lanes run: error ceiling, throughput floors,
    token-vs-mTLS throughput ratio, and (when measured) relative p95."""
    print_results(token, mtls)

    assert (
        token.error_rate <= thresholds.max_error_rate
        and mtls.error_rate <= thresholds.max_error_rate
    ), (
        "Error rate too high "
        f"(token error rate={token.error_rate:.4f}, mtls error rate={mtls.error_rate:.4f}, "
        f"max={thresholds.max_error_rate})"
    )
    assert token.rps >= thresholds.min_token_rps, (
        f"Token throughput too low (got {token.rps:.2f} req/s, "
        f"expected >={thresholds.min_token_rps} req/s)"
    )
    assert mtls.rps >= thresholds.min_mtls_rps, (
        f"mTLS throughput too low (got {mtls.rps:.2f} req/s, "
        f"expected >={thresholds.min_mtls_rps} req/s)"
    )
    assert token.rps >= thresholds.min_token_rps_ratio * mtls.rps, (
        "Token throughput too low relative to mTLS "
        f"(token rps={token.rps:.2f}, mTLS rps={mtls.rps:.2f}, "
        f"ratio={token.rps / mtls.rps if mtls.rps > 0 else 0:.2f}, "
        f"min ratio={thresholds.min_token_rps_ratio})"
    )
    if token.p95_ms is not None and mtls.p95_ms is not None:
        assert token.p95_ms <= thresholds.max_token_p95_ratio * mtls.p95_ms, (
            "Token endpoint too slow relative to mTLS "
            f"(token p95={token.p95_ms} ms, mTLS p95={mtls.p95_ms} ms, "
            f"max ratio={thresholds.max_token_p95_ratio})"
        )


def summarize_samples(samples: List[Dict]) -> Dict:
    """Reduce closed-loop samples [{latency_ms, ok}] to aggregate fields."""
    requests = len(samples)
    errors = sum(1 for s in samples if not s["ok"])
    latencies = [s["latency_ms"] for s in samples]
    p50, p95, p99 = percentiles(latencies)
    return {
        "requests": requests,
        "error_rate": errors / requests if requests else 1.0,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }
