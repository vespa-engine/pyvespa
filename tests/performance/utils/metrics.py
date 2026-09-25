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
    mean_ms: Optional[float] = None
    # Little's law: rps * mean latency = requests actually in flight. Compared
    # with `concurrency` it shows whether this client kept the instance's queue
    # full, which neither its CPU use nor the instance's CPU can tell.
    achieved_in_flight: Optional[float] = None
    # Worker processes the concurrency was spread over (pyvespa lane).
    processes: int = 1
    # Connections per transport the lane opened (concurrency / connections
    # requests multiplexed per connection). Same in both lanes by construction.
    connections: Optional[int] = None
    # Client CPU per request, summed over worker processes. Largely runner-
    # independent, so it is the client-efficiency number to track; None for k6.
    cpu_ms_per_request: Optional[float] = None
    # Share of measured requests answered 429 (backpressure). Both lanes run
    # without retries, so this is the same quantity on both sides.
    rate_limited_rate: Optional[float] = None
    # Validity evidence (see utils/cpu_probes.py): CPU busy fraction of the load
    # generator over the window, and the Vespa instance's node CPU per cluster
    # right after the window. All 0..1, None when not measurable.
    client_cpu_fraction: Optional[float] = None
    server_container_cpu_util: Optional[float] = None
    server_content_cpu_util: Optional[float] = None
    # HTTP status (or "error" for no response) -> count over measured requests
    # in the pyvespa lane, for diagnosing a non-zero error rate.
    status_counts: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class Thresholds:
    max_error_rate: float
    min_token_rps: float
    min_mtls_rps: float
    min_token_rps_ratio: float
    max_token_p95_ratio: float


@dataclass(frozen=True)
class ValidityLimits:
    """When a measurement does not count as a measurement of the instance."""

    # Above this share of 429s the run measured backpressure handling, not
    # capacity: lower the concurrency.
    max_rate_limited_rate: float
    # Above this the load generator was CPU-bound and its numbers are about
    # the runner, not Vespa.
    max_client_cpu_fraction: float
    # Below this the instance was not the bottleneck. 0 disables the check
    # (until the sweep has shown what "saturated" looks like here).
    min_server_container_cpu_util: float
    # Below this share of the configured concurrency actually in flight, the
    # client did not keep the instance's queue full and its rps is its own limit.
    min_in_flight_fraction: float = 0.0


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def assert_measurement_valid(results: List[LaneResult], limits: ValidityLimits):
    """Fail the test when the numbers cannot be about the instance."""
    for r in results:
        in_flight = (
            f"{r.achieved_in_flight:.0f}/{r.concurrency}"
            if r.achieved_in_flight is not None
            else "n/a"
        )
        print(
            f"Validity {r.transport}: 429 rate={r.rate_limited_rate if r.rate_limited_rate is not None else 'n/a'}, "
            f"in flight={in_flight}, "
            f"client cpu={_pct(r.client_cpu_fraction)}, "
            f"server container cpu={_pct(r.server_container_cpu_util)}, "
            f"content cpu={_pct(r.server_content_cpu_util)}"
        )
    for r in results:
        if r.achieved_in_flight is not None and limits.min_in_flight_fraction > 0:
            assert (
                r.achieved_in_flight >= limits.min_in_flight_fraction * r.concurrency
            ), (
                f"{r.transport}: only {r.achieved_in_flight:.0f} of {r.concurrency} "
                f"requests in flight (min {limits.min_in_flight_fraction:.0%}); the "
                "client could not keep the instance's queue full. Result invalid."
            )
        if r.rate_limited_rate is not None:
            assert r.rate_limited_rate <= limits.max_rate_limited_rate, (
                f"{r.transport}: {r.rate_limited_rate:.4f} of requests were 429 "
                f"(max {limits.max_rate_limited_rate}); the instance was overloaded, "
                "not saturated. Lower LoadProfile.concurrency."
            )
        if r.client_cpu_fraction is not None:
            assert r.client_cpu_fraction <= limits.max_client_cpu_fraction, (
                f"{r.transport}: load generator CPU at {_pct(r.client_cpu_fraction)} "
                f"(max {_pct(limits.max_client_cpu_fraction)}); the client, not the "
                "instance, was the bottleneck. Result invalid."
            )
        if (
            limits.min_server_container_cpu_util > 0
            and r.server_container_cpu_util is not None
        ):
            assert (
                r.server_container_cpu_util >= limits.min_server_container_cpu_util
            ), (
                f"{r.transport}: container CPU at {_pct(r.server_container_cpu_util)} "
                f"(min {_pct(limits.min_server_container_cpu_util)}); the instance "
                "was not saturated. Raise LoadProfile.concurrency."
            )


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


def _fmt_cpu(value: Optional[float]) -> str:
    return f", cpu={value:.3f}ms/req" if value is not None else ""


def print_results(token: LaneResult, mtls: LaneResult) -> None:
    print(
        f"\n=== Results: {token.lane}/{token.method}/{token.http} "
        f"(concurrency={token.concurrency}/transport over "
        f"{token.connections} connections, processes={token.processes}) ==="
    )
    print(
        f"Token: {token.rps:.2f} req/s, p95={_fmt_ms(token.p95_ms)}, "
        f"error_rate={token.error_rate:.4f} ({token.requests} reqs)"
        f"{_fmt_cpu(token.cpu_ms_per_request)}"
    )
    print(
        f"mTLS:  {mtls.rps:.2f} req/s, p95={_fmt_ms(mtls.p95_ms)}, "
        f"error_rate={mtls.error_rate:.4f} ({mtls.requests} reqs)"
        f"{_fmt_cpu(mtls.cpu_ms_per_request)}"
    )
    print(f"Token/mTLS ratio: {token.rps / mtls.rps if mtls.rps > 0 else 0:.2f}")
    if token.p50_ms is not None and mtls.p50_ms is not None:
        # The token path's own cost (its auth hop): the one quantity the
        # token/mTLS ratio actually varies with. 10 ms here moves the ratio
        # by ~0.06 at these latencies, so read this, not the ratio.
        print(
            f"Token extra latency: p50 {token.p50_ms - mtls.p50_ms:+.1f} ms, "
            f"p95 {token.p95_ms - mtls.p95_ms:+.1f} ms"
        )
    for r in (token, mtls):
        if r.status_counts and r.error_rate > 0:
            top = sorted(r.status_counts.items(), key=lambda kv: -kv[1])[:6]
            print(f"{r.transport} statuses: " + ", ".join(f"{k}={v}" for k, v in top))


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
