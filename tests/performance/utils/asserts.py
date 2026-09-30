# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from typing import List, Optional

from utils.metrics import LaneResult, Thresholds, ValidityLimits


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _fmt_ms(value: Optional[float]) -> str:
    return f"{value:.2f}ms" if value is not None else "n/a"


def _fmt_cpu(value: Optional[float]) -> str:
    return f", cpu={value:.3f}ms/req" if value is not None else ""


def print_results(token: LaneResult, mtls: LaneResult) -> None:
    print(
        f"\n=== Results: {token.lane}/{token.method} "
        f"(concurrency={token.concurrency} per transport) ==="
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
        print(
            f"Token extra latency: p50 {token.p50_ms - mtls.p50_ms:+.1f} ms, "
            f"p95 {token.p95_ms - mtls.p95_ms:+.1f} ms"
        )
    for r in (token, mtls):
        if r.status_counts and r.error_rate > 0:
            top = sorted(r.status_counts.items(), key=lambda kv: -kv[1])[:6]
            print(f"{r.transport} statuses: " + ", ".join(f"{k}={v}" for k, v in top))


def print_gzip_effect(plain: LaneResult, gzipped: LaneResult) -> None:
    """Same document with and without request compression, same transport."""
    cpu = ""
    if plain.cpu_ms_per_request and gzipped.cpu_ms_per_request:
        cpu = f", client cpu/req x{gzipped.cpu_ms_per_request / plain.cpu_ms_per_request:.2f}"
    print(
        f"gzip {plain.transport}: {plain.rps:.0f} -> {gzipped.rps:.0f} rps "
        f"(x{gzipped.rps / plain.rps if plain.rps else 0:.2f}){cpu}"
    )


def print_validity(results: List[LaneResult]) -> None:
    for r in results:
        in_flight = (
            f"{r.achieved_in_flight:.0f}/{r.concurrency}"
            if r.achieved_in_flight is not None
            else "n/a"
        )
        print(
            f"Validity {r.transport}: 429 rate={r.rate_limited_rate if r.rate_limited_rate is not None else 'n/a'}, "
            f"in flight={in_flight}, "
            f"client cpu={_pct(r.client_cpu_fraction)} of the machine"
            f"{f' ({r.client_cpu_cores:.2f} cores)' if r.client_cpu_cores is not None else ''}, "
            f"server container cpu={_pct(r.server_container_cpu_util)}, "
            f"content cpu={_pct(r.server_content_cpu_util)}"
        )


def assert_floor(result: LaneResult, min_rps: float, max_error_rate: float) -> None:
    """The single-transport assert set: error ceiling and throughput floor."""
    print(
        f"\n=== Results: {result.lane}/{result.method} {result.transport}: "
        f"{result.rps:.2f} req/s, error_rate={result.error_rate:.4f} "
        f"({result.requests} reqs){_fmt_cpu(result.cpu_ms_per_request)} ==="
    )
    assert result.error_rate <= max_error_rate, (
        f"Error rate too high ({result.error_rate:.4f}, max={max_error_rate})"
    )
    assert result.rps >= min_rps, (
        f"Throughput too low (got {result.rps:.2f} req/s, expected >={min_rps} req/s)"
    )


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


def assert_measurement_valid(results: List[LaneResult], limits: ValidityLimits):
    """Fail the test when the numbers cannot be about the instance."""
    print_validity(results)
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
