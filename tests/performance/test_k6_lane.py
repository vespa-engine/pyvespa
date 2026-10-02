# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
from typing import List

import pytest

from utils.k6_lane import run_k6
from utils.asserts import (
    assert_floor,
    assert_measurement_valid,
    assert_token_vs_mtls,
    print_gzip_effect,
)
from utils.metrics import LaneResult, Thresholds, resolve_report_dir, write_records
from utils.config import LARGE, LARGE_GZIP, LATENCY_PROBE, SMALL, VALIDITY

# Explained under "Thresholds" in tests/performance/README.md.
K6_THRESHOLDS = Thresholds(
    max_error_rate=0.02,
    min_token_rps=1400,
    min_mtls_rps=1600,
    min_token_rps_ratio=0.4,
    max_token_p95_ratio=4.0,
)
MAX_TOKEN_HOP_MS = 50.0
K6_4K_MIN_MTLS_RPS = 1050  # the 4 KB document, plain and gzipped, on mTLS


def _measure(
    endpoints,
    report_dir,
    name: str,
    profile=None,
    case=SMALL,
    transports=("token", "mtls"),
) -> List[LaneResult]:
    results = [
        run_k6(
            endpoints,
            profile or endpoints.profile_for(case),
            report_dir / f"{name}_{transport}_summary.json",
            transport,
            case,
        )
        for transport in transports
    ]
    write_records(results, report_dir, name)
    return results


def _check(results: List[LaneResult]) -> None:
    token, mtls = results
    assert_token_vs_mtls(token, mtls, K6_THRESHOLDS)
    assert_measurement_valid([token, mtls], VALIDITY)


@pytest.mark.performance
def test_token_hop_latency(vespa_cloud_token_endpoints, tmp_path):
    """Token p50 minus mTLS p50 at one request in flight: the token hop, no queueing."""
    token, mtls = _measure(
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "k6_token_hop",
        LATENCY_PROBE,
    )
    assert token.p50_ms is not None and mtls.p50_ms is not None, "no latency samples"
    hop_ms = token.p50_ms - mtls.p50_ms
    print(
        f"\n=== Token hop: p50 token {token.p50_ms:.1f} ms, mTLS {mtls.p50_ms:.1f} ms, "
        f"difference {hop_ms:+.1f} ms ==="
    )
    assert max(token.error_rate, mtls.error_rate) <= K6_THRESHOLDS.max_error_rate
    assert hop_ms <= MAX_TOKEN_HOP_MS, (
        f"Token path adds {hop_ms:.1f} ms per request at one in flight "
        f"(max {MAX_TOKEN_HOP_MS:.0f} ms)"
    )


@pytest.mark.performance
def test_token_vs_mtls_performance(vespa_cloud_token_endpoints, tmp_path, run_state):
    """Opening k6 run: the instance ceiling before the pyvespa methods."""
    results = _measure(
        vespa_cloud_token_endpoints, resolve_report_dir(tmp_path), "k6_token_vs_mtls"
    )
    run_state["k6_first"] = results
    _check(results)


@pytest.mark.performance
def test_gzip_4k_performance(vespa_cloud_token_endpoints, tmp_path):
    """The 4 KB document plain and gzipped on mTLS, the standard transport, so
    compression's cost or gain on the wire shows against the same instance."""
    endpoints = vespa_cloud_token_endpoints
    report_dir = resolve_report_dir(tmp_path)
    (plain,) = _measure(
        endpoints, report_dir, "k6_4k", case=LARGE, transports=("mtls",)
    )
    (gzipped,) = _measure(
        endpoints, report_dir, "k6_4k_gzip", case=LARGE_GZIP, transports=("mtls",)
    )
    print_gzip_effect(plain, gzipped)
    for result in (plain, gzipped):
        assert_floor(result, K6_4K_MIN_MTLS_RPS, K6_THRESHOLDS.max_error_rate)
        assert_measurement_valid([result], VALIDITY)


@pytest.mark.performance
@pytest.mark.performance_last
def test_token_vs_mtls_performance_last(
    vespa_cloud_token_endpoints, tmp_path, run_state
):
    """Closing k6 run; the difference to the opening run is the instance's drift."""
    report_dir = resolve_report_dir(tmp_path)
    last = _measure(vespa_cloud_token_endpoints, report_dir, "k6_token_vs_mtls_last")
    first = run_state.get("k6_first")
    if first and sum(r.rps for r in first) > 0:
        # Written before the asserts so the artifact has it when a threshold fails.
        first_total = sum(r.rps for r in first)
        last_total = sum(r.rps for r in last)
        drift_pct = (last_total - first_total) / first_total * 100
        print(
            f"\n=== Instance drift over the session: k6 total {first_total:.0f} -> "
            f"{last_total:.0f} rps ({drift_pct:+.1f}%) ==="
        )
        (report_dir / "k6_drift.json").write_text(
            json.dumps(
                {
                    "first_total_rps": first_total,
                    "last_total_rps": last_total,
                    "drift_pct": drift_pct,
                },
                indent=2,
            )
        )
    else:
        print("No opening k6 run in this session; drift not measured.")
    _check(last)
