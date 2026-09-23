# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""pyvespa lane: the real pyvespa feed code paths under the same load shape as
k6. Each method runs in PROFILE.processes worker processes per transport (see
utils/loadgen.py), token and mTLS concurrently, with retries and compression
off, so the instance rather than the Python client is the bottleneck and a 429
means the same thing in both lanes."""

import pytest

from utils.loadgen import Target, run_method
from utils.metrics import (
    assert_measurement_valid,
    assert_token_vs_mtls,
    resolve_report_dir,
    write_records,
)
from utils.workloads import MIN_PYVESPA_VS_K6_RATIO, PYVESPA_THRESHOLDS, VALIDITY


def _targets(endpoints) -> list:
    return [
        Target(transport="token", url=endpoints.token_url, token=endpoints.token),
        Target(
            transport="mtls",
            url=endpoints.mtls_url,
            cert_path=endpoints.cert_path,
            key_path=endpoints.key_path,
        ),
    ]


def _run_pair(endpoints, report_dir, method: str, run_state: dict) -> None:
    profile = endpoints.profile
    share = profile.per_process()
    expected_s = int(profile.warmup_s + profile.duration_s)
    print(
        f"\n=== Running pyvespa {method} "
        f"(concurrency={profile.concurrency} per transport as "
        f"{profile.processes} processes x {share.concurrency}, "
        f"token+mtls concurrently, ~{expected_s}s) ==="
    )
    token, mtls = run_method(
        method, _targets(endpoints), profile, metrics_app=endpoints.mtls_app
    )
    write_records([token, mtls], report_dir, f"pyvespa_{method}")
    assert_token_vs_mtls(token, mtls, PYVESPA_THRESHOLDS[method])
    assert_measurement_valid([token, mtls], VALIDITY)
    k6_first = run_state.get("k6_first")
    if not k6_first:
        print("No opening k6 run in this session; pyvespa-vs-k6 ratio not checked.")
        return
    total = token.rps + mtls.rps
    k6_total = sum(r.rps for r in k6_first)
    print(
        f"pyvespa/k6 total rps: {total:.0f} / {k6_total:.0f} = {total / k6_total:.2f}"
    )
    assert total >= MIN_PYVESPA_VS_K6_RATIO * k6_total, (
        f"{method}: {total:.0f} rps is below {MIN_PYVESPA_VS_K6_RATIO:.0%} of the k6 "
        f"ceiling {k6_total:.0f} rps measured in this session; the client path, not "
        "the instance, limited throughput."
    )


@pytest.mark.performance
def test_sync_feed_data_point(vespa_cloud_token_endpoints, tmp_path, run_state):
    _run_pair(
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "sync_feed_data_point",
        run_state,
    )


@pytest.mark.performance
def test_async_feed_data_point(vespa_cloud_token_endpoints, tmp_path, run_state):
    _run_pair(
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "async_feed_data_point",
        run_state,
    )


@pytest.mark.performance
def test_feed_iterable(vespa_cloud_token_endpoints, tmp_path, run_state):
    _run_pair(
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "feed_iterable",
        run_state,
    )


@pytest.mark.performance
def test_feed_async_iterable(vespa_cloud_token_endpoints, tmp_path, run_state):
    _run_pair(
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "feed_async_iterable",
        run_state,
    )
