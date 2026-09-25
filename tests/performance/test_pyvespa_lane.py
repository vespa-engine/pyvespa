# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Compare four pyvespa feed paths against the opening k6 baseline."""

import pytest

from utils.pyvespa_lane import Target, run_pyvespa
from utils.metrics import (
    assert_measurement_valid,
    assert_token_vs_mtls,
    resolve_report_dir,
    write_records,
)
from utils.config import (
    MIN_PYVESPA_VS_K6_RATIO,
    PYVESPA_METHODS,
    THRESHOLDS,
    VALIDITY,
)


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


@pytest.mark.performance
@pytest.mark.parametrize("method", PYVESPA_METHODS)
def test_pyvespa_performance(vespa_cloud_token_endpoints, tmp_path, run_state, method):
    endpoints = vespa_cloud_token_endpoints
    report_dir = resolve_report_dir(tmp_path)
    profile = endpoints.profile
    share = profile.per_process()
    expected_s = int(profile.warmup_s + profile.duration_s)
    print(
        f"\n=== Running pyvespa {method}: {profile.concurrency} in flight as "
        f"{profile.processes} processes x {share.concurrency}, one transport at a "
        f"time, ~{expected_s}s each ==="
    )
    token, mtls = (
        run_pyvespa(method, [target], profile, metrics_app=endpoints.mtls_app)[0]
        for target in _targets(endpoints)
    )
    write_records([token, mtls], report_dir, f"pyvespa_{method}")
    assert_token_vs_mtls(token, mtls, THRESHOLDS)
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
