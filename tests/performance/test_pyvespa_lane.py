# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""pyvespa lane: the batch feed APIs in one process, read against the k6 ceiling."""

import pytest

from utils.pyvespa_lane import client, run_pyvespa
from utils.asserts import (
    assert_floor,
    assert_token_vs_mtls,
    print_gzip_effect,
    print_validity,
)
from utils.metrics import Thresholds, resolve_report_dir, write_records
from utils.config import LARGE, LARGE_GZIP, PYVESPA_METHODS, SMALL

# Explained under "Thresholds" in tests/performance/README.md.
PYVESPA_THRESHOLDS = {
    "feed_iterable": Thresholds(
        max_error_rate=0.02,
        min_token_rps=1950,
        min_mtls_rps=2150,
        min_token_rps_ratio=0.4,
        max_token_p95_ratio=4.0,
    ),
    "feed_async_iterable": Thresholds(
        max_error_rate=0.02,
        min_token_rps=1550,
        min_mtls_rps=1950,
        min_token_rps_ratio=0.4,
        max_token_p95_ratio=4.0,
    ),
}
PYVESPA_4K_MIN_MTLS_RPS = 1100  # the 4 KB document, plain and gzipped, on mTLS


def _clients(endpoints) -> dict:
    return {
        "token": client("token", endpoints.token_url, token=endpoints.token),
        "mtls": client(
            "mtls",
            endpoints.mtls_url,
            cert_path=endpoints.cert_path,
            key_path=endpoints.key_path,
        ),
    }


def _measure(
    endpoints, report_dir, method: str, case=SMALL, transports=("token", "mtls")
):
    """One method through each transport, one transport at a time."""
    profile = endpoints.profile_for(case)
    name = method + case.suffix
    print(
        f"\n=== Running pyvespa {name}: {case.docs or profile.iterable_docs} docs "
        f"per transport with max_workers={profile.pyvespa_workers}, one transport "
        "at a time ==="
    )
    clients = _clients(endpoints)
    results = [
        run_pyvespa(
            method, clients[transport], transport, profile, endpoints.mtls_app, case
        )
        for transport in transports
    ]
    write_records(results, report_dir, f"pyvespa_{name}")
    return results


@pytest.mark.performance
@pytest.mark.parametrize("method", PYVESPA_METHODS)
def test_pyvespa_token_vs_mtls_performance(
    vespa_cloud_token_endpoints, tmp_path, run_state, method
):
    endpoints = vespa_cloud_token_endpoints
    token, mtls = _measure(endpoints, resolve_report_dir(tmp_path), method)
    floors = PYVESPA_THRESHOLDS[method].scaled(endpoints.runner_speed)
    assert_token_vs_mtls(token, mtls, floors)
    print_validity([token, mtls])
    k6_first = run_state.get("k6_first")
    if k6_first:
        share = (token.rps + mtls.rps) / sum(r.rps for r in k6_first)
        print(f"{method} delivers {share:.0%} of the k6 ceiling")


@pytest.mark.performance
def test_pyvespa_gzip_4k_performance(vespa_cloud_token_endpoints, tmp_path):
    """The 4 KB document plain and gzipped through feed_iterable, the one batch
    API with a compression parameter, on mTLS, the standard transport."""
    endpoints = vespa_cloud_token_endpoints
    report_dir = resolve_report_dir(tmp_path)
    (plain,) = _measure(endpoints, report_dir, "feed_iterable", LARGE, ("mtls",))
    (gzipped,) = _measure(endpoints, report_dir, "feed_iterable", LARGE_GZIP, ("mtls",))
    print_gzip_effect(plain, gzipped)
    # Only down: on a fast runner the 4 KB batch hits the instance's ceiling first.
    floor = PYVESPA_4K_MIN_MTLS_RPS * min(1.0, endpoints.runner_speed)
    for result in (plain, gzipped):
        assert_floor(
            result,
            floor,
            PYVESPA_THRESHOLDS["feed_iterable"].max_error_rate,
        )
        print_validity([result])
