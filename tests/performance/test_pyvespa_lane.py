# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""pyvespa lane: the batch feed APIs in one process, read against the k6 ceiling."""

from dataclasses import replace

import pytest

from utils.pyvespa_lane import client, run_pyvespa
from utils.asserts import assert_token_vs_mtls, print_gzip_effect, print_validity
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
# Large documents, plain and gzipped: floors are set after the first calibration run.
PYVESPA_4K_THRESHOLDS = replace(
    PYVESPA_THRESHOLDS["feed_iterable"], min_token_rps=0, min_mtls_rps=0
)


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


def _measure(endpoints, report_dir, method: str, case=SMALL):
    """One method through the token and the mTLS endpoint, one after the other."""
    profile = endpoints.profile
    name = method + case.suffix
    print(
        f"\n=== Running pyvespa {name}: {case.docs or profile.iterable_docs} docs "
        f"per transport with max_workers={profile.pyvespa_workers}, one transport "
        "at a time ==="
    )
    token, mtls = (
        run_pyvespa(method, app, transport, profile, endpoints.mtls_app, case)
        for transport, app in _clients(endpoints).items()
    )
    write_records([token, mtls], report_dir, f"pyvespa_{name}")
    return token, mtls


@pytest.mark.performance
@pytest.mark.parametrize("method", PYVESPA_METHODS)
def test_pyvespa_token_vs_mtls_performance(
    vespa_cloud_token_endpoints, tmp_path, run_state, method
):
    endpoints = vespa_cloud_token_endpoints
    token, mtls = _measure(endpoints, resolve_report_dir(tmp_path), method)
    assert_token_vs_mtls(token, mtls, PYVESPA_THRESHOLDS[method])
    print_validity([token, mtls])
    k6_first = run_state.get("k6_first")
    if k6_first:
        share = (token.rps + mtls.rps) / sum(r.rps for r in k6_first)
        print(f"{method} delivers {share:.0%} of the k6 ceiling")


@pytest.mark.performance
def test_pyvespa_gzip_4k_performance(vespa_cloud_token_endpoints, tmp_path):
    """The 4 KB document plain and gzipped through feed_iterable, the one batch
    API with a compression parameter, on the same runner and instance."""
    endpoints = vespa_cloud_token_endpoints
    report_dir = resolve_report_dir(tmp_path)
    plain = _measure(endpoints, report_dir, "feed_iterable", LARGE)
    gzipped = _measure(endpoints, report_dir, "feed_iterable", LARGE_GZIP)
    for before, after in zip(plain, gzipped):
        print_gzip_effect(before, after)
    for token, mtls in (plain, gzipped):
        assert_token_vs_mtls(token, mtls, PYVESPA_4K_THRESHOLDS)
        print_validity([token, mtls])
