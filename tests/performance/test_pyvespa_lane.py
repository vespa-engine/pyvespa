# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""pyvespa lane: the batch feed APIs in one process, read against the k6 ceiling."""

from dataclasses import replace

import pytest

from utils.pyvespa_lane import client, run_pyvespa
from utils.asserts import assert_token_vs_mtls, print_gzip_effect, print_validity
from utils.metrics import Thresholds, resolve_report_dir, write_records
from utils.config import PYVESPA_RUNS

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
PYVESPA_THRESHOLDS["feed_iterable_4k"] = replace(
    PYVESPA_THRESHOLDS["feed_iterable"], min_token_rps=0, min_mtls_rps=0
)
PYVESPA_THRESHOLDS["feed_iterable_4k_gzip"] = PYVESPA_THRESHOLDS["feed_iterable_4k"]


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


@pytest.mark.performance
@pytest.mark.parametrize(
    "method,case", PYVESPA_RUNS, ids=[m + c.suffix for m, c in PYVESPA_RUNS]
)
def test_pyvespa_token_vs_mtls_performance(
    vespa_cloud_token_endpoints, tmp_path, run_state, method, case
):
    """One batch API through the token and the mTLS endpoint, one after the other."""
    endpoints = vespa_cloud_token_endpoints
    report_dir = resolve_report_dir(tmp_path)
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
    run_state[name] = (token, mtls)
    assert_token_vs_mtls(token, mtls, PYVESPA_THRESHOLDS[name])
    print_validity([token, mtls])
    if case.gzip and name.removesuffix("_gzip") in run_state:
        for plain, gzipped in zip(run_state[name.removesuffix("_gzip")], (token, mtls)):
            print_gzip_effect(plain, gzipped)
    k6_first = run_state.get("k6_first")
    if k6_first and not case.body_bytes:
        k6_total = sum(r.rps for r in k6_first)
        total = token.rps + mtls.rps
        print(f"{name} delivers {total / k6_total:.0%} of the k6 ceiling")
