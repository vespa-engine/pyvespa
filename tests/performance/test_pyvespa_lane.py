# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""pyvespa lane: the batch feed APIs as a user calls them, one process, one
transport at a time. k6 is the instance ceiling; this lane measures how much
of it pyvespa delivers, so only its own floors and error limits are asserted."""

import pytest

from utils.pyvespa_lane import client, run_pyvespa
from utils.metrics import (
    assert_token_vs_mtls,
    print_validity,
    resolve_report_dir,
    write_records,
)
from utils.config import PYVESPA_METHODS, PYVESPA_THRESHOLDS


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
@pytest.mark.parametrize("method", PYVESPA_METHODS)
def test_pyvespa_token_vs_mtls_performance(
    vespa_cloud_token_endpoints, tmp_path, run_state, method
):
    """One batch API through the token and the mTLS endpoint, one after the other."""
    endpoints = vespa_cloud_token_endpoints
    report_dir = resolve_report_dir(tmp_path)
    profile = endpoints.profile
    print(
        f"\n=== Running pyvespa {method}: {profile.iterable_docs} docs per "
        "transport, one transport at a time ==="
    )
    token, mtls = (
        run_pyvespa(method, app, transport, profile, metrics_app=endpoints.mtls_app)
        for transport, app in _clients(endpoints).items()
    )
    write_records([token, mtls], report_dir, f"pyvespa_{method}")
    assert_token_vs_mtls(token, mtls, PYVESPA_THRESHOLDS[method])
    # Evidence only: one Python process is the bottleneck here by design.
    print_validity([token, mtls])
    k6_first = run_state.get("k6_first")
    if k6_first:
        k6_total = sum(r.rps for r in k6_first)
        total = token.rps + mtls.rps
        print(f"{method} delivers {total / k6_total:.0%} of the k6 ceiling")
