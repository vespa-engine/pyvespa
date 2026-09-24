# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""k6 lane: raw HTTP at PROFILE.concurrency VUs per transport. Runs twice per
session, first and last (`perf_last`), so the pyvespa methods are bracketed
and the instance's drift within the run is measured rather than guessed."""

import json
import shutil
from typing import List

import pytest

from utils.k6_lane import run_k6
from utils.metrics import (
    LaneResult,
    assert_measurement_valid,
    assert_token_vs_mtls,
    resolve_report_dir,
    write_records,
)
from utils.workloads import THRESHOLDS, VALIDITY


if shutil.which("k6") is None:
    pytest.skip("k6 binary not found in PATH", allow_module_level=True)


def _measure(endpoints, report_dir, name: str) -> List[LaneResult]:
    token, mtls = run_k6(
        endpoints,
        endpoints.profile,
        report_dir / f"{name}_summary.json",
    )
    write_records([token, mtls], report_dir, name)
    return [token, mtls]


def _check(results: List[LaneResult]) -> None:
    token, mtls = results
    assert_token_vs_mtls(token, mtls, THRESHOLDS)
    assert_measurement_valid([token, mtls], VALIDITY)


@pytest.mark.performance
def test_token_vs_mtls_performance(vespa_cloud_token_endpoints, tmp_path, run_state):
    """Opening k6 run: the instance ceiling before the pyvespa methods."""
    results = _measure(
        vespa_cloud_token_endpoints, resolve_report_dir(tmp_path), "k6_token_vs_mtls"
    )
    run_state["k6_first"] = results
    _check(results)


@pytest.mark.performance
@pytest.mark.perf_last
def test_token_vs_mtls_performance_last(
    vespa_cloud_token_endpoints, tmp_path, run_state
):
    """Closing k6 run. The difference to the opening run is how much the
    instance itself moved during the session: read every pyvespa-vs-k6 gap
    smaller than that as noise."""
    report_dir = resolve_report_dir(tmp_path)
    last = _measure(vespa_cloud_token_endpoints, report_dir, "k6_token_vs_mtls_last")
    first = run_state.get("k6_first")
    if first:
        # Drift is recorded before any assertion so the artifact has it even
        # when a threshold fails.
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
