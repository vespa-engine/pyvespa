# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""k6 concurrency sweep: find where the instance's throughput flattens.

Runs only when PERFORMANCE_K6_SWEEP is set to a comma-separated list of VUs
per transport (workflow_dispatch input `k6_sweep`, e.g. "100,200,400,800").
Each level runs the normal warmup + hold profile and records throughput,
latency, 429 share, runner CPU and instance CPU. Use the table to set
LoadProfile.concurrency just past the knee (total rps flat, latency rising,
429 share still ~0) and VALIDITY.min_server_container_cpu_util from the
container CPU seen there. Informational: no thresholds are asserted.
"""

import os
import shutil
from dataclasses import replace
from typing import List

import pytest

from utils.k6_lane import run_k6
from utils.metrics import LaneResult, resolve_report_dir, write_records
from utils.workloads import PROFILE

SWEEP = os.environ.get("PERFORMANCE_K6_SWEEP", "").strip()
if not SWEEP:
    pytest.skip("PERFORMANCE_K6_SWEEP not set", allow_module_level=True)
if shutil.which("k6") is None:
    pytest.skip("k6 binary not found in PATH", allow_module_level=True)

LEVELS = [int(level) for level in SWEEP.split(",") if level.strip()]


def _pct(value) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _table(results: List[List[LaneResult]]) -> str:
    lines = [
        "## k6 concurrency sweep",
        "",
        "| VUs/transport | token rps | mTLS rps | total rps | token p50/p95 ms "
        "| mTLS p50/p95 ms | 429 share | runner CPU | container CPU | content CPU |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for token, mtls in results:
        lines.append(
            f"| {token.concurrency} | {token.rps:.0f} | {mtls.rps:.0f} "
            f"| {token.rps + mtls.rps:.0f} "
            f"| {token.p50_ms:.0f} / {token.p95_ms:.0f} "
            f"| {mtls.p50_ms:.0f} / {mtls.p95_ms:.0f} "
            f"| {max(token.rate_limited_rate or 0, mtls.rate_limited_rate or 0):.4f} "
            f"| {_pct(token.client_cpu_fraction)} "
            f"| {_pct(token.server_container_cpu_util)} "
            f"| {_pct(token.server_content_cpu_util)} |"
        )
    lines += [
        "",
        "Set `LoadProfile.concurrency` just past the level where total rps stops "
        "growing while latency keeps rising and the 429 share is still ~0, and "
        "`VALIDITY.min_server_container_cpu_util` a little under the container "
        "CPU seen there. Runner CPU must stay well under 70% at that level.",
        "",
    ]
    return "\n".join(lines)


@pytest.mark.performance
def test_k6_concurrency_sweep(vespa_cloud_token_endpoints, tmp_path):
    report_dir = resolve_report_dir(tmp_path)
    results: List[List[LaneResult]] = []
    for vus in LEVELS:
        profile = replace(PROFILE, concurrency=vus)
        token, mtls = run_k6(
            vespa_cloud_token_endpoints,
            profile,
            report_dir / f"k6_sweep_{vus}_summary.json",
            extra_env=None,
        )
        print(
            f"sweep {vus} VUs/transport: token {token.rps:.0f} rps "
            f"(p95 {token.p95_ms:.0f} ms), mTLS {mtls.rps:.0f} rps "
            f"(p95 {mtls.p95_ms:.0f} ms), 429 share "
            f"{max(token.rate_limited_rate or 0, mtls.rate_limited_rate or 0):.4f}, "
            f"runner CPU {_pct(token.client_cpu_fraction)}, container CPU "
            f"{_pct(token.server_container_cpu_util)}"
        )
        results.append([token, mtls])

    write_records([r for pair in results for r in pair], report_dir, "k6_sweep")
    table = _table(results)
    (report_dir / "k6_sweep.md").write_text(table)
    print("\n" + table)
