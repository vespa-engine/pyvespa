# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import os
import subprocess
from pathlib import Path
import shutil

import pytest

from utils.metrics import (
    LaneResult,
    assert_token_vs_mtls,
    resolve_report_dir,
    write_records,
)
from utils.workloads import K6_THRESHOLDS, PROFILE


if shutil.which("k6") is None:
    pytest.skip("k6 binary not found in PATH", allow_module_level=True)


def _require_metric_key(metrics: dict, key: str) -> dict:
    if key in metrics:
        return metrics[key]
    pytest.fail(
        f"Missing metric '{key}' in k6 summary. Available keys: {list(metrics.keys())}"
    )


def _metric_value(metric: dict, field: str):
    if field in metric:
        return metric[field]
    return metric.get("values", {}).get(field)


def _require_value(metric: dict, fields: tuple, label: str, metrics: dict) -> float:
    for field in fields:
        value = _metric_value(metric, field)
        if value is not None:
            return value
    dump = json.dumps(metric, indent=2)
    pytest.fail(
        f"Missing {fields} for {label}.\n\n"
        f"Metric dump:\n{dump}\n\n"
        f"Available metric keys:\n{list(metrics.keys())}"
    )


def _lane_result(metrics: dict, transport: str, concurrency: int) -> LaneResult:
    duration = _require_metric_key(metrics, f"{transport}_req_duration")
    fail = _require_metric_key(metrics, f"{transport}_fail_rate")
    reqs = _require_metric_key(metrics, f"{transport}_reqs")

    # k6 Rate metrics export the rate under "value"; keep "rate" as fallback.
    error_rate = _require_value(fail, ("value", "rate"), transport, metrics)
    rps = _require_value(reqs, ("rate",), transport, metrics)
    count = int(_require_value(reqs, ("count",), transport, metrics))

    return LaneResult(
        lane="k6",
        method="http_post",
        transport=transport,
        http="negotiate",
        rps=rps,
        error_rate=error_rate,
        requests=count,
        duration_s=count / rps if rps > 0 else 0.0,
        concurrency=concurrency,
        p50_ms=_metric_value(duration, "med"),
        p95_ms=_require_value(duration, ("p(95)",), transport, metrics),
        p99_ms=_metric_value(duration, "p(99)"),
    )


@pytest.mark.performance
def test_token_vs_mtls_performance(vespa_cloud_token_endpoints, tmp_path):
    """Run k6 against token and mTLS endpoints and assert thresholds and relative latency."""

    report_dir = resolve_report_dir(tmp_path)
    summary_file = report_dir / "k6_token_vs_mtls_summary.json"
    script = Path(__file__).parent / "k6" / "token_vs_mtls.js"

    env = {
        **os.environ,
        "TOKEN_URL": vespa_cloud_token_endpoints.token_url,
        "MTLS_URL": vespa_cloud_token_endpoints.mtls_url,
        "TOKEN_AUTH_HEADER": f"Bearer {vespa_cloud_token_endpoints.token}",
        "MTLS_CERT_PATH": vespa_cloud_token_endpoints.cert_path,
        "MTLS_KEY_PATH": vespa_cloud_token_endpoints.key_path,
        **PROFILE.k6_env(),
    }

    k6_command = [
        "k6",
        "run",
        "--summary-export",
        str(summary_file),
    ]
    if os.environ.get("CI"):
        k6_command.append("--quiet")
    k6_command.append(str(script))

    expected_s = int(PROFILE.warmup_s + PROFILE.duration_s)
    print(f"\n=== Running k6: {script.name} (~{expected_s}s + graceful stop) ===")

    try:
        result = subprocess.run(
            k6_command,
            env=env,
            capture_output=False,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("k6 binary not found in PATH")

    assert result.returncode == 0, "k6 run failed (see output above)"

    metrics = json.loads(summary_file.read_text()).get("metrics", {})
    token = _lane_result(metrics, "token", PROFILE.concurrency)
    mtls = _lane_result(metrics, "mtls", PROFILE.concurrency)

    write_records([token, mtls], report_dir, "k6_token_vs_mtls")
    assert_token_vs_mtls(token, mtls, K6_THRESHOLDS)
