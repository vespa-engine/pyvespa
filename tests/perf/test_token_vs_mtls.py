# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import os
import subprocess
from pathlib import Path
import shutil

import pytest


if shutil.which("k6") is None:
    pytest.skip("k6 binary not found in PATH", allow_module_level=True)


@pytest.mark.perf
def test_token_vs_mtls_perf(vespa_cloud_token_endpoints, tmp_path):
    """Run k6 against token and mTLS endpoints and assert thresholds and relative latency."""

    summary_file = tmp_path / "k6_summary.json"
    script = Path(__file__).parent / "k6_token_vs_mtls.js"

    env = {
        **os.environ,
        "TOKEN_URL": vespa_cloud_token_endpoints["token_url"],
        "MTLS_URL": vespa_cloud_token_endpoints["mtls_url"],
        "TOKEN_AUTH_HEADER": f"Bearer {vespa_cloud_token_endpoints['token']}",
        "MTLS_CERT_PATH": vespa_cloud_token_endpoints["cert_path"],
        "MTLS_KEY_PATH": vespa_cloud_token_endpoints["key_path"],
        "VUS": os.getenv("K6_VUS", "50"),
        "DURATION": os.getenv("K6_DURATION", "60s"),
    }
    min_token_rps_ratio = float(os.getenv("K6_TOKEN_RPS_RATIO", "0.7"))

    try:
        result = subprocess.run(
            [
                "k6",
                "run",
                "--summary-export",
                str(summary_file),
                "--summary-trend-stats=min,avg,med,p(95),p(99),max",
                str(script),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("k6 binary not found in PATH")

    if result.stdout:
        print("\n=== k6 stdout ===\n", result.stdout)
    if result.stderr:
        print("\n=== k6 stderr ===\n", result.stderr)

    assert result.returncode == 0, (
        "k6 thresholds failed\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )

    summary = json.loads(summary_file.read_text())
    metrics = summary.get("metrics", {})

    def require_metric_key(key: str) -> dict:
        if key in metrics:
            return metrics[key]
        pytest.fail(
            f"Missing metric '{key}' in k6 summary. Available keys: {list(metrics.keys())}"
        )

    def metric_value(metric: dict, field: str):
        if field in metric:
            return metric[field]
        return metric.get("values", {}).get(field)

    mtls_duration = require_metric_key("mtls_req_duration")
    token_duration = require_metric_key("token_req_duration")
    mtls_fail = require_metric_key("mtls_fail_rate")
    token_fail = require_metric_key("token_fail_rate")
    mtls_reqs = require_metric_key("mtls_reqs")
    token_reqs = require_metric_key("token_reqs")

    def p95(metric: dict, label: str) -> float:
        value = metric_value(metric, "p(95)")
        if value is None:
            dump = json.dumps(metric, indent=2)
            pytest.fail(
                f"Missing p(95) for {label}.\n\n"
                f"Metric dump:\n{dump}\n\n"
                f"Available metric keys:\n{list(metrics.keys())}"
            )
        return value

    def fail_rate(metric: dict, label: str) -> float:
        # k6 Rate metrics export the rate under "value"; keep "rate" as fallback for safety.
        rate = metric_value(metric, "value")
        if rate is None:
            rate = metric_value(metric, "rate")
        if rate is None:
            dump = json.dumps(metric, indent=2)
            pytest.fail(
                f"Missing fail rate for {label}.\n\n"
                f"Metric dump:\n{dump}\n\n"
                f"Available metric keys:\n{list(metrics.keys())}"
            )
        return rate

    def req_rate(metric: dict, label: str) -> float:
        rate = metric_value(metric, "rate")
        if rate is None:
            dump = json.dumps(metric, indent=2)
            pytest.fail(
                f"Missing request rate for {label}.\n\n"
                f"Metric dump:\n{dump}\n\n"
                f"Available metric keys:\n{list(metrics.keys())}"
            )
        return rate

    token_p95 = p95(token_duration, "token")
    mtls_p95 = p95(mtls_duration, "mtls")

    token_fail_rate = fail_rate(token_fail, "token")
    mtls_fail_rate = fail_rate(mtls_fail, "mtls")
    token_req_rate = req_rate(token_reqs, "token")
    mtls_req_rate = req_rate(mtls_reqs, "mtls")

    assert token_fail_rate <= 0.01 and mtls_fail_rate <= 0.01, (
        "Error rate too high "
        f"(token fail rate={token_fail_rate}, mtls fail rate={mtls_fail_rate})"
    )
    assert mtls_req_rate > 0, "mTLS request rate is zero; throughput comparison invalid"
    assert token_req_rate >= min_token_rps_ratio * mtls_req_rate, (
        "Token throughput too low relative to mTLS "
        f"(token rps={token_req_rate}, mTLS rps={mtls_req_rate}, "
        f"min ratio={min_token_rps_ratio})"
    )
    assert token_p95 <= 3 * mtls_p95, (
        "Token endpoint too slow relative to mTLS"
        f" (token p95={token_p95} ms, mTLS p95={mtls_p95} ms)"
    )
