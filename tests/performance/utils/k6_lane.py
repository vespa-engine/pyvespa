# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Run k6 and parse its measured-window metrics."""

import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import List

from utils.metrics import LaneResult
from utils.saturation import RunnerCpu, ServerCpuSampler
from utils.workloads import CONTAINER_CLUSTER, CONTENT_CLUSTER, LoadProfile

SCRIPT = Path(__file__).parent.parent / "k6" / "token_vs_mtls.js"


class K6Error(AssertionError):
    pass


def _require_metric_key(metrics: dict, key: str) -> dict:
    if key in metrics:
        return metrics[key]
    raise K6Error(
        f"Missing metric '{key}' in k6 summary. Available keys: {list(metrics.keys())}"
    )


def _metric_value(metric: dict, field: str):
    if field in metric:
        return metric[field]
    return metric.get("values", {}).get(field)


def _require_value(metric: dict, fields: tuple, label: str) -> float:
    for field in fields:
        value = _metric_value(metric, field)
        if value is not None:
            return value
    raise K6Error(
        f"Missing {fields} for {label}.\n\nMetric dump:\n{json.dumps(metric, indent=2)}"
    )


def lane_result(metrics: dict, transport: str, profile: LoadProfile) -> LaneResult:
    duration = _require_metric_key(metrics, f"{transport}_req_duration")
    fail = _require_metric_key(metrics, f"{transport}_fail_rate")
    reqs = _require_metric_key(metrics, f"{transport}_reqs")
    # k6 omits a Counter that never got a sample, so a missing 429 counter is 0.
    rate_limited = metrics.get(f"{transport}_rate_limited", {})

    # k6 Rate metrics export the rate under "value"; keep "rate" as fallback.
    error_rate = _require_value(fail, ("value", "rate"), transport)
    count = int(_require_value(reqs, ("count",), transport))
    limited = int(_metric_value(rate_limited, "count") or 0)
    # The script only counts requests completed inside the hold window, so
    # rps is count / hold exactly like the pyvespa lane (the summary's own
    # "rate" divides by the whole run including ramp-up and graceful stop).
    return LaneResult(
        lane="k6",
        method="http_post",
        transport=transport,
        http="negotiate",
        rps=count / profile.duration_s,
        error_rate=error_rate,
        requests=count,
        duration_s=profile.duration_s,
        concurrency=profile.concurrency,
        connections=profile.connections(),
        p50_ms=_metric_value(duration, "med"),
        p95_ms=_require_value(duration, ("p(95)",), transport),
        p99_ms=_metric_value(duration, "p(99)"),
        rate_limited_rate=limited / count if count else None,
    )


def run_k6(endpoints, profile: LoadProfile, summary_file: Path) -> List[LaneResult]:
    """Run k6 and attach runner and instance CPU measurements to each result."""
    env = {
        **os.environ,
        "TOKEN_URL": endpoints.token_url,
        "MTLS_URL": endpoints.mtls_url,
        "TOKEN_AUTH_HEADER": f"Bearer {endpoints.token}",
        "MTLS_CERT_PATH": endpoints.cert_path,
        "MTLS_KEY_PATH": endpoints.key_path,
        **profile.k6_env(),
    }
    command = ["k6", "run", "--summary-export", str(summary_file)]
    if os.environ.get("CI"):
        command.append("--quiet")
    command.append(str(SCRIPT))

    expected_s = int(profile.warmup_s + profile.duration_s)
    print(
        f"\n=== Running k6: {SCRIPT.name} ({profile.concurrency} in flight per "
        f"transport over {profile.connections()} connections, "
        f"~{expected_s}s + graceful stop) ==="
    )
    load_start = time.time()
    runner_cpu = RunnerCpu().start()
    sampler = ServerCpuSampler(endpoints.mtls_app).start()
    result = subprocess.run(command, env=env, capture_output=False, text=True)
    runner_fraction = runner_cpu.stop()
    server = sampler.stop(load_start)
    if result.returncode != 0:
        raise K6Error(f"k6 exited with {result.returncode} (see output above)")

    metrics = json.loads(summary_file.read_text()).get("metrics", {})
    return [
        replace(
            lane_result(metrics, transport, profile),
            client_cpu_fraction=runner_fraction,
            server_container_cpu_util=server.get(f"container/{CONTAINER_CLUSTER}"),
            server_content_cpu_util=server.get(f"content/{CONTENT_CLUSTER}"),
        )
        for transport in ("token", "mtls")
    ]
