# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from utils.metrics import LaneResult
from utils.cpu_probes import LoadGeneratorCpu, InstanceCpuSampler
from utils.config import (
    CONTAINER_CLUSTER,
    CONTENT_CLUSTER,
    SMALL,
    FeedCase,
    LoadProfile,
)

SCRIPT = Path(__file__).parent.parent / "k6" / "token_vs_mtls.js"


def _require_metric_key(metrics: dict, key: str) -> dict:
    if key in metrics:
        return metrics[key]
    raise KeyError(
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
    raise KeyError(
        f"Missing {fields} for {label}.\n\nMetric dump:\n{json.dumps(metric, indent=2)}"
    )


def lane_result(
    metrics: dict, transport: str, profile: LoadProfile, method: str = "http_post"
) -> LaneResult:
    """Turn one transport's k6 summary into a LaneResult: the same shape the pyvespa
    lane produces, which the tests assert on and export to Prometheus."""
    duration = _require_metric_key(metrics, f"{transport}_req_duration")
    failed = _require_metric_key(metrics, f"{transport}_fail_rate")
    requests = _require_metric_key(metrics, f"{transport}_reqs")
    # k6 omits a Counter that never got a sample, so a missing 429 counter is 0.
    rate_limited = metrics.get(f"{transport}_rate_limited", {})

    count = int(_require_value(requests, ("count",), transport))
    limited = int(_metric_value(rate_limited, "count") or 0)
    # k6's own "rate" divides by the whole run; the script only counts inside the hold.
    rps = count / profile.duration_s
    mean_ms = _metric_value(duration, "avg")

    return LaneResult(
        lane="k6",
        method=method,
        transport=transport,
        concurrency=profile.concurrency,
        duration_s=profile.duration_s,
        requests=count,
        rps=rps,
        error_rate=_require_value(failed, ("value", "rate"), transport),
        rate_limited_rate=limited / count if count else None,
        p50_ms=_metric_value(duration, "med"),
        p95_ms=_require_value(duration, ("p(95)",), transport),
        p99_ms=_metric_value(duration, "p(99)"),
        mean_ms=mean_ms,
        achieved_in_flight=rps * mean_ms / 1000 if mean_ms is not None else None,
    )


def run_k6(
    endpoints,
    profile: LoadProfile,
    summary_file: Path,
    transport: str,
    case: FeedCase = SMALL,
) -> LaneResult:
    env = {
        **os.environ,
        "TRANSPORT": transport,
        "TOKEN_URL": endpoints.token_url,
        "MTLS_URL": endpoints.mtls_url,
        "TOKEN_AUTH_HEADER": f"Bearer {endpoints.token}",
        "MTLS_CERT_PATH": endpoints.cert_path,
        "MTLS_KEY_PATH": endpoints.key_path,
        "BODY_BYTES": str(case.body_bytes),
        "COMPRESSION": "gzip" if case.gzip else "",
        **profile.k6_env(),
    }
    command = ["k6", "run", "--summary-export", str(summary_file)]
    if os.environ.get("CI"):
        command.append("--quiet")
    command.append(str(SCRIPT))

    expected_s = int(profile.warmup_s + profile.duration_s)
    print(
        f"\n=== Running k6 {transport}{case.suffix}: {profile.concurrency} in flight "
        f"over {profile.k6_connections} connections, ~{expected_s}s + graceful stop ==="
    )
    load_start = time.time()
    runner_cpu = LoadGeneratorCpu().start()
    sampler = InstanceCpuSampler(endpoints.mtls_app).start()
    try:
        subprocess.run(command, env=env, check=True)
    finally:
        runner_fraction = runner_cpu.stop()
        server = sampler.stop(load_start)

    metrics = json.loads(summary_file.read_text()).get("metrics", {})
    return replace(
        lane_result(metrics, transport, profile, "http_post" + case.suffix),
        client_cpu_fraction=runner_fraction,
        server_container_cpu_util=server.get(f"container/{CONTAINER_CLUSTER}"),
        server_content_cpu_util=server.get(f"content/{CONTENT_CLUSTER}"),
    )
