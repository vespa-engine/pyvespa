# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Labels and fields are documented in tests/performance/README.md, Metrics reference.
RECORD_LABELS = ("lane", "method", "transport", "concurrency")
RECORD_FIELDS = (
    "rps",
    "error_rate",
    "requests",
    "duration_s",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "achieved_in_flight",
    "cpu_ms_per_request",
    "rate_limited_rate",
    "client_cpu_fraction",
    "client_cpu_cores",
    "server_container_cpu_util",
    "server_content_cpu_util",
    "server_cpu_samples",
)

Sample = Tuple[str, str, float]  # metric name, label string, value


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _labels(values: dict) -> str:
    return ",".join(f'{k}="{_sanitize(str(v))}"' for k, v in values.items() if v)


def _numeric_fields(metric: dict) -> dict:
    fields = {}
    for source in (metric, metric.get("values", {})):
        for key, value in source.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields[key] = value
    return fields


def run_identity(report_dir: Path) -> dict:
    """Commit, workflow run and runner CPU model, empty outside CI."""
    runner = report_dir / "runner.md"
    cpu = re.search(r"CPU: (.*?) \(", runner.read_text()) if runner.exists() else None
    return {
        "commit": os.environ.get("GITHUB_SHA", "")[:8],
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "runner_cpu": cpu.group(1) if cpu else "",
    }


def k6_summary_samples(summary_file: Path, runner_cpu: str) -> List[Sample]:
    metrics = json.loads(summary_file.read_text()).get("metrics", {})
    labels = _labels({"source": summary_file.stem, "runner_cpu": runner_cpu})
    return [
        (f"k6_{_sanitize(name)}_{_sanitize(field)}", labels, value)
        for name, metric in sorted(metrics.items())
        for field, value in sorted(_numeric_fields(metric).items())
    ]


def record_samples(records_file: Path, runner_cpu: str) -> List[Sample]:
    records = json.loads(records_file.read_text()).get("records", [])
    samples = []
    for record in records:
        fields = {name: record.get(name, "unknown") for name in RECORD_LABELS}
        labels = _labels(
            {"source": records_file.stem, "runner_cpu": runner_cpu, **fields}
        )
        for field in RECORD_FIELDS:
            value = record.get(field)
            if value is not None:
                samples.append((f"perf_{field}", labels, value))
    return samples


def exposition(samples: List[Sample]) -> str:
    """Prometheus text format: every sample of a metric in one group, one TYPE line."""
    by_metric: Dict[str, List[str]] = {}
    for name, labels, value in samples:
        by_metric.setdefault(name, []).append(f"{name}{{{labels}}} {value}")
    lines = []
    for name in sorted(by_metric):
        lines.append(f"# TYPE {name} gauge")
        lines += by_metric[name]
    return "\n".join(lines) + "\n"


def main() -> int:
    report_dir = Path(sys.argv[1])
    summaries = sorted(report_dir.glob("*summary.json"))
    record_files = sorted(report_dir.glob("*records.json"))
    drift_file = report_dir / "k6_drift.json"
    if not summaries and not record_files:
        print(f"No report files in {report_dir}; nothing to convert.")
        return 0

    run = run_identity(report_dir)
    runner_cpu = run["runner_cpu"]
    samples: List[Sample] = []
    if run["run_id"]:
        samples.append(("perf_run_info", _labels(run), 1))
    if drift_file.exists():
        drift = json.loads(drift_file.read_text()).get("drift_pct")
        if isinstance(drift, (int, float)):
            labels = _labels({"runner_cpu": runner_cpu})
            samples.append(("perf_instance_drift_pct", labels, drift))
    for summary_file in summaries:
        samples += k6_summary_samples(summary_file, runner_cpu)
    for records_file in record_files:
        samples += record_samples(records_file, runner_cpu)

    out = report_dir / "metrics.prom"
    out.write_text(exposition(samples))
    print(
        f"Wrote {len(samples)} metrics from {len(summaries) + len(record_files)} "
        f"file(s) to {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
