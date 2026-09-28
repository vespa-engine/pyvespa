import json
import re
import sys
from pathlib import Path

# Labels and fields are documented in tests/performance/README.md, Metrics reference.
RECORD_LABELS = ("lane", "method", "transport", "http", "concurrency")
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
    "server_container_cpu_util",
    "server_content_cpu_util",
)


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _numeric_fields(metric: dict) -> dict:
    fields = {}
    for source in (metric, metric.get("values", {})):
        for key, value in source.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields[key] = value
    return fields


def _typed(prom_name: str, typed_names: set) -> list:
    if prom_name in typed_names:
        return []
    typed_names.add(prom_name)
    return [f"# TYPE {prom_name} gauge"]


def convert_k6_summary(summary_file: Path, typed_names: set) -> list:
    metrics = json.loads(summary_file.read_text()).get("metrics", {})
    source = _sanitize(summary_file.stem)
    lines = []
    for name, metric in sorted(metrics.items()):
        for field, value in sorted(_numeric_fields(metric).items()):
            prom_name = f"k6_{_sanitize(name)}_{_sanitize(field)}"
            lines += _typed(prom_name, typed_names)
            lines.append(f'{prom_name}{{source="{source}"}} {value}')
    return lines


def _labels(record: dict, names: tuple) -> str:
    return ",".join(
        f'{name}="{_sanitize(str(record.get(name, "unknown")))}"' for name in names
    )


def _sample(prom_name: str, labels: str, value, typed_names: set) -> list:
    return _typed(prom_name, typed_names) + [f"{prom_name}{{{labels}}} {value}"]


def convert_records(records_file: Path, typed_names: set) -> list:
    records = json.loads(records_file.read_text()).get("records", [])
    lines = []
    for record in records:
        source = f'source="{_sanitize(records_file.stem)}"'
        labels = f"{source},{_labels(record, RECORD_LABELS)}"
        for field in RECORD_FIELDS:
            value = record.get(field)
            if value is not None:
                lines += _sample(f"perf_{field}", labels, value, typed_names)
    return lines


def main() -> int:
    report_dir = Path(sys.argv[1])
    summaries = sorted(report_dir.glob("*summary.json"))
    record_files = sorted(report_dir.glob("*records.json"))
    drift_file = report_dir / "k6_drift.json"
    if not summaries and not record_files:
        print(f"No report files in {report_dir}; nothing to convert.")
        return 0
    lines = []
    typed_names = set()
    if drift_file.exists():
        drift = json.loads(drift_file.read_text()).get("drift_pct")
        if isinstance(drift, (int, float)):
            lines += _typed("perf_instance_drift_pct", typed_names)
            lines.append(f"perf_instance_drift_pct {drift}")
    for summary_file in summaries:
        lines += convert_k6_summary(summary_file, typed_names)
    for records_file in record_files:
        lines += convert_records(records_file, typed_names)
    out = report_dir / "metrics.prom"
    out.write_text("\n".join(lines) + "\n")
    samples = len(lines) - len(typed_names)
    print(
        f"Wrote {samples} metrics from {len(summaries) + len(record_files)} "
        f"file(s) to {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
