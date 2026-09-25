import json
import re
import sys
from pathlib import Path

RECORD_LABELS = (
    "lane",
    "method",
    "transport",
    "http",
    "concurrency",
    "connections",
    "processes",
)
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


def convert_records(records_file: Path, typed_names: set) -> list:
    records = json.loads(records_file.read_text()).get("records", [])
    lines = []
    # Token path's extra latency over mTLS for the same lane/method: the auth
    # hop by itself, cleaner to graph than the token/mTLS rps ratio.
    by_transport = {r.get("transport"): r for r in records}
    token, mtls = by_transport.get("token"), by_transport.get("mtls")
    if token and mtls:
        for field in ("p50_ms", "p95_ms"):
            if token.get(field) is not None and mtls.get(field) is not None:
                labels = ",".join(
                    f'{label}="{_sanitize(str(token.get(label, "unknown")))}"'
                    for label in RECORD_LABELS
                    if label != "transport"
                )
                prom_name = f"perf_token_extra_{_sanitize(field)}"
                lines += _typed(prom_name, typed_names)
                lines.append(f"{prom_name}{{{labels}}} {token[field] - mtls[field]}")
    for record in records:
        # Sanitize label values so record content cannot break the exposition
        # format or inject labels.
        labels = ",".join(
            f'{label}="{_sanitize(str(record.get(label, "unknown")))}"'
            for label in RECORD_LABELS
        )
        for field in RECORD_FIELDS:
            value = record.get(field)
            if value is None:
                continue
            prom_name = f"perf_{_sanitize(field)}"
            lines += _typed(prom_name, typed_names)
            lines.append(f"{prom_name}{{{labels}}} {value}")
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
        # Instance drift between the opening and closing k6 runs of the session.
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
