import json
import re
import sys
from pathlib import Path


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)

def _numeric_fields(metric: dict) -> dict:
    fields = {}
    for source in (metric, metric.get("values", {})):
        for key, value in source.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields[key] = value
    return fields

def convert(summary_file: Path) -> list:
    metrics = json.loads(summary_file.read_text()).get("metrics", {})
    source = _sanitize(summary_file.stem)
    lines = []
    for name, metric in sorted(metrics.items()):
        for field, value in sorted(_numeric_fields(metric).items()):
            prom_name = f"k6_{_sanitize(name)}_{_sanitize(field)}"
            lines.append(f"# TYPE {prom_name} gauge")
            lines.append(f'{prom_name}{{source="{source}"}} {value}')
    return lines

def main() -> int:
    report_dir = Path(sys.argv[1])
    summaries = sorted(report_dir.glob("*summary.json"))
    if not summaries:
        print(f"No *summary.json files in {report_dir}; nothing to convert.")
        return 0
    lines = []
    for summary_file in summaries:
        lines += convert(summary_file)
    out = report_dir / "metrics.prom"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines) // 2} metrics from {len(summaries)} file(s) to {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
