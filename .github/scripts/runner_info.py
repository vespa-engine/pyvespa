"""Record the CI runner's hardware next to the performance report.

GitHub-hosted runners land on different CPU models between runs, which moves
client-bound (Python) throughput by up to ~1.8x while the server-bound k6 lane
barely changes. Writing the CPU model plus a short single-thread Python score
into the report makes that visible per run and lets a dashboard normalize.

Usage: runner_info.py <report_dir>
Writes <report_dir>/runner.json and prints a Markdown summary to stdout.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

# Same payload shape as the workloads feed, so the score tracks the Python
# work a feed request does (serialize, parse) on this runner.
_DOC = {"fields": {"id": "x" * 16, "title": "performance-doc", "body": "benchmark run"}}
_SCORE_ITERATIONS = 200_000


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def mem_total_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return 0.0


def python_cpu_score() -> float:
    """Single-thread json dumps+loads iterations per second."""
    started = time.perf_counter()
    for _ in range(_SCORE_ITERATIONS):
        json.loads(json.dumps(_DOC))
    return round(_SCORE_ITERATIONS / (time.perf_counter() - started))


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: runner_info.py <report_dir>", file=sys.stderr)
        return 2
    report_dir = Path(sys.argv[1])
    report_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count() or 0,
        "mem_total_gb": mem_total_gb(),
        "python": platform.python_version(),
        "python_cpu_score_ops_per_s": python_cpu_score(),
        "runner_image": os.environ.get("ImageOS", ""),
        "runner_image_version": os.environ.get("ImageVersion", ""),
    }
    (report_dir / "runner.json").write_text(json.dumps(info, indent=2))

    print("## Runner")
    print("")
    print(f"- CPU: {info['cpu_model']} ({info['cpu_count']} vCPU)")
    print(f"- Memory: {info['mem_total_gb']} GB")
    print(
        f"- Python {info['python']}, single-thread json score: "
        f"{info['python_cpu_score_ops_per_s']:,} ops/s"
    )
    if info["runner_image_version"]:
        print(f"- Image: {info['runner_image']} {info['runner_image_version']}")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
