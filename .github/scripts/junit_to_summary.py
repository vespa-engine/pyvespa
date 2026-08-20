#!/usr/bin/env python3
"""Render a JUnit XML file as a Markdown test summary.

Usage: junit_to_summary.py <junit.xml>

Prints Markdown to stdout; in CI, redirect to $GITHUB_STEP_SUMMARY.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    junit_file = Path(sys.argv[1])
    if not junit_file.exists():
        print(f"No test results found ({junit_file} missing).")
        return 0

    root = ET.parse(junit_file).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")

    lines = ["# Test results", ""]
    total = failed = errored = skipped = 0
    failures = []
    for suite in suites:
        total += int(suite.get("tests", 0))
        failed += int(suite.get("failures", 0))
        errored += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
        for case in suite.iter("testcase"):
            for problem in list(case.iter("failure")) + list(case.iter("error")):
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                message = (problem.get("message") or "").strip()
                failures.append((name, message))

    passed = total - failed - errored - skipped
    lines.append(
        f"**{passed} passed**, {failed} failed, {errored} errored, "
        f"{skipped} skipped ({total} total)"
    )
    if failures:
        lines += ["", "## Failures", ""]
        for name, message in failures:
            lines.append(f"### ❌ `{name}`")
            lines.append("")
            lines.append("```")
            lines.append(message or "(no message)")
            lines.append("```")
            lines.append("")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
