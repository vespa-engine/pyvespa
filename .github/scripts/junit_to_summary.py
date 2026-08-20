#!/usr/bin/env python3
"""Render a JUnit XML file as a Markdown test summary.

Usage: junit_to_summary.py <junit.xml>

Prints Markdown to stdout; in CI, redirect to $GITHUB_STEP_SUMMARY.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: junit_to_summary.py <junit.xml>", file=sys.stderr)
        return 2
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
            # Indented code block: immune to backticks inside the message.
            for message_line in (message or "(no message)").splitlines():
                lines.append(f"    {message_line}")
            lines.append("")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
