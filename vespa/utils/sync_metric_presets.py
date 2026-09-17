# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""
Keep the vendored metric preset list in sync with the Vespa CLI.

`vespa/resources/metric-presets.json` is a verbatim copy of the Vespa CLI's
`metric-presets.json`, and `vespa/resources/metric-presets.source.json` records
where it came from (repository, path and git ref). This script resolves the
latest Vespa CLI release tag, fetches the file at that tag and, if its bytes
differ from the vendored copy, overwrites the copy and bumps the `ref`.

It is run by `.github/workflows/sync-metric-presets.yml`, which opens a PR when
something changed. It only needs the standard library.

Usage:
    python vespa/utils/sync_metric_presets.py            # latest release
    python vespa/utils/sync_metric_presets.py --tag v8.751.13
    python vespa/utils/sync_metric_presets.py --dry-run

Exit code 0 means "done" whether or not anything changed; the outcome is
printed and, when GITHUB_OUTPUT is set, written there as `changed`, `tag`,
`previous_ref` and `summary`.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
PRESETS_PATH = RESOURCES_DIR / "metric-presets.json"
SOURCE_PATH = RESOURCES_DIR / "metric-presets.source.json"

# Vespa CLI releases are tagged vX.Y.Z. The same repository also tags the
# language server (lsp-vX.Y.Z), which must be skipped.
RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+$")
USER_AGENT = "pyvespa-sync-metric-presets"


def _headers(accept: str) -> Dict[str, str]:
    headers = {"Accept": accept, "User-Agent": USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(url: str, accept: str) -> bytes:
    request = urllib.request.Request(url, headers=_headers(accept))
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def pick_release_tag(tags: Sequence[str]) -> Optional[str]:
    """Return the first Vespa CLI release tag in `tags`, skipping e.g. `lsp-*`."""
    for tag in tags:
        if RELEASE_TAG.match(tag):
            return tag
    return None


def latest_release_tag(repository: str) -> str:
    """Resolve the latest Vespa CLI release tag, newest first, from GitHub."""
    url = f"https://api.github.com/repos/{repository}/releases?per_page=30"
    releases = json.loads(_get(url, "application/vnd.github+json"))
    tags = [r["tag_name"] for r in releases if not r["draft"] and not r["prerelease"]]
    tag = pick_release_tag(tags)
    if tag is None:
        raise RuntimeError(f"No release tag matching vX.Y.Z among {tags}")
    return tag


def fetch_upstream(repository: str, path: str, tag: str) -> Optional[bytes]:
    """The raw file at `tag`, or None if the tag predates the file (HTTP 404)."""
    url = f"https://raw.githubusercontent.com/{repository}/{tag}/{path}"
    try:
        return _get(url, "application/octet-stream")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def preset_diff(old: bytes, new: bytes) -> Dict[str, List[str]]:
    """Preset names added and removed between two versions of the file."""
    old_names, new_names = set(json.loads(old)), set(json.loads(new))
    return {
        "added": sorted(new_names - old_names),
        "removed": sorted(old_names - new_names),
    }


def format_summary(tag: str, previous_ref: str, diff: Dict[str, List[str]]) -> str:
    lines = [f"Updated metric presets from {previous_ref} to {tag}."]
    for kind in ("added", "removed"):
        names = diff[kind]
        if names:
            lines.append("")
            lines.append(f"{kind.capitalize()} ({len(names)}):")
            lines.extend(f"- {name}" for name in names)
    if not diff["added"] and not diff["removed"]:
        lines.append("")
        lines.append(
            "No preset names changed; only the file contents/formatting differ."
        )
    return "\n".join(lines)


def write_outputs(values: Dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for key, value in values.items():
            if "\n" in value:
                f.write(f"{key}<<EOF_{key}\n{value}\nEOF_{key}\n")
            else:
                f.write(f"{key}={value}\n")


def sync(tag: Optional[str] = None, dry_run: bool = False) -> bool:
    """Sync the vendored list against `tag` (default: latest). Returns True if changed."""
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    repository, path, previous_ref = source["repository"], source["path"], source["ref"]

    tag = tag or latest_release_tag(repository)
    print(f"Vendored ref: {previous_ref}; checking {repository}@{tag}:{path}")

    upstream = fetch_upstream(repository, path, tag)
    if upstream is None:
        print(f"{path} does not exist at {tag}; nothing to do.")
        write_outputs({"changed": "false", "tag": tag, "previous_ref": previous_ref})
        return False

    current = PRESETS_PATH.read_bytes()
    if upstream == current:
        print(f"Vendored list is byte-identical to {tag}; nothing to do.")
        write_outputs({"changed": "false", "tag": tag, "previous_ref": previous_ref})
        return False

    summary = format_summary(tag, previous_ref, preset_diff(current, upstream))
    print(summary)
    if dry_run:
        print("Dry run: not writing files.")
    else:
        PRESETS_PATH.write_bytes(upstream)
        source["ref"] = tag
        SOURCE_PATH.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {PRESETS_PATH.name} and set ref={tag} in {SOURCE_PATH.name}")
    write_outputs(
        {
            "changed": "true",
            "tag": tag,
            "previous_ref": previous_ref,
            "summary": summary,
        }
    )
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tag",
        help="Vespa CLI release tag to sync against (default: latest release)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report drift without modifying the vendored files",
    )
    args = parser.parse_args(argv)
    sync(tag=args.tag, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
