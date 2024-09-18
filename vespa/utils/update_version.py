# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

# This script should modify top level pyproject.toml to update the version of pyvespa

import sys
import re
from pathlib import Path
import argparse

PYPROJECT_TOML_PATH = Path(__file__).parent.parent.parent / "pyproject.toml"


def update_version(new_version: str):
    # Update version in pyproject.toml
    with open(PYPROJECT_TOML_PATH, "r") as f:
        content = f.read()
    new_content = re.sub(r'version = ".*"', f'version = "{new_version}"', content)
    with open(PYPROJECT_TOML_PATH, "w") as f:
        f.write(new_content)
    print(f"Updated version to {new_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update pyvespa version")
    parser.add_argument(
        "-v", "--version", type=str, help="New version to set", required=True
    )
    args = parser.parse_args()
    update_version(args.version)
    sys.exit(0)
