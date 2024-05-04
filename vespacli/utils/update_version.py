# This script should modify top level pyproject.toml to update the version of vespacli
# And modify vespacli/_version_generated.py to update the version of vespacli

import toml
import sys
import re
from pathlib import Path
import argparse

PYPROJECT_TOML_PATH = Path(__file__).parent.parent / "pyproject.toml"
VERSION_FILE_PATH = Path(__file__).parent.parent / "vespacli" / "_version_generated.py"


def update_version(new_version: str):
    # Update version in pyproject.toml
    with open(PYPROJECT_TOML_PATH, "r") as f:
        data = toml.load(f)
    data["project"]["version"] = new_version
    with open("pyproject.toml", "w") as f:
        toml.dump(data, f)

    # Update version in vespacli/_version_generated.py
    with open(VERSION_FILE_PATH, "r") as f:
        content = f.read()
    new_content = re.sub(
        r'vespa_version = ".*"', f'vespa_version = "{new_version}"', content
    )
    with open("vespacli/_version_generated.py", "w") as f:
        f.write(new_content)
    print(f"Updated version to {new_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update vespacli version")
    parser.add_argument(
        "-v", "--version", type=str, help="New version to set", required=True
    )
    args = parser.parse_args()
    update_version(args.version)
    sys.exit(0)
