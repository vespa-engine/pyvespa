# This script should modify top level pyproject.toml to update the version of vespacli

import toml
import sys
from pathlib import Path
import argparse

PYPROJECT_TOML_PATH = Path(__file__).parent.parent / "pyproject.toml"


def update_version(new_version: str):
    # Update version in pyproject.toml
    with open(PYPROJECT_TOML_PATH, "r") as f:
        data = toml.load(f)
    data["project"]["version"] = new_version
    with open("pyproject.toml", "w") as f:
        toml.dump(data, f)

    print(f"Updated version to {new_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update vespacli version")
    parser.add_argument(
        "-v", "--version", type=str, help="New version to set", required=True
    )
    args = parser.parse_args()
    update_version(args.version)
    sys.exit(0)
