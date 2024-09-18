from download_binaries import VespaBinaryDownloader
import sys
import requests
from packaging import version


def get_latest_pypi_version() -> version.Version:
    response = requests.get(
        "https://pypi.org/simple/vespacli/",
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
    )
    latest_version = response.json()["versions"][-1]
    return version.parse(latest_version)


def get_latest_github_version() -> version.Version:
    downloader = VespaBinaryDownloader()
    return version.parse(downloader.get_latest_version())


if __name__ == "__main__":
    gh_release = get_latest_github_version()
    pypi_release = get_latest_pypi_version()
    if gh_release > pypi_release:
        print(f"{gh_release}")
    else:
        print("NA")
    sys.exit(0)
