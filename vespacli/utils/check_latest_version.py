from download_binaries import VespaBinaryDownloader
from vespacli._version_generated import vespa_version
import sys

if __name__ == "__main__":
    downloader = VespaBinaryDownloader()
    new_version = downloader.get_latest_version()
    found_newer = new_version != vespa_version
    if found_newer:
        print(f"{new_version}")
    else:
        print("NA")
    sys.exit(0)
