import argparse
import tempfile
import os
import shutil
from git import Repo
from pathlib import Path


def clone_repo_shallow(repo_url, temp_dir):
    """Clone the given GitHub repository to a temporary directory with depth 1 and without blobs."""
    repo = Repo.clone_from(repo_url, temp_dir, depth=1, no_checkout=True)
    return repo


def sparse_checkout(repo, paths):
    """Perform a sparse checkout of specified paths."""
    git_dir = repo.git_dir
    sparse_checkout_file = Path(git_dir) / "info" / "sparse-checkout"
    sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)

    print("Sparse checkout paths:")
    with sparse_checkout_file.open("w") as f:
        for path in paths:
            print(f"Adding to sparse checkout: {path}")
            f.write(f"{path}\n")

    repo.git.config("core.sparseCheckout", "true")
    repo.git.checkout()


def copy_files_preserving_structure(temp_dir, output_dir, file_name, subfolder):
    """Copy all files with a specific name to a given subfolder within the output directory, preserving the original directory structure."""
    dest_base_dir = os.path.join(output_dir, subfolder)
    os.makedirs(dest_base_dir, exist_ok=True)

    print(f"Searching for {file_name} in {temp_dir}...")
    for root, dirs, files in os.walk(temp_dir):
        if file_name in files:
            src_file = os.path.join(root, file_name)
            relative_path = os.path.relpath(root, temp_dir)
            dest_dir = os.path.join(dest_base_dir, relative_path)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_file, dest_dir)
            print(f"Copied {src_file} to {dest_dir}")


def main(output_dir):
    repo_url = "https://github.com/vespa-engine/sample-apps/"

    # Create a temporary directory to clone the repo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning repo to temporary directory {temp_dir}...")

        # Shallow clone the repo without blobs
        repo = clone_repo_shallow(repo_url, temp_dir)

        # Specify the files you want to checkout
        paths = ["**/services.xml", "**/validation-overrides.xml", "**/hosts.xml"]
        sparse_checkout(repo, paths)

        # Copy files to respective folders in the output directory, preserving directory structure
        copy_files_preserving_structure(
            temp_dir, output_dir, "services.xml", "services"
        )
        copy_files_preserving_structure(
            temp_dir, output_dir, "validation-overrides.xml", "validations"
        )
        copy_files_preserving_structure(temp_dir, output_dir, "hosts.xml", "hosts")
        print(f"Files copied to {output_dir} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy specific XML files from a GitHub repo to an output directory, preserving directory structure."
    )
    parser.add_argument(
        "output_dir", type=str, help="The directory to output the copied files."
    )
    args = parser.parse_args()

    main(args.output_dir)
