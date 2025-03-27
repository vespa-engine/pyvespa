import pathlib
import pytest

from mktestdocs import check_md_file

doc_paths = pathlib.Path("docs/sphinx/source/api").glob("**/*.md")

@pytest.mark.parametrize('fpath', doc_paths, ids=str)

def test_member(fpath):
    check_md_file(fpath,lang="python")
    check_md_file(fpath,lang="bash")
    