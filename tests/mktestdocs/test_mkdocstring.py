
import pathlib
import pytest

from mktestdocs import check_docstring

doc_paths = pathlib.Path("vespa").glob("**/*.py")

@pytest.mark.parametrize('fpath', doc_paths, ids=str)

def test_member(fpath):
    check_docstring(fpath)