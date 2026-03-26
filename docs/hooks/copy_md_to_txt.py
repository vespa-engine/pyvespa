"""MkDocs hook to copy generated .md files to .md.txt for browser display.

Browsers download .md files (content-type: text/markdown) instead of
displaying them. This hook creates .md.txt copies (served as text/plain)
so the "View as Markdown" button opens readable text in the browser.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig


def on_post_build(config: MkDocsConfig) -> None:
    site_dir = Path(config["site_dir"])
    for md_file in site_dir.rglob("*.md"):
        txt_file = md_file.with_suffix(".md.txt")
        shutil.copy2(md_file, txt_file)
