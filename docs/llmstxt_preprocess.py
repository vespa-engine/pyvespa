"""Preprocess hook for mkdocs-llmstxt plugin.

Rewrites .html references to .md so that links in llms-full.txt point to
the Markdown versions of pages instead of the HTML versions.
This is needed because we use ``use_directory_urls: false`` in mkdocs.yml,
which causes MkDocs to generate .html links in the rendered HTML.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bs4 import NavigableString

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

PYVESPA_DOCS_BASE_URL = "https://vespa-engine.github.io/pyvespa/"

_PYVESPA_HTML_RE = re.compile(
    r"(" + re.escape(PYVESPA_DOCS_BASE_URL) + r"[^\s)\"'>]*)\.html([#\s)\"'>]|$)"
)


def preprocess(soup: BeautifulSoup, output: str) -> None:  # noqa: ARG001
    # Rewrite href attributes on <a> tags (only relative or pyvespa URLs).
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not isinstance(href, str) or ".html" not in href:
            continue
        if href.startswith(PYVESPA_DOCS_BASE_URL) or not href.startswith(
            ("http://", "https://", "//")
        ):
            link["href"] = re.sub(r"\.html(#|$)", r".md\1", href)

    # Rewrite .html URLs that appear in link text or bare text nodes.
    for text_node in soup.find_all(string=_PYVESPA_HTML_RE):
        if isinstance(text_node, NavigableString):
            new_text = _PYVESPA_HTML_RE.sub(r"\1.md\2", text_node)
            if new_text != text_node:
                text_node.replace_with(NavigableString(new_text))
