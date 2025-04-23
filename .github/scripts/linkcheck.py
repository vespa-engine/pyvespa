#!/usr/bin/env python3
# /// script
# dependencies = [
#   "aiohttp>=3.8.0",
#   "rich>=13.0.0",
#   "beautifulsoup4>=4.12.0",
#   "markdown>=3.4.0",
#   "nbformat>=5.7.0",
#   "tqdm>=4.65.0",
#   "httpx>=0.24.0",
# ]
# ///

import asyncio
import argparse
import glob
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
import markdown
import nbformat
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich import box
from rich.theme import Theme
from tqdm.asyncio import tqdm_asyncio


class LinkStatus(Enum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class LinkCheckResult:
    url: str
    status_code: Optional[int] = None
    status: LinkStatus = LinkStatus.OK
    error_message: str = ""
    source_file: str = ""
    source_line: Optional[int] = None
    response_time: float = 0


@dataclass
class LinkCheckerConfig:
    directories: List[str]
    recursive: bool = True
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    concurrency: int = 10
    include_patterns: List[str] = field(
        default_factory=lambda: ["*.md", "*.html", "*.ipynb"]
    )
    exclude_patterns: List[str] = field(default_factory=lambda: [])
    acceptable_status_codes: Set[int] = field(
        default_factory=lambda: {200, 201, 202, 203, 204, 205, 206, 207, 208, 226}
    )
    user_agent: str = "DocLinkChecker/1.0"
    check_fragments: bool = True
    verify_ssl: bool = True
    skip_external: bool = False
    verbose: bool = False
    output_file: Optional[str] = None
    fail_on_error: bool = False
    follow_redirects: bool = True
    max_redirects: int = 5
    ignore_urls: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class LinkCheckSummary:
    total_links: int = 0
    checked_links: int = 0
    ok_links: int = 0
    error_links: int = 0
    skipped_links: int = 0
    internal_links: int = 0
    external_links: int = 0
    total_files: int = 0
    start_time: float = field(default_factory=time.time)
    results: List[LinkCheckResult] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class LinkChecker:
    def __init__(self, config: LinkCheckerConfig):
        self.config = config
        self.summary = LinkCheckSummary()
        self.console = Console(
            theme=Theme(
                {
                    "success": "green",
                    "error": "red",
                    "warning": "yellow",
                    "info": "cyan",
                }
            )
        )
        self.base_urls = {}
        self.seen_urls = set()
        self.in_progress = set()
        self.url_pattern = re.compile(
            r'<?(https?://[^\s<>"\'()]+|www\.[^\s<>"\'()]+)>?', re.IGNORECASE
        )

    async def check_links(self) -> LinkCheckSummary:
        files_to_check = self._get_files_to_check()
        self.summary.total_files = len(files_to_check)

        if not files_to_check:
            self.console.print(
                "[warning]No files found matching the specified patterns.[/warning]"
            )
            return self.summary

        self.console.print(f"[info]Found {len(files_to_check)} files to check[/info]")

        links_by_file = {}

        # Extract all links from files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Extracting links from files...", total=len(files_to_check)
            )

            for file_path in files_to_check:
                progress.update(
                    task,
                    advance=1,
                    description=f"Extracting links from {os.path.basename(file_path)}",
                )
                links = self._extract_links_from_file(file_path)
                if links:
                    links_by_file[file_path] = links
                    self.summary.total_links += len(links)

        # Check all the links
        self.console.print(
            f"[info]Found {self.summary.total_links} links to check[/info]"
        )

        if self.summary.total_links == 0:
            self.console.print(
                "[warning]No links found in the specified files.[/warning]"
            )
            return self.summary

        # Group by URL to avoid checking the same URL multiple times
        url_to_sources = {}
        for file_path, links in links_by_file.items():
            for link, line_num in links:
                # Clean URLs that might be wrapped in angle brackets or have invalid trailing characters
                clean_link = self._clean_url(link)

                if clean_link not in url_to_sources:
                    url_to_sources[clean_link] = []
                url_to_sources[clean_link].append((file_path, line_num))

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.concurrency)

        # Set up session with common parameters
        timeout = httpx.Timeout(self.config.timeout)

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=self.config.follow_redirects,
            max_redirects=self.config.max_redirects,
            verify=self.config.verify_ssl,
            headers={"User-Agent": self.config.user_agent, **self.config.headers},
        ) as client:
            tasks = []
            # Only check each unique URL once
            checked_urls = set()
            for url, sources in url_to_sources.items():
                if url in checked_urls:
                    continue

                checked_urls.add(url)
                # Only create one task per URL, using the first source location
                file_path, line_num = sources[0]
                task = self._check_link(client, url, file_path, line_num, semaphore)
                tasks.append(task)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task_id = progress.add_task("Checking links...", total=len(tasks))

                for future in tqdm_asyncio.as_completed(tasks):
                    result = await future
                    self.summary.results.append(result)
                    self.summary.checked_links += 1

                    if result.status == LinkStatus.OK:
                        self.summary.ok_links += 1
                    elif result.status == LinkStatus.ERROR:
                        self.summary.error_links += 1
                    elif result.status == LinkStatus.SKIPPED:
                        self.summary.skipped_links += 1

                    is_external = urlparse(result.url).netloc != ""
                    if is_external:
                        self.summary.external_links += 1
                    else:
                        self.summary.internal_links += 1

                    progress_desc = f"Checking links... {self.summary.checked_links}/{self.summary.total_links}"
                    if self.config.verbose and result.status == LinkStatus.ERROR:
                        progress_desc = f"Error: {result.url} - {result.error_message}"

                    progress.update(task_id, advance=1, description=progress_desc)

        return self.summary

    async def _check_link(
        self,
        client: httpx.AsyncClient,
        url: str,
        source_file: str,
        line_num: Optional[int],
        semaphore: asyncio.Semaphore,
    ) -> LinkCheckResult:
        result = LinkCheckResult(url=url, source_file=source_file, source_line=line_num)

        # Skip checking for certain types of links
        if self._should_skip_link(url):
            result.status = LinkStatus.SKIPPED
            result.error_message = "Skipped (matched ignore pattern)"
            return result

        # Handle fragments separately for local files
        if url.startswith("#"):
            # Skip fragment checks if configured
            if not self.config.check_fragments:
                result.status = LinkStatus.SKIPPED
                result.error_message = "Skipped (fragment link)"
                return result

            # Check if the fragment exists in the file
            return await self._check_fragment(url, source_file, result)

        # Handle relative paths
        url_obj = urlparse(url)
        if not url_obj.netloc and not url_obj.scheme:
            # It's a relative path
            if source_file in self.base_urls:
                base_url = self.base_urls[source_file]
                url = urljoin(base_url, url)
            else:
                # For local file links, check if the file exists
                file_path = self._resolve_relative_path(source_file, url)
                if file_path:
                    if os.path.exists(file_path):
                        result.status_code = 200
                        # If there's a fragment, check it
                        if "#" in url and self.config.check_fragments:
                            fragment = url.split("#")[1]
                            if fragment:
                                return await self._check_fragment(
                                    f"#{fragment}", file_path, result
                                )
                    else:
                        result.status = LinkStatus.ERROR
                        result.error_message = f"File not found: {file_path}"
                    return result

        # Skip external URLs if configured to do so
        if self.config.skip_external and url_obj.netloc:
            result.status = LinkStatus.SKIPPED
            result.error_message = "Skipped (external URL)"
            return result

        # Use semaphore to limit concurrency
        async with semaphore:
            start_time = time.time()
            retries = 0

            while retries <= self.config.max_retries:
                try:
                    response = await client.head(url)
                    if response.status_code == 405:  # Method not allowed
                        # Try with GET instead
                        response = await client.get(url, follow_redirects=True)

                    result.status_code = response.status_code
                    result.response_time = time.time() - start_time

                    if response.status_code in self.config.acceptable_status_codes:
                        result.status = LinkStatus.OK
                    else:
                        result.status = LinkStatus.ERROR
                        result.error_message = (
                            f"HTTP status code: {response.status_code}"
                        )

                    break

                except httpx.RequestError as e:
                    retries += 1
                    if retries > self.config.max_retries:
                        result.status = LinkStatus.ERROR
                        result.error_message = str(e)
                    else:
                        # Wait before retrying
                        await asyncio.sleep(self.config.retry_delay)

        return result

    async def _check_fragment(
        self, fragment: str, file_path: str, result: LinkCheckResult
    ) -> LinkCheckResult:
        """Check if a fragment (anchor) exists in the file."""
        fragment_id = fragment[1:]  # Remove the # character

        if not fragment_id:
            # Empty fragment is valid (links to the top of the page)
            result.status = LinkStatus.OK
            result.status_code = 200
            return result

        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check based on file extension
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".md":
                # For Markdown, convert to HTML and then check
                html = markdown.markdown(content)
                soup = BeautifulSoup(html, "html.parser")
                if soup.find(id=fragment_id) or soup.find(attrs={"name": fragment_id}):
                    result.status = LinkStatus.OK
                    result.status_code = 200
                else:
                    result.status = LinkStatus.ERROR
                    result.error_message = (
                        f"Fragment #{fragment_id} not found in {file_path}"
                    )

            elif ext == ".html":
                # For HTML, use BeautifulSoup to find the fragment
                soup = BeautifulSoup(content, "html.parser")
                if soup.find(id=fragment_id) or soup.find(attrs={"name": fragment_id}):
                    result.status = LinkStatus.OK
                    result.status_code = 200
                else:
                    result.status = LinkStatus.ERROR
                    result.error_message = (
                        f"Fragment #{fragment_id} not found in {file_path}"
                    )

            elif ext == ".ipynb":
                # For Jupyter notebooks, look for headers that might match the fragment
                result.status = LinkStatus.SKIPPED
                result.error_message = (
                    "Fragment checking in Jupyter notebooks is not supported"
                )

            else:
                # For other files, we can't check fragments reliably
                result.status = LinkStatus.SKIPPED
                result.error_message = (
                    f"Fragment checking not supported for {ext} files"
                )

        except Exception as e:
            result.status = LinkStatus.ERROR
            result.error_message = f"Error checking fragment: {str(e)}"

        return result

    def _get_files_to_check(self) -> List[str]:
        """Get all files matching the patterns in the specified directories."""
        all_files = []

        for directory in self.config.directories:
            # If it's a direct file path, add it directly
            if os.path.isfile(directory):
                all_files.append(directory)
                continue

            # Handle glob patterns
            if any(char in directory for char in "*?[]"):
                matching_files = glob.glob(directory, recursive=self.config.recursive)
                all_files.extend(matching_files)
                continue

            # Otherwise, it's a directory - get all matching files
            for pattern in self.config.include_patterns:
                if self.config.recursive:
                    search_pattern = os.path.join(directory, "**", pattern)
                    matching_files = glob.glob(search_pattern, recursive=True)
                else:
                    search_pattern = os.path.join(directory, pattern)
                    matching_files = glob.glob(search_pattern)

                all_files.extend(matching_files)

        # Filter out excluded patterns
        for exclude_pattern in self.config.exclude_patterns:
            all_files = [
                f for f in all_files if not glob.fnmatch.fnmatch(f, exclude_pattern)
            ]

        # Remove duplicates and sort
        return sorted(list(set(all_files)))

    def _extract_links_from_file(
        self, file_path: str
    ) -> List[Tuple[str, Optional[int]]]:
        """Extract all links from a file based on its type."""
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".md":
                return self._extract_links_from_markdown(file_path)
            elif ext == ".html":
                return self._extract_links_from_html(file_path)
            elif ext == ".ipynb":
                return self._extract_links_from_notebook(file_path)
            else:
                self.console.print(
                    f"[warning]Unsupported file type: {file_path}[/warning]"
                )
                return []
        except Exception as e:
            self.console.print(
                f"[error]Error extracting links from {file_path}: {str(e)}[/error]"
            )
            return []

    def _extract_links_from_markdown(
        self, file_path: str
    ) -> List[Tuple[str, Optional[int]]]:
        """Extract links from a Markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.readlines()

        links = []

        # Regular expression for Markdown links
        md_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

        for i, line in enumerate(content, 1):
            # Find Markdown style links [text](url)
            for match in md_link_pattern.finditer(line):
                url = match.group(2).split(" ")[
                    0
                ]  # Handle cases with title: [text](url "title")
                # Clean the URL
                url = self._clean_url(url)
                links.append((url, i))

            # Also find plain URLs in the text
            for match in self.url_pattern.finditer(line):
                url = match.group(0)
                # Clean the URL
                url = self._clean_url(url)
                links.append((url, i))

        return links

    def _extract_links_from_html(
        self, file_path: str
    ) -> List[Tuple[str, Optional[int]]]:
        """Extract links from an HTML file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Set the base URL for relative links
        base_url = None
        soup = BeautifulSoup(content, "html.parser")
        base_tag = soup.find("base", href=True)
        if base_tag:
            base_url = base_tag["href"]
            self.base_urls[file_path] = base_url

        links = []

        # Extract links from <a> tags
        for a_tag in soup.find_all("a", href=True):
            line_num = self._get_line_number(content, a_tag)
            url = self._clean_url(a_tag["href"])
            links.append((url, line_num))

        # Extract links from <img> tags
        for img_tag in soup.find_all("img", src=True):
            line_num = self._get_line_number(content, img_tag)
            url = self._clean_url(img_tag["src"])
            links.append((url, line_num))

        # Extract links from <link> tags
        for link_tag in soup.find_all("link", href=True):
            line_num = self._get_line_number(content, link_tag)
            url = self._clean_url(link_tag["href"])
            links.append((url, line_num))

        # Extract links from <script> tags
        for script_tag in soup.find_all("script", src=True):
            line_num = self._get_line_number(content, script_tag)
            url = self._clean_url(script_tag["src"])
            links.append((url, line_num))

        # Extract links from inline styles with url()
        style_pattern = re.compile(r'url\([\'"]?([^\'")]+)[\'"]?\)')
        for style_tag in soup.find_all("style"):
            if style_tag.string:
                line_num = self._get_line_number(content, style_tag)
                for match in style_pattern.finditer(style_tag.string):
                    url = self._clean_url(match.group(1))
                    links.append((url, line_num))

        # Extract links from inline style attributes
        for tag in soup.find_all(style=True):
            line_num = self._get_line_number(content, tag)
            for match in style_pattern.finditer(tag["style"]):
                url = self._clean_url(match.group(1))
                links.append((url, line_num))  # Use the cleaned URL, not original match

        return links

    def _extract_links_from_notebook(
        self, file_path: str
    ) -> List[Tuple[str, Optional[int]]]:
        """Extract links from a Jupyter notebook."""
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        links = []
        cell_num = 0

        for cell in notebook.cells:
            cell_num += 1

            if cell.cell_type == "markdown":
                # Extract Markdown links
                md_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
                for match in md_link_pattern.finditer(cell.source):
                    url = match.group(2).split(" ")[0]
                    # Clean the URL
                    url = self._clean_url(url)
                    links.append((url, cell_num))

                # Extract plain URLs
                for match in self.url_pattern.finditer(cell.source):
                    url = match.group(0)
                    # Clean the URL
                    url = self._clean_url(url)
                    links.append((url, cell_num))

                # Also parse the HTML that might be embedded in Markdown
                html = markdown.markdown(cell.source)
                soup = BeautifulSoup(html, "html.parser")

                for a_tag in soup.find_all("a", href=True):
                    url = self._clean_url(a_tag["href"])
                    links.append((url, cell_num))

                for img_tag in soup.find_all("img", src=True):
                    url = self._clean_url(img_tag["src"])
                    links.append((url, cell_num))

            elif cell.cell_type == "code":
                # Look for URLs in code comments and strings
                for line in cell.source.split("\n"):
                    for match in self.url_pattern.finditer(line):
                        url = self._clean_url(match.group(0))
                        links.append((url, cell_num))

        return links

    def _get_line_number(self, content: str, tag) -> Optional[int]:
        """Try to determine the line number of a tag in the content."""
        try:
            # Get the string representation of the tag
            tag_str = str(tag)
            # Find the position in the content
            pos = content.find(tag_str)
            if pos == -1:
                return None

            # Count newlines up to this position
            return content[:pos].count("\n") + 1
        except Exception:
            return None

    def _should_skip_link(self, url: str) -> bool:
        """Check if a link should be skipped based on configuration."""
        # Skip empty URLs
        if not url:
            return True

        # Clean URL from angle brackets and trailing invalid characters
        url = self._clean_url(url)

        # Skip URLs in the ignore list
        for ignore_pattern in self.config.ignore_urls:
            if re.search(ignore_pattern, url):
                return True

        # Skip data URLs
        if url.startswith("data:"):
            return True

        # Skip javascript: URLs
        if url.startswith("javascript:"):
            return True

        # Skip mailto: URLs
        if url.startswith("mailto:"):
            return True

        # Skip tel: URLs
        if url.startswith("tel:"):
            return True

        # Skip image files
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".svg",
            ".bmp",
            ".webp",
            ".ico",
        ]
        if any(url.lower().endswith(ext) for ext in image_extensions):
            return True

        return False

    def _clean_url(self, url: str) -> str:
        """Clean a URL by removing angle brackets and trailing invalid characters."""
        # Remove angle brackets if they match
        if url.startswith("<") and url.endswith(">"):
            url = url[1:-1]
        # Remove leading angle bracket even if no trailing one
        elif url.startswith("<"):
            url = url[1:]

        # Remove common trailing characters that shouldn't be part of URLs
        invalid_trailing_chars = ")]},;:'\"`.[]"  # Added period, backtick and brackets
        while url and url[-1] in invalid_trailing_chars:
            url = url[:-1]

        return url

    def _resolve_relative_path(
        self, source_file: str, relative_url: str
    ) -> Optional[str]:
        """Resolve a relative URL from a source file to an absolute file path."""
        try:
            # Handle fragment-only URLs
            if relative_url.startswith("#"):
                return source_file

            # Split the URL to remove any fragments
            url_parts = relative_url.split("#")
            path_part = url_parts[0]

            # Get the directory of the source file
            source_dir = os.path.dirname(os.path.abspath(source_file))

            # Join with the relative path
            resolved_path = os.path.normpath(os.path.join(source_dir, path_part))

            return resolved_path
        except Exception:
            return None

    def print_summary(self) -> None:
        """Print a summary of the link checking results."""
        table = Table(title="Link Check Summary", box=box.ROUNDED)

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Files", f"{self.summary.total_files}")
        table.add_row("Total Links", f"{self.summary.total_links}")
        table.add_row("Checked Links", f"{self.summary.checked_links}")
        table.add_row("OK Links", f"[success]{self.summary.ok_links}[/success]")
        table.add_row("Error Links", f"[error]{self.summary.error_links}[/error]")
        table.add_row(
            "Skipped Links", f"[warning]{self.summary.skipped_links}[/warning]"
        )
        table.add_row("Internal Links", f"{self.summary.internal_links}")
        table.add_row("External Links", f"{self.summary.external_links}")
        table.add_row("Elapsed Time", f"{self.summary.elapsed_time:.2f} seconds")

        self.console.print(table)

        if self.summary.error_links > 0:
            self._print_error_links()

    def _print_error_links(self) -> None:
        """Print a table of all links with errors."""
        table = Table(title="Links with Errors", box=box.ROUNDED, show_lines=True)

        table.add_column("URL", style="cyan", ratio=3, overflow="fold")
        table.add_column("Status", style="red", justify="center", width=8)
        table.add_column("Error Message", style="yellow", ratio=2, overflow="fold")
        table.add_column("Source File", style="green", ratio=1, overflow="fold")
        table.add_column("Line", style="blue", justify="center", width=6)

        # Keep track of URLs we've already reported
        reported_urls = set()

        for result in self.summary.results:
            if result.status == LinkStatus.ERROR:
                if result.url in reported_urls:
                    continue

                reported_urls.add(result.url)
                source_file = os.path.basename(result.source_file)
                line = str(result.source_line) if result.source_line else "N/A"
                status = str(result.status_code) if result.status_code else "N/A"

                table.add_row(
                    result.url, status, result.error_message, source_file, line
                )

        self.console.print(table)

    def save_results(self) -> None:
        """Save the results to a file if output_file is specified."""
        if not self.config.output_file:
            return

        try:
            with open(self.config.output_file, "w", encoding="utf-8") as f:
                # Write summary
                f.write("# Link Check Summary\n\n")
                f.write(f"- Total Files: {self.summary.total_files}\n")
                f.write(f"- Total Links: {self.summary.total_links}\n")
                f.write(f"- Checked Links: {self.summary.checked_links}\n")
                f.write(f"- OK Links: {self.summary.ok_links}\n")
                f.write(f"- Error Links: {self.summary.error_links}\n")
                f.write(f"- Skipped Links: {self.summary.skipped_links}\n")
                f.write(f"- Internal Links: {self.summary.internal_links}\n")
                f.write(f"- External Links: {self.summary.external_links}\n")
                f.write(f"- Elapsed Time: {self.summary.elapsed_time:.2f} seconds\n\n")

                # Write error links
                if self.summary.error_links > 0:
                    f.write("# Links with Errors\n\n")
                    for result in self.summary.results:
                        if result.status == LinkStatus.ERROR:
                            source_file = result.source_file
                            line = (
                                str(result.source_line) if result.source_line else "N/A"
                            )
                            status = (
                                str(result.status_code) if result.status_code else "N/A"
                            )

                            f.write(f"## {result.url}\n")
                            f.write(f"- Status: {status}\n")
                            f.write(f"- Error: {result.error_message}\n")
                            f.write(f"- Source: {source_file}:{line}\n\n")

            self.console.print(
                f"[success]Results saved to {self.config.output_file}[/success]"
            )

        except Exception as e:
            self.console.print(f"[error]Error saving results: {str(e)}[/error]")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Asynchronous documentation link checker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories or glob patterns to check (e.g., 'docs/**/*.md')",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search directories",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=30,
        help="Timeout for HTTP requests in seconds",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests",
    )

    parser.add_argument(
        "--retry-delay", type=int, default=1, help="Delay between retries in seconds"
    )

    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=100,
        help="Maximum number of concurrent requests",
    )

    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        default=["*.md", "*.html", "*.ipynb"],
        help="File patterns to include",
    )

    parser.add_argument(
        "-e", "--exclude", nargs="+", default=[], help="File patterns to exclude"
    )

    parser.add_argument(
        "--status-codes",
        type=lambda s: {int(x) for x in s.split(",")},
        default="200,201,202,203,204,205,206,207,208,226,301,302,307,308",
        help="Comma-separated list of acceptable HTTP status codes",
    )

    parser.add_argument(
        "-u",
        "--user-agent",
        default="DocLinkChecker/1.0",
        help="User agent to use for HTTP requests",
    )

    parser.add_argument(
        "--no-check-fragments",
        action="store_false",
        dest="check_fragments",
        help="Skip checking URL fragments (anchors)",
    )

    parser.add_argument(
        "--no-verify-ssl",
        action="store_false",
        dest="verify_ssl",
        help="Skip SSL certificate verification",
    )

    parser.add_argument(
        "--skip-external", action="store_true", help="Skip checking external links"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument("-o", "--output", type=str, help="File to save the results")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with a non-zero status if any errors are found",
    )
    parser.add_argument(
        "--follow-redirects", action="store_true", help="Follow HTTP redirects"
    )
    parser.add_argument(
        "--max-redirects",
        type=int,
        default=5,
        help="Maximum number of redirects to follow",
    )
    parser.add_argument(
        "--ignore-urls",
        nargs="+",
        default=[],
        help="Regular expressions for URLs to ignore",
    )
    parser.add_argument(
        "--ignore-url-file",
        type=str,
        help="Path to a file containing URL patterns to ignore (one per line, like .gitignore)",
    )
    parser.add_argument(
        "--headers",
        nargs="+",
        default=[],
        help="Custom headers to include in requests (key:value)",
    )
    parser.add_argument("--skip", action="store_true", help="Skip checking links")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    return parser.parse_args()


async def main(args):
    # Parse headers from key:value format
    headers = {}
    for header in args.headers:
        if ":" in header:
            key, value = header.split(":", 1)
            headers[key.strip()] = value.strip()

    # Load ignore patterns from file if provided
    ignore_urls = list(
        args.ignore_urls
    )  # Make a copy to avoid modifying the original args
    if args.ignore_url_file:
        try:
            with open(args.ignore_url_file, "r", encoding="utf-8") as f:
                for line in f:
                    pattern = line.strip()
                    # Ignore empty lines and comments (like .gitignore)
                    if pattern and not pattern.startswith("#"):
                        ignore_urls.append(pattern)
        except FileNotFoundError:
            console = Console()
            console.print(
                f"[error]Ignore URL file not found: {args.ignore_url_file}[/error]"
            )
            sys.exit(1)
        except Exception as e:
            console = Console()
            console.print(
                f"[error]Error reading ignore URL file {args.ignore_url_file}: {e}[/error]"
            )
            sys.exit(1)

    # Create configuration
    config = LinkCheckerConfig(
        directories=args.directories,
        recursive=args.recursive,
        timeout=args.timeout,
        max_retries=args.retries,
        retry_delay=args.retry_delay,
        concurrency=args.concurrency,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        acceptable_status_codes=args.status_codes,
        user_agent=args.user_agent,
        check_fragments=args.check_fragments,
        verify_ssl=args.verify_ssl,
        skip_external=args.skip_external,
        verbose=args.verbose,
        output_file=args.output,
        fail_on_error=args.fail_on_error,
        follow_redirects=args.follow_redirects,
        max_redirects=args.max_redirects,
        ignore_urls=ignore_urls,  # Use the combined list
        headers=headers,
    )

    # Create and run the link checker
    checker = LinkChecker(config)

    if args.skip:
        console = Console()
        console.print("[warning]Link checking skipped due to --skip flag[/warning]")
        return 0

    # Run the link checker
    summary = await checker.check_links()

    # Print the summary
    checker.print_summary()

    # Save the results if requested
    if config.output_file:
        checker.save_results()

    # Return exit code based on results
    if config.fail_on_error and summary.error_links > 0:
        return 1

    return 0


if __name__ == "__main__":
    args = parse_arguments()

    if args.debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    try:
        exit_code = asyncio.run(main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nLink checking interrupted by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        console = Console()
        console.print(f"[error]Error: {str(e)}[/error]")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)
