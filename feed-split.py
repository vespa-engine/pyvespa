#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import copy
import json
import sys
from bs4 import BeautifulSoup
from markdownify import markdownify
import random
import re
from xml.sax.saxutils import escape
import tiktoken
import urllib.parse
import shutil
import os
import hashlib
from pathlib import Path

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
note_pattern = re.compile(r"{%\s*include.*?%}", flags=re.DOTALL)
highlight_pattern = re.compile(r"{%\s*.*?\s%}", flags=re.DOTALL)

exclude_feeding = ["/404.html"]

WRITE_MARKDOWN = True  # Set to True to write content to markdown files for debugging
MARKDOWN_DIR = "markdown_after"  # Directory to write markdown files to


def what_language(el):
    z = re.match(r"{%\s*highlight\s*(\w+)\s%}", el.text)
    if z:
        lang = z.group(1)
        return lang
    if el.text.find("curl") > 0:
        return "bash"
    if el.text.find("import com.yahoo") > 0:
        return "java"
    return ""


def remove_jekyll(text):
    text = note_pattern.sub("", text)
    text = highlight_pattern.sub("", text)
    return text


def xml_fixup(text):
    regex = r"{%\s*highlight xml\s*%}(.*?){%\s*endhighlight\s*%}"
    matches = re.findall(regex, text, re.DOTALL)
    for match in matches:
        escaped_match = escape(match)
        text = text.replace(match, escaped_match)
    return text


def create_text_doc(doc, paragraph, paragraph_id, header):
    id = doc["put"]
    # id:open:doc::open/en/access-logging.html#
    _, namespace, doc_type, _, id = id.split(":")
    # print("n={},doc_type={},id={}".format(namespace,doc_type,id))

    new_namespace = namespace + "-p"
    id = "id:{}:{}::{}".format(new_namespace, "paragraph", id)
    fields = doc["fields"]
    n_tokens = len(encoding.encode(paragraph))
    new_doc = {
        "put": id,
        "fields": {
            "title": fields["title"],
            "path": fields["path"],
            "doc_id": fields["path"],
            "namespace": new_namespace,
            "content": paragraph,
            "content_tokens": n_tokens,
            "base_uri": sys.argv[2],
        },
    }

    if header:
        title = fields["title"]
        new_title = title + " - " + header
        new_doc["fields"]["title"] = new_title

    if paragraph_id is None:
        paragraph_id = str(random.randint(0, 1000))

    new_doc["fields"]["path"] = new_doc["fields"]["path"] + "#" + paragraph_id
    new_doc["put"] = new_doc["put"] + "-" + urllib.parse.quote(paragraph_id)

    return new_doc


def is_function_signature(line: str) -> bool:
    line = line.replace("\\", "")
    # Exclude __init__-functions, as these belong with their class.
    if line.startswith("__init__"):
        return False
    # Updated regex pattern to match function signatures in the given HTML
    pattern = r"^[a-z_]+\s*\(.*"
    # Check if the line matches the pattern
    match = re.match(pattern, line)
    return bool(match)


def extract_function_name(line: str) -> str:
    line = line.replace("\\", "")
    # Regex pattern to capture the part before the "("
    pattern = r"^([a-z_]+)\(*"
    # Search for the pattern in the line
    match = re.search(pattern, line)
    # If there's a match, return the captured group (function name)
    if match:
        return match.group(1)
    return ""


def split_reference(markdown_text):
    """
    Splits the given markdown text into logical chunks based on headers and class definitions.

    Args:
        markdown_text (str): The markdown text to be split into chunks.

    Returns:
        list of tuples: Each tuple contains an identifier, header, and the chunk of text.
    """

    chunks = []  # List to hold the final chunks of text
    lines = markdown_text.split("\n")  # Split the text into lines
    current_chunk = []  # Temporary storage for the current chunk of lines
    current_header = ""  # Store the current header text
    for line in lines:
        line = line.rstrip()  # Remove any trailing whitespace
        if line.startswith("###"):
            # Handle a new header, finalize the current chunk if it exists
            if current_chunk:
                _id = "-".join(
                    current_header.split()
                ).lower()  # Create an ID from the header
                chunks.append((_id, current_header, "\n".join(current_chunk)))
                current_chunk = []
            current_header = line.strip("# ").strip()  # Update the current header
            class_header = current_header

        elif line.startswith("*class*"):
            # Detect the start of a class definition
            current_chunk.append(line)

        elif is_function_signature(line):
            # Handle a class method (not __init__), finalize the current chunk if needed, and initialize a new one
            if current_chunk:
                _id = "-".join(current_header.split()).lower()
                chunks.append((_id, current_header, "\n".join(current_chunk)))
                current_chunk = []
                current_header = class_header + "." + extract_function_name(line)
            current_chunk.append(line)
        else:
            # Add the line to the current chunk
            current_chunk.append(line)

    # Add the final chunk if it exists
    if current_chunk:
        _id = "-".join(current_header.split()).lower()
        chunks.append((_id, current_header, "\n".join(current_chunk)))

    return chunks
def get_markdown_notebook_path(path):
    markdown_noteboks = Path('markdown_noteboks')
    notebook_names = [file.stem for file in markdown_noteboks.iterdir() if file.is_file()]

    match_path = re.search(r"/([^.]+)", path)
    
    if match_path:
        rel_path = match_path.group(1)
        match_examples_path = re.search(r"examples/([^.]+)", rel_path)
        if rel_path in notebook_names:
            return f"markdown_noteboks/{rel_path}.md"               
        elif match_examples_path:
            rel_examples_path = match_examples_path.group(1)
            return f"markdown_noteboks/{rel_examples_path}.md" 
        else:
            None
           

def split_md(md, path):

    lines = md.split("\n")
    header = ""
    text = ""
    id = ""
    data = []
    in_code_block = False  # Track whether we're inside a code block

    for line in lines:

        # Toggle in_code_block state if we detect code block
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        if in_code_block and line.strip().startswith("#"):
            continue


        if line.startswith("#") and not in_code_block:
            if text:
                data.append((id, header, text))
                text = ""
            header = line.lstrip("#").strip()

            blacktick_match = re.match(r"`?", header)
            if blacktick_match:
                header = re.sub("`", "", header)

            # Check if it's a function/method signature
            if "(" in header and ")" in header:
                # Strip argument list from header
                match = re.match(r"([a-zA-Z0-9_]+)\s*\(", header)
                if match:
                    id = match.group(1)
                    header = match.group(1)
                else:
                    id = "-".join(header.split())
            elif "=" in header:
                header = header.split('=')[0]
                id = "-".join(header.split())
            else:
                # Normal header, create id based on path rule
                if path in {
                    "/",
                    "/examples",
                    "/api",
                    "/troubleshooting.html",
                }:
                    id = "-".join(header.split()).lower()
                else:
                    id = "-".join(header.split())
        else:
            text = text + "\n" + line

    data.append((id, header, text))  # Flush any last data

    return data


def split_text(path, soup = None):
    split_tables(soup)
    split_lists(soup)
    md = markdownify(
        str(soup),
        heading_style="ATX",
        code_language_callback=what_language,
        strip=["img"],
        )
    local_nb_path = get_markdown_notebook_path(path)   
    if path == "/reference-api.html":
        data = split_reference(md)
    elif local_nb_path:
        with open(local_nb_path, "r") as md_f:
            md_content = md_f.read()
        md = re.sub(r"<style.*?>.*?</style>", "", md_content, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<script.*?>.*?</script>", "", md_content, flags=re.DOTALL | re.IGNORECASE)
        data = split_md(md, path) 
    else:
        data = split_md(md, path)
    return data

def remove_notext_tags(soup):
    for remove_tag in soup.find_all(["style"]):
        remove_tag.decompose()


def split_lists(soup):
    for list in soup.body.find_all(["ul", "ol"]):
        for list_item in list.find_all("li"):
            move_linkable_item_to_single_entity(soup, list_item)


def split_tables(soup):
    top_level_tables = soup.body.find_all(
        "table", recursive=False
    )  # i.e., do not find rows in tables within tables
    for table in top_level_tables:
        tbody = table.find("tbody")  # tbody is implicit and always there
        for row in tbody.find_all("tr", recursive=False):
            move_linkable_item_to_single_entity(soup, row)


def table_header_row(row):
    thead = row.find_parent("table").find("thead")
    if thead is not None:
        return thead.find("tr")
    return None


def move_linkable_item_to_single_entity(soup, item):
    id_elem = item.find("p", {"id": True})
    if id_elem is not None:
        new_h4 = soup.new_tag("h4", id=id_elem["id"])
        new_h4.string = id_elem["id"]
        if item.name == "tr":
            header_row = table_header_row(item)
            new_container = soup.new_tag("table")
            if header_row is not None:
                new_container.insert(0, copy.copy(header_row))
        else:
            new_container = soup.new_tag(item.parent.name)  # ul or ol

        new_container.append(item)
        soup.append(new_h4)
        soup.append(new_container)


def remove_notebook_cells(text):
    """Remove pattern matching notebook cells from text. Example: ```\n[1]:\n```"""
    pattern = r"^\s*```\s*\n\s*\[\d{1,3}\]:\s*\n\s*```\s*$"
    return re.sub(pattern, "", text, flags=re.MULTILINE)


def replace_long_integer_sequences(text):
    """
    Some of the notebooks contain prints of vectors, resulting in too many (irrelevant) tokens
    This function replaces more than 10 consecutive integers in lists with the first 10 and an ellipsis
    """

    def replace_func(match):
        numbers = match.group(0).split(",")
        return ",".join(numbers[:10]) + ",..."

    pattern = r"((?:-?\d+\s*,\s*){10,})-?\d+(?:\s*,\s*-?\d+)*"
    return re.sub(pattern, replace_func, text)


def main():
    with open(sys.argv[1]) as fp:
        random.seed(42)
        docs = json.load(fp)
        operations = []
        for doc in docs:
            if doc["fields"]["path"] in exclude_feeding:
                continue
            html_doc = doc["fields"]["html"]
            html_doc = xml_fixup(html_doc)
            soup = BeautifulSoup(html_doc, "html5lib")
            remove_notext_tags(soup)
            data = split_text(doc["fields"]["path"], soup)
            for paragraph_id, header, paragraph in data:
                paragraph = paragraph.lstrip("\n").lstrip(" ")
                paragraph = paragraph.rstrip("\n")

                paragraph = re.sub(r"\n{2,}", "\n\n", paragraph)

                paragraph = re.sub(r"\n*```", "\n```", paragraph)
                paragraph = re.sub(r"```\n*", "```\n", paragraph)
                paragraph = re.sub(r"window.MathJax = {.*}", "", paragraph)
                paragraph = re.sub(
                    r"© Copyright Copyright Vespa.ai.*?\);",
                    "",
                    paragraph,
                    flags=re.DOTALL,
                )
                paragraph = re.sub("\uf0c1", "", paragraph)

                paragraph = paragraph.replace("```\njson", "```json")
                paragraph = paragraph.replace("```\nxml", "```xml")
                paragraph = paragraph.replace("```\nbash", "```bash")
                paragraph = paragraph.replace("```\nsh", "```sh")
                paragraph = paragraph.replace("```\nraw", "```\n")
                paragraph = paragraph.replace("```\njava", "```java\n")
                paragraph = paragraph.replace("\n```\n[ ]:\n```", "\n")
                paragraph = paragraph.replace("\n```\n[1]:\n```", "\n")
                # Strip backslashes to avoid double escaping
                paragraph = paragraph.replace(
                    "\\", ""
                )  # Necessary backslashes and quotes will be added when json-serialized.
                paragraph = remove_jekyll(paragraph)
                paragraph = remove_notebook_cells(paragraph)
                paragraph = replace_long_integer_sequences(paragraph)

                if paragraph:
                    paragraph_doc = create_text_doc(
                        doc, paragraph, paragraph_id, header
                    )
                    n_tokens = paragraph_doc["fields"]["content_tokens"]
                    if n_tokens > 4096:
                        print(
                            f"Warning: paragraph with {n_tokens} tokens: {paragraph_doc['fields']['path']}"
                        )
                    operations.append(paragraph_doc)

    # Merge question expansion
    questions_expansion = dict()
    with open(sys.argv[3]) as fp:
        for line in fp:
            op = json.loads(line)
            id = op["update"]
            fields = op["fields"]
            if "questions" in fields:
                questions = fields["questions"]["assign"]
                questions_expansion[id] = questions
    # Remove and recreate markdown directory
    if WRITE_MARKDOWN:
        shutil.rmtree(MARKDOWN_DIR, ignore_errors=True)
        os.makedirs(MARKDOWN_DIR, exist_ok=True)

    def safe_id(name, max_length=100):
        if len(name) <= max_length:
            return name
        hash_part = hashlib.md5(name.encode()).hexdigest()
        base = name[:max_length - len(hash_part) - 1]
        return f"{base}_{hash_part}"

    for op in operations:
        id = op["put"]
        doc_id = op["fields"]["path"]
        doc_id = doc_id.replace("/", "-")
        safe_doc_id = safe_id(doc_id)
        if WRITE_MARKDOWN:
            with open(f"{MARKDOWN_DIR}/{safe_doc_id}.md", "w") as f:
                f.write(op["fields"]["content"])
        if id in questions_expansion:
            op["fields"]["questions"] = questions_expansion[id]
        else:
            op["fields"]["questions"] = [op["fields"]["title"]]
    with open("paragraph_index.json", "w") as fp:
        json.dump(operations, fp)


if __name__ == "__main__":
    main()
