# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""docs/llms.txt is an index of the docs for language models: every page it links to must exist."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
URL = re.compile(r"\]\(https://raw\.githubusercontent\.com/decitre/lmlite/main/([^)]+)\)")


def test_every_linked_page_exists():
    text = (ROOT / "docs/llms.txt").read_text()
    pages = URL.findall(text)

    assert len(pages) >= 10 and text.startswith("# lmLite\n\n> ")
    assert [page for page in pages if not (ROOT / page).is_file()] == []


def test_every_guide_and_reference_page_is_listed():
    text = (ROOT / "docs/llms.txt").read_text()
    pages = {
        path.relative_to(ROOT).as_posix()
        for folder in ("guides", "reference", "explanation")
        for path in (ROOT / "docs" / folder).glob("*.md")
        if path.name != "index.md"
    }

    assert sorted(page for page in pages if f"/{page})" not in text) == []
