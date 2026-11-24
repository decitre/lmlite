# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""docs/reference/api.md generates the API reference from the docstrings: it must cover the public API."""

import re
from pathlib import Path

import lmlite

PAGE = Path(__file__).parents[1] / "docs/reference/api.md"
NOT_DOCUMENTED = {"__version__"}  # a string, not an API


def test_every_public_name_is_in_the_api_reference():
    documented = set(re.findall(r"^\.\. auto\w+:: lmlite\.(\w+)$", PAGE.read_text(), re.M))

    assert sorted(set(lmlite.__all__) - NOT_DOCUMENTED - documented) == []
