# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

# Sphinx with MyST (Markdown) and Furo. Build: sphinx-build -W -b html docs docs/_build
from lmlite import __version__

project = "lmLite"
author = "Emmanuel Decitre"
release = version = __version__

extensions = ["myst_parser"]
html_theme = "furo"
html_logo = "logo.svg"

# Headings get anchors, so links such as guides/tokenizers.md#tokenize-only-handles are checked by the build.
myst_heading_anchors = 3

# USAGE.md only points at the new pages, for links that predate the split; the pages above it are in the toctrees.
exclude_patterns = ["_build", "USAGE.md", ".DS_Store"]

# llms.txt (an index of these pages for language models) is served at the site root.
html_extra_path = ["llms.txt"]
