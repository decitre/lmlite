# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""The object-detection section of the vision notebook, run in real Pyodide and xeus-python kernels (needs network)."""

import json
from pathlib import Path

import pytest
from kernels import run_in_pyodide, run_in_xeus

pytestmark = pytest.mark.browser

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/ch13_vision.ipynb"

# Pyodide has no IPython: a stand-in keeps the notebook's own `from IPython.display import display` working.
STAND_IN = """
import sys, types
ipython = types.ModuleType("IPython")
display = types.ModuleType("IPython.display")
display.display = lambda *objects: None
ipython.display = display
sys.modules.update({"IPython": ipython, "IPython.display": display})
"""

# After the notebook's cells: what the test asserts on, including that the drawing changed the photo.
SUMMARY = """
from PIL import ImageChops

def summary(detections):
    return "error" if isinstance(detections, Exception) else len(detections)

drawn = [
    ImageChops.difference(image, annotate(image, detections)).getbbox() is not None
    for image, detections in zip(images, results)
    if isinstance(detections, list) and detections
]
print("RESULT" + json.dumps({
    "photos": len(photos),
    "results": [summary(r) for r in results],
    "drawn": drawn,
    "repeats": len(first_titles & {title for title, _ in photos}),
    "seen": len(seen),
}))
"""


def notebook_code(stand_in=""):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    start = next(i for i, cell in enumerate(cells) if "".join(cell["source"]).startswith("## Object detection"))
    sources = ["".join(cell["source"]) for cell in cells[start:] if cell["cell_type"] == "code"]
    picking = next(source for source in sources if "count = 8" in source)
    # The picking cell runs a second time, to see that it shows photos it has not shown before.
    again = f"first_titles = {{title for title, _ in photos}}\n{picking}"
    code = "\n\n".join(source + (f"\n\n{again}" if source is picking else "") for source in sources)
    # Seeded, so the photos are the same between runs.
    return f"{stand_in}\nimport random\nrandom.seed(0)\nfrom lmlite import LLM\n{code}\n{SUMMARY}"


def check(result):
    assert result["photos"] == 8 and len(result["results"]) == 8
    found = [count for count in result["results"] if isinstance(count, int) and count > 0]
    assert found, f"No object found in 8 photos: {result['results']}"
    assert result["drawn"] and all(result["drawn"])  # the boxes and labels were drawn onto the photos
    assert result["repeats"] == 0 and result["seen"] == 16  # the second pick shows 8 new photos


def test_detections_are_drawn_in_pyodide(source_server):
    check(run_in_pyodide(source_server, notebook_code(STAND_IN), packages=("tqdm", "pillow", "pyodide-http")))


def test_detections_are_drawn_in_xeus(xeus_site):
    check(run_in_xeus(xeus_site, notebook_code()))
