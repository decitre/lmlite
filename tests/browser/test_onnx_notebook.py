# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""The bare-ONNX notebook's code, run in real Pyodide and xeus-python kernels (needs network access)."""

import json
import os
import urllib.parse
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/ch14_onnx-models.ipynb"
TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "300000"))
PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js"

# Seeded, so the random photo is the same between runs; the notebook looks for a face in up to 8 photos.
SUMMARY = """
print("RESULT" + json.dumps({
    "faces": [[box.tolist(), score] for box, score in faces],
    "author": author,
    "info": model.model_info(),
}))
"""


def notebook_code():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    code = "\n\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    return f"import json, random\nrandom.seed(0)\n{code}\n{SUMMARY}"


def check(result):
    assert result["faces"], "No face found in the random photos"
    for box, score in result["faces"]:
        assert 0.7 < score <= 1 and all(0 <= value <= 1 for value in box) and box[0] < box[2] and box[1] < box[3]
    assert result["author"] and result["info"]["size"] == 1270727
    assert result["info"]["backend"] == "wasm" and result["info"]["cache_status"] in {"downloaded", "cached"}


def test_notebook_runs_in_pyodide(source_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=PYODIDE_URL)
            output = page.evaluate(
                """async ({server, code}) => {
                  const pyodide = await loadPyodide();
                  await pyodide.loadPackage(['numpy', 'pillow', 'pyodide-http']);
                  pyodide.FS.mkdirTree('/lib/lmlite/bridge');
                  for (const name of ['__init__.py', '_bridge.py', '_session.py', 'classifier.py', 'embedder.py', 'llm.py', 'lmlite.js', 'models.py', 'onnx_model.py', 'pipeline.py', 'tokenizer.py', 'tools.py', 'version.py', 'bridge/__init__.py', 'bridge/runtime.py']) {
                    pyodide.FS.writeFile(`/lib/lmlite/${name}`, await (await fetch(`${server}/src/lmlite/${name}`)).text());
                  }
                  pyodide.runPython("import sys; sys.path.insert(0, '/lib')");
                  let printed = '';
                  pyodide.setStdout({batched: (line) => { printed += line + '\\n'; }});
                  await pyodide.runPythonAsync(code);
                  return printed;
                }""",
                {"server": source_server, "code": notebook_code()},
            )
        finally:
            browser.close()
    check(json.loads(next(line for line in output.splitlines() if line.startswith("RESULT")).removeprefix("RESULT")))


def test_notebook_runs_in_xeus(xeus_site):
    url = f"{xeus_site}/repl/index.html?kernel=xpython&execute=1&code={urllib.parse.quote(notebook_code())}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url)
            # The echoed input never contains 'RESULT{', so this waits for kernel output.
            page.wait_for_function(
                "document.body.innerText.includes('RESULT{') || document.body.innerText.includes('Traceback')",
                timeout=TIMEOUT,
            )
            text = page.inner_text("body")
        finally:
            browser.close()
    lines = [line for line in text.splitlines() if line.startswith("RESULT{")]
    assert lines, f"Kernel failed:\n{text[-3000:]}"
    check(json.loads(lines[-1].removeprefix("RESULT")))
