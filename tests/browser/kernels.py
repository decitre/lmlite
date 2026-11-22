# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Run Python code in a real Pyodide or xeus-python kernel and read back one RESULT line it prints."""

import json
import os
import urllib.parse

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "300000"))


def webgpu_args(backend="webgpu"):
    """Chromium flags for a run on `backend`. Headless Chromium gives WebGPU a software adapter (no float16
    shaders, 1 GB buffers) unless told to use the GPU: set LMLITE_WEBGPU_ARGS, for example on macOS to
    "--enable-unsafe-webgpu --enable-gpu --use-angle=metal --ignore-gpu-blocklist"."""
    if backend != "webgpu":
        return []
    return os.getenv("LMLITE_WEBGPU_ARGS", "--enable-unsafe-webgpu").split()


PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js"


def run_in_pyodide(source_server, code, packages=("tqdm",), requests=None, args=()):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=list(args))
        try:
            page = browser.new_page()
            page.set_default_timeout(TIMEOUT)
            if requests is not None:
                page.on("request", lambda request: requests.append(request.url))
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=PYODIDE_URL)
            output = page.evaluate(
                """async ({server, code, packages}) => {
                  const pyodide = await loadPyodide();
                  await pyodide.loadPackage(packages);
                  pyodide.FS.mkdirTree('/lib/lmlite/bridge');
                  for (const name of ['__init__.py', 'llm.py', 'llm.js', 'onnx_model.py', 'version.py', 'bridge/__init__.py', 'bridge/runtime.py']) {
                    pyodide.FS.writeFile(`/lib/lmlite/${name}`, await (await fetch(`${server}/src/lmlite/${name}`)).text());
                  }
                  pyodide.runPython("import sys; sys.path.insert(0, '/lib')");
                  let printed = '';
                  pyodide.setStdout({batched: (line) => { printed += line + '\\n'; }});
                  await pyodide.runPythonAsync(code);
                  return printed;
                }""",
                {"server": source_server, "code": code, "packages": list(packages)},
            )
        finally:
            browser.close()
    line = next(line for line in output.splitlines() if line.startswith("RESULT"))
    return json.loads(line.removeprefix("RESULT"))


def run_in_xeus(xeus_site, code, requests=None, args=()):
    url = f"{xeus_site}/repl/index.html?kernel=xpython&execute=1&code={urllib.parse.quote(code)}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=list(args))
        try:
            page = browser.new_page()
            console = []
            page.on("console", lambda message: console.append(f"{message.type}: {message.text[:300]}"))
            if requests is not None:
                page.on("request", lambda request: requests.append(request.url))
            page.goto(url)
            # The echoed input never contains 'RESULT{', so this waits for kernel output.
            try:
                page.wait_for_function(
                    "document.body.innerText.includes('RESULT{') || document.body.innerText.includes('Traceback')",
                    timeout=TIMEOUT,
                )
            except PlaywrightTimeoutError:
                output = page.inner_text("body").split("]:", 1)[-1][-2500:]
                pytest.fail(
                    f"Kernel gave no result in {TIMEOUT} ms. Page end:\n{output}\nConsole:\n" + "\n".join(console[-10:])
                )
            text = page.inner_text("body")
        finally:
            browser.close()
    lines = [line for line in text.splitlines() if line.startswith("RESULT{")]
    assert lines, f"Kernel failed:\n{text[-3000:]}"
    return json.loads(lines[-1].removeprefix("RESULT"))
