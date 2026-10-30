# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""LLM.tokenizer() in a real xeus-python kernel, built into a JupyterLite site from the current source."""

import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.parse
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/models/lmlite/tiny-unigram"
TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "300000"))
ENVIRONMENT = """name: xeus-kernels
channels:
  - https://prefix.dev/emscripten-forge-dev
  - https://prefix.dev/conda-forge
dependencies:
  - xeus-python
  - tqdm
  - ipywidgets
"""

# Runs in the kernel; prints one RESULT line with everything the test asserts on.
KERNEL_CODE = r"""
import json, time
from lmlite import LLM

cases = json.load(open("models/lmlite/tiny-unigram/fixture.json", encoding="utf-8"))["cases"]
res = {}
tok = await LLM.tokenizer("lmlite/tiny-unigram")
bare = await LLM.tokenizer(tokenizer_json="files/models/lmlite/tiny-unigram/tokenizer.json")
res["vocab_size"] = tok.vocab_size
res["ids"] = [tok.encode(c["text"]) for c in cases]
res["ids_with_special_tokens"] = [tok.encode(c["text"], add_special_tokens=True) for c in cases]
res["bare_ids"] = [bare.encode(c["text"]) for c in cases]
res["plain_ints"] = all(type(ids) is list and all(type(i) is int for i in ids) for ids in res["ids"])
res["decoded"] = tok.decode(res["ids"][0])
start = time.perf_counter()
for _ in range(1000):
    tok.encode(cases[0]["text"])
res["mean_ms"] = time.perf_counter() - start
try:
    tok.encode(b"x")
except TypeError:
    res["type_error"] = True
tok.close()
tok.close()
try:
    tok.encode("x")
except RuntimeError as exc:
    res["closed"] = str(exc)
try:
    await LLM.tokenizer(tokenizer_json="files/models/missing/tokenizer.json")
except Exception as exc:
    res["missing"] = str(exc)
print("RESULT" + json.dumps(res))
"""


@pytest.fixture(scope="module")
def xeus_site(tmp_path_factory):
    jupyter_lite = shutil.which("jupyter-lite")
    if jupyter_lite is None:
        pytest.skip("jupyterlite-core and jupyterlite-xeus are required; run `pixi run xeus-tokenizer-smoke`")
    root = tmp_path_factory.mktemp("xeus")
    models = root / "content/models"
    shutil.copytree(FIXTURE_DIR, models / "lmlite/tiny-unigram")
    (models / "manifest.json").write_text(json.dumps({"models": ["lmlite/tiny-unigram"]}))
    (root / "environment.yml").write_text(ENVIRONMENT)
    subprocess.run(
        [
            jupyter_lite,
            "build",
            "--contents",
            str(root / "content"),
            "--output-dir",
            str(root / "site"),
            f"--XeusAddon.environment_file={root / 'environment.yml'}",
            "--XeusAddon.mount_jupyterlite_content=True",
            f"--XeusAddon.mounts={REPO_ROOT / 'src/lmlite'}:/lib/python3.13/site-packages/lmlite",
        ],
        cwd=root,
        check=True,
        stdout=sys.stderr,
    )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root / "site"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_xeus_encode_matches_reference_with_plain_ints(xeus_site):
    url = f"{xeus_site}/repl/index.html?kernel=xpython&execute=1&code={urllib.parse.quote(KERNEL_CODE)}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            requests = []
            page.on("request", lambda request: requests.append(request.url))
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
    result = json.loads(lines[-1].removeprefix("RESULT"))
    print(f"xeus-python encode mean: {result['mean_ms']:.4f} ms")

    fixture = json.loads((FIXTURE_DIR / "fixture.json").read_text(encoding="utf-8"))
    assert result["plain_ints"]
    assert result["ids"] == [case["ids"] for case in fixture["cases"]]
    assert result["ids_with_special_tokens"] == [case["ids_with_special_tokens"] for case in fixture["cases"]]
    assert result["bare_ids"] == result["ids"]
    assert result["vocab_size"] == fixture["vocab_size"]
    assert result["decoded"].strip() == fixture["cases"][0]["text"]
    assert result["type_error"]
    assert "closed" in result["closed"]
    assert "Failed to load tokenizer from 'files/models/missing/tokenizer.json'" in result["missing"]
    weights = (".onnx", ".onnx_data", ".safetensors", ".bin")
    assert not [u for u in requests if "huggingface.co" in u or u.split("?", 1)[0].endswith(weights)]
