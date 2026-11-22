# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import functools
import http.server
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_FIXTURE_DIR = REPO_ROOT / "tests/fixtures/models/lmlite/tiny-unigram"
ONNX_FIXTURE_DIR = REPO_ROOT / "tests/fixtures/onnx"
# Both xeus tests share one site: it is built once per session.
ENVIRONMENT = """name: xeus-kernels
channels:
  - https://prefix.dev/emscripten-forge-4x
  - https://prefix.dev/conda-forge
dependencies:
  - xeus-python
  - tqdm
  - ipywidgets
  - numpy
  - pillow
  - anywidget
"""


@pytest.fixture(scope="module")
def source_server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture(scope="session")
def xeus_site(tmp_path_factory):
    jupyter_lite = shutil.which("jupyter-lite")
    if jupyter_lite is None:
        pytest.skip("jupyterlite-core and jupyterlite-xeus are required; run `pixi run xeus-tokenizer-smoke`")
    root = tmp_path_factory.mktemp("xeus")
    models = root / "content/models"
    shutil.copytree(TOKENIZER_FIXTURE_DIR, models / "lmlite/tiny-unigram")
    (models / "manifest.json").write_text(json.dumps({"models": ["lmlite/tiny-unigram"]}))
    shutil.copytree(ONNX_FIXTURE_DIR, root / "content/onnx")
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
