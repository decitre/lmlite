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
    for notebook in (REPO_ROOT / "notebooks").glob("*.ipynb"):  # for the tests that open them in Lab
        shutil.copy(notebook, root / "content" / notebook.name)
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


class _CorsHandler(http.server.SimpleHTTPRequestHandler):
    """Static files that a page on another origin may fetch, as a hosted models site's files are."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture(scope="session")
def hosted_speech_models(tmp_path_factory):
    """A hosted models tree with Xenova/mms-tts-eng (its JSON files and quantized weights), served over HTTP."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    root = tmp_path_factory.mktemp("hosted-models")
    huggingface_hub.snapshot_download(
        "Xenova/mms-tts-eng",
        local_dir=root / "Xenova/mms-tts-eng",
        allow_patterns=["*.json", "onnx/model_quantized.onnx"],
    )
    (root / "manifest.json").write_text(json.dumps({"models": ["Xenova/mms-tts-eng"]}))
    handler = functools.partial(_CorsHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture(scope="session")
def hosted_onnx_models():
    """The hosted ONNX fixture tree (a manifest and lmlite/scale-add/model.onnx), served with CORS headers."""
    handler = functools.partial(_CorsHandler, directory=str(ONNX_FIXTURE_DIR / "hosted"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
