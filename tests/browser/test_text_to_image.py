# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""The text-to-image notebook's code on a hardware GPU, in real Pyodide and xeus-python kernels.

Opt-in beyond `-m browser`: it needs a local copy of the 2.4 GB SD-Turbo files (non-commercial research licence,
https://huggingface.co/schmuell/sd-turbo-ort-web) in a directory named by LMLITE_SD_TURBO_DIR, with text_encoder/,
unet/ and vae_decoder/ each holding a model.onnx, and a browser with a hardware WebGPU adapter that has float16
shaders. LMLITE_WEBGPU_ARGS overrides the Chromium flags that select the GPU (default: Metal on macOS).
"""

import functools
import http.server
import json
import os
import shlex
import sys
import threading
from pathlib import Path

import pytest
from kernels import run_in_pyodide, run_in_xeus

pytestmark = pytest.mark.browser

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/ch08g_text-to-image.ipynb"
HUB_URL = "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main"
DEFAULT_ARGS = "--enable-unsafe-webgpu --enable-gpu --ignore-gpu-blocklist" + (
    " --use-angle=metal" if sys.platform == "darwin" else ""
)
MODEL_DIR = os.getenv("LMLITE_SD_TURBO_DIR")
GPU_ARGS = shlex.split(os.getenv("LMLITE_WEBGPU_ARGS", DEFAULT_ARGS))

# Pyodide has no IPython: a stand-in keeps the notebook's own `from IPython.display import ...` working.
STAND_IN = """
import sys, types
ipython = types.ModuleType("IPython")
display = types.ModuleType("IPython.display")
display.display = lambda *objects: None
display.Image = lambda *args, **kwargs: None
ipython.display = display
sys.modules.update({"IPython": ipython, "IPython.display": display})
"""

# After the generating cell, before the notebook releases the models: a second image with another seed.
SECOND = """
other, _ = await generate("a photo of an astronaut riding a horse on mars", seed=8)
"""

# After all the cells: what the test asserts on.
SUMMARY = """
def decode(data):
    import io
    return np.asarray(PilImage.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
first, second = decode(png), decode(other)
print("RESULT" + json.dumps({
    "shape": list(first.shape), "std": float(first.std()), "mean": float(first.mean()),
    "difference": float(np.abs(first - second).mean()), "seconds": seconds, "load_seconds": load_seconds,
    "adapter": adapter, "png_signature": png[:8].hex(),
}))
"""


@pytest.fixture(scope="module")
def models_url():
    if not MODEL_DIR or not all(
        (Path(MODEL_DIR) / part / "model.onnx").is_file() for part in ("text_encoder", "unet", "vae_decoder")
    ):
        pytest.skip("LMLITE_SD_TURBO_DIR does not point at a directory with the SD-Turbo model files")

    class Cors(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Cors, directory=MODEL_DIR))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def notebook_code(models_url, stand_in=""):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    code = "\n\n".join(source + (SECOND if "seed=7" in source else "") for source in sources)
    assert HUB_URL in code
    return f"{stand_in}\n{code.replace(HUB_URL, models_url)}\n{SUMMARY}"


def check(result):
    assert result["png_signature"] == "89504e470d0a1a0a" and result["shape"] == [512, 512, 3]
    assert result["std"] > 20, "the image is flat"  # an astronaut on Mars, not a blank or noise-free frame
    assert result["difference"] > 5, "two seeds gave the same image"
    assert result["adapter"]["shader_f16"] and result["adapter"]["fallback"] is not True
    assert 0 < result["seconds"]["total"] < 60 and set(result["load_seconds"]) == {
        "text_encoder",
        "unet",
        "vae_decoder",
    }


def test_text_to_image_in_pyodide(source_server, models_url):
    result = run_in_pyodide(
        source_server, notebook_code(models_url, STAND_IN), packages=("tqdm", "numpy", "pillow"), args=GPU_ARGS
    )
    check(result)


def test_text_to_image_in_xeus(xeus_site, models_url):
    check(run_in_xeus(xeus_site, notebook_code(models_url), args=GPU_ARGS))
