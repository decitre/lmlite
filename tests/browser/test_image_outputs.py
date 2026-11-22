# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for the tasks that return images (downloads models from the Hub)."""

import os
import struct

import pytest
from kernels import TIMEOUT, run_in_pyodide, run_in_xeus
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

TIGER_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg"
MODELS = {
    "background-removal": os.getenv("LMLITE_TEST_BACKGROUND_REMOVAL_MODEL", "Xenova/modnet"),
    "depth-estimation": os.getenv("LMLITE_TEST_DEPTH_MODEL", "onnx-community/depth-anything-v2-small"),
    "image-segmentation": os.getenv("LMLITE_TEST_SEGMENTATION_MODEL", "Xenova/detr-resnet-50-panoptic"),
    "image-to-image": os.getenv("LMLITE_TEST_IMAGE_TO_IMAGE_MODEL", "Xenova/swin2SR-lightweight-x2-64"),
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_size(data):
    """(width, height) from a PNG's IHDR chunk, after checking the signature."""
    data = bytes(data)
    assert data[:8] == PNG_SIGNATURE and data[12:16] == b"IHDR", "not a PNG file"
    return struct.unpack(">II", data[16:24])


# In the page: runs the task on the tiger photo and returns each image buffer's size and first 32 bytes.
RUN = """async ({task, model, url, inWorker}) => {
  const run = async ({task, model, url}) => {
    const llm = globalThis.lmLite.createLLM({task, task_model: model, backend: 'wasm'});
    await llm.init();
    const [structure, ...buffers] = await llm.runTask(url, '{}');
    await llm.dispose();
    return {structure: JSON.parse(structure), heads: buffers.map(b => [...b.subarray(0, 32)]), sizes: buffers.map(b => b.byteLength),
            canvas: typeof document === 'undefined' ? 'worker' : 'page'};
  };
  if (!inWorker) {
    await new Promise(resolve => setTimeout(resolve, 0));
    try { return {ok: true, ...await run({task, model, url})}; } catch (error) { return {ok: false, error: String(error?.stack ?? error)}; }
  }
  const code = `self.onmessage = async ({data}) => {
    try {
      await import(data.src);
      const llm = globalThis.lmLite.createLLM({task: data.task, task_model: data.model, backend: 'wasm'});
      await llm.init();
      const [structure, ...buffers] = await llm.runTask(data.url, '{}');
      await llm.dispose();
      self.postMessage({ok: true, structure: JSON.parse(structure), heads: buffers.map(b => [...b.subarray(0, 32)]),
                        sizes: buffers.map(b => b.byteLength), canvas: typeof document === 'undefined' ? 'worker' : 'page'});
    } catch (error) { self.postMessage({ok: false, error: String(error?.stack ?? error)}); }
  };`;
  const worker = new Worker(URL.createObjectURL(new Blob([code], {type: 'text/javascript'})), {type: 'module'});
  try {
    return await new Promise((resolve) => {
      worker.onmessage = ({data}) => resolve(data);
      worker.onerror = (event) => resolve({ok: false, error: event.message});
      worker.postMessage({src: new URL('/src/lmlite/llm.js', location.href).href, task, model, url});
    });
  } finally { worker.terminate(); }
}"""


def run_in_chromium(source_server, task, in_worker):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            result = page.evaluate(RUN, {"task": task, "model": MODELS[task], "url": TIGER_URL, "inWorker": in_worker})
        finally:
            browser.close()
    assert result["ok"], result.get("error")
    return result


@pytest.mark.parametrize("task", MODELS)
def test_each_task_returns_png_images_with_the_right_sizes(source_server, task):
    result = run_in_chromium(source_server, task, in_worker=False)
    structure, heads = result["structure"], result["heads"]
    if task == "depth-estimation":
        assert structure["predicted_depth"]["dtype"] == "float32" and structure["predicted_depth"]["shape"] == [
            408,
            612,
        ]
        assert result["sizes"][structure["predicted_depth"]["$buffer"]] == 408 * 612 * 4
        assert png_size(heads[structure["depth"]["$buffer"]]) == (612, 408)
    elif task == "image-segmentation":
        assert structure and all(isinstance(item["label"], str) for item in structure)
        assert all(png_size(heads[item["mask"]["$buffer"]]) == (612, 408) for item in structure)
    elif task == "background-removal":
        assert [png_size(head) for head in heads] == [(612, 408)]
        assert heads[0][25] == 6  # an RGBA PNG: the cut-out has an alpha channel
    else:  # image-to-image: a 2x upscale, padded to the model's window
        width, height = png_size(heads[0])
        assert 2 * 612 <= width <= 2 * 616 and height >= 2 * 408


def test_png_encoding_works_inside_a_web_worker(source_server):
    """Notebook kernels run in a worker, which has no <canvas>: the PNG must come from OffscreenCanvas."""
    result = run_in_chromium(source_server, "background-removal", in_worker=True)

    assert result["canvas"] == "worker"
    assert png_size(result["heads"][0]) == (612, 408) and result["structure"]["channels"] == 4


# Runs in a Python kernel: every task through LLM.pipeline, with PNG bytes and, for one, NumPy arrays.
KERNEL_CODE = r"""
import json, struct
import numpy as np
from lmlite import LLM

URL = "TIGER"

def size(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR"
    return list(struct.unpack(">II", png[16:24]))

res = {}
for task, model in MODELS.items():
    pipe = await LLM.pipeline(task, model=model)
    out = await pipe(URL)
    if task == "depth-estimation":
        res[task] = {"depth": size(out["depth"]), "predicted": [type(out["predicted_depth"]).__name__, list(out["predicted_depth"].shape), str(out["predicted_depth"].dtype)]}
    elif task == "image-segmentation":
        res[task] = {"segments": len(out), "labels_are_str": all(isinstance(s["label"], str) for s in out), "mask": size(out[0]["mask"])}
    else:
        res[task] = {"png": size(out)}
    if task == "background-removal":
        array = await pipe(URL, as_array=True)
        res[task]["array"] = [type(array).__name__, list(array.shape), str(array.dtype)]
        many = await pipe.map([URL, URL])
        res[task]["map"] = [size(item) for item in many]
    await pipe.dispose()
print("RESULT" + json.dumps(res))
"""


def kernel_code():
    from json import dumps

    return KERNEL_CODE.replace("MODELS", dumps(MODELS)).replace("TIGER", TIGER_URL)


def check_kernel_result(result):
    assert result["background-removal"]["png"] == [612, 408]
    assert result["background-removal"]["array"] == ["ndarray", [408, 612, 4], "uint8"]
    assert result["background-removal"]["map"] == [[612, 408], [612, 408]]
    assert result["depth-estimation"]["depth"] == [612, 408]
    assert result["depth-estimation"]["predicted"] == ["ndarray", [408, 612], "float32"]
    assert result["image-segmentation"]["segments"] >= 1 and result["image-segmentation"]["labels_are_str"]
    assert result["image-segmentation"]["mask"] == [612, 408]
    assert result["image-to-image"]["png"][0] >= 2 * 612


def test_image_outputs_in_pyodide(source_server):
    check_kernel_result(run_in_pyodide(source_server, kernel_code(), packages=("tqdm", "numpy")))


def test_image_outputs_in_xeus(xeus_site):
    check_kernel_result(run_in_xeus(xeus_site, kernel_code()))
