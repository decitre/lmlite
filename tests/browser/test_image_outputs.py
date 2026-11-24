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
CATS_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"
# DETR panoptic: the quantized default returns wrong masks (one segment over the whole photo) and fp16 can fail
# with std::bad_alloc in a fresh page, so ask for fp32.
DTYPES = {"image-segmentation": "fp32"}
URLS = {"image-segmentation": CATS_URL}
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
RUN = """async ({task, model, url, dtype, inWorker}) => {
  const run = async ({task, model, url, dtype}) => {
    const llm = globalThis.lmLite.createRuntime({task, task_model: model, backend: 'wasm', ...(dtype ? {dtype} : {})});
    await llm.init();
    const [structure, ...buffers] = await llm.runTask(url, '{}');
    await llm.dispose();
    return {structure: JSON.parse(structure), heads: buffers.map(b => [...b.subarray(0, 32)]), sizes: buffers.map(b => b.byteLength),
            canvas: typeof document === 'undefined' ? 'worker' : 'page'};
  };
  if (!inWorker) {
    await new Promise(resolve => setTimeout(resolve, 0));
    try { return {ok: true, ...await run({task, model, url, dtype})}; } catch (error) { return {ok: false, error: String(error?.stack ?? error)}; }
  }
  const code = `self.onmessage = async ({data}) => {
    try {
      await import(data.src);
      const llm = globalThis.lmLite.createRuntime({task: data.task, task_model: data.model, backend: 'wasm', ...(data.dtype ? {dtype: data.dtype} : {})});
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
      worker.postMessage({src: new URL('/src/lmlite/lmlite.js', location.href).href, task, model, url, dtype});
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
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
            args = {
                "task": task,
                "model": MODELS[task],
                "url": URLS.get(task, TIGER_URL),
                "dtype": DTYPES.get(task),
                "inWorker": in_worker,
            }
            result = page.evaluate(RUN, args)
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
        labels = [item["label"] for item in structure]
        assert labels.count("cat") == 2 and "couch" in labels  # the two cats photo, with fp32
        assert len({png_size(heads[item["mask"]["$buffer"]]) for item in structure}) == 1
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


# Runs in a Python kernel: every task through LLM.pipeline, with PNG bytes, NumPy arrays and the content of the images.
KERNEL_CODE = r"""
import json, struct
import numpy as np
from lmlite import LLM

URLS = __URLS__
DTYPES = __DTYPES__
TIGER = "__TIGER__"

def size(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR"
    return list(struct.unpack(">II", png[16:24]))

res = {}
for task, model in __MODELS__.items():
    url = URLS.get(task, TIGER)
    pipe = await LLM.pipeline(task, model=model, **({"dtype": DTYPES[task]} if task in DTYPES else {}))
    out = await pipe(url)
    pixels = await pipe(url, as_array=True)  # the same result as arrays, to look at what is in the images
    if task == "depth-estimation":
        depth = pixels["depth"][..., 0].astype(np.float32)
        res[task] = {"depth": size(out["depth"]), "depth_std": float(depth.std()), "depth_range": [float(depth.min()), float(depth.max())],
                     "predicted": [type(out["predicted_depth"]).__name__, list(out["predicted_depth"].shape), str(out["predicted_depth"].dtype)]}
    elif task == "image-segmentation":
        res[task] = {"segments": [[s["label"], round(s["score"], 3), float((m["mask"][..., 0] == 255).mean()), size(s["mask"])]
                                  for s, m in zip(out, pixels)]}
    elif task == "background-removal":
        alpha = pixels[..., 3]
        res[task] = {"png": size(out), "array": [type(pixels).__name__, list(pixels.shape), str(pixels.dtype)],
                     "transparent": float((alpha < 128).mean()), "opaque": float((alpha > 200).mean())}
        many = await pipe.map([url, url])
        res[task]["map"] = [size(item) for item in many]
    else:
        res[task] = {"png": size(out), "std": float(pixels[..., :3].std()), "mean": float(pixels[..., :3].mean())}
    await pipe.dispose()
print("RESULT" + json.dumps(res))
"""


def kernel_code():
    from json import dumps

    return (
        KERNEL_CODE.replace("__MODELS__", dumps(MODELS))
        .replace("__URLS__", dumps(URLS))
        .replace("__DTYPES__", dumps(DTYPES))
        .replace("__TIGER__", TIGER_URL)
    )


def check_kernel_result(result):
    cutout = result["background-removal"]
    assert cutout["png"] == [612, 408] and cutout["array"] == ["ndarray", [408, 612, 4], "uint8"]
    assert cutout["map"] == [[612, 408], [612, 408]]
    assert 0.15 < cutout["transparent"] < 0.85 and cutout["opaque"] > 0.1  # a cut-out: part background, part tiger
    depth = result["depth-estimation"]
    assert depth["depth"] == [612, 408] and depth["predicted"] == ["ndarray", [408, 612], "float32"]
    assert (
        depth["depth_std"] > 20 and depth["depth_range"][1] - depth["depth_range"][0] > 150
    )  # a map, not a flat image
    segments = result["image-segmentation"]["segments"]
    by_label = {}
    for label, score, white, mask_size in segments:
        by_label.setdefault(label, []).append(white)
        assert 0.5 < score <= 1 and mask_size[0] >= 320
    assert len(by_label["cat"]) == 2 and "couch" in by_label  # two cats on a couch, not one mask over the whole photo
    assert all(0.05 < white < 0.4 for white in by_label["cat"])
    assert 0.3 < by_label["couch"][0] < 0.8
    upscale = result["image-to-image"]
    assert upscale["png"][0] >= 2 * 612 and upscale["std"] > 30 and 30 < upscale["mean"] < 220


def test_image_outputs_in_pyodide(source_server):
    check_kernel_result(run_in_pyodide(source_server, kernel_code(), packages=("tqdm", "numpy")))


def test_image_outputs_in_xeus(xeus_site):
    check_kernel_result(run_in_xeus(xeus_site, kernel_code()))
