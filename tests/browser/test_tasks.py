# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for the generic task pipeline interface (downloads models from the Hub)."""

import os

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

MODEL_TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "900000"))
BACKENDS = os.getenv("LMLITE_TEST_TASK_BACKENDS", "wasm,webgpu").split(",")
AUDIO_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
DOCS_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main"
TIGER_URL = f"{DOCS_URL}/tiger.jpg"
CATS_URL = f"{DOCS_URL}/cats.jpg"

TASKS = {
    "zero-shot-classification": (
        os.getenv("LMLITE_TEST_ZERO_SHOT_MODEL", "Xenova/mobilebert-uncased-mnli"),
        "This notebook runs in a browser on a laptop computer",
        {"candidate_labels": ["technology", "sports", "cooking"]},
    ),
    "translation": (
        os.getenv("LMLITE_TEST_TRANSLATION_MODEL", "Xenova/opus-mt-en-fr"),
        "Hello, how are you?",
        {"max_new_tokens": 32},
    ),
    "automatic-speech-recognition": (
        os.getenv("LMLITE_TEST_ASR_MODEL", "Xenova/whisper-tiny.en"),
        AUDIO_URL,
        {},
    ),
    "image-classification": (
        os.getenv("LMLITE_TEST_IMAGE_CLASSIFICATION_MODEL", "Xenova/vit-base-patch16-224"),
        TIGER_URL,
        {"top_k": 3},
    ),
    "zero-shot-image-classification": (
        os.getenv("LMLITE_TEST_ZERO_SHOT_IMAGE_MODEL", "Xenova/clip-vit-base-patch32"),
        TIGER_URL,
        {"candidate_labels": ["a photo of a tiger", "a photo of food", "a photo of a face"]},
    ),
    "image-feature-extraction": (
        os.getenv("LMLITE_TEST_IMAGE_FEATURES_MODEL", "Xenova/dinov2-small"),
        TIGER_URL,
        {},
    ),
    "object-detection": (
        os.getenv("LMLITE_TEST_OBJECT_DETECTION_MODEL", "Xenova/detr-resnet-50"),
        CATS_URL,
        {"threshold": 0.9, "percentage": False},
    ),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("task", TASKS)
def test_task_pipeline_runs_in_browser(source_server, task, backend):
    model, inputs, options = TASKS[task]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--enable-unsafe-webgpu"] if backend == "webgpu" else [])
        try:
            page = browser.new_page()
            page.set_default_timeout(MODEL_TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            if backend == "webgpu" and not page.evaluate("async () => Boolean(await navigator.gpu?.requestAdapter())"):
                pytest.skip("WebGPU adapter unavailable in this browser")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            result = page.evaluate(
                """async ({task, model, backend, inputs, options}) => {
                  const llm = globalThis.lmLite.createLLM({task, task_model: model, backend});
                  try {
                    await llm.init();
                    const output = await llm.runTask(inputs, JSON.stringify(options));
                    await llm.dispose();
                    return {ok: true, output};
                  } catch (error) {
                    return {ok: false, error: String(error?.stack ?? error)};
                  }
                }""",
                {"task": task, "model": model, "backend": backend, "inputs": inputs, "options": options},
            )
        finally:
            browser.close()

    assert result["ok"], f"{task} with {model} ({backend}) failed: {result.get('error')}"
    output = result["output"]
    if task == "zero-shot-classification":
        assert sorted(output["labels"]) == sorted(options["candidate_labels"])
        assert sum(output["scores"]) == pytest.approx(1.0, abs=1e-3)
        assert output["labels"][0] == "technology"
    elif task == "translation":
        assert output["translation_text"].strip()
    elif task == "image-classification":
        assert "tiger" in output[0]["label"].lower()
    elif task == "zero-shot-image-classification":
        assert max(output, key=lambda item: item["score"])["label"] == "a photo of a tiger"
    elif task == "image-feature-extraction":
        assert len(output) > 1 and all(isinstance(value, (int, float)) for value in output)
    elif task == "object-detection":
        assert any(item["label"] == "cat" for item in output)
        assert all(item["box"]["xmax"] > 1 for item in output)  # pixels, not fractions
    else:
        assert "country" in output["text"].lower()


def test_asr_url_runs_in_worker(source_server):
    """Notebook kernels run in a Web Worker, where AudioContext is unavailable."""
    model = TASKS["automatic-speech-recognition"][0]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(MODEL_TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            result = page.evaluate(
                """async ({src, model, url}) => {
                  const code = `
                    self.onmessage = async ({data}) => {
                      try {
                        await import(data.src);
                        const llm = globalThis.lmLite.createLLM({task: 'automatic-speech-recognition', task_model: data.model, backend: 'wasm'});
                        await llm.init();
                        const output = await llm.runTask(data.url, '{}');
                        await llm.dispose();
                        self.postMessage({ok: true, audioContext: typeof AudioContext, output});
                      } catch (error) {
                        self.postMessage({ok: false, error: String(error?.stack ?? error)});
                      }
                    };`;
                  const worker = new Worker(URL.createObjectURL(new Blob([code], {type: 'text/javascript'})), {type: 'module'});
                  try {
                    return await new Promise((resolve) => {
                      worker.onmessage = ({data}) => resolve(data);
                      worker.onerror = (event) => resolve({ok: false, error: event.message});
                      worker.postMessage({src, model, url});
                    });
                  } finally {
                    worker.terminate();
                  }
                }""",
                {"src": f"{source_server}/src/lmlite/llm.js", "model": model, "url": AUDIO_URL},
            )
        finally:
            browser.close()

    assert result["ok"], f"ASR in a worker failed: {result.get('error')}"
    assert result["audioContext"] == "undefined"
    assert "country" in result["output"]["text"].lower()


def test_image_bytes_and_url_run_in_worker(source_server):
    """Notebook kernels run in a Web Worker, where RawImage must decode with OffscreenCanvas."""
    model = TASKS["image-classification"][0]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(MODEL_TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            result = page.evaluate(
                """async ({src, model, url}) => {
                  const code = `
                    self.onmessage = async ({data}) => {
                      try {
                        await import(data.src);
                        const llm = globalThis.lmLite.createLLM({task: 'image-classification', task_model: data.model, backend: 'wasm'});
                        await llm.init();
                        const bytes = new Uint8Array(await (await fetch(data.url)).arrayBuffer());
                        const fromBytes = await llm.runTask(bytes, '{"top_k": 1}');
                        const fromUrl = await llm.runTask(data.url, '{"top_k": 1}');
                        const bad = await llm.runTask(new Uint8Array([1, 2, 3]), '{}').catch(error => String(error));
                        await llm.dispose();
                        self.postMessage({ok: true, offscreen: typeof OffscreenCanvas, fromBytes, fromUrl, bad});
                      } catch (error) {
                        self.postMessage({ok: false, error: String(error?.stack ?? error)});
                      }
                    };`;
                  const worker = new Worker(URL.createObjectURL(new Blob([code], {type: 'text/javascript'})), {type: 'module'});
                  try {
                    return await new Promise((resolve) => {
                      worker.onmessage = ({data}) => resolve(data);
                      worker.onerror = (event) => resolve({ok: false, error: event.message});
                      worker.postMessage({src, model, url});
                    });
                  } finally {
                    worker.terminate();
                  }
                }""",
                {"src": f"{source_server}/src/lmlite/llm.js", "model": model, "url": TIGER_URL},
            )
        finally:
            browser.close()

    assert result["ok"], f"Image input in a worker failed: {result.get('error')}"
    assert "tiger" in result["fromBytes"][0]["label"].lower()
    assert result["fromBytes"] == result["fromUrl"]
    assert "could not be decoded" in result["bad"]


@pytest.mark.parametrize(
    "task", ["image-classification", "zero-shot-image-classification", "image-feature-extraction", "object-detection"]
)
def test_model_info_reports_vision_tasks(source_server, task):
    model = TASKS[task][0]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(MODEL_TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            info = page.evaluate(
                """async ({task, model}) => {
                  await new Promise(resolve => setTimeout(resolve, 0));
                  return await globalThis.lmLite.getModelInfo(model, task, 'wasm');
                }""",
                {"task": task, "model": model},
            )
        finally:
            browser.close()

    assert info["error"] is None, info["error"]
    assert info["task"] == task and info["model_id"] == model
    assert info["dtype"] and info["files"]
    assert info["download_bytes"] and info["download_bytes"] > 1_000_000
