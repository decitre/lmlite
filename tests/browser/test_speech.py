# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for text-to-speech pipelines (downloads a model from the Hub)."""

import os
import struct

import pytest
from kernels import TIMEOUT, run_in_pyodide, run_in_xeus
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

SPEECH_MODEL = os.getenv("LMLITE_TEST_SPEECH_MODEL", "Xenova/mms-tts-eng")
BACKENDS = os.getenv("LMLITE_TEST_TASK_BACKENDS", "wasm,webgpu").split(",")


@pytest.mark.parametrize("task", ["text-to-speech", "text-to-audio"])
def test_raw_samples_cross_the_bridge_as_one_buffer(source_server, task):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
            result = page.evaluate(
                """async ({task, model}) => {
                  await new Promise(resolve => setTimeout(resolve, 0));
                  try {
                    const llm = globalThis.lmLite.createRuntime({task, task_model: model, backend: 'wasm'});
                    await llm.init();
                    const [rate, bytes] = await llm.runTask('Hello from the browser', '{}');
                    const batch = await llm.runTasks(['One.', 'Two.'], 8, {}, null, true);
                    await llm.dispose();
                    return {ok: true, rate, isBytes: bytes instanceof Uint8Array, samples: [...bytes],
                            batch: batch.results.map(([r, b]) => [r, b.byteLength])};
                  } catch (error) {
                    return {ok: false, error: String(error?.stack ?? error)};
                  }
                }""",
                {"task": task, "model": SPEECH_MODEL},
            )
        finally:
            browser.close()

    assert result["ok"], result.get("error")
    raw = bytes(result["samples"])
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    assert result["rate"] == 16000 and len(raw) % 4 == 0 and result["isBytes"]
    assert 0.5 < len(values) / result["rate"] < 6  # a short sentence is a few seconds of audio
    assert 0.05 < max(abs(v) for v in values) <= 1.5  # audible, not silence or noise at full scale
    assert len(result["batch"]) == 2 and all(
        rate == 16000 and size > 0 for rate, size in result["batch"]
    )  # one text per call


@pytest.mark.parametrize("backend", BACKENDS)
def test_model_info_reports_the_speech_task(source_server, backend):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
            info = page.evaluate(
                """async ({model, backend}) => {
                  await new Promise(resolve => setTimeout(resolve, 0));
                  return await globalThis.lmLite.getModelInfo(model, 'text-to-speech', backend);
                }""",
                {"model": SPEECH_MODEL, "backend": backend},
            )
        finally:
            browser.close()

    assert info["error"] is None, info["error"]
    assert info["task"] == "text-to-speech" and info["dtype"] and info["files"]
    assert info["download_bytes"] and info["download_bytes"] > 1_000_000


# Runs in a Python kernel: one call, a batch with map(), and the objects they return.
KERNEL_CODE = r"""
import json
from lmlite import LLM

tts = await LLM.pipeline("text-to-speech", model="MODEL")
speech = await tts("Hello from the browser")
many = await tts.map(["One.", "Two."])
try:
    await tts(["not one text"])
    error = None
except TypeError as exc:
    error = str(exc)
await tts.dispose()
wav = speech.to_wav()
print("RESULT" + json.dumps({
    "rate": speech.sampling_rate,
    "count": len(speech.samples),
    "kind": type(speech.samples).__module__ + "." + type(speech.samples).__name__,
    "peak": max(abs(float(x)) for x in speech.samples),
    "duration": speech.duration,
    "wav": [wav[:4].decode(), wav[8:12].decode(), len(wav) - 44 - 2 * len(speech.samples)],
    "html": speech._repr_html_().startswith('<audio controls src="data:audio/wav;base64,'),
    "map": [type(item).__name__ for item in many],
    "error": error,
}))
"""


def kernel_code():
    return KERNEL_CODE.replace("MODEL", SPEECH_MODEL)


def check_kernel_result(result, numpy):
    assert result["rate"] == 16000 and 0.5 < result["duration"] < 6
    assert result["count"] == round(result["duration"] * 16000)
    assert result["kind"] == ("numpy.ndarray" if numpy else "array.array")
    assert 0.05 < result["peak"] <= 1.5
    assert result["wav"] == ["RIFF", "WAVE", 0]  # 16-bit PCM: two bytes per sample after the 44-byte header
    assert result["html"] is True
    assert result["map"] == ["Speech", "Speech"]
    assert "takes one text" in result["error"]


def test_speech_in_pyodide(source_server):
    # NumPy is not loaded here, so the samples are an array.array.
    check_kernel_result(run_in_pyodide(source_server, kernel_code()), numpy=False)


def test_speech_in_xeus(xeus_site):
    check_kernel_result(run_in_xeus(xeus_site, kernel_code()), numpy=True)


MODEL_PATH = "Xenova/mms-tts-eng/onnx/model_quantized.onnx"


def test_speech_loads_from_a_hosted_snapshot_without_the_hub(source_server, hosted_speech_models):
    requests = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(TIMEOUT)
            page.on("request", lambda request: requests.append(request.url))
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
            result = page.evaluate(
                """async ({base, model}) => {
                  await new Promise(resolve => setTimeout(resolve, 0));
                  try {
                    const hosted = await globalThis.lmLite.getHostedModelBase(model, base);
                    const llm = globalThis.lmLite.createRuntime({
                      task: 'text-to-speech', task_model: model, backend: 'wasm',
                      hosted_model_base: hosted, hosted_models: [model],
                    });
                    await llm.init();
                    const [rate, bytes] = await llm.runTask('Hello from a hosted model', '{}');
                    await llm.dispose();
                    return {ok: true, hosted, rate, seconds: bytes.byteLength / 4 / rate};
                  } catch (error) {
                    return {ok: false, error: String(error?.stack ?? error)};
                  }
                }""",
                {"base": hosted_speech_models, "model": "Xenova/mms-tts-eng"},
            )
        finally:
            browser.close()

    assert result["ok"], result.get("error")
    assert result["hosted"] == hosted_speech_models and result["rate"] == 16000 and 0.3 < result["seconds"] < 6
    assert hosted_speech_models + MODEL_PATH in requests  # the weights came from the hosted tree
    assert not [url for url in requests if "huggingface.co" in url]


# Runs in a Python kernel: the speech pipeline from a hosted tree, found through its manifest.
HOSTED_KERNEL_CODE = r"""
import json
from lmlite import LLM

tts = await LLM.pipeline("text-to-speech", model="Xenova/mms-tts-eng", hosted_models_url="BASE")
speech = await tts("Hello from a hosted model")
await tts.dispose()
print("RESULT" + json.dumps({"rate": speech.sampling_rate, "duration": speech.duration}))
"""


def check_hosted_kernel_result(result, requests, base):
    assert result["rate"] == 16000 and 0.3 < result["duration"] < 6
    assert base + MODEL_PATH in requests
    assert not [url for url in requests if "huggingface.co" in url]


def test_hosted_speech_in_pyodide(source_server, hosted_speech_models):
    requests = []
    code = HOSTED_KERNEL_CODE.replace("BASE", hosted_speech_models)
    result = run_in_pyodide(source_server, code, requests=requests)
    check_hosted_kernel_result(result, requests, hosted_speech_models)


def test_hosted_speech_in_xeus(xeus_site, hosted_speech_models):
    requests = []
    code = HOSTED_KERNEL_CODE.replace("BASE", hosted_speech_models)
    result = run_in_xeus(xeus_site, code, requests=requests)
    check_hosted_kernel_result(result, requests, hosted_speech_models)
