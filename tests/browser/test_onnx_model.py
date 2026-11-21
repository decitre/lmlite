# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for bare ONNX models (downloads onnxruntime-web from jsDelivr)."""

import hashlib
import os
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

BACKENDS = os.getenv("LMLITE_TEST_TASK_BACKENDS", "wasm,webgpu").split(",")
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/onnx/scale_add.onnx"


@pytest.mark.parametrize("backend", BACKENDS)
def test_bare_onnx_model_runs_in_browser(source_server, backend):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--enable-unsafe-webgpu"] if backend == "webgpu" else [])
        try:
            page = browser.new_page()
            page.set_default_timeout(120000)
            page.goto(source_server, wait_until="domcontentloaded")
            if backend == "webgpu" and not page.evaluate("async () => Boolean(await navigator.gpu?.requestAdapter())"):
                pytest.skip("WebGPU adapter unavailable in this browser")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            result = page.evaluate(
                """async ({url, backend}) => {
                  await new Promise(resolve => setTimeout(resolve, 0));
                  try {
                    const model = await globalThis.lmLite.loadOnnxModel(url, backend);
                    const f32 = (values) => new Uint8Array(new Float32Array(values).buffer);
                    const run = async (rows) => {
                      const meta = JSON.stringify([
                        {name: 'x', dtype: 'float32', dims: [rows, 3]},
                        {name: 'b', dtype: 'float32', dims: [3]},
                      ]);
                      const [outMeta, y] = await model.run(meta, [f32(new Array(rows * 3).fill(1)), f32([0, 1, 2])]);
                      return {meta: JSON.parse(outMeta), y: [...new Float32Array(y.buffer, y.byteOffset, y.byteLength / 4)]};
                    };
                    const output = {info: JSON.parse(model.describe()), one: await run(1), two: await run(2)};
                    await model.dispose();
                    return {ok: true, ...output};
                  } catch (error) {
                    return {ok: false, error: String(error?.stack ?? error)};
                  }
                }""",
                {"url": f"{source_server}/tests/fixtures/onnx/scale_add.onnx", "backend": backend},
            )
        finally:
            browser.close()

    assert result["ok"], result.get("error")
    info = result["info"]
    assert [i["name"] for i in info["inputs"]] == ["x", "b"]
    assert info["inputs"][0]["shape"] == ["batch", 3] and info["inputs"][0]["dtype"] == "float32"
    assert info["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert info["backend"] == info["execution_provider"] == backend
    assert result["one"]["y"] == [2.0, 3.0, 4.0]
    assert result["two"]["meta"] == [{"name": "y", "dtype": "float32", "dims": [2, 3]}]  # dynamic batch dimension
    assert result["two"]["y"] == [2.0, 3.0, 4.0] * 2
