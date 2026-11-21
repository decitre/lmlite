# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for bare ONNX models (downloads onnxruntime-web from jsDelivr)."""

import hashlib
import json
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


def test_onnx_model_runs_through_the_pyodide_bridge(source_server):
    """Python in Pyodide: a URL and a file path load, NumPy goes in, NumPy comes back."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(300000)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url="https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js")
            result = page.evaluate(
                """async ({server}) => {
                  const pyodide = await loadPyodide();
                  await pyodide.loadPackage('numpy');
                  pyodide.FS.mkdirTree('/lib/lmlite/bridge');
                  for (const name of ['__init__.py', 'llm.py', 'llm.js', 'onnx_model.py', 'version.py', 'bridge/__init__.py', 'bridge/runtime.py']) {
                    pyodide.FS.writeFile(`/lib/lmlite/${name}`, await (await fetch(`${server}/src/lmlite/${name}`)).text());
                  }
                  const model = new Uint8Array(await (await fetch(`${server}/tests/fixtures/onnx/scale_add.onnx`)).arrayBuffer());
                  pyodide.FS.writeFile('/scale_add.onnx', model);
                  pyodide.globals.set('url', `${server}/tests/fixtures/onnx/scale_add.onnx`);
                  return await pyodide.runPythonAsync(`
import json, sys
sys.path.insert(0, '/lib')
import numpy as np
from lmlite import OnnxModel

out = {}
for label, source in (('url', url), ('path', '/scale_add.onnx')):
    model = await OnnxModel.load(source, licence='CC0')
    y = (await model.run({'x': np.ones((2, 3), np.float32), 'b': np.array([0, 1, 2], np.float32)}))['y']
    out[label] = {'dtype': str(y.dtype), 'shape': list(y.shape), 'values': y.tolist(), 'info': model.model_info()}
    if label == 'url':
        try:
            await model.run({'x': np.ones((2, 3), np.float64), 'b': np.zeros(3, np.float32)})
        except TypeError as exc:
            out['dtype_error'] = str(exc)
    await model.dispose()
json.dumps(out)
                  `);
                }""",
                {"server": source_server},
            )
        finally:
            browser.close()

    result = json.loads(result)
    for label in ("url", "path"):
        assert result[label]["dtype"] == "float32" and result[label]["shape"] == [2, 3]
        assert result[label]["values"] == [[2.0, 3.0, 4.0]] * 2
        assert result[label]["info"]["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        assert result[label]["info"]["licence"] == "CC0"
    assert "Input 'x' must be float32, got float64" in result["dtype_error"]
