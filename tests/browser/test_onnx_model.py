# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for bare ONNX models (downloads onnxruntime-web from jsDelivr)."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from kernels import webgpu_args
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

BACKENDS = os.getenv("LMLITE_TEST_TASK_BACKENDS", "wasm,webgpu").split(",")
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures/onnx"
FIXTURE = FIXTURE_DIR / "scale_add.onnx"


@pytest.mark.parametrize("backend", BACKENDS)
def test_bare_onnx_model_runs_in_browser(source_server, backend):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=webgpu_args(backend))
        try:
            page = browser.new_page()
            page.set_default_timeout(120000)
            page.goto(source_server, wait_until="domcontentloaded")
            if backend == "webgpu" and not page.evaluate("async () => Boolean(await navigator.gpu?.requestAdapter())"):
                pytest.skip("WebGPU adapter unavailable in this browser")
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
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
                """async ({server, digest}) => {
                  const pyodide = await loadPyodide();
                  await pyodide.loadPackage('numpy');
                  pyodide.FS.mkdirTree('/lib/lmlite/bridge');
                  for (const name of ['__init__.py', '_bridge.py', '_session.py', 'llm.py', 'lmlite.js', 'models.py', 'onnx_model.py', 'pipeline.py', 'tokenizer.py', 'version.py', 'bridge/__init__.py', 'bridge/runtime.py']) {
                    pyodide.FS.writeFile(`/lib/lmlite/${name}`, await (await fetch(`${server}/src/lmlite/${name}`)).text());
                  }
                  const model = new Uint8Array(await (await fetch(`${server}/tests/fixtures/onnx/scale_add.onnx`)).arrayBuffer());
                  pyodide.FS.writeFile('/scale_add.onnx', model);
                  pyodide.globals.set('url', `${server}/tests/fixtures/onnx/scale_add.onnx`);
                  pyodide.globals.set('half_url', `${server}/tests/fixtures/onnx/scale_add_fp16.onnx`);
                  pyodide.globals.set('hosted_url', `${server}/tests/fixtures/onnx/hosted/`);
                  pyodide.globals.set('digest', digest);
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

# a hosted ID from a manifest, a SHA-256 check, the Cache API, and float16 through the bridge
hosted = await OnnxModel.load("lmlite/scale-add", hosted_models_url=hosted_url, sha256=digest)
out["hosted"] = (await hosted.run({'x': np.ones((1, 3), np.float32), 'b': np.zeros(3, np.float32)}))['y'].tolist()
out["hosted_cache"] = [hosted.model_info()["cache_status"]]
await hosted.dispose()
again = await OnnxModel.load("lmlite/scale-add", hosted_models_url=hosted_url)
out["hosted_cache"].append(again.model_info()["cache_status"])
await again.dispose()
try:
    await OnnxModel.load(url, sha256="0" * 64)
except Exception as exc:
    out["sha_error"] = str(exc)
half = await OnnxModel.load(half_url)
y16 = (await half.run({'x': np.ones((2, 3), np.float16), 'b': np.array([0, 1, 2], np.float16)}))['y']
out["float16"] = {'dtype': str(y16.dtype), 'values': y16.tolist()}
await half.dispose()
json.dumps(out)
                  `);
                }""",
                {"server": source_server, "digest": hashlib.sha256(FIXTURE.read_bytes()).hexdigest()},
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
    assert result["hosted"] == [[2.0, 2.0, 2.0]]
    assert result["hosted_cache"] == ["downloaded", "cached"]  # the second load makes no request
    assert "SHA-256 mismatch" in result["sha_error"]
    assert result["float16"] == {"dtype": "float16", "values": [[2.0, 3.0, 4.0]] * 2}


RUN_JS = """async ({url, backend, dtype, x, b}) => {
  await new Promise(resolve => setTimeout(resolve, 0));
  try {
    const model = await globalThis.lmLite.loadOnnxModel(url, backend);
    const rows = x.length / 3 / (dtype === 'float16' ? 2 : 4);
    const meta = JSON.stringify([{name: 'x', dtype, dims: [rows, 3]}, {name: 'b', dtype, dims: [3]}]);
    const [outMeta, y] = await model.run(meta, [new Uint8Array(x), new Uint8Array(b)]);
    await model.dispose();
    return {ok: true, meta: JSON.parse(outMeta), y: [...y]};
  } catch (error) {
    return {ok: false, error: String(error?.stack ?? error)};
  }
}"""


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_file,dtype", [("scale_add.onnx", "float32"), ("scale_add_fp16.onnx", "float16")])
def test_browser_matches_native_onnxruntime(source_server, backend, model_file, dtype):
    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3)).astype(dtype)
    b = rng.normal(size=3).astype(dtype)
    expected = ort.InferenceSession(str(FIXTURE_DIR / model_file)).run(None, {"x": x, "b": b})[0]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=webgpu_args(backend))
        try:
            page = browser.new_page()
            page.set_default_timeout(120000)
            page.goto(source_server, wait_until="domcontentloaded")
            if backend == "webgpu" and not page.evaluate("async () => Boolean(await navigator.gpu?.requestAdapter())"):
                pytest.skip("WebGPU adapter unavailable in this browser")
            if (
                backend == "webgpu"
                and dtype == "float16"
                and not page.evaluate("async () => (await navigator.gpu.requestAdapter()).features.has('shader-f16')")
            ):
                pytest.skip("This WebGPU adapter has no shader-f16 support")
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
            result = page.evaluate(
                RUN_JS,
                {
                    "url": f"{source_server}/tests/fixtures/onnx/{model_file}",
                    "backend": backend,
                    "dtype": dtype,
                    "x": list(x.tobytes()),
                    "b": list(b.tobytes()),
                },
            )
        finally:
            browser.close()

    assert result["ok"], result.get("error")
    assert result["meta"] == [{"name": "y", "dtype": dtype, "dims": [5, 3]}]
    actual = np.frombuffer(bytes(result["y"]), dtype=dtype).reshape(5, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-3 if dtype == "float16" else 1e-6)


def test_urls_are_cached_and_checked_against_sha256(source_server):
    url = f"{source_server}/tests/fixtures/onnx/scale_add.onnx"
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(120000)
            requests = []
            page.on("request", lambda request: requests.append(request.url))
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
            result = page.evaluate(
                """async ({url, digest}) => {
                  await new Promise(resolve => setTimeout(resolve, 0));
                  const lmLite = globalThis.lmLite;
                  const status = async (...args) => {
                    const model = await lmLite.loadOnnxModel(url, 'wasm', ...args);
                    const info = JSON.parse(model.describe());
                    await model.dispose();
                    return info.cache_status;
                  };
                  const out = {};
                  out.first = await status(digest);
                  out.second = await status(digest);
                  out.uncached = await lmLite.loadOnnxModel(url, 'wasm', null, false).then(m => JSON.parse(m.describe()).cache_status);
                  out.mismatch = await status('0'.repeat(64)).catch(error => String(error));
                  out.afterMismatch = await status(digest);
                  await lmLite.clearModelCache();
                  out.afterClear = await status(null);
                  return out;
                }""",
                {"url": url, "digest": digest},
            )
        finally:
            browser.close()

    assert result["first"] == "downloaded" and result["second"] == "cached"
    assert result["uncached"] is None
    assert "SHA-256 mismatch" in result["mismatch"]  # a changed file is fetched again, and checked
    assert result["afterMismatch"] == "downloaded"  # the stale entry was dropped
    assert result["afterClear"] == "downloaded"  # clear_cache() removes it
    # Requests: first load, uncached load, the mismatch refetch, the load after it, the load after the clear.
    assert requests.count(url) == 5
