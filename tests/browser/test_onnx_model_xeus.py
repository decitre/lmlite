# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""OnnxModel in a real xeus-python kernel (the pyjs bridge), built into a JupyterLite site from the current source."""

import hashlib
import json
import os
import urllib.parse
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/onnx/scale_add.onnx"
TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "300000"))

# Runs in the kernel; prints one RESULT line with everything the test asserts on.
KERNEL_CODE = r"""
import json
import js
import numpy as np
from lmlite import OnnxModel

x = np.ones((2, 3), np.float32)
b = np.array([0, 1, 2], np.float32)
res = {}
for label, source in (("path", "onnx/scale_add.onnx"), ("url", f"{js.location.origin}/files/onnx/scale_add.onnx")):
    model = await OnnxModel.load(source, licence="CC0")
    y = (await model.run({"x": x, "b": b}))["y"]
    res[label] = {"dtype": str(y.dtype), "shape": list(y.shape), "values": y.tolist(), "info": model.model_info()}
    if label == "path":
        try:
            await model.run({"x": x.astype(np.float64), "b": b})
        except TypeError as exc:
            res["dtype_error"] = str(exc)
    await model.dispose()
print("RESULT" + json.dumps(res))
"""


def test_onnx_model_runs_through_the_xeus_bridge(xeus_site):
    url = f"{xeus_site}/repl/index.html?kernel=xpython&execute=1&code={urllib.parse.quote(KERNEL_CODE)}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url)
            # The echoed input never contains 'RESULT{', so this waits for kernel output.
            page.wait_for_function(
                "document.body.innerText.includes('RESULT{') || document.body.innerText.includes('Traceback')",
                timeout=TIMEOUT,
            )
            text = page.inner_text("body")
        finally:
            browser.close()

    lines = [line for line in text.splitlines() if line.startswith("RESULT{")]
    assert lines, f"Kernel failed:\n{text[-3000:]}"
    result = json.loads(lines[-1].removeprefix("RESULT"))
    for label in ("path", "url"):
        assert result[label]["dtype"] == "float32" and result[label]["shape"] == [2, 3]
        assert result[label]["values"] == [[2.0, 3.0, 4.0]] * 2
        assert result[label]["info"]["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        assert result[label]["info"]["licence"] == "CC0"
    assert "Input 'x' must be float32, got float64" in result["dtype_error"]
