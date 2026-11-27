# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser check of the WebGPU to WASM fallback (it downloads a small model from the Hub). Headless Chromium
without --enable-unsafe-webgpu has no WebGPU adapter."""

import pytest
from kernels import run_in_pyodide

pytestmark = pytest.mark.browser

KERNEL_CODE = r"""
import json
import js
from lmlite import Classifier

MODEL = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
has_adapter = bool(await js.eval("(async () => Boolean(await navigator.gpu?.requestAdapter()))()"))

fallen_back = await Classifier.create(MODEL, backend="webgpu")
backend = fallen_back.backend
label = (await fallen_back.classify("I love this"))[0]["label"]
await fallen_back.dispose()

try:
    pinned = await Classifier.create(MODEL, backend="webgpu", backend_fallback=False)
    pinned_error = None
    await pinned.dispose()
except Exception as error:
    pinned_error = str(error)

print("RESULT" + json.dumps({"has_adapter": has_adapter, "backend": backend, "label": label, "pinned_error": pinned_error}))
"""


def test_webgpu_falls_back_to_wasm_without_an_adapter_and_can_be_pinned(source_server):
    result = run_in_pyodide(source_server, KERNEL_CODE, packages=("tqdm",))  # no WebGPU flags
    assert not result["has_adapter"], "this check needs a browser without WebGPU"
    assert result["backend"] == "wasm" and result["label"] == "POSITIVE"
    assert result["pinned_error"] and "webgpu" in result["pinned_error"].lower()
