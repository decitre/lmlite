# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""OnnxModel in a real xeus-python kernel (the pyjs bridge), built into a JupyterLite site from the current source."""

import hashlib
from pathlib import Path

import pytest
from kernels import run_in_xeus, webgpu_args, webgpu_has_f16

pytestmark = pytest.mark.browser

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures/onnx"
DIGEST = hashlib.sha256((FIXTURE_DIR / "scale_add.onnx").read_bytes()).hexdigest()

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
    model = await OnnxModel.load(source, backend="BACKEND", licence="CC0")
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

# The options added after the first version: a hosted ID, the SHA-256 check, the Cache API, and float16.
OPTIONS_CODE = r"""
import json
import js
import numpy as np
from lmlite import OnnxModel

x = np.ones((2, 3), np.float32)
b = np.array([0, 1, 2], np.float32)
res = {}
hosted = await OnnxModel.load("lmlite/scale-add", backend="BACKEND", hosted_models_url="HOSTED", sha256="DIGEST")
res["hosted"] = (await hosted.run({"x": x, "b": b}))["y"].tolist()
res["status"] = [hosted.model_info()["cache_status"]]
await hosted.dispose()
again = await OnnxModel.load("lmlite/scale-add", backend="BACKEND", hosted_models_url="HOSTED")
res["status"].append(again.model_info()["cache_status"])
await again.dispose()
try:
    await OnnxModel.load(f"{js.location.origin}/files/onnx/scale_add.onnx", backend="BACKEND", sha256="0" * 64)
except Exception as exc:
    res["sha_error"] = str(exc)
if FLOAT16:
    half = await OnnxModel.load("onnx/scale_add_fp16.onnx", backend="BACKEND")
    y16 = (await half.run({"x": x.astype(np.float16), "b": b.astype(np.float16)}))["y"]
    res["float16"] = {"dtype": str(y16.dtype), "values": y16.tolist()}
    await half.dispose()
print("RESULT" + json.dumps(res))
"""


@pytest.mark.parametrize("backend", ["wasm", "webgpu"])
def test_onnx_model_runs_through_the_xeus_bridge(xeus_site, backend):
    result = run_in_xeus(xeus_site, KERNEL_CODE.replace("BACKEND", backend), args=webgpu_args(backend))

    for label in ("path", "url"):
        assert result[label]["dtype"] == "float32" and result[label]["shape"] == [2, 3]
        assert result[label]["values"] == [[2.0, 3.0, 4.0]] * 2
        assert result[label]["info"]["sha256"] == DIGEST
        assert result[label]["info"]["licence"] == "CC0"
        assert result[label]["info"]["backend"] == backend
    assert "Input 'x' must be float32, got float64" in result["dtype_error"]


@pytest.mark.parametrize("backend", ["wasm", "webgpu"])
def test_hosted_id_sha256_cache_and_float16_in_xeus(xeus_site, source_server, hosted_onnx_models, backend):
    args = webgpu_args(backend)
    float16 = backend == "wasm" or webgpu_has_f16(source_server, args)  # WebGPU needs shader-f16 for float16
    code = (
        OPTIONS_CODE.replace("BACKEND", backend)
        .replace("HOSTED", hosted_onnx_models)
        .replace("DIGEST", DIGEST)
        .replace("FLOAT16", str(float16))
    )

    result = run_in_xeus(xeus_site, code, args=args)

    assert result["hosted"] == [[2.0, 3.0, 4.0]] * 2
    assert result["status"] == ["downloaded", "cached"]  # the second load makes no request
    assert "SHA-256 mismatch" in result["sha_error"]
    if float16:
        assert result["float16"] == {"dtype": "float16", "values": [[2.0, 3.0, 4.0]] * 2}
    else:
        assert "float16" not in result  # no shader-f16 on this adapter: not run
