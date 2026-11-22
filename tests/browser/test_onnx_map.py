# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""OnnxModel.map() through the real Python/JavaScript bridge, in Pyodide and xeus-python kernels."""

import pytest
from kernels import run_in_pyodide, run_in_xeus, webgpu_args

pytestmark = pytest.mark.browser

# Runs in the kernel: a lazy generator with one bad input, compared with one run() per input.
KERNEL_CODE = r"""
import asyncio, json
import numpy as np
from lmlite import OnnxModel

model = await OnnxModel.load("SOURCE", backend="BACKEND")
b = np.array([0, 1, 2], np.float32)

def feed(n):
    return {"x": np.full((1 + n % 3, 3), n, np.float32), "b": b}

def generate():
    for n in range(10):
        yield feed(n) if n != 4 else {"x": np.ones((1, 3), np.float64), "b": b}  # a wrong dtype

progress = []
mapped = await model.map(generate(), on_error="return", progress_callback=lambda done, total: progress.append([done, total]))
good = [n for n in range(10) if n != 4]
single = [(await model.run(feed(n)))["y"] for n in good]
same = all(np.array_equal(m["y"], s) for m, s in zip([m for m in mapped if isinstance(m, dict)], single))

stopped = []
def stop_after_three(done, total):
    if done == 3:
        stopped.append(asyncio.ensure_future(model.cancel()))
cancelled = await model.map([feed(n) for n in range(30)], progress_callback=stop_after_three)
cancel_returned = await stopped[0]
after = len(await model.map([feed(0), feed(1)]))
await model.dispose()
print("RESULT" + json.dumps({
    "types": [type(item).__name__ for item in mapped],
    "same_outputs": same,
    "progress": [progress[0], progress[-1], len(progress)],
    "cancelled": [len(cancelled), cancel_returned],
    "after_cancel": after,
}))
"""


def check(result):
    assert result["types"] == ["dict"] * 4 + ["TypeError"] + ["dict"] * 5  # the bad input is returned in its place
    assert result["same_outputs"]  # the rest equal one run() per input
    assert result["progress"] == [[1, None], [10, None], 10]  # a generator has no total, and every input counts
    assert 3 <= result["cancelled"][0] <= 4 and result["cancelled"][1] is True
    assert result["after_cancel"] == 2  # the model stays usable


@pytest.mark.parametrize("backend", ["wasm", "webgpu"])
def test_map_through_the_pyodide_bridge(source_server, backend):
    code = KERNEL_CODE.replace("SOURCE", f"{source_server}/tests/fixtures/onnx/scale_add.onnx").replace(
        "BACKEND", backend
    )
    flags = webgpu_args(backend)
    check(run_in_pyodide(source_server, code, packages=("tqdm", "numpy"), args=flags))


def test_map_through_the_xeus_bridge(xeus_site):
    check(run_in_xeus(xeus_site, KERNEL_CODE.replace("SOURCE", "onnx/scale_add.onnx").replace("BACKEND", "wasm")))
