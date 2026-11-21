# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Write tests/fixtures/onnx/scale_add.onnx and scale_add_fp16.onnx: y = x * 2 + b.

x has shape (batch, 3) and b shape (3,); the second model takes and returns float16.
A hosted copy of the float32 model goes in hosted/lmlite/scale-add/model.onnx, listed in hosted/manifest.json.
Run with: uv run --with onnx python scripts/generate_onnx_fixture.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/onnx"


def scale_add(name, onnx_type, numpy_type):
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "two"], ["scaled"]), helper.make_node("Add", ["scaled", "b"], ["y"])],
        "scale_add",
        [
            helper.make_tensor_value_info("x", onnx_type, ["batch", 3]),
            helper.make_tensor_value_info("b", onnx_type, [3]),
        ],
        [helper.make_tensor_value_info("y", onnx_type, ["batch", 3])],
        [numpy_helper.from_array(np.array(2.0, dtype=numpy_type), "two")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    (OUT / name).write_bytes(model.SerializeToString())
    print(f"Wrote {OUT / name} ({(OUT / name).stat().st_size} bytes)")


scale_add("scale_add.onnx", TensorProto.FLOAT, np.float32)
scale_add("scale_add_fp16.onnx", TensorProto.FLOAT16, np.float16)

hosted = OUT / "hosted"
(hosted / "lmlite/scale-add").mkdir(parents=True, exist_ok=True)
shutil.copy(OUT / "scale_add.onnx", hosted / "lmlite/scale-add/model.onnx")
(hosted / "manifest.json").write_text(json.dumps({"models": ["lmlite/scale-add"]}) + "\n")
