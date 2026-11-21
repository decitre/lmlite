# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Write tests/fixtures/onnx/scale_add.onnx: y = x * 2 + b, with x of shape (batch, 3) and b of shape (3,).

Run with: uv run --with onnx python scripts/generate_onnx_fixture.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/onnx/scale_add.onnx"

graph = helper.make_graph(
    [helper.make_node("Mul", ["x", "two"], ["scaled"]), helper.make_node("Add", ["scaled", "b"], ["y"])],
    "scale_add",
    [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3]),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [3]),
    ],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
    [numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
OUT.write_bytes(model.SerializeToString())
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
