# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Run a bare ``.onnx`` file (no Hugging Face repository layout) with NumPy in and NumPy out.

In a browser kernel the session is an onnxruntime-web ``InferenceSession`` (the build Transformers.js uses);
elsewhere it is an onnxruntime one from ``pip install 'lmlite[native]'``. Pre- and post-processing stay in Python.
"""

import json
import os

from .llm import _js_function, _native, _to_js, _to_python_deep

# onnxruntime tensor type -> NumPy dtype name, as onnxruntime-web reports it.
_NATIVE_DTYPES = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
    "tensor(int16)": "int16",
    "tensor(uint16)": "uint16",
    "tensor(int32)": "int32",
    "tensor(uint32)": "uint32",
    "tensor(int64)": "int64",
    "tensor(uint64)": "uint64",
    "tensor(bool)": "bool",
}


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ModuleNotFoundError("OnnxModel needs NumPy: pip install numpy", name="numpy") from exc
    return np


def _bytes_of(value):
    """The raw bytes of a JavaScript Uint8Array proxy (Pyodide or xeus-python)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if callable(getattr(value, "to_bytes", None)):  # Pyodide JsBuffer
        return value.to_bytes()
    converted = value.to_py() if callable(getattr(value, "to_py", None)) else value
    return bytes(converted)


def _read_source(source):
    """The model's bytes for a path, or None for a URL (the browser fetches those itself)."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if os.path.isfile(source):
        with open(source, "rb") as file:
            return file.read()
    if not source.startswith(("http://", "https://", "blob:", "data:")) and not _in_browser():
        raise FileNotFoundError(f"ONNX model not found: {source!r}. Pass a file path, a URL or the model's bytes.")
    return None


def _in_browser():
    try:
        import js  # noqa: F401
    except ImportError:
        return False
    return True


def _check_feeds(feeds, inputs):
    """Validate ``feeds`` against the model's inputs and return ``[(name, array)]`` in the model's order."""
    np = _numpy()
    expected = {spec["name"]: spec for spec in inputs}
    missing = [name for name in expected if name not in feeds]
    if missing:
        raise ValueError(f"Missing input {', '.join(map(repr, missing))}; the model takes {list(expected)}.")
    unknown = [name for name in feeds if name not in expected]
    if unknown:
        raise ValueError(f"Unknown input {', '.join(map(repr, unknown))}; the model takes {list(expected)}.")
    checked = []
    for name, spec in expected.items():
        array = np.asarray(feeds[name])
        if spec["dtype"] and str(array.dtype) != spec["dtype"]:
            raise TypeError(f"Input {name!r} must be {spec['dtype']}, got {array.dtype}. Convert it with astype().")
        shape = spec["shape"]
        if shape is not None:
            if array.ndim != len(shape):
                raise ValueError(f"Input {name!r} must have rank {len(shape)} {shape}, got shape {array.shape}.")
            for axis, (size, dim) in enumerate(zip(array.shape, shape)):
                if isinstance(dim, int) and dim >= 0 and size != dim:
                    raise ValueError(f"Input {name!r} must have size {dim} on axis {axis}, got shape {array.shape}.")
        checked.append((name, np.ascontiguousarray(array)))
    return checked


class OnnxModel:
    """A loaded ``.onnx`` model; create it with ``await OnnxModel.load(...)``."""

    def __init__(self, handle, info, licence=None):
        self._handle = handle
        self._info = info
        self._licence = licence
        self.inputs = info["inputs"]
        self.outputs = info["outputs"]

    def __repr__(self):
        return f"OnnxModel(inputs={[i['name'] for i in self.inputs]}, outputs={[o['name'] for o in self.outputs]})"

    @classmethod
    async def load(cls, source, backend="wasm", licence=None):
        """Load ``source``: a file path, a URL or the model's bytes.

        ``backend`` is ``"wasm"`` or ``"webgpu"`` in a browser kernel. Outside one, or with ``"native"``, the
        model runs with onnxruntime. ``licence`` is a free-form string that ``model_info()`` returns.
        """
        native = _native(backend == "native")
        data = _read_source(source)
        if native:
            return await native.load_onnx_model(cls, data if data is not None else source, licence)
        handle = await _js_function("loadOnnxModel")(_to_js(data) if data is not None else source, backend)
        return cls(handle, json.loads(str(handle.describe())), licence)

    def model_info(self):
        """The model's hash, size, backend and execution provider, its inputs and outputs, and the licence."""
        return {**self._info, "licence": self._licence}

    async def run(self, feeds):
        """Run the model. ``feeds`` maps input names to NumPy arrays; returns a dict of NumPy arrays.

        Each call crosses the browser bridge once, with raw buffers. One call per image is the intended use:
        some detectors return about 1 MB of floats per image.
        """
        np = _numpy()
        checked = _check_feeds(feeds, self.inputs)
        if self._info["backend"] == "native":
            return await self._handle.run({name: array for name, array in checked})
        meta = json.dumps([{"name": name, "dtype": str(a.dtype), "dims": list(a.shape)} for name, a in checked])
        result = _to_python_deep(await self._handle.run(meta, _to_js([a.tobytes() for _, a in checked])))
        described = json.loads(str(result[0]))
        return {
            spec["name"]: np.frombuffer(_bytes_of(buffer), dtype=spec["dtype"]).reshape(spec["dims"])
            for spec, buffer in zip(described, result[1:])
        }

    async def dispose(self):
        """Release the session."""
        await self._handle.dispose()
