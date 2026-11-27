# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Run a bare ``.onnx`` file (no Hugging Face repository layout) with NumPy in and NumPy out.

In a browser kernel the session is an onnxruntime-web ``InferenceSession`` (the build Transformers.js uses);
elsewhere it is an onnxruntime one from ``pip install 'lmlite[native]'``. Pre- and post-processing stay in Python.
"""

import asyncio
import hashlib
import json
import os
import re

from ._bridge import _bytes_of, _js_function, _native, _to_js, _to_python_deep

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


def _verify(data, sha256):
    if sha256 and (actual := hashlib.sha256(data).hexdigest()) != sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {sha256}, got {actual}")


def _is_hosted_name(source):
    """A ``namespace/model`` ID, as listed in a hosted models manifest; model files end with ``.onnx``."""
    return isinstance(source, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", source) and not source.lower().endswith(".onnx")


def _read_source(source):
    """The model's bytes for a path or bytes, or None for a URL or a hosted name (the browser fetches those)."""
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


class _MapRun:
    """The state of a running ``OnnxModel.map()``."""

    def __init__(self):
        self.cancelled = False
        self.finished = asyncio.Event()


class OnnxModel:
    """A single ``.onnx`` file with no Hugging Face repository around it, run with NumPy in and NumPy out.

    Load it with ``await OnnxModel.load(...)``. You do the pre- and post-processing in NumPy.

    Attributes:
        inputs: The model's inputs, a list of dicts with ``name``, ``dtype`` and ``shape``; symbolic dimensions stay
            strings such as ``"batch"``.
        outputs: The model's outputs, in the same form.

    Example:
        .. code-block:: python

            model = await OnnxModel.load("models/detector.onnx", backend="wasm", licence="Apache-2.0")
            outputs = await model.run({"input": blob.astype(np.float32)})
            await model.dispose()
    """

    def __init__(self, handle, info, licence=None):
        self._handle = handle
        self._info = info
        self._licence = licence
        self._map_run = None
        self.inputs = info["inputs"]
        self.outputs = info["outputs"]

    def __repr__(self):
        return f"OnnxModel(inputs={[i['name'] for i in self.inputs]}, outputs={[o['name'] for o in self.outputs]})"

    @classmethod
    async def load(
        cls,
        source,
        backend="wasm",
        licence=None,
        sha256=None,
        cache=True,
        hosted_models_url=None,
        backend_fallback=True,
    ):
        """Load a model.

        Args:
            source: A file path, a URL, the model's bytes, or a hosted model ID such as ``"lmlite/detector"`` (listed in
                the site's ``manifest.json``, or the one at ``hosted_models_url``; it loads ``<id>/model.onnx``).
            backend: ``"wasm"`` (the default) or ``"webgpu"`` in a browser kernel; ``"native"``, or any backend outside
                a browser, runs onnxruntime (``pip install 'lmlite[native]'``). float16 models on WebGPU need an adapter with
                ``shader-f16``.
            licence: A free-form string that ``model_info()`` returns.
            sha256: A hash the model's bytes must match.
            cache: In a browser kernel, keep URLs in the Cache API so a later load of the same URL makes no request.
            hosted_models_url: An explicit URL of a models directory, for hosted IDs.
            backend_fallback: ``True`` (the default) loads on ``"wasm"`` when ``backend="webgpu"`` but the browser has no
                WebGPU adapter, or the session cannot be created for a reason specific to WebGPU (an operator or
                ``float16`` the adapter lacks). The model then runs on WASM, which can be much slower or run out of
                memory; ``model_info()`` reports the ``backend`` that ran and the ``backend_fallback_reason``. ``False``
                fails instead.

        Returns:
            The loaded ``OnnxModel``.

        Raises:
            ValueError: The bytes do not match ``sha256``.
            FileNotFoundError: A path does not exist, or a hosted ID is not listed in the manifest.
        """
        native = _native(backend == "native")
        if _is_hosted_name(source) and not os.path.isfile(source):
            if native:
                raise FileNotFoundError(f"ONNX model not found: {source!r}. Hosted IDs resolve in a browser kernel.")
            base = await _js_function("getHostedModelBase")(source, hosted_models_url)
            if base is None:
                raise FileNotFoundError(f"{source!r} is not a hosted model: list it in the manifest.json of the site.")
            source = f"{base}{source}/model.onnx"
        data = _read_source(source)
        if data is not None:
            _verify(data, sha256)
        if native:
            return await native.load_onnx_model(cls, data if data is not None else source, licence, sha256)
        handle = await _js_function("loadOnnxModel")(
            _to_js(data) if data is not None else source, backend, sha256, cache, backend_fallback
        )
        return cls(handle, json.loads(str(handle.describe())), licence)

    def model_info(self):
        """What is known about the loaded model.

        Returns:
            A dict with the model's SHA-256 and size, the backend and execution provider, the cache status, the
            inputs and outputs, and the ``licence`` passed to ``load()``.
        """
        return {**self._info, "licence": self._licence}

    async def run(self, feeds):
        """Run the model once.

        Each call crosses the browser bridge once, with raw buffers: call it once per image rather than with a large
        batch, since some detectors return about 1 MB of floats per image.

        Args:
            feeds: A dict from input names to NumPy arrays. The dtypes must be the ones the model declares (no silent
                casts: use ``astype()``), and the shape must fit; dynamic dimensions accept any size.

        Returns:
            A dict from output names to NumPy arrays.

        Raises:
            ValueError: An input is missing or unknown, or has the wrong rank or size on a fixed axis.
            TypeError: An input has the wrong dtype.
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

    async def map(self, feeds, progress_callback=None, on_error="raise"):
        """Run the model over many inputs, one ``run()`` each.

        The loop is in Python, one bridge call per input.

        Args:
            feeds: An iterable, or async iterable, of feed dicts. A generator that preprocesses one image at a time keeps
                the inputs from being in memory together.
            progress_callback: Called as ``progress_callback(done, total)`` after each input; ``total`` is ``None`` when
                ``feeds`` has no length.
            on_error: ``"raise"`` raises the first error; ``"skip"`` leaves a failed input out of the results; ``"return"``
                puts an exception object where its outputs would be. It covers errors from ``run()``, such as a wrong
                dtype or shape. An error raised by ``feeds`` itself ends the run.

        Returns:
            The outputs in input order. If ``cancel()`` stopped the run, the outputs so far; the model stays usable.

        Raises:
            ValueError: ``on_error`` is not one of the three values.
            RuntimeError: Another ``map()`` is already running on this model.
        """
        if on_error not in ("raise", "skip", "return"):
            raise ValueError(f"on_error must be 'raise', 'skip' or 'return', got {on_error!r}")
        if self._map_run is not None:
            raise RuntimeError("Another map() is running on this model; await it or call cancel() first.")
        run = self._map_run = _MapRun()
        total = len(feeds) if hasattr(feeds, "__len__") else None
        asynchronous = hasattr(feeds, "__aiter__")
        iterator = feeds.__aiter__() if asynchronous else iter(feeds)
        results = []
        done = 0
        try:
            while not run.cancelled:
                try:
                    feed = await iterator.__anext__() if asynchronous else next(iterator)
                except (StopAsyncIteration, StopIteration):
                    break
                try:
                    results.append(await self.run(feed))
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    if on_error == "return":
                        results.append(exc)
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total)
            return results
        finally:
            self._map_run = None
            run.finished.set()

    async def cancel(self):
        """Stop a running ``map()`` after its current input.

        Returns:
            ``False`` if nothing was running.
        """
        run = self._map_run
        if run is None:
            return False
        run.cancelled = True
        await run.finished.wait()
        return True

    async def dispose(self):
        """Release the session."""
        await self._handle.dispose()
