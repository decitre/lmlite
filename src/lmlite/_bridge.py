# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import contextlib

from .bridge.runtime import ensure_runtime


def _loading_progress():
    import sys

    from tqdm import tqdm as text_tqdm

    if sys.platform in {"emscripten", "wasi"}:
        # Browser kernels cannot start tqdm's background monitor thread.
        text_tqdm.monitor_interval = 0
        from tqdm.notebook import IProgress, tqdm as notebook_tqdm

        if IProgress is not None:
            # Xeus is not always recognized by tqdm.auto's IPython kernel detection.
            notebook_tqdm.monitor_interval = 0
            progress_class = notebook_tqdm
        else:
            import warnings

            warnings.warn(
                "Notebook progress widgets are unavailable. Install ipywidgets in the kernel "
                "and enable widget support in JupyterLite; using text progress instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            progress_class = text_tqdm
    else:
        from tqdm.auto import tqdm as progress_class

    return progress_class(
        desc="Preparing models", total=None, unit="B", unit_scale=True, mininterval=0.1, bar_format="{desc}"
    )


def _to_python_deep(value):
    """Convert JS proxy values into plain Python objects."""
    if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return value

    # Pyodide conversion first (Pyodide/JupyterLite).
    try:
        from pyodide.ffi import to_py as pyodide_to_py

        converted = pyodide_to_py(value)
        if converted is not value:
            return _to_python_deep(converted)
    except (ImportError, TypeError):
        pass

    # xeus-python fallback.
    try:
        from pyjs import to_py as pyjs_to_py

        converted = pyjs_to_py(value)
        if converted is not value:
            return _to_python_deep(converted)
    except (ImportError, TypeError, AttributeError):
        pass

    # JsProxy often exposes a .to_py() method.
    to_py_method = getattr(value, "to_py", None)
    if callable(to_py_method):
        try:
            converted = to_py_method()
            if converted is not value:
                return _to_python_deep(converted)
        except TypeError:
            pass

    if isinstance(value, dict):
        return {key: _to_python_deep(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return tuple(_to_python_deep(item) for item in value)

    if isinstance(value, list):
        return [_to_python_deep(item) for item in value]

    # Final fallback for iterable proxy arrays.
    if hasattr(value, "__iter__"):
        try:
            return [_to_python_deep(item) for item in value]
        except TypeError:
            pass

    return value


def _bytes_of(value):
    """The raw bytes of a JavaScript Uint8Array proxy (Pyodide or xeus-python)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if callable(getattr(value, "to_bytes", None)):  # Pyodide JsBuffer
        return value.to_bytes()
    converted = value.to_py() if callable(getattr(value, "to_py", None)) else value
    return bytes(converted)


def _to_js(value):
    try:
        from pyodide.ffi import to_js
    except ImportError:
        try:
            from pyjs import to_js
        except ImportError:
            raise RuntimeError("No Python-to-JavaScript bridge found: need pyodide.ffi or pyjs") from None
    return to_js(value)


def _runtime():
    """The JS runtime namespace, ``globalThis.lmLite``, or None when it is not loaded."""
    try:
        import js
    except ImportError as exc:
        raise ModuleNotFoundError(
            "This lmlite call needs a browser kernel (JupyterLite pyodide or xeus-python). Outside the browser, "
            "only LLM.create() embed()/classify()/generate() and LLM.tokenizer() run, with pip install 'lmlite[native]'.",
            name="js",
        ) from exc

    return getattr(js, "lmLite", None)


def _native(requested=False):
    """The native backend module, used when requested or when there is no browser JS runtime; else None."""
    if not requested:
        try:
            import js  # noqa: F401

            return None
        except ImportError:
            pass
    try:
        from . import native
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"No browser JS runtime and the native backend is unavailable ({exc}). "
            "Install it with: pip install 'lmlite[native]'"
        ) from exc
    return native


@contextlib.contextmanager
def _js_callable(function):
    """``function`` as something JavaScript can call, released on exit: a Pyodide proxy or a xeus-python callable."""
    proxy = handle = None
    try:
        from pyodide.ffi import create_proxy
    except ImportError:
        try:
            from pyjs import create_callable
        except ImportError:
            callback = function
        else:
            # Xeus exposes Python functions as objects, not JavaScript callables.
            callback, handle = create_callable(function)
    else:
        proxy = callback = create_proxy(function)
    try:
        yield callback
    finally:
        if proxy is not None:
            proxy.destroy()
        if handle is not None:
            handle.delete()


def _js_function(name):
    ensure_runtime()

    fn = getattr(_runtime(), name, None)
    if fn is None:
        raise RuntimeError(f"JS runtime missing {name}; restart the kernel to reload lmLite")
    return fn
