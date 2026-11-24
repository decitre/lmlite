# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import contextlib
from dataclasses import dataclass

from .bridge.runtime import ensure_runtime

DEFAULT_MODEL_NAMESPACE = "Xenova"

# Tasks that return images (PNG bytes, or NumPy arrays with as_array=True); their inputs are images too.
IMAGE_OUTPUT_TASKS = ("background-removal", "depth-estimation", "image-segmentation", "image-to-image")

# Tasks whose inputs are images; keep in sync with lmlite.js.
IMAGE_TASKS = (
    "image-classification",
    "zero-shot-image-classification",
    "image-feature-extraction",
    "object-detection",
    *IMAGE_OUTPUT_TASKS,
)

# Text-to-speech under both names Transformers.js accepts; keep in sync with lmlite.js.
SPEECH_TASKS = ("text-to-speech", "text-to-audio")
# Tasks whose input is audio: a WAV URL or file, WAV bytes, or mono samples at 16 kHz.
AUDIO_INPUT_TASKS = ("automatic-speech-recognition", "audio-classification")

# Transformers.js tasks available through LLM.pipeline(); keep in sync with lmlite.js.
SUPPORTED_TASKS = (
    "zero-shot-classification",
    *AUDIO_INPUT_TASKS,
    "translation",
    *IMAGE_TASKS,
    *SPEECH_TASKS,
)

# Sampling rate expected by Whisper and most Transformers.js speech recognition models.
AUDIO_SAMPLING_RATE = 16000


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


def resolve_model_id(model_name: str) -> str:
    model_id, revision = split_model_name_and_revision(model_name)
    if "/" not in model_id:
        model_id = f"{DEFAULT_MODEL_NAMESPACE}/{model_id}"
    if revision:
        return f"{model_id}@{revision}"
    return model_id


def split_model_name_and_revision(model_name: str) -> tuple[str, str | None]:
    if "@" in model_name:
        model_id, revision = model_name.split("@", 1)
        return model_id, revision
    return model_name, None


def _filename_slug(value: str) -> str:
    return value.replace("/", "-")


def _legacy_filename_slug(value: str) -> str:
    return value.replace("/", "--")


def _is_full_git_sha(value: str) -> bool:
    if len(value) != 40:
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in value)


async def _get_latest_revision(model_id: str) -> str | None:
    """Fetch the latest commit SHA from Hugging Face for a model.

    Tries JS fetch first (works in Pyodide/JupyterLite), falls back to urllib.
    """
    # Try JS fetch first (for Pyodide/JupyterLite environments)
    try:
        get_revision_js = getattr(_runtime(), "getModelLatestRevision", None)
        if get_revision_js is not None:
            try:
                result = await get_revision_js(model_id)
                if result:
                    return result
            except Exception:
                pass  # Fall through to urllib
    except (ImportError, AttributeError):
        pass

    # Fall back to urllib (for server-side Python)
    try:
        import urllib.request
        import json

        url = f"https://huggingface.co/api/models/{model_id}"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            return data.get("sha")
    except Exception as e:
        # For debugging: log the error to help diagnose why revision fetching failed
        import sys

        print(f"Warning: Could not fetch latest revision for {model_id}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


async def _zip_revision_label(revision: str | None, model_id: str | None = None) -> str:
    if not revision:
        if model_id:
            latest_sha = await _get_latest_revision(model_id)
            if latest_sha and _is_full_git_sha(latest_sha):
                return latest_sha[:6].lower()
            elif latest_sha:
                # SHA was returned but not in expected format; use it anyway (shortened if possible)
                return latest_sha[:6].lower() if len(latest_sha) >= 6 else latest_sha.lower()
        # Only print warning once per model, so we check for model_id
        if model_id:
            import sys

            print(
                f"Warning: Using fallback 'main' for {model_id}. If this is incorrect, pass a revision explicitly.",
                file=sys.stderr,
            )
        return "main"
    if _is_full_git_sha(revision):
        return revision[:6].lower()
    return revision


async def model_zip_name(model_name: str) -> str:
    resolved = resolve_model_id(model_name)
    model_id, revision = split_model_name_and_revision(resolved)
    zip_revision = await _zip_revision_label(revision, model_id)
    return f"model-{_filename_slug(model_id)}@{_filename_slug(zip_revision)}.zip"


async def model_zip_candidates(model_name: str) -> list[str]:
    resolved = resolve_model_id(model_name)
    resolved_model_id, resolved_revision = split_model_name_and_revision(resolved)
    zip_revision = await _zip_revision_label(resolved_revision, resolved_model_id)
    candidates = [
        await model_zip_name(model_name),
        f"model-{_legacy_filename_slug(resolved_model_id)}@{_filename_slug(zip_revision)}.zip",
        f"model-{_filename_slug(resolved_model_id)}--{_filename_slug(zip_revision)}.zip",
        f"model-{_filename_slug(resolved_model_id)}.zip",
        f"model-{_legacy_filename_slug(resolved_model_id)}.zip",
    ]
    original_model_id, _ = split_model_name_and_revision(model_name)
    if "/" not in original_model_id:
        candidates.append(f"xenova-{original_model_id}.zip")
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


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


def _read_wav(source, sampling_rate=AUDIO_SAMPLING_RATE):
    """Decode PCM WAV bytes or a file path into mono float samples at ``sampling_rate``."""
    import array
    import io
    import sys
    import wave

    fileobj = io.BytesIO(source) if isinstance(source, (bytes, bytearray, memoryview)) else source
    try:
        with wave.open(fileobj, "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Audio must be a PCM WAV file: {exc}") from exc

    if width == 1:
        samples = [(byte - 128) / 128 for byte in frames]
    elif width == 3:
        samples = [int.from_bytes(frames[i : i + 3], "little", signed=True) / 8388608 for i in range(0, len(frames), 3)]
    else:
        typecode = {2: "h", 4: "i"}.get(width)
        if typecode is None:
            raise ValueError(f"Unsupported WAV sample width: {width} bytes")
        pcm = array.array(typecode, frames)
        if sys.byteorder == "big":
            pcm.byteswap()
        scale = float(2 ** (8 * width - 1))
        samples = [value / scale for value in pcm]

    if channels > 1:
        samples = [sum(samples[i : i + channels]) / channels for i in range(0, len(samples), channels)]
    if rate != sampling_rate and samples:
        # Linear interpolation is sufficient for speech; resample offline for best quality.
        count = max(1, round(len(samples) * sampling_rate / rate))
        step = (len(samples) - 1) / max(1, count - 1)
        resampled = []
        for i in range(count):
            position = i * step
            left = int(position)
            right = min(left + 1, len(samples) - 1)
            resampled.append(samples[left] + (samples[right] - samples[left]) * (position - left))
        samples = resampled
    return samples


def _audio_input(audio):
    """Normalize the input of an audio task to a URL string or a list of float samples."""
    import os

    if isinstance(audio, str):
        if audio.startswith(("http://", "https://", "blob:", "data:")):
            return audio
        if os.path.isfile(audio):
            return _read_wav(audio)
        raise FileNotFoundError(
            f"Audio file not found: {audio!r}. Pass an HTTP(S) URL, a path to a PCM WAV file, "
            f"WAV bytes, or mono samples at {AUDIO_SAMPLING_RATE} Hz."
        )
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return _read_wav(audio)
    return [float(sample) for sample in audio]


_IMAGE_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM")


def _encoded_image(data):
    data = bytes(data)
    is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if not (is_webp or data.startswith(_IMAGE_SIGNATURES)):
        raise ValueError("The image could not be decoded: expected JPEG, PNG, WebP, GIF or BMP bytes.")
    return data


def _image_input(image):
    """Normalize one image to a URL string, encoded bytes, or ``["raw", pixels, width, height, channels]``."""
    import io
    import os

    if isinstance(image, str):
        if image.startswith(("http://", "https://", "blob:", "data:")):
            return image
        if os.path.isfile(image):
            with open(image, "rb") as file:
                return _encoded_image(file.read())
        raise FileNotFoundError(
            f"Image file not found: {image!r}. Pass an HTTP(S) URL, a file path, encoded image bytes, "
            "a uint8 NumPy array of shape (H, W, 3 or 4), or a PIL image."
        )
    if isinstance(image, (bytes, bytearray, memoryview)):
        return _encoded_image(image)
    if hasattr(image, "save") and hasattr(image, "mode"):  # PIL image
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    if hasattr(image, "shape") and hasattr(image, "dtype"):  # NumPy array
        if str(image.dtype) != "uint8" or len(image.shape) != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"A NumPy image must be uint8 with shape (H, W, 3 or 4), got {image.dtype} {image.shape}.")
        height, width, channels = (int(n) for n in image.shape)
        return ["raw", image.tobytes(), width, height, channels]
    raise TypeError(f"Unsupported image input: {type(image).__name__}")


def _image_inputs(inputs):
    """One image, or a list of images for a batch."""
    single = (
        isinstance(inputs, (str, bytes, bytearray, memoryview)) or hasattr(inputs, "shape") or hasattr(inputs, "mode")
    )
    return _image_input(inputs) if single else [_image_input(item) for item in inputs]


def _unpack_images(result, as_array=False):
    """An image-output task's result: ``[JSON structure, one buffer per image or tensor]`` back into dicts and lists.

    Each image is PNG bytes, or a ``uint8`` array of shape (height, width, channels) with ``as_array``; a float
    tensor (the raw depth) is a float32 array, or nested lists without NumPy.
    """
    import json

    try:
        import numpy as np
    except ImportError:
        np = None
        if as_array:
            raise ModuleNotFoundError("as_array needs NumPy: pip install numpy", name="numpy") from None
    buffers = [_bytes_of(buffer) for buffer in result[1:]]

    def build(node):
        if isinstance(node, dict) and "$buffer" in node:
            data = buffers[node["$buffer"]]
            if node.get("dtype") == "float32":
                rows, columns = node["shape"]
                if np is not None:
                    return np.frombuffer(data, dtype="<f4").reshape(rows, columns).astype(np.float32)
                import array

                flat = array.array("f")
                flat.frombytes(data)
                return [flat[row * columns : (row + 1) * columns].tolist() for row in range(rows)]
            if as_array:
                return (
                    np.frombuffer(data, dtype=np.uint8).reshape(node["height"], node["width"], node["channels"]).copy()
                )
            return data
        if isinstance(node, list):
            return [build(item) for item in node]
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        return node

    return build(json.loads(str(result[0])))


class Speech:
    """Synthesised speech: mono float32 ``samples`` at ``sampling_rate`` Hz.

    ``samples`` is a NumPy array when NumPy is installed, else an ``array.array("f")``. Evaluating a ``Speech``
    in a notebook cell shows an audio player; ``to_wav()`` returns a WAV file's bytes.
    """

    def __init__(self, samples, sampling_rate):
        self.samples = samples
        self.sampling_rate = sampling_rate

    @classmethod
    def _from_bytes(cls, sampling_rate, raw):
        """From little-endian float32 bytes, which is how the browser hands over the samples."""
        try:
            import numpy as np
        except ImportError:
            import array

            samples = array.array("f")
            samples.frombytes(raw)
        else:
            samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        return cls(samples, int(sampling_rate))

    @property
    def duration(self):
        """Length in seconds."""
        return len(self.samples) / self.sampling_rate

    def __repr__(self):
        return f"Speech(duration={self.duration:.2f}s, sampling_rate={self.sampling_rate})"

    def to_wav(self):
        """The speech as a 16-bit PCM mono WAV file."""
        import array
        import io
        import sys
        import wave

        pcm = array.array("h", (max(-32768, min(32767, round(float(x) * 32767))) for x in self.samples))
        if sys.byteorder == "big":
            pcm.byteswap()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sampling_rate)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def _repr_html_(self):
        import base64

        data = base64.b64encode(self.to_wav()).decode("ascii")
        return f'<audio controls src="data:audio/wav;base64,{data}"></audio>'


class Pipeline:
    """A loaded Transformers.js task pipeline, created with ``LLM.pipeline()``."""

    def __init__(self, llm, task, model):
        self.llm = llm
        self.task = task
        self.model = model

    def __repr__(self):
        return f"Pipeline(task={self.task!r}, model={self.model!r})"

    async def __call__(self, inputs, **kwargs):
        """Run the task. Keyword arguments are forwarded as Transformers.js task options."""
        import json

        if self.task in AUDIO_INPUT_TASKS:
            inputs = _audio_input(inputs)
        elif self.task in IMAGE_TASKS:
            inputs = _image_inputs(inputs)
        elif self.task in SPEECH_TASKS and not isinstance(inputs, str):
            raise TypeError(f"{self.task} takes one text; use map() for several, got {type(inputs).__name__}")
        elif not isinstance(inputs, str):
            inputs = list(inputs)
        if not isinstance(inputs, str):
            inputs = _to_js(inputs)
        options = json.dumps(kwargs, allow_nan=False)
        result = await self.llm.js.runTask(inputs, options)
        return self._result(_to_python_deep(result), as_array=kwargs.get("as_array", False))

    def _result(self, result, as_array=False):
        """A task result as a Python object: speech arrives as ``[sampling_rate, float32 bytes]``, an image result
        as ``[JSON structure, buffers]``."""
        if self.task in SPEECH_TASKS:
            return Speech._from_bytes(result[0], _bytes_of(result[1]))
        if self.task in IMAGE_OUTPUT_TASKS:
            return _unpack_images(result, as_array)
        return result

    def _prepare(self, item):
        """One input of ``map()`` in the form the JavaScript task takes."""
        if self.task in AUDIO_INPUT_TASKS:
            return _audio_input(item)
        if self.task in IMAGE_TASKS:
            return _image_input(item)
        if not isinstance(item, str):
            raise TypeError(f"map() inputs for {self.task} must be strings, got {type(item).__name__}")
        return item

    async def map(self, inputs, batch_size=8, progress_callback=None, on_error="raise", **kwargs):
        """Run the task over many inputs, with the loop in JavaScript so a notebook makes one call.

        ``batch_size`` inputs go to each Transformers.js call (1 runs them one by one).
        ``progress_callback(done, total)`` is called after each batch. ``on_error`` is ``"raise"`` (the first
        error is raised), ``"skip"`` (failed inputs are left out of the results) or ``"return"`` (an exception
        object takes the place of a failed input's result). Keyword arguments are task options, as in a
        normal call. Results are in input order, as many as ran: ``await pipe.cancel()`` stops after the
        current batch and ``map()`` returns what it has.
        """
        import json

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if on_error not in ("raise", "skip", "return"):
            raise ValueError(f"on_error must be 'raise', 'skip' or 'return', got {on_error!r}")
        items = list(inputs)
        prepared = []  # (index, input) for the inputs JavaScript will run
        failures = {}  # index -> exception, for inputs that could not be prepared
        for index, item in enumerate(items):
            try:
                prepared.append((index, self._prepare(item)))
            except Exception as exc:
                if on_error == "raise":
                    raise
                failures[index] = exc

        def report(done, total):
            if progress_callback is not None:
                progress_callback(int(done) + len(failures), len(items))

        outcome = {"results": [], "errors": []}
        if prepared:
            with _js_callable(report) as callback:
                outcome = _to_python_deep(
                    await self.llm.js.runTasks(
                        _to_js([value for _, value in prepared]),
                        batch_size,
                        json.dumps(kwargs, allow_nan=False),
                        callback,
                        on_error != "raise",
                    )
                )
        results = outcome["results"]
        errors = {position: message for position, message in outcome["errors"]}
        # A cancelled run covers a prefix of the inputs.
        done = len(results)
        limit = len(items) if done == len(prepared) else prepared[done - 1][0] + 1 if done else 0
        positions = {index: position for position, (index, _) in enumerate(prepared)}
        output = []
        for index in range(limit):
            if index in failures:
                value = failures[index]
            elif positions[index] in errors:
                value = RuntimeError(errors[positions[index]])
            else:
                value = self._result(results[positions[index]], as_array=kwargs.get("as_array", False))
            if on_error == "skip" and isinstance(value, Exception):
                continue
            output.append(value)
        return output

    async def cancel(self):
        """Stop a running ``map()`` after its current batch; return False if nothing was running."""
        return await self.llm.cancel()

    async def dispose(self):
        """Release the loaded pipeline without deleting cached model files."""
        await self.llm.dispose()

    async def export_models(self):
        """Export the pipeline model as a zip file to /drive/models."""
        await self.llm.export_models()


class Tokenizer:
    """A tokenize-only handle created with ``LLM.tokenizer()``; ``encode()`` and ``decode()`` are synchronous."""

    def __init__(self, js_tokenizer, source):
        self._js = js_tokenizer
        self.source = source
        self.vocab_size = int(_to_python_deep(js_tokenizer.vocab_size))

    def __repr__(self):
        return f"Tokenizer(source={self.source!r}, vocab_size={self.vocab_size})"

    def _handle(self):
        if self._js is None:
            raise RuntimeError(f"Tokenizer for {self.source!r} is closed; load it again with LLM.tokenizer().")
        return self._js

    def encode(self, text, *, add_special_tokens=False):
        """Return the token ids of ``text`` as a list of ints, without truncation."""
        import json

        if not isinstance(text, str):
            raise TypeError(f"encode() expects a str, got {type(text).__name__}")
        return json.loads(_to_python_deep(self._handle().encode(text, bool(add_special_tokens))))

    def decode(self, ids):
        """Return the text of a sequence of token ids, keeping special tokens."""
        import json

        return _to_python_deep(self._handle().decode(json.dumps([int(i) for i in ids])))

    def close(self):
        """Release the loaded tokenizer; later ``encode()``/``decode()`` calls raise. Repeated calls are safe."""
        self._js = None


@dataclass
class ModelInfo:
    """Best-effort, pre-load metadata returned by ``LLM.model_info()``.

    ``cache_status`` is ``"cached"``, ``"partial"``, ``"not_cached"`` or ``"unknown"`` and
    only reflects the Transformers.js Cache API. ``download_bytes`` is the summed file
    size, or ``None`` when any size is unknown; it is not a runtime memory estimate.
    """

    model_id: str
    revision: str
    task: str
    backend: str
    dtype: str | None
    cache_status: str
    download_bytes: int | None
    files: list
    error: str | None = None


def _check_alternatives(alternatives):
    if isinstance(alternatives, bool) or not isinstance(alternatives, int) or alternatives < 0:
        raise ValueError(f"alternatives must be a nonnegative integer, got {alternatives!r}")


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


async def clear_cache(model=None):
    """Clear browser caches/databases used by Transformers.js model downloads.

    With no ``model``, deletes all caches and returns
    ``{"deleted_caches": [...], "deleted_databases": [...]}``. With a model name,
    deletes only that model's cached files (all revisions) and returns
    ``{"model_id": ..., "deleted_files": [urls]}``.
    """
    result = await _js_function("clearModelCache")(model)
    return _to_python_deep(result)


async def cached_models():
    """List models in the Transformers.js browser cache as dicts, or None if the Cache API is unavailable.

    Each dict has ``model_id``, ``revision``, ``model_type`` (from the cached config.json), ``task`` (the Hub
    pipeline tag, None offline), ``files``, ``file_sizes`` (name to size, None if unknown) and ``size``.
    Hosted model files and /drive/models ZIPs are not included.
    """
    return _to_python_deep(await _js_function("getCachedModels")())


class LLM:
    DEFAULTS = {
        "generator_model": None,
        "embedding_model": None,
        "classification_model": None,
        "backend": "wasm",
        "embedding_backend": None,
        "dtype": None,
        "dtype_fallbacks": None,
        # Explicit dtype for the embedding (feature-extraction) pipeline, e.g.
        # "q8". Pins the quantization so embeddings are reproducible against
        # offline-precomputed vectors; None keeps the Transformers.js default.
        "embedding_dtype": None,
        "max_new_tokens": 50,
        "return_full_text": True,
        "temperature": 0.7,
        "top_k": 50,
        "do_sample": True,
        "local_models_path": "/drive/models",
        "auto_detect_local": True,
    }

    def __init__(self, js_instance, models=None):
        self.js = js_instance
        self._models = [m for m in (models or []) if m]

    # --- lifecycle ---

    @classmethod
    async def model_info(cls, model, task=None, backend="wasm", dtype=None):
        """Inspect a model's files, download size and cache status without loading it.

        Without ``task``, the model's Hub pipeline tag is used (``info.task``); it needs the network.
        """
        get_model_info = _js_function("getModelInfo")
        return ModelInfo(**_to_python_deep(await get_model_info(model, task, backend, dtype)))

    @classmethod
    async def model_task(cls, model):
        """Return the model's Hub pipeline tag (e.g. ``"text-generation"``), or None offline or if untagged."""
        return _to_python_deep(await _js_function("getModelTask")(model))

    @classmethod
    async def create(cls, progress_callback=None, **overrides):
        """Load models. ``progress_callback`` receives a dict per loading event, ending with
        status ``"ready"`` or ``"error"``.

        Outside a browser kernel, or with ``backend="native"``, the ``lmlite[native]`` onnxruntime backend
        runs the same ONNX files for embed()/classify()/generate(). It reads models from a directory path,
        ``local_models_path/<model id>/``, or the Hugging Face Hub cache."""
        with _loading_progress() as progress:
            try:
                return await cls._create(progress, progress_callback, **overrides)
            except Exception as exc:
                if progress_callback is not None:
                    progress_callback(
                        {"model": "", "status": "error", "file": "", "loaded": 0, "total": 0, "error": str(exc)}
                    )
                raise

    @classmethod
    async def _create(cls, progress, progress_callback, **overrides):
        import asyncio
        import os
        import sys
        import zipfile

        config = {**cls.DEFAULTS, **overrides}
        native = _native(config.get("backend") == "native")
        createRuntime = None if native else _js_function("createRuntime")

        # Auto-detect local zip and inject files directly into the browser Cache API.
        # This avoids any filesystem extraction: bytes flow zip → memory → Cache API.
        if not native and config.get("auto_detect_local", True):
            models_to_detect = []
            for model_name in (
                config.get("generator_model"),
                config.get("embedding_model"),
                config.get("classification_model"),
                config.get("task_model"),
            ):
                if model_name and model_name not in models_to_detect:
                    models_to_detect.append(model_name)

            if models_to_detect:
                js_runtime = _runtime()
                is_model_cached = getattr(js_runtime, "isModelCached", None)
                load_model_to_cache = getattr(js_runtime, "loadModelToCache", None)
                hosted_model_base = None
                if sys.platform in {"emscripten", "wasi"} or config.get("hosted_models_url"):
                    hosted_model_base = getattr(js_runtime, "getHostedModelBase", None)
                    if hosted_model_base is None and config.get("hosted_models_url"):
                        raise RuntimeError("JS runtime missing getHostedModelBase; restart the kernel to reload lmLite")

                for model_to_detect in models_to_detect:
                    progress.set_description(f"Checking {model_to_detect}")
                    resolved_model_id = resolve_model_id(model_to_detect)
                    if hosted_model_base is not None:
                        base = await hosted_model_base(model_to_detect, config.get("hosted_models_url"))
                        if base:
                            config["hosted_model_base"] = str(base)
                            config.setdefault("hosted_models", []).append(resolved_model_id.removesuffix("@main"))
                            print(f"Using hosted files for {resolved_model_id} from {base}")
                            continue
                    zip_path = None
                    for zip_filename in await model_zip_candidates(model_to_detect):
                        candidate = f"{config['local_models_path']}/{zip_filename}"
                        if os.path.exists(candidate):
                            zip_path = candidate
                            break

                    cached_in_browser = False
                    if is_model_cached is not None:
                        cached_in_browser = bool(await is_model_cached(model_to_detect))
                        if cached_in_browser:
                            print(f"{resolved_model_id} model found in browser cache")

                    if zip_path:
                        if cached_in_browser:
                            print(f"Skipping local zip injection from {zip_path}")
                        else:
                            if load_model_to_cache is None:
                                raise RuntimeError("JS runtime missing loadModelToCache")
                            print(f"Found zipped model at {zip_path}, loading into browser cache...")
                            model_prefix = f"{resolve_model_id(model_to_detect).split('@', 1)[0]}/"
                            progress.set_description(f"Opening local archive for {model_to_detect}")
                            # Flush widget messages before the filesystem fetches the archive.
                            await asyncio.sleep(0)
                            with zipfile.ZipFile(zip_path, "r") as zf:
                                members = []
                                for name in zf.namelist():
                                    if name.endswith("/"):
                                        continue
                                    normalized_name = name.replace("\\", "/")
                                    if normalized_name.startswith(model_prefix):
                                        filename = normalized_name[len(model_prefix) :]
                                    else:
                                        filename = normalized_name.rsplit("/", 1)[-1]
                                    if (
                                        not filename
                                        or filename.startswith("/")
                                        or any(part in {"", ".", ".."} for part in filename.split("/"))
                                    ):
                                        continue
                                    members.append((name, filename))

                                total_bytes = sum(zf.getinfo(name).file_size for name, _ in members)
                                progress.bar_format = None
                                progress.reset(total=total_bytes)
                                progress.set_description(f"Reading {model_to_detect}")
                                for name, filename in members:
                                    progress.set_description(f"Reading {model_to_detect}: {filename}")
                                    with zf.open(name) as source:
                                        data = bytearray()
                                        while chunk := source.read(1024 * 1024):
                                            data.extend(chunk)
                                            progress.update(len(chunk))
                                            # Let the notebook deliver output while reading large tensors.
                                            await asyncio.sleep(0)
                                    progress.set_description(f"Caching {model_to_detect}: {filename}")
                                    await asyncio.sleep(0)
                                    files = {filename: data}
                                    js_files = _to_js(files)
                                    await load_model_to_cache(model_to_detect, js_files)
                                    del js_files, files, data
                                progress.refresh()

                            progress.bar_format = "{desc}"
                            print(f"Loaded {len(members)} files for {model_to_detect} into browser cache")
                    elif not cached_in_browser:
                        local_models_path = config["local_models_path"].rstrip("/") + "/"
                        print(
                            f"No {resolved_model_id} model found in {local_models_path} nor in browser cache, "
                            "will download from 🤗"
                        )

        js_config = _to_js(config)

        current_file = None

        def report_progress(model, status, filename, loaded, total):
            nonlocal current_file
            if status == "progress":
                progress.bar_format = None
                key = (model, filename)
                if key != current_file or loaded < progress.n or (total and total != progress.total):
                    progress.total = total or None
                    # Pass total to reset so the notebook widget's maximum is updated too.
                    progress.reset(total=total or None)
                    current_file = key
                progress.set_description(f"Downloading {model}: {filename}", refresh=False)
                progress.update(loaded - progress.n)
            else:
                current_file = None
                progress.bar_format = "{desc}"
                progress.set_description(f"{status.capitalize()} {model}")
            if progress_callback is not None:
                progress_callback(
                    {"model": model, "status": status, "file": filename, "loaded": loaded, "total": total}
                )

        progress.bar_format = "{desc}"
        progress.set_description("Initializing browser runtime")
        with _js_callable(report_progress) as callback:
            instance = native.NativeLLM(config) if native else createRuntime(js_config)
            await instance.init(callback)
            progress.bar_format = "{desc}{percentage:3.0f}%|{bar}|"
            progress.reset(total=1)
            progress.update(1)
            progress.set_description("Models ready")

        models = [
            config.get("generator_model"),
            config.get("embedding_model"),
            config.get("classification_model"),
            config.get("task_model"),
        ]
        return cls(instance, models=models)

    @classmethod
    async def pipeline(cls, task, model, **options):
        """Load a Transformers.js task pipeline.

        Supported tasks are listed in ``SUPPORTED_TASKS``. Loading options such as
        ``backend``, ``dtype``, ``dtype_fallbacks``, ``hosted_models_url`` and
        ``local_models_path`` behave as in ``LLM.create()``.
        """
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task {task!r}. Supported pipeline tasks are: {', '.join(SUPPORTED_TASKS)}. "
                "Use LLM.create() with generate()/chat(), embed() or classify() for text generation, "
                "embeddings and text classification."
            )
        if not model:
            raise ValueError(f"Task {task!r} requires a model.")
        model_options = {"generator_model", "embedding_model", "classification_model", "task", "task_model"}
        conflicting = sorted(model_options & options.keys())
        if conflicting:
            raise TypeError(f"LLM.pipeline() does not accept {', '.join(conflicting)}; use LLM.create() instead.")
        llm = await cls.create(task=task, task_model=model, **options)
        return Pipeline(llm, task, model)

    @classmethod
    async def tokenizer(cls, model=None, *, tokenizer_json=None, tokenizer_config=None, hosted_models_url=None):
        """Load only a tokenizer, without model weights, and return a ``Tokenizer``.

        Pass exactly one of ``model`` (hosted ``files/models/`` first, then Hub files, as in
        ``LLM.create()``) or ``tokenizer_json`` (an HTTP(S) URL, or a path relative to the
        JupyterLite site root such as ``"files/models/org/name/tokenizer.json"``), optionally with
        ``tokenizer_config``. Each call loads an independent handle.
        """
        if bool(model) == bool(tokenizer_json):
            raise ValueError("Pass exactly one of model= or tokenizer_json=.")
        if model and tokenizer_config:
            raise ValueError("tokenizer_config= requires tokenizer_json=; model= loads the model's own config.")
        source = model or tokenizer_json
        native = _native()
        load_tokenizer = native.load_tokenizer if native else _js_function("loadTokenizer")
        tokenizer = Tokenizer(await load_tokenizer(model, tokenizer_json, tokenizer_config, hosted_models_url), source)
        # Fail at load time, not on the first call, if this kernel's bridge cannot call encode synchronously.
        try:
            tokenizer.encode("a")
        except Exception as exc:
            raise RuntimeError(f"This kernel cannot call the tokenizer for {source!r} synchronously: {exc!r}") from exc
        return tokenizer

    @classmethod
    async def hosted_base(cls, model_id, *, hosted_models_url=None):
        """Return the hosted models base URL serving ``model_id``, or None if it is not hosted.

        Uses the same discovery as ``LLM.create()``: the site's ``files/models/manifest.json``, or
        ``hosted_models_url`` when given. Revisions other than ``@main`` are never hosted. Raises
        when an explicit ``hosted_models_url`` manifest is missing or invalid.
        """
        base = await _js_function("getHostedModelBase")(model_id, hosted_models_url)
        return str(base) if base else None

    # --- core API ---

    async def dispose(self):
        """Release loaded inference resources without deleting cached model files.

        Await active operations before disposal. Repeated calls are safe;
        failed pipeline cleanups raise and can be retried.
        """
        await self.js.dispose()

    async def generate(self, prompt: str):
        """Generate text, including the prompt unless created with return_full_text=False."""
        return await self.js.generate(prompt)

    async def chat(self, messages):
        """Return an assistant response using the loaded tokenizer's chat template.

        Messages require string ``role`` and ``content`` fields. Models without
        a chat template use generic role labels, not instruction tuning.
        The response excludes the prompt regardless of ``return_full_text``.
        Template errors (including unsupported roles) propagate without fallback.
        """
        import json

        # A JSON string crosses both browser bridges without runtime-specific
        # dict conversion options; JavaScript parses it into plain objects.
        return await self.js.chat(json.dumps(messages, allow_nan=False))

    def chat_stream(self, messages):
        """Async iterator over ``chat()`` response text chunks as they are generated.

        ``"".join(chunks).strip()`` is the ``chat()`` answer. Generation starts on the
        first iteration; leaving the loop early (``break``, an exception or an interrupt)
        cancels it. One generation runs per instance at a time.
        """
        import json

        return self._stream("chatStream", json.dumps(messages, allow_nan=False))

    def generate_stream(self, prompt: str):
        """Async iterator over newly generated text chunks, never including the prompt."""
        return self._stream("generateStream", prompt)

    def chat_tokens(self, messages, alternatives=1):
        """Async iterator over ``chat()`` response tokens with their probabilities.

        Yields ``{"text", "prob", "alternatives": [{"text", "prob"}, ...]}`` per token, where
        ``alternatives`` are the ``alternatives`` most probable other tokens at that step.
        Probabilities are those the token picker used (after repetition penalty, and temperature
        when sampling). Joined ``text`` values equal joined ``chat_stream()`` chunks. Cancellation
        and concurrency behave as in ``chat_stream()``.
        """
        import json

        _check_alternatives(alternatives)
        return self._stream("chatTokens", json.dumps(messages, allow_nan=False), alternatives, parse=json.loads)

    def generate_tokens(self, prompt: str, alternatives=1):
        """Like ``chat_tokens()`` for a plain prompt; never includes the prompt."""
        import json

        _check_alternatives(alternatives)
        return self._stream("generateTokens", prompt, alternatives, parse=json.loads)

    async def _stream(self, method, *args, parse=None):
        stream = await getattr(self.js, method)(*args)
        finished = False
        try:
            while chunk := _to_python_deep(await stream.read()):
                yield parse(chunk) if parse else chunk
            finished = True
        finally:
            if not finished:
                # Not awaited: the interrupt is synchronous in JS, and a generator closed
                # by garbage collection cannot await.
                self.js.cancel()

    async def cancel(self):
        """Stop the running generation after its current token; return False if none was running.

        Streams end early without error and ``generate()``/``chat()`` return the partial text.
        """
        return bool(_to_python_deep(await self.js.cancel()))

    async def embed(self, text: str):
        vec = await self.js.embed(text)
        return list(vec)

    async def classify(self, text: str, top_k=None):
        if top_k is None:
            result = await self.js.classify(text)
        else:
            result = await self.js.classify(text, top_k)
        return _to_python_deep(result)

    # --- similarity ---

    async def similarity_search(self, query, docs, k=None):
        """Search for the most similar documents to a query.

        Args:
            query: The query text
            docs: List of documents to search
            k: Number of top results to return (default: all)

        Returns:
            List of (doc, similarity_score) tuples, sorted by similarity descending
        """
        cosineSimilarityBatch = _js_function("cosineSimilarityBatch")

        query_vec = await self.embed(query)

        doc_vecs = []
        for doc in docs:
            doc_vecs.append(await self.embed(doc))

        # Convert Python lists to JS arrays for the JS function
        js_query_vec = _to_js(query_vec)
        js_doc_vecs = _to_js(doc_vecs)

        sims = cosineSimilarityBatch(js_query_vec, js_doc_vecs)

        results = list(zip(docs, sims))
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top k results if specified
        if k is not None:
            return results[:k]
        return results

    # --- model export ---

    async def export_models(self):
        """Export all configured models as zip files to /drive/models."""
        import os
        import zipfile

        if not self._models:
            msg = "No models configured on this instance."
            print(msg)
            return

        for model_name in self._models:
            zip_path = f"/drive/models/{await model_zip_name(model_name)}"

            if os.path.exists(zip_path):
                message = f"Model already exported at {zip_path}"
                print(message)
                continue

            result = await self.js.exportModelFiles(model_name)

            if hasattr(result, "success") and result.success:
                print(f"Exporting {len(result.files)} files for {result.model}")
                os.makedirs("/drive/models", exist_ok=True)

                # Deduplicate files by name, keeping the largest version
                files_by_name = {}
                for file_obj in result.files:
                    filename = file_obj.filename
                    file_bytes = bytes(file_obj.data.to_py())
                    if filename not in files_by_name or len(file_bytes) > len(files_by_name[filename]):
                        files_by_name[filename] = file_bytes

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for filename, file_bytes in files_by_name.items():
                        arcname = f"{resolve_model_id(result.model)}/{filename}"
                        zipf.writestr(arcname, file_bytes)
                        print(f"Added to zip: {arcname} ({len(file_bytes)} bytes)")

                message = f"Successfully exported {len(files_by_name)} files to {zip_path}"
                print(message)
            else:
                print(f"Export result: {result}")
