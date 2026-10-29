# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from .bridge.runtime import ensure_runtime

DEFAULT_MODEL_NAMESPACE = "Xenova"

# Transformers.js tasks available through LLM.pipeline(); keep in sync with llm.js.
SUPPORTED_TASKS = ("zero-shot-classification", "automatic-speech-recognition", "translation")

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
        from pjs import to_py as pjs_to_py

        converted = pjs_to_py(value)
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


def _to_js(value):
    try:
        from pyodide.ffi import to_js

        return to_js(value)
    except ImportError:
        try:
            from pjs import to_js

            return to_js(value)
        except ImportError:
            return value


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
    """Normalize ASR input to a URL string or a list of float samples."""
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

        if self.task == "automatic-speech-recognition":
            inputs = _audio_input(inputs)
        elif not isinstance(inputs, str):
            inputs = list(inputs)
        if not isinstance(inputs, str):
            inputs = _to_js(inputs)
        options = json.dumps(kwargs, allow_nan=False)
        result = await self.llm.js.runTask(inputs, options)
        return _to_python_deep(result)

    async def dispose(self):
        """Release the loaded pipeline without deleting cached model files."""
        await self.llm.dispose()

    async def export_models(self):
        """Export the pipeline model as a zip file to /drive/models."""
        await self.llm.export_models()


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
    import js

    return getattr(js, "lmLite", None)


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
        status ``"ready"`` or ``"error"``."""
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

        createLLM = _js_function("createLLM")

        config = {**cls.DEFAULTS, **overrides}

        # Auto-detect local zip and inject files directly into the browser Cache API.
        # This avoids any filesystem extraction: bytes flow zip → memory → Cache API.
        if config.get("auto_detect_local", True):
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
                                    js_files = files
                                    try:
                                        from pyodide.ffi import to_js

                                        js_files = to_js(files)
                                    except ImportError:
                                        try:
                                            from pjs import to_js

                                            js_files = to_js(files)
                                        except ImportError:
                                            pass

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

        js_config = config
        try:
            from pyodide.ffi import to_js

            js_config = to_js(config)
        except ImportError:
            try:
                from pjs import to_js

                js_config = to_js(config)
            except ImportError:
                pass

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
        callback_handle = None
        try:
            from pyodide.ffi import create_proxy
        except ImportError:
            try:
                from pyjs import create_callable
            except ImportError:
                callback = report_progress
            else:
                # Xeus exposes Python functions as objects, not JavaScript callables.
                callback, callback_handle = create_callable(report_progress)
            proxy = None
        else:
            proxy = callback = create_proxy(report_progress)

        try:
            instance = createLLM(js_config)
            await instance.init(callback)
            progress.bar_format = "{desc}{percentage:3.0f}%|{bar}|"
            progress.reset(total=1)
            progress.update(1)
            progress.set_description("Models ready")
        finally:
            if proxy is not None:
                proxy.destroy()
            if callback_handle is not None:
                callback_handle.delete()

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
        try:
            from pyodide.ffi import to_js

            js_query_vec = to_js(query_vec)
            js_doc_vecs = to_js(doc_vecs)
        except ImportError:
            try:
                from pjs import to_js

                js_query_vec = to_js(query_vec)
                js_doc_vecs = to_js(doc_vecs)
            except ImportError:
                # Fallback: hope it works
                js_query_vec = query_vec
                js_doc_vecs = doc_vecs

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
