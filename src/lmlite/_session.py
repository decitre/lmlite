# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

from ._bridge import _js_callable, _js_function, _loading_progress, _native, _runtime, _to_js
from .models import model_zip_candidates, model_zip_name, resolve_model_id


class Session:
    """A loaded set of models in one runtime: the browser's JavaScript runtime, or the native backend."""

    DEFAULTS = {
        "generator_model": None,
        "embedding_model": None,
        "classification_model": None,
        "backend": "wasm",
        "embedding_backend": None,
        "backend_fallback": True,
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

    @property
    def backend(self):
        """The backend that loaded the models: ``"webgpu"``, ``"wasm"`` or ``"native"``. After a fallback from WebGPU
        (see ``backend_fallback``) it is ``"wasm"``, and ``backend_fallback_reason`` says why."""
        return getattr(self.js, "backend", None) or "native"  # the native backend has no such attribute

    @property
    def backend_fallback_reason(self):
        """Why WebGPU was given up for WASM, or ``None`` if it was not."""
        reason = getattr(self.js, "fallbackReason", None)
        return str(reason) if reason else None

    # --- lifecycle ---

    @classmethod
    async def create(cls, progress_callback=None, **overrides):
        """Load models. ``progress_callback`` receives a dict per loading event, ending with
        status ``"ready"`` or ``"error"``.

        Outside a browser kernel, or with ``backend="native"``, the ``lmlite[native]`` onnxruntime backend
        runs the same ONNX files for embed()/classify()/generate(). It reads models from a directory path,
        ``local_models_path/<model id>/``, or the Hugging Face Hub cache."""
        with _loading_progress(quiet=bool(overrides.get("quiet"))) as progress:
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
        say = (lambda *args, **kwargs: None) if config.get("quiet") else print  # quiet: no messages and no progress bar
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
                            say(f"Using hosted files for {resolved_model_id} from {base}")
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
                            say(f"{resolved_model_id} model found in browser cache")

                    if zip_path:
                        if cached_in_browser:
                            say(f"Skipping local zip injection from {zip_path}")
                        else:
                            if load_model_to_cache is None:
                                raise RuntimeError("JS runtime missing loadModelToCache")
                            say(f"Found zipped model at {zip_path}, loading into browser cache...")
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
                            say(f"Loaded {len(members)} files for {model_to_detect} into browser cache")
                    elif not cached_in_browser:
                        local_models_path = config["local_models_path"].rstrip("/") + "/"
                        say(
                            f"No {resolved_model_id} model found in {local_models_path} nor in browser cache, "
                            "will download from 🤗"
                        )

        js_config = config if native else _to_js(config)  # the native backend has no JavaScript runtime

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

        reason = None if native else getattr(instance, "fallbackReason", None)
        if reason:  # WebGPU was asked for and given up: say so, so a benchmark or a bug report is not misread
            say(f"WebGPU was not used, the models run on WASM: {reason}")

        models = [
            config.get("generator_model"),
            config.get("embedding_model"),
            config.get("classification_model"),
            config.get("task_model"),
        ]
        return cls(instance, models=models)

    async def dispose(self):
        """Release the loaded models and their inference sessions, including GPU resources, without deleting cached model
        files.

        Await any running call first. Calling it again is safe; if a release fails it raises, and the call can be retried.
        After a successful dispose the object rejects further inference calls; create a new one to load from the cache
        again. Memory is reclaimed when the browser decides, which may be later.
        """
        await self.js.dispose()

    async def export_models(self):
        """Save the configured models from the browser cache as zip files in ``/drive/models``.

        The files are named ``model-<namespace>-<model>@<revision>.zip`` and can be downloaded from JupyterLite. They can be
        imported again through the ``local_models_path`` option of the ``create()`` methods.
        """
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
