# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Model names, revisions, ZIP names, the browser model cache, and what a model needs before it loads."""

from dataclasses import dataclass

from ._bridge import _js_function, _runtime, _to_python_deep

DEFAULT_MODEL_NAMESPACE = "Xenova"


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


async def model_info(model, task=None, backend="wasm", dtype=None):
    """Inspect a model's files, download size and cache status without loading it.

    Without ``task``, the model's Hub pipeline tag is used (``info.task``); it needs the network.
    """
    get_model_info = _js_function("getModelInfo")
    return ModelInfo(**_to_python_deep(await get_model_info(model, task, backend, dtype)))


async def model_task(model):
    """Return the model's Hub pipeline tag (e.g. ``"text-generation"``), or None offline or if untagged."""
    return _to_python_deep(await _js_function("getModelTask")(model))


async def hosted_base(model_id, *, hosted_models_url=None):
    """Return the hosted models base URL serving ``model_id``, or None if it is not hosted.

    Uses the same discovery as ``LLM.create()``: the site's ``files/models/manifest.json``, or
    ``hosted_models_url`` when given. Revisions other than ``@main`` are never hosted. Raises
    when an explicit ``hosted_models_url`` manifest is missing or invalid.
    """
    base = await _js_function("getHostedModelBase")(model_id, hosted_models_url)
    return str(base) if base else None
