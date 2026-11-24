# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

from .llm import LLM, SUPPORTED_TASKS, ModelInfo, Pipeline, Speech, Tokenizer, cached_models, clear_cache
from .onnx_model import OnnxModel
from .version import __version__


# The functions that are not about language models, at the top level: ``lmlite.pipeline(...)`` is
# ``LLM.pipeline(...)``, which keeps working. They look ``LLM`` up when called.
async def pipeline(task, model, **options):
    """Load a Transformers.js task and return a ``Pipeline``: see ``LLM.pipeline``."""
    return await LLM.pipeline(task, model, **options)


async def tokenizer(model=None, *, tokenizer_json=None, tokenizer_config=None, hosted_models_url=None):
    """Load only a tokenizer: see ``LLM.tokenizer``."""
    return await LLM.tokenizer(
        model, tokenizer_json=tokenizer_json, tokenizer_config=tokenizer_config, hosted_models_url=hosted_models_url
    )


async def model_info(model, task=None, backend="wasm", dtype=None):
    """Download size and cache status of a model before loading it: see ``LLM.model_info``."""
    return await LLM.model_info(model, task, backend, dtype)


async def model_task(model):
    """The Hub pipeline tag of a model: see ``LLM.model_task``."""
    return await LLM.model_task(model)


async def hosted_base(model_id, *, hosted_models_url=None):
    """The hosted base URL serving a model, or None: see ``LLM.hosted_base``."""
    return await LLM.hosted_base(model_id, hosted_models_url=hosted_models_url)


__all__ = [
    "LLM",
    "ModelInfo",
    "OnnxModel",
    "Pipeline",
    "SUPPORTED_TASKS",
    "Speech",
    "Tokenizer",
    "cached_models",
    "clear_cache",
    "hosted_base",
    "model_info",
    "model_task",
    "pipeline",
    "tokenizer",
    "__version__",
]
