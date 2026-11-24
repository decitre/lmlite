# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import warnings

from ._bridge import _to_python_deep
from ._session import Session
from .classifier import _classify
from .embedder import _embed, _similarity_search
from .models import hosted_base, model_info, model_task
from .pipeline import pipeline
from .tokenizer import tokenizer


def _deprecated(old, new):
    warnings.warn(f"{old} is deprecated and will be removed in 0.15: use {new}.", DeprecationWarning, stacklevel=3)


def _check_alternatives(alternatives):
    if isinstance(alternatives, bool) or not isinstance(alternatives, int) or alternatives < 0:
        raise ValueError(f"alternatives must be a nonnegative integer, got {alternatives!r}")


class LLM(Session):
    """Generation: the large language model loaded by ``LLM.create(generator_model=...)``."""

    # --- core API ---

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

    # --- deprecated: embeddings and classification are not about language models ---

    @classmethod
    async def create(cls, progress_callback=None, **overrides):
        """Load the generator model: ``LLM.create(generator_model=...)``. See ``Session.create``.

        ``embedding_model`` and ``classification_model`` still load, with a ``DeprecationWarning``: use
        ``Embedder.create()`` and ``Classifier.create()``. They will be removed in 0.15.
        """
        legacy = [name for name in ("embedding_model", "classification_model") if overrides.get(name)]
        if legacy:
            _deprecated(
                f"LLM.create({', '.join(f'{name}=' for name in legacy)})", "Embedder.create() or Classifier.create()"
            )
        return await super().create(progress_callback, **overrides)

    async def embed(self, text: str):
        """Deprecated: use ``Embedder.embed``."""
        _deprecated("LLM.embed()", "Embedder.create(...).embed()")
        return await _embed(self, text)

    async def classify(self, text: str, top_k=None):
        """Deprecated: use ``Classifier.classify``."""
        _deprecated("LLM.classify()", "Classifier.create(...).classify()")
        return await _classify(self, text, top_k)

    async def similarity_search(self, query, docs, k=None):
        """Deprecated: use ``Embedder.similarity_search``."""
        _deprecated("LLM.similarity_search()", "Embedder.create(...).similarity_search()")
        return await _similarity_search(self, query, docs, k)

    # --- not about language models: the same as the functions in lmlite ---

    @classmethod
    async def model_info(cls, model, task=None, backend="wasm", dtype=None):
        """See ``lmlite.model_info``."""
        return await model_info(model, task, backend, dtype)

    @classmethod
    async def model_task(cls, model):
        """See ``lmlite.model_task``."""
        return await model_task(model)

    @classmethod
    async def pipeline(cls, task, model, **options):
        """See ``lmlite.pipeline``."""
        return await pipeline(task, model, **options)

    @classmethod
    async def tokenizer(cls, model=None, *, tokenizer_json=None, tokenizer_config=None, hosted_models_url=None):
        """See ``lmlite.tokenizer``."""
        return await tokenizer(
            model, tokenizer_json=tokenizer_json, tokenizer_config=tokenizer_config, hosted_models_url=hosted_models_url
        )

    @classmethod
    async def hosted_base(cls, model_id, *, hosted_models_url=None):
        """See ``lmlite.hosted_base``."""
        return await hosted_base(model_id, hosted_models_url=hosted_models_url)
