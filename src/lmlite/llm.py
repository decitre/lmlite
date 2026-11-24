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
    """A large language model for text generation, chat and streaming.

    Load it with ``await LLM.create(generator_model=...)``. For embeddings use ``Embedder``, for classification
    ``Classifier``, and for everything else ``pipeline()``.

    Example:
        .. code-block:: python

            llm = await LLM.create(generator_model="HuggingFaceTB/SmolLM2-360M-Instruct", backend="wasm", dtype="q4")
            await llm.chat([{"role": "user", "content": "Explain embeddings in simple terms."}])
            async for chunk in llm.chat_stream([{"role": "user", "content": "Hello"}]):
                print(chunk, end="")
            await llm.dispose()
    """

    # --- core API ---

    async def generate(self, prompt: str):
        """Generate text from a prompt.

        Args:
            prompt: The text to continue.

        Returns:
            The prompt followed by the generated text, or only the generated text when the model was loaded with
            ``return_full_text=False``. If ``cancel()`` stopped it, the text so far.
        """
        return await self.js.generate(prompt)

    async def chat(self, messages):
        """Answer a conversation, using the chat template of the loaded model's tokenizer.

        Args:
            messages: A list of ``{"role": ..., "content": ...}`` dicts with string values, such as ``"system"``,
                ``"user"`` and ``"assistant"``. A model without a chat template gets generic role labels, which does
                not make a completion model follow instructions.

        Returns:
            The assistant's reply, never the prompt, whatever ``return_full_text`` is. If ``cancel()`` stopped it, the
            text so far.

        Raises:
            Exception: A template error, such as a role the template does not support. There is no fallback format.
        """
        import json

        # A JSON string crosses both browser bridges without runtime-specific
        # dict conversion options; JavaScript parses it into plain objects.
        return await self.js.chat(json.dumps(messages, allow_nan=False))

    def chat_stream(self, messages):
        """Stream the answer to a conversation as it is generated.

        Generation starts on the first iteration. Leaving the loop early (``break``, an exception or a kernel
        interrupt) cancels it. One generation runs per instance at a time.

        Args:
            messages: As for ``chat()``.

        Yields:
            Text chunks, usually a word with its whitespace. ``"".join(chunks).strip()`` is the ``chat()`` answer.

        Example:
            .. code-block:: python

                async for chunk in llm.chat_stream([{"role": "user", "content": "Explain embeddings simply."}]):
                    print(chunk, end="", flush=True)
        """
        import json

        return self._stream("chatStream", json.dumps(messages, allow_nan=False))

    def generate_stream(self, prompt: str):
        """Stream newly generated text for a prompt.

        Args:
            prompt: The text to continue.

        Yields:
            Text chunks, never including the prompt, whatever ``return_full_text`` is. Cancellation and concurrency are
            as for ``chat_stream()``.
        """
        return self._stream("generateStream", prompt)

    def chat_tokens(self, messages, alternatives=1):
        """Stream the answer to a conversation one token at a time, with the probabilities.

        Cancellation and concurrency are as for ``chat_stream()``.

        Args:
            messages: As for ``chat()``.
            alternatives: How many other likely tokens to report at each step (default 1; 0 for none).

        Yields:
            One dict per token: ``{"text": str, "prob": float, "alternatives": [{"text": str, "prob": float}, ...]}``.
            The probabilities are those the token picker used, after the repetition penalty and, when sampling, the
            temperature. The joined ``text`` values equal the joined ``chat_stream()`` chunks.

        Raises:
            ValueError: ``alternatives`` is not a nonnegative integer.
        """
        import json

        _check_alternatives(alternatives)
        return self._stream("chatTokens", json.dumps(messages, allow_nan=False), alternatives, parse=json.loads)

    def generate_tokens(self, prompt: str, alternatives=1):
        """Like ``chat_tokens()`` for a plain prompt. The prompt is never included.

        Args:
            prompt: The text to continue.
            alternatives: How many other likely tokens to report at each step (default 1; 0 for none).

        Yields:
            Dicts as for ``chat_tokens()``.
        """
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
        """Stop the running generation after its current token.

        A stream ends early without an error, and ``generate()`` and ``chat()`` return the text so far. The instance stays
        usable.

        Returns:
            ``False`` if nothing was running.
        """
        return bool(_to_python_deep(await self.js.cancel()))

    # --- loading ---

    @classmethod
    async def create(cls, progress_callback=None, **overrides):
        """Load a generator model.

        Progress is shown in the notebook cell, with status messages while the model is prepared, loaded and warmed
        up. The cell stays busy until that finishes.

        Args:
            generator_model: A Hugging Face model ID such as ``"HuggingFaceTB/SmolLM2-360M-Instruct"``, or a short name,
                which gets the default ``Xenova/`` namespace.
            backend: ``"wasm"`` (the default) or ``"webgpu"``; ``"native"`` runs onnxruntime outside a browser
                (``pip install 'lmlite[native]'``).
            dtype: The weights to load, such as ``"q4"`` or ``"fp32"``. By default the first of ``q8``, ``int8`` and
                ``fp32`` that exists. On WebGPU ``"q4"`` is the fastest; see the model guide.
            dtype_fallbacks: Dtypes to try in order; takes precedence over ``dtype``.
            max_new_tokens: How many tokens to generate at most (default 50).
            return_full_text: ``True`` (the default) makes ``generate()`` return the prompt followed by the
                continuation; ``False`` returns only the continuation.
            temperature: Sampling temperature (default 0.7).
            top_k: Sampling considers this many most likely tokens (default 50).
            do_sample: ``True`` (the default) samples; ``False`` picks the most probable token each time.
            hosted_models_url: An explicit URL of a models directory that has a ``manifest.json``.
            auto_detect_local: Look for hosted model files and ZIPs automatically (default ``True``).
            local_models_path: The directory of ZIP archives to import (default ``"/drive/models"``).
            progress_callback: Called with a dict per loading event (``model``, ``status``, ``file``, ``loaded``,
                ``total``); the last event has the status ``"ready"`` or ``"error"``.
            quiet: Show no progress bar and no loading messages (default ``False``), for an app that has its own.

        Returns:
            The loaded ``LLM``; call ``dispose()`` when you are done.
        """
        legacy = [name for name in ("embedding_model", "classification_model") if overrides.get(name)]
        if legacy:
            _deprecated(
                f"LLM.create({', '.join(f'{name}=' for name in legacy)})", "Embedder.create() or Classifier.create()"
            )
        return await super().create(progress_callback, **overrides)

    # --- deprecated: embeddings and classification are not about language models; removed in 0.15 ---

    async def embed(self, text: str):
        _deprecated("LLM.embed()", "Embedder.create(...).embed()")
        return await _embed(self, text)

    async def classify(self, text: str, top_k=None):
        _deprecated("LLM.classify()", "Classifier.create(...).classify()")
        return await _classify(self, text, top_k)

    async def similarity_search(self, query, docs, k=None):
        _deprecated("LLM.similarity_search()", "Embedder.create(...).similarity_search()")
        return await _similarity_search(self, query, docs, k)

    # --- not about language models: the same as the functions in lmlite ---

    @classmethod
    async def model_info(cls, model, task=None, backend="wasm", dtype=None):
        return await model_info(model, task, backend, dtype)

    @classmethod
    async def model_task(cls, model):
        return await model_task(model)

    @classmethod
    async def pipeline(cls, task, model, **options):
        return await pipeline(task, model, **options)

    @classmethod
    async def tokenizer(cls, model=None, *, tokenizer_json=None, tokenizer_config=None, hosted_models_url=None):
        return await tokenizer(
            model, tokenizer_json=tokenizer_json, tokenizer_config=tokenizer_config, hosted_models_url=hosted_models_url
        )

    @classmethod
    async def hosted_base(cls, model_id, *, hosted_models_url=None):
        return await hosted_base(model_id, hosted_models_url=hosted_models_url)
