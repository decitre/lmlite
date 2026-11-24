# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

from ._bridge import _js_function, _to_js, _to_python_deep
from ._session import Session
from .models import hosted_base, model_info, model_task
from .pipeline import pipeline
from .tokenizer import tokenizer


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

    async def embed(self, text: str):
        vec = await self.js.embed(text)
        return list(vec)

    async def classify(self, text: str, top_k=None):
        if top_k is None:
            result = await self.js.classify(text)
        else:
            result = await self.js.classify(text, top_k)
        return _to_python_deep(result)

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
