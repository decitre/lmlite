# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Sentence embeddings and similarity search."""

from ._bridge import _js_function, _to_js
from ._session import Session


async def _embed(session, text):
    vec = await session.js.embed(text)
    return list(vec)


async def _similarity_search(session, query, docs, k=None):
    cosineSimilarityBatch = _js_function("cosineSimilarityBatch")

    query_vec = await _embed(session, query)

    doc_vecs = []
    for doc in docs:
        doc_vecs.append(await _embed(session, doc))

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


class Embedder(Session):
    """Sentence embeddings: the model loaded by ``Embedder.create(model)``."""

    @classmethod
    async def create(cls, model, *, backend=None, dtype=None, progress_callback=None, **options):
        """Load an embedding model.

        ``backend`` is ``"wasm"`` or ``"webgpu"``, or ``"native"`` for the onnxruntime backend; ``None`` leaves the device
        to Transformers.js. ``dtype``
        pins the weights (for example ``"q8"``) with no fallback to another dtype, so the vectors are
        reproducible; ``None`` keeps the Transformers.js default. ``options`` are the loading options of
        ``LLM.create()``: ``hosted_models_url``, ``auto_detect_local`` and ``local_models_path``.
        """
        native = backend == "native"  # the onnxruntime backend runs the whole session, not just a device
        return await super().create(
            progress_callback,
            embedding_model=model,
            embedding_backend=None if native else backend,
            embedding_dtype=dtype,
            **({"backend": "native"} if native else {}),
            **options,
        )

    async def embed(self, text: str):
        """A flat sentence vector: attention-mask-weighted mean pooling, then L2 normalisation."""
        return await _embed(self, text)

    async def similarity_search(self, query, docs, k=None):
        """Search for the most similar documents to a query.

        Args:
            query: The query text
            docs: List of documents to search
            k: Number of top results to return (default: all)

        Returns:
            List of (doc, similarity_score) tuples, sorted by similarity descending
        """
        return await _similarity_search(self, query, docs, k)
