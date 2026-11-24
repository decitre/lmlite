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
    """Sentence embeddings and similarity search.

    Load it with ``await Embedder.create(model)``. The vectors come from attention-mask-weighted mean pooling followed
    by L2 normalisation, so a dot product of two vectors is their cosine similarity.

    Example:
        .. code-block:: python

            embedder = await Embedder.create("all-MiniLM-L6-v2", backend="wasm", dtype="q8")
            vector = await embedder.embed("Hello world")
            await embedder.similarity_search("Where does JupyterLite run?", docs, k=2)
            await embedder.dispose()
    """

    @classmethod
    async def create(cls, model, *, backend=None, dtype=None, progress_callback=None, **options):
        """Load an embedding model.

        Args:
            model: A Hugging Face model ID such as ``"Xenova/all-MiniLM-L6-v2"``, or a short name, which gets the default
                ``Xenova/`` namespace.
            backend: ``"wasm"`` or ``"webgpu"``, or ``"native"`` for the onnxruntime backend outside a browser. ``None``
                (the default) leaves the device to Transformers.js. A backend that does not suit the model fails; there is
                no retry on another one.
            dtype: Pins the weights, for example ``"q8"``, so the vectors are reproducible. There is no fallback to
                another dtype. ``None`` (the default) keeps the Transformers.js default.
            progress_callback: Called with a dict per loading event, as for ``LLM.create()``.
            **options: The other loading options of ``LLM.create()``: ``hosted_models_url``, ``auto_detect_local`` and
                ``local_models_path``.

        Returns:
            The loaded ``Embedder``; call ``dispose()`` when you are done.
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
        """The sentence vector of a text.

        Args:
            text: The text to embed.

        Returns:
            A flat list of floats, normalised to unit length. Its length is the model's dimension, 384 for MiniLM.
        """
        return await _embed(self, text)

    async def similarity_search(self, query, docs, k=None):
        """Rank documents by their cosine similarity to a query.

        The query and every document are embedded on each call, in memory; nothing is stored between calls.

        Args:
            query: The query text.
            docs: The documents to search, a list of strings.
            k: How many results to return. By default, all of them.

        Returns:
            A list of ``(doc, score)`` tuples, the most similar first.
        """
        return await _similarity_search(self, query, docs, k)
