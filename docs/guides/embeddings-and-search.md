# Embeddings, classification and similarity search

## Embeddings

```python
# Generate embeddings
embedding = await llm.embed("Hello world")
print(embedding[:5])  # First 5 dimensions
```

## Text Classification

```python
# Classify text (requires classification_model)
llm_cls = await LLM.create(classification_model="distilbert-base-uncased-finetuned-sst-2-english")
labels = await llm_cls.classify("I love transformers!", top_k=2)
print(labels)
```

## Similarity Search

```python
# Similarity search
docs = [
    "JupyterLite runs entirely in the browser.",
    "Python is widely used for machine learning.",
    "TypeScript is great for frontend applications.",
]

# Search with all results
results = await llm.similarity_search(
    "Where does JupyterLite run?",
    docs
)

for doc, score in results:
    print(f"{score:.3f} -> {doc}")

# Search for top 2 results
top_results = await llm.similarity_search(
    "Where does JupyterLite run?",
    docs,
    k=2
)
```

## Sentence Embeddings and Dtype Pinning

`embed(text)` returns a flat, fixed-dimension sentence vector using attention-mask-weighted mean pooling and L2 normalization. Pooling makes token-level outputs suitable for sentence comparison; normalization makes dot products equivalent to cosine similarity for nonzero vectors. `similarity_search()` already normalizes when computing cosine scores.

Pin the embedding weights independently of generation and classification:

```python
llm_embed = await LLM.create(
    embedding_model="all-MiniLM-L6-v2",
    embedding_dtype="q8",
)
embedding = await llm_embed.embed("JupyterLite runs in the browser.")
```

Set `embedding_backend="wasm"` or `"webgpu"` to choose the embedding device independently of `backend`. The default, `None`, omits the device option and lets Transformers.js choose; it does not inherit `backend`. For example, use `backend="wasm", embedding_backend="webgpu"` for WASM generation and WebGPU embeddings. Unsupported device/model/dtype combinations fail initialization; lmLite does not retry on another embedding backend.

`embedding_dtype=None` (the default) leaves dtype selection to Transformers.js. An explicit value selects only that dtype: unavailable or incompatible weights fail initialization rather than falling back. `dtype` and `dtype_fallbacks` do not override the embedding pin or affect unpinned embeddings. No embedding warm-up is performed. This opt-in setting preserves existing embedding loading behavior when omitted.

For offline-precomputed vectors, use the same model snapshot, ONNX weights/dtype, tokenization, model-specific input prefixes, pooling, and normalization in both environments. Pinning dtype alone does not guarantee identical results across runtimes. Regenerate stored vectors and recalibrate thresholds when these settings change. See [embedding dtype selection](../MODELS.md#embedding-dtype-selection) for details.
