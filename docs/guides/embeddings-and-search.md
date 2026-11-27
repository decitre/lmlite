# Embeddings, classification and similarity search

`Embedder` and `Classifier` load a model each, independently of a language model and of one another. Before 0.15 they were loaded with `LLM.create(embedding_model=...)` and `LLM.create(classification_model=...)` and used through `llm.embed()`, `llm.classify()` and `llm.similarity_search()`. Those are gone: `LLM.create()` raises a `TypeError` for the two model options.

## Embeddings

```python
from lmlite import Classifier, Embedder

embedder = await Embedder.create("all-MiniLM-L6-v2")

# Generate embeddings
embedding = await embedder.embed("Hello world")
print(embedding[:5])  # First 5 dimensions
```

## Text Classification

```python
# Classify text
classifier = await Classifier.create("distilbert-base-uncased-finetuned-sst-2-english")
labels = await classifier.classify("I love transformers!", top_k=2)
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
results = await embedder.similarity_search(
    "Where does JupyterLite run?",
    docs
)

for doc, score in results:
    print(f"{score:.3f} -> {doc}")

# Search for top 2 results
top_results = await embedder.similarity_search(
    "Where does JupyterLite run?",
    docs,
    k=2
)
```

## Sentence Embeddings and Dtype Pinning

`embed(text)` returns a flat, fixed-dimension sentence vector using attention-mask-weighted mean pooling and L2 normalization. Pooling makes token-level outputs suitable for sentence comparison; normalization makes dot products equivalent to cosine similarity for nonzero vectors. `similarity_search()` already normalizes when computing cosine scores.

Pin the embedding weights with `dtype`:

```python
embedder = await Embedder.create("all-MiniLM-L6-v2", dtype="q8")
embedding = await embedder.embed("JupyterLite runs in the browser.")
```

Set `backend="wasm"` or `"webgpu"` to choose the embedding device. The default, `None`, omits the device option and lets Transformers.js choose. Unsupported device/model/dtype combinations fail initialization. With `"webgpu"`, a failure specific to WebGPU is retried once on WASM, as described in [WebGPU fallback](../reference/parameters.md#webgpu-fallback); pass `backend_fallback=False` to fail instead.

`dtype=None` (the default) leaves dtype selection to Transformers.js. An explicit value selects only that dtype: unavailable or incompatible weights fail initialization rather than falling back. No embedding warm-up is performed. This opt-in setting preserves existing embedding loading behavior when omitted.

For offline-precomputed vectors, use the same model snapshot, ONNX weights/dtype, tokenization, model-specific input prefixes, pooling, and normalization in both environments. Pinning dtype alone does not guarantee identical results across runtimes. Regenerate stored vectors and recalibrate thresholds when these settings change. See [embedding dtype selection](../MODELS.md#embedding-dtype-selection) for details.
