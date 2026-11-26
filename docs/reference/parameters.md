# `LLM`, `Embedder` and `Classifier`: parameters, models and backends

## `LLM.create()`

`LLM` is for large language models: generation, chat, streaming and token probabilities.

```python
llm = await LLM.create(
    generator_model="gpt2",           # or "distilgpt2", etc.
    backend="wasm",                   # "wasm" or "webgpu"
    dtype=None,                       # e.g. "fp32"
    dtype_fallbacks=None,             # retry order; takes precedence over dtype
    max_new_tokens=50,
    return_full_text=True,            # False returns only newly generated text
    temperature=0.7,
    top_k=50,
    do_sample=True,
    auto_detect_local=True,           # Discover hosted model files automatically
    hosted_models_url=None,           # Optional explicit HTTP models directory
    local_models_path="/drive/models",  # Portable ZIP import directory
    progress_callback=None,           # Called with a dict per loading event (see below)
    quiet=False,                      # No progress bar and no loading messages, for an app that has its own
)
```

`LLM.create(embedding_model=..., embedding_backend=..., embedding_dtype=..., classification_model=...)` still loads those models, with a `DeprecationWarning`: use `Embedder.create()` and `Classifier.create()`. The options will be removed in 0.15.

## `Embedder.create()` and `Classifier.create()`

```python
embedder = await Embedder.create(
    "all-MiniLM-L6-v2",               # the model
    backend=None,                     # None (Transformers.js chooses), "wasm", "webgpu" or "native"
    dtype=None,                       # pin the weights (e.g. "q8"); no dtype fallback
    auto_detect_local=True,
    hosted_models_url=None,
    local_models_path="/drive/models",
    progress_callback=None,
)

classifier = await Classifier.create(
    "distilbert-base-uncased-finetuned-sst-2-english",
    backend="wasm",                   # "wasm", "webgpu" or "native"
    dtype=None,                       # e.g. "fp32"
    dtype_fallbacks=None,             # retry order; takes precedence over dtype
    auto_detect_local=True,
    hosted_models_url=None,
    local_models_path="/drive/models",
    progress_callback=None,
)
```

Each class loads one model, so load only what you use: an `Embedder` next to an `LLM` is two `create()` calls, and each is disposed on its own.

## Model selection

Pass full Hugging Face IDs (e.g. `Xenova/gpt2`, `my-org/my-model`) for explicit control, or pass short names to use the default `Xenova/` namespace.

## Backend and dtype selection

`backend="webgpu"` enables GPU inference for generation, classification and embeddings, when the model, dtype, runtime and browser support it. `Embedder.create()` leaves the device to Transformers.js when `backend` is `None`.

Model support depends on the current Transformers.js runtime and the ONNX/dtype files available for that model. If a model fails to load, check the Transformers.js WebGPU and dtype guides for compatible browser models.

The fastest generation dtype depends on the backend: `q4` on WebGPU, `fp16` (or `q8` for files over 2 GB) on WASM, where `q4` is 4 to 8× slower than `q8`. The default fallback order starts with `q8`, the slowest WebGPU choice, so pass `dtype="q4"` with `backend="webgpu"`. Measurements are in [MODELS.md](../MODELS.md#generation-dtype-selection).

If you hit ONNX runtime errors such as `Missing required scale ... DequantizeLinear` on WASM, try `backend="webgpu", dtype="fp32"` or set `dtype_fallbacks=["fp32"]` to avoid quantized variants.

```python
# WebGPU for generation with fp32
llm = await LLM.create(
    generator_model="gpt2",
    backend="webgpu",
    dtype="fp32"
)

# WASM generation, and WebGPU embeddings in a second model
llm = await LLM.create(generator_model="gpt2", backend="wasm")
embedder = await Embedder.create("all-MiniLM-L6-v2", backend="webgpu")
```

When a generator or task model fails to load, the `RuntimeError` keeps the runtime's own message and adds the stage (`load`, or `warm up` for the one-token test that follows a generator load), the backend, the dtypes tried and the Transformers.js and onnxruntime-web versions. A message that says the runtime ran out of memory (for example `RangeError: Out of memory` on WASM) means the model is too large for the browser or device, not that it is unsupported: try a smaller model or dtype, close other tabs or reload. A message that says a fetch failed (`Load failed` in Safari, `Failed to fetch` in Chrome, `NetworkError` in Firefox) means a file could not be downloaded or loaded, and says nothing about whether the model is supported: try again, look at the browser's network tab, and if it fails with `webgpu` only, try `wasm`. The same applies at the warm-up stage. Embedding and classification models keep their shorter messages.
