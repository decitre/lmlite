# `LLM.create()` parameters, models and backends

## Parameters

```python
llm = await LLM.create(
    generator_model="gpt2",           # or "distilgpt2", etc. (optional)
    embedding_model="all-MiniLM-L6-v2",  # (optional)
    classification_model=None,        # (optional)
    backend="wasm",                   # Generation/classification: "wasm" or "webgpu"
    embedding_backend=None,           # Embedding-only device: None, "wasm", or "webgpu"
    dtype=None,                       # Generation/classification dtype (e.g. "fp32")
    dtype_fallbacks=None,             # Generation/classification retry order; takes precedence over dtype
    embedding_dtype=None,             # Embedding-only pin (e.g. "q8"); no dtype fallback
    max_new_tokens=50,
    return_full_text=True,            # False returns only newly generated text
    temperature=0.7,
    top_k=50,
    do_sample=True,
    auto_detect_local=True,           # Discover hosted model files automatically
    hosted_models_url=None,           # Optional explicit HTTP models directory
    local_models_path="/drive/models",  # Portable ZIP import directory
    progress_callback=None,           # Called with a dict per loading event (see below)
)
```

## Model selection

**Models are optional**: You only need to specify the models you actually use. For example, use `classification_model` alone for classification-only workflows, or `generator_model` alone for text generation.

Pass full Hugging Face IDs (e.g. `Xenova/gpt2`, `my-org/my-model`) for explicit control, or pass short names to use the default `Xenova/` namespace.

```python
# Create instance with only classification
llm = await LLM.create(
    classification_model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Or with text generation and embeddings
llm = await LLM.create(
    generator_model="gpt2",
    embedding_model="all-MiniLM-L6-v2"
)
```

## Backend and dtype selection

`backend="webgpu"` enables GPU inference for generation and classification. Use `embedding_backend="webgpu"` independently for embeddings, when the model, dtype, runtime, and browser support it. Leaving `embedding_backend=None` retains Transformers.js's default device selection.

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

# WASM generation with WebGPU embeddings
llm = await LLM.create(
    generator_model="gpt2",
    embedding_model="all-MiniLM-L6-v2",
    backend="wasm",
    embedding_backend="webgpu"
)
```

When a generator or task model fails to load, the `RuntimeError` keeps the runtime's own message and adds the stage (`load`, or `warm up` for the one-token test that follows a generator load), the backend, the dtypes tried and the Transformers.js and onnxruntime-web versions. A message that says the runtime ran out of memory (for example `RangeError: Out of memory` on WASM) means the model is too large for the browser or device, not that it is unsupported: try a smaller model or dtype, close other tabs or reload. Embedding and classification models keep their shorter messages.
