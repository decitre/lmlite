# Usage Guide

This guide covers detailed API usage, configuration options, and common workflows for lmLite.

## Basic Usage

### Creating an LLM Instance

`LLM.create(...)` displays `tqdm` download progress in the notebook cell, with status messages during model preparation, initialization, and warm-up. Browser kernels explicitly use notebook widgets when `ipywidgets` is available, without starting a background monitor thread. Missing widget support produces a warning and text progress instead. The JupyterLite frontend must also support widgets to render the bar. The cell stays busy until initialization and warm-up finish.

Phases without measurable byte progress show a status rather than a zero-byte counter; successful initialization ends at 100%. Cached models may initialize without any download progress. Model weights and inference sessions still require sufficient browser memory even when files are fetched individually.

```python
from lmlite import LLM

# Create LLM instance (downloads model on first run)
# You can pass either a full model ID ("Xenova/gpt2") or a short name ("gpt2").
# Short names default to the Xenova namespace.
llm = await LLM.create(
    generator_model="gpt2",
    embedding_model="all-MiniLM-L6-v2",
)
```

### Text Generation

```python
# Generate text
text = await llm.generate("Python is a great language because")
print(text)
```

### Chat with Message History

```python
# Chat with message history
messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What can I use it for?"}
]
response = await llm.chat(messages)
print(response)
```

### Embeddings

```python
# Generate embeddings
embedding = await llm.embed("Hello world")
print(embedding[:5])  # First 5 dimensions
```

### Text Classification

```python
# Classify text (requires classification_model)
llm_cls = await LLM.create(classification_model="distilbert-base-uncased-finetuned-sst-2-english")
labels = await llm_cls.classify("I love transformers!", top_k=2)
print(labels)
```

### Similarity Search

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

### Task Pipelines

`LLM.pipeline(task, model=...)` loads one additional Transformers.js task and returns an awaitable callable. Supported tasks are listed in `lmlite.SUPPORTED_TASKS`: `"zero-shot-classification"`, `"automatic-speech-recognition"` and `"translation"`. Other task names raise a `ValueError` that lists the supported tasks; generation, embeddings and text classification keep their dedicated methods.

```python
classifier = await LLM.pipeline("zero-shot-classification", model="mobilebert-uncased-mnli")
result = await classifier("This notebook runs in a browser", candidate_labels=["technology", "sports"])
# {'sequence': 'This notebook runs in a browser', 'labels': ['technology', 'sports'], 'scores': [0.93, 0.07]}

translator = await LLM.pipeline("translation", model="opus-mt-en-fr")
await translator("Hello, how are you?", max_new_tokens=64)
# {'translation_text': 'Bonjour, comment allez-vous ?'}
await translator(["Hello!", "Goodbye!"])
# [{'translation_text': 'Bonjour !'}, {'translation_text': 'Au revoir !'}]

transcriber = await LLM.pipeline("automatic-speech-recognition", model="whisper-tiny.en")
await transcriber("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav")
# {'text': ' And so my fellow Americans ask not what your country can do for you...'}

await classifier.dispose()
```

Loading options are the same as for `LLM.create()`: `backend`, `dtype`, `dtype_fallbacks`, `hosted_models_url`, `auto_detect_local` and `local_models_path`. Model names, hosted files, browser caching, progress display and `export_models()` work the same way too. Load failures name the task, model and backend.

Keyword arguments to a pipeline call are forwarded as Transformers.js options for that task, and must be JSON-serializable. Results are plain Python dicts and lists:

| Task | Input | Task-specific arguments | Output |
|------|-------|-------------------------|--------|
| `zero-shot-classification` | string or list of strings | `candidate_labels` (required), `hypothesis_template`, `multi_label` | `{"sequence", "labels", "scores"}` sorted by score, or a list for list input |
| `translation` | string or list of strings | generation options such as `max_new_tokens`; `src_lang`/`tgt_lang` for multilingual models (e.g. NLLB, M2M100) | `{"translation_text"}`, or a list for list input |
| `automatic-speech-recognition` | audio (see below) | `return_timestamps`, `chunk_length_s`, `stride_length_s`, `language`/`task` for multilingual Whisper | `{"text"}`, plus `"chunks"` with timestamps |

Speech recognition accepts an HTTP(S) URL, a path to a PCM WAV file (e.g. a file uploaded to the JupyterLite file browser), WAV bytes, or a sequence of mono float samples at 16 kHz. WAV files are decoded in Python, mixed down to mono and linearly resampled to 16 kHz. Notebook kernels run in a Web Worker without `AudioContext`, so URLs are fetched and decoded the same way and must point to WAV files. Compressed formats (MP3, OGG, M4A) are not decoded in the kernel: convert them to WAV first. URL fetches must be allowed by CORS. Whisper processes 30-second windows, so pass `chunk_length_s=30` for longer recordings.

Expect a pipeline in a browser kernel to be slower than native inference, especially sequence-to-sequence tasks (translation and speech recognition), which decode token by token. WebGPU can help when the browser, model and dtype support it. Each pipeline holds its own model in browser memory; call `dispose()` before loading the next one.

## Configuration Options

### All Available Parameters

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

### Model Selection

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

### Backend and Dtype Selection

`backend="webgpu"` enables GPU inference for generation and classification. Use `embedding_backend="webgpu"` independently for embeddings, when the model, dtype, runtime, and browser support it. Leaving `embedding_backend=None` retains Transformers.js's default device selection.

Model support depends on the current Transformers.js runtime and the ONNX/dtype files available for that model. If a model fails to load, check the Transformers.js WebGPU and dtype guides for compatible browser models.

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

### Hosted Models Discovery

Hosted models are discovered from the browser kernel's JupyterLite URL. If your kernel uses a URL that cannot identify the site root (for example, a blob URL), pass `hosted_models_url="https://your-host/path/to/site/files/models/"` to `LLM.create(...)`.

Hosted trees represent a single snapshot: explicit revisions other than `@main` use the existing model-resolution path rather than the hosted snapshot. `auto_detect_local=False` disables automatic model-source discovery. Clear the model cache after changing files in a hosted snapshot if you want to force fresh downloads.

```python
# Explicit hosted models URL
llm = await LLM.create(
    generator_model="gpt2",
    hosted_models_url="https://my-site.com/files/models/",
    auto_detect_local=True
)
```

## Advanced Features

### Generation Without Prompt Echo

For generation without prompt echo, pass `return_full_text=False` to `LLM.create()`. This forwards the option to Transformers.js; `generate(prompt)` then returns only the continuation, with no Python-side prompt removal needed.

The default, `True`, preserves full-text output. Custom chat wrappers using `False` should consume the returned answer directly rather than stripping a prompt prefix. This option does not apply a model-specific chat template or prevent repetition.

```python
llm = await LLM.create(
    generator_model="gpt2",
    return_full_text=False
)

# Returns only the generated continuation
continuation = await llm.generate("Python is")
```

### Model-Aware Chat

`chat(messages)` uses the chat template supplied by the loaded generator's tokenizer through Transformers.js, rather than guessing a format from the model name. It returns only the assistant response, regardless of `return_full_text`. Messages require string `role` and `content` fields.

```python
answer = await llm.chat([
    {"role": "system", "content": "Answer directly and concisely."},
    {"role": "user", "content": "What are text embeddings?"},
])
```

Models without a tokenizer chat template retain the generic `User:` / `Assistant:` format; this does not make completion models instruction-tuned. Invalid messages, unsupported template roles, and template failures raise errors without retrying with a different format. Chat does not manage memory or retrieve documents automatically; the lcLite notebook demonstrates those separately.

### Streaming and Cancellation

`chat_stream(messages)` and `generate_stream(prompt)` are async iterators that yield text chunks as Transformers.js generates them, so the notebook shows output while the model runs.

```python
async for chunk in llm.chat_stream([
    {"role": "user", "content": "Explain embeddings simply."},
]):
    print(chunk, end="", flush=True)
```

- **Chunks** are nonempty strings from Transformers.js's `TextStreamer` (special tokens skipped). It emits whole words, so a chunk is usually a word with its whitespace, not a single token. `"".join(chunks).strip()` is the `chat()` answer for the same tokens. `generate_stream()` never includes the prompt, whatever `return_full_text` is.
- **Laziness**: generation starts when you first iterate, not when you call `chat_stream()`.
- **Cancellation**: `await llm.cancel()` stops the running `generate()`, `chat()` or stream after its current token and returns `False` if nothing was running. A cancelled stream ends early without error; `generate()`/`chat()` return the partial text. The instance remains usable.
- **Early exit**: leaving the loop (`break`, an exception, or a kernel interrupt) cancels the generation. Kernel interrupts reach Python only between chunks and only where the kernel supports interrupts; if one never arrives, restart the kernel.
- **Concurrency**: one generation runs per instance at a time. Starting another while one is running raises an error (for streams, on the first chunk); a generation started right after a cancellation waits for it to stop instead.
- **Errors** during generation are raised from the iterator.
- **Backends**: streaming works with any generator model and backend. Cancellation is checked between tokens, so a slow first token (prompt processing on a long input) is not interrupted.

### Sentence Embeddings and Dtype Pinning

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

For offline-precomputed vectors, use the same model snapshot, ONNX weights/dtype, tokenization, model-specific input prefixes, pooling, and normalization in both environments. Pinning dtype alone does not guarantee identical results across runtimes. Regenerate stored vectors and recalibrate thresholds when these settings change. See [embedding dtype selection](MODELS.md#embedding-dtype-selection) for details.

### Model Info, Cache Status and Loading Progress

Inspect a model before loading it:

```python
from lmlite import LLM

info = await LLM.model_info("HuggingFaceTB/SmolLM2-360M-Instruct", backend="wasm", dtype="q4")
print(info.cache_status, info.download_bytes)
for f in info.files:
    print(f["name"], f["size"], f["cached"])

llm = await LLM.create(
    generator_model="HuggingFaceTB/SmolLM2-360M-Instruct",
    backend="wasm", dtype="q4",
    progress_callback=lambda event: print(event),
)
```

`LLM.model_info(model, task=None, backend="wasm", dtype=None)` returns a `ModelInfo` built from the Transformers.js `ModelRegistry`. Pass the `task` you will load (`"text-generation"`, `"feature-extraction"` for embeddings, `"text-classification"`, or a `SUPPORTED_TASKS` entry). Without a task, the model's Hub pipeline tag is used. That needs the network, and a model without a tag reports `task=None` with `error` set. To get only the tag, use `await LLM.model_task(model)`: a single Hub request (5-second timeout) that returns e.g. `"text-generation"`, or `None` offline or for untagged models. `ModelInfo` contains:

- `model_id`, `revision`, `task`, `backend` and `dtype`. Without a dtype, the first of `q8`, `int8` or `fp32` published for the model is reported, matching what `LLM.create()` tries; `feature-extraction` keeps the Transformers.js default and reports `None`.
- `files`: the files the pipeline will fetch, each with `name`, `size` in bytes (or `None`) and `cached`.
- `cache_status`: `"cached"`, `"partial"`, `"not_cached"` or `"unknown"`.
- `download_bytes`: the sum of the file sizes, or `None` if any size is unknown. It is the download size, not an estimate of runtime memory. Runtime memory depends on the backend, the dtype, the browser and the length of what you generate, and lmLite does not estimate it.
- `error`: why the metadata is incomplete, or `None`.

Limitations:

- `cache_status` only reflects the Transformers.js Cache API (`transformers-cache`). It does not inspect the browser HTTP cache, IndexedDB, hosted JupyterLite model files, or ZIPs in `local_models_path`, even though `LLM.create()` can load from any of these.
- When the Cache API is unavailable, for example outside a secure context, `cache_status` is `"unknown"`.
- Offline, sizes of files that are not cached are `None`. If `dtype` is omitted, the dtype lookup needs the network, so you get `cache_status="unknown"` with `error` set. Pass `dtype` explicitly to inspect cached models offline.
- File sizes are read from the cache, or from a 1-byte range request to the Hub. Nothing is downloaded.
- The check works the same in Xeus-Python and Pyodide; it runs entirely in the browser JS runtime.

`progress_callback` receives dicts with `model`, `status`, `file`, `loaded` and `total`:

- `status` is `"initializing"`, `"loading"`, `"progress"` (byte counts for `file`), `"warming up"`, `"ready"` (all models loaded) or `"error"`.
- An `"error"` event also has an `error` message. The exception is still raised.
- The tqdm progress bar keeps working alongside the callback.
- Reading a local ZIP archive is shown only in the progress bar.

### Portable ZIP Import and Export

ZIP archives are a separate portability feature, not an input to the JupyterLite hosting build. Export all models configured on an instance from the browser cache:

```python
await llm.export_models()
```

Exports are saved under `/drive/models/` as `model-<namespace>-<model>@<revision>.zip` and can be downloaded from JupyterLite. Without an explicit revision, naming uses the latest Hub commit when available, otherwise `main`; full commit SHAs are shortened to six characters.

Existing model archives can be imported through `LLM.create(...)`'s `local_models_path` option when no hosted model is selected. Its default is `"/drive/models"`. Opening an archive may fetch the entire compressed file before ZIP-reading progress begins; entries are then read and cached one file at a time.

For preparing a hosted site, prefer the individual-file download workflow above. The current Python export API produces ZIPs; it does not expose a directory-export option.

### Release Loaded Models

```python
await llm.dispose()
```

This releases loaded generation, embedding, and classification pipelines without deleting cached model files. Use it between model trials to release inference sessions and GPU resources. Await all active loading/inference/export operations first; concurrent use and disposal of the same instance are not supported.

Repeated disposal is safe. Cleanup is attempted for every loaded pipeline; failures raise an error, and failed pipelines remain available for a disposal retry. Successfully disposed pipelines reject subsequent inference calls. Create a new instance with `LLM.create(...)` to load models again from cache. Actual memory reclamation timing depends on the browser and runtime; disposal does not guarantee an immediate drop in reported memory.

### Clear Browser Model Cache

```python
from lmlite import clear_cache

result = await clear_cache()
print(result)
```

This clears all cached model files from the browser's cache. Use this if you need to force re-downloading models or if you're experiencing cache-related issues.

To see what is cached, or to remove a single model (all revisions):

```python
from lmlite import cached_models, clear_cache

for entry in await cached_models() or []:
    print(entry["model_type"], entry["task"], entry["model_id"], entry["revision"], entry["size"])

await clear_cache(model="Xenova/gpt2")  # {"model_id": ..., "deleted_files": [...]}
```

`cached_models()` only lists the Transformers.js browser Cache API (not hosted model files or `/drive/models` ZIPs). It returns `None` if the Cache API is unavailable, and `size` is `None` when a file's size is unknown. `model_type` is the architecture from the cached `config.json` (e.g. `gpt2`, `bert`). `task` is the Hub pipeline tag (e.g. `text-generation`) and needs the network, so it is `None` offline. A cache manager widget built on these is demonstrated in `notebooks/model-info.ipynb`.
