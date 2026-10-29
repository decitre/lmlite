[![Python package][ci-badge]][ci-url]
[![codecov-python][codecov-python-badge]][codecov-url]
[![codecov-js][codecov-js-badge]][codecov-url]
[![Pixi Badge][pixi-badge]][pixi]
[![Transformers.js][transformers-badge]][transformers-url]

[![xeus-badge][xeus-badge]][xeus-url]
[![pyodide-badge][pyodide-badge]][pyodide-url]


# lmLite

Package version: `0.8.1`

**A Python bridge to Transformers.js for JupyterLite** — run language models and embedding workflows entirely in your browser.


## Quick Start

**Try online** (no installation required):
- [Launch with xeus-python][xeus-url] (recommended)
- [Launch with Pyodide][pyodide-url]

**Basic usage:**

```python
from lmlite import LLM

# Instruction-tuned chat with SmolLM2
llm = await LLM.create(
    generator_model="HuggingFaceTB/SmolLM2-360M-Instruct",
    backend="wasm",
    dtype="q4",
)
response = await llm.chat([
    {"role": "user", "content": "Explain embeddings in simple terms."}
])
print(response)

# Text completion with GPT-2
llm = await LLM.create(generator_model="gpt2")
text = await llm.generate("Python is a great language because")
print(text)

# Embeddings and similarity search
llm = await LLM.create(embedding_model="all-MiniLM-L6-v2")
embedding = await llm.embed("Hello world")

docs = ["JupyterLite runs in the browser.", "Python is widely used for ML."]
results = await llm.similarity_search("Where does JupyterLite run?", docs)
print(results)
```

## Features

- **Browser-native**: Run LLMs entirely in the browser, no server required
- **Instruction-tuned models**: Chat and instruction-following with browser-compatible models such as SmolLM2-360M-Instruct.
- **Composable workflows**: Build simple RAG pipelines and conversational interfaces with memory using Python.
- **Privacy-first**: All computation happens locally in your browser
- **Zero setup**: No installation or API keys for hosted demos
- **Multiple backends**: WebAssembly (WASM) or WebGPU
- **Multiple models**: GPT-2, DistilGPT-2, TinyLlama, and more via Transformers.js
- **Embeddings & classification**: Sentence vectors and text classification
- **Task pipelines**: Zero-shot classification, speech recognition and translation via `LLM.pipeline(task, model=...)`
- **Model caching**: Browser-cached downloads, portable ZIP export/import, and cache inspection before loading (`LLM.model_info()`, `cached_models()`, per-model `clear_cache()`) with Python progress callbacks
- **Pythonic API**: Clean, async Python interface

## What is lmLite?

lmLite is a lightweight Python-to-JavaScript bridge for **local LLM inference in the browser**, designed for JupyterLite notebooks. It provides:

- Async Python API for text generation, embeddings, classification, similarity search, and additional Transformers.js task pipelines
- Model loading, caching, and portable ZIP workflows
- Independent backend/dtype controls for embeddings and generation

**Inference is powered by [Transformers.js](https://github.com/huggingface/transformers.js) and ONNX Runtime**. lmLite is not a new model engine or native Python inference backend. Its similarity search is an in-memory cosine ranking helper, not a vector database.

**Best suited for:** Teaching, interactive experiments, and browser-local prototypes where avoiding remote inference services matters.

> **Note:** Installing the Python package on desktop does **not** enable desktop inference. `LLM.create()` requires a supported browser-based Python environment (xeus-python or the JupyterLite Pyodide kernel) with JavaScript interoperability. See [docs/RUNTIME.md](https://github.com/decitre/lmlite/blob/main/docs/RUNTIME.md) for details.


## Feedback and Contributions

lmLite is an experimental project exploring Python access to browser-native machine learning through Transformers.js.

Feedback is particularly welcome on:

- The usefulness and design of the asynchronous Python API.
- Additional Transformers.js tasks and capabilities to expose.
- Model compatibility, performance, and browser runtime limitations.
- Potential applications beyond JupyterLite.

Please share suggestions, bug reports, and use cases through [GitHub Issues](https://github.com/decitre/lmlite/issues).

### AI-Assisted Development

Parts of this project were developed with the assistance of generative AI tools. The maintainer is responsible for the architecture, review, testing, and published code.

AI-assisted contributions are welcome, provided contributors understand, review, and test their changes and take responsibility for their submissions.

## Documentation

- **[Getting Started](https://github.com/decitre/lmlite/blob/main/docs/DEVELOPMENT.md)** - Installation, development setup, and testing
- **[Usage Guide](https://github.com/decitre/lmlite/blob/main/docs/USAGE.md)** - Detailed API examples and configuration
- **[Model Guide](https://github.com/decitre/lmlite/blob/main/docs/MODELS.md)** - Recommended models and dtype selection
- **[Runtime & Integration](https://github.com/decitre/lmlite/blob/main/docs/RUNTIME.md)** - Platform support and integration boundaries
- **[Glossary](https://github.com/decitre/lmlite/blob/main/docs/GLOSSARY.md)** - Key concepts and terminology

## Interactive Notebooks

Try lmLite directly in your browser, without installing anything.

- **[Basic demo](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch01_demo.ipynb)** — Text generation, embeddings, classification, and similarity search.
- **[Task pipelines](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch08_tasks.ipynb)** — Zero-shot classification, translation and speech recognition with `LLM.pipeline()`.
- **[Model info and cache manager](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch99_model-info.ipynb)** — Download size and cache status before loading, loading progress from Python, and a widget to list, load and delete cached models.
- **[LLM orchestration](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch09_orchestration.ipynb)** — A simple retrieval-augmented generation (RAG) workflow and interactive multi-turn chat with conversation memory and streamed, stoppable answers, using SmolLM2-360M-Instruct (WASM/q4) and MiniLM embeddings (WASM/q8).
- **[Streaming chat](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch10_streaming.ipynb)** — Print a chat answer as it is generated, stop early with `break`, and cancel a running generation with `cancel()`, using SmolLM2-135M-Instruct (WASM).
- **[Token probabilities](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch11_token-scores.ipynb)** — Stream each generated token with its probability, plus the runner-up token at every step, using SmolLM2-135M-Instruct (WASM).

These examples run locally in the browser using Transformers.js.

## How It Works

```
┌─────────────────────────────────────┐
│   Python (JupyterLite)              │
│   ├─ xeus-python (recommended)      │
│   └─ pyodide (alternative)          │
│                                     │
│   from lmlite import LLM            │
│   llm = await LLM.create()          │
│   text = await llm.generate(...)    │
└─────────────┬───────────────────────┘
              │ Bridge (xeus-python: pjs / Pyodide: pyodide.ffi)
┌─────────────▼───────────────────────┐
│   JavaScript (Browser)              │
│   Transformers.js + ONNX Runtime    │
│   ├─ WebAssembly (WASM)             │
│   └─ WebGPU (optional)              │
└─────────────────────────────────────┘
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

**Dependencies:** [Transformers.js](https://github.com/huggingface/transformers.js) v4 (Apache 2.0)

**Note:** Individual model licenses may vary. Check model cards on HuggingFace before use.

## Citation

```bibtex
@software{lmlite2026,
  author = {Decitre, Emmanuel},
  title = {lmLite: Browser-native LLM orchestration in JupyterLite},
  year = {2026},
  url = {https://github.com/decitre/lmlite}
}
```

## Acknowledgements

lmLite builds on several open-source projects:

- [Transformers.js](https://github.com/huggingface/transformers.js) — browser-native model inference.
- [JupyterLite](https://jupyterlite.readthedocs.io/) — the browser-based Jupyter environment.
- [xeus-python](https://github.com/jupyter-xeus/xeus-python) and [Pyodide](https://pyodide.org/) — Python execution in the browser.

Thanks to the developers and maintainers of these projects and the broader Jupyter and Hugging Face communities.

---

[ci-badge]: https://github.com/decitre/lmlite/actions/workflows/test.yml/badge.svg
[codecov-python-badge]: https://img.shields.io/codecov/c/github/decitre/lmlite/main?flag=python&label=py%20cov&logo=codecov&token=CODECOV_TOKEN
[codecov-js-badge]: https://img.shields.io/codecov/c/github/decitre/lmlite/main?flag=javascript&label=js%20cov&logo=codecov&token=CODECOV_TOKEN
[ci-url]: https://github.com/decitre/lmlite/actions/workflows/test.yml
[codecov-url]: https://codecov.io/gh/decitre/lmlite
[xeus-url]: https://decitre.github.io/lmlite/xeus/lab/?path=ch01_demo.ipynb
[pyodide-url]: https://decitre.github.io/lmlite/pyodide/lab/?path=ch01_demo.ipynb
[pyodide-badge]: https://jupyterlite.rtfd.io/en/latest/_static/badge.svg
[xeus-badge]: https://img.shields.io/badge/try-lite%20now-f7dc1e.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0Ni43IDQ2LjciPgogIDxwYXRoCiAgICBmaWxsPSIjNDBCM0MzIgogICAgZD0iTTUgMGMyLjUuOCA0LjUgMiA0LjYgMi4xIDEuOSAxLjMgMy42IDMuMSA0LjggNS4xLjQuNiA1LjIgNy44IDguNiAxMi45LS44IDEuMi0xLjUgMi4zLTIuMSAzLjEgMC0uMS0uMS0uMS0uMS0uMkMxNy42IDE3LjkgMTEuNCA4LjYgMTEgNy45Yy0uOS0xLjMtMi0yLjUtMy41LTMuNCAwIDAtMS41LTEtMy41LTEuNkMzLjIgMi42IDEuOCAyLjQgMCAyLjJ2NDIuNGMxLjgtLjIgMy4yLS40IDQtLjcgMi0uNiAzLjUtMS42IDMuNS0xLjYgMS41LTEgMi43LTIuMiAzLjUtMy40LjQtLjcgNi42LTkuOSAxMC4xLTE1LjJDMjQgMTkuMSAzMS40IDguMSAzMS45IDcuM2MxLjItMiAyLjktMy44IDQuOC01LjEgMC0uMSAyLjEtMS4zIDQuNi0yLjFMNDEuOCAwSDV6CiAgICAgICBNNDYuNyAyLjJjLTEuOC4yLTMuMi40LTQgLjctMi4xLjYtMy42IDEuNi0zLjYgMS42LTEuNS45LTIuNiAyLjEtMy41IDMuNC0uNC43LTYuNiA5LjktMTAuMSAxNS4yLTIuOSA0LjQtMTAuMyAxNS41LTEwLjggMTYuMi0xLjIgMi0yLjkgMy44LTQuOCA1LjEgMCAuMS0yLjEgMS4zLTQuNiAyLjFsLS41LjFoMzYuOGwtLjQtLjFjLTIuNS0uOC00LjUtMi00LjYtMi4xLTEuOS0xLjMtMy42LTMuMS00LjgtNS4xLS40LS42LTUuMi03LjgtOC42LTEyLjkuOC0xLjIgMS41LTIuMyAyLjEtMy4xIDAgLjEuMS4xLjEuMiAzLjUgNS4zIDkuNyAxNC41IDEwLjEgMTUuMi45IDEuMyAyIDIuNSAzLjUgMy40IDAgMCAxLjUgMSAzLjUgMS42LjguMyAyLjIuNSA0IC43VjIuMnoiCiAgLz4KPC9zdmc+Cgo=
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
[pixi]: https://pixi.sh
[transformers-badge]: https://img.shields.io/badge/🤗-Transformers.js-172033?style=flat&labelColor=172033&color=172033
[transformers-url]: https://huggingface.co/docs/transformers.js/en/index
