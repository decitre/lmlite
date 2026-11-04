<p>
  <img src="https://raw.githubusercontent.com/decitre/lmlite/main/docs/logo.svg"
       alt="lmLite — a Python bridge to Transformers.js for JupyterLite"
       width="300">
</p>

[![Tests][ci-badge]][ci-url]
[![codecov-python][codecov-python-badge]][codecov-url]
[![codecov-js][codecov-js-badge]][codecov-url]
[![Transformers.js][transformers-badge]][transformers-url]
[![Try xeus-python][xeus-badge]][xeus-url]
[![Try Pyodide][pyodide-badge]][pyodide-url]


# lmLite

Package version: `0.11.1`

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
- **Headless native backend**: `pip install 'lmlite[native]'` runs the same ONNX files with onnxruntime outside the browser for fast tuning loops, checked for parity against the browser

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
- **[Native Tuning](https://github.com/decitre/lmlite/blob/main/docs/NATIVE.md)** - Tune headless with the native backend, then ship to the browser
- **[Glossary](https://github.com/decitre/lmlite/blob/main/docs/GLOSSARY.md)** - Key concepts and terminology

## Interactive Notebooks

Try lmLite directly in your browser, without installing anything.

- **[Basic demo](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch01_demo.ipynb)** — Text generation, embeddings, classification, and similarity search.
- **[Task pipelines](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch08_tasks.ipynb)** — Zero-shot classification, translation and speech recognition with `LLM.pipeline()`.
- **[Model info and cache manager](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch99_model-info.ipynb)** — Download size and cache status before loading, loading progress from Python, and a widget to list, load and delete cached models.
- **[LLM orchestration](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch09_orchestration.ipynb)** — A simple retrieval-augmented generation (RAG) workflow and interactive multi-turn chat with conversation memory and streamed, stoppable answers, using SmolLM2-360M-Instruct (WASM/q4) and MiniLM embeddings (WASM/q8).
- **[Streaming chat](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch10_streaming.ipynb)** — Print a chat answer as it is generated, stop early with `break`, and cancel a running generation with `cancel()`, using SmolLM2-135M-Instruct (WASM).
- **[Token probabilities](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch11_token-scores.ipynb)** — Stream each generated token with its probability, plus the runner-up token at every step, using SmolLM2-135M-Instruct (WASM).
- **[Tokenizers](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch12_tokenizers.ipynb)** — Load only the tokenizers of GPT-2, multilingual BERT and POTION (no weights), compare how they split text, and count tokens synchronously with `LLM.tokenizer()`.
- **[Proving GPU inference](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch13_webgpu.ipynb)** — Show that WebGPU inference really runs on the GPU: the hardware adapter, model tensors held in GPU memory, and WebGPU vs WASM timing for the same greedy generation, using SmolLM2-135M-Instruct (q4).

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
[pyodide-badge]: https://img.shields.io/badge/try-Pyodide-f7dc1e.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI2OCIgaGVpZ2h0PSI2OCIgdmlld0JveD0iMCAwIDE3Ljk5MiAxNy45OTIiPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKC40NiAuNDQ4KSI+PGNpcmNsZSBjeD0iODMuNzI3IiBjeT0iODUuNjMzIiByPSI2LjU5OCIgc3R5bGU9ImZpbGw6I2Y3ZGMxZTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6LjI2NDU4MzMyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzUuMTkxIC03OS4wMjUpIi8+PHJlY3Qgd2lkdGg9IjUuNjUzIiBoZWlnaHQ9IjEuMDEiIHg9IjgwLjgzMSIgeT0iOTIuMzQ0IiByeD0iLjQxMyIgcnk9Ii40MTMiIHN0eWxlPSJmaWxsOiNmOWY5Zjk7c3Ryb2tlOm5vbmU7c3Ryb2tlLXdpZHRoOjEuOTk5OTk5ODgiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03NS4xOTEgLTc5LjAyNSkiLz48cmVjdCB3aWR0aD0iNS42NDEiIGhlaWdodD0iMS4wMzEiIHg9IjgwLjgzNyIgeT0iOTMuNjcxIiByeD0iLjQyMSIgcnk9Ii40MjEiIHN0eWxlPSJmaWxsOiNmOWY5Zjk7c3Ryb2tlOm5vbmU7c3Ryb2tlLXdpZHRoOjEuOTk5OTk5NzYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03NS4xOTEgLTc5LjAyNSkiLz48cmVjdCB3aWR0aD0iMy4yNCIgaGVpZ2h0PSIxLjAyIiB4PSI4Mi4wMzgiIHk9Ijk1LjAxOSIgcng9Ii40MTciIHJ5PSIuNDE3IiBzdHlsZT0iZmlsbDojZjlmOWY5O3N0cm9rZTpub25lO3N0cm9rZS13aWR0aDoxLjk5OTk5OTg4IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzUuMTkxIC03OS4wMjUpIi8+PGcgc3R5bGU9ImZpbGw6IzFhMWExYSI+PGcgc3R5bGU9ImZpbGw6IzFhMWExYTtzdHJva2Utd2lkdGg6LjI2NDU4MzM1Ij48cGF0aCBkPSJNODcuMDY1IDkwLjI4OXEtLjk5Ni0uMDQxLTEuNDc1LTEuMjE2LS4zNTYtLjg4NC0uMzU2LTIuMTY2IDAtLjA2Ni4wMDgtLjE3LjA1OC0uODcyLjMyMy0xLjYuNDMtMS4xNzMgMS4yMTUtMS4xNzMuMDQyIDAgLjA4Ny4wMDQuNzE1LjA0NS43MTUuODMgMCAuODU3LS43MzYgMS45MDItLjQ2Ny42NjYtLjk4IDEuMDguMTQxLjc4NS4zMSAxLjE3OC4zMTUuNzI3LjkzNS43MjcuMDQxIDAgLjA4Ny0uMDA4LjEzNi0uMDI1LjM4LS4xOTguMjM2LS4xNjIuMzY4LS4xNjIuMDU4IDAgLjExMi4wMzguMTIuMDgyLjEyLjIyMy0uMDA5LjIxNS0uMzA2LjQ0Mi0uMzcyLjI2OS0uODA3LjI2OXptLS4yOC01LjcwMXEtLjI0NS4wMDQtLjQzLjIxLS40NzIuNTM0LS41MTcgMi4yMy4yNDgtLjI1My40OTYtLjU5Ni42ODYtLjkzOS42ODYtMS41ODQgMC0uMDMzLS4wMTctLjA3OC0uMDY2LS4xODItLjIxOS0uMTgyeiIgYXJpYS1sYWJlbD0ibCIgc3R5bGU9ImZpbGw6IzFhMWExYTtzdHJva2Utd2lkdGg6LjI2NDU4MzM1IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzYuNDA5IC03OS40NzMpIi8+PC9nPjxwYXRoIGQ9Ik04NC45OTggODMuNjYzYS42OTQuNjk0IDAgMCAxLS42OTQuNjk1LjY5NC42OTQgMCAwIDEtLjY5NC0uNjk1LjY5NC42OTQgMCAwIDEgLjY5NC0uNjk0LjY5NC42OTQgMCAwIDEgLjY5NC42OTR6IiBzdHlsZT0iZmlsbDojMWExYTFhO3N0cm9rZTpub25lO3N0cm9rZS13aWR0aDouMjY0NTgzMzIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03Ni40MDkgLTc5LjQ3MykiLz48ZyBzdHlsZT0iZmlsbDojMWExYTFhO3N0cm9rZS13aWR0aDouMjY0NTgzMzUiPjxwYXRoIGQ9Ik04NC4xNTUgODQuNjY3cS4yNjkuMDA0LjI3Ny4zNWwtLjQ1OSAzLjI4M3ExLjAzNC0uNzkgMS4xNjItLjg5My4xNTMtLjEzMi4zMTQtLjEzNi4wNTggMCAuMTEyLjAzNy4xMi4wODMuMTIuMjExLS4wMDUuMTk5LS4yMjQuMzg1LS42MDcuNTEyLTEuNjA4IDEuMjZsLS4xOSAxLjM2OXEtLjAxMi4wOTUtLjAzNy4xODYtLjE5LjcyNy0uOTU1LjcyNy0uMzggMC0uNjctLjI4NS0uMjg1LS4yODUtLjI4NS0uNjQgMC0uMzczLjE3OC0uNjEyLjA5OS0uMTI0LjM4OC0uMzRsLjk5Mi0uNzQ0cS41MjUtMy40NjguNTM4LTMuNjQ2IDAtLjAyNS4wMDQtLjA0NS4wNDUtLjQ2Ny4zNDMtLjQ2N3ptLTEuMDg3IDUuNTM1LjA3LS41MDQtLjQ1OS4zNTFxLS4yOTguMjIzLS4zMDIuNDIyIDAgLjA3LjAyNS4xMzIuMDcuMTgyLjI2NS4xODIuMDc4IDAgLjE0LS4wMjUuMTk5LS4wODcuMjYtLjU1OHptLjg5Ny02LjMyNXEwLS4wNzQuMDM3LS4xNC4wODctLjE2Mi4yNjktLjE2Mi4wNzQgMCAuMTQ0LjAzNy4xNTguMDg3LjE1OC4yNjUgMCAuMDc0LS4wMzguMTQ1LS4wODYuMTU3LS4yNjQuMTU3LS4wNzUgMC0uMTQ1LS4wMzgtLjE2MS0uMDgyLS4xNjEtLjI2NHoiIGFyaWEtbGFiZWw9ImoiIHN0eWxlPSJmaWxsOiMxYTFhMWE7c3Ryb2tlLXdpZHRoOi4yNjQ1ODMzNSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTc2LjQwOSAtNzkuNDczKSIvPjwvZz48L2c+PC9nPjwvc3ZnPiA=
[xeus-badge]: https://img.shields.io/badge/try-xeus--python-f7dc1e.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0Ni43IDQ2LjciPgogIDxwYXRoCiAgICBmaWxsPSIjNDBCM0MzIgogICAgZD0iTTUgMGMyLjUuOCA0LjUgMiA0LjYgMi4xIDEuOSAxLjMgMy42IDMuMSA0LjggNS4xLjQuNiA1LjIgNy44IDguNiAxMi45LS44IDEuMi0xLjUgMi4zLTIuMSAzLjEgMC0uMS0uMS0uMS0uMS0uMkMxNy42IDE3LjkgMTEuNCA4LjYgMTEgNy45Yy0uOS0xLjMtMi0yLjUtMy41LTMuNCAwIDAtMS41LTEtMy41LTEuNkMzLjIgMi42IDEuOCAyLjQgMCAyLjJ2NDIuNGMxLjgtLjIgMy4yLS40IDQtLjcgMi0uNiAzLjUtMS42IDMuNS0xLjYgMS41LTEgMi43LTIuMiAzLjUtMy40LjQtLjcgNi42LTkuOSAxMC4xLTE1LjJDMjQgMTkuMSAzMS40IDguMSAzMS45IDcuM2MxLjItMiAyLjktMy44IDQuOC01LjEgMC0uMSAyLjEtMS4zIDQuNi0yLjFMNDEuOCAwSDV6CiAgICAgICBNNDYuNyAyLjJjLTEuOC4yLTMuMi40LTQgLjctMi4xLjYtMy42IDEuNi0zLjYgMS42LTEuNS45LTIuNiAyLjEtMy41IDMuNC0uNC43LTYuNiA5LjktMTAuMSAxNS4yLTIuOSA0LjQtMTAuMyAxNS41LTEwLjggMTYuMi0xLjIgMi0yLjkgMy44LTQuOCA1LjEgMCAuMS0yLjEgMS4zLTQuNiAyLjFsLS41LjFoMzYuOGwtLjQtLjFjLTIuNS0uOC00LjUtMi00LjYtMi4xLTEuOS0xLjMtMy42LTMuMS00LjgtNS4xLS40LS42LTUuMi03LjgtOC42LTEyLjkuOC0xLjIgMS41LTIuMyAyLjEtMy4xIDAgLjEuMS4xLjEuMiAzLjUgNS4zIDkuNyAxNC41IDEwLjEgMTUuMi45IDEuMyAyIDIuNSAzLjUgMy40IDAgMCAxLjUgMSAzLjUgMS42LjguMyAyLjIuNSA0IC43VjIuMnoiCiAgLz4KPC9zdmc+Cgo=
[transformers-badge]: https://img.shields.io/badge/🤗-Transformers.js-172033?style=flat&labelColor=172033&color=172033
[transformers-url]: https://huggingface.co/docs/transformers.js/en/index
