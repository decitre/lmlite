[![Python package][ci-badge]][ci-url]
[![codecov-python][codecov-python-badge]][codecov-url]
[![codecov-js][codecov-js-badge]][codecov-url]
[![Pixi Badge][pixi-badge]][pixi]
[![Transformers.js][transformers-badge]][transformers-url]

[![xeus-badge][xeus-badge]][xeus-url]
[![pyodide-badge][pyodide-badge]][pyodide-url]


# lmLite

Package version: `0.5.5`

**Browser-native LLM orchestration in JupyterLite** - run language models entirely in your browser using WebAssembly or WebGPU.

## Quick Start

**Try online** (no installation required):
- [Launch with Xeus-Python][xeus-url] (recommended)
- [Launch with Pyodide][pyodide-url]

**Basic usage:**

```python
from lmlite import LLM

# Create LLM instance (downloads model on first run)
llm = await LLM.create(
    generator_model="gpt2",
    embedding_model="all-MiniLM-L6-v2"
)

# Generate text
text = await llm.generate("Python is a great language because")

# Generate embeddings
embedding = await llm.embed("Hello world")

# Similarity search
docs = ["JupyterLite runs in the browser.", "Python is widely used for ML."]
results = await llm.similarity_search("Where does JupyterLite run?", docs)
```

## Features

- **Browser-native**: Run LLMs entirely in the browser, no server required
- **Privacy-first**: All computation happens locally in your browser
- **Zero setup**: No installation or API keys for hosted demos
- **Multiple backends**: WebAssembly (WASM) or WebGPU
- **Multiple models**: GPT-2, DistilGPT-2, TinyLlama, and more via Transformers.js
- **Embeddings & classification**: Sentence vectors and text classification
- **Model caching**: Browser-cached downloads and portable ZIP export/import
- **Pythonic API**: Clean, async Python interface

## What is lmLite?

lmLite is a lightweight Python-to-JavaScript bridge for **local LLM inference in the browser**, designed for JupyterLite notebooks. It provides:

- Async Python API for text generation, embeddings, classification, and similarity search
- Model loading, caching, and portable ZIP workflows
- Independent backend/dtype controls for embeddings and generation

**Inference is powered by [Transformers.js](https://github.com/huggingface/transformers.js) and ONNX Runtime**. lmLite is not a new model engine or native Python inference backend. Its similarity search is an in-memory cosine ranking helper, not a vector database.

**Best suited for:** Teaching, interactive experiments, and browser-local prototypes where avoiding remote inference services matters.

> **Note:** Installing the Python package on desktop does **not** enable desktop inference. `LLM.create()` requires a browser Python runtime (xeus-python or Pyodide) with JavaScript bridge. See [docs/RUNTIME.md](docs/RUNTIME.md) for details.

## Documentation

- **[Getting Started](docs/DEVELOPMENT.md)** - Installation, development setup, and testing
- **[Usage Guide](docs/USAGE.md)** - Detailed API examples and configuration
- **[Model Guide](docs/MODELS.md)** - Recommended models and dtype selection
- **[Runtime & Integration](docs/RUNTIME.md)** - Platform support and integration boundaries
- **[Glossary](docs/GLOSSARY.md)** - Key concepts and terminology

## Notebooks

Open [`demo.ipynb`](notebooks/demo.ipynb) in the [online demo][xeus-url] to see lmLite in action. See [notebooks/README.md](notebooks/README.md) for a complete notebook guide.

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
              │ Bridge (pyodide.ffi / pjs)
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

## Acknowledgments

- [Transformers.js](https://github.com/huggingface/transformers.js) by [@xenova](https://github.com/xenova) / HuggingFace
- [JupyterLite](https://jupyterlite.readthedocs.io/) team
- [Pyodide](https://pyodide.org/) project
- [Hugging Face](https://huggingface.co/) for model hosting

---

[ci-badge]: https://github.com/decitre/lmlite/actions/workflows/test.yml/badge.svg
[codecov-python-badge]: https://img.shields.io/codecov/c/github/decitre/lmlite/main?flag=python&label=py%20cov&logo=codecov&token=CODECOV_TOKEN
[codecov-js-badge]: https://img.shields.io/codecov/c/github/decitre/lmlite/main?flag=javascript&label=js%20cov&logo=codecov&token=CODECOV_TOKEN
[ci-url]: https://github.com/decitre/lmlite/actions/workflows/test.yml
[codecov-url]: https://codecov.io/gh/decitre/lmlite
[xeus-url]: https://decitre.github.io/lmlite/xeus/lab/?path=demo.ipynb
[pyodide-url]: https://decitre.github.io/lmlite/pyodide/lab/?path=demo.ipynb
[pyodide-badge]: https://jupyterlite.rtfd.io/en/latest/_static/badge.svg
[xeus-badge]: https://img.shields.io/badge/try-lite%20now-f7dc1e.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0Ni43IDQ2LjciPgogIDxwYXRoCiAgICBmaWxsPSIjNDBCM0MzIgogICAgZD0iTTUgMGMyLjUuOCA0LjUgMiA0LjYgMi4xIDEuOSAxLjMgMy42IDMuMSA0LjggNS4xLjQuNiA1LjIgNy44IDguNiAxMi45LS44IDEuMi0xLjUgMi4zLTIuMSAzLjEgMC0uMS0uMS0uMS0uMS0uMkMxNy42IDE3LjkgMTEuNCA4LjYgMTEgNy45Yy0uOS0xLjMtMi0yLjUtMy41LTMuNCAwIDAtMS41LTEtMy41LTEuNkMzLjIgMi42IDEuOCAyLjQgMCAyLjJ2NDIuNGMxLjgtLjIgMy4yLS40IDQtLjcgMi0uNiAzLjUtMS42IDMuNS0xLjYgMS41LTEgMi43LTIuMiAzLjUtMy40LjQtLjcgNi42LTkuOSAxMC4xLTE1LjJDMjQgMTkuMSAzMS40IDguMSAzMS45IDcuM2MxLjItMiAyLjktMy44IDQuOC01LjEgMC0uMSAyLjEtMS4zIDQuNi0yLjFMNDEuOCAwSDV6CiAgICAgICBNNDYuNyAyLjJjLTEuOC4yLTMuMi40LTQgLjctMi4xLjYtMy42IDEuNi0zLjYgMS42LTEuNS45LTIuNiAyLjEtMy41IDMuNC0uNC43LTYuNiA5LjktMTAuMSAxNS4yLTIuOSA0LjQtMTAuMyAxNS41LTEwLjggMTYuMi0xLjIgMi0yLjkgMy44LTQuOCA1LjEgMCAuMS0yLjEgMS4zLTQuNiAyLjFsLS41LjFoMzYuOGwtLjQtLjFjLTIuNS0uOC00LjUtMi00LjYtMi4xLTEuOS0xLjMtMy42LTMuMS00LjgtNS4xLS40LS42LTUuMi03LjgtOC42LTEyLjkuOC0xLjIgMS41LTIuMyAyLjEtMy4xIDAgLjEuMS4xLjEuMiAzLjUgNS4zIDkuNyAxNC41IDEwLjEgMTUuMi45IDEuMyAyIDIuNSAzLjUgMy40IDAgMCAxLjUgMSAzLjUgMS42LjguMyAyLjIuNSA0IC43VjIuMnoiCiAgLz4KPC9zdmc+Cgo=
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
[pixi]: https://pixi.sh
[transformers-badge]: https://img.shields.io/badge/🤗-Transformers.js-172033?style=flat&labelColor=172033&color=172033
[transformers-url]: https://huggingface.co/docs/transformers.js/en/index
