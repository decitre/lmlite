# Runtime scope and positioning

lmLite is a lightweight orchestration layer for browser-local language-model
inference from Python. JupyterLite, with Xeus-Python or Pyodide, is its primary
integration. The package provides async generation, embeddings, classification,
and in-memory cosine similarity search, together with browser model caching,
hosted assets, and portable model ZIP import/export.

## What lmLite adds

Transformers.js and ONNX Runtime load and execute the models. lmLite bridges
those capabilities into browser Python and handles notebook-oriented loading
progress, model-source discovery, and independent embedding device/dtype
configuration. Models and browser/runtime capabilities determine inference
quality, compatibility, and resource requirements.

This makes lmLite useful for teaching, interactive experimentation, and small
browser-local prototypes without a remote inference service. Model and runtime
downloads may still require network access; local inference does not mean a
fresh installation works offline.

lmLite is not a model-training toolkit, a native Python inference engine, a
vector database, or a complete RAG framework. Its similarity helper embeds
documents and ranks them by cosine similarity in memory. Applications remain
responsible for chunking, model-specific input preparation, persistent indexes,
retrieval policy, prompt construction, and evaluation.

## Inference outside JupyterLite

### Standalone Pyodide browser applications

JupyterLite's notebook interface is not inherently required for browser
inference. A standalone Pyodide application is a potential integration target,
but there is no turnkey standalone integration documented or guaranteed here.
The host would need to provide the Python-to-JavaScript bridge, load the package
and Transformers.js dependencies, and integrate asynchronous initialization
and progress display.

Notebook widget progress is tied to notebook frontend support. A custom host
would need to accommodate that behavior or provide suitable progress rendering.
It must also account for browser memory, model/backend/dtype compatibility,
and HTTP(S) access and CORS for model files.

Automatic hosted-model discovery recognizes JupyterLite URLs. A host without
those URLs should provide `hosted_models_url` pointing to an HTTP(S) model
directory with `manifest.json` and the expected namespace/model layout.
Model preparation is described in [the README](https://github.com/decitre/lmlite/blob/main/README.md#prepare-models-for-another-hosted-site)
and [the model guide](https://github.com/decitre/lmlite/blob/main/docs/MODELS.md).

### JavaScript browser applications

**JavaScript runtime:** `llm.js` exposes `BrowserLLM` and helper functions
under a single `globalThis.lmLite` namespace for integration with the Python API. These interfaces are
internal implementation details, not a separately documented or versioned
JavaScript SDK. Direct JavaScript usage may be possible, but its initialization,
integration, and compatibility are not guaranteed independently of the Python API.

For an application already written in JavaScript, direct Transformers.js is
usually the simpler choice unless lmLite's model-loading and cache workflows
provide a specific benefit.

## Native Python and Node.js

The Python inference API calls a browser `js` bridge. A conventional CPython
process or server-backed Jupyter kernel does not supply that bridge, and lmLite
has no native inference fallback. Installing the package there does not make
`LLM.create()` an inference service. For native Python inference, use a runtime
designed for that environment rather than relying on lmLite's browser bridge.

Node.js is also not supported for inference as-is: the runtime uses
browser-oriented dependency loading and APIs such as the browser Cache API.
Using Node.js for development tests is separate from supporting it as an
application inference runtime.

## Desktop and CI model preparation

Model preparation does work outside the browser. The `lmlite-models` CLI and
`host_models` / `pack_models` helpers in `lmlite.model_assets` prepare downloaded
assets for static hosting or ZIP portability without performing inference.
They can be used from desktop Python or CI independently of JupyterLite.
