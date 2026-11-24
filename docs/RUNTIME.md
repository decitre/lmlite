# Runtime scope and positioning

lmLite is a lightweight orchestration layer for browser-local language-model
inference from Python. JupyterLite, with Xeus-Python or Pyodide, is its primary
integration. The package provides async generation, embeddings, classification,
and in-memory cosine similarity search, Transformers.js task pipelines (text,
vision, speech and image-output tasks), bare ONNX models through `OnnxModel`,
and a microphone widget, together with browser model caching, hosted assets,
and portable model ZIP import/export.

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

lmLite is not a model-training toolkit, a production inference service, a
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
Model preparation is described in [the model guide](MODELS.md).

### JavaScript browser applications

**JavaScript runtime:** `lmlite.js` exposes `BrowserRuntime` and helper functions
under a single `globalThis.lmLite` namespace for integration with the Python API. These interfaces are
internal implementation details, not a separately documented or versioned
JavaScript SDK. Direct JavaScript usage may be possible, but its initialization,
integration, and compatibility are not guaranteed independently of the Python API.

For an application already written in JavaScript, direct Transformers.js is
usually the simpler choice unless lmLite's model-loading and cache workflows
provide a specific benefit.

## Native Python and Node.js

The Python API calls a browser `js` bridge. A conventional CPython process or
server-backed Jupyter kernel does not supply that bridge. There, `pip install
'lmlite[native]'` adds a headless onnxruntime backend that runs the same ONNX
files for `Embedder`, `Classifier`, greedy `LLM.generate()`, `OnnxModel` and
`tokenizer()` (see [the native guide](guides/native-backend.md)); the other
calls, such as `pipeline()`, raise an error that says they need a browser
kernel. The native backend is for tuning models before shipping them, not an
inference service.

Node.js is also not supported for inference as-is: the runtime uses
browser-oriented dependency loading and APIs such as the browser Cache API.
Using Node.js for development tests is separate from supporting it as an
application inference runtime.

## Desktop and CI model preparation

Model preparation does work outside the browser. The `lmlite-models` CLI and
`host_models` / `pack_models` helpers in `lmlite.model_assets` prepare downloaded
assets for static hosting or ZIP portability without performing inference.
They can be used from desktop Python or CI independently of JupyterLite.

## Choosing a kernel: xeus-python or Pyodide

lmLite runs the same Python API on both, and both are covered by the browser tests (see [DEVELOPMENT.md](DEVELOPMENT.md)). The model code is the same JavaScript (`lmlite.js`, Transformers.js, ONNX Runtime Web) in both, so inference speed does not depend on the kernel.

| | xeus-python | Pyodide |
|:--|:--|:--|
| What it is | CPython built with emscripten, packages from the emscripten-forge conda channel, bridged with `pyjs` | CPython built with emscripten, wheels from the Pyodide distribution and PyPI, bridged with `pyodide.ffi` |
| Cross-origin isolation (multi-threaded WASM) | works with pyjs 4.1.0 or later (jl-forge 0.6.0 and later); 137 ms to 50 ms on a long input | works; the same 2.5 to 3.5 times speedup (see [NATIVE.md](NATIVE.md)) |
| Python call overhead | about 0.05 ms per `encode()` | about 0.02 ms per `encode()` |
| Differences lmLite works around | a function passed to JavaScript is an object, not a callable; binary widget traits set by the browser never reach the kernel (the microphone recording travels as base64); loading a FreeType font hangs (use `ImageFont.load_default_imagefont()`) | none found so far |
| Site build | `jl-forge` solves a conda environment on the first build | `jl-forge` with `uv` and the wheels in `pyproject.toml`; the site fetches nothing from PyPI |

Both kernels need the same things from the host: the page's service worker proxies cross-origin requests, so headers for isolation have to come from the server (GitHub Pages cannot send them).

**Recommendation.** Use xeus-python by default: it is the kernel of the hosted demos and of most tests (the real JupyterLab UI test, `lab-smoke`, runs on it). Use Pyodide when you want the smallest differences from desktop Python: an install of any pure-Python wheel from PyPI at run time, no workarounds in widget or font code, and a site build without a conda solve. The workarounds above are in lmLite and the notebooks, so a notebook of this project runs on either.

Not measured: kernel start-up time, the size of each site, and memory use of the kernel itself. Measure those with your own notebooks before choosing on size or start-up.

## Browsers tested

- **Chromium and Firefox:** the notebooks for images, speech and the microphone run in the real JupyterLab UI with xeus-python (`lab-smoke`, with Playwright's browsers; see [DEVELOPMENT.md](DEVELOPMENT.md)). A user report adds stock Firefox 159 on an Apple-silicon Mac: model downloads, text to speech and the text-to-image recipe on WebGPU worked.
- **Safari:** only Playwright's WebKit is tested, for cross-origin isolation (see [NATIVE.md](NATIVE.md)). The notebooks have not been run in Safari itself.
- **WebGPU:** needs a hardware adapter. Headless Chromium uses a software adapter without float16 shaders, which is enough for small models but not for the text-to-image recipe.
