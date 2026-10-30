# Development Guide

This guide covers installation, development setup, testing, and building lmLite.

## Prerequisites

- [pixi](https://pixi.sh/) - Package manager (required)
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) - Conda package manager (required for JupyterLite builds)

**Supported Platforms**: macOS (Intel/Apple Silicon) and Linux. Windows is not currently supported for development.

## Installation

1. Clone the repository and install dependencies with [pixi](https://pixi.sh/):
   ```shell
   git clone https://github.com/decitre/lmlite.git
   cd lmlite
   pixi install
   ```

2. Run the tests to verify installation:
   ```shell
   pixi run tests
   ```

**Note**: If `pixi run tests` fails with "Error launching 'pytest': No such file or directory", the `.pixi` environment cache may be corrupted. Rebuild it with:
```shell
rm -rf .pixi
pixi install --all
```

## Development Tasks

### Running Tests

```bash
# Python tests
pixi run tests

# JavaScript tests
pixi run js-tests

# Python coverage
pixi run coverage

# JavaScript coverage
pixi run js-coverage

# Linting
pixi run lint
pixi run lint-check
```

### Node.js Version for JavaScript Tests

JavaScript tests and coverage require a modern Node.js runtime. If `pixi run js-tests` or `pixi run js-coverage` fails with `Cannot find module 'node:path'`, Pixi may be picking up an old Node.js installation from your shell (e.g., Node.js 10 managed by nvm). Add Node.js 22 to the project's Pixi `dev` feature:

```bash
pixi add --feature dev 'nodejs>=22,<23'
pixi run js-coverage
```

The `pixi add` command updates the dependency manifest and lockfile; retain those changes to make the runtime requirement reproducible for other developers.

### Browser Smoke Tests

#### Model Loading Smoke Test

The browser smoke test loads `Xenova/gpt2` by default:

```bash
pixi run install-browser
pixi run model-smoke
```

To test TinyLlama instead:

```bash
LMLITE_TEST_MODEL_ID=Xenova/TinyLlama-1.1B-Chat-v1.0 \
LMLITE_TEST_BACKEND=wasm pixi run model-smoke
```

#### Embedding Smoke Tests

JavaScript unit tests mock Transformers.js. The separate, opt-in `embedding-smoke` task exercises lmLite's actual browser runtime against local bundled, randomly initialized BERT weights using Playwright, without downloading model assets during the test. This is a real transformer encoder, not a mocked pipeline; meaningful semantic rankings are not required for these contract checks:

```bash
pixi run install-browser
pixi run embedding-smoke
```

Both fp32 and q8 runs must return finite, 16-dimensional, unit-length sentence vectors for texts of different lengths. The harness also checks successful HTTP loading of the requested ONNX variant and tokenizer assets from the local server. It rejects alternate weight downloads.

The current Transformers.js runtime makes Hub metadata probes even for HTTP-hosted models; for the bundled fixture, a test-only fetch adapter redirects those probes to the real local files. It does not mock pipeline outputs or change lmLite's production loading behavior. This verifies real pooling, normalization, and dtype file selection, not merely mocked pipeline options. It does not claim bit-identical embeddings across devices or that quantization must change every output value.

The original fixture's six assets total 83002 bytes (about 81 KiB) and are included under `tests/fixtures/models/lmlite/tiny-bert/`, licensed Apache-2.0. They contain no pretrained weights or third-party tokenizer vocabulary. The smoke test verifies asset SHA-256 hashes recorded in `fixture.json`. Do not treat this randomly initialized model as a useful sentence-embedding model.

**Regenerating the fixture:**

```bash
pixi run --environment model-fixture generate-embedding-fixture
```

The export environment supports Apple Silicon macOS and Linux; normal smoke execution also supports Intel macOS. See the [fixture provenance and regeneration notes](https://github.com/decitre/lmlite/blob/main/tests/fixtures/models/lmlite/tiny-bert/README.md).

**Optional representative sentence-model run:**

```bash
pixi run hf Xenova/all-MiniLM-L6-v2 -- \
  config.json tokenizer.json tokenizer_config.json \
  onnx/model.onnx onnx/model_quantized.onnx
LMLITE_TEST_EMBEDDING_MODEL=Xenova/all-MiniLM-L6-v2 pixi run embedding-smoke
```

That run checks 384-dimensional vectors with the same normalization and local-weight loading assertions. It permits upstream `Range: bytes=0-0` metadata probes but rejects non-metadata Hub requests.

**WebGPU testing:**

WASM is the default. To exercise WebGPU explicitly:

```bash
LMLITE_TEST_EMBEDDING_BACKEND=webgpu pixi run embedding-smoke
```

WebGPU requires support in the browser, execution environment, model, and dtype; the test does not skip failures or fall back to WASM. Optional MiniLM model files stay under `models/Xenova/all-MiniLM-L6-v2/`; missing files fail with preparation instructions. Chromium and the Transformers.js/ONNX runtime dependencies must also be available. Runtime dependencies are loaded from the CDN, so the test is not fully offline. Only the optional MiniLM run also needs Hub metadata access.

#### Task Pipeline Smoke Tests

The opt-in `tokenizer-smoke` task checks `LLM.tokenizer()` in Chromium. It compares Transformers.js ids with reference ids from the Rust `tokenizers` library for a small bundled Unigram/Metaspace tokenizer (`tests/fixtures/models/lmlite/tiny-unigram/`). It also checks that no weight file or Hub request is made, and that `encode()` returns plain Python ints in Pyodide, which it loads from the jsDelivr CDN:

```bash
pixi run tokenizer-smoke
```

`xeus-tokenizer-smoke` runs the same checks in a real xeus-python kernel. It builds a JupyterLite xeus site from the current `src/lmlite` into a temporary directory, so it needs the `xeus-site` environment and network access to emscripten-forge:

```bash
pixi run -e xeus-site xeus-tokenizer-smoke
```

Regenerate the tokenizer and its expected ids, with `tokenizers` pinned, after changing the spec or test texts:

```bash
pixi run --environment model-fixture generate-tokenizer-fixture
```

The opt-in `task-smoke` task loads each `LLM.pipeline()` task (zero-shot classification, translation and speech recognition) through the browser runtime, with real models downloaded from the Hub:

```bash
pixi run task-smoke
```

It runs each task on WASM and WebGPU. WebGPU runs are skipped when Chromium exposes no WebGPU adapter. Set `LMLITE_TEST_TASK_BACKENDS=wasm` to restrict backends. Override the models with `LMLITE_TEST_ZERO_SHOT_MODEL`, `LMLITE_TEST_TRANSLATION_MODEL` and `LMLITE_TEST_ASR_MODEL`.

Set `LMLITE_MODEL_TIMEOUT_MS` to adjust the inference wait limit (default: 900000 ms). None of the browser smoke tasks run in the default non-browser test suite.

### Building a Wheel

```bash
pixi run wheel
```

### Version Management

Set a specific package version in README.md and `src/lmlite/version.py` without creating a commit or tag:

```bash
pixi run set-version <M>.<m>.<p>
```

## Building JupyterLite Demo

```bash
pixi run jupyterlite
```

Follow the instructions provided by the command.

### Hosting Models in JupyterLite

Model bundling is opt-in and applies to both the Xeus and Pyodide sites. Download models into a Hugging Face-style `models/<namespace>/<model>/` directory, then build with `-m`:

The `hf` Pixi task downloads a complete model repository into that layout, including its `onnx/` files; no `--include` option is needed:

```bash
pixi run hf Xenova/gpt2
pixi run jupyterlite -m
```

For a specific ONNX variant, select the required files explicitly:

```bash
pixi run hf HuggingFaceTB/SmolLM2-360M-Instruct -- \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json vocab.json merges.txt onnx/model_q4.onnx

pixi run jupyterlite -m
```

Download the configuration, tokenizer assets, and the ONNX variant you intend to run—not only the ONNX weight file. Include any external ONNX data files required by the model. For models stored elsewhere, pass their containing `<namespace>/<model>/` tree with `-M`; keep that source outside `workspace/`, which the build recreates:

```bash
pixi run jupyterlite -M /path/to/models
```

The `-m`/`-M` options stage individual model files in `content-xeus/models/` and `content-pyodide/models/` before invoking JupyterLite's builds. JupyterLite publishes them at each site's HTTP `files/models/<namespace>/<model>/` URLs. A `files/models/manifest.json` lets `LLM.create(...)` discover the hosted models and load their files directly over HTTP with browser caching and download progress, bypassing the `/drive` filesystem. Missing files in a hosted model produce an error rather than downloading a different copy from Hugging Face.

The build requires individual model directories containing configuration, tokenizer assets, and the selected ONNX weights.

For hosted files, enable gzip or Brotli **HTTP compression** on your hosting server when supported. Serve the original filenames with the appropriate `Content-Encoding` header; browsers decompress them transparently. This build does not generate `.gz` filenames or configure your server. Python's simple HTTP server works with the uncompressed files. Hosting models increases site size and requires a source outside `workspace/`.

See [MODELS.md](MODELS.md#downloading-and-bundling-models) for download filtering and matching the selected ONNX variant to `LLM.create(...)`.

### Preparing Models for Another Hosted Site

The installed package includes `lmlite-models`; no repository checkout is required. It copies individual assets from a `<namespace>/<model>/` source tree and writes `manifest.json`, excluding download metadata and backup files:

```bash
lmlite-models --source-dir /path/to/models --output-dir /path/to/site/files/models
```

The output must be outside the source directory. Publish it over HTTP and pass its URL as `hosted_models_url` when automatic discovery does not apply.

Use `--format zip` explicitly to prepare portable model archives instead; ZIP mode caches archives in the source directory and reuses them on subsequent runs.

The same CLI is available through `python -m lmlite.model_assets`. Python callers can use `host_models` and `pack_models` from `lmlite.model_assets`.

The JupyterLite build invokes `lmlite-models` directly with `--format files`. When using `-m` or `-M`, lmLite must be installed and `lmlite-models` must be available on `PATH`; builds without model hosting do not require the CLI. The build script does not invoke Pixi.

## Troubleshooting

### Models downloading every time?

Check browser console for Cache API availability. Some privacy settings may disable caching.

### Out of memory errors?

Try smaller models like `distilgpt2` or reduce `max_new_tokens`.

### CORS errors?

Ensure you're running from `http://localhost` or a proper HTTPS domain, not `file://`.

### "Failed to load text-generation model 'undefined'" on Pyodide?

This error indicates the configuration parameters aren't being properly passed to the JavaScript layer. This was a known issue with dictionary conversion in older versions. The fix is included in the latest version. If you encounter this issue:

1. Ensure you're running the latest version of lmlite
2. Clear your browser cache (IndexedDB and Cache API)
3. Restart the kernel and try again
