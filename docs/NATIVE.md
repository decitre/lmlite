# Tuning With the Native Backend

lmlite runs models in the browser. Tuning them means picking an embedding model, a dtype, a similarity threshold or margin, and calibration texts. That takes many runs over the same inputs, and in a browser each run is slow and hard to automate. The native backend (`pip install 'lmlite[native]'`) loads the same ONNX files and `tokenizer.json` in CPython with onnxruntime. You tune headless, then ship the result to the browser unchanged.

The browser stays the source of truth. Native exists to make tuning fast, and this guide shows what transfers exactly and what does not.

## Why tune natively

Median latency per `embed()` call on an Apple M-series laptop. Browser numbers are Chromium with ONNX Runtime Web (wasm, single thread, as lmlite serves it today). Native is onnxruntime 1.31.1 on the CPU. "Short" is the median over the 120 parity texts (mostly under 20 tokens); "long" is one ~250-token passage.

| model | browser short | browser long | native short | native long | speedup (long) |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (fp16) | 5.2 ms | 122 ms | 1.1 ms | 13 ms | 9× |
| bge-small-en-v1.5 (fp16) | 10.0 ms | 248 ms | 2.1 ms | 26 ms | 10× |
| multilingual-e5-small (fp16) | 12.0 ms | 276 ms | 2.3 ms | 29 ms | 10× |
| paraphrase-multilingual-MiniLM-L12-v2 (fp16) | 9.8 ms | 280 ms | 2.1 ms | 29 ms | 10× |
| all-mpnet-base-v2 (fp16) | 34.4 ms | 832 ms | 6.0 ms | 77 ms | 11× |

Native is also:

- **Scriptable:** runs in pytest, CI, notebooks, and next to numpy, scikit-learn or pandas, with no browser kernel.
- **Repeatable:** sweeping thresholds over thousands of labelled pairs takes seconds instead of minutes.
- **The same API:** `LLM.create(...)`, `embed()`, `classify()` and greedy `generate()` behave as in the browser, so tuning code moves to a notebook as is.

## What transfers exactly, and what does not

Every model and dtype below was measured with `scripts/native_parity_report.py` over `tests/fixtures/parity/texts.json`: 120 texts covering short queries, long passages, 22 languages, emoji, code and empty input. Drift is `1 - cos` between the browser and native vectors for the same text. Error is `1 - cos` against native fp32, which matches browser fp32 to about `1e-12`.

| model | dtype | size | drift max | texts drifting over 1e-4 | error vs fp32 (mean) |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | q8 | 23 MB | 1e-2 | 9/120 | 1e-2 |
| | fp16 | 45 MB | 7e-8 | 0/120 | 6e-7 |
| | fp32 | 90 MB | 8e-13 | 0/120 | 0 |
| bge-small-en-v1.5 | q8 | 34 MB | 3e-3 | 11/120 | 5e-3 |
| | fp16 | 67 MB | 1e-7 | 0/120 | 6e-8 |
| | fp32 | 133 MB | 4e-13 | 0/120 | 0 |
| multilingual-e5-small | q8 | 118 MB | 3e-3 | 29/120 | 4e-3 |
| | fp16 | 235 MB | 5e-8 | 0/120 | 6e-8 |
| | fp32 | 470 MB | 2e-13 | 0/120 | 0 |
| paraphrase-multilingual-MiniLM-L12-v2 | q8 | 118 MB | 4e-3 | 21/120 | 5e-3 |
| | fp16 | 235 MB | 9e-9 | 0/120 | 6e-7 |
| | fp32 | 470 MB | 1e-13 | 0/120 | 0 |
| all-mpnet-base-v2 | q8 | 110 MB | **0.18** | 35/120 | **0.1** |
| | fp16 | 218 MB | 4e-8 | 0/120 | 8e-6 |
| | fp32 | 436 MB | 7e-13 | 0/120 | 0 |

- **fp32 and fp16 transfer exactly** for any practical purpose. A threshold tuned natively makes the same decisions in the browser.
- **q8 does not.** These models quantize activations to 8 bits at runtime, and the browser and native float kernels differ by about one rounding step (ulp). Most texts come out identical, but now and then a value sits on a rounding edge, lands in the next 8-bit step, and the error grows through the remaining layers. The result is a tail of texts that drift by up to the size of q8's own error. Session options and onnxruntime versions do not remove it. A decision whose cosine falls within the drift of a threshold can flip between native and browser, in either direction.
- **all-mpnet-base-v2 q8 is unusable.** Its error against fp32 averages 0.1. Use fp16 or fp32.
- **Generation:** greedy (`do_sample=False`) fp32 output matches token for token. int8 logits drift by up to about 1, which picks a different token at near-ties. Sampled output never matches, because the two backends use different random generators.

## Choosing the dtype you ship

Latency in the browser does not depend on the dtype. ONNX Runtime Web's wasm kernels run q8, fp16 and fp32 at about the same speed: within 3% on the four small models, and q8 is 8% slower on all-mpnet-base-v2 (long input: q8 898 ms, fp16 832 ms, fp32 827 ms). Quantization saves download size and memory, not compute time.

| ship | when |
|---|---|
| **fp16** (recommended) | Tuned results must hold in the browser. Half fp32's download, same accuracy and latency. |
| q8 | Download size is the hard limit (first visit on mobile data). Leave every threshold a margin of at least the q8 drift above (about `5e-3` for most models, `1.5e-2` for MiniLM-L6), or tune with the browser as the reference. |
| fp32 | Exact bit-level parity, or a model without fp16 weights. |

Downloads are cached by the browser after the first visit, so the size difference only matters once per user and model.

## Browser latency: enable cross-origin isolation

ONNX Runtime Web can run wasm on several threads only when the page is cross-origin isolated, which means it is served with these headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

lmlite does not send them today, so the browser runs single-threaded. With them, the same models got 2.5 to 3.5× faster (run with `--isolated`):

| model (fp16) | single thread, long | isolated, long |
|---|---|---|
| all-MiniLM-L6-v2 | 122 ms | 37 ms |
| multilingual-e5-small | 276 ms | 83 ms |
| all-mpnet-base-v2 | 832 ms | 243 ms |

Isolation is set by the web server or host that serves the JupyterLite site, not by lmlite. Measured on the lmlite 0.11.0 Pyodide site (JupyterLite 0.8.7) with all-MiniLM-L6-v2 fp16 on the long input, run in the JupyterLite REPL:

| browser | Pyodide site, no headers | Pyodide, `credentialless` | Pyodide, `require-corp` |
|---|---|---|---|
| Chrome (Chromium) | 129 ms | 50 ms | 50 ms |
| Firefox 159 | 120 ms | 38 ms | 38 ms |
| Safari 26.6.2 | 136 ms | 137 ms (not isolated) | 76 ms |
| WebKit 26.6 (Playwright) | 131 ms | 130 ms (not isolated) | 52 ms |

The xeus-python site, measured the same way on the lmlite 0.12.0 site built with jl-forge 0.6.0 (xeus-python 0.19.0, pyjs 4.1.0 from emscripten-forge 4x), in Playwright's browsers:

| browser | xeus site, no headers | xeus, `credentialless` | xeus, `require-corp` |
|---|---|---|---|
| Chromium 153 | 140 ms | 57 ms | 54 ms |
| Firefox 155 | 137 ms | 49 ms | 49 ms |
| WebKit 26.6 | 136 ms | 135 ms (not isolated) | 50 ms |

- **Both sites** work isolated in every browser tested, and models still download from the Hugging Face Hub. Safari ignores `credentialless` and stays single-threaded but working, so `credentialless` speeds up Chrome and Firefox without breaking Safari. `require-corp` also isolates Safari (1.8× faster in Safari itself on the Pyodide site), but blocks any cross-origin resource that is not served with CORS or a `Cross-Origin-Resource-Policy` header (for example images linked from notebooks).
- **xeus-python site: needs pyjs 4.1.0 or later.** With older xeus-python environments (jl-forge 0.5.0 and earlier, pyjs 3.2.0), the kernel dies on an isolated page (`generic_type: type "XKernel" is already registered!`), so even `print()` gets no output. Fixed upstream in [jupyterlite/xeus#378](https://github.com/jupyterlite/xeus/issues/378). The xeus site was not retested in Safari itself.

Static hosts such as GitHub Pages cannot set response headers; a service worker that adds them would compete with JupyterLite's own `service-worker.js`. Check that every other cross-origin resource your site loads still works with these headers.

## Guide: from native tuning to a browser deployment

### 1. Get the model files

Download the files the browser will load into `./models/<id>/`, so native and browser read the same bytes:

```bash
hf download Xenova/multilingual-e5-small --local-dir models/Xenova/multilingual-e5-small \
  --include "*.json" "onnx/model_fp16.onnx" "onnx/model.onnx"
```

`pixi run hf <model>` downloads every dtype of the model.

### 2. Tune natively

```python
import numpy as np
from lmlite import LLM

llm = await LLM.create(
    backend="native",
    embedding_model="Xenova/multilingual-e5-small",
    embedding_dtype="fp16",  # the dtype you will ship
    local_models_path="models",
)

anchors = {"transport": ["query: train station", "query: airport terminal"], "health": ["query: a pharmacy"]}
labelled = [("query: coach station for long distance buses", "transport"), ("query: where can I buy medicine", "health")]

anchor_vectors = {label: np.array([await llm.embed(t) for t in texts]) for label, texts in anchors.items()}
def best(vector):
    scores = {label: float((vectors @ vector).max()) for label, vectors in anchor_vectors.items()}
    return max(scores, key=scores.get), scores

for text, expected in labelled:
    label, scores = best(np.array(await llm.embed(text)))
    print(text, label == expected, scores)
```

`embed()` returns unit vectors, so a dot product is the cosine. Sweep thresholds or margins over your labelled set and keep the value with the best accuracy. Note how close the nearest decisions sit to it.

### 3. Check that the tuning transfers

Two checks run the same inputs in Chromium and natively:

```bash
# Drift, dtype error and latency for your model on the parity corpus (or --texts your.json)
pixi run python scripts/native_parity_report.py Xenova/multilingual-e5-small --dtypes fp16

# Gate check: identical accept/reject decisions at your tuned margin
LMLITE_TEST_EMBEDDING_MODEL=Xenova/multilingual-e5-small LMLITE_PARITY_MARGIN=0.05 pixi run native-parity
```

If any of your decisions sits closer to the threshold than the drift for your dtype, move to fp16 or widen the margin.

### 4. Ship to the browser

Use the same model id and dtype in the notebook, and host the same files with the site (see "Hosted Models Discovery" in `USAGE.md`):

```python
llm = await LLM.create(embedding_model="Xenova/multilingual-e5-small", embedding_dtype="fp16")
```

Set `embedding_dtype` explicitly. Without it, the browser picks the Transformers.js default (q8 on wasm), which is not what you tuned.

## Parity test data

`tests/fixtures/parity/texts.json` is the shared corpus behind the numbers above. Both the report script and your own checks can use it with `--texts`. It is small (120 texts) and lives in the repository.

A natural next step is to host **browser baseline vectors**: the Chromium embeddings of this corpus per model and dtype, about 0.2 MB per model and dtype at 384 dimensions, keyed by the Transformers.js version and the ONNX file sha256. With those published (for example as a Hugging Face dataset or a release asset), native-only CI and downstream projects could check drift and decision stability without launching a browser, and regenerate the baselines only when the Transformers.js pin or the model changes.
