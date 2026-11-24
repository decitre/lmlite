# Backends and the Python/JavaScript bridge

This page explains why lmLite behaves the way it does across backends and kernels. The guides and the reference say what to call; this says why the results differ.

## One API, three ways to run a model

In a browser kernel a model runs in onnxruntime-web, on `wasm` or `webgpu`. Outside a browser, `pip install 'lmlite[native]'` runs the same ONNX files with native onnxruntime, for fast headless tuning (see [Run natively for tuning](../guides/native-backend.md)). The browser is the source of truth: `tests/browser/test_native_parity.py` checks that the native backend agrees with it.

The fastest generation dtype depends on the backend: `q4` on WebGPU, `fp16` (or `q8` for files over 2 GB) on WASM, where `q4` is 4 to 8 times slower than `q8`. The default fallback order starts with `q8`, the slowest WebGPU choice, so pass `dtype="q4"` with `backend="webgpu"`. The measurements are in [the model guide](../MODELS.md#generation-dtype-selection).

## Why quantized results differ between backends

Quantized (`q8`/`int8`) embeddings are not bit-identical across backends. These models quantize activations at runtime (`DynamicQuantizeLinear`), and onnxruntime-web and native onnxruntime float kernels differ by about one ulp. Usually that changes nothing, but a value sitting on a rounding edge lands in the next uint8 step, and the error compounds layer after layer. On `Xenova/multilingual-e5-small` q8, most vectors match to `1 - cos < 1e-7`, while a tail of inputs drifts up to about `4e-3`. That is the same size as q8's own error against fp32 (mean `3.5e-3`), and session options (graph optimization level, threads) do not reduce it. If a cosine gate or parity test needs the backends to agree, use `embedding_dtype="fp16"` on both: it measured `1 - cos < 1e-7` across backends, sits within `2e-7` of fp32, and is half fp32's size (235 MB vs 470 MB; q8 is 118 MB). fp32 agrees to `1e-12`. Weight-only `q4` also agrees to `1e-12`, but at 399 MB with a larger error against fp32 (mean `1.3e-2`) it is not worth it for e5. Otherwise, leave q8 gates a margin of about `5e-3`. The same applies to greedy generation: int8 GPT-2 logits drift by up to about 1 between backends, enough to pick a different token at a near-tie, while fp32 matches token for token.

## The bridge

A notebook kernel runs in a Web Worker, so Python reaches the browser's JavaScript runtime (`lmlite.js`, Transformers.js, onnxruntime-web) through a bridge: `pyodide.ffi` on Pyodide, `pyjs` on xeus-python. Four consequences follow from that.

- **No page APIs in the kernel.** A worker has no `<canvas>`, no `AudioContext` and cannot open the microphone. Images are encoded to PNG with `OffscreenCanvas`, and the microphone widget records on the page and sends the recording back (as base64 text in xeus-python, where a binary widget trait set by the browser never arrives).
- **Data crosses as one buffer per value.** Images, audio samples and ONNX tensors go over the bridge as raw buffers, not as nested lists; token ids as a JSON string. A call costs far less than a model run, so lmLite makes one call per input and, for `Pipeline.map()`, loops in JavaScript.
- **Callbacks.** A Python function passed to JavaScript is a proxy on Pyodide and an object on xeus-python; lmLite hides the difference for `progress_callback`.
- **The two kernels differ in small ways.** [Choosing a kernel](../RUNTIME.md#choosing-a-kernel-xeus-python-or-pyodide) lists them and recommends one.
