# Run a bare ONNX model

## Bare ONNX models

`OnnxModel` runs a single `.onnx` file that has no Hugging Face repository around it (no `config.json`, no tokenizer): a face or object detector, an OCR stage, a custom classifier. NumPy goes in, a dict of NumPy arrays comes out, and you do the pre- and post-processing in NumPy.

```python
import numpy as np
from lmlite import OnnxModel

model = await OnnxModel.load("models/detector.onnx", backend="wasm", licence="Apache-2.0")  # path, URL, bytes or hosted ID
model.inputs, model.outputs    # [{"name", "dtype", "shape"}], symbolic dimensions stay strings such as "batch"
outputs = await model.run({"input": blob.astype(np.float32)})
model.model_info()             # sha256, size, backend, execution provider, cache status, inputs, outputs, licence
await model.dispose()
```

- **Backend:** `"wasm"` or `"webgpu"` in a browser kernel (onnxruntime-web, the version Transformers.js uses; it fails to load if the browser has no WebGPU). Outside a browser, or with `backend="native"`, the same API runs onnxruntime from `pip install 'lmlite[native]'`.
- **Inputs:** `run()` checks each input before running and raises an error that names it: a missing or unknown name, a dtype that does not match the model (no silent casts: use `astype()`), a wrong rank, or a wrong size on a fixed axis. Dynamic dimensions accept any size.
- **Many inputs:** `await model.map(feeds, progress_callback=None, on_error="raise")` calls `run()` for each element of `feeds` and returns the outputs in order. `feeds` is any iterable, or async iterable, of feed dicts, so a generator that decodes and resizes one image at a time never holds them all in memory. `progress_callback(done, total)` is called after each input (`total` is `None` for a generator, which has no length). `on_error` works as for `Pipeline.map()`: `"raise"`, `"skip"` (a failed input is left out, so the results are shorter than the inputs) or `"return"` (an exception object in its place, so a wrong dtype or shape fails one input and not the run). An error raised by the iterable itself ends the run. `await model.cancel()` stops after the current input and `map()` returns the outputs so far; the model stays usable, and only one `map()` runs at a time. Unlike `Pipeline.map()` the loop runs in Python, one bridge call per input: an ONNX model's outputs can be large (about 1 MB per image for a detector) and a bridge call costs far less than the inference, so a JavaScript loop would gain little.
- **Bridge cost:** each `run()` crosses the Python/JavaScript bridge once, with raw buffers. Use one call per image: a detector with 16,800 anchors of 15 floats returns about 1 MB per image.
- **Browser files:** a path is read in Python and sent as bytes; a URL is fetched by the browser and kept in the Cache API (`cache=False` turns that off), so loading the same URL again makes no request, and `clear_cache()` removes it. `cached_models()` lists such a file as an entry whose `model_id` is `None`, and the cache widget does not show it. `model_info()["cache_status"]` is `"cached"` when the bytes came from the cache, `"downloaded"` when they were just stored, and `None` otherwise. Pass `sha256="..."` to require a hash: a mismatch raises an error, and a cached copy that no longer matches is dropped and fetched again. A model with external data files is not supported, and the whole file must fit in browser memory.
- **Hosted IDs:** a `namespace/model` ID that does not end in `.onnx` is a hosted model: list it in the site's `files/models/manifest.json` (or the manifest at `hosted_models_url=`) and put the file at `files/models/<id>/model.onnx`. `OnnxModel.load("lmlite/detector")` then loads it from the site, with caching. Outside a browser, give a path or URL.
- **onnxruntime-web:** its JavaScript is loaded from the site's `vendor/transformers/` when `scripts/lite-build.sh` put it there, else from jsDelivr.
- **float16:** supported. On WebGPU it also needs an adapter with `shader-f16`: otherwise the run fails with an onnxruntime-web error. WASM has no such requirement. Checked against native onnxruntime on an Apple Metal GPU; headless Chromium gives WebGPU a software adapter without float16 unless you pass the flags that select the GPU.
- **A large example:** [`ch08g_text-to-image.ipynb`](https://github.com/decitre/lmlite/blob/main/notebooks/ch08g_text-to-image.ipynb) runs three float16 models (a 1.7 GB UNet among them) with `OnnxModel`; the numbers and the limits are in [MODELS.md](../MODELS.md#text-to-image-a-recipe-not-a-pipeline-task).
