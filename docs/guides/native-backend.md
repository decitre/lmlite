# Run natively for tuning

Outside a browser kernel, `pip install 'lmlite[native]'` adds an onnxruntime backend that runs the same ONNX files and `tokenizer.json` the browser loads, for fast headless iteration. It is picked automatically when no browser JS runtime exists, or explicitly with `backend="native"`:

```python
from lmlite import Embedder, tokenizer

embedder = await Embedder.create(
    "intfloat/multilingual-e5-small",  # or a model directory path
    dtype="q8",
    local_models_path="site/files/models",  # looks in <path>/<model id>/, then the Hugging Face Hub cache
)
vec = await embedder.embed("query: sushi next to a pharmacy")
tok = await tokenizer(tokenizer_json="site/files/models/org/name/tokenizer.json")
```

It supports `embed()`, `classify()`, `generate()` (greedy, or top-k sampling with temperature and `repetition_penalty`) and `tokenizer()`; other calls raise. dtype selection follows the browser: an `Embedder`'s `dtype`, else the model's `transformers.js_config` dtype, else `q8`; `dtype`/`dtype_fallbacks`, else `q8`, `int8`, `fp32` for generation and classification. Sampled text cannot match the browser token for token (different random generators); use `do_sample=False` to compare. The browser remains the source of truth; `tests/browser/test_native_parity.py` (`pixi run native-parity`) checks the two agree. See [NATIVE.md](../NATIVE.md) for why and how to tune natively, measured parity and latency per model and dtype, and browser deployment.
