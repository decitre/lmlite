# Load only a tokenizer

## Tokenize-Only Handles

`tokenizer()` (also `LLM.tokenizer()`) loads only a tokenizer, with no model weights and no inference pipeline, for workloads such as static embeddings (model2vec/POTION) that tokenize, gather rows from a precomputed table and pool. Loading is async; `encode()` is synchronous and returns a plain `list[int]`:

```python
from lmlite import tokenizer

tok = await tokenizer(model="minishlab/potion-multilingual-128M")
# or: tok = await tokenizer(tokenizer_json="files/models/<org>/<name>/tokenizer.json",
#                               tokenizer_config="files/models/<org>/<name>/tokenizer_config.json")

ids = tok.encode("sushi right next door to a pharmacy")   # sync, no await
tok.encode("text", add_special_tokens=True)              # default False, unlike Transformers.js
tok.decode(ids), tok.vocab_size
tok.close()                                              # later calls raise; repeated close() is safe
```

- **Arguments**: pass exactly one of `model=` or `tokenizer_json=`; `tokenizer_config=` goes only with `tokenizer_json=`. Errors (bad arguments, missing files, unsupported tokenizer types) raise in `tokenizer()`, not on the first `encode()`.
- **`model=`** fetches `tokenizer.json` and, when present, `tokenizer_config.json` from the hosted `files/models/` snapshot first, then from the Hub, like `LLM.create()` (`hosted_models_url=` works the same way). It never makes the Hub task lookup, so hosted tokenizers work offline. `/drive/models` ZIPs are not read.
- **`tokenizer_json=` / `tokenizer_config=`** are HTTP(S) URLs. Relative paths resolve against the JupyterLite site root, so `files/models/...` is the hosted files directory. Paths in the kernel's virtual filesystem are not supported.
- **Without a config**, ids come from `tokenizer.json` alone, which holds the vocabulary, normalizer, pre-tokenizer and special-token template. The config only selects the Transformers.js tokenizer class (`tokenizer_class`) and named special tokens such as `eos_token`, which a few tokenizer classes use when encoding.
- **Encoding** never truncates. Non-`str` input raises `TypeError`. The empty string gives what the Rust `tokenizers` library gives for the same spec (`[]` for the Unigram/Metaspace test fixture).
- **Lifetime**: each call loads an independent handle; closing one does not affect others.
- **Cache and progress**: tokenizer files go through the browser's HTTP cache only. They do not appear in `cached_models()`: a tokenizer-only Cache API entry would make `LLM.create()` treat the whole model as already cached. There is no `progress_callback`, because tokenizer files are small.
- **Kernels**: works on Pyodide and xeus-python. Ids cross the bridge as a JSON string, so both kernels return plain `int`s. Both have opt-in browser tests. If a kernel cannot call `encode()` synchronously, `tokenizer()` raises at load time.
- **Latency**: about 0.02 ms (Pyodide) and 0.05 ms (xeus-python) per `encode()` of a short sentence, warm, in headless Chromium on Apple Silicon with the 97-token test vocabulary. Measure with your own tokenizer before relying on these numbers.
