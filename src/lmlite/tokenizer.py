# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Load only a tokenizer, without model weights."""

from ._bridge import _js_function, _native, _to_python_deep


class Tokenizer:
    """A tokenize-only handle created with ``tokenizer()``; ``encode()`` and ``decode()`` are synchronous."""

    def __init__(self, js_tokenizer, source):
        self._js = js_tokenizer
        self.source = source
        self.vocab_size = int(_to_python_deep(js_tokenizer.vocab_size))

    def __repr__(self):
        return f"Tokenizer(source={self.source!r}, vocab_size={self.vocab_size})"

    def _handle(self):
        if self._js is None:
            raise RuntimeError(f"Tokenizer for {self.source!r} is closed; load it again with tokenizer().")
        return self._js

    def encode(self, text, *, add_special_tokens=False):
        """Return the token ids of ``text`` as a list of ints, without truncation."""
        import json

        if not isinstance(text, str):
            raise TypeError(f"encode() expects a str, got {type(text).__name__}")
        return json.loads(_to_python_deep(self._handle().encode(text, bool(add_special_tokens))))

    def decode(self, ids):
        """Return the text of a sequence of token ids, keeping special tokens."""
        import json

        return _to_python_deep(self._handle().decode(json.dumps([int(i) for i in ids])))

    def close(self):
        """Release the loaded tokenizer; later ``encode()``/``decode()`` calls raise. Repeated calls are safe."""
        self._js = None


async def tokenizer(model=None, *, tokenizer_json=None, tokenizer_config=None, hosted_models_url=None):
    """Load only a tokenizer, without model weights, and return a ``Tokenizer``.

    Pass exactly one of ``model`` (hosted ``files/models/`` first, then Hub files, as in
    ``LLM.create()``) or ``tokenizer_json`` (an HTTP(S) URL, or a path relative to the
    JupyterLite site root such as ``"files/models/org/name/tokenizer.json"``), optionally with
    ``tokenizer_config``. Each call loads an independent handle.
    """
    if bool(model) == bool(tokenizer_json):
        raise ValueError("Pass exactly one of model= or tokenizer_json=.")
    if model and tokenizer_config:
        raise ValueError("tokenizer_config= requires tokenizer_json=; model= loads the model's own config.")
    source = model or tokenizer_json
    native = _native()
    load_tokenizer = native.load_tokenizer if native else _js_function("loadTokenizer")
    tokenizer = Tokenizer(await load_tokenizer(model, tokenizer_json, tokenizer_config, hosted_models_url), source)
    # Fail at load time, not on the first call, if this kernel's bridge cannot call encode synchronously.
    try:
        tokenizer.encode("a")
    except Exception as exc:
        raise RuntimeError(f"This kernel cannot call the tokenizer for {source!r} synchronously: {exc!r}") from exc
    return tokenizer
