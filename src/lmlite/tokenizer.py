# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Load only a tokenizer, without model weights."""

from ._bridge import _js_function, _native, _to_python_deep


class Tokenizer:
    """A loaded tokenizer, created with ``tokenizer()``. It has no model weights.

    ``encode()`` and ``decode()`` are synchronous, so they can be used in plain loops.

    Attributes:
        source: The model ID or URL it was loaded from.
        vocab_size: The number of tokens in the vocabulary.
    """

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
        """The token ids of a text.

        Args:
            text: The text to encode.
            add_special_tokens: Add the tokenizer's special tokens, such as a start or end token. The default is
                ``False``, unlike in Transformers.js.

        Returns:
            A list of ints. Nothing is truncated.

        Raises:
            TypeError: ``text`` is not a ``str``.
            RuntimeError: The tokenizer was closed.
        """
        import json

        if not isinstance(text, str):
            raise TypeError(f"encode() expects a str, got {type(text).__name__}")
        return json.loads(_to_python_deep(self._handle().encode(text, bool(add_special_tokens))))

    def decode(self, ids):
        """The text of a sequence of token ids, special tokens included.

        Args:
            ids: The token ids.

        Returns:
            The decoded string.
        """
        import json

        return _to_python_deep(self._handle().decode(json.dumps([int(i) for i in ids])))

    def close(self):
        """Release the loaded tokenizer.

        Later ``encode()`` and ``decode()`` calls raise. Calling it again is safe.
        """
        self._js = None


async def tokenizer(model=None, *, tokenizer_json=None, tokenizer_config=None, hosted_models_url=None):
    """Load only a tokenizer, with no model weights and no inference pipeline.

    Args:
        model: A model ID or name. Its ``tokenizer.json`` (and ``tokenizer_config.json`` when there is one) come from
            the hosted ``files/models/`` snapshot first, then from the Hub, as for ``LLM.create()``. Pass this or
            ``tokenizer_json``, not both.
        tokenizer_json: An HTTP(S) URL, or a path relative to the JupyterLite site root such as
            ``"files/models/org/name/tokenizer.json"``. Paths in the kernel's virtual file system are not supported.
        tokenizer_config: The URL or path of a ``tokenizer_config.json``, only with ``tokenizer_json``.
        hosted_models_url: An explicit URL of a hosted models directory, as for ``LLM.create()``.

    Returns:
        A ``Tokenizer``. Each call loads an independent handle.

    Raises:
        ValueError: The arguments are wrong (neither or both of ``model`` and ``tokenizer_json``, or a config
            with ``model``).
        RuntimeError: The kernel cannot call ``encode()`` synchronously.

    Example:
        .. code-block:: python

            tok = await tokenizer(model="minishlab/potion-multilingual-128M")
            tok.encode("sushi right next door to a pharmacy")
            tok.close()
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
