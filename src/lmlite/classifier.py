# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Text classification."""

from ._bridge import _to_python_deep
from ._session import Session


async def _classify(session, text, top_k=None):
    if top_k is None:
        result = await session.js.classify(text)
    else:
        result = await session.js.classify(text, top_k)
    return _to_python_deep(result)


class Classifier(Session):
    """Text classification.

    Load it with ``await Classifier.create(model)``. The labels are the model's own: a sentiment model such as
    ``distilbert-base-uncased-finetuned-sst-2-english`` has POSITIVE and NEGATIVE and no neutral class.

    Example:
        .. code-block:: python

            classifier = await Classifier.create("distilbert-base-uncased-finetuned-sst-2-english")
            await classifier.classify("I love using lmlite in JupyterLite!", top_k=2)
            await classifier.dispose()
    """

    @classmethod
    async def create(
        cls, model, *, backend="wasm", dtype=None, dtype_fallbacks=None, progress_callback=None, **options
    ):
        """Load a text classification model.

        Args:
            model: A Hugging Face model ID, or a short name, which gets the default ``Xenova/`` namespace.
            backend: ``"wasm"`` (the default) or ``"webgpu"``, or ``"native"`` for the onnxruntime backend outside a
                browser.
            dtype: The weights to load, such as ``"fp32"``. By default the first of ``q8``, ``int8`` and ``fp32`` that
                exists.
            dtype_fallbacks: Dtypes to try in order; takes precedence over ``dtype``.
            progress_callback: Called with a dict per loading event, as for ``LLM.create()``.
            **options: The other loading options of ``LLM.create()``: ``hosted_models_url``, ``auto_detect_local`` and
                ``local_models_path``, ``backend_fallback`` and ``quiet`` (no progress bar and no loading messages).

        Returns:
            The loaded ``Classifier``; call ``dispose()`` when you are done.
        """
        return await super().create(
            progress_callback,
            classification_model=model,
            backend=backend,
            dtype=dtype,
            dtype_fallbacks=dtype_fallbacks,
            **options,
        )

    async def classify(self, text: str, top_k=None):
        """Classify a text.

        Args:
            text: The text to classify.
            top_k: How many labels to return. By default, only the best one.

        Returns:
            A list of ``{"label": str, "score": float}`` dicts, the best first.
        """
        return await _classify(self, text, top_k)
