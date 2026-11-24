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
    """Text classification: the model loaded by ``Classifier.create(model)``."""

    @classmethod
    async def create(
        cls, model, *, backend="wasm", dtype=None, dtype_fallbacks=None, progress_callback=None, **options
    ):
        """Load a text classification model.

        ``backend`` is ``"wasm"`` or ``"webgpu"``. ``dtype`` and ``dtype_fallbacks`` work as in ``LLM.create()``.
        ``options`` are its other loading options: ``hosted_models_url``, ``auto_detect_local`` and
        ``local_models_path``.
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
        """The labels and scores of ``text``, best first; ``top_k`` limits how many."""
        return await _classify(self, text, top_k)
