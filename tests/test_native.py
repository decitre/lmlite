# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Native (onnxruntime) backend checks; browser parity lives in tests/browser/test_native_parity.py."""

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

from lmlite import LLM

FIXTURES = Path(__file__).parent / "fixtures/models/lmlite"
GPT2 = Path(__file__).parents[1] / "models/Xenova/gpt2"
needs_native = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("onnxruntime", "tokenizers")), reason="needs lmlite[native]"
)


async def test_missing_native_extra_raises_actionable_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "js", raising=False)
    monkeypatch.setitem(sys.modules, "lmlite.native", None)
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'lmlite\[native\]'"):
        await LLM.create(embedding_model="any")


async def test_browser_only_calls_explain_the_native_limits(monkeypatch):
    monkeypatch.delitem(sys.modules, "js", raising=False)
    from lmlite import clear_cache

    with pytest.raises(ModuleNotFoundError, match="needs a browser kernel"):
        await clear_cache()


@needs_native
async def test_tokenizer_matches_reference_ids():
    # The browser tokenizer test asserts Transformers.js produces these same ids.
    root = FIXTURES / "tiny-unigram"
    manifest = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    tok = await LLM.tokenizer(
        tokenizer_json=str(root / "tokenizer.json"), tokenizer_config=str(root / "tokenizer_config.json")
    )
    assert tok.vocab_size == manifest["vocab_size"]
    for case in manifest["cases"]:
        assert tok.encode(case["text"]) == case["ids"], case["text"]
        assert tok.encode(case["text"], add_special_tokens=True) == case["ids_with_special_tokens"], case["text"]
    assert tok.decode(tok.encode(manifest["cases"][0]["text"])).strip() == manifest["cases"][0]["text"]
    assert (await LLM.tokenizer(str(root))).encode("sushi") == tok.encode("sushi")


@needs_native
@pytest.mark.parametrize("dtype", ["fp32", "q8"])
async def test_embed_is_pooled_normalized_and_uses_requested_weights(monkeypatch, dtype):
    monkeypatch.delitem(sys.modules, "js", raising=False)
    events = []
    llm = await LLM.create(
        embedding_model=str(FIXTURES / "tiny-bert"), embedding_dtype=dtype, progress_callback=events.append
    )
    assert llm.js.embedder.path.endswith("model.onnx" if dtype == "fp32" else "model_quantized.onnx")
    short, long = await llm.embed("Hello."), await llm.embed("A considerably longer sentence to embed.")
    for vector in (short, long):
        assert len(vector) == 16
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0, abs=1e-5)
    assert short != long
    assert events[-1]["status"] == "ready"
    with pytest.raises(NotImplementedError, match="native backend"):
        await llm.chat([{"role": "user", "content": "hi"}])


@needs_native
async def test_explicit_native_backend_and_models_path_lookup():
    llm = await LLM.create(
        backend="native",
        embedding_model="lmlite/tiny-bert",
        local_models_path=str(FIXTURES.parent),
        embedding_dtype="q8",
    )
    assert len(await llm.embed("Hello.")) == 16


@needs_native
async def test_classify_softmax_top_k_and_labels():
    import numpy as np

    from lmlite.native import NativeLLM

    llm = NativeLLM({})
    llm.classifier = types.SimpleNamespace(
        config={"id2label": {"0": "neg", "1": "pos", "2": "mixed"}},
        tokenizer=types.SimpleNamespace(ids=lambda text, truncate: [1, 2]),
        encoder_feed=lambda ids: {},
        run=lambda feed: {"logits": np.array([[0.0, 2.0, 1.0]], np.float32)},
    )
    wrapped = LLM(llm)
    top = await wrapped.classify("x")
    assert [r["label"] for r in top] == ["pos"] and top[0]["score"] == pytest.approx(0.66524, abs=1e-5)
    every = await wrapped.classify("x", top_k=3)
    assert [r["label"] for r in every] == ["pos", "mixed", "neg"]
    assert sum(r["score"] for r in every) == pytest.approx(1.0, abs=1e-6)
    llm.classifier.config = {"problem_type": "multi_label_classification"}
    assert (await wrapped.classify("x"))[0] == {"label": "LABEL_1", "score": pytest.approx(1 / (1 + math.exp(-2)))}


@needs_native
@pytest.mark.skipif(not (GPT2 / "onnx/model_int8.onnx").is_file(), reason="needs `pixi run hf Xenova/gpt2`")
async def test_greedy_generation_is_deterministic():
    llm = await LLM.create(
        backend="native", generator_model=str(GPT2), do_sample=False, max_new_tokens=8, return_full_text=False
    )
    assert llm.js.generator.dtype == "int8"  # no model_quantized.onnx locally: falls back like llm.js
    first = await llm.generate("The capital of France is")
    assert first and first == await llm.generate("The capital of France is")
