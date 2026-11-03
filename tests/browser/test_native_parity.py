# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Parity gate: the native onnxruntime backend must match the browser on the same ONNX files.

Embeddings must agree within EPSILON and an embedding margin gate must make identical accept/reject
decisions on a calibration set. Set LMLITE_TEST_EMBEDDING_MODEL (a model under ./models, as in
test_embeddings.py) and LMLITE_PARITY_MARGIN to check a real model at its tuned threshold.
"""

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest
from playwright.sync_api import sync_playwright

pytest.importorskip("onnxruntime")

from lmlite import LLM  # noqa: E402

pytestmark = pytest.mark.browser

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL_ID = "lmlite/tiny-bert"
MODEL_ID = os.getenv("LMLITE_TEST_EMBEDDING_MODEL", TINY_MODEL_ID)
MODEL_BASE = "tests/fixtures/models" if MODEL_ID == TINY_MODEL_ID else "models"
GPT2_DIR = REPO_ROOT / "models/Xenova/gpt2"
MARGIN = float(os.getenv("LMLITE_PARITY_MARGIN", "0.02"))
# Max L2 distance between unit vectors from ONNX Runtime Web (wasm) and CPU. q8 is looser: ulp-level kernel
# differences flip runtime activation quantization steps (~4e-3 = 1-cos, L2 ~0.09, on multilingual-e5-small).
EPSILON = {"fp32": 1e-4, "fp16": 1e-3, "q8": 0.1}
MODEL_TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "900000"))

ANCHORS = {
    "food": ["sushi restaurant", "pizza delivery", "a bakery selling bread"],
    "health": ["a pharmacy", "the hospital emergency room", "dentist appointment"],
    "transport": ["train station", "airport terminal", "bus stop downtown"],
}
QUERIES = [
    "Hello.",
    "where can I buy medicine",
    "cheap noodles near me",
    "next flight to Paris",
    "sushi right next door to a pharmacy",
    "A considerably longer sentence with many more tokens to embed, to exercise attention over a longer input.",
    "¿Dónde está la estación?",
    "",
]

# Hosted fixture files are served locally; redirect Transformers.js metadata probes to them (see test_embeddings).
BUNDLED_FETCH = """
  const {env} = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
  const fetch = env.fetch;
  env.fetch = (url, options) => {
    const remote = new URL(String(url), location.href);
    const prefix = `/${modelId}/resolve/main/`;
    if (remote.hostname === 'huggingface.co' && remote.pathname.startsWith(prefix)
        && new Headers(options?.headers).get('range') === 'bytes=0-0') {
      return fetch(new URL(`${modelId}/${remote.pathname.slice(prefix.length)}`, base), options);
    }
    return fetch(url, options);
  };
"""


def browser_run(source_server, model_base, model_id, config, inputs, body, bundled=False):
    """Run ``body`` (JS with ``llm`` initialized and ``inputs``) in Chromium and return its value."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            page.wait_for_function("!!globalThis.lmLite?.createLLM")
            return page.evaluate(
                f"""async ({{base, modelId, config, inputs}}) => {{
                  {BUNDLED_FETCH if bundled else ""}
                  const llm = globalThis.lmLite.createLLM({{...config, hosted_model_base: base, hosted_models: [modelId]}});
                  await llm.init();
                  try {{ {body} }} finally {{ await llm.dispose(); }}
                }}""",
                {"base": f"{source_server}/{model_base}/", "modelId": model_id, "config": config, "inputs": inputs},
            )
        finally:
            browser.close()


async def native_run(model_base, config, method, inputs):
    llm = await LLM.create(backend="native", local_models_path=str(REPO_ROOT / model_base), **config)
    return [await getattr(llm, method)(item) for item in inputs]


def gate(vectors, labels):
    """Nearest-anchor label per query and whether its margin over the next label clears MARGIN."""
    anchors = np.array(vectors[: len(labels)])
    decisions = []
    for query in np.array(vectors[len(labels) :]):
        best = {}
        for label, score in zip(labels, anchors @ query):
            best[label] = max(best.get(label, -np.inf), score)
        (top, top_score), (_, second) = sorted(best.items(), key=lambda item: -item[1])[:2]
        decisions.append((top, bool(top_score - second >= MARGIN), float(top_score - second)))
    return decisions


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "q8"])
def test_native_embeddings_and_gate_match_browser(source_server, dtype):
    if dtype == "fp16" and not (REPO_ROOT / MODEL_BASE / MODEL_ID / "onnx/model_fp16.onnx").is_file():
        pytest.skip(f"{MODEL_ID} has no fp16 weights")
    labels = [label for label, texts in ANCHORS.items() for _ in texts]
    texts = [text for group in ANCHORS.values() for text in group] + QUERIES
    config = {"embedding_model": MODEL_ID, "embedding_dtype": dtype, "embedding_backend": "wasm"}
    browser = browser_run(
        source_server,
        MODEL_BASE,
        MODEL_ID,
        config,
        texts,
        "const out = []; for (const t of inputs) out.push(await llm.embed(t)); return out;",
        bundled=MODEL_ID == TINY_MODEL_ID,
    )
    native = asyncio.run(native_run(MODEL_BASE, config, "embed", texts))

    distances = np.linalg.norm(np.array(native) - np.array(browser), axis=1)
    assert distances.max() <= EPSILON[dtype], dict(zip(texts, distances.round(7)))
    browser_gate, native_gate = gate(browser, labels), gate(native, labels)
    assert [d[:2] for d in native_gate] == [d[:2] for d in browser_gate], list(zip(QUERIES, browser_gate, native_gate))


@pytest.mark.skipif(not (GPT2_DIR / "onnx/model.onnx").is_file(), reason="needs `pixi run hf Xenova/gpt2`")
def test_native_greedy_generation_matches_browser(source_server):
    prompts = ["The capital of France is", "Once upon a time"]
    # fp32: int8 logits drift ~1 between runtimes (runtime activation quantization), enough to flip greedy near-ties.
    config = {
        "generator_model": "Xenova/gpt2",
        "dtype": "fp32",
        "do_sample": False,
        "max_new_tokens": 12,
        "return_full_text": True,
    }
    browser = browser_run(
        source_server,
        "models",
        "Xenova/gpt2",
        config,
        prompts,
        "const out = []; for (const p of inputs) out.push(await llm.generate(p)); return out;",
    )
    assert asyncio.run(native_run("models", config, "generate", prompts)) == browser
