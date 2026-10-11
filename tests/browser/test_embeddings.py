# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in inference checks against tiny or representative local ONNX weights."""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

TINY_MODEL_ID = "lmlite/tiny-bert"
MODEL_ID = os.getenv("LMLITE_TEST_EMBEDDING_MODEL", TINY_MODEL_ID)
MODEL_DIMENSIONS = {TINY_MODEL_ID: 16, "Xenova/all-MiniLM-L6-v2": 384}
MODEL_BASE = "tests/fixtures/models" if MODEL_ID == TINY_MODEL_ID else "models"
MODEL_DIR = Path(__file__).resolve().parents[2] / MODEL_BASE / MODEL_ID
BACKEND = os.getenv("LMLITE_TEST_EMBEDDING_BACKEND", "wasm")
MODEL_TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "900000"))


@pytest.mark.parametrize("dtype,weights", [("fp32", "model.onnx"), ("q8", "model_quantized.onnx")])
def test_local_embeddings_are_pooled_normalized_and_use_requested_weights(source_server, dtype, weights):
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", f"onnx/{weights}"]
    missing = [name for name in required if not (MODEL_DIR / name).is_file()]
    assert not missing, (
        f"Missing local assets for {MODEL_ID}: {missing}. "
        "Download them with the embedding-smoke instructions in README.md."
    )
    assert MODEL_ID in MODEL_DIMENSIONS, f"Unsupported smoke model: {MODEL_ID}"
    if MODEL_ID == TINY_MODEL_ID:
        manifest = json.loads((MODEL_DIR / "fixture.json").read_text())
        for name, metadata in manifest["assets"].items():
            digest = hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest()
            assert digest == metadata["sha256"], f"Fixture hash mismatch: {name}"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            requests = []
            responses = []
            failed_requests = []
            hub_probes = []

            def record_request(request):
                requests.append(request.url)
                if request.url.startswith("https://huggingface.co/") and request.headers.get("range") == "bytes=0-0":
                    hub_probes.append(request.url)

            page.on("request", record_request)
            page.on("response", lambda response: responses.append((response.url, response.status)))
            page.on("requestfailed", lambda request: failed_requests.append((request.url, request.failure)))

            def handle_hub_request(route):
                # Current Transformers.js probes metadata remotely even for HTTP-hosted models.
                if route.request.headers.get("range") == "bytes=0-0":
                    route.continue_()
                else:
                    route.abort()

            page.route("https://huggingface.co/**", handle_hub_request)
            page.evaluate(
                """async ({base, modelId, backend, dtype, bundled}) => {
                  if (bundled) {
                    const {env} = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
                    const fetch = env.fetch;
                    // The upstream metadata helper does not probe HTTP localModelPath.
                    // Redirect its metadata probes to actual bundled assets, not mock responses.
                    env.fetch = (url, options) => {
                      const remote = new URL(String(url), location.href);
                      const prefix = `/${modelId}/resolve/main/`;
                      if (remote.hostname === 'huggingface.co' && remote.pathname.startsWith(prefix)
                          && new Headers(options?.headers).get('range') === 'bytes=0-0') {
                        return fetch(new URL(`${modelId}/${remote.pathname.slice(prefix.length)}`, base), options);
                      }
                      return fetch(url, options);
                    };
                  }
                  globalThis.embeddingSmoke = {done: false};
                  (async () => {
                    const llm = globalThis.createLLM({
                      embedding_model: modelId,
                      embedding_backend: backend,
                      embedding_dtype: dtype,
                      hosted_model_base: base,
                      hosted_models: [modelId],
                    });
                    try {
                      await llm.init();
                      const vectors = [];
                      const references = [];
                      for (const text of [
                        "Hello.",
                        "A considerably longer sentence with many more tokens to embed.",
                      ]) {
                        vectors.push(await llm.embed(text));
                        const reference = await llm.embedder(text, {pooling: 'mean', normalize: true});
                        references.push(Array.from(reference.data));
                      }
                      await llm.dispose();
                      if (llm.embedder !== null) throw new Error('Disposed embedder reference was not cleared');
                      await llm.dispose();
                      let rejected = false;
                      try { await llm.embed('Hello.'); } catch (error) {
                        rejected = String(error.message).includes('No embedding model configured');
                      }
                      if (!rejected) throw new Error('Embedding after disposal must fail clearly');
                      await llm.init();
                      const reloaded = await llm.embed('Hello.');
                      await llm.dispose();
                      globalThis.embeddingSmoke = {done: true, vectors, references, reloaded};
                    } catch (error) {
                      globalThis.embeddingSmoke = {done: true, error: String(error?.stack ?? error)};
                    }
                  })();
                }""",
                {
                    "base": f"{source_server}/{MODEL_BASE}/",
                    "modelId": MODEL_ID,
                    "backend": BACKEND,
                    "dtype": dtype,
                    "bundled": MODEL_ID == TINY_MODEL_ID,
                },
            )
            page.wait_for_function("globalThis.embeddingSmoke.done", timeout=MODEL_TIMEOUT)
            result = page.evaluate("globalThis.embeddingSmoke")
        finally:
            browser.close()

    assert "error" not in result, (
        f"{MODEL_ID} ({BACKEND}, {dtype}) failed: {result.get('error')}. Failed requests: {failed_requests}"
    )
    for vector, reference in zip(result["vectors"], result["references"], strict=True):
        assert len(vector) == MODEL_DIMENSIONS[MODEL_ID], (
            "embed() must return one sentence vector, not flattened token states"
        )
        assert all(isinstance(value, (float, int)) and math.isfinite(value) for value in vector)
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0, abs=1e-5)
        assert vector == pytest.approx(reference, abs=1e-5), "embed() must use mean pooling, not another pooling method"

    expected_url = f"{source_server}/{MODEL_BASE}/{MODEL_ID}/onnx/{weights}"
    assert result["reloaded"] == pytest.approx(result["vectors"][0], abs=1e-5)
    assert (expected_url, 200) in responses, "The requested dtype's real ONNX weights must be loaded"
    for asset in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        assert (f"{source_server}/{MODEL_BASE}/{MODEL_ID}/{asset}", 200) in responses
    onnx_requests = [url for url in requests if url not in hub_probes and url.split("?", 1)[0].endswith(".onnx")]
    assert set(onnx_requests) == {expected_url}, "Do not load or fall back to another ONNX dtype"
    unexpected_hub_requests = [
        url for url in requests if url.startswith("https://huggingface.co/") and url not in hub_probes
    ]
    assert not unexpected_hub_requests, f"Non-metadata Hub requests: {unexpected_hub_requests}"
