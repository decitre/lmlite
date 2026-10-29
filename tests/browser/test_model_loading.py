# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser smoke test for loading a real Transformers.js model."""

import json
import os

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

MODEL_ID = os.getenv("LMLITE_TEST_MODEL_ID", "Xenova/gpt2")
BACKEND = os.getenv("LMLITE_TEST_BACKEND", "wasm")
MODEL_TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "900000"))


def test_transformers_model_loads_and_generates(source_server):
    """Load and run a model through the same browser runtime used by lmlite."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(MODEL_TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")

            result = page.evaluate(
                """async ({modelId, backend}) => {
                  const llm = globalThis.lmLite.createLLM({
                    generator_model: modelId,
                    backend,
                    max_new_tokens: 1,
                    do_sample: false,
                  });
                  try {
                    await llm.init();
                    const generated = await llm.generate("The sky is");
                    return {ok: true, generated};
                  } catch (error) {
                    return {ok: false, error: String(error?.stack ?? error)};
                  }
                }""",
                {"modelId": MODEL_ID, "backend": BACKEND},
            )
        finally:
            browser.close()

    assert result["ok"], f"Model {MODEL_ID!r} failed to load or generate: {result.get('error')}"
    assert result["generated"].strip()


def test_hosted_config_loads_directly_over_http(source_server):
    """Exercise Transformers.js's real HTTP local-model loader without downloading weights."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(60000)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            page.route(
                "**/files/models/manifest.json",
                lambda route: route.fulfill(
                    body=json.dumps({"models": ["org/test-model"]}), content_type="application/json"
                ),
            )
            page.route(
                "**/files/models/org/test-model/config.json",
                lambda route: route.fulfill(
                    body=json.dumps({"model_type": "gpt2", "n_layer": 1}), content_type="application/json"
                ),
            )
            requests = []
            page.on("request", lambda request: requests.append(request.url))

            result = page.evaluate(
                """async (base) => {
                  const hosted = await globalThis.lmLite.getHostedModelBase('org/test-model', base);
                  await globalThis.lmLite.createLLM({
                    hosted_model_base: hosted,
                    hosted_models: ['org/test-model'],
                  }).init();
                  const { AutoConfig } = await import(
                    'https://cdn.jsdelivr.net/npm/@huggingface/transformers'
                  );
                  const progress = [];
                  const config = await AutoConfig.from_pretrained('org/test-model', {
                    local_files_only: true,
                    progress_callback: event => progress.push(event.status),
                  });
                  return {modelType: config.model_type, progress};
                }""",
                f"{source_server}/files/models/",
            )
        finally:
            browser.close()

    assert result["modelType"] == "gpt2"
    assert "progress" in result["progress"]
    assert f"{source_server}/files/models/org/test-model/config.json" in requests
    assert not any("huggingface.co/" in url or url.endswith(".zip") for url in requests)
