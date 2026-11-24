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
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")

            result = page.evaluate(
                """async ({modelId, backend}) => {
                  const llm = globalThis.lmLite.createRuntime({
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
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
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
                  await globalThis.lmLite.createRuntime({
                    hosted_model_base: hosted,
                    hosted_models: ['org/test-model'],
                  }).init();
                  const { AutoConfig } = await import(
                    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2'
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


def discover_hosted_base(source_server, manifest_answers):
    """Call getHostedModelBase('gpt2') on a page under lab/ (so the site's manifest is looked up on its own), with
    the manifest route answering from `manifest_answers` in turn, the last answer repeating. Returns (result, calls)."""
    page_url = f"{source_server}/lab/lmlite-test.html"
    calls = []
    warnings = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(60000)
            page.on("console", lambda message: warnings.append(message.text) if message.type == "warning" else None)

            def answer(route):
                status, body = manifest_answers[min(len(calls), len(manifest_answers) - 1)]
                calls.append(status)
                route.fulfill(status=status, body=body, content_type="application/json")

            page.route("**/files/models/manifest.json", answer)
            page.route(page_url, lambda route: route.fulfill(body="<html></html>", content_type="text/html"))
            page.goto(page_url)
            page.add_script_tag(url=f"{source_server}/src/lmlite/lmlite.js", type="module")
            page.wait_for_function("typeof globalThis.lmLite === 'object'")
            result = page.evaluate("async () => await globalThis.lmLite.getHostedModelBase('gpt2')")
        finally:
            browser.close()
    return result, calls, warnings


def test_a_flaky_site_manifest_does_not_break_hosted_discovery(source_server):
    """GitHub Pages answers 503 now and then: the probe retries, and the model is still found when it is hosted."""
    listed = json.dumps({"models": ["Xenova/gpt2"]})

    result, calls, warnings = discover_hosted_base(source_server, [(503, ""), (503, ""), (200, listed)])

    assert result == f"{source_server}/files/models/"
    assert calls == [503, 503, 200]
    assert not warnings


def test_an_unavailable_site_manifest_means_nothing_is_hosted(source_server):
    """A manifest that stays 503 must not make LLM.pipeline() fail: the model then loads from the Hub."""
    result, calls, warnings = discover_hosted_base(source_server, [(503, "")])

    assert result is None and calls == [503, 503, 503]
    assert any("loading from the Hub" in warning for warning in warnings)


def test_a_missing_site_manifest_is_asked_for_once(source_server):
    result, calls, warnings = discover_hosted_base(source_server, [(404, "")])

    assert result is None and calls == [404] and not warnings
