# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser check of a site built with `scripts/lite-build.sh -m`: Transformers.js, onnxruntime-web and the
model all load from the site, with every request to another host blocked."""

import functools
import http.server
import os
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC = Path(os.getenv("LMLITE_PUBLIC", REPO_ROOT / "workspace/public"))
MODEL_ID = os.getenv("LMLITE_TEST_MODEL_ID", "Xenova/gpt2")
MODEL_TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "900000"))


@pytest.fixture(scope="module")
def public_server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("site", ["pyodide", "xeus"])
def test_model_generates_without_other_hosts(public_server, site):
    if not (PUBLIC / site / "files/models" / MODEL_ID / "config.json").is_file():
        pytest.skip(f"no {MODEL_ID} hosted in {PUBLIC / site}; build with scripts/lite-build.sh -m")
    external = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(MODEL_TIMEOUT)

            def block_external(route):
                if route.request.url.startswith(public_server):
                    route.continue_()
                else:
                    external.append(route.request.url)
                    route.abort()

            page.route("**/*", block_external)
            # An empty page under lab/, so llm.js finds the site root as it does in JupyterLite.
            page.route(
                f"{public_server}/{site}/lab/lmlite-test.html", lambda route: route.fulfill(body="<html></html>")
            )
            page.goto(f"{public_server}/{site}/lab/lmlite-test.html")
            page.add_script_tag(path=REPO_ROOT / "src/lmlite/llm.js", type="module")
            page.wait_for_function("globalThis.lmLite")

            result = page.evaluate(
                """async (modelId) => {
                  try {
                    const hosted = await globalThis.lmLite.getHostedModelBase(modelId);
                    const llm = globalThis.lmLite.createLLM({
                      generator_model: modelId,
                      hosted_model_base: hosted,
                      hosted_models: [modelId],
                      max_new_tokens: 1,
                      do_sample: false,
                    });
                    await llm.init();
                    return {ok: true, hosted, generated: await llm.generate("The sky is")};
                  } catch (error) {
                    return {ok: false, error: String(error?.stack ?? error)};
                  }
                }""",
                MODEL_ID,
            )
        finally:
            browser.close()

    assert result["ok"], (result.get("error"), external)
    assert result["hosted"] == f"{public_server}/{site}/files/models/"
    assert result["generated"].strip()
    assert external == []
