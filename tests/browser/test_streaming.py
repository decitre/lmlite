# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser check for chat streaming and cancellation (downloads a model from the Hub)."""

import os

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

MODEL_TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "900000"))
CHAT_MODEL = os.getenv("LMLITE_TEST_CHAT_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")


def test_chat_stream_and_cancel_in_browser(source_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(MODEL_TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            result = page.evaluate(
                """async (model) => {
                  const llm = globalThis.lmLite.createLLM({generator_model: model, max_new_tokens: 40, do_sample: false});
                  const messages = JSON.stringify([{role: 'user', content: 'Explain embeddings simply.'}]);
                  const readAll = async (stream, limit = Infinity) => {
                    const chunks = [];
                    for (let chunk; chunks.length < limit && (chunk = await stream.read());) chunks.push(chunk);
                    return chunks;
                  };
                  try {
                    await llm.init();
                    const full = await llm.chatStream(messages);
                    // Resume through a macrotask, as kernels resume Python, to catch event-loop starvation.
                    const first = await full.read();
                    await new Promise((resolve) => setTimeout(resolve, 0));
                    const streamedLive = llm._generation !== null;
                    const chunks = [first, ...await readAll(full)];
                    const answer = await llm.chat(messages);
                    const stream = await llm.chatStream(messages);
                    const head = await readAll(stream, 2);
                    const cancelled = await llm.cancel();
                    const tail = await readAll(stream);
                    const after = await llm.chat(messages);
                    await llm.dispose();
                    return {ok: true, streamedLive, chunks, answer, cancelled, partial: head.length + tail.length, after};
                  } catch (error) {
                    return {ok: false, error: String(error?.stack ?? error)};
                  }
                }""",
                CHAT_MODEL,
            )
        finally:
            browser.close()

    assert result["ok"], result.get("error")
    assert result["streamedLive"], "first chunk reached a macrotask consumer only after generation ended"
    assert len(result["chunks"]) > 1
    assert "".join(result["chunks"]).strip() == result["answer"]
    assert result["cancelled"] is True
    assert result["partial"] < len(result["chunks"])
    assert result["after"] == result["answer"]


def test_chat_tokens_report_probabilities_in_browser(source_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(MODEL_TIMEOUT)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            result = page.evaluate(
                """async (model) => {
                  const llm = globalThis.lmLite.createLLM({generator_model: model, max_new_tokens: 40, do_sample: false});
                  const messages = JSON.stringify([{role: 'user', content: 'Explain embeddings simply.'}]);
                  try {
                    await llm.init();
                    const stream = await llm.chatTokens(messages, 2);
                    const tokens = [];
                    for (let chunk; (chunk = await stream.read());) tokens.push(JSON.parse(chunk));
                    const answer = await llm.chat(messages);
                    await llm.dispose();
                    return {ok: true, tokens, answer};
                  } catch (error) {
                    return {ok: false, error: String(error?.stack ?? error)};
                  }
                }""",
                CHAT_MODEL,
            )
        finally:
            browser.close()

    assert result["ok"], result.get("error")
    tokens = result["tokens"]
    assert len(tokens) > 1
    assert "".join(t["text"] for t in tokens).strip() == result["answer"]
    for token in tokens:
        probs = [token["prob"], *(alt["prob"] for alt in token["alternatives"])]
        assert len(probs) == 3 and 0 < probs[0] <= 1
        # Greedy decoding picks the most probable token, so the list is descending.
        assert probs == sorted(probs, reverse=True) and sum(probs) <= 1 + 1e-6
