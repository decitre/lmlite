# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for Pipeline.map() (downloads models from the Hub)."""

import json
import os

import pytest
from kernels import run_in_pyodide, run_in_xeus
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "300000"))
ZERO_SHOT_MODEL = os.getenv("LMLITE_TEST_ZERO_SHOT_MODEL", "Xenova/mobilebert-uncased-mnli")
IMAGE_MODEL = os.getenv("LMLITE_TEST_IMAGE_CLASSIFICATION_MODEL", "Xenova/vit-base-patch16-224")
TIGER_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg"
TEXTS = [
    "The match ended in a draw",
    "A recipe for fresh pasta",
    "The new phone has a faster chip",
    "She scored in the final minute",
    "Whisk the eggs with sugar",
    "The laptop ships with more memory",
    "A late goal won the cup",
    "Simmer the sauce for ten minutes",
    "The update improves battery life",
    "He trained for the marathon",
]
LABELS = ["technology", "sports", "cooking"]


def in_page(source_server, script, args, timeout=TIMEOUT):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(timeout)
            page.goto(source_server, wait_until="domcontentloaded")
            page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
            return page.evaluate(script, args)
        finally:
            browser.close()


def test_batches_match_single_calls_report_progress_and_cancel(source_server):
    result = in_page(
        source_server,
        """async ({model, texts, labels}) => {
          await new Promise(resolve => setTimeout(resolve, 0));
          try {
            const llm = globalThis.lmLite.createLLM({task: 'zero-shot-classification', task_model: model, backend: 'wasm'});
            await llm.init();
            const options = {candidate_labels: labels};
            const single = [];
            for (const text of texts) {
              single.push(await llm.runTask(text, JSON.stringify(options)));
            }
            const progress = [];
            const batched = await llm.runTasks(texts, 4, options, (done, total) => progress.push([done, total]));
            // Cancel from the progress callback of the first batch: the run ends after that batch.
            const cancelled = await llm.runTasks(texts, 4, options, () => { llm.cancel(); });
            const after = await llm.runTasks(texts.slice(0, 2), 2, options);
            await llm.dispose();
            return {ok: true, single, batched, progress, cancelled, afterCount: after.results.length};
          } catch (error) {
            return {ok: false, error: String(error?.stack ?? error)};
          }
        }""",
        {"model": ZERO_SHOT_MODEL, "texts": TEXTS, "labels": LABELS},
    )

    assert result["ok"], result.get("error")
    assert result["progress"] == [[4, 10], [8, 10], [10, 10]]  # ceil(10 / 4) calls
    batched = result["batched"]["results"]
    assert len(batched) == len(TEXTS) and result["batched"]["errors"] == []
    for one, many in zip(result["single"], batched):
        assert one["labels"] == many["labels"]
        assert one["scores"] == pytest.approx(many["scores"], abs=1e-3)
    assert result["cancelled"]["cancelled"] is True and len(result["cancelled"]["results"]) == 4
    assert result["afterCount"] == 2  # the pipeline is still usable


def test_an_undecodable_image_fails_only_itself(source_server):
    result = in_page(
        source_server,
        """async ({model, url}) => {
          await new Promise(resolve => setTimeout(resolve, 0));
          try {
            const llm = globalThis.lmLite.createLLM({task: 'image-classification', task_model: model, backend: 'wasm'});
            await llm.init();
            const good = new Uint8Array(await (await fetch(url)).arrayBuffer());
            // The PNG signature passes Python's check, but the data is not an image.
            const corrupt = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]);
            const run = await llm.runTasks([good, corrupt, url], 3, {top_k: 1}, null, true);
            const raised = await llm.runTasks([good, corrupt], 2, {top_k: 1}).catch(error => String(error));
            await llm.dispose();
            return {ok: true, run, raised};
          } catch (error) {
            return {ok: false, error: String(error?.stack ?? error)};
          }
        }""",
        {"model": IMAGE_MODEL, "url": TIGER_URL},
    )

    assert result["ok"], result.get("error")
    run = result["run"]
    assert "tiger" in run["results"][0][0]["label"].lower() and run["results"][2] == run["results"][0]
    assert run["results"][1] is None and [index for index, _ in run["errors"]] == [1]
    assert "could not be decoded" in run["errors"][0][1]
    assert "could not be decoded" in result["raised"]


# Runs in a Python kernel: the same inputs one by one and with map(), through the bridge's progress callback.
KERNEL_CODE = r"""
import json
from lmlite import LLM

texts = TEXTS_JSON
labels = ["technology", "sports", "cooking"]
pipe = await LLM.pipeline("zero-shot-classification", model="MODEL")
single = [await pipe(text, candidate_labels=labels) for text in texts]
progress = []
batched = await pipe.map(texts, batch_size=4, progress_callback=lambda *args: progress.append(list(args)), candidate_labels=labels)
returned = await pipe.map([texts[0], 3, texts[1]], on_error="return", candidate_labels=labels)
await pipe.dispose()
print("RESULT" + json.dumps({
    "same_labels": [a["labels"] == b["labels"] for a, b in zip(single, batched)],
    "max_score_diff": max(abs(x - y) for a, b in zip(single, batched) for x, y in zip(a["scores"], b["scores"])),
    "count": len(batched),
    "progress": progress,
    "returned": [type(item).__name__ for item in returned],
}))
"""


def kernel_code():
    return KERNEL_CODE.replace("TEXTS_JSON", json.dumps(TEXTS)).replace("MODEL", ZERO_SHOT_MODEL)


def check_kernel_result(result):
    assert all(result["same_labels"]) and result["count"] == len(TEXTS)
    assert result["max_score_diff"] < 1e-3
    assert result["progress"] == [[4, 10], [8, 10], [10, 10]]
    assert result["returned"] == ["dict", "TypeError", "dict"]


def test_map_in_pyodide(source_server):
    check_kernel_result(run_in_pyodide(source_server, kernel_code()))


def test_map_in_xeus(xeus_site):
    check_kernel_result(run_in_xeus(xeus_site, kernel_code()))
