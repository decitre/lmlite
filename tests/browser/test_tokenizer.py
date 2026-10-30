# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Tokenize-only handles against reference ids from the Rust `tokenizers` library."""

import json
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

MODEL_ID = "lmlite/tiny-unigram"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures/models" / MODEL_ID
FIXTURE = json.loads((FIXTURE_DIR / "fixture.json").read_text(encoding="utf-8"))
PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js"
TOKENIZER = json.loads((FIXTURE_DIR / "tokenizer.json").read_text(encoding="utf-8"))
UNSUPPORTED = {**TOKENIZER, "model": {"type": "Foo"}}
WEIGHTS = (".onnx", ".onnx_data", ".safetensors", ".bin")


@pytest.fixture
def page(source_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(120000)
            page.goto(source_server, wait_until="domcontentloaded")
            page.route(
                "**/fixtures/models/manifest.json",
                lambda route: route.fulfill(body=json.dumps({"models": [MODEL_ID]}), content_type="application/json"),
            )
            page.route(
                "**/fixtures/models/org/bad/tokenizer.json",
                lambda route: route.fulfill(body=json.dumps(UNSUPPORTED), content_type="application/json"),
            )
            page.requests = []
            page.on("request", lambda request: page.requests.append(request.url))
            yield page
        finally:
            browser.close()


def test_ids_match_reference_without_weights_or_hub(page, source_server):
    page.add_script_tag(url=f"{source_server}/src/lmlite/llm.js", type="module")
    page.wait_for_function("globalThis.lmLite")
    base = f"{source_server}/tests/fixtures/models/"
    result = page.evaluate(
        """async ({base, modelId, texts}) => {
          const encodeAll = (tok) => texts.map(t => [JSON.parse(tok.encode(t)), JSON.parse(tok.encode(t, true))]);
          const hosted = await globalThis.lmLite.loadTokenizer(modelId, null, null, base);
          const bare = await globalThis.lmLite.loadTokenizer(null, `${base}${modelId}/tokenizer.json`);
          let error = null;
          try { await globalThis.lmLite.loadTokenizer(null, `${base}org/bad/tokenizer.json`); } catch (e) { error = e.message; }
          const cached = [];
          for (const name of await caches.keys()) cached.push(...(await (await caches.open(name)).keys()));
          return {hosted: encodeAll(hosted), bare: encodeAll(bare), vocabSize: hosted.vocab_size,
                  decoded: hosted.decode(hosted.encode(texts[0])), error, cached: cached.length};
        }""",
        {"base": base, "modelId": MODEL_ID, "texts": [case["text"] for case in FIXTURE["cases"]]},
    )

    for name in ["hosted", "bare"]:  # bare: tokenizer.json alone gives the same ids
        for case, (ids, special) in zip(FIXTURE["cases"], result[name], strict=True):
            assert ids == case["ids"], (name, case["text"])
            assert special == case["ids_with_special_tokens"], (name, case["text"])
    assert any(case["ids"] != case["ids_with_special_tokens"] for case in FIXTURE["cases"])
    assert result["vocabSize"] == FIXTURE["vocab_size"]
    assert result["decoded"].strip() == FIXTURE["cases"][0]["text"]
    assert "org/bad/tokenizer.json'" in result["error"] and "Unknown TokenizerModel type: Foo" in result["error"]
    assert result["cached"] == 0, "Tokenizer files must not enter the Transformers.js Cache API"
    assert not [url for url in page.requests if "huggingface.co" in url]
    assert not [url for url in page.requests if url.split("?", 1)[0].endswith(WEIGHTS)]
    assert f"{base}{MODEL_ID}/tokenizer_config.json" in page.requests


def test_pyodide_encode_returns_plain_ints_synchronously(page, source_server):
    page.add_script_tag(url=PYODIDE_URL)
    case = FIXTURE["cases"][0]
    result = page.evaluate(
        """async ({server, text}) => {
          const pyodide = await loadPyodide();
          for (const name of ['__init__.py', 'llm.py', 'llm.js', 'version.py', 'bridge/__init__.py', 'bridge/runtime.py']) {
            pyodide.FS.mkdirTree('/lib/lmlite/bridge');
            pyodide.FS.writeFile(`/lib/lmlite/${name}`, await (await fetch(`${server}/src/lmlite/${name}`)).text());
          }
          pyodide.globals.set('server', server);
          pyodide.globals.set('text', text);
          return await pyodide.runPythonAsync(`
import json, sys, time
sys.path.insert(0, '/lib')
from lmlite import LLM

tok = await LLM.tokenizer('lmlite/tiny-unigram', hosted_models_url=f'{server}/tests/fixtures/models/')
ids = tok.encode(text)
# Browser timers are coarse; time a batch.
start = time.perf_counter()
for _ in range(1000):
    tok.encode(text)
mean_ms = time.perf_counter() - start
tok.close()
tok.close()
try:
    tok.encode(text)
    closed = None
except RuntimeError as exc:
    closed = str(exc)
json.dumps({'ids': ids, 'list': type(ids) is list, 'ints': all(type(i) is int for i in ids),
            'mean_ms': mean_ms, 'closed': closed})
`);
        }""",
        {"server": source_server, "text": case["text"]},
    )
    result = json.loads(result)
    print(f"Pyodide encode mean: {result['mean_ms']:.4f} ms")
    assert result["list"] and result["ints"]
    assert result["ids"] == case["ids"]
    assert "closed" in result["closed"]
