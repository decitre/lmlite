# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Measure browser (Chromium, ONNX Runtime Web wasm) vs native embedding drift, dtype error and latency.

Models are read from ./models/<id>/ (config, tokenizer and onnx/model{,_fp16,_quantized}.onnx).
Native fp32 is the error reference: it matches browser fp32 to ~1e-12. Example:

    pixi run python scripts/native_parity_report.py Xenova/all-MiniLM-L6-v2 --dtypes q8 fp16 fp32
"""

import argparse
import asyncio
import functools
import http.server
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from lmlite import LLM

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "Xenova/all-MiniLM-L6-v2",
    "Xenova/bge-small-en-v1.5",
    "Xenova/multilingual-e5-small",
    "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    "Xenova/all-mpnet-base-v2",
]
WARMUP = 3
# ~250 tokens: latency of a passage-length input, where dtype kernels matter more than call overhead.
LONG = " ".join(["The city council approved a new plan to extend the tram line to the northern suburbs."] * 14)
LONG_RUNS = 5


class Handler(http.server.SimpleHTTPRequestHandler):
    isolated = False

    def end_headers(self):
        if self.isolated:  # crossOriginIsolated lets ONNX Runtime Web use wasm threads
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def log_message(self, *args):
        pass


def browser_embed(page, origin, model, dtype, texts):
    page.goto(origin, wait_until="domcontentloaded")
    page.add_script_tag(url=f"{origin}/src/lmlite/llm.js", type="module")
    page.wait_for_function("!!globalThis.lmLite?.createLLM")
    return page.evaluate(
        """async ({base, model, dtype, texts, warmup, long, runs}) => {
          const llm = globalThis.lmLite.createLLM({embedding_model: model, embedding_dtype: dtype,
            embedding_backend: 'wasm', hosted_model_base: base, hosted_models: [model]});
          let t = performance.now();
          await llm.init();
          const init_ms = performance.now() - t;
          try {
            for (const text of texts.slice(0, warmup)) await llm.embed(text);
            const vectors = [], times = [];
            for (const text of texts) {
              t = performance.now();
              vectors.push(await llm.embed(text));
              times.push(performance.now() - t);
            }
            const long_times = [];
            for (let i = 0; i < runs; i++) {
              t = performance.now();
              await llm.embed(long);
              long_times.push(performance.now() - t);
            }
            return {init_ms, vectors, times, long_times, threads: globalThis.crossOriginIsolated ? 'multi' : 1};
          } finally { await llm.dispose(); }
        }""",
        {
            "base": f"{origin}/models/",
            "model": model,
            "dtype": dtype,
            "texts": texts,
            "warmup": WARMUP,
            "long": LONG,
            "runs": LONG_RUNS,
        },
    )


async def native_embed(model, dtype, texts):
    start = time.perf_counter()
    llm = await LLM.create(
        backend="native", local_models_path=str(REPO_ROOT / "models"), embedding_model=model, embedding_dtype=dtype
    )
    init_ms = (time.perf_counter() - start) * 1000
    for text in texts[:WARMUP]:
        await llm.embed(text)
    vectors, times = [], []
    for text in texts:
        start = time.perf_counter()
        vectors.append(await llm.embed(text))
        times.append((time.perf_counter() - start) * 1000)
    long_times = []
    for _ in range(LONG_RUNS):
        start = time.perf_counter()
        await llm.embed(LONG)
        long_times.append((time.perf_counter() - start) * 1000)
    return {"init_ms": init_ms, "vectors": vectors, "times": times, "long_times": long_times}


def one_minus_cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return 1 - (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def p(values, q):
    return float(np.percentile(values, q))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--dtypes", nargs="+", default=["q8", "fp16", "fp32"])
    parser.add_argument("--texts", default=str(REPO_ROOT / "tests/fixtures/parity/texts.json"))
    parser.add_argument("--isolated", action="store_true", help="serve with COOP/COEP so wasm runs multithreaded")
    parser.add_argument("--json", help="also write the full report (without vectors) to this path")
    args = parser.parse_args()
    texts = json.loads(Path(args.texts).read_text(encoding="utf-8"))

    Handler.isolated = args.isolated
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(REPO_ROOT)))
    server.handle_error = lambda *args: None  # the browser aborts range requests mid-transfer; not an error
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    # e5 models expect a task prefix; texts that already carry one are left alone.
    inputs = {
        model: [t if "e5" not in model or t.startswith(("query: ", "passage: ")) else f"query: {t}" for t in texts]
        for model in args.models
    }
    web = {}
    try:
        with sync_playwright() as playwright:  # its event loop blocks asyncio.run(), so native runs after
            browser = playwright.chromium.launch()
            try:
                for model in args.models:
                    for dtype in args.dtypes:
                        page = browser.new_page()
                        try:
                            web[model, dtype] = browser_embed(page, origin, model, dtype, inputs[model])
                        finally:
                            page.close()
                        print(f"browser {model} {dtype}: {statistics.median(web[model, dtype]['times']):.1f} ms")
            finally:
                browser.close()
    finally:
        server.shutdown()

    rows = []
    for model in args.models:
        reference = asyncio.run(native_embed(model, "fp32", inputs[model]))["vectors"]
        for dtype in args.dtypes:
            native = asyncio.run(native_embed(model, dtype, inputs[model]))
            w = web[model, dtype]
            drift = one_minus_cos(w["vectors"], native["vectors"])
            error = one_minus_cos(native["vectors"], reference)
            r = {
                "model": model,
                "dtype": dtype,
                "drift_max": float(drift.max()),
                "drift_p99": p(drift, 99),
                "drift_median": float(np.median(drift)),
                "drift_over_1e-4": int((drift > 1e-4).sum()),
                "error_vs_fp32_mean": float(error.mean()),
                "error_vs_fp32_max": float(error.max()),
                "browser_init_ms": w["init_ms"],
                "browser_ms_median": statistics.median(w["times"]),
                "browser_ms_p90": p(w["times"], 90),
                "browser_threads": w["threads"],
                "native_ms_median": statistics.median(native["times"]),
                "browser_long_ms": statistics.median(w["long_times"]),
                "native_long_ms": statistics.median(native["long_times"]),
                "texts": len(texts),
            }
            rows.append(r)
            print(
                f"{model:46s} {dtype:5s} drift max {r['drift_max']:.2g} p99 {r['drift_p99']:.2g} "
                f"(>1e-4: {r['drift_over_1e-4']}/{len(texts)})  err vs fp32 mean "
                f"{r['error_vs_fp32_mean']:.2g} max {r['error_vs_fp32_max']:.2g}  browser "
                f"{r['browser_ms_median']:.1f} ms (p90 {r['browser_ms_p90']:.1f}, init "
                f"{r['browser_init_ms'] / 1000:.1f} s, long {r['browser_long_ms']:.0f} ms)  native "
                f"{r['native_ms_median']:.1f} ms (long {r['native_long_ms']:.0f} ms)",
                flush=True,
            )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
