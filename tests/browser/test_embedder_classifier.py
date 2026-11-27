# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for Embedder and Classifier in Pyodide and xeus-python (they download models from the Hub),
and that LLM no longer loads them."""

import pytest
from kernels import run_in_pyodide, run_in_xeus

pytestmark = pytest.mark.browser

KERNEL_CODE = r"""
import json, math
from lmlite import LLM, Classifier, Embedder

docs = ["JupyterLite runs entirely in the browser.", "Python is widely used for machine learning.", "Bananas are yellow."]
embedder = await Embedder.create("Xenova/all-MiniLM-L6-v2", backend="wasm", dtype="q8")
vector = await embedder.embed("Where does JupyterLite run?")
ranked = await embedder.similarity_search("Where does JupyterLite run?", docs, k=2)
await embedder.dispose()

classifier = await Classifier.create("Xenova/distilbert-base-uncased-finetuned-sst-2-english", backend="wasm")
labels = await classifier.classify("I love this, it is wonderful!", top_k=2)
await classifier.dispose()

try:
    await LLM.create(embedding_model="Xenova/all-MiniLM-L6-v2")
    refused = None
except TypeError as error:
    refused = str(error)

print("RESULT" + json.dumps({
    "dimensions": len(vector),
    "norm": math.sqrt(sum(v * v for v in vector)),
    "ranked": [doc for doc, _ in ranked],
    "labels": [item["label"] for item in labels],
    "refused": refused,
}))
"""


def check(result):
    assert result["dimensions"] == 384 and abs(result["norm"] - 1) < 1e-4
    assert result["ranked"][0] == "JupyterLite runs entirely in the browser." and len(result["ranked"]) == 2
    assert result["labels"][0] == "POSITIVE" and len(result["labels"]) == 2
    assert result["refused"] and "Embedder.create()" in result["refused"]  # LLM.create no longer loads an embedder


def test_embedder_and_classifier_in_pyodide(source_server):
    check(run_in_pyodide(source_server, KERNEL_CODE, packages=("tqdm",)))


def test_embedder_and_classifier_in_xeus(xeus_site):
    check(run_in_xeus(xeus_site, KERNEL_CODE))
