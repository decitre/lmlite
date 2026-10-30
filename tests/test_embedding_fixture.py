# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path


def test_bundled_embedding_fixture_integrity_and_size():
    root = Path(__file__).parent / "fixtures/models/lmlite/tiny-bert"
    manifest = json.loads((root / "fixture.json").read_text())
    assert manifest["license"] == "Apache-2.0"
    assert manifest["seed"] == 2026
    assert manifest["hidden_size"] == 16
    assert set(manifest["assets"]) == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "onnx/model.onnx",
        "onnx/model_quantized.onnx",
    }
    total = 0
    for name, asset in manifest["assets"].items():
        data = (root / name).read_bytes()
        assert len(data) == asset["size"]
        assert hashlib.sha256(data).hexdigest() == asset["sha256"]
        total += len(data)
    assert total < 100_000, "Keep the bundled model small enough for fast repository checkouts"
    config = json.loads((root / "config.json").read_text())
    assert config["hidden_size"] == manifest["hidden_size"]
    assert config["num_hidden_layers"] == 1


def test_tokenizer_fixture_matches_its_reference_ids():
    root = Path(__file__).parent / "fixtures/models/lmlite/tiny-unigram"
    manifest = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    # Editing tokenizer.json without rerunning the generator would leave stale expected ids.
    for name, asset in manifest["assets"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == asset["sha256"], name
    texts = [case["text"] for case in manifest["cases"]]
    assert "" in texts and any(len(text) > 1000 for text in texts)
