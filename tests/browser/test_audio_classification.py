# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for audio classification, with the instrument model (downloads it from the Hub).

The audio is synthetic (a tone with harmonics), so no recording is stored: the checks are about the bridge and the
result format, not about which instrument the model hears."""

import os

import pytest
from kernels import run_in_pyodide, run_in_xeus

pytestmark = pytest.mark.browser

MODEL = os.getenv("LMLITE_TEST_AUDIO_CLASSIFICATION_MODEL", "onnx-community/Musical-Instrument-Classification-ONNX")
LABELS = {
    "Acoustic_Guitar",
    "Bass_Guitar",
    "Drum_set",
    "Electro_Guitar",
    "flute",
    "Hi_Hats",
    "Keyboard",
    "Trumpet",
    "Violin",
}

KERNEL_CODE = r"""
import io, json, math, struct, wave
from lmlite import LLM

rate = 16000
samples = [sum(math.sin(2 * math.pi * 220 * k * t / rate) / k for k in (1, 2, 3)) / 2 for t in range(2 * rate)]
buffer = io.BytesIO()
with wave.open(buffer, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    wav.writeframes(b"".join(struct.pack("<h", int(s * 20000)) for s in samples))

pipe = await LLM.pipeline("audio-classification", model="__MODEL__", backend="wasm")
from_wav = await pipe(buffer.getvalue(), top_k=3)
from_samples = await pipe(samples[:rate], top_k=2)
many = await pipe.map([buffer.getvalue(), samples[:rate]], top_k=1)
await pipe.dispose()
print("RESULT" + json.dumps({"wav": from_wav, "samples": from_samples, "many": many}))
"""


def check(result):
    for key, count in (("wav", 3), ("samples", 2)):
        items = result[key]
        assert len(items) == count and {item["label"] for item in items} <= LABELS
        scores = [item["score"] for item in items]
        assert scores == sorted(scores, reverse=True) and all(0 < score <= 1 for score in scores)
    assert [len(item) for item in result["many"]] == [1, 1]


def test_audio_classification_in_pyodide(source_server):
    check(run_in_pyodide(source_server, KERNEL_CODE.replace("__MODEL__", MODEL), packages=("tqdm",)))


def test_audio_classification_in_xeus(xeus_site):
    check(run_in_xeus(xeus_site, KERNEL_CODE.replace("__MODEL__", MODEL)))
