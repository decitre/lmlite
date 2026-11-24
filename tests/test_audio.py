# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""The microphone recorder widget: its Python side, with the browser's recording replaced by setting ``audio``."""

import asyncio
import base64
from pathlib import Path

import pytest

pytest.importorskip("anywidget")

from lmlite.audio import AudioRecorder  # noqa: E402

WAV = b"RIFF$\x00\x00\x00WAVE"


def reply(recorder, wav, peak=0.3):
    """What the browser does when a recording stops: the description first, then the recording."""
    recorder.info = {"seconds": 1.0, "peak": peak, "channels": 2, "sampleRate": 48000, "mimeType": "audio/webm"}
    recorder.recording = base64.b64encode(wav).decode()


def test_the_widget_carries_the_runtime_and_uses_the_lmlite_global():
    recorder = AudioRecorder()

    assert recorder.runtime == (Path(__file__).parents[1] / "src/lmlite/lmlite.js").read_text()
    assert "globalThis.lmLite.AudioRecorder" in recorder._esm and 'model.get("runtime")' in recorder._esm
    assert recorder.audio == b"" and recorder.recording == ""


async def test_wait_returns_the_next_recording_and_then_the_last_one():
    recorder = AudioRecorder()
    waiting = asyncio.ensure_future(recorder.wait())
    await asyncio.sleep(0)

    reply(recorder, WAV)
    assert await waiting == WAV
    assert await recorder.wait() == WAV  # a recording is already there


async def test_clear_makes_wait_block_until_a_new_recording():
    recorder = AudioRecorder()
    reply(recorder, WAV)
    recorder.clear()
    waiting = asyncio.ensure_future(recorder.wait())
    await asyncio.sleep(0)

    assert not waiting.done()
    reply(recorder, WAV + b"2")
    assert await waiting == WAV + b"2"


async def test_every_waiter_gets_the_recording():
    recorder = AudioRecorder()
    first, second = asyncio.ensure_future(recorder.wait()), asyncio.ensure_future(recorder.wait())
    await asyncio.sleep(0)

    reply(recorder, WAV)

    assert await first == await second == WAV


async def test_info_describes_the_recording_and_is_forgotten_by_clear():
    recorder = AudioRecorder()
    assert recorder.info == {}

    reply(recorder, WAV, peak=0.0)  # a silent recording is still delivered, with its peak to say so
    assert await recorder.wait() == WAV and recorder.info["peak"] == 0.0 and recorder.info["channels"] == 2
    recorder.clear()

    assert recorder.info == {} and recorder.audio == b""


def test_a_recording_is_input_for_speech_recognition():
    # What the recorder returns is a 16 kHz mono 16-bit WAV, which _audio_input decodes without resampling.
    import io
    import struct
    import wave

    from lmlite.llm import _audio_input

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<3h", 0, 16384, -16384))

    assert _audio_input(buffer.getvalue()) == pytest.approx([0.0, 0.5, -0.5], abs=1e-4)
