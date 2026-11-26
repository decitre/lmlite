# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""``request()`` on the camera and microphone widgets, with the browser's replies set by hand."""

import asyncio
import base64
import pytest

pytest.importorskip("anywidget")

from lmlite.audio import AudioRecorder  # noqa: E402
from lmlite.camera import CameraCapture, CaptureCancelled  # noqa: E402

MEDIA = {CameraCapture: b"\xff\xd8\xff\xe0JFIF", AudioRecorder: b"RIFF$\x00\x00\x00WAVE"}
FIELD = {CameraCapture: "photo", AudioRecorder: "recording"}


@pytest.fixture(params=[CameraCapture, AudioRecorder])
def widget(request):
    widget = request.param()
    widget.kind = request.param
    return widget


def capture(widget, data=None):
    """What the browser does when the user taps for the request: the media first, then the id it answers."""
    setattr(widget, FIELD[widget.kind], base64.b64encode(data or MEDIA[widget.kind]).decode())


async def start(widget, **kwargs):
    task = asyncio.ensure_future(widget.request("Tap to do it", **kwargs))
    await asyncio.sleep(0)
    return task


async def test_a_request_shows_its_prompt_and_resolves_with_the_media_captured_for_it(widget):
    task = await start(widget)

    assert widget.pending_request == {"id": 1, "prompt": "Tap to do it"}  # what the widget shows, even if shown later
    capture(widget)
    await asyncio.sleep(0)
    assert not task.done()  # media with no request id is an ordinary capture, not the answer
    widget.answered = 1
    assert await task == MEDIA[widget.kind]
    assert widget.pending_request == {}  # the widget clears the prompt


async def test_only_media_for_this_request_resolves_it(widget):
    capture(widget)  # a photo or recording that was already there
    widget.answered = 1
    first = await start(widget)  # request 1 is answered by the 'answered' that is already 1: no change, so it waits
    await asyncio.sleep(0)
    assert not first.done()
    first.cancel()
    await asyncio.sleep(0)

    second = await start(widget)
    widget.answered = 1  # a stale answer for the first request
    await asyncio.sleep(0)
    assert not second.done()
    capture(widget, b"new")
    widget.answered = 2
    assert await second == b"new"


async def test_cancel_raises_capture_cancelled(widget):
    task = await start(widget)

    widget.cancelled = 1

    with pytest.raises(CaptureCancelled):
        await task


@pytest.mark.parametrize(
    "name, expected",
    [("NotAllowedError", PermissionError), ("SecurityError", PermissionError), ("NotFoundError", RuntimeError)],
)
async def test_a_refused_device_raises(widget, name, expected):
    task = await start(widget)

    widget.error = {"id": 1, "name": name, "message": "no device"}

    with pytest.raises(expected, match="no device"):
        await task


async def test_an_error_for_another_request_is_ignored(widget):
    task = await start(widget)

    widget.error = {"id": 7, "name": "NotAllowedError", "message": "old"}
    await asyncio.sleep(0)

    assert not task.done()
    task.cancel()


async def test_a_timeout_clears_the_prompt_and_frees_the_widget(widget):
    with pytest.raises(TimeoutError):
        await widget.request("Tap to do it", timeout=0.01)

    assert widget.pending_request == {}  # the widget clears the prompt and releases the device
    task = await start(widget)  # not busy
    assert widget.pending_request["id"] == 2
    widget.cancelled = 2
    with pytest.raises(CaptureCancelled):
        await task


async def test_cancelling_the_task_clears_the_prompt(widget):
    task = await start(widget)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert widget.pending_request == {}


async def test_the_prompt_is_cleared_whether_the_request_succeeds_or_not(widget):
    for request_id, outcome in ((1, "answered"), (2, "cancelled")):
        task = await start(widget)
        assert widget.pending_request["id"] == request_id
        setattr(widget, outcome, request_id)
        await asyncio.gather(task, return_exceptions=True)
        assert widget.pending_request == {}


async def test_one_request_at_a_time(widget):
    task = await start(widget)

    with pytest.raises(RuntimeError, match="already pending"):
        await widget.request("Another")

    task.cancel()


@pytest.mark.parametrize("prompt, timeout", [("", None), ("   ", None), (None, None), ("Tap", 0), ("Tap", -1)])
async def test_bad_arguments_are_rejected_before_anything_is_sent(widget, prompt, timeout):
    with pytest.raises(ValueError):
        await widget.request(prompt, timeout)

    assert widget.pending_request == {}


async def test_a_new_request_forgets_the_last_error(widget):
    widget.error = {"id": 9, "name": "NotAllowedError", "message": "old"}

    task = await start(widget)

    assert widget.error == {}
    task.cancel()


async def test_the_microphone_can_stop_by_itself():
    recorder = AudioRecorder()

    task = asyncio.ensure_future(recorder.request("Say it", max_seconds=5))
    await asyncio.sleep(0)

    assert recorder.pending_request == {"id": 1, "prompt": "Say it", "max_seconds": 5}
    task.cancel()
    with pytest.raises(ValueError):
        await recorder.request("Say it", max_seconds=0)
