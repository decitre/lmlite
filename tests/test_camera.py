# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""The camera widget: its Python side, with the browser's photo replaced by setting ``photo``."""

import asyncio
import base64

import pytest

pytest.importorskip("anywidget")

from lmlite.camera import CameraCapture  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF"


def reply(camera, jpeg):
    """What the browser does when a photo is taken: the description first, then the photo."""
    camera.info = {"width": 640, "height": 480, "bytes": len(jpeg)}
    camera.photo = base64.b64encode(jpeg).decode()


def test_a_new_widget_has_no_photo():
    camera = CameraCapture()

    assert camera.image == b"" and camera.photo == "" and camera.info == {}


async def test_wait_returns_the_next_photo_and_then_the_last_one():
    camera = CameraCapture()
    waiting = asyncio.ensure_future(camera.wait())
    await asyncio.sleep(0)

    reply(camera, JPEG)
    assert await waiting == JPEG
    assert await camera.wait() == JPEG  # a photo is already there


async def test_clear_makes_wait_block_until_a_new_photo():
    camera = CameraCapture()
    reply(camera, JPEG)
    camera.clear()
    waiting = asyncio.ensure_future(camera.wait())
    await asyncio.sleep(0)

    assert not waiting.done() and camera.info == {}
    reply(camera, JPEG + b"2")
    assert await waiting == JPEG + b"2"


async def test_every_waiter_gets_the_photo():
    camera = CameraCapture()
    first, second = asyncio.ensure_future(camera.wait()), asyncio.ensure_future(camera.wait())
    await asyncio.sleep(0)

    reply(camera, JPEG)
    assert await first == JPEG and await second == JPEG
