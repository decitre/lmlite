# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""The two Voici apps, in a real browser with fake devices (they download models from the Hub).

Opt-in beyond `-m browser`: they need the site that `scripts/lite-build.sh` builds, in workspace/public (`pixi run
jupyterlite`, or `LITE_SITES=voici bash scripts/lite-build.sh` for the apps alone). A fake camera shows a photo of
two cats, and a fake microphone plays a recording of an English sentence.
"""

import functools
import http.server
import threading
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

PUBLIC = Path(__file__).resolve().parents[2] / "workspace/public"
CATS_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"
JFK_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
FAKE_DEVICES = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
TIMEOUT = 480000

BUTTONS = """() => [...document.querySelectorAll('button')]
  .filter((button) => button.offsetParent !== null)
  .map((button) => ({text: button.textContent.trim(), disabled: button.disabled,
                     left: Math.round(button.getBoundingClientRect().left)}))"""


@pytest.fixture(scope="module")
def site():
    if not (PUBLIC / "apps/voici/render/camera-objects.html").is_file():
        pytest.skip("no Voici site in workspace/public: run `LITE_SITES=voici bash scripts/lite-build.sh`")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(PUBLIC)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def open_app(playwright, site, name, permission, args):
    browser = playwright.chromium.launch(args=[*FAKE_DEVICES, *args])
    context = browser.new_context(viewport={"width": 700, "height": 900})
    context.grant_permissions([permission])
    page = context.new_page()
    page.set_default_timeout(TIMEOUT)
    page.goto(f"{site}/apps/voici/render/{name}.html")
    return browser, page


def visible(page, text):
    """The visible buttons with this label (an app has the widget's own, hidden, next to its own)."""
    return [button for button in page.evaluate(BUTTONS) if button["text"] == text]


def test_camera_app_shows_the_camera_then_the_objects_with_a_retry_button(site, tmp_path):
    cats = tmp_path / "cats.mjpeg"
    cats.write_bytes(urllib.request.urlopen(CATS_URL).read())
    with sync_playwright() as playwright:
        browser, page = open_app(
            playwright, site, "camera-objects", "camera", [f"--use-file-for-fake-video-capture={cats}"]
        )
        try:
            page.wait_for_function(
                "[...document.querySelectorAll('button')].some(b => b.textContent.trim() === 'Take a shot' && !b.disabled)"
            )
            live = page.evaluate(
                "() => { const v = document.querySelector('video'); "
                "return {playing: v.offsetParent !== null && !v.paused, "
                "images: [...document.querySelectorAll('img')].filter(i => i.naturalWidth && i.offsetParent).length}; }"
            )
            first = visible(page, "Take a shot")
            page.get_by_role("button", name="Take a shot", exact=True).click()
            page.get_by_role("button", name="Retry", exact=True).wait_for()
            text = page.inner_text("body")
            result = page.evaluate(
                "() => { const v = document.querySelector('video'); "
                "return {camera_hidden: v.offsetParent === null, "
                "images: [...document.querySelectorAll('img')].filter(i => i.naturalWidth && i.offsetParent).length}; }"
            )
            retry = visible(page, "Retry")
            page.get_by_role("button", name="Retry", exact=True).click()
            page.wait_for_function(
                "[...document.querySelectorAll('button')].some(b => b.textContent.trim() === 'Take a shot' && !b.disabled)"
            )
            back = page.evaluate(
                "() => { const v = document.querySelector('video'); return v.offsetParent !== null && !v.paused; }"
            )
        finally:
            browser.close()

    assert live == {"playing": True, "images": 0}  # the camera's picture, nothing else
    assert first[0]["left"] == retry[0]["left"] < 100  # one button, top left, whose label changes
    assert "cat 1.00" in text and "couch" in text  # the objects, from the photo of the two cats
    assert result == {"camera_hidden": True, "images": 1}  # the result replaces the camera
    assert back is True  # Retry gives the camera back


def test_translator_app_records_then_gives_the_french_text_and_voice_with_a_retry_button(site, tmp_path):
    english = tmp_path / "jfk.wav"
    english.write_bytes(urllib.request.urlopen(JFK_URL).read())
    with sync_playwright() as playwright:
        browser, page = open_app(
            playwright, site, "speak-french", "microphone", [f"--use-file-for-fake-audio-capture={english}"]
        )
        try:
            page.get_by_role("button", name="Record", exact=True).wait_for()
            idle = visible(page, "Record")
            page.get_by_role("button", name="Record", exact=True).click()
            page.get_by_role("button", name="Stop", exact=True).wait_for()
            recording = visible(page, "Stop")
            page.wait_for_timeout(12500)  # the recording is 11 s long
            page.get_by_role("button", name="Stop", exact=True).click()
            page.get_by_role("button", name="Working...", exact=True).wait_for()
            working = visible(page, "Working...")
            page.get_by_role("button", name="Retry", exact=True).wait_for()
            text = page.inner_text("body")
            voice = page.evaluate(
                "() => [...document.querySelectorAll('audio')].map(a => ({duration: a.duration, error: a.error?.message ?? null}))"
            )
            page.get_by_role("button", name="Retry", exact=True).click()
            page.get_by_role("button", name="Record", exact=True).wait_for()
            cleared = "English:" not in page.inner_text("body")
        finally:
            browser.close()

    assert idle[0]["left"] == recording[0]["left"] == working[0]["left"] < 100 and working[0]["disabled"]
    assert "English:" in text and "French:" in text and "pays" in text  # "country", in the French sentence
    assert len(voice) == 1 and voice[0]["error"] is None and voice[0]["duration"] > 4
    assert cleared  # Retry starts again
