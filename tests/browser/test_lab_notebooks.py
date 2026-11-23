# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Notebooks in the real JupyterLab UI of the xeus-python site: what a reader sees (needs network access)."""

import re
import urllib.request

import lab
import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

FAKE_MICROPHONE = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
FIREFOX_PREFS = {"media.navigator.streams.fake": True, "media.navigator.permission.disabled": True}
CATS_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"
BROWSERS = ["chromium", "firefox"]  # Firefox is the reporter's browser; Safari is not covered


def in_lab(xeus_site, notebook, action=None, args=(), permissions=(), browser_name="chromium"):
    """Open `notebook`, run `action(page)` (default: run all cells), wait for the cells, return what they show.

    `args` are Chromium flags, and `permissions` are granted in Chromium; Firefox gets a fake microphone by its prefs."""
    with sync_playwright() as playwright:
        if browser_name == "firefox":
            browser = playwright.firefox.launch(firefox_user_prefs=FIREFOX_PREFS)
        else:
            browser = playwright.chromium.launch(args=list(args))
        try:
            # JupyterLite's service worker answers every cross-origin GET itself with fetch(request), and in Playwright's
            # Firefox build that never completes for a kernel's model downloads (the same fetch works with service
            # workers blocked, and in Chromium). Block it there to test the rest of Firefox.
            context = browser.new_context(
                viewport={"width": 1400, "height": 1000},
                **({"service_workers": "block"} if browser_name == "firefox" else {}),
            )
            if permissions and browser_name == "chromium":
                context.grant_permissions(list(permissions))
            page = context.new_page()
            page.set_default_timeout(lab.TIMEOUT)
            page.on("console", lambda message: lab.console.append(f"{message.type}: {message.text[:200]}"))
            page.on("pageerror", lambda error: lab.console.append(f"pageerror: {str(error)[:200]}"))
            lab.console.clear()
            lab.open_notebook(page, xeus_site, notebook)
            (action or lab.run_all)(page)
            lab.wait_for_all_cells(page)
            return lab.outputs(page)
        finally:
            browser.close()


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_image_outputs_notebook_shows_its_images(xeus_site, browser_name):
    shown = in_lab(xeus_site, "ch10_image-outputs.ipynb", browser_name=browser_name)

    assert shown["errors"] == []
    loaded = [image for image in shown["images"] if image["width"] > 0]
    assert len(loaded) == len(shown["images"])  # every <img> decoded, none is broken
    # the photo from a URL, the cut-out, the depth map, the cats photo, one mask per segment and the upscale
    assert len(loaded) >= 9, shown["images"]
    assert any(image["width"] >= 1232 for image in loaded)  # the 2x upscale, as the browser decoded it
    text = "".join(shown["text"])
    assert "cat 0.99" in text and "couch" in text  # named segments, not LABEL_<id> over the whole photo


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_speech_notebook_shows_audio_players_that_decode(xeus_site, browser_name):
    shown = in_lab(xeus_site, "ch12_speech.ipynb", browser_name=browser_name)

    assert shown["errors"] == []
    # the sentence, then one player per mapped sentence
    assert len(shown["audios"]) >= 4, shown["audios"]
    assert all(audio["src"].startswith("data:audio/wav") and audio["error"] is None for audio in shown["audios"])
    text = "".join(shown["text"])
    assert "Speech(duration=" in text and "progress: ['1/3', '2/3', '3/3']" in text


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_microphone_notebook_records_in_the_widget_and_transcribes(xeus_site, browser_name):
    def record_when_the_widget_shows(page):
        lab.run_all(page)
        page.get_by_role("button", name="Record", exact=True).click()  # the anywidget, rendered in Lab
        stop = page.get_by_role("button", name="Stop", exact=True)
        stop.wait_for()  # it appears once the browser has opened the microphone
        page.wait_for_timeout(1500)
        stop.click()

    shown = in_lab(
        xeus_site, "ch13_microphone.ipynb", record_when_the_widget_shows, FAKE_MICROPHONE, ["microphone"], browser_name
    )

    assert shown["errors"] == []
    text = "".join(shown["text"])
    assert " s of audio, " in text and "peak " in text and "raw input: False" in text
    assert "silent" not in text  # the fake microphone beeps
    assert len(shown["audios"]) == 1 and shown["audios"][0]["error"] is None  # the player for the recording


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_camera_notebook_takes_a_photo_in_the_widget_and_detects_objects(xeus_site, browser_name, tmp_path):
    cats = (
        tmp_path / "cats.mjpeg"
    )  # Chromium plays one JPEG as a still camera picture; Firefox's fake camera is a pattern
    cats.write_bytes(urllib.request.urlopen(CATS_URL).read())

    def take_a_photo_when_the_widget_shows(page):
        lab.run_all(page)
        page.get_by_role("button", name="Start camera", exact=True).click()  # the anywidget, rendered in Lab
        take = page.get_by_role("button", name="Take photo", exact=True)
        page.wait_for_function(
            "[...document.querySelectorAll('button')].some(b => b.textContent === 'Take photo' && !b.disabled)"
        )
        take.click()

    shown = in_lab(
        xeus_site,
        "ch11_camera.ipynb",
        take_a_photo_when_the_widget_shows,
        [*FAKE_MICROPHONE, f"--use-file-for-fake-video-capture={cats}"],
        ["camera"],
        browser_name,
    )

    assert shown["errors"] == []
    text = "".join(shown["text"])
    assert " pixels, " in text and " bytes" in text
    assert all(image["width"] > 0 for image in shown["images"]) and len(shown["images"]) >= 3  # preview, photo, boxes
    if browser_name == "chromium":
        assert len(re.findall(r"cat \d\.\d\d", text)) == 2 and "couch" in text  # the fake camera shows the cats photo
