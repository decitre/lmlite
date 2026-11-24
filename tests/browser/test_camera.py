# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for the camera widget, with Chromium's fake camera playing a photo of two cats."""

import urllib.request
from importlib.resources import files

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

CATS_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"
DETECTION_MODEL = "Xenova/detr-resnet-50"
FAKE_CAMERA = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
PAGE_URL = "https://camera.test/"  # a secure context, which getUserMedia needs; the page is empty

# On the page: renders the widget with a fake model, starts the camera, takes a photo and returns what was sent.
WIDGET = """async ({esm, runtimeUrl, detect}) => {
  const url = URL.createObjectURL(new Blob([esm], {type: 'text/javascript'}));
  const widget = (await import(url)).default;
  const sent = {};
  const model = {set: (name, value) => { sent[name] = value; }, save_changes: () => { sent.saved = true; }};
  const el = document.createElement('div');
  document.body.append(el);
  try {
    const cleanup = widget.render({model, el});
    const [controls, video, photo] = el.children;
    const [start, take, , status] = controls.children;
    const labels = [start.textContent, take.disabled];
    start.click();
    for (let i = 0; i < 100 && take.disabled; i++) await new Promise(resolve => setTimeout(resolve, 100));
    labels.push(start.textContent, take.disabled, video.style.display);
    take.click();
    for (let i = 0; i < 100 && !sent.photo; i++) await new Promise(resolve => setTimeout(resolve, 100));
    const bytes = Uint8Array.from(atob(sent.photo), (char) => char.charCodeAt(0));
    const result = {ok: true, labels, info: sent.info, saved: sent.saved, status: status.textContent,
                    head: [...bytes.subarray(0, 3)], size: bytes.length, shown: photo.style.display};
    if (detect) {
      await import(runtimeUrl);
      const llm = globalThis.lmLite.createRuntime({task: 'object-detection', task_model: detect, backend: 'wasm'});
      await llm.init();
      const blobUrl = URL.createObjectURL(new Blob([bytes], {type: 'image/jpeg'}));
      const found = await llm.runTask(blobUrl, JSON.stringify({threshold: 0.7}));
      await llm.dispose();
      result.labels_found = found.map((item) => item.label);
    }
    start.click();  // stop the camera
    await new Promise(resolve => setTimeout(resolve, 100));
    result.after = [start.textContent, take.disabled, video.srcObject === null];
    (await cleanup)?.();
    return result;
  } catch (error) {
    return {ok: false, error: String(error?.stack ?? error)};
  }
}"""


def run(tmp_path, detect=None):
    from lmlite.camera import CameraCapture

    cats = tmp_path / "cats.mjpeg"  # Chromium plays an MJPEG file as the camera; one JPEG is a still picture
    cats.write_bytes(urllib.request.urlopen(CATS_URL).read())
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=[*FAKE_CAMERA, f"--use-file-for-fake-video-capture={cats}"])
        try:
            context = browser.new_context()
            context.grant_permissions(["camera"])
            page = context.new_page()
            page.set_default_timeout(120000)
            page.route(PAGE_URL, lambda route: route.fulfill(body="<html></html>", content_type="text/html"))
            page.goto(PAGE_URL)
            runtime = files("lmlite").joinpath("lmlite.js").read_text()
            blob = page.evaluate("(src) => URL.createObjectURL(new Blob([src], {type: 'text/javascript'}))", runtime)
            return page.evaluate(WIDGET, {"esm": CameraCapture._esm, "runtimeUrl": blob, "detect": detect})
        finally:
            browser.close()


def test_the_widget_takes_a_jpeg_photo_and_releases_the_camera(tmp_path):
    result = run(tmp_path)

    assert result["ok"], result.get("error")
    assert result["labels"][:2] == ["Start camera", True]  # a photo needs a running camera
    assert result["labels"][2:] == ["Stop camera", False, ""]
    assert result["head"] == [0xFF, 0xD8, 0xFF] and result["size"] > 5000  # a JPEG, not an empty file
    assert result["saved"] is True and result["shown"] == ""
    info = result["info"]
    assert 0 < max(info["width"], info["height"]) <= 1024 and info["bytes"] == result["size"]
    assert result["status"] == f"Photo {info['width']} x {info['height']}"
    assert result["after"] == ["Start camera", True, True]


def test_a_photo_from_the_camera_goes_through_object_detection(tmp_path):
    result = run(tmp_path, detect=DETECTION_MODEL)

    assert result["ok"], result.get("error")
    assert result["labels_found"].count("cat") == 2 and "couch" in result["labels_found"]  # the cats photo


# A phone browser can refuse video.play() after the permission prompt (Brave on Android: "play() can only be initiated
# by a user gesture."), and does not autoplay either. This page does the same until the video is tapped.
REFUSE_PLAY = """
const play = HTMLMediaElement.prototype.play;
Object.defineProperty(HTMLMediaElement.prototype, 'autoplay', { get: () => false, set: () => {} });  // no autoplay either
let tapped = false;
document.addEventListener('click', (event) => { tapped = tapped || event.target.tagName === 'VIDEO'; }, true);
HTMLMediaElement.prototype.play = function () {
  return tapped ? play.call(this)
    : Promise.reject(new DOMException('play() can only be initiated by a user gesture.', 'NotAllowedError'));
};
"""

TAP_TO_START = """async ({esm}) => {
  const url = URL.createObjectURL(new Blob([esm], {type: 'text/javascript'}));
  const widget = (await import(url)).default;
  const sent = {};
  const model = {set: (name, value) => { sent[name] = value; }, save_changes: () => {}};
  const el = document.createElement('div');
  document.body.append(el);
  widget.render({model, el});
  const [controls, video] = el.children;
  const [start, take, , status] = controls.children;
  const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  start.click();
  for (let i = 0; i < 100 && start.textContent !== 'Stop camera'; i++) await wait(100);
  await wait(500);
  const before = {status: status.textContent, take: take.disabled, shown: video.style.display, start: start.textContent};
  video.click();  // the tap on the picture
  for (let i = 0; i < 100 && take.disabled; i++) await wait(100);
  const after = {status: status.textContent, take: take.disabled, paused: video.paused};
  take.click();
  for (let i = 0; i < 100 && !sent.photo; i++) await wait(100);
  const head = [...Uint8Array.from(atob(sent.photo), (char) => char.charCodeAt(0)).subarray(0, 3)];
  return {before, after, head};
}"""


def test_a_refused_preview_starts_when_the_picture_is_tapped(tmp_path):
    from lmlite.camera import CameraCapture

    cats = tmp_path / "cats.mjpeg"
    cats.write_bytes(urllib.request.urlopen(CATS_URL).read())
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=[*FAKE_CAMERA, f"--use-file-for-fake-video-capture={cats}"])
        try:
            context = browser.new_context()
            context.grant_permissions(["camera"])
            page = context.new_page()
            page.set_default_timeout(60000)
            page.route(PAGE_URL, lambda route: route.fulfill(body="<html></html>", content_type="text/html"))
            page.add_init_script(REFUSE_PLAY)
            page.goto(PAGE_URL)
            result = page.evaluate(TAP_TO_START, {"esm": CameraCapture._esm})
        finally:
            browser.close()

    # The camera stays on and says what to do, instead of stopping with the browser's error.
    assert result["before"] == {
        "status": "Tap the picture to start the preview.",
        "take": True,
        "shown": "",
        "start": "Stop camera",
    }
    assert result["after"] == {"status": "", "take": False, "paused": False}
    assert result["head"] == [0xFF, 0xD8, 0xFF]
