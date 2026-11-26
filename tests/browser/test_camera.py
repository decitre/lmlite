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
  const model = {get: () => undefined, set: (name, value) => { sent[name] = value; }, save_changes: () => { sent.saved = true; }};
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
  const model = {get: () => undefined, set: (name, value) => { sent[name] = value; }, save_changes: () => {}};
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


# App mode: no buttons, the camera starts by itself, and Python takes the photo with a message (camera.take()).
APP_MODE = """async ({esm}) => {
  const url = URL.createObjectURL(new Blob([esm], {type: 'text/javascript'}));
  const widget = (await import(url)).default;
  const sent = {};
  const values = {controls: false, autostart: true, show_photo: false};
  const handlers = {};
  const model = {get: (name) => values[name], on: (name, handler) => { handlers[name] = handler; },
                 set: (name, value) => { sent[name] = value; }, save_changes: () => {}};
  const el = document.createElement('div');
  document.body.append(el);
  widget.render({model, el});
  const [controls, video, photo] = el.children;
  const [start, take, facing, status] = controls.children;
  const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  for (let i = 0; i < 100 && (video.paused || !video.videoWidth); i++) await wait(100);
  const hidden = [start, take, facing].map((control) => control.style.display);
  const running = {paused: video.paused, shown: video.style.display, width: video.videoWidth > 0};
  handlers['msg:custom']({do: 'take'});  // camera.take() in Python
  for (let i = 0; i < 100 && !sent.photo; i++) await wait(100);
  const head = [...Uint8Array.from(atob(sent.photo), (char) => char.charCodeAt(0)).subarray(0, 3)];
  const late = await (async () => { handlers['msg:custom']({do: 'stop'}); await wait(200); return video.srcObject === null; })();
  return {hidden, running, head, still: photo.style.display, stopped: late};
}"""


def test_app_mode_starts_by_itself_hides_the_buttons_and_takes_a_photo_on_a_message(tmp_path):
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
            page.goto(PAGE_URL)
            result = page.evaluate(APP_MODE, {"esm": CameraCapture._esm})
        finally:
            browser.close()

    assert result["hidden"] == ["none", "none", "none"]
    assert result["running"] == {"paused": False, "shown": "", "width": True}
    assert result["head"] == [0xFF, 0xD8, 0xFF] and result["still"] == "none"  # a JPEG, and no still under the preview
    assert result["stopped"] is True


# request(): Python asks, the widget shows the prompt and takes nothing until the user taps, and only that tap answers.
REQUESTS = """async ({esm}) => {
  const url = URL.createObjectURL(new Blob([esm], {type: 'text/javascript'}));
  const widget = (await import(url)).default;
  const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const until = async (done) => { for (let i = 0; i < 100 && !done(); i++) await wait(100); };
  const make = (initial = {}) => {
    const sent = {};
    const handlers = {};
    const values = {...initial};
    const model = {get: (name) => values[name], on: (name, handler) => { handlers[name] = handler; },
                   set: (name, value) => { sent[name] = value; }, save_changes: () => {}};
    const el = document.createElement('div');
    document.body.append(el);
    const cleanup = widget.render({model, el});
    const [controls, video, , bar] = el.children;
    const [start, take] = controls.children;
    const [ask, shoot, dismiss] = bar.children;
    return {sent, start, take, shoot, dismiss, video, cleanup,
            // request(): Python sets pending_request, and clears it when the request ends however it ends
            send: (message) => {
              values.pending_request = message.do === 'request' ? {id: message.id, prompt: message.prompt} : {};
              handlers['change:pending_request']();
            },
            state: () => ({bar: bar.style.display, ask: ask.textContent, camera: start.textContent, stream: video.srcObject !== null})};
  };
  const out = {};

  let w = make();  // a tap answers the request, and the camera the request started is released
  w.send({do: 'request', id: 1, prompt: 'Tap to take a photo'});
  await until(() => !w.take.disabled);
  await wait(300);
  out.asked = {...w.state(), photo: w.sent.photo, answered: w.sent.answered};
  w.shoot.click();
  await until(() => w.sent.answered);
  out.answered = {id: w.sent.answered, jpeg: [...Uint8Array.from(atob(w.sent.photo), (c) => c.charCodeAt(0)).subarray(0, 3)],
                  ...w.state()};
  w.cleanup();

  w = make();  // Cancel: nothing is captured, the user's cancel is reported, the camera is released
  w.send({do: 'request', id: 2, prompt: 'Tap to take a photo'});
  await until(() => !w.take.disabled);
  w.dismiss.click();
  await wait(200);
  out.cancelled = {id: w.sent.cancelled, photo: w.sent.photo, answered: w.sent.answered, ...w.state()};
  w.cleanup();

  w = make();  // Python gave up (a timeout): the prompt goes and the camera is released, nothing is reported
  w.send({do: 'request', id: 3, prompt: 'Tap to take a photo'});
  await until(() => !w.take.disabled);
  w.send({do: 'cancel_request', id: 3});
  await wait(200);
  out.timedOut = {cancelled: w.sent.cancelled, error: w.sent.error, ...w.state()};
  w.cleanup();

  const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Permission denied', 'NotAllowedError'); };
  w = make();  // the browser refuses the camera: the request fails with the reason and the prompt goes
  w.send({do: 'request', id: 4, prompt: 'Tap to take a photo'});
  await until(() => w.sent.error);
  await wait(100);
  out.denied = {error: w.sent.error, ...w.state()};
  w.cleanup();
  navigator.mediaDevices.getUserMedia = getUserMedia;

  w = make({pending_request: {id: 6, prompt: 'Tap to take a photo'}});  // asked before the widget was shown
  await until(() => !w.take.disabled);
  out.early = w.state();
  w.shoot.click();
  await until(() => w.sent.answered);
  out.earlyAnswered = w.sent.answered;
  w.cleanup();

  w = make();  // a camera the user had started stays on after the request
  w.start.click();
  await until(() => !w.take.disabled);
  w.send({do: 'request', id: 5, prompt: 'Tap to take a photo'});
  w.shoot.click();
  await until(() => w.sent.answered);
  await wait(200);
  out.kept = {id: w.sent.answered, ...w.state()};
  w.cleanup();
  return out;
}"""


def test_a_request_needs_a_tap_and_ends_with_the_camera_released(tmp_path):
    from lmlite.camera import CameraCapture

    cats = tmp_path / "cats.mjpeg"
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
            result = page.evaluate(REQUESTS, {"esm": CameraCapture._esm})
        finally:
            browser.close()

    prompt = "Tap to take a photo"
    # The prompt and the live picture are up, and nothing was captured before the tap.
    assert result["asked"] == {
        "bar": "",
        "ask": prompt,
        "camera": "Stop camera",
        "stream": True,
        "photo": None,
        "answered": None,
    }
    # The tap sends a JPEG and the id of the request, and the camera the request started is stopped.
    assert result["answered"] == {
        "id": 1,
        "jpeg": [0xFF, 0xD8, 0xFF],
        "bar": "none",
        "ask": prompt,
        "camera": "Start camera",
        "stream": False,
    }
    assert result["cancelled"] == {
        "id": 2,
        "photo": None,
        "answered": None,
        "bar": "none",
        "ask": prompt,
        "camera": "Start camera",
        "stream": False,
    }
    assert result["timedOut"] == {
        "cancelled": None,
        "error": None,
        "bar": "none",
        "ask": prompt,
        "camera": "Start camera",
        "stream": False,
    }
    assert result["denied"]["error"] == {"id": 4, "name": "NotAllowedError", "message": "Permission denied"}
    assert result["denied"]["bar"] == "none" and result["denied"]["stream"] is False
    assert result["early"] == {
        "bar": "",
        "ask": prompt,
        "camera": "Stop camera",
        "stream": True,
    }  # shown when it renders
    assert result["earlyAnswered"] == 6
    assert result["kept"] == {"id": 5, "bar": "none", "ask": prompt, "camera": "Stop camera", "stream": True}


# The phone case of #76 inside a request: the browser refuses the preview until the picture is tapped.
REQUEST_TAP_TO_START = """async ({esm}) => {
  const url = URL.createObjectURL(new Blob([esm], {type: 'text/javascript'}));
  const widget = (await import(url)).default;
  const sent = {};
  const handlers = {};
  const values = {pending_request: {id: 1, prompt: 'Tap to take a photo'}};  // asked before the widget was shown
  const model = {get: (name) => values[name], on: (name, handler) => { handlers[name] = handler; },
                 set: (name, value) => { sent[name] = value; }, save_changes: () => {}};
  const el = document.createElement('div');
  document.body.append(el);
  widget.render({model, el});
  const [controls, video, , bar] = el.children;
  const [, take, , status] = controls.children;
  const [ask, shoot] = bar.children;
  const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  await wait(1500);
  const before = {bar: bar.style.display, ask: ask.textContent, status: status.textContent, take: take.disabled};
  shoot.click();  // not running yet: the request stays up
  await wait(200);
  const early = {bar: bar.style.display, answered: sent.answered, photo: sent.photo};
  video.click();  // the tap on the picture
  for (let i = 0; i < 100 && take.disabled; i++) await wait(100);
  shoot.click();
  for (let i = 0; i < 100 && !sent.answered; i++) await wait(100);
  return {before, early, answered: sent.answered, bar: bar.style.display, stream: video.srcObject !== null};
}"""


def test_a_request_waits_for_the_tap_that_starts_a_refused_preview(tmp_path):
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
            result = page.evaluate(REQUEST_TAP_TO_START, {"esm": CameraCapture._esm})
        finally:
            browser.close()

    assert result["before"] == {
        "bar": "",
        "ask": "Tap to take a photo",
        "status": "Tap the picture to start the preview.",
        "take": True,
    }
    assert result["early"] == {
        "bar": "",
        "answered": None,
        "photo": None,
    }  # nothing is taken from a preview that is not on
    assert result["answered"] == 1 and result["bar"] == "none" and result["stream"] is False


KERNEL_REQUESTS = r"""
import json
from lmlite.camera import CameraCapture, CaptureCancelled

camera = CameraCapture()
display(camera)
try:
    await camera.request("First prompt", timeout=120)
    first = "answered"
except CaptureCancelled:
    first = "cancelled"
try:
    await camera.request("Second prompt", timeout=3)
    second = "answered"
except TimeoutError:
    second = "timeout"
jpeg = await camera.request("Third prompt", timeout=120)
print("RESULT" + json.dumps({"first": first, "second": second, "head": list(jpeg[:3]), "bytes": len(jpeg)}))
"""


def test_a_request_is_answered_cancelled_and_timed_out_in_a_real_xeus_kernel(xeus_site, tmp_path):
    from lab import run_in_repl, tap

    cats = tmp_path / "cats.mjpeg"
    cats.write_bytes(urllib.request.urlopen(CATS_URL).read())

    def taps(page):
        page.get_by_text("First prompt", exact=True).wait_for()
        tap(page, "First prompt", 2)  # Cancel
        page.get_by_text("Second prompt", exact=True).wait_for()
        page.get_by_text("Second prompt", exact=True).wait_for(state="hidden")  # timed out: the prompt goes
        page.get_by_text("Third prompt", exact=True).wait_for()
        page.wait_for_function(
            "[...document.querySelectorAll('button')].some(b => b.textContent === 'Take photo' && !b.disabled)"
        )
        tap(page, "Third prompt", 1)  # Take photo

    result, released = run_in_repl(
        xeus_site, KERNEL_REQUESTS, taps, [*FAKE_CAMERA, f"--use-file-for-fake-video-capture={cats}"], ["camera"]
    )

    assert result["first"] == "cancelled" and result["second"] == "timeout"
    assert result["head"] == [0xFF, 0xD8, 0xFF] and result["bytes"] > 5000
    assert released == [True]  # the request started the camera and the answer stopped it
