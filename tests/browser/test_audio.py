# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for microphone recording, with Chromium's fake audio device (a repeating beep)."""

import json
import math
import struct
import urllib.parse
import wave

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

FAKE_MICROPHONE = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]


def in_page(source_server, script, args=None, capture=None):
    """Run `script` on a page whose microphone plays the WAV file `capture` (a beep when None)."""
    flags = [f"--use-file-for-fake-audio-capture={capture}"] if capture else []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=[*FAKE_MICROPHONE, *flags])
        try:
            context = browser.new_context()
            context.grant_permissions(["microphone"])
            page = context.new_page()
            page.set_default_timeout(60000)
            page.goto(source_server, wait_until="domcontentloaded")
            return page.evaluate(script, args)
        finally:
            browser.close()


def check_wav(data, seconds):
    data = bytes(data)
    assert data[:4] == b"RIFF" and data[8:16] == b"WAVEfmt "
    channels, rate = struct.unpack("<HI", data[22:28])
    assert (channels, rate) == (1, 16000)
    samples = struct.unpack(f"<{(len(data) - 44) // 2}h", data[44:])
    assert 0.5 * seconds < len(samples) / 16000 < 2.5 * seconds
    assert max(abs(sample) for sample in samples) > 500  # the fake device beeps, so this is not silence


def test_recorder_returns_a_16khz_mono_wav(source_server):
    result = in_page(
        source_server,
        """async (url) => {
          await import(url);
          try {
            const recorder = new globalThis.lmLite.AudioRecorder();
            await recorder.start();
            const during = recorder.recording;
            await new Promise(resolve => setTimeout(resolve, 1200));
            const wav = await recorder.stop();
            // Recording again works, and the microphone was released in between.
            await recorder.start();
            const again = await new Promise(resolve => setTimeout(resolve, 300)).then(() => recorder.stop());
            return {ok: true, during, after: recorder.recording, wav: [...wav], againBytes: again.byteLength};
          } catch (error) {
            return {ok: false, error: String(error?.stack ?? error)};
          }
        }""",
        f"{source_server}/src/lmlite/lmlite.js",
    )

    assert result["ok"], result.get("error")
    assert result["during"] is True and result["after"] is False and result["againBytes"] > 44
    check_wav(result["wav"], 1.2)


def test_widget_loads_the_runtime_on_the_page_and_sends_the_recording(source_server):
    anywidget_module = pytest.importorskip("anywidget")
    assert anywidget_module
    from lmlite.audio import AudioRecorder

    widget = AudioRecorder()
    result = in_page(
        source_server,
        """async ({esm, runtime}) => {
          // The page has no lmLite: the widget has to load the runtime it carries.
          const hadRuntime = Boolean(globalThis.lmLite);
          const url = URL.createObjectURL(new Blob([esm], {type: 'text/javascript'}));
          const widget = (await import(url)).default;
          const sent = [];
          const model = {get: (name) => ({runtime, raw: false, controls: true})[name], set: (name, value) => sent.push([name, value]), save_changes: () => sent.push(['saved'])};
          const el = document.createElement('div');
          document.body.append(el);
          try {
            await widget.render({model, el});
            const [button, , status] = el.children;  // the record button, the raw input box, the status
            const labels = [button.textContent];
            button.click();
            await new Promise(resolve => setTimeout(resolve, 1500));
            labels.push(button.textContent, status.textContent);
            button.click();
            for (let i = 0; i < 300 && !sent.some(([key]) => key === 'recording'); i++) {
              await new Promise(resolve => setTimeout(resolve, 100));
            }
            const [name, text] = sent.find(([key]) => key === 'recording');
            return {ok: true, hadRuntime, hasRuntime: Boolean(globalThis.lmLite?.AudioRecorder), labels,
                    final: [button.textContent, status.textContent], name, saved: sent.some(([key]) => key === 'saved'),
                    wav: [...atob(text)].map((char) => char.charCodeAt(0))};
          } catch (error) {
            return {ok: false, error: String(error?.stack ?? error)};
          }
        }""",
        {"esm": widget._esm, "runtime": widget.runtime},
    )

    assert result["ok"], result.get("error")
    assert result["hadRuntime"] is False and result["hasRuntime"] is True
    assert result["labels"] == ["Record", "Stop", "Recording..."]
    assert result["name"] == "recording" and result["saved"] is True
    assert result["final"][0] == "Record again" and result["final"][1].startswith("Recorded ")
    check_wav(result["wav"], 1.5)


# Runs in the kernel: shows the widget, then waits for the recording that the test makes with the page's buttons.
KERNEL_CODE = r"""
import json
from lmlite.audio import AudioRecorder

recorder = AudioRecorder()
display(recorder)
wav = await recorder.wait()
print("RESULT" + json.dumps({"bytes": len(wav), "header": wav[:4].decode(), "rate": int.from_bytes(wav[24:28], "little")}))
"""


def test_widget_records_into_a_real_xeus_kernel(xeus_site):
    url = f"{xeus_site}/repl/index.html?kernel=xpython&execute=1&code={urllib.parse.quote(KERNEL_CODE)}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=FAKE_MICROPHONE)
        try:
            context = browser.new_context()
            context.grant_permissions(["microphone"])
            page = context.new_page()
            page.set_default_timeout(300000)
            console = []
            page.on("console", lambda message: console.append(f"{message.type}: {message.text[:300]}"))
            page.on("pageerror", lambda error: console.append(f"pageerror: {error}"))
            page.goto(url)
            page.get_by_role("button", name="Record", exact=True).click()
            page.wait_for_timeout(1500)
            page.get_by_role("button", name="Stop", exact=True).click()
            try:
                page.wait_for_function("document.body.innerText.includes('RESULT{')", timeout=60000)
            except Exception:
                pytest.fail(
                    f"No RESULT. Page:\n{page.inner_text('body')[-2500:]}\nConsole:\n" + "\n".join(console[-15:])
                )
            text = page.inner_text("body")
        finally:
            browser.close()

    line = next(line for line in text.splitlines() if line.startswith("RESULT{"))
    result = json.loads(line.removeprefix("RESULT"))
    assert result["header"] == "RIFF" and result["rate"] == 16000
    assert 0.5 * 1.5 * 32000 < result["bytes"] < 2.5 * 1.5 * 32000  # about 1.5 s of 16-bit samples


def write_input(path, channels, polarity):
    """Three seconds of a 440 Hz tone at half scale in each channel, times `polarity` (silence for 0)."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        frames = bytearray()
        for n in range(48000 * 3):
            value = 0.5 * math.sin(2 * math.pi * 440 * n / 48000)
            frames += struct.pack(f"<{channels}h", *[int(32767 * value * sign) for sign in polarity])
        wav.writeframes(bytes(frames))
    return path


# (channels, polarity of each channel, whether there is sound, and whether the raw mode is needed to hear it)
INPUTS = {
    "mono": (1, (1,), True),
    "stereo": (2, (1, 1), True),
    "opposite_polarity": (2, (1, -1), True),  # the browser's processing mixes these channels to exact silence
    "left_channel_only": (2, (1, 0), True),
    "silence": (1, (0,), False),
}

RECORD = """async ({url, raw}) => {
  await import(url);
  const recorder = new globalThis.lmLite.AudioRecorder();
  await recorder.start({raw});
  await new Promise(resolve => setTimeout(resolve, 1500));
  const wav = await recorder.stop();
  return {wav: [...wav], info: recorder.info};
}"""


def peak_of(result):
    samples = struct.unpack(f"<{(len(result['wav']) - 44) // 2}h", bytes(result["wav"])[44:])
    return max(abs(sample) for sample in samples) / 32767


@pytest.mark.parametrize("name", INPUTS)
def test_raw_recording_keeps_the_sound_of_any_input_and_levels_it(source_server, tmp_path, name):
    channels, polarity, audible = INPUTS[name]
    capture = write_input(tmp_path / f"{name}.wav", channels, polarity)

    result = in_page(source_server, RECORD, {"url": f"{source_server}/src/lmlite/lmlite.js", "raw": True}, capture)

    peak = peak_of(result)
    info = result["info"]
    assert info["raw"] is True and info["channels"] in (1, 2) and info["sampleRate"] > 0
    if audible:
        # The channels survive (opposite polarity no longer cancels), and the level is raised to a peak of 0.9.
        assert 0.85 < peak < 0.95, f"{name}: peak {peak}"
        assert info["peak"] == pytest.approx(0.5, abs=0.1) and info["gain"] == pytest.approx(0.9 / info["peak"])
    else:
        assert peak < 0.005 and info["gain"] == 1


@pytest.mark.parametrize("name", ["mono", "stereo", "silence"])
def test_default_recording_describes_what_it_recorded(source_server, tmp_path, name):
    channels, polarity, audible = INPUTS[name]
    capture = write_input(tmp_path / f"{name}.wav", channels, polarity)

    result = in_page(source_server, RECORD, {"url": f"{source_server}/src/lmlite/lmlite.js", "raw": False}, capture)

    peak = peak_of(result)
    assert result["info"]["peak"] == pytest.approx(peak, abs=0.01) and result["info"]["raw"] is False
    assert (0.3 < peak < 0.7) if audible else peak < 0.005  # no gain is applied without the raw mode


WIDGET = """async ({esm, runtime}) => {
  const url = URL.createObjectURL(new Blob([esm], {type: 'text/javascript'}));
  const widget = (await import(url)).default;
  const sent = {};
  const model = {get: () => runtime, set: (name, value) => { sent[name] = value; }, save_changes: () => {}};
  const el = document.createElement('div');
  document.body.append(el);
  await widget.render({model, el});
  const [button, rawLabel, status] = el.children;
  const record = async (raw) => {
    delete sent.recording;
    rawLabel.querySelector('input').checked = raw;
    button.click();
    await new Promise(resolve => setTimeout(resolve, 1500));
    button.click();
    for (let i = 0; i < 100 && !sent.recording; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return {status: status.textContent, info: sent.info, hasRecording: Boolean(sent.recording)};
  };
  const first = await record(false);
  const second = await record(true);
  return {first, second, order: Object.keys(sent)};
}"""


def test_widget_says_what_to_do_about_a_silent_recording(source_server, tmp_path):
    pytest.importorskip("anywidget")
    from lmlite.audio import AudioRecorder

    widget = AudioRecorder()
    # Opposite polarity: silent with the browser's processing, audible with Raw input ticked.
    capture = write_input(tmp_path / "opposite.wav", 2, (1, -1))

    result = in_page(source_server, WIDGET, {"esm": widget._esm, "runtime": widget.runtime}, capture)

    first, second = result["first"], result["second"]
    status = first["status"]
    assert "is silent" in status and "input level" in status and "microphone" in status
    assert "tick Raw input" in status and status.index("input level") < status.index(
        "Raw input"
    )  # the usual cause first
    assert first["info"]["peak"] < 0.005 and first["hasRecording"]  # still delivered, with the description
    assert second["status"].startswith("Recorded ") and "silent" not in second["status"]
    assert second["info"]["raw"] is True and second["info"]["peak"] > 0.3
    updates = [name for name in result["order"] if name != "active"]  # active follows the state, not the data
    assert updates[:2] == ["info", "recording"]  # the description arrives before the recording


def test_widget_does_not_suggest_raw_input_when_it_was_ticked(source_server, tmp_path):
    pytest.importorskip("anywidget")
    from lmlite.audio import AudioRecorder

    widget = AudioRecorder()
    result = in_page(
        source_server,
        WIDGET.replace("const first = await record(false);", "const first = await record(true);"),
        {"esm": widget._esm, "runtime": widget.runtime},
        write_input(tmp_path / "silence.wav", 1, (0,)),
    )

    status = result["first"]["status"]
    assert "is silent" in status and "input level" in status and "Raw input" not in status


# request(): Python asks, the widget shows the prompt and records nothing until the user taps, and only that
# recording answers. Every way it ends releases the microphone.
REQUESTS = """async ({esm, runtime}) => {
  const url = URL.createObjectURL(new Blob([esm], {type: 'text/javascript'}));
  const widget = (await import(url)).default;
  const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const until = async (done) => { for (let i = 0; i < 100 && !done(); i++) await wait(100); };
  const streams = [];
  const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (constraints) => { const s = await real(constraints); streams.push(s); return s; };
  const live = () => streams.filter((s) => s.getTracks().some((track) => track.readyState === 'live')).length;
  const make = async (initial = {}) => {
    const sent = {};
    const handlers = {};
    const values = {runtime, raw: false, controls: true, ...initial};
    const model = {get: (name) => values[name], on: (name, handler) => { handlers[name] = handler; },
                   set: (name, value) => { sent[name] = value; }, save_changes: () => {}};
    const el = document.createElement('div');
    document.body.append(el);
    const cleanup = await widget.render({model, el});
    const [button, , status, bar] = el.children;
    const [ask, go, dismiss] = bar.children;
    return {sent, button, go, dismiss, cleanup,
            // request(): Python sets pending_request, and clears it when the request ends however it ends
            send: (message) => {
              values.pending_request = message.do === 'request'
                ? {id: message.id, prompt: message.prompt, max_seconds: message.max_seconds} : {};
              handlers['change:pending_request']();
            },
            state: () => ({bar: bar.style.display, ask: ask.textContent, go: go.textContent, live: live()})};
  };
  const wavSeconds = (text) => (atob(text).length - 44) / 2 / 16000;
  const out = {};

  let w = await make();  // a tap on Record, a tap on Stop: the recording carries the id of the request
  w.send({do: 'request', id: 1, prompt: 'Tap Record'});
  out.asked = {...w.state(), recording: w.sent.recording, active: w.sent.active};
  w.go.click();
  await until(() => w.sent.active === true);
  out.recording = w.state();
  await wait(800);
  w.go.click();
  await until(() => w.sent.answered);
  out.answered = {id: w.sent.answered, riff: atob(w.sent.recording).slice(0, 4), ...w.state()};
  await w.cleanup();

  w = await make();  // Cancel before recording
  w.send({do: 'request', id: 2, prompt: 'Tap Record'});
  w.dismiss.click();
  await wait(200);
  out.cancelledIdle = {id: w.sent.cancelled, recording: w.sent.recording, ...w.state()};
  await w.cleanup();

  w = await make();  // Cancel while recording: the recording is dropped and the microphone released
  w.send({do: 'request', id: 3, prompt: 'Tap Record'});
  w.go.click();
  await until(() => w.sent.active === true);
  await wait(400);
  w.dismiss.click();
  await until(() => w.sent.cancelled);
  out.cancelledRecording = {id: w.sent.cancelled, recording: w.sent.recording, answered: w.sent.answered, active: w.sent.active,
                            ...w.state()};
  await w.cleanup();

  w = await make();  // max_seconds stops the recording by itself
  w.send({do: 'request', id: 4, prompt: 'Tap Record', max_seconds: 1});
  w.go.click();
  await until(() => w.sent.answered);
  out.limited = {id: w.sent.answered, seconds: wavSeconds(w.sent.recording), ...w.state()};
  await w.cleanup();

  w = await make();  // Python gave up while recording (a timeout): dropped, released, nothing reported
  w.send({do: 'request', id: 5, prompt: 'Tap Record'});
  w.go.click();
  await until(() => w.sent.active === true);
  await wait(300);
  w.send({do: 'cancel_request', id: 5});
  await until(() => w.sent.active === false);
  await wait(200);
  out.timedOut = {recording: w.sent.recording, answered: w.sent.answered, cancelled: w.sent.cancelled, error: w.sent.error, ...w.state()};
  await w.cleanup();

  w = await make();  // a recording that started before the request does not answer it
  w.button.click();
  await until(() => w.sent.active === true);
  await wait(300);
  w.send({do: 'request', id: 6, prompt: 'Tap Record'});
  await until(() => w.sent.active === false);
  await wait(200);
  out.earlier = {recording: w.sent.recording, answered: w.sent.answered, ...w.state()};
  w.dismiss.click();
  await wait(100);
  await w.cleanup();

  w = await make({pending_request: {id: 8, prompt: 'Tap Record'}});  // asked before the widget was shown
  out.early = w.state();
  w.go.click();
  await until(() => w.sent.active === true);
  await wait(500);
  w.go.click();
  await until(() => w.sent.answered);
  out.earlyAnswered = w.sent.answered;
  await w.cleanup();

  navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Permission denied', 'NotAllowedError'); };
  w = await make();  // the browser refuses the microphone: the request fails with the reason
  w.send({do: 'request', id: 7, prompt: 'Tap Record'});
  w.go.click();
  await until(() => w.sent.error);
  await wait(100);
  out.denied = {error: w.sent.error, ...w.state()};
  await w.cleanup();
  navigator.mediaDevices.getUserMedia = real;
  out.streams = streams.length;
  return out;
}"""


def test_a_request_needs_a_tap_and_ends_with_the_microphone_released(source_server):
    anywidget_module = pytest.importorskip("anywidget")
    assert anywidget_module
    from lmlite.audio import AudioRecorder

    widget = AudioRecorder()
    result = in_page(source_server, REQUESTS, {"esm": widget._esm, "runtime": widget.runtime})

    prompt = "Tap Record"
    # The prompt is up and nothing records before the tap.
    assert result["asked"] == {"bar": "", "ask": prompt, "go": "Record", "live": 0, "recording": None, "active": None}
    assert result["recording"] == {"bar": "", "ask": prompt, "go": "Stop", "live": 1}
    assert result["answered"] == {"id": 1, "riff": "RIFF", "bar": "none", "ask": prompt, "go": "Record", "live": 0}
    assert result["cancelledIdle"] == {
        "id": 2,
        "recording": None,
        "bar": "none",
        "ask": prompt,
        "go": "Record",
        "live": 0,
    }
    cancelled = result["cancelledRecording"]
    assert cancelled["id"] == 3 and cancelled["recording"] is None and cancelled["answered"] is None
    assert cancelled["active"] is False and cancelled["bar"] == "none" and cancelled["live"] == 0
    assert result["limited"]["id"] == 4 and 0.5 < result["limited"]["seconds"] < 2.5 and result["limited"]["live"] == 0
    timed_out = result["timedOut"]
    assert timed_out["recording"] is None and timed_out["answered"] is None and timed_out["cancelled"] is None
    assert timed_out["error"] is None and timed_out["bar"] == "none" and timed_out["live"] == 0
    assert result["earlier"]["recording"] is None and result["earlier"]["answered"] is None
    assert result["earlier"]["live"] == 0
    assert result["early"] == {"bar": "", "ask": prompt, "go": "Record", "live": 0}  # shown when it renders
    assert result["earlyAnswered"] == 8
    assert result["denied"]["error"] == {"id": 7, "name": "NotAllowedError", "message": "Permission denied"}
    assert result["denied"]["bar"] == "none"
    assert result["streams"] >= 5  # the microphone was really opened and closed each time


# The same request flow through a real xeus-python kernel: a cancel, a timeout and a recording, each as a message
# from the page to the kernel's awaiting request().
KERNEL_REQUESTS = r"""
import json
from lmlite.audio import AudioRecorder, CaptureCancelled

recorder = AudioRecorder()
display(recorder)
try:
    await recorder.request("First prompt", timeout=120)
    first = "answered"
except CaptureCancelled:
    first = "cancelled"
try:
    await recorder.request("Second prompt", timeout=3)
    second = "answered"
except TimeoutError:
    second = "timeout"
wav = await recorder.request("Third prompt", timeout=120, max_seconds=1)
print("RESULT" + json.dumps({"first": first, "second": second, "header": wav[:4].decode(),
                             "rate": int.from_bytes(wav[24:28], "little"), "bytes": len(wav)}))
"""


def test_a_request_is_answered_cancelled_and_timed_out_in_a_real_xeus_kernel(xeus_site):
    from lab import run_in_repl, tap

    def taps(page):
        page.get_by_text("First prompt", exact=True).wait_for()
        tap(page, "First prompt", 2)  # Cancel
        page.get_by_text("Second prompt", exact=True).wait_for()
        page.get_by_text("Second prompt", exact=True).wait_for(state="hidden")  # timed out: the prompt goes
        page.get_by_text("Third prompt", exact=True).wait_for()
        tap(page, "Third prompt", 1)  # Record: it stops by itself after max_seconds

    result, _ = run_in_repl(xeus_site, KERNEL_REQUESTS, taps, FAKE_MICROPHONE, ["microphone"])

    assert result["first"] == "cancelled" and result["second"] == "timeout"
    assert result["header"] == "RIFF" and result["rate"] == 16000
    assert 0.3 * 32000 < result["bytes"] < 3 * 32000  # about a second of 16-bit samples
