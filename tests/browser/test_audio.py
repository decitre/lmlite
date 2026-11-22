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
        f"{source_server}/src/lmlite/llm.js",
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
          const model = {get: (name) => runtime, set: (name, value) => sent.push([name, value]), save_changes: () => sent.push(['saved'])};
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
            for (let i = 0; i < 100 && !sent.some(([key]) => key === 'recording'); i++) {
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

    result = in_page(source_server, RECORD, {"url": f"{source_server}/src/lmlite/llm.js", "raw": True}, capture)

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

    result = in_page(source_server, RECORD, {"url": f"{source_server}/src/lmlite/llm.js", "raw": False}, capture)

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
    assert "is silent" in first["status"] and "Tick Raw input" in first["status"] and "microphone" in first["status"]
    assert first["info"]["peak"] < 0.005 and first["hasRecording"]  # still delivered, with the description
    assert second["status"].startswith("Recorded ") and "silent" not in second["status"]
    assert second["info"]["raw"] is True and second["info"]["peak"] > 0.3
    assert result["order"][:2] == ["info", "recording"]  # the description arrives before the recording


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

    assert "is silent" in result["first"]["status"] and "Tick Raw input" not in result["first"]["status"]
