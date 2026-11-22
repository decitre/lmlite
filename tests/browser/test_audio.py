# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Opt-in browser checks for microphone recording, with Chromium's fake audio device (a repeating beep)."""

import json
import struct
import urllib.parse

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

FAKE_MICROPHONE = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]


def in_page(source_server, script, args=None):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=FAKE_MICROPHONE)
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
            const [button, status] = el.children;
            const labels = [button.textContent];
            button.click();
            await new Promise(resolve => setTimeout(resolve, 1500));
            labels.push(button.textContent, status.textContent);
            button.click();
            for (let i = 0; i < 100 && !sent.length; i++) {
              await new Promise(resolve => setTimeout(resolve, 100));
            }
            const [name, text] = sent[0];
            return {ok: true, hadRuntime, hasRuntime: Boolean(globalThis.lmLite?.AudioRecorder), labels,
                    final: [button.textContent, status.textContent], name, saved: sent[1][0],
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
    assert result["name"] == "recording" and result["saved"] == "saved"
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
