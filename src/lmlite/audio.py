# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Record the microphone in a JupyterLite notebook. Imported separately: it needs ``pip install 'lmlite[audio]'``.

Kernels run in a Web Worker, which cannot open the microphone, so the widget works on the page: its JavaScript loads
the same ``llm.js`` the kernel uses (``globalThis.lmLite.AudioRecorder``), records, and hands the kernel a WAV file.
"""

import asyncio
import base64
from importlib.resources import files

import anywidget
import traitlets

_ESM = """
export default {
  async render({ model, el }) {
    if (!globalThis.lmLite?.AudioRecorder) {
      // The page has no lmLite yet (the kernel's copy lives in its worker): load the runtime from the widget.
      const url = URL.createObjectURL(new Blob([model.get("runtime")], { type: "text/javascript" }));
      try {
        await import(url);
      } finally {
        URL.revokeObjectURL(url);
      }
    }
    const recorder = new globalThis.lmLite.AudioRecorder();
    const button = document.createElement("button");
    const raw = document.createElement("input");
    raw.type = "checkbox";
    const rawLabel = document.createElement("label");
    rawLabel.style.marginLeft = "0.75em";
    rawLabel.title = "Turn off the browser's echo cancellation, noise suppression and gain control. Try it when a recording is silent.";
    rawLabel.append(raw, " Raw input");
    const status = document.createElement("span");
    status.style.marginLeft = "0.75em";
    button.textContent = "Record";
    el.append(button, rawLabel, status);
    button.onclick = async () => {
      button.disabled = true;
      try {
        if (recorder.recording) {
          status.textContent = "Processing...";
          const wav = await recorder.stop();
          let text = "";
          for (let i = 0; i < wav.length; i += 0x8000) {
            text += String.fromCharCode(...wav.subarray(i, i + 0x8000));
          }
          raw.disabled = false;
          const { seconds, peak, channels, sampleRate, mimeType } = recorder.info;
          model.set("info", recorder.info);  // before the recording, which wakes up whoever waits for it
          model.set("recording", btoa(text));
          model.save_changes();
          status.textContent = peak < 0.005
            ? `Recorded ${seconds.toFixed(1)} s, but it is silent. ${raw.checked ? "" : "Tick Raw input and record again. "}Check that the browser may use the microphone and that the right input is selected in the system settings. (${channels} channel(s), ${sampleRate} Hz, ${mimeType})`
            : `Recorded ${seconds.toFixed(1)} s`;
          button.textContent = "Record again";
        } else {
          await recorder.start({ raw: raw.checked });
          raw.disabled = true;
          status.textContent = "Recording...";
          button.textContent = "Stop";
        }
      } catch (error) {
        raw.disabled = false;
        status.textContent = String(error?.message ?? error);
      } finally {
        button.disabled = false;
      }
    };
    return () => {
      if (recorder.recording) {
        recorder.stop().catch(() => {});  // release the microphone when the output is removed
      }
    };
  },
};
"""


class AudioRecorder(anywidget.AnyWidget):
    """A Record/Stop button. ``audio`` holds the last recording: a 16 kHz mono 16-bit WAV file's bytes, which a
    speech recognition pipeline takes as input. Recording needs a page on HTTPS (or localhost) and the user's
    permission for the microphone; the browser asks the first time.
    """

    _esm = _ESM
    runtime = traitlets.Unicode().tag(sync=True)
    # The WAV as base64 text: in xeus-python, a binary trait set by the browser never reaches the kernel, while
    # a string does.
    recording = traitlets.Unicode("").tag(sync=True)
    # How the browser recorded it: seconds, peak (0 to 1; near 0 means silence), channels, sampleRate, mimeType,
    # raw (the "Raw input" box was ticked) and gain (the factor applied to a raw recording).
    info = traitlets.Dict().tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(runtime=files("lmlite").joinpath("llm.js").read_text(), **kwargs)
        self._waiters = []
        self.observe(self._received, names="recording")

    @property
    def audio(self):
        """The last recording as WAV bytes, or ``b""`` if there is none."""
        return base64.b64decode(self.recording)

    def _received(self, change):
        if change["new"]:
            waiters, self._waiters = self._waiters, []
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(self.audio)

    async def wait(self):
        """The last recording, waiting for one if there is none yet."""
        if self.audio:
            return self.audio
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        return await waiter

    def clear(self):
        """Forget the recording, so ``wait()`` waits for a new one."""
        self.recording = ""
        self.info = {}
