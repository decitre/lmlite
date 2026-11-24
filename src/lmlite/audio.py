# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Record the microphone in a JupyterLite notebook. Imported separately: it needs ``pip install 'lmlite[audio]'``.

Kernels run in a Web Worker, which cannot open the microphone, so the widget works on the page: its JavaScript loads
the same ``lmlite.js`` the kernel uses (``globalThis.lmLite.AudioRecorder``), records, and hands the kernel a WAV file.
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
    rawLabel.title = "Turn off the browser's echo cancellation, noise suppression and gain control. Try it when a recording is silent and the input level in the system settings is fine.";
    rawLabel.append(raw, " Raw input");
    const status = document.createElement("span");
    status.style.marginLeft = "0.75em";
    button.textContent = "Record";
    el.append(button, rawLabel, status);
    raw.checked = model.get("raw") === true;
    raw.onchange = () => {
      model.set("raw", raw.checked);
      model.save_changes();
    };
    model.on?.("change:raw", () => {
      raw.checked = model.get("raw") === true;
    });
    if (model.get("controls") === false) {
      // An app drives the recording with its own buttons (record(), stop()): the status line stays.
      button.style.display = rawLabel.style.display = "none";
    }

    const begin = async () => {
      await recorder.start({ raw: raw.checked });
      raw.disabled = true;
      status.textContent = "Recording...";
      button.textContent = "Stop";
      model.set("active", true);
      model.save_changes();
    };
    const finish = async () => {
      status.textContent = "Processing...";
      const wav = await recorder.stop();
      let text = "";
      for (let i = 0; i < wav.length; i += 0x8000) {
        text += String.fromCharCode(...wav.subarray(i, i + 0x8000));
      }
      raw.disabled = false;
      const { seconds, peak, channels, sampleRate, mimeType } = recorder.info;
      model.set("info", recorder.info);  // before the recording, which wakes up whoever waits for it
      model.set("active", false);
      model.set("recording", btoa(text));
      model.save_changes();
      status.textContent = peak < 0.005
        ? `Recorded ${seconds.toFixed(1)} s, but it is silent. Check the microphone's input level in the system settings (it may be at 0), that the right input is selected, and that the browser may use the microphone. ${raw.checked ? "" : "If those are fine, tick Raw input and record again. "}(${channels} channel(s), ${sampleRate} Hz, ${mimeType})`
        : `Recorded ${seconds.toFixed(1)} s`;
      button.textContent = "Record again";
    };
    const run = async (action) => {
      button.disabled = true;
      try {
        await action();
      } catch (error) {
        raw.disabled = false;
        status.textContent = String(error?.message ?? error);
        model.set("active", recorder.recording);
        model.save_changes();
      } finally {
        button.disabled = false;
      }
    };
    button.onclick = () => run(recorder.recording ? finish : begin);
    model.on?.("change:recording", () => {
      if (!model.get("recording") && !recorder.recording) {  // clear(): ready to record again
        status.textContent = "";
        button.textContent = "Record";
      }
    });
    model.on?.("msg:custom", (message) => {
      if (message.do === "record" && !recorder.recording) run(begin);
      if (message.do === "stop" && recorder.recording) run(finish);
    });
    return () => {
      if (recorder.recording) {
        recorder.stop().catch(() => {});  // release the microphone when the output is removed
      }
    };
  },
};
"""


class AudioRecorder(anywidget.AnyWidget):
    """A Record and Stop button that records the microphone of a laptop or tablet, in a notebook.

    Install it with ``pip install 'lmlite[audio]'``. The kernel runs in a Web Worker, which cannot open the
    microphone, so the widget records on the page and hands the kernel the recording. The page must be on HTTPS (or
    localhost), and the browser asks for permission the first time.

    Tick **Raw input** to turn off the browser's echo cancellation, noise suppression and gain control, which can
    silence a recording on some devices. ``info`` describes the last recording: ``seconds``, ``peak`` (0 to 1; near 0
    means silence), ``channels``, ``sampleRate``, ``mimeType``, and ``raw`` and ``gain`` for a raw recording.

    Example:
        .. code-block:: python

            recorder = AudioRecorder()
            recorder                      # tap Record, speak, tap Stop
            wav = await recorder.wait()   # 16 kHz mono 16-bit WAV bytes
            text = (await transcriber(wav))["text"]
    """

    _esm = _ESM
    runtime = traitlets.Unicode().tag(sync=True)
    # The WAV as base64 text: in xeus-python, a binary trait set by the browser never reaches the kernel, while
    # a string does.
    recording = traitlets.Unicode("").tag(sync=True)
    # Show the Record button and the Raw input box (the default). An app turns them off and calls record() and stop().
    controls = traitlets.Bool(True).tag(sync=True)
    # Turn off the browser's echo cancellation, noise suppression and gain control: the Raw input box.
    raw = traitlets.Bool(False).tag(sync=True)
    # True while the microphone is recording: an app with its own buttons follows it.
    active = traitlets.Bool(False).tag(sync=True)
    # How the browser recorded it: seconds, peak (0 to 1; near 0 means silence), channels, sampleRate, mimeType,
    # raw (the "Raw input" box was ticked) and gain (the factor applied to a raw recording).
    info = traitlets.Dict().tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(runtime=files("lmlite").joinpath("lmlite.js").read_text(), **kwargs)
        self._waiters = []
        self.observe(self._received, names="recording")

    @property
    def audio(self):
        """The last recording.

        Returns:
            The bytes of a 16 kHz mono 16-bit WAV file, or ``b""`` if there is none. Speech recognition and audio
            classification pipelines take it as input.
        """
        return base64.b64decode(self.recording)

    def _received(self, change):
        if change["new"]:
            waiters, self._waiters = self._waiters, []
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(self.audio)

    async def wait(self):
        """The last recording, waiting for one if there is none yet.

        Returns:
            The WAV bytes, as ``audio``.
        """
        if self.audio:
            return self.audio
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        return await waiter

    def record(self):
        """Start recording, as the Record button does. Needs a page that shows the widget; ``stop()`` ends it."""
        self.send({"do": "record"})

    def stop(self):
        """Stop recording, as the Stop button does: the recording arrives in ``audio``, and ``wait()`` returns it."""
        self.send({"do": "stop"})

    def clear(self):
        """Forget the recording, so that ``wait()`` waits for a new one."""
        self.recording = ""
        self.info = {}
