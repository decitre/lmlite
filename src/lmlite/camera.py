# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Take a photo with the camera of a laptop, tablet or phone in a JupyterLite notebook. Imported separately: it needs
``pip install 'lmlite[camera]'``.

Kernels run in a Web Worker, which cannot open the camera, so the widget works on the page: it shows the camera,
takes a JPEG when asked and hands the kernel its bytes, which the image pipelines take as input.
"""

import asyncio
import base64

import anywidget
import traitlets

_ESM = """
const LONGEST_SIDE = 1024;  // pixels: a larger photo costs more to send and the detectors shrink it anyway

export default {
  render({ model, el }) {
    const video = document.createElement("video");
    video.autoplay = video.muted = video.playsInline = true;  // playsInline: Safari on iPhone and iPad would go fullscreen
    video.style.cssText = "display:none;max-width:100%;width:480px";
    const photo = document.createElement("img");
    photo.style.cssText = "display:none;max-width:100%;width:480px";
    const start = document.createElement("button");
    start.textContent = "Start camera";
    const take = document.createElement("button");
    take.textContent = "Take photo";
    take.disabled = true;
    const facing = document.createElement("select");
    facing.title = "Which camera to use when the device has two";
    facing.innerHTML = '<option value="environment">Rear camera</option><option value="user">Front camera</option>';
    const status = document.createElement("span");
    status.style.marginLeft = "0.75em";
    const controls = document.createElement("div");
    controls.append(start, take, facing, status);
    el.append(controls, video, photo);

    let stream = null;
    const stop = () => {
      stream?.getTracks().forEach((track) => track.stop());
      stream = null;
      video.srcObject = null;
      video.style.display = "none";
      start.textContent = "Start camera";
      take.disabled = true;
      facing.disabled = false;
    };

    start.onclick = async () => {
      if (stream) {
        stop();
        return;
      }
      start.disabled = true;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: facing.value } }, audio: false });
        video.srcObject = stream;
        await video.play();
        video.style.display = "";
        photo.style.display = "none";
        facing.disabled = true;
        take.disabled = false;
        start.textContent = "Stop camera";
        status.textContent = "";
      } catch (error) {
        stop();
        status.textContent = String(error?.message ?? error);
      } finally {
        start.disabled = false;
      }
    };

    take.onclick = async () => {
      const scale = Math.min(1, LONGEST_SIDE / Math.max(video.videoWidth, video.videoHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let text = "";
      for (let i = 0; i < bytes.length; i += 0x8000) {
        text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      }
      photo.src = URL.createObjectURL(blob);
      photo.style.display = "";
      model.set("info", { width: canvas.width, height: canvas.height, bytes: bytes.length });  // before the photo, which wakes up whoever waits for it
      model.set("photo", btoa(text));
      model.save_changes();
      status.textContent = `Photo ${canvas.width} x ${canvas.height}`;
    };

    return stop;  // release the camera when the output is removed
  },
};
"""


class CameraCapture(anywidget.AnyWidget):
    """Start camera, Take photo and Stop camera buttons for the camera of a laptop, tablet or phone, in a notebook.

    Install it with ``pip install 'lmlite[camera]'``. The kernel runs in a Web Worker, which cannot open the camera, so the
    widget shows it on the page and hands the kernel a photo. The page must be on HTTPS (or localhost), and the browser
    asks for permission the first time. "Rear camera" is a preference: a device with one camera uses it. ``info``
    holds the last photo's ``width``, ``height`` and ``bytes``.

    Example:
        .. code-block:: python

            camera = CameraCapture()
            camera                        # Start camera, aim, Take photo
            jpeg = await camera.wait()
            detections = await detector(jpeg, threshold=0.6)
    """

    _esm = _ESM
    # The JPEG as base64 text: in xeus-python, a binary trait set by the browser never reaches the kernel, while
    # a string does.
    photo = traitlets.Unicode("").tag(sync=True)
    # What the browser took: width, height (pixels) and bytes.
    info = traitlets.Dict().tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._waiters = []
        self.observe(self._received, names="photo")

    @property
    def image(self):
        """The last photo.

        Returns:
            The bytes of a JPEG file, at most 1024 pixels on the longer side, or ``b""`` if there is none. Any image
            pipeline takes it as input.
        """
        return base64.b64decode(self.photo)

    def _received(self, change):
        if change["new"]:
            waiters, self._waiters = self._waiters, []
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(self.image)

    async def wait(self):
        """The last photo, waiting for one if there is none yet.

        Returns:
            The JPEG bytes, as ``image``.
        """
        if self.image:
            return self.image
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        return await waiter

    def clear(self):
        """Forget the photo, so that ``wait()`` waits for a new one."""
        self.photo = ""
        self.info = {}
