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

from lmlite._request import CaptureCancelled, _Requests  # noqa: F401  (CaptureCancelled: for callers of request())

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
    const bar = document.createElement("div");  // a pending request(): what is asked, then Take photo and Cancel
    bar.style.display = "none";
    const ask = document.createElement("span");
    ask.style.marginRight = "0.75em";
    const shoot = document.createElement("button");
    shoot.textContent = "Take photo";
    shoot.disabled = true;  // like the Take photo button: only a picture that plays can be photographed
    const dismiss = document.createElement("button");
    dismiss.textContent = "Cancel";
    bar.append(ask, shoot, dismiss);
    el.append(controls, video, photo, bar);

    let stream = null;
    let pending = null;  // {id, ownStream}: the request in progress; ownStream is true when it started the camera
    const stop = () => {
      stream?.getTracks().forEach((track) => track.stop());
      stream = null;
      video.srcObject = null;
      video.style.display = "none";
      start.textContent = "Start camera";
      take.disabled = shoot.disabled = true;
      facing.disabled = false;
      model.set("running", false);
      model.save_changes();
    };

    let last = 0;  // the newest request this view has ended: a request is shown once
    const endRequest = () => {
      const release = pending?.ownStream;
      last = Math.max(last, pending?.id ?? 0);
      pending = null;
      bar.style.display = "none";
      if (release) {
        stop();  // the request started the camera: do not leave it on
      }
    };
    const refuse = (error) => {
      if (pending) {
        model.set("error", { id: pending.id, name: String(error?.name ?? "Error"), message: String(error?.message ?? error) });
        model.save_changes();
        endRequest();
      }
    };

    video.onplaying = () => {
      take.disabled = shoot.disabled = false;
      status.textContent = "";
      model.set("running", true);
      model.save_changes();
    };
    video.onclick = () => video.play().catch((error) => {
      status.textContent = String(error?.message ?? error);
    });

    const startCamera = async () => {
      if (stream) {
        return;
      }
      start.disabled = true;
      const asked = pending;  // the request that started the camera, if any
      try {
        const granted = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: facing.value } }, audio: false });
        if (asked && pending !== asked) {
          granted.getTracks().forEach((track) => track.stop());  // the request ended while the permission prompt was open
          return;
        }
        stream = granted;
        video.srcObject = stream;
        video.style.display = "";
        photo.style.display = "none";
        facing.disabled = true;
        start.textContent = "Stop camera";
        status.textContent = "";
        await video.play().catch(() => {
          // A browser can refuse to start the preview without a tap that comes after the permission prompt (seen
          // in Brave on Android): keep the camera on, and start the preview when the picture is tapped.
          status.textContent = "Tap the picture to start the preview.";
        });
      } catch (error) {
        stop();
        status.textContent = String(error?.message ?? error);
        refuse(error);
      } finally {
        start.disabled = false;
      }
    };
    start.onclick = () => (stream ? stop() : startCamera());

    const takePhoto = async () => {
      if (!stream || !video.videoWidth) {
        status.textContent = "The camera is not running yet.";
        return;
      }
      const answering = pending?.id;  // the request this photo answers, if one is waiting for a tap
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
      photo.style.display = model.get("show_photo") === false ? "none" : "";
      model.set("info", { width: canvas.width, height: canvas.height, bytes: bytes.length });  // before the photo, which wakes up whoever waits for it
      model.set("photo", btoa(text));
      if (answering !== undefined && pending?.id === answering) {
        model.set("answered", answering);  // after the photo: the request resolves with it
      }
      model.save_changes();
      status.textContent = `Photo ${canvas.width} x ${canvas.height}`;
      if (answering !== undefined && pending?.id === answering) {
        endRequest();
      }
    };
    take.onclick = shoot.onclick = takePhoto;
    dismiss.onclick = () => {
      if (pending) {
        model.set("cancelled", pending.id);
        model.save_changes();
        endRequest();
      }
    };
    model.on?.("change:photo", () => {
      if (!model.get("photo")) {  // clear(): back to the live picture
        photo.style.display = "none";
        status.textContent = stream ? "" : status.textContent;
      }
    });

    if (model.get("controls") === false) {
      // An app drives the camera with its own buttons (start(), take(), stop()): the status line stays.
      start.style.display = take.style.display = facing.style.display = "none";
    }
    model.on?.("msg:custom", (message) => {
      if (message.do === "start") startCamera();
      if (message.do === "take") takePhoto();
      if (message.do === "stop") stop();
    });
    if (model.get("autostart") === true) {
      startCamera();
    }
    // request(): Python sets pending_request and clears it when the request ends, however it ends.
    const syncRequest = () => {
      const wanted = model.get("pending_request") ?? {};
      if (pending && pending.id !== wanted.id) {
        endRequest();
      }
      if (!pending && wanted.id > last) {
        pending = { id: wanted.id, ownStream: !stream };
        ask.textContent = wanted.prompt;
        bar.style.display = "";
        startCamera();
      }
    };
    model.on?.("change:pending_request", syncRequest);
    syncRequest();  // a request made before the widget was shown

    return stop;  // release the camera when the output is removed
  },
};
"""


class CameraCapture(_Requests, anywidget.AnyWidget):
    """Start camera, Take photo and Stop camera buttons for the camera of a laptop, tablet or phone, in a notebook.

    Install it with ``pip install 'lmlite[camera]'``. The kernel runs in a Web Worker, which cannot open the camera, so the
    widget shows it on the page and hands the kernel a photo. The page must be on HTTPS (or localhost), and the browser
    asks for permission the first time. "Rear camera" is a preference: a device with one camera uses it. ``info``
    holds the last photo's ``width``, ``height`` and ``bytes``.

    To build an app, turn the buttons off and drive the camera from your own: ``CameraCapture(controls=False,
    autostart=True, show_photo=False)`` shows just the live picture and starts it, ``take()`` takes the photo, and
    ``image`` or ``wait()`` returns it.

    To ask the user for a photo from code, such as a tool an assistant calls, ``await camera.request("Tap to take a
    photo of the desk")``: the widget shows the question with Take photo and Cancel buttons and the live picture, and
    nothing is captured until the user taps.

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
    # Show the Start, Take photo and Stop buttons and the camera choice (the default). An app turns them off and
    # calls start(), take() and stop().
    controls = traitlets.Bool(True).tag(sync=True)
    # Start the camera as soon as the widget is shown.
    autostart = traitlets.Bool(False).tag(sync=True)
    # Show the photo under the preview (the default). An app that shows its own result turns it off.
    show_photo = traitlets.Bool(True).tag(sync=True)
    # True while the camera picture plays: an app with its own buttons follows it.
    running = traitlets.Bool(False).tag(sync=True)
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

    def _media(self):
        return self.image

    async def request(self, prompt, timeout=None):
        """Ask the user for a new photo and wait for it.

        The widget shows ``prompt`` with a live picture, a Take photo button and a Cancel button, and starts the camera
        if it is not running. It takes nothing until the user taps. Only a photo taken for this request resolves it,
        not one that was already there. The camera the request started is stopped when it ends, however it ends.

        Args:
            prompt: What the user is asked to do, shown in the widget.
            timeout: Seconds to wait, including the permission prompt, or ``None`` to wait for ever.

        Returns:
            The JPEG bytes, as ``image``.

        Raises:
            CaptureCancelled: The user tapped Cancel.
            TimeoutError: Nobody took a photo within ``timeout`` seconds.
            PermissionError: The browser did not allow the camera.
            RuntimeError: The camera could not start (none found, in use), or a request is already pending.

        Example:
            .. code-block:: python

                try:
                    jpeg = await camera.request("Tap to take a photo of the desk", timeout=60)
                except CaptureCancelled:
                    ...  # the user declined: answer without a photo
        """
        return await self._request(prompt, timeout)

    def start(self):
        """Start the camera, as the Start camera button does. Needs a page that shows the widget."""
        self.send({"do": "start"})

    def take(self):
        """Take a photo, as the Take photo button does: it arrives in ``image``, and ``wait()`` returns it."""
        self.send({"do": "take"})

    def stop(self):
        """Stop the camera and release it, as the Stop camera button does."""
        self.send({"do": "stop"})

    def clear(self):
        """Forget the photo, so that ``wait()`` waits for a new one."""
        self.photo = ""
        self.info = {}
