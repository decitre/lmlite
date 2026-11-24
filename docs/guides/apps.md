# Build an app

An app is a notebook that [Voici](https://github.com/voila-dashboards/voici) renders as a page: the code is hidden, the widgets are shown, and the kernel runs in the browser, as in the Lab sites. lmLite has two, in the `apps/` directory:

- **`camera-objects`**: the camera shows its picture; **Take a shot** finds the objects in it with `detr-resnet-50` (43 MB) and shows them, with **Retry** to go back to the camera.
- **`speak-french`**: **Record** an English sentence, **Stop**, and the browser transcribes it (`whisper-tiny.en`), translates it (`opus-mt-en-fr`) and says the translation in French (`mms-tts-fra`, non-commercial licence), with **Retry** for the next one.

They are published at `https://decitre.github.io/lmlite/apps/voici/render/camera-objects.html` and `.../speak-french.html`.

## Build them

`scripts/lite-build.sh` builds the apps as a site of their own, `workspace/public/apps`, with a xeus-python kernel and `jl-forge`. They are not part of the Lab site because `voici-core` holds `jupyterlite-core` at 0.7, and the Lab sites use 0.8 (see jl-forge's [Voici guide](https://github.com/decitre/jl-forge/blob/main/docs/VOICI.md)). To build only the apps while you work on one:

```bash
LITE_SITES=voici bash scripts/lite-build.sh
```

The `voici-smoke` task opens both apps in Chromium with a fake camera and a fake microphone and goes through the buttons.

## Write one

A notebook in `apps/` is a Markdown cell for the title and the instructions, and a code cell that builds the widgets. The pattern of both apps is one button, top left, whose label follows the state (`Take a shot`, `Retry`; `Record`, `Stop`, `Working...`, `Retry`):

- **The widgets in app mode.** `CameraCapture(controls=False, autostart=True, show_photo=False)` shows only the live picture and starts it; `camera.take()` takes the photo. `AudioRecorder(controls=False)` shows only a status line; `recorder.record()` and `recorder.stop()` drive it, and its `raw` trait is the Raw input box. The `running` trait of the camera and the `active` trait of the recorder say when the picture plays and when the microphone records, so the button can follow them.
- **No progress bars.** `quiet=True` in `pipeline(...)` shows no progress bar and no loading messages, which would appear under the app.
- **Show results with `widgets.Image` and `widgets.HTML`.** On xeus-python in Voici, an `Output` widget stays empty when it is written from a callback ([voici#127](https://github.com/voila-dashboards/voici/issues/127)).
- **One model at a time on a phone.** `speak-french` loads each pipeline when it needs it and disposes it as soon as it has answered, so three models are never in memory together.
- **A browser can refuse to autoplay audio**, so the voice has controls to tap.
