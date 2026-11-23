# Run task pipelines

`LLM.pipeline(task, model=...)` loads one additional Transformers.js task and returns an awaitable callable. Supported tasks are listed in `lmlite.SUPPORTED_TASKS`: `"zero-shot-classification"`, `"automatic-speech-recognition"`, `"translation"` and the vision tasks `"image-classification"`, `"zero-shot-image-classification"`, `"image-feature-extraction"` and `"object-detection"`, the image-output tasks `"background-removal"`, `"depth-estimation"`, `"image-segmentation"` and `"image-to-image"`, and `"text-to-speech"` (also spelled `"text-to-audio"`). Other task names raise a `ValueError` that lists the supported tasks; generation, embeddings and text classification keep their dedicated methods.

```python
classifier = await LLM.pipeline("zero-shot-classification", model="mobilebert-uncased-mnli")
result = await classifier("This notebook runs in a browser", candidate_labels=["technology", "sports"])
# {'sequence': 'This notebook runs in a browser', 'labels': ['technology', 'sports'], 'scores': [0.93, 0.07]}

translator = await LLM.pipeline("translation", model="opus-mt-en-fr")
await translator("Hello, how are you?", max_new_tokens=64)
# {'translation_text': 'Bonjour, comment allez-vous ?'}
await translator(["Hello!", "Goodbye!"])
# [{'translation_text': 'Bonjour !'}, {'translation_text': 'Au revoir !'}]

transcriber = await LLM.pipeline("automatic-speech-recognition", model="whisper-tiny.en")
await transcriber("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav")
# {'text': ' And so my fellow Americans ask not what your country can do for you...'}

clip = await LLM.pipeline("zero-shot-image-classification", model="clip-vit-base-patch32")
await clip(jpeg_bytes, candidate_labels=["a photo of a person's face", "a photo of food"])
# [{'label': 'a photo of food', 'score': 0.97}, {'label': "a photo of a person's face", 'score': 0.03}]

detector = await LLM.pipeline("object-detection", model="detr-resnet-50")
await detector(jpeg_bytes, threshold=0.5)
# [{'label': 'cat', 'score': 0.99, 'box': {'xmin': 11, 'ymin': 54, 'xmax': 316, 'ymax': 472}}, ...]

await classifier.dispose()
```

The models each task was tested with, their licences and the ones known not to work are in [MODELS.md](../MODELS.md#task-pipelines-for-llmpipelinetask-model).

Loading options are the same as for `LLM.create()`: `backend`, `dtype`, `dtype_fallbacks`, `hosted_models_url`, `auto_detect_local` and `local_models_path`. Model names, hosted files, browser caching, progress display and `export_models()` work the same way too. Load failures name the task, model and backend.

Image tasks accept encoded image bytes (JPEG, PNG, WebP, GIF or BMP), an HTTP(S) URL, a path to a file in the JupyterLite file system, a `uint8` NumPy array of shape `(H, W, 3)` or `(H, W, 4)`, a PIL image, or a list of any of these. Encoded bytes are the recommended form: they cross the Python/JavaScript bridge as one buffer, and a 250 px JPEG is about 15 KB against about 190 KB of RGB pixels. Bytes that are not an image raise a `ValueError`. NumPy arrays and PIL images need NumPy and Pillow in the kernel: `pip install 'lmlite[images]'`, or the same extra on a JupyterLite site build. Encoded bytes and URLs need neither. Image tasks are not available in the native backend. Output sizes can be large for detectors with many anchors, so call a pipeline once per image rather than in large batches.

The four image-output tasks return images as PNG bytes, which `IPython.display.Image(data=png)` shows and `open(..., "wb")` saves; pass `as_array=True` for a `uint8` NumPy array of shape (height, width, channels) instead (it needs NumPy). Each image crosses the Python/JavaScript bridge as one buffer, and the PNG is encoded in the browser with `OffscreenCanvas`, so Pillow is not needed. The raw depth values (`predicted_depth`) are a float32 NumPy array, or nested lists when NumPy is not installed. Results can be large: for the 612 x 408 test photo the cut-out is 381 KB, the raw depth values 1 MB and the 2x upscale 2.5 MB, so use `map()` and not one huge batch for many photos. `map()` runs these tasks one photo per call.

Text to speech returns a `Speech`: `samples` (a NumPy `float32` array when NumPy is installed, else an `array.array("f")`), `sampling_rate`, `duration` and `to_wav()` (16-bit PCM mono WAV bytes). Evaluating a `Speech` in a notebook cell shows an audio player. The samples cross the Python/JavaScript bridge as one buffer. A call takes one text and raises `TypeError` for a list; use `map()` for several, which returns one `Speech` per text. Only models that Transformers.js runs through its text-to-speech pipeline work: `Xenova/mms-tts-eng` is tested (CC-BY-NC-4.0, non-commercial use only). In Transformers.js 4.3.2, `Xenova/speecht5_tts` fails with "Missing the following inputs: speaker_embeddings, encoder_hidden_states, output_sequence" (with `fp32` and `q8`, with embeddings as a URL or a `Float32Array`), and Kokoro is not supported by the pipeline ("Unsupported model type style_text_to_speech_2"). The speech model can be hosted with a site like any other: a models directory with a `manifest.json` that lists `Xenova/mms-tts-eng` and the model's files (see [MODELS.md](../MODELS.md#notes-on-the-tested-models)) makes `LLM.pipeline("text-to-speech", model="Xenova/mms-tts-eng", hosted_models_url=...)` load without contacting the Hub. Sound generation such as MusicGen is not covered.

Speech recognition accepts an HTTP(S) URL, a path to a PCM WAV file (e.g. a file uploaded to the JupyterLite file browser), WAV bytes, or a sequence of mono float samples at 16 kHz. WAV files are decoded in Python, mixed down to mono and linearly resampled to 16 kHz. Notebook kernels run in a Web Worker without `AudioContext`, so URLs are fetched and decoded the same way and must point to WAV files. Compressed formats (MP3, OGG, M4A) are not decoded in the kernel: convert them to WAV first. URL fetches must be allowed by CORS. Whisper processes 30-second windows, so pass `chunk_length_s=30` for longer recordings.

## Many inputs: `map()`

`await pipe.map(inputs, batch_size=8, progress_callback=None, on_error="raise", **options)` runs the task over any iterable of inputs with the loop in JavaScript, so a notebook makes one call instead of one per input, and can show progress while it stays responsive. Each input is whatever the task accepts on its own (a string, an image, an audio clip); keyword arguments are task options, as in a normal call.

```python
bar = widgets.IntProgress(max=len(photos))
results = await classifier.map(photos, batch_size=8, progress_callback=lambda done, total: setattr(bar, "value", done), top_k=3)
```

- `batch_size` inputs go to each Transformers.js call; `1` runs them one by one, still in JavaScript. Speech recognition always runs one clip per call.
- `progress_callback(done, total)` is called after each batch, so `ceil(len(inputs) / batch_size)` times.
- `on_error`: `"raise"` raises the first error. `"skip"` leaves failed inputs out of the results (so the results are shorter than the inputs), and `"return"` puts an exception object in a failed input's place. When a batch fails, its inputs are retried one by one, so one bad image does not lose the others. An input that cannot be prepared in Python (undecodable image bytes, a missing file) counts as failed too, and counts as done in the progress.
- `await pipe.cancel()` stops after the current batch, and `map()` returns the results so far. The pipeline stays usable. Only one `map()` runs on a pipeline at a time.
- Batched results can differ from single calls by rounding error, because batches are padded together.

## Recording the microphone

`lmlite.audio.AudioRecorder` is an [anywidget](https://anywidget.dev/) with a Record/Stop button, for notebooks on a laptop or a tablet. Install it with `pip install 'lmlite[audio]'` (a JupyterLite site build takes the same extra, and `lite-build.sh` does).

```python
from lmlite.audio import AudioRecorder

recorder = AudioRecorder()
recorder                           # tap Record, speak, tap Stop
wav = await recorder.wait()        # the last recording, waiting for one if there is none (recorder.clear() to forget it)
text = (await transcriber(wav))["text"]
```

A notebook kernel runs in a Web Worker, which cannot open the microphone, so the widget records on the page: its JavaScript loads `llm.js` there (`globalThis.lmLite.AudioRecorder`), records with `MediaRecorder`, decodes and resamples with `AudioContext`, and returns a 16 kHz mono 16-bit WAV, which speech recognition takes as bytes. The page must be on HTTPS or localhost, and the browser asks for microphone permission the first time. `recorder.info` describes the last recording: `seconds`, `peak` (0 to 1; near 0 means silence), the browser's `channels`, `sampleRate` and `mimeType`, and the `raw` option and `gain` applied. The widget says when a recording is silent, and what to check, starting with the system's input level: a microphone whose input level is at 0 in the system settings (as happened on macOS in Firefox) records pure silence, and a wrong selected input or a missing browser permission do the same. If those are fine, there is one more cause. By default the browser processes the microphone (echo cancellation, noise suppression, gain control), and any of them mixes the input to mono first: a stereo input whose channels have opposite polarity then becomes exact silence (reproduced in Chromium with a test file, not checked on real hardware). Tick **Raw input** to turn that processing off, keep the channels (a channel mix that cannot cancel is used) and level a quiet recording to a peak of 0.9, at most 20 times; the price is no noise suppression.

The recording reaches the kernel as base64 text in the `recording` trait (`audio` is the decoded bytes), because in xeus-python a binary trait set by the browser never arrives.

## Taking a photo with the camera

`lmlite.camera.CameraCapture` is an [anywidget](https://anywidget.dev/) with Start camera, Take photo and Stop camera buttons and a Rear/Front choice, for notebooks on a laptop, tablet or phone. Install it with `pip install 'lmlite[camera]'` (a JupyterLite site build takes the same extra, and `lite-build.sh` does).

```python
from lmlite.camera import CameraCapture

camera = CameraCapture()
camera                              # Start camera, aim, Take photo
jpeg = await camera.wait()          # the last photo, waiting for one if there is none (camera.clear() to forget it)
detections = await detector(jpeg, threshold=0.6)   # any image pipeline takes the bytes
```

The photo is a JPEG of at most 1024 pixels on the longer side, taken from the page's video preview. A notebook kernel runs in a Web Worker, which cannot open the camera, so the widget works on the page, as `AudioRecorder` does. The page must be on HTTPS or localhost, and the browser asks for camera permission the first time. "Rear camera" is a preference: a device with one camera uses it. `camera.info` holds the photo's `width`, `height` and `bytes`. The photo reaches the kernel as base64 text in the `photo` trait (`image` is the decoded bytes), for the same reason as the recording. The widget shows one photo at a time and has no live detection: each detection is one call to the pipeline, which takes a few seconds with `detr-resnet-50` on WebAssembly.

Expect a pipeline in a browser kernel to be slower than native inference, especially sequence-to-sequence tasks (translation and speech recognition), which decode token by token. WebGPU can help when the browser, model and dtype support it. Each pipeline holds its own model in browser memory; call `dispose()` before loading the next one.
