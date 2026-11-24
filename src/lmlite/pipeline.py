# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Transformers.js task pipelines, and the audio and image inputs and outputs they use."""

from ._bridge import _bytes_of, _js_callable, _to_js, _to_python_deep
from ._session import Session

# Tasks that return images (PNG bytes, or NumPy arrays with as_array=True); their inputs are images too.
IMAGE_OUTPUT_TASKS = ("background-removal", "depth-estimation", "image-segmentation", "image-to-image")


# Tasks whose inputs are images; keep in sync with lmlite.js.
IMAGE_TASKS = (
    "image-classification",
    "zero-shot-image-classification",
    "image-feature-extraction",
    "object-detection",
    *IMAGE_OUTPUT_TASKS,
)


# Text-to-speech under both names Transformers.js accepts; keep in sync with lmlite.js.
SPEECH_TASKS = ("text-to-speech", "text-to-audio")


# Tasks whose input is audio: a WAV URL or file, WAV bytes, or mono samples at 16 kHz.
AUDIO_INPUT_TASKS = ("automatic-speech-recognition", "audio-classification")


# Transformers.js tasks available through pipeline(); keep in sync with lmlite.js.
SUPPORTED_TASKS = (
    "zero-shot-classification",
    *AUDIO_INPUT_TASKS,
    "translation",
    *IMAGE_TASKS,
    *SPEECH_TASKS,
)


# Sampling rate expected by Whisper and most Transformers.js speech recognition models.
AUDIO_SAMPLING_RATE = 16000


def _read_wav(source, sampling_rate=AUDIO_SAMPLING_RATE):
    """Decode PCM WAV bytes or a file path into mono float samples at ``sampling_rate``."""
    import array
    import io
    import sys
    import wave

    fileobj = io.BytesIO(source) if isinstance(source, (bytes, bytearray, memoryview)) else source
    try:
        with wave.open(fileobj, "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Audio must be a PCM WAV file: {exc}") from exc

    if width == 1:
        samples = [(byte - 128) / 128 for byte in frames]
    elif width == 3:
        samples = [int.from_bytes(frames[i : i + 3], "little", signed=True) / 8388608 for i in range(0, len(frames), 3)]
    else:
        typecode = {2: "h", 4: "i"}.get(width)
        if typecode is None:
            raise ValueError(f"Unsupported WAV sample width: {width} bytes")
        pcm = array.array(typecode, frames)
        if sys.byteorder == "big":
            pcm.byteswap()
        scale = float(2 ** (8 * width - 1))
        samples = [value / scale for value in pcm]

    if channels > 1:
        samples = [sum(samples[i : i + channels]) / channels for i in range(0, len(samples), channels)]
    if rate != sampling_rate and samples:
        # Linear interpolation is sufficient for speech; resample offline for best quality.
        count = max(1, round(len(samples) * sampling_rate / rate))
        step = (len(samples) - 1) / max(1, count - 1)
        resampled = []
        for i in range(count):
            position = i * step
            left = int(position)
            right = min(left + 1, len(samples) - 1)
            resampled.append(samples[left] + (samples[right] - samples[left]) * (position - left))
        samples = resampled
    return samples


def _audio_input(audio):
    """Normalize the input of an audio task to a URL string or a list of float samples."""
    import os

    if isinstance(audio, str):
        if audio.startswith(("http://", "https://", "blob:", "data:")):
            return audio
        if os.path.isfile(audio):
            return _read_wav(audio)
        raise FileNotFoundError(
            f"Audio file not found: {audio!r}. Pass an HTTP(S) URL, a path to a PCM WAV file, "
            f"WAV bytes, or mono samples at {AUDIO_SAMPLING_RATE} Hz."
        )
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return _read_wav(audio)
    return [float(sample) for sample in audio]


_IMAGE_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"BM")


def _encoded_image(data):
    data = bytes(data)
    is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if not (is_webp or data.startswith(_IMAGE_SIGNATURES)):
        raise ValueError("The image could not be decoded: expected JPEG, PNG, WebP, GIF or BMP bytes.")
    return data


def _image_input(image):
    """Normalize one image to a URL string, encoded bytes, or ``["raw", pixels, width, height, channels]``."""
    import io
    import os

    if isinstance(image, str):
        if image.startswith(("http://", "https://", "blob:", "data:")):
            return image
        if os.path.isfile(image):
            with open(image, "rb") as file:
                return _encoded_image(file.read())
        raise FileNotFoundError(
            f"Image file not found: {image!r}. Pass an HTTP(S) URL, a file path, encoded image bytes, "
            "a uint8 NumPy array of shape (H, W, 3 or 4), or a PIL image."
        )
    if isinstance(image, (bytes, bytearray, memoryview)):
        return _encoded_image(image)
    if hasattr(image, "save") and hasattr(image, "mode"):  # PIL image
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    if hasattr(image, "shape") and hasattr(image, "dtype"):  # NumPy array
        if str(image.dtype) != "uint8" or len(image.shape) != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"A NumPy image must be uint8 with shape (H, W, 3 or 4), got {image.dtype} {image.shape}.")
        height, width, channels = (int(n) for n in image.shape)
        return ["raw", image.tobytes(), width, height, channels]
    raise TypeError(f"Unsupported image input: {type(image).__name__}")


def _image_inputs(inputs):
    """One image, or a list of images for a batch."""
    single = (
        isinstance(inputs, (str, bytes, bytearray, memoryview)) or hasattr(inputs, "shape") or hasattr(inputs, "mode")
    )
    return _image_input(inputs) if single else [_image_input(item) for item in inputs]


def _unpack_images(result, as_array=False):
    """An image-output task's result: ``[JSON structure, one buffer per image or tensor]`` back into dicts and lists.

    Each image is PNG bytes, or a ``uint8`` array of shape (height, width, channels) with ``as_array``; a float
    tensor (the raw depth) is a float32 array, or nested lists without NumPy.
    """
    import json

    try:
        import numpy as np
    except ImportError:
        np = None
        if as_array:
            raise ModuleNotFoundError("as_array needs NumPy: pip install numpy", name="numpy") from None
    buffers = [_bytes_of(buffer) for buffer in result[1:]]

    def build(node):
        if isinstance(node, dict) and "$buffer" in node:
            data = buffers[node["$buffer"]]
            if node.get("dtype") == "float32":
                rows, columns = node["shape"]
                if np is not None:
                    return np.frombuffer(data, dtype="<f4").reshape(rows, columns).astype(np.float32)
                import array

                flat = array.array("f")
                flat.frombytes(data)
                return [flat[row * columns : (row + 1) * columns].tolist() for row in range(rows)]
            if as_array:
                return (
                    np.frombuffer(data, dtype=np.uint8).reshape(node["height"], node["width"], node["channels"]).copy()
                )
            return data
        if isinstance(node, list):
            return [build(item) for item in node]
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        return node

    return build(json.loads(str(result[0])))


class Speech:
    """Synthesised speech, returned by the text-to-speech pipelines.

    Evaluating a ``Speech`` in a notebook cell shows an audio player.

    Attributes:
        samples: Mono float32 samples: a NumPy array when NumPy is installed, else an ``array.array("f")``.
        sampling_rate: Samples per second.

    Example:
        .. code-block:: python

            tts = await pipeline("text-to-speech", model="Xenova/mms-tts-eng")
            speech = await tts("Hello from the browser.")
            speech.duration, speech.sampling_rate
            open("hello.wav", "wb").write(speech.to_wav())
    """

    def __init__(self, samples, sampling_rate):
        self.samples = samples
        self.sampling_rate = sampling_rate

    @classmethod
    def _from_bytes(cls, sampling_rate, raw):
        """From little-endian float32 bytes, which is how the browser hands over the samples."""
        try:
            import numpy as np
        except ImportError:
            import array

            samples = array.array("f")
            samples.frombytes(raw)
        else:
            samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        return cls(samples, int(sampling_rate))

    @property
    def duration(self):
        """Length in seconds."""
        return len(self.samples) / self.sampling_rate

    def __repr__(self):
        return f"Speech(duration={self.duration:.2f}s, sampling_rate={self.sampling_rate})"

    def to_wav(self):
        """The speech as a WAV file.

        Returns:
            The bytes of a 16-bit PCM mono WAV file at ``sampling_rate``.
        """
        import array
        import io
        import sys
        import wave

        pcm = array.array("h", (max(-32768, min(32767, round(float(x) * 32767))) for x in self.samples))
        if sys.byteorder == "big":
            pcm.byteswap()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sampling_rate)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def _repr_html_(self):
        import base64

        data = base64.b64encode(self.to_wav()).decode("ascii")
        return f'<audio controls src="data:audio/wav;base64,{data}"></audio>'


class Pipeline:
    """A loaded Transformers.js task, created with ``pipeline()``.

    Call it to run the task on one input (``await pipe(input, **options)``), or use ``map()`` for many.

    Attributes:
        task: The task name, such as ``"object-detection"``.
        model: The model ID or name that was loaded.
    """

    def __init__(self, llm, task, model):
        self.llm = llm
        self.task = task
        self.model = model

    def __repr__(self):
        return f"Pipeline(task={self.task!r}, model={self.model!r})"

    async def __call__(self, inputs, **kwargs):
        """Run the task on one input, or on a list of inputs for the tasks that take a list.

        Args:
            inputs: What the task takes. Text tasks (``zero-shot-classification``, ``translation``): a string or a
                list of strings. Audio tasks (``automatic-speech-recognition``, ``audio-classification``): an HTTP(S)
                URL, a path to a PCM WAV file, WAV bytes, or mono samples at 16 kHz. Image tasks: encoded image
                bytes, an HTTP(S) URL, a file path, a NumPy array of shape ``(H, W, 3)`` or ``(H, W, 4)``, a PIL image,
                or a list of these. Text to speech: one string.
            **kwargs: Task options, forwarded to Transformers.js. They must be JSON-serializable, for example
                ``candidate_labels``, ``top_k``, ``threshold`` or ``max_new_tokens``. The image-output tasks also take
                ``as_array=True``, which returns NumPy pixels instead of PNG bytes.

        Returns:
            The task's result as plain Python objects: dicts and lists, ``Speech`` for text to speech, PNG bytes (or
            arrays) for the image-output tasks.

        Raises:
            TypeError: A list was given to text to speech, which takes one string (use ``map()``).
            ValueError: The input cannot be read, for example bytes that are not an image or a WAV file that is not PCM.
        """
        import json

        if self.task in AUDIO_INPUT_TASKS:
            inputs = _audio_input(inputs)
        elif self.task in IMAGE_TASKS:
            inputs = _image_inputs(inputs)
        elif self.task in SPEECH_TASKS and not isinstance(inputs, str):
            raise TypeError(f"{self.task} takes one text; use map() for several, got {type(inputs).__name__}")
        elif not isinstance(inputs, str):
            inputs = list(inputs)
        if not isinstance(inputs, str):
            inputs = _to_js(inputs)
        options = json.dumps(kwargs, allow_nan=False)
        result = await self.llm.js.runTask(inputs, options)
        return self._result(_to_python_deep(result), as_array=kwargs.get("as_array", False))

    def _result(self, result, as_array=False):
        """A task result as a Python object: speech arrives as ``[sampling_rate, float32 bytes]``, an image result
        as ``[JSON structure, buffers]``."""
        if self.task in SPEECH_TASKS:
            return Speech._from_bytes(result[0], _bytes_of(result[1]))
        if self.task in IMAGE_OUTPUT_TASKS:
            return _unpack_images(result, as_array)
        return result

    def _prepare(self, item):
        """One input of ``map()`` in the form the JavaScript task takes."""
        if self.task in AUDIO_INPUT_TASKS:
            return _audio_input(item)
        if self.task in IMAGE_TASKS:
            return _image_input(item)
        if not isinstance(item, str):
            raise TypeError(f"map() inputs for {self.task} must be strings, got {type(item).__name__}")
        return item

    async def map(self, inputs, batch_size=8, progress_callback=None, on_error="raise", **kwargs):
        """Run the task over many inputs, with the loop in JavaScript so a notebook makes one call.

        Args:
            inputs: Any iterable of inputs, each in the form the task takes on its own call.
            batch_size: How many inputs go to each Transformers.js call; ``1`` runs them one by one. The audio,
                image-output and text-to-speech tasks always run one input per call.
            progress_callback: Called as ``progress_callback(done, total)`` after each batch.
            on_error: ``"raise"`` raises the first error; ``"skip"`` leaves failed inputs out of the results (so the
                results can be shorter than the inputs); ``"return"`` puts an exception object where a failed input's
                result would be. A failed batch is retried input by input.
            **kwargs: Task options, as for a normal call.

        Returns:
            The results in input order. If ``cancel()`` stopped the run, the results of the inputs that ran.

        Raises:
            ValueError: ``batch_size`` is not a positive integer, or ``on_error`` is not one of the three values.
            RuntimeError: Another ``map()`` is already running on this pipeline.
        """
        import json

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if on_error not in ("raise", "skip", "return"):
            raise ValueError(f"on_error must be 'raise', 'skip' or 'return', got {on_error!r}")
        items = list(inputs)
        prepared = []  # (index, input) for the inputs JavaScript will run
        failures = {}  # index -> exception, for inputs that could not be prepared
        for index, item in enumerate(items):
            try:
                prepared.append((index, self._prepare(item)))
            except Exception as exc:
                if on_error == "raise":
                    raise
                failures[index] = exc

        def report(done, total):
            if progress_callback is not None:
                progress_callback(int(done) + len(failures), len(items))

        outcome = {"results": [], "errors": []}
        if prepared:
            with _js_callable(report) as callback:
                outcome = _to_python_deep(
                    await self.llm.js.runTasks(
                        _to_js([value for _, value in prepared]),
                        batch_size,
                        json.dumps(kwargs, allow_nan=False),
                        callback,
                        on_error != "raise",
                    )
                )
        results = outcome["results"]
        errors = {position: message for position, message in outcome["errors"]}
        # A cancelled run covers a prefix of the inputs.
        done = len(results)
        limit = len(items) if done == len(prepared) else prepared[done - 1][0] + 1 if done else 0
        positions = {index: position for position, (index, _) in enumerate(prepared)}
        output = []
        for index in range(limit):
            if index in failures:
                value = failures[index]
            elif positions[index] in errors:
                value = RuntimeError(errors[positions[index]])
            else:
                value = self._result(results[positions[index]], as_array=kwargs.get("as_array", False))
            if on_error == "skip" and isinstance(value, Exception):
                continue
            output.append(value)
        return output

    async def cancel(self):
        """Stop a running ``map()`` after its current batch.

        Returns:
            ``False`` if nothing was running. The pipeline stays usable.
        """
        return await self.llm.cancel()

    async def dispose(self):
        """Release the loaded model without deleting its cached files.

        Await any running call first. Calling it again is safe.
        """
        await self.llm.dispose()

    async def export_models(self):
        """Save the model from the browser cache as a zip file in ``/drive/models``, ready to download from JupyterLite."""
        await self.llm.export_models()


async def pipeline(task, model, **options):
    """Load a Transformers.js task and return a callable ``Pipeline``.

    ``pipeline()`` is the way to run everything except text generation, embeddings and text classification (see
    ``LLM``, ``Embedder`` and ``Classifier``). The supported tasks are listed in ``SUPPORTED_TASKS``; the pipeline task
    table in the reference says what each one takes and returns.

    Args:
        task: One of ``SUPPORTED_TASKS``, for example ``"zero-shot-classification"``, ``"translation"``,
            ``"automatic-speech-recognition"``, ``"object-detection"`` or ``"text-to-speech"``.
        model: A Hugging Face model ID such as ``"Xenova/detr-resnet-50"``, or a short name, which gets the default
            ``Xenova/`` namespace, or a model hosted with the site.
        **options: Loading options, as for ``LLM.create()``: ``backend`` (``"wasm"``, the default, or ``"webgpu"``),
            ``dtype``, ``dtype_fallbacks``, ``hosted_models_url``, ``auto_detect_local``, ``local_models_path``,
            ``progress_callback`` and ``quiet``.

    Returns:
        The loaded ``Pipeline``. Call it with the task's input, and ``dispose()`` it when you are done.

    Raises:
        ValueError: ``task`` is not supported, or ``model`` is empty.
        TypeError: ``options`` contains a model option such as ``generator_model``; this function loads one task.

    Example:
        .. code-block:: python

            classifier = await pipeline("zero-shot-classification", model="mobilebert-uncased-mnli")
            await classifier("This notebook runs in a browser", candidate_labels=["technology", "sports"])
            await classifier.dispose()
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task {task!r}. Supported pipeline tasks are: {', '.join(SUPPORTED_TASKS)}. "
            "Use LLM for text generation, Embedder for embeddings and Classifier for text classification."
        )
    if not model:
        raise ValueError(f"Task {task!r} requires a model.")
    model_options = {"generator_model", "embedding_model", "classification_model", "task", "task_model"}
    conflicting = sorted(model_options & options.keys())
    if conflicting:
        raise TypeError(f"pipeline() does not accept {', '.join(conflicting)}; use LLM.create() instead.")
    session = await Session.create(task=task, task_model=model, **options)
    return Pipeline(session, task, model)
