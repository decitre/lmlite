# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Exercise example control flow without downloading models or requiring a browser."""

import ast
import asyncio
import inspect
import json
import random
import re
import sys
import types
from unittest.mock import AsyncMock
from pathlib import Path

import pytest

from lmlite import SUPPORTED_TASKS, ModelInfo

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda path: path.stem)
async def test_notebooks_run_in_order_and_dispose_models(path, monkeypatch):
    instances = []
    tokenizers = []
    prompts = []
    chats = []
    hub_tags = {}  # model -> Hub pipeline tag; unlisted models are text generation
    hosted_checks = []

    class ExampleLLM:
        def __init__(self, config):
            self.config = config
            self.disposed = False
            self.js = types.SimpleNamespace(backend=config.get("backend"))  # the BrowserLLM handle
            instances.append(self)

        @classmethod
        async def create(cls, **config):
            assert all(instance.disposed for instance in instances), "Dispose previous instances before loading another"
            return cls(config)

        async def generate(self, prompt):
            assert not self.disposed and self.config.get("generator_model")
            prompts.append(prompt)
            completion = " Example completion."
            return prompt + completion if self.config.get("return_full_text", True) else completion

        async def chat(self, messages):
            assert not self.disposed and self.config.get("generator_model")
            chats.append(messages)
            return "Example completion."

        async def chat_stream(self, messages):
            self.streaming = True
            try:
                for chunk in (await self.chat(messages)).split(" "):
                    if not self.streaming:
                        return
                    yield chunk + " "
                    await asyncio.sleep(0)
            finally:
                self.streaming = False

        async def chat_tokens(self, messages, alternatives=1):
            async for chunk in self.chat_stream(messages):
                yield {"text": chunk, "prob": 0.6, "alternatives": [{"text": " alt", "prob": 0.3}][:alternatives]}

        async def cancel(self):
            running, self.streaming = getattr(self, "streaming", False), False
            return running

        async def embed(self, text):
            assert not self.disposed and self.config.get("embedding_model")
            return [1.0, 0.0]

        async def similarity_search(self, query, docs, k=None):
            await self.embed(query)
            return [(doc, 0.9) for doc in docs[:k]]

        async def classify(self, text, top_k=None):
            assert not self.disposed and self.config.get("classification_model")
            return [{"label": "POSITIVE", "score": 0.9}][:top_k]

        @classmethod
        async def model_task(cls, model):
            return hub_tags.get(model, "text-generation")

        @classmethod
        async def model_info(cls, model, task=None, backend="wasm", dtype=None):
            task = task or await cls.model_task(model)
            files = [
                {"name": "config.json", "size": 10, "cached": False},
                {"name": "onnx/model.onnx", "size": None, "cached": False},
            ]
            return ModelInfo(model, "main", task, backend, dtype, "partial", None, files)

        @classmethod
        async def hosted_base(cls, model_id, *, hosted_models_url=None):
            hosted_checks.append(model_id)
            return None

        @classmethod
        async def pipeline(cls, task, model, **options):
            assert task in SUPPORTED_TASKS and model
            return await cls.create(task=task, task_model=model, **options)

        async def __call__(self, inputs, **kwargs):
            assert not self.disposed and self.config.get("task_model")
            task = self.config["task"]
            if task == "zero-shot-classification":
                labels = kwargs["candidate_labels"]
                return {"sequence": inputs, "labels": labels, "scores": [1 / len(labels)] * len(labels)}
            if task == "translation":
                if isinstance(inputs, str):
                    return {"translation_text": "Exemple."}
                return [{"translation_text": "Exemple."} for _ in inputs]
            if task == "image-classification":
                return [{"label": "tiger", "score": 0.9}, {"label": "cat", "score": 0.1}][: kwargs.get("top_k", 2)]
            if task == "zero-shot-image-classification":
                return [
                    {"label": label, "score": 1 / len(kwargs["candidate_labels"])}
                    for label in kwargs["candidate_labels"]
                ]
            if task == "image-feature-extraction":
                return [[1.0, 0.0], [0.6, 0.8]] if isinstance(inputs, list) else [1.0, 0.0]
            if task == "object-detection":
                if isinstance(inputs, bytes):  # a photo from the fake Commons: 64 px wide has objects, 80 none
                    import io

                    from PIL import Image

                    width = Image.open(io.BytesIO(inputs)).width
                    if width == 96:
                        raise RuntimeError("could not be decoded")
                    if width == 80:
                        return []
                return [
                    {"label": "cat", "score": 0.99, "box": {"xmin": 1.2, "ymin": 2.0, "xmax": 30.5, "ymax": 40.0}},
                    {"label": "cat", "score": 0.8, "box": {"xmin": 0.0, "ymin": 0.0, "xmax": 10.0, "ymax": 12.0}},
                ]
            if task == "text-to-speech":
                from lmlite import Speech

                return Speech([0.0, 0.25, -0.25] * 1600, 16000)
            return {"text": "Example transcript."}

        async def map(self, inputs, *, progress_callback=None, on_error="raise", **kwargs):
            results = []
            for done, item in enumerate(inputs, 1):
                try:
                    results.append(await self(item, **kwargs))
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    results.append(exc)
                if progress_callback:
                    progress_callback(done, len(inputs))
            return results

        async def export_models(self):
            assert not self.disposed

        async def dispose(self):
            self.disposed = True

        @classmethod
        async def tokenizer(cls, model=None, **options):
            tokenizer = ExampleTokenizer(model)
            tokenizers.append(tokenizer)
            return tokenizer

    class ExampleTokenizer:
        vocab_size = 100

        def __init__(self, model):
            self.model = model
            self.closed = False

        def encode(self, text, *, add_special_tokens=False):
            assert not self.closed and isinstance(text, str)
            return (
                [0, *range(1, len(text.split()) + 1), 0]
                if add_special_tokens
                else list(range(1, len(text.split()) + 1))
            )

        def decode(self, ids):
            assert not self.closed
            return " ".join(f"t{i}" for i in ids)

        def close(self):
            self.closed = True

    monkeypatch.setattr("lmlite.LLM", ExampleLLM)
    monkeypatch.setattr("lmlite.clear_cache", AsyncMock(return_value={"deleted_caches": [], "deleted_databases": []}))
    monkeypatch.setattr(
        "lmlite.cached_models",
        AsyncMock(
            return_value=[
                {
                    "model_id": "Xenova/distilgpt2",
                    "revision": "main",
                    "model_type": "gpt2",
                    "task": "text-generation",
                    "files": ["config.json", "onnx/model.onnx"],
                    "file_sizes": {"config.json": 1_000_000, "onnx/model.onnx": None},
                    "size": 1_000_000,
                },
                {
                    "model_id": None,
                    "revision": None,
                    "model_type": None,
                    "task": None,
                    "files": ["other.bin"],
                    "file_sizes": {"other.bin": None},
                    "size": None,
                },
            ]
        ),
        raising=False,
    )
    # The cache manager widget imports these from lmlite.llm, not from the lmlite package.
    for name in ("LLM", "cached_models", "clear_cache"):
        monkeypatch.setattr(f"lmlite.widgets.{name}", getattr(sys.modules["lmlite"], name))
    installer = types.ModuleType("install")
    installer._ = AsyncMock()
    monkeypatch.setitem(sys.modules, "install", installer)
    browser = types.ModuleType("js")

    def js_eval(source):
        # The GPU notebook's JavaScript probes: the adapter, then per-backend tensor locations.
        if "requestAdapter" in source:
            return AsyncMock(return_value='{"vendor": "example", "fallback": false}')
        return AsyncMock(
            side_effect=lambda llm: '{"cpu": 1, "gpu-buffer": 2}' if llm.backend == "webgpu" else '{"cpu": 3}'
        )

    browser.eval = js_eval
    monkeypatch.setitem(sys.modules, "js", browser)
    if path.name == "ch08b_vision.ipynb":
        _fake_commons(monkeypatch)
        random.seed(0)
    if path.name == "ch08e_microphone.ipynb":
        _fake_microphone(monkeypatch)
    onnx_models = []
    if path.name == "ch08c_onnx-models.ipynb":
        onnx_models = _fake_onnx_notebook_world(monkeypatch)
    notebook = json.loads(path.read_text())
    namespace = {}
    for cell in notebook["cells"]:
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        if cell["cell_type"] == "markdown":
            for target in re.findall(r"\]\(([^)]+\.ipynb)\)", source):
                assert (path.parent / target).is_file(), f"Broken notebook link: {target}"
            continue
        code = compile(source, str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        result = eval(code, namespace)
        if inspect.isawaitable(result):
            await result
    if path.name == "ch08e_microphone.ipynb":
        assert [instance.config["task"] for instance in instances] == ["automatic-speech-recognition"]
        assert namespace["wav"][:4] == b"RIFF" and namespace["result"]["text"] == "Example transcript."
    if path.name == "ch08d_speech.ipynb":
        assert [instance.config["task"] for instance in instances] == ["text-to-speech"]
        assert namespace["progress"] == ["1/3", "2/3", "3/3"] and len(namespace["clips"]) == 3
        assert namespace["wav"][:4] == b"RIFF" and namespace["speech"].sampling_rate == 16000
    if path.name == "ch08c_onnx-models.ipynb":
        assert len(namespace["faces"]) == 1  # the overlapping box was suppressed
        assert namespace["author"] == "Ann Author"
    if path.name == "ch09_orchestration.ipynb":
        panel = namespace["panel"]
        panel.question.value = "<script>alert('test')</script>"
        panel.send.click()
        assert panel.busy and panel.send.disabled and panel.clear.disabled and panel.release.disabled
        task = panel.task
        panel.send.click()
        panel.clear.click()
        panel.release.click()
        assert panel.task is task
        await task
        assert not panel.busy and not panel.send.disabled
        assert "&lt;script&gt;" in panel.transcript.children[0].value
        assert namespace["memory"].get()[-1]["content"] == "Example completion."
        assert "Example completion." in panel.transcript.children[-1].value
        assert panel.stop.disabled

        original_chat = instances[0].chat
        instances[0].chat = AsyncMock(return_value="one two three four five six")
        panel.question.value = "Stop early"
        panel.send.click()
        assert not panel.stop.disabled
        while "Assistant" not in panel.transcript.children[-1].value:
            await asyncio.sleep(0)  # wait for the first streamed chunk
        panel.stop.click()
        await panel.task
        partial = namespace["memory"].get()[-1]["content"]
        assert (
            partial and "one two three four five six".startswith(partial) and partial != "one two three four five six"
        )
        assert panel.status.value.startswith("Stopped") and panel.stop.disabled and not panel.busy
        instances[0].chat = original_chat
        panel.clear.click()
        assert not panel.transcript.children and not namespace["memory"].get()

        original_chat = instances[0].chat
        instances[0].chat = AsyncMock(side_effect=RuntimeError("<failure>"))
        panel.question.value = "Retry this"
        panel.send.click()
        await panel.task
        assert not panel.busy and not panel.send.disabled
        assert not namespace["memory"].get()
        assert "&lt;failure&gt;" in panel.transcript.children[-1].value
        assert panel.question.value == "Retry this"
        instances[0].chat = original_chat

        original_dispose = instances[0].dispose
        instances[0].dispose = AsyncMock(side_effect=RuntimeError("release failed"))
        panel.release.click()
        await panel.task
        assert panel.send.disabled and not panel.release.disabled
        instances[0].dispose = original_dispose
        panel.release.click()
        await panel.task
        assert panel.closed and panel.send.disabled and panel.release.disabled
    if path.name == "ch99_model-info.ipynb":
        # Widget behavior is covered by test_widgets.py; check the notebook wires it up.
        (model_row,) = namespace["manager"].rows.children
        assert model_row.children[2].titles[0].startswith("Xenova/distilgpt2@main")
    assert instances or tokenizers or onnx_models
    assert all(instance.disposed for instance in [*instances, *onnx_models])
    assert all(tokenizer.closed for tokenizer in tokenizers)
    if path.name == "ch01_demo.ipynb":
        assert len(instances) == 1
        assert instances[0].config["generator_model"] == "distilgpt2"
        assert instances[0].config["embedding_model"] == "all-MiniLM-L6-v2"
        assert instances[0].config["backend"] == "wasm"
    if path.name in {
        "ch01_demo.ipynb",
        "ch09_orchestration.ipynb",
        "ch04_generation-tinyllama.ipynb",
        "ch05_generation-lamini.ipynb",
    }:
        assert instances[0].config["return_full_text"] is False
    if path.name == "ch99_model-info.ipynb":
        assert callable(instances[0].config["progress_callback"])
        assert hosted_checks == ["Xenova/distilgpt2"]
    if path.name == "ch10_streaming.ipynb":
        assert instances[0].config["generator_model"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert len(namespace["chunks"]) > 1
        assert namespace["task"].result() == ["Example "]
    if path.name == "ch11_token-scores.ipynb":
        assert instances[0].config["do_sample"] is False
        assert namespace["runner_ups"] == [{"text": " alt", "prob": 0.3}] * 2
    if path.name == "ch12_tokenizers.ipynb":
        assert not instances, "The tokenizer notebook must not load model weights"
        assert [t.model for t in tokenizers] == [
            "gpt2",
            "Xenova/bert-base-multilingual-cased",
            "minishlab/potion-base-8M",
        ]
    if path.name == "ch13_webgpu.ipynb":
        assert [instance.config["backend"] for instance in instances] == ["webgpu", "wasm"]
        assert instances[0].config["dtype"] == instances[1].config["dtype"] and prompts[0] == prompts[1]
    if path.name == "ch08_tasks.ipynb":
        tasks = [instance.config["task"] for instance in instances]
        assert tasks == ["zero-shot-classification", "translation", "automatic-speech-recognition"]
    if path.name == "ch08b_vision.ipynb":
        # Run the picking cell again: no photo repeats until all 36 have been shown, then it starts over.
        cell = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "count = 8" in "".join(cell["source"])
        )
        namespace["detector"].disposed = False  # the notebook's last cell released it; use it again here
        shown = [[title for title, _ in namespace["photos"]]]
        for _ in range(3):
            await eval(compile(cell, str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
            shown.append([title for title, _ in namespace["photos"]])
        flat = [title for batch in shown for title in batch]
        assert len(flat) == 32 and len(set(flat)) == 32 and set(namespace["seen"]) == set(flat)
        assert all(isinstance(file, bytes) for _, file in namespace["seen"].values())  # the downloads are kept
        await eval(compile(cell, str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)  # 4 left: restart
        assert len(namespace["seen"]) == 8 and len(namespace["photos"]) == 8
        namespace["detector"].disposed = True
        results = namespace["results"]
        assert len(results) == 8 and len(namespace["photos"]) == 8
        assert any(isinstance(r, list) and r for r in results)  # a photo with objects
        assert any(r == [] for r in results)  # one with nothing found
        assert any(isinstance(r, RuntimeError) for r in results)  # one that failed, without losing the others
        tasks = [instance.config["task"] for instance in instances]
        assert tasks == [
            "image-classification",
            "zero-shot-image-classification",
            "image-feature-extraction",
            "object-detection",
        ]
    if path.name == "ch09_orchestration.ipynb":
        assert instances[0].config["generator_model"] == "HuggingFaceTB/SmolLM2-360M-Instruct"
        assert instances[0].config["backend"] == "wasm"
        assert chats
        assert all(messages[0]["role"] == "system" for messages in chats)
        assert all(messages[-1]["role"] == "user" for messages in chats)
        assert not prompts


def _fake_microphone(monkeypatch):
    """The recorder widget hands over a short WAV, as if the user had recorded and tapped Stop."""
    pytest.importorskip("anywidget")
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x01" * 16000)
    monkeypatch.setattr("lmlite.audio.AudioRecorder.wait", AsyncMock(return_value=buffer.getvalue()))


def _fake_onnx_notebook_world(monkeypatch):
    """OnnxModel returns one confident face; urllib serves the model, a Commons listing and a photo."""
    np = pytest.importorskip("numpy")
    models = []

    class FakeOnnxModel:
        disposed = False
        inputs = [{"name": "input", "dtype": "float32", "shape": [1, 3, 240, 320]}]
        outputs = [{"name": "scores", "dtype": "float32", "shape": [1, 4420, 2]}]

        @classmethod
        async def load(cls, source, **options):
            assert source.endswith("version-RFB-320.onnx") and options["backend"] == "wasm"
            models.append(cls())
            return models[-1]

        def model_info(self):
            return {"size": 1270727, "backend": "wasm", "execution_provider": "wasm", "cache_status": "downloaded"}

        async def run(self, feeds):
            assert feeds["input"].shape == (1, 3, 240, 320) and feeds["input"].dtype == np.float32
            scores = np.zeros((1, 4420, 2), np.float32)
            scores[0, :3, 1] = [0.9, 0.8, 0.1]  # two overlapping faces, then noise
            boxes = np.zeros((1, 4420, 4), np.float32)
            boxes[0, :2] = [[0.25, 0.25, 0.75, 0.75], [0.26, 0.26, 0.76, 0.76]]
            return {"scores": scores, "boxes": boxes}

        async def dispose(self):
            self.disposed = True

    _fake_commons(monkeypatch)
    monkeypatch.setattr("lmlite.OnnxModel", FakeOnnxModel)
    return models


def _fake_commons(monkeypatch):
    """urllib serves Wikimedia Commons listings and photos: thumbnails alternate 64 and 80 pixels wide, so a fake
    detector can find something in one kind and nothing in the other (and fail on a 96 pixel one)."""
    pil = pytest.importorskip("PIL.Image")
    import io
    import urllib.parse
    import urllib.request

    def photo(width):
        buffer = io.BytesIO()
        pil.new("RGB", (width, 48)).save(buffer, format="JPEG")
        return buffer.getvalue()

    def imageinfo(title):
        number = int(title.rsplit("-", 1)[-1].removesuffix(".jpg").removeprefix("File:") or 0)
        return {
            "title": title,
            "imageinfo": [
                {
                    "thumburl": f"https://example.test/{(64, 80, 96)[number % 3]}.jpg",
                    "extmetadata": {
                        "Artist": {"value": '<a href="x">Ann Author</a>'},
                        "LicenseShortName": {"value": "CC BY 4.0"},
                    },
                }
            ],
        }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    def urlopen(request):
        url = request.full_url
        assert url.startswith(("https://commons.wikimedia.org/", "https://example.test/"))
        if url.startswith("https://example.test/"):
            return Response(photo(int(url.rsplit("/", 1)[-1].removesuffix(".jpg"))))
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if query.get("list") == ["categorymembers"]:  # titles of a category, six photos each
            category = query["cmtitle"][0].removeprefix("Category:")
            members = [{"title": f"File:{category}-{n}.jpg"} for n in range(6)]
            return Response(json.dumps({"query": {"categorymembers": members}}).encode())
        if "titles" in query:  # the details of the chosen photos
            pages = {str(n): imageinfo(title) for n, title in enumerate(query["titles"][0].split("|"))}
            return Response(json.dumps({"query": {"pages": pages}}).encode())
        pages = {"1": imageinfo("File:Example-0.jpg")}  # a category listing with details, for the ONNX notebook
        return Response(json.dumps({"query": {"pages": pages}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_demo_lists_notebooks_in_file_order():
    demo = json.loads((NOTEBOOKS / "ch01_demo.ipynb").read_text())
    source = demo["cells"][0]["source"]
    links = re.findall(r"\]\(([^)]+\.ipynb)\)", "".join(source) if isinstance(source, list) else source)
    others = sorted(path.name for path in NOTEBOOKS.glob("ch*.ipynb") if path.name != "ch01_demo.ipynb")
    assert links == others
    assert links[-1] == "ch99_model-info.ipynb"
