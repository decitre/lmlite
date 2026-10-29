# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Exercise example control flow without downloading models or requiring a browser."""

import ast
import asyncio
import inspect
import json
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
    prompts = []
    chats = []
    hub_tags = {}  # model -> Hub pipeline tag; unlisted models are text generation

    class ExampleLLM:
        def __init__(self, config):
            self.config = config
            self.disposed = False
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
            return {"text": "Example transcript."}

        async def export_models(self):
            assert not self.disposed

        async def dispose(self):
            self.disposed = True

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
    assert instances
    assert all(instance.disposed for instance in instances)
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
    if path.name == "ch10_streaming.ipynb":
        assert instances[0].config["generator_model"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert len(namespace["chunks"]) > 1
        assert namespace["task"].result() == ["Example "]
    if path.name == "ch11_token-scores.ipynb":
        assert instances[0].config["do_sample"] is False
        assert namespace["runner_ups"] == [{"text": " alt", "prob": 0.3}] * 2
    if path.name == "ch08_tasks.ipynb":
        tasks = [instance.config["task"] for instance in instances]
        assert tasks == ["zero-shot-classification", "translation", "automatic-speech-recognition"]
    if path.name == "ch09_orchestration.ipynb":
        assert instances[0].config["generator_model"] == "HuggingFaceTB/SmolLM2-360M-Instruct"
        assert instances[0].config["backend"] == "wasm"
        assert chats
        assert all(messages[0]["role"] == "system" for messages in chats)
        assert all(messages[-1]["role"] == "user" for messages in chats)
        assert not prompts


def test_demo_lists_notebooks_in_file_order():
    demo = json.loads((NOTEBOOKS / "ch01_demo.ipynb").read_text())
    source = demo["cells"][0]["source"]
    links = re.findall(r"\]\(([^)]+\.ipynb)\)", "".join(source) if isinstance(source, list) else source)
    others = sorted(path.name for path in NOTEBOOKS.glob("ch*.ipynb") if path.name != "ch01_demo.ipynb")
    assert links == others
    assert links[-1] == "ch99_model-info.ipynb"
