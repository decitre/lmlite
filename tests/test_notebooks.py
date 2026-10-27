# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Exercise example control flow without downloading models or requiring a browser."""

import ast
import inspect
import json
import re
import sys
import types
from unittest.mock import AsyncMock
from pathlib import Path

import pytest

from lmlite import SUPPORTED_TASKS

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda path: path.stem)
async def test_notebooks_run_in_order_and_dispose_models(path, monkeypatch):
    instances = []
    prompts = []
    chats = []

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
    if path.name == "orchestration.ipynb":
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
    assert instances
    assert all(instance.disposed for instance in instances)
    if path.name == "demo.ipynb":
        assert len(instances) == 1
        assert instances[0].config["generator_model"] == "distilgpt2"
        assert instances[0].config["embedding_model"] == "all-MiniLM-L6-v2"
        assert instances[0].config["backend"] == "wasm"
    if path.name in {"demo.ipynb", "orchestration.ipynb", "generation-tinyllama.ipynb", "generation-lamini.ipynb"}:
        assert instances[0].config["return_full_text"] is False
    if path.name == "tasks.ipynb":
        tasks = [instance.config["task"] for instance in instances]
        assert tasks == ["zero-shot-classification", "translation", "automatic-speech-recognition"]
    if path.name == "orchestration.ipynb":
        assert instances[0].config["generator_model"] == "HuggingFaceTB/SmolLM2-360M-Instruct"
        assert instances[0].config["backend"] == "wasm"
        assert chats
        assert all(messages[0]["role"] == "system" for messages in chats)
        assert all(messages[-1]["role"] == "user" for messages in chats)
        assert not prompts
