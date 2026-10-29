# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from lmlite import widgets

ENTRIES = [
    {
        "model_id": "Xenova/distilgpt2",
        "revision": "main",
        "model_type": "gpt2",
        "task": "text-generation",
        "files": ["onnx/model.onnx", "config.json", "<b>.json"],
        "file_sizes": {"onnx/model.onnx": None, "config.json": 1_000_000, "<b>.json": 5},
        "size": 1_000_000,
    },
    # Shared ONNX Runtime files cached outside any model.
    {
        "model_id": None,
        "revision": None,
        "model_type": None,
        "task": None,
        "files": ["ort.wasm"],
        "file_sizes": {"ort.wasm": None},
        "size": None,
    },
]


class FakeLLM:
    instances = []
    tags = {}

    def __init__(self, config):
        self.config = config
        self.disposed = False
        FakeLLM.instances.append(self)

    @classmethod
    async def create(cls, **config):
        return cls(config)

    @classmethod
    async def pipeline(cls, task, model, **options):
        return cls({"task": task, "task_model": model, **options})

    @classmethod
    async def model_task(cls, model):
        return cls.tags.get(model, "text-generation")

    async def dispose(self):
        self.disposed = True


@pytest.fixture
def manager(monkeypatch):
    FakeLLM.instances.clear()
    FakeLLM.tags.clear()
    monkeypatch.setattr(widgets, "LLM", FakeLLM)
    monkeypatch.setattr(widgets, "cached_models", AsyncMock(return_value=ENTRIES))
    monkeypatch.setattr(widgets, "clear_cache", AsyncMock())
    return widgets.CacheManager()


async def load(manager, name, task=None):
    manager.model.value = name
    manager.task_choice.value = task
    manager.load.click()
    await manager.task


async def test_lists_models_with_expandable_files(manager):
    await manager.refresh()
    assert [label.value for label in manager.header.children] == ["Architecture", "Task", "Model"]
    (row,) = manager.rows.children  # the non-model entry is hidden
    assert [label.value for label in row.children[:2]] == ["gpt2", "text-generation"]
    files = row.children[2]
    assert files.titles == ("Xenova/distilgpt2@main (3 files, 1.0 MB)",)
    assert files.selected_index is None  # collapsed until clicked
    table = files.children[0].value
    # Sorted by name, with names HTML-escaped.
    assert table.index("&lt;b&gt;.json") < table.index("config.json") < table.index("onnx/model.onnx")
    assert "1.00 MB" in table and "0.00 MB" in table and ">?<" in table
    assert manager.status.value == "1 cached, 1.0 MB"


async def test_delete_clears_one_model_and_refreshes(manager):
    await manager.refresh()
    manager.rows.children[0].children[-1].click()
    assert manager.load.disabled
    await manager.task
    widgets.clear_cache.assert_awaited_once_with(model="Xenova/distilgpt2")
    assert not manager.load.disabled and widgets.cached_models.await_count == 2


@pytest.mark.parametrize(
    "task, expected",
    [
        (None, {"generator_model": "Xenova/gpt2"}),  # the Hub tag
        ("feature-extraction", {"embedding_model": "Xenova/gpt2"}),
        ("translation", {"task": "translation", "task_model": "Xenova/gpt2"}),
    ],
)
async def test_load_fills_the_cache_and_releases_the_model(manager, task, expected):
    await load(manager, " Xenova/gpt2 ", task)
    (llm,) = FakeLLM.instances
    assert expected.items() <= llm.config.items() and callable(llm.config["progress_callback"]) and llm.disposed


@pytest.mark.parametrize(
    "name, tag, message",
    [
        ("", "text-generation", "Enter a model ID"),
        ("Xenova/gpt2", None, "No task found on the Hub"),
        ("Xenova/gpt2", "image-to-text", "cannot load 'image-to-text'"),
    ],
)
async def test_load_errors_are_shown_in_the_status(manager, name, tag, message):
    FakeLLM.tags["Xenova/gpt2"] = tag
    await load(manager, name)
    assert message in manager.status.value and not FakeLLM.instances and not manager.load.disabled


async def test_reports_unavailable_cache_and_refresh_errors(manager):
    widgets.cached_models.return_value = None
    await manager.refresh()
    assert manager.status.value == "Cache inspection is unavailable in this browser."
    widgets.cached_models.side_effect = RuntimeError("JS runtime missing getCachedModels")
    await manager.refresh()
    assert manager.status.value == "RuntimeError: JS runtime missing getCachedModels"
