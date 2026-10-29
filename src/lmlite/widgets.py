# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Notebook widgets built on the lmLite API. Imported separately so ``import lmlite`` stays light."""

import asyncio
from html import escape

import ipywidgets as widgets

from .llm import LLM, SUPPORTED_TASKS, cached_models, clear_cache

# LLM.create() option for each task; SUPPORTED_TASKS load through LLM.pipeline().
_LOAD_OPTIONS = {
    "text-generation": "generator_model",
    "feature-extraction": "embedding_model",
    "sentence-similarity": "embedding_model",
    "text-classification": "classification_model",
}
_TASKS = [
    ("auto (Hub task)", None),
    *((task, task) for task in ("text-generation", "feature-extraction", "text-classification", *SUPPORTED_TASKS)),
]


def _columns(architecture, task, model, *buttons, style=None):
    # The header and model lines share column widths so they line up; `model` may be a widget.
    labels = [
        widgets.Label(text, style=style or {}, layout=widgets.Layout(width=width))
        for text, width in ((architecture, "8em"), (task, "13em"))
    ]
    if isinstance(model, str):
        model = widgets.Label(model, style=style or {})
    model.layout.flex = "1"
    return widgets.HBox([*labels, model, *buttons], layout=widgets.Layout(width="100%", align_items="flex-start"))


def _megabytes(size):
    return "?" if size is None else f"{size / 1e6:.2f} MB"


def _file_table(entry):
    rows = "".join(
        f"<tr><td>{escape(name)}</td>"
        f"<td style='text-align:right; padding-left:2em'>{_megabytes(entry['file_sizes'][name])}</td></tr>"
        for name in sorted(entry["files"])
    )
    return widgets.HTML(f"<table>{rows}</table>")


class CacheManager:
    """Widget to list, load and delete models in the Transformers.js browser cache.

    Display ``manager.view`` and ``await manager.refresh()`` to fill it. Each line shows a model's
    architecture, Hub task and name; clicking the name expands its cached files and sizes. Load
    downloads a model into the cache and releases it right away. Hosted model files and ZIPs under
    ``/drive/models`` are not listed.
    """

    def __init__(self):
        self.task = None
        self.header = _columns("Architecture", "Task", "Model", style={"font_weight": "bold"})
        self.rows = widgets.VBox()
        self.model = widgets.Text(placeholder="Xenova/distilgpt2", layout=widgets.Layout(flex="1"))
        self.task_choice = widgets.Dropdown(options=_TASKS, layout=widgets.Layout(width="16em"))
        self.load = widgets.Button(description="Load", button_style="primary")
        self.status = widgets.Label()
        self.view = widgets.VBox(
            [self.header, self.rows, widgets.HBox([self.model, self.task_choice, self.load]), self.status]
        )
        self.load.on_click(lambda _: self.start(self.download))

    def set_busy(self, busy):
        for control in (self.load, *(row.children[-1] for row in self.rows.children)):
            control.disabled = busy

    def start(self, action):
        if self.task is None or self.task.done():
            self.set_busy(True)
            self.task = asyncio.create_task(self.refresh(action))

    async def refresh(self, action=None):
        try:
            if action:
                await action()
            self.show(await cached_models())
        except Exception as error:
            # Surface callback failures in the widget, not a background console.
            self.status.value = f"{type(error).__name__}: {error}"
        finally:
            self.set_busy(False)

    async def download(self):
        name = self.model.value.strip()
        if not name:
            raise ValueError("Enter a model ID first.")
        task = self.task_choice.value
        if task is None:
            self.status.value = f"Looking up the task of {name}..."
            task = await LLM.model_task(name)
            if task is None:
                raise ValueError(f"No task found on the Hub for {name}; pick one in the list.")
        self.status.value = f"Loading {name} as {task}..."
        if task in _LOAD_OPTIONS:
            llm = await LLM.create(**{_LOAD_OPTIONS[task]: name}, progress_callback=self.on_progress)
        elif task in SUPPORTED_TASKS:
            llm = await LLM.pipeline(task, name, progress_callback=self.on_progress)
        else:
            raise ValueError(f"lmLite cannot load {task!r} models; pick a task in the list.")
        await llm.dispose()  # Only fill the cache; later loads read from it.

    def on_progress(self, event):
        self.status.value = f"{event['status']} {event['model']} {event['file']}".strip()

    async def delete(self, model):
        self.status.value = f"Deleting {model}..."
        await clear_cache(model=model)

    def show(self, models):
        for row in self.rows.children:
            row.close()
        self.rows.children = ()
        if models is None:
            self.status.value = "Cache inspection is unavailable in this browser."
            return
        # Non-model entries (the shared ONNX Runtime files) are not listed.
        models = [entry for entry in models if entry["model_id"]]
        rows = []
        for entry in models:
            model = entry["model_id"]
            size = "size unknown" if entry["size"] is None else f"{entry['size'] / 1e6:.1f} MB"
            delete = widgets.Button(description="Delete", button_style="danger")
            delete.on_click(lambda _, model=model: self.start(lambda: self.delete(model)))
            name = f"{model}@{entry['revision']} ({len(entry['files'])} files, {size})"
            # The title expands to the cached files when clicked.
            files = widgets.Accordion(children=[_file_table(entry)], titles=(name,), selected_index=None)
            rows.append(_columns(entry["model_type"] or "?", entry["task"] or "?", files, delete))
        self.rows.children = rows
        known = sum(entry["size"] or 0 for entry in models)
        unknown = " (some sizes unknown)" if any(entry["size"] is None for entry in models) else ""
        self.status.value = f"{len(models)} cached, {known / 1e6:.1f} MB{unknown}" if models else "No models cached."
