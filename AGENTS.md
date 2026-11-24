# AGENTS.md

lmLite is browser machine learning from Python: Transformers.js pipelines, bare ONNX models and browser devices (microphone, camera), driven from a Python kernel in JupyterLite (xeus-python or Pyodide), with an optional headless onnxruntime backend. Read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup and the test tasks; this file is the short version, plus what is easy to get wrong.

## Layout

- `src/lmlite/lmlite.js`: the JavaScript runtime (`BrowserRuntime`, exposed as `globalThis.lmLite`). The Python side calls it through the bridge.
- `src/lmlite/`: `llm.py` (`LLM`, generation only), `embedder.py`, `classifier.py`, `pipeline.py` (tasks, `Pipeline`, `Speech`), `models.py` (names, cache, `ModelInfo`), `tokenizer.py`, `onnx_model.py`, `audio.py` and `camera.py` (widgets), `widgets.py` (cache manager), `native.py` (onnxruntime backend), and the private `_session.py` and `_bridge.py`. Everything public is exported from `lmlite/__init__.py`.
- `notebooks/chNN_*.ipynb`: the tutorials, numbered. `apps/*.ipynb`: the Voici apps, built into their own site by `scripts/lite-build.sh` (`LITE_SITES=voici` builds only them) and tested by `voici-smoke`, not by the notebook tests. `docs/`: Diátaxis (guides, reference, explanation), built with Sphinx. `tests/`: unit tests; `tests/browser/`: opt-in real-browser tests.

## Commands (pixi)

- `pixi run tests` (Python, no browser), `pixi run js-tests` (vitest), `pixi run lint` (fixes) or `lint-check`. Ruff, line length 120.
- `pixi run --environment docs docs`: the docs build. Warnings are errors, so a broken link or heading anchor fails it.
- Browser tests are opt-in (`-m browser`, need network and Playwright's Chromium: `pixi run install-browser`), one task per area: `task-smoke`, `speech-smoke`, `camera-smoke`, `embedder-smoke`, `lab-smoke`, and so on. The xeus-python variants run in `-e xeus-site`, which builds a site first. See DEVELOPMENT.md for the list and for `LMLITE_WEBGPU_ARGS` (headless Chromium has no real WebGPU).

## What is easy to get wrong

- **New Python module:** four browser tests copy the package files into Pyodide from a hard-coded list (`tests/browser/kernels.py`, `test_onnx_model.py`, `test_onnx_notebook.py`, `test_tokenizer.py`). Add the file there or those tests fail with an import error.
- **New notebook:** `tests/test_notebooks.py` runs every notebook against fakes, and `ch01_demo` must list the notebooks in file order. Keep each cell's `source` as it is (string or list) so diffs stay small.
- **Public API:** add the name to `__init__.py` and `__all__`, and to `docs/reference/api.md` (a test checks), whose entries come from the docstrings: write one. `LLM.create(embedding_model=...)`, `llm.embed()`, `llm.classify()` and `llm.similarity_search()` are deprecated and go in 0.15; use `Embedder` and `Classifier`.
- **JavaScript:** the Transformers.js version is the `TRANSFORMERS_VERSION` constant in `lmlite.js`, and `scripts/lite-build.sh` reads it with `sed`: do not reformat that line. In the same script, `jl-forge$JL_FORGE_VERSION` must stay on one line (a weekly job rewrites it).
- **xeus-python:** binary widget traits set by the browser never reach the kernel (send base64 text), and loading a FreeType font hangs (use `ImageFont.load_default_imagefont()`).
- **Models:** several tested models are non-commercial (`mms-tts-eng`, `musicgen-small`, SD-Turbo); `docs/MODELS.md` lists the licences. Do not bundle them into a published site.
- **Local files:** `models/`, `workspace/`, `coverage*`, `build/`, screenshots and `release-notes.md` are untracked. Do not `git add -A` at the root; add paths.
- **Pull requests:** squash merges. For stacked PRs, retarget the second to `main` once the first merges: GitHub does not do it while the base branch exists, and the merge lands in the old base instead.
- **Releases:** `bumpver update --minor` commits, tags and pushes to `main`, and publishing a GitHub release uploads to PyPI, which cannot be undone. Do it only when asked.

## Before you open a PR

Run `pixi run tests`, `pixi run js-tests`, `pixi run lint-check` and the docs build, plus the browser task for the area you touched. Say in the PR what you ran and what you did not.
