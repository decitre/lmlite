<p>
  <img src="https://raw.githubusercontent.com/decitre/lmlite/main/docs/logo.svg"
       alt="lmLite — browser machine learning from Python"
       width="300">
</p>

[![Tests][ci-badge]][ci-url]
[![codecov-python][codecov-python-badge]][codecov-url]
[![codecov-js][codecov-js-badge]][codecov-url]
[![Transformers.js][transformers-badge]][transformers-url]
[![Docs][docs-badge]][docs-url]
[![Try xeus-python][xeus-badge]][xeus-url]
[![Try Pyodide][pyodide-badge]][pyodide-url]


# lmLite

Package version: `0.14.0`

**Browser machine learning from Python** — run language models, embeddings, vision, speech and bare ONNX models entirely in your browser, and reach the microphone and camera, from a Python kernel in JupyterLite (or headless, with onnxruntime).

**Try it, nothing to install:** [xeus-python][xeus-url] (recommended) or [Pyodide][pyodide-url]. [Documentation][docs-url] · [All notebooks](https://decitre.github.io/lmlite/tutorials/index.html)

## Quick start

```python
from lmlite import LLM, Classifier, Embedder, pipeline

# Chat with a small instruction-tuned model
llm = await LLM.create(generator_model="HuggingFaceTB/SmolLM2-360M-Instruct", backend="wasm", dtype="q4")
print(await llm.chat([{"role": "user", "content": "Explain embeddings in simple terms."}]))

# Chat with memory: keep the conversation in a list and pass it back at each turn
history = [{"role": "user", "content": "What is Python?"}]
history.append({"role": "assistant", "content": await llm.chat(history)})
history.append({"role": "user", "content": "What can I use it for?"})  # "it" is Python
print(await llm.chat(history))

# Embeddings and similarity search
embedder = await Embedder.create("all-MiniLM-L6-v2")
docs = ["JupyterLite runs in the browser.", "Python is widely used for ML."]
print(await embedder.similarity_search("Where does JupyterLite run?", docs))

# Text classification
classifier = await Classifier.create("distilbert-base-uncased-finetuned-sst-2-english")
print(await classifier.classify("I love lmLite!", top_k=2))

# Any Transformers.js task is an awaitable callable: here, the objects in a photo
detector = await pipeline("object-detection", model="detr-resnet-50")
print(await detector("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg", threshold=0.6))
```

## What you get

- **`pipeline()`**: Transformers.js tasks as awaitable Python callables (vision, speech, audio, images out, translation, zero-shot classification), with `map()` over many inputs, progress and cancel.
- **`LLM`, `Embedder`, `Classifier`**: generation and chat with streaming and token probabilities; sentence embeddings with similarity search; text classification.
- **`OnnxModel`**: any single `.onnx` file, NumPy in and NumPy out, with caching, hosted files and a SHA-256 check.
- **Notebook widgets**: record the microphone, take a photo with the camera, manage the model cache.
- **Local and private**: inference runs in your browser, on WebAssembly or WebGPU, with no server and no API key. Downloaded models stay cached and can be exported as ZIP files.
- **A headless native backend**: `pip install 'lmlite[native]'` runs the same ONNX files with onnxruntime, for fast tuning loops checked against the browser.

Extras: `images` (NumPy and Pillow), `audio` and `camera` (the widgets), `native`.

> **Note:** Installing the package on a desktop does **not** run models in a browser. `pipeline()`, `OnnxModel`, `LLM`, `Embedder`, `Classifier` and the widgets need a browser kernel (xeus-python or the JupyterLite Pyodide kernel) with JavaScript interoperability. The native extra covers `Embedder`, `Classifier`, greedy `LLM.generate()`, `OnnxModel` and `tokenizer()`. See [docs/RUNTIME.md](https://github.com/decitre/lmlite/blob/main/docs/RUNTIME.md).

## Notebooks

<p align="center">
  <img src="https://raw.githubusercontent.com/decitre/lmlite/main/docs/images/sd-turbo-astronaut.jpg"
       alt="An astronaut riding a horse on Mars, generated with SD-Turbo from the prompt 'a photo of an astronaut riding a horse on mars'"
       width="320"><br>
  <em>Generated in the browser with SD-Turbo by the text-to-image notebook.</em>
</p>

- **[Detect objects with the camera](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch11_camera.ipynb)**: take a photo on a laptop, tablet or phone and find the objects in it.
- **[Speak to the notebook](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch13_microphone.ipynb)**: record your voice, transcribe it, or name the instrument you play.
- **[Pipelines that return images](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch10_image-outputs.ipynb)**: background removal, depth estimation, segmentation and a 2x upscale.
- **[Text to image](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch15_text-to-image.ipynb)**: SD-Turbo on WebGPU with prompt widgets (a hardware GPU and about 7 GB of memory).
- **[Chat with memory](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch16_orchestration.ipynb)**: a retrieval-augmented answer, then a multi-turn chat widget with conversation memory and streamed, stoppable answers.

[All the notebooks](https://decitre.github.io/lmlite/tutorials/index.html), from the first demo to a retrieval-augmented chat, are listed in the documentation.

## Documentation

The [documentation site][docs-url] has the tutorials (the notebooks), [guides](https://github.com/decitre/lmlite/blob/main/docs/guides/index.md) for one task at a time, a [reference](https://github.com/decitre/lmlite/blob/main/docs/reference/index.md) with the API generated from the docstrings, and [explanations](https://github.com/decitre/lmlite/blob/main/docs/explanation/index.md) of backends and the Python/JavaScript bridge. Setup and tests are in the [development guide](https://github.com/decitre/lmlite/blob/main/docs/DEVELOPMENT.md).

## Contributing

lmLite is experimental. Feedback on the API, on other Transformers.js tasks to expose, and on model compatibility and browser limits is welcome through [GitHub Issues](https://github.com/decitre/lmlite/issues).

Parts of this project were developed with the assistance of generative AI tools. The maintainer is responsible for the architecture, review, testing and published code. AI-assisted contributions are welcome, provided contributors understand, review and test their changes and take responsibility for them. [AGENTS.md](https://github.com/decitre/lmlite/blob/main/AGENTS.md) is a short guide to the layout, the commands and the usual mistakes, for people and for coding agents.

## License

Apache License 2.0. See [LICENSE](LICENSE). Models have their own licences, and some tested ones are non-commercial: check the model card before use, and see the [model guide](https://github.com/decitre/lmlite/blob/main/docs/MODELS.md).

## Citation

```bibtex
@software{lmlite2026,
  author = {Decitre, Emmanuel},
  title = {lmLite: Browser machine learning from Python},
  year = {2026},
  url = {https://github.com/decitre/lmlite}
}
```

## Acknowledgements

lmLite builds on [Transformers.js](https://github.com/huggingface/transformers.js) and [ONNX Runtime](https://onnxruntime.ai) for inference, on [JupyterLite](https://jupyterlite.readthedocs.io/), [xeus-python](https://github.com/jupyter-xeus/xeus-python) and [Pyodide](https://pyodide.org/) for Python in the browser, and on the models that the Hugging Face community publishes. The full list is on the [credits page](https://decitre.github.io/lmlite/credits.html).

---

[ci-badge]: https://github.com/decitre/lmlite/actions/workflows/test.yml/badge.svg
[codecov-python-badge]: https://img.shields.io/codecov/c/github/decitre/lmlite/main?flag=python&label=python&logo=codecov
[codecov-js-badge]: https://img.shields.io/codecov/c/github/decitre/lmlite/main?flag=javascript&label=javascript&logo=codecov
[ci-url]: https://github.com/decitre/lmlite/actions/workflows/test.yml
[codecov-url]: https://codecov.io/gh/decitre/lmlite
[docs-badge]: https://img.shields.io/website?url=https%3A%2F%2Fdecitre.github.io%2Flmlite%2F&label=docs&up_message=online&down_message=offline
[docs-url]: https://decitre.github.io/lmlite/
[xeus-url]: https://decitre.github.io/lmlite/xeus/lab/?path=ch01_demo.ipynb
[pyodide-url]: https://decitre.github.io/lmlite/pyodide/lab/?path=ch01_demo.ipynb
[pyodide-badge]: https://img.shields.io/badge/try-Pyodide-f7dc1e.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI2OCIgaGVpZ2h0PSI2OCIgdmlld0JveD0iMCAwIDE3Ljk5MiAxNy45OTIiPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKC40NiAuNDQ4KSI+PGNpcmNsZSBjeD0iODMuNzI3IiBjeT0iODUuNjMzIiByPSI2LjU5OCIgc3R5bGU9ImZpbGw6I2Y3ZGMxZTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6LjI2NDU4MzMyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzUuMTkxIC03OS4wMjUpIi8+PHJlY3Qgd2lkdGg9IjUuNjUzIiBoZWlnaHQ9IjEuMDEiIHg9IjgwLjgzMSIgeT0iOTIuMzQ0IiByeD0iLjQxMyIgcnk9Ii40MTMiIHN0eWxlPSJmaWxsOiNmOWY5Zjk7c3Ryb2tlOm5vbmU7c3Ryb2tlLXdpZHRoOjEuOTk5OTk5ODgiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03NS4xOTEgLTc5LjAyNSkiLz48cmVjdCB3aWR0aD0iNS42NDEiIGhlaWdodD0iMS4wMzEiIHg9IjgwLjgzNyIgeT0iOTMuNjcxIiByeD0iLjQyMSIgcnk9Ii40MjEiIHN0eWxlPSJmaWxsOiNmOWY5Zjk7c3Ryb2tlOm5vbmU7c3Ryb2tlLXdpZHRoOjEuOTk5OTk5NzYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03NS4xOTEgLTc5LjAyNSkiLz48cmVjdCB3aWR0aD0iMy4yNCIgaGVpZ2h0PSIxLjAyIiB4PSI4Mi4wMzgiIHk9Ijk1LjAxOSIgcng9Ii40MTciIHJ5PSIuNDE3IiBzdHlsZT0iZmlsbDojZjlmOWY5O3N0cm9rZTpub25lO3N0cm9rZS13aWR0aDoxLjk5OTk5OTg4IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzUuMTkxIC03OS4wMjUpIi8+PGcgc3R5bGU9ImZpbGw6IzFhMWExYSI+PGcgc3R5bGU9ImZpbGw6IzFhMWExYTtzdHJva2Utd2lkdGg6LjI2NDU4MzM1Ij48cGF0aCBkPSJNODcuMDY1IDkwLjI4OXEtLjk5Ni0uMDQxLTEuNDc1LTEuMjE2LS4zNTYtLjg4NC0uMzU2LTIuMTY2IDAtLjA2Ni4wMDgtLjE3LjA1OC0uODcyLjMyMy0xLjYuNDMtMS4xNzMgMS4yMTUtMS4xNzMuMDQyIDAgLjA4Ny4wMDQuNzE1LjA0NS43MTUuODMgMCAuODU3LS43MzYgMS45MDItLjQ2Ny42NjYtLjk4IDEuMDguMTQxLjc4NS4zMSAxLjE3OC4zMTUuNzI3LjkzNS43MjcuMDQxIDAgLjA4Ny0uMDA4LjEzNi0uMDI1LjM4LS4xOTguMjM2LS4xNjIuMzY4LS4xNjIuMDU4IDAgLjExMi4wMzguMTIuMDgyLjEyLjIyMy0uMDA5LjIxNS0uMzA2LjQ0Mi0uMzcyLjI2OS0uODA3LjI2OXptLS4yOC01LjcwMXEtLjI0NS4wMDQtLjQzLjIxLS40NzIuNTM0LS41MTcgMi4yMy4yNDgtLjI1My40OTYtLjU5Ni42ODYtLjkzOS42ODYtMS41ODQgMC0uMDMzLS4wMTctLjA3OC0uMDY2LS4xODItLjIxOS0uMTgyeiIgYXJpYS1sYWJlbD0ibCIgc3R5bGU9ImZpbGw6IzFhMWExYTtzdHJva2Utd2lkdGg6LjI2NDU4MzM1IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzYuNDA5IC03OS40NzMpIi8+PC9nPjxwYXRoIGQ9Ik04NC45OTggODMuNjYzYS42OTQuNjk0IDAgMCAxLS42OTQuNjk1LjY5NC42OTQgMCAwIDEtLjY5NC0uNjk1LjY5NC42OTQgMCAwIDEgLjY5NC0uNjk0LjY5NC42OTQgMCAwIDEgLjY5NC42OTR6IiBzdHlsZT0iZmlsbDojMWExYTFhO3N0cm9rZTpub25lO3N0cm9rZS13aWR0aDouMjY0NTgzMzIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03Ni40MDkgLTc5LjQ3MykiLz48ZyBzdHlsZT0iZmlsbDojMWExYTFhO3N0cm9rZS13aWR0aDouMjY0NTgzMzUiPjxwYXRoIGQ9Ik04NC4xNTUgODQuNjY3cS4yNjkuMDA0LjI3Ny4zNWwtLjQ1OSAzLjI4M3ExLjAzNC0uNzkgMS4xNjItLjg5My4xNTMtLjEzMi4zMTQtLjEzNi4wNTggMCAuMTEyLjAzNy4xMi4wODMuMTIuMjExLS4wMDUuMTk5LS4yMjQuMzg1LS42MDcuNTEyLTEuNjA4IDEuMjZsLS4xOSAxLjM2OXEtLjAxMi4wOTUtLjAzNy4xODYtLjE5LjcyNy0uOTU1LjcyNy0uMzggMC0uNjctLjI4NS0uMjg1LS4yODUtLjI4NS0uNjQgMC0uMzczLjE3OC0uNjEyLjA5OS0uMTI0LjM4OC0uMzRsLjk5Mi0uNzQ0cS41MjUtMy40NjguNTM4LTMuNjQ2IDAtLjAyNS4wMDQtLjA0NS4wNDUtLjQ2Ny4zNDMtLjQ2N3ptLTEuMDg3IDUuNTM1LjA3LS41MDQtLjQ1OS4zNTFxLS4yOTguMjIzLS4zMDIuNDIyIDAgLjA3LjAyNS4xMzIuMDcuMTgyLjI2NS4xODIuMDc4IDAgLjE0LS4wMjUuMTk5LS4wODcuMjYtLjU1OHptLjg5Ny02LjMyNXEwLS4wNzQuMDM3LS4xNC4wODctLjE2Mi4yNjktLjE2Mi4wNzQgMCAuMTQ0LjAzNy4xNTguMDg3LjE1OC4yNjUgMCAuMDc0LS4wMzguMTQ1LS4wODYuMTU3LS4yNjQuMTU3LS4wNzUgMC0uMTQ1LS4wMzgtLjE2MS0uMDgyLS4xNjEtLjI2NHoiIGFyaWEtbGFiZWw9ImoiIHN0eWxlPSJmaWxsOiMxYTFhMWE7c3Ryb2tlLXdpZHRoOi4yNjQ1ODMzNSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTc2LjQwOSAtNzkuNDczKSIvPjwvZz48L2c+PC9nPjwvc3ZnPiA=
[xeus-badge]: https://img.shields.io/badge/try-xeus--python-f7dc1e.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0Ni43IDQ2LjciPgogIDxwYXRoCiAgICBmaWxsPSIjNDBCM0MzIgogICAgZD0iTTUgMGMyLjUuOCA0LjUgMiA0LjYgMi4xIDEuOSAxLjMgMy42IDMuMSA0LjggNS4xLjQuNiA1LjIgNy44IDguNiAxMi45LS44IDEuMi0xLjUgMi4zLTIuMSAzLjEgMC0uMS0uMS0uMS0uMS0uMkMxNy42IDE3LjkgMTEuNCA4LjYgMTEgNy45Yy0uOS0xLjMtMi0yLjUtMy41LTMuNCAwIDAtMS41LTEtMy41LTEuNkMzLjIgMi42IDEuOCAyLjQgMCAyLjJ2NDIuNGMxLjgtLjIgMy4yLS40IDQtLjcgMi0uNiAzLjUtMS42IDMuNS0xLjYgMS41LTEgMi43LTIuMiAzLjUtMy40LjQtLjcgNi42LTkuOSAxMC4xLTE1LjJDMjQgMTkuMSAzMS40IDguMSAzMS45IDcuM2MxLjItMiAyLjktMy44IDQuOC01LjEgMC0uMSAyLjEtMS4zIDQuNi0yLjFMNDEuOCAwSDV6CiAgICAgICBNNDYuNyAyLjJjLTEuOC4yLTMuMi40LTQgLjctMi4xLjYtMy42IDEuNi0zLjYgMS42LTEuNS45LTIuNiAyLjEtMy41IDMuNC0uNC43LTYuNiA5LjktMTAuMSAxNS4yLTIuOSA0LjQtMTAuMyAxNS41LTEwLjggMTYuMi0xLjIgMi0yLjkgMy44LTQuOCA1LjEgMCAuMS0yLjEgMS4zLTQuNiAyLjFsLS41LjFoMzYuOGwtLjQtLjFjLTIuNS0uOC00LjUtMi00LjYtMi4xLTEuOS0xLjMtMy42LTMuMS00LjgtNS4xLS40LS42LTUuMi03LjgtOC42LTEyLjkuOC0xLjIgMS41LTIuMyAyLjEtMy4xIDAgLjEuMS4xLjEuMiAzLjUgNS4zIDkuNyAxNC41IDEwLjEgMTUuMi45IDEuMyAyIDIuNSAzLjUgMy40IDAgMCAxLjUgMSAzLjUgMS42LjguMyAyLjIuNSA0IC43VjIuMnoiCiAgLz4KPC9zdmc+Cgo=
[transformers-badge]: https://img.shields.io/badge/🤗-Transformers.js-172033?style=flat&labelColor=172033&color=172033
[transformers-url]: https://huggingface.co/docs/transformers.js/en/index
