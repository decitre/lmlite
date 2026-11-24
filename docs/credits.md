# Credits

lmLite is a thin layer over other people's work. This is what it uses, by what it does for the project.

## Inference

- [Transformers.js](https://github.com/huggingface/transformers.js): the browser runtime for the task pipelines, tokenizers and model loading. The version is pinned in `lmlite.js`.
- [ONNX Runtime](https://onnxruntime.ai): runs every model, on WebAssembly and WebGPU in the browser (onnxruntime-web, the build Transformers.js uses) and natively with the `lmlite[native]` extra.
- [tokenizers](https://github.com/huggingface/tokenizers) and [huggingface_hub](https://github.com/huggingface/huggingface_hub): the native backend's tokenizer and model download.

## Python in the browser

- [JupyterLite](https://jupyterlite.readthedocs.io/) and [jupyterlite-xeus](https://github.com/jupyterlite/xeus): the browser-based Jupyter environment.
- [xeus-python](https://github.com/jupyter-xeus/xeus-python) and [pyjs](https://github.com/emscripten-forge/pyjs), with the packages from [emscripten-forge](https://emscripten-forge.org): the xeus-python kernel and its bridge to JavaScript.
- [Pyodide](https://pyodide.org/): the other kernel, with `pyodide.ffi` as its bridge.
- [jl-forge](https://github.com/decitre/jl-forge): builds the JupyterLite sites from this repository.

## Notebooks and widgets

- [ipywidgets](https://ipywidgets.readthedocs.io/) and [tqdm](https://github.com/tqdm/tqdm): progress bars and the cache manager.
- [anywidget](https://anywidget.dev/): the microphone and camera widgets.
- [NumPy](https://github.com/numpy/numpy) and [Pillow](https://python-pillow.github.io/): arrays and images, in the optional `images` extra.

## Models

Every demo downloads its models from the [Hugging Face Hub](https://huggingface.co/). They are the work of their authors, and the browser-ready ONNX versions are mostly the work of the [Xenova](https://huggingface.co/Xenova) and [onnx-community](https://huggingface.co/onnx-community) accounts. [The model guide](MODELS.md) names each tested model with its page and licence; several are non-commercial.

Two exports deserve their own mention. The text-to-image notebook downloads its SD-Turbo files from [schmuell/sd-turbo-ort-web](https://huggingface.co/schmuell/sd-turbo-ort-web). The instrument classifier, `onnx-community/Musical-Instrument-Classification-ONNX`, is an ONNX version of `Bhaveen/Musical-Instrument-Classification`.

## Documentation and tests

- [Sphinx](https://www.sphinx-doc.org/), [MyST](https://myst-parser.readthedocs.io/) and [Furo](https://pradyunsg.me/furo): the documentation site. Its layout (tutorials, guides, reference, explanation) follows [here-search-demo](https://github.com/heremaps/here-search-demo).
- [Playwright](https://playwright.dev/): the real-browser tests.
