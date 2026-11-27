# Tutorials

Learn by doing: each notebook runs in your browser without installing anything, and the chapters build on one another. Run a notebook in order, in a fresh kernel.

- **[Basic demo](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch01_demo.ipynb)** — Text generation, embeddings, classification, and similarity search.
- **[Tool calling](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch06_tool-calling.ipynb)** — Let a small model (Qwen2.5-0.5B-Instruct, WASM) choose between Python functions with `ToolRunner`: every reply is validated before a tool runs, and a scripted model shows how invalid replies, the step limit and the recorded steps behave.
- **[Streaming chat](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch07_streaming.ipynb)** — Print a chat answer as it is generated, stop early with `break`, and cancel a running generation with `cancel()`, using SmolLM2-135M-Instruct (WASM).
- **[Token probabilities](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch08_token-scores.ipynb)** — Stream each generated token with its probability, plus the runner-up token at every step, using SmolLM2-135M-Instruct (WASM).
- **[LLM orchestration](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch11_orchestration.ipynb)** — A simple retrieval-augmented generation (RAG) workflow and interactive multi-turn chat with conversation memory and streamed, stoppable answers, using SmolLM2-360M-Instruct (WASM/q4) and MiniLM embeddings (WASM/q8).
- **[Task pipelines](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch12_tasks.ipynb)** — Zero-shot classification, translation and speech recognition with `pipeline()`.
- **[Vision pipelines](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch13_vision.ipynb)** — Image classification, zero-shot image classification, image features and object detection with `pipeline()`.
- **[Pipelines that return images](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch14_image-outputs.ipynb)** — Background removal, depth estimation, image segmentation and a 2x upscale with `pipeline()`, shown as PNG images.
- **[Detect objects with the camera](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch15_camera.ipynb)** — Take a photo with the camera of a laptop, tablet or phone with an `anywidget` and find the objects in it in the browser.
- **[Text to speech](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch16_speech.ipynb)** — Generate speech with `pipeline("text-to-speech")` and play it in the notebook.
- **[Speak to the notebook](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch17_microphone.ipynb)** — Record the microphone (laptop or tablet) with an `anywidget` and transcribe it in the browser.
- **[Tokenizers](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch18_tokenizers.ipynb)** — Load only the tokenizers of GPT-2, multilingual BERT and POTION (no weights), compare how they split text, and count tokens synchronously with `tokenizer()`.
- **[Bare ONNX models](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch19_onnx-models.ipynb)** — Run a single `.onnx` file (the UltraFace face detector) with `OnnxModel`, NumPy in and NumPy out.
- **[Text to image: a recipe](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch20_text-to-image.ipynb)** — SD-Turbo in one step on WebGPU with `OnnxModel`, a CLIP tokenizer and a NumPy scheduler, with prompt widgets to generate and save images: a research-licence recipe that needs a hardware GPU, 2.4 GB of download and about 7 GB of memory.
- **[Proving GPU inference](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch21_webgpu.ipynb)** — Show that WebGPU inference really runs on the GPU: the hardware adapter, model tensors held in GPU memory, and WebGPU vs WASM timing for the same greedy generation, using SmolLM2-135M-Instruct (q4).
- **[Model info and cache manager](https://decitre.github.io/lmlite/xeus/lab/index.html?path=ch99_model-info.ipynb)** — Download size and cache status before loading, loading progress from Python, and a widget to list, load and delete cached models.

These examples run locally in the browser using Transformers.js. For one task at a time, see the [guides](../guides/index.md).
