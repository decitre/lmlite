# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Native (CPython) backend: runs the same ONNX files as the browser with onnxruntime.

A dev/test/tuning accelerator mirroring ``llm.js`` (Transformers.js 4.3.2) semantics; the browser
stays the source of truth. Requires ``pip install 'lmlite[native]'``.
"""

import asyncio
import json
import os
import re

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer as HFTokenizer

from .llm import resolve_model_id, split_model_name_and_revision
from .model_assets import DTYPE_SUFFIXES

# llm.js DEFAULT_DTYPES, tried in order for generation and classification models.
DEFAULT_DTYPES = ("q8", "int8", "fp32")


def _model_file(model, filename, models_path=None):
    """Local path of ``filename`` from a model directory, ``models_path/<id>/``, else the Hub cache.

    Raises FileNotFoundError when the model has no such file.
    """
    if os.path.isdir(model):
        path = os.path.join(model, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    model_id, revision = split_model_name_and_revision(resolve_model_id(model))
    if models_path:
        path = os.path.join(models_path, model_id, filename)
        if os.path.isfile(path):
            return path
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        return hf_hub_download(model_id, filename, revision=revision)
    except EntryNotFoundError as exc:
        raise FileNotFoundError(f"{model_id}: {filename}") from exc


def _read_json(model, filename, models_path=None):
    try:
        with open(_model_file(model, filename, models_path), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _clean_up_tokenization(text):
    # Same replacements as @huggingface/tokenizers' clean_up_tokenization.
    for before, after in ((" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't")):
        text = text.replace(before, after)
    return re.sub(r" '(m|s|ve|re)", r"'\1", text)


class NativeTokenizer:
    """tokenizer.json + tokenizer_config.json with Transformers.js encode/decode behavior."""

    def __init__(self, tokenizer_json, config=None):
        self._tok = HFTokenizer.from_str(tokenizer_json)
        # Transformers.js ignores tokenizer.json padding/truncation and truncates ids itself.
        self._tok.no_truncation()
        self._tok.no_padding()
        self.config = config or {}
        self.vocab_size = self._tok.get_vocab_size()

    def ids(self, text, add_special_tokens=True, truncate=False):
        ids = self._tok.encode(text, add_special_tokens=add_special_tokens).ids
        max_length = self.config.get("model_max_length")
        # Transformers.js truncation cuts the id list, dropping trailing special tokens too.
        return ids[: int(max_length)] if truncate and max_length and max_length < len(ids) else ids

    def text(self, ids, skip_special_tokens=False):
        text = self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
        return _clean_up_tokenization(text) if self.config.get("clean_up_tokenization_spaces", True) else text

    # JSON strings match the JS handle that lmlite.llm.Tokenizer wraps.
    def encode(self, text, add_special_tokens=False):
        return json.dumps(self.ids(text, add_special_tokens))

    def decode(self, ids):
        return self.text(json.loads(ids))


def _load_tokenizer(model, models_path):
    with open(_model_file(model, "tokenizer.json", models_path), encoding="utf-8") as f:
        return NativeTokenizer(f.read(), _read_json(model, "tokenizer_config.json", models_path))


async def load_tokenizer(model=None, tokenizer_json=None, tokenizer_config=None, hosted_models_url=None):
    """Native ``loadTokenizer``.

    ``model`` is a model directory or id, looked up under the local directory ``hosted_models_url`` and then
    the Hub; ``tokenizer_json``/``tokenizer_config`` are local paths or HTTP(S) URLs.
    """

    def read(path):
        if path.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(path, timeout=30) as response:
                return response.read().decode("utf-8")
        with open(path, encoding="utf-8") as f:
            return f.read()

    try:
        if model:
            return _load_tokenizer(model, hosted_models_url)
        config = json.loads(read(tokenizer_config)) if tokenizer_config else {}
        return NativeTokenizer(read(tokenizer_json), config)
    except Exception as exc:
        raise RuntimeError(f"Failed to load tokenizer from '{model or tokenizer_json}': {exc}") from exc


class _Model:
    """One ONNX session with its config and tokenizer, loaded like a Transformers.js pipeline."""

    def __init__(self, model, dtypes, models_path):
        self.config = _read_json(model, "config.json", models_path)
        js_config = self.config.get("transformers.js_config") or {}
        errors = []
        for dtype in dtypes:
            if dtype not in DTYPE_SUFFIXES:
                raise ValueError(f"Invalid dtype: {dtype}. Should be one of: {', '.join(DTYPE_SUFFIXES)}")
            filename = f"onnx/model{DTYPE_SUFFIXES[dtype]}.onnx"
            try:
                path = _model_file(model, filename, models_path)
                if js_config.get("use_external_data_format"):
                    # shortcut: assumes one external data file; add numbered chunks if a model ships them.
                    _model_file(model, f"{filename}_data", models_path)
            except FileNotFoundError as exc:
                errors.append(str(exc))
                continue
            self.dtype, self.path = dtype, path
            break
        else:
            raise FileNotFoundError(f"No ONNX weights for dtypes {', '.join(dtypes)}: {'; '.join(errors)}")
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.tokenizer = _load_tokenizer(model, models_path)

    def run(self, feed):
        return dict(zip(self.output_names, self.session.run(None, feed)))

    def encoder_feed(self, ids):
        feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros((1, len(ids)), np.int64)
        return feed


class NativeLLM:
    """CPython counterpart of llm.js ``BrowserLLM`` for embed(), classify() and generate()."""

    def __init__(self, config):
        self.config = config
        self.return_full_text = config.get("return_full_text", True)
        self.generator = self.embedder = self.classifier = None
        self._generating = False
        self._cancelled = False

    def _dtypes(self):
        return self.config.get("dtype_fallbacks") or ([self.config["dtype"]] if self.config.get("dtype") else None)

    async def init(self, progress_callback=None):
        def report(model, status):
            if progress_callback:
                progress_callback(model, status, "", 0, 0)

        if self.config.get("task") or self.config.get("task_model"):
            raise NotImplementedError("The native backend does not support LLM.pipeline() tasks.")
        report("", "initializing")
        models_path = self.config.get("local_models_path")
        dtypes = self._dtypes() or DEFAULT_DTYPES
        if model := self.config.get("generator_model"):
            report(model, "loading")
            self.generator = _Model(model, dtypes, models_path)
        if model := self.config.get("embedding_model"):
            report(model, "loading")
            config = _read_json(model, "config.json", models_path)
            js_dtype = (config.get("transformers.js_config") or {}).get("dtype")
            # Transformers.js wasm default is q8 unless config.json pins a dtype; embedding_dtype never falls back.
            dtype = self.config.get("embedding_dtype") or (js_dtype if isinstance(js_dtype, str) else "q8")
            self.embedder = _Model(model, [dtype], models_path)
        if model := self.config.get("classification_model"):
            report(model, "loading")
            self.classifier = _Model(model, dtypes, models_path)
        report("", "ready")

    async def dispose(self):
        self.generator = self.embedder = self.classifier = None

    async def embed(self, text):
        if self.embedder is None:
            raise RuntimeError("No embedding model configured")
        model = self.embedder
        outputs = model.run(model.encoder_feed(model.tokenizer.ids(text, truncate=True)))
        hidden = next(outputs[name] for name in ("last_hidden_state", "logits", "token_embeddings") if name in outputs)
        # Batch of one without padding: attention-mask mean pooling is a plain mean. Float32 like Transformers.js.
        vector = hidden[0].astype(np.float64).mean(axis=0).astype(np.float32)
        return (vector / np.linalg.norm(vector.astype(np.float64))).astype(np.float32).tolist()

    async def classify(self, text, top_k=1):
        if self.classifier is None:
            raise RuntimeError("No classification model configured")
        model = self.classifier
        logits = model.run(model.encoder_feed(model.tokenizer.ids(text, truncate=True)))["logits"][0]
        logits = logits.astype(np.float64)
        if model.config.get("problem_type") == "multi_label_classification":
            scores = 1 / (1 + np.exp(-logits))
        else:
            scores = np.exp(logits - logits.max())
            scores /= scores.sum()
        scores = scores.astype(np.float32)
        order = np.argsort(-scores, kind="stable")[: top_k if top_k else None]
        id2label = model.config.get("id2label")
        return [{"label": id2label[str(i)] if id2label else f"LABEL_{i}", "score": float(scores[i])} for i in order]

    async def cancel(self):
        if not self._generating:
            return False
        self._cancelled = True
        return True

    async def generate(self, prompt, return_full_text=None):
        if self.generator is None:
            raise RuntimeError("Generator model not initialized. Call init() first.")
        if self._generating:
            raise RuntimeError("Another generation is running on this instance; await it or call cancel() first.")
        self._generating, self._cancelled = True, False
        try:
            return await self._generate(prompt, self.return_full_text if return_full_text is None else return_full_text)
        finally:
            self._generating = False

    async def _generate(self, prompt, return_full_text):
        # shortcut: greedy/top-k sampling with temperature and repetition_penalty only; other generation_config
        # processors (no_repeat_ngram_size, min_length, bad_words_ids, ...) are ignored. Add them if a model needs one.
        model, config = self.generator, self.config
        tokenizer = model.tokenizer
        generation = _read_json(config["generator_model"], "generation_config.json", config.get("local_models_path"))
        add_special = bool(tokenizer.config.get("add_bos_token") or tokenizer.config.get("add_eos_token"))
        prompt_ids = tokenizer.ids(prompt, add_special_tokens=add_special, truncate=True)
        max_new_tokens = config.get("max_new_tokens", 50)
        temperature = config.get("temperature", 0.7)
        top_k = config.get("top_k", 50)
        do_sample = config.get("do_sample", True) and temperature != 0
        penalty = generation.get("repetition_penalty") or 1.0
        eos = generation.get("eos_token_id", model.config.get("eos_token_id"))
        eos = set(eos if isinstance(eos, list) else [] if eos is None else [eos])

        past = {}
        for name, info in model.inputs.items():
            if name.startswith("past_key_values"):
                if not all(isinstance(d, int) for d in info.shape[1:2] + info.shape[3:]):
                    raise NotImplementedError(f"{name} has symbolic head dimensions {info.shape}")
                dtype = np.float16 if info.type == "tensor(float16)" else np.float32
                past[name] = np.zeros((1, info.shape[1], 0, info.shape[3]), dtype)
        ids = list(prompt_ids)
        step_ids = prompt_ids
        rng = np.random.default_rng()
        while True:
            feed = {"input_ids": np.array([step_ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64)}
            if "position_ids" in model.inputs:
                # shortcut: 0-indexed positions; Transformers.js starts paligemma/gemma3 at 1.
                feed["position_ids"] = np.arange(len(ids) - len(step_ids), len(ids), dtype=np.int64)[None]
            if "use_cache_branch" in model.inputs:  # merged decoders: False only on the prompt pass
                feed["use_cache_branch"] = np.array([step_ids is not prompt_ids])
            if "num_logits_to_keep" in model.inputs:
                feed["num_logits_to_keep"] = np.array(1, np.int64)
            outputs = model.run({**feed, **past})
            logits = outputs["logits"][0, -1].astype(np.float32)
            if penalty != 1.0:
                for token in set(ids):
                    logits[token] = logits[token] * penalty if logits[token] < 0 else logits[token] / penalty
            if do_sample:
                if temperature != 1.0:
                    logits = logits / np.float32(temperature)
                k = min(top_k, len(logits)) if top_k > 0 else len(logits)
                top = np.argsort(-logits, kind="stable")[:k]
                probabilities = np.exp(logits[top].astype(np.float64) - logits[top].max())
                token = int(top[rng.choice(k, p=probabilities / probabilities.sum())])
            else:
                token = int(np.argmax(logits))
            ids.append(token)
            if len(ids) - len(prompt_ids) >= max(1, max_new_tokens) or token in eos or self._cancelled:
                break
            past = {
                k.replace("present", "past_key_values", 1): v for k, v in outputs.items() if k.startswith("present")
            }
            step_ids = [token]
            await asyncio.sleep(0)  # lets cancel() run between tokens

        text = tokenizer.text(ids, skip_special_tokens=True)
        if not return_full_text and prompt_ids:
            text = text[len(tokenizer.text(prompt_ids, skip_special_tokens=True)) :]
        return text

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        # chat(), streams and model export need the Transformers.js runtime.
        raise NotImplementedError(f"{name}() is not supported by the native backend; run it in a browser kernel.")
