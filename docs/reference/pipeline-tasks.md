# Pipeline tasks: arguments and results

Keyword arguments to a pipeline call are forwarded as Transformers.js options for that task, and must be JSON-serializable. Results are plain Python dicts and lists:

| Task | Input | Task-specific arguments | Output |
|------|-------|-------------------------|--------|
| `zero-shot-classification` | string or list of strings | `candidate_labels` (required), `hypothesis_template`, `multi_label` | `{"sequence", "labels", "scores"}` sorted by score, or a list for list input |
| `translation` | string or list of strings | generation options such as `max_new_tokens`; `src_lang`/`tgt_lang` for multilingual models (e.g. NLLB, M2M100) | `{"translation_text"}`, or a list for list input |
| `image-classification` | image (see below) or list of images | `top_k` | `[{"label", "score"}]`, or a list of those for list input |
| `zero-shot-image-classification` | image or list of images | `candidate_labels` (required), `hypothesis_template` | `[{"label", "score"}]` sorted by score |
| `image-feature-extraction` | image or list of images | `pool=True` to use the model's pooled output, for models that have a pooler layer (default: the first, CLS, token's vector) | a list of floats per image, or a list of those for list input |
| `object-detection` | image or list of images | `threshold`, `percentage` (default `False`: box coordinates in pixels) | `[{"label", "score", "box": {"xmin", "ymin", "xmax", "ymax"}}]` |
| `background-removal` | image or list of images | none | PNG bytes of the cut-out (RGBA, transparent background), or a list of them |
| `depth-estimation` | image or list of images | none | `{"depth": PNG bytes, "predicted_depth": float32 array of shape (height, width)}` |
| `image-segmentation` | image or list of images | Transformers.js options for the model, none tried here; pass `dtype="fp32"` to `LLM.pipeline()` for DETR panoptic: its quantized default gives wrong masks and its fp16 file can run out of memory | `[{"label", "score", "mask": PNG bytes}]`, `score` is `None` when the model gives none |
| `image-to-image` | image or list of images | none | PNG bytes of the result (a 2x upscale for Swin2SR) |
| `text-to-speech` / `text-to-audio` | one string | model options such as `speaker_embeddings` (a URL string), if the model takes them | a `Speech` object (see below) |
| `automatic-speech-recognition` | audio (see below) | `return_timestamps`, `chunk_length_s`, `stride_length_s`, `language`/`task` for multilingual Whisper | `{"text"}`, plus `"chunks"` with timestamps |
| `audio-classification` | audio (as for speech recognition) or a list of clips | `top_k` | `[{"label", "score"}]` sorted by score, one list per clip |
