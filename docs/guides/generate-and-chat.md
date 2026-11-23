# Generate text and chat

## Creating an LLM Instance

`LLM.create(...)` displays `tqdm` download progress in the notebook cell, with status messages during model preparation, initialization, and warm-up. Browser kernels explicitly use notebook widgets when `ipywidgets` is available, without starting a background monitor thread. Missing widget support produces a warning and text progress instead. The JupyterLite frontend must also support widgets to render the bar. The cell stays busy until initialization and warm-up finish.

Phases without measurable byte progress show a status rather than a zero-byte counter; successful initialization ends at 100%. Cached models may initialize without any download progress. Model weights and inference sessions still require sufficient browser memory even when files are fetched individually.

```python
from lmlite import LLM

# Create LLM instance (downloads model on first run)
# You can pass either a full model ID ("Xenova/gpt2") or a short name ("gpt2").
# Short names default to the Xenova namespace.
llm = await LLM.create(
    generator_model="gpt2",
    embedding_model="all-MiniLM-L6-v2",
)
```

## Text Generation

```python
# Generate text
text = await llm.generate("Python is a great language because")
print(text)
```

## Chat with Message History

```python
# Chat with message history
messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What can I use it for?"}
]
response = await llm.chat(messages)
print(response)
```

## Generation Without Prompt Echo

For generation without prompt echo, pass `return_full_text=False` to `LLM.create()`. This forwards the option to Transformers.js; `generate(prompt)` then returns only the continuation, with no Python-side prompt removal needed.

The default, `True`, preserves full-text output. Custom chat wrappers using `False` should consume the returned answer directly rather than stripping a prompt prefix. This option does not apply a model-specific chat template or prevent repetition.

```python
llm = await LLM.create(
    generator_model="gpt2",
    return_full_text=False
)

# Returns only the generated continuation
continuation = await llm.generate("Python is")
```

## Model-Aware Chat

`chat(messages)` uses the chat template supplied by the loaded generator's tokenizer through Transformers.js, rather than guessing a format from the model name. It returns only the assistant response, regardless of `return_full_text`. Messages require string `role` and `content` fields.

```python
answer = await llm.chat([
    {"role": "system", "content": "Answer directly and concisely."},
    {"role": "user", "content": "What are text embeddings?"},
])
```

Models without a tokenizer chat template retain the generic `User:` / `Assistant:` format; this does not make completion models instruction-tuned. Invalid messages, unsupported template roles, and template failures raise errors without retrying with a different format. Chat does not manage memory or retrieve documents automatically; the lcLite notebook demonstrates those separately.

## Streaming and Cancellation

`chat_stream(messages)` and `generate_stream(prompt)` are async iterators that yield text chunks as Transformers.js generates them, so the notebook shows output while the model runs.

```python
async for chunk in llm.chat_stream([
    {"role": "user", "content": "Explain embeddings simply."},
]):
    print(chunk, end="", flush=True)
```

- **Chunks** are nonempty strings from Transformers.js's `TextStreamer` (special tokens skipped). It emits whole words, so a chunk is usually a word with its whitespace, not a single token. `"".join(chunks).strip()` is the `chat()` answer for the same tokens. `generate_stream()` never includes the prompt, whatever `return_full_text` is.
- **Laziness**: generation starts when you first iterate, not when you call `chat_stream()`.
- **Cancellation**: `await llm.cancel()` stops the running `generate()`, `chat()` or stream after its current token and returns `False` if nothing was running. A cancelled stream ends early without error; `generate()`/`chat()` return the partial text. The instance remains usable.
- **Early exit**: leaving the loop (`break`, an exception, or a kernel interrupt) cancels the generation. Kernel interrupts reach Python only between chunks and only where the kernel supports interrupts; if one never arrives, restart the kernel.
- **Concurrency**: one generation runs per instance at a time. Starting another while one is running raises an error (for streams, on the first chunk); a generation started right after a cancellation waits for it to stop instead.
- **Errors** during generation are raised from the iterator.
- **Backends**: streaming works with any generator model and backend. Cancellation is checked between tokens, so a slow first token (prompt processing on a long input) is not interrupted.

## Token Probabilities

`chat_tokens(messages, alternatives=1)` and `generate_tokens(prompt, alternatives=1)` stream one record per generated token instead of text chunks:

```python
async for token in llm.chat_tokens(messages, alternatives=1):
    runner_up = token["alternatives"][0]
    print(f"{token['text']}[{token['prob']:.2f}] (runner-up {runner_up['text']!r} {runner_up['prob']:.2f})")
```

- **Records**: `{"text": str, "prob": float, "alternatives": [{"text": str, "prob": float}, ...]}`. `alternatives` lists the `alternatives` most probable tokens other than the picked one, most probable first (pass `0` for none).
- **Probabilities** are those the token picker used: the softmax of the final logits after repetition penalty and, when sampling, temperature. With `do_sample=False` the picked token is always the most probable, so the first alternative is the true runner-up. With sampling, an alternative can be more probable than the pick, and top-k sampling draws from probabilities renormalized over its k tokens.
- **Text**: joined `text` values equal joined `chat_stream()` chunks. A token holding only part of a multi-byte character has empty `text`; the completed character appears with the next token. Special tokens such as end-of-turn are not reported. An alternative's `text` is decoded alone, so a partial character shows as `�`, and tokenizers that drop a word's leading space when decoding a single token (SentencePiece, e.g. Llama) show it without the space.
- **Cost**: about a millisecond per token for a softmax over the vocabulary. Laziness, cancellation, concurrency and errors behave as in `chat_stream()`.
