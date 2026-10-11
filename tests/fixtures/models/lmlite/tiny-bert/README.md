# Tiny BERT contract fixture

Copyright 2026 Emmanuel Decitre. SPDX-License-Identifier: Apache-2.0.
These original randomly initialized weights and the hand-written minimal
vocabulary are distributed under the repository's [Apache-2.0 license](../../../../../LICENSE).
No pretrained weights or third-party tokenizer vocabulary are included.

Regenerate from the repository root:

```bash
pixi run --environment model-fixture generate-embedding-fixture
```

The generator uses seed 2026, a one-layer BERT with hidden size 16, two attention
heads, a 32-unit feed-forward layer, and a 17-token vocabulary. Both ONNX variants
return unpooled `last_hidden_state` with dynamic batch and sequence dimensions.
The q8 variant quantizes constant MatMul weights to unsigned 8-bit integers.
The model is not trained and is not suitable for semantic search.

`fixture.json` records asset sizes, SHA-256 hashes, and generator library versions.
The smoke harness verifies these hashes before inference. Export tooling belongs
to the separate `model-fixture` environment; normal smoke execution needs only
the checked-in artifacts and the existing Playwright environment.

Binary exports can vary across platforms/toolchain versions. Regeneration must
be reviewed together with the manifest and exercised with `pixi run embedding-smoke`.
Export tooling supports Apple Silicon macOS and Linux (PyTorch 2.8 has no Intel
macOS wheels); inference with the checked-in artifacts does not require PyTorch.
