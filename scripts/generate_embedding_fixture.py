# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Generate original, randomly initialized BERT smoke assets; no pretrained weights."""

import hashlib
import json
from pathlib import Path

import onnx
import torch
import transformers
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/models/lmlite/tiny-bert"
SEED = 2026


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "onnx").mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "hello",
        ".",
        "a",
        "considerably",
        "longer",
        "sentence",
        "with",
        "many",
        "more",
        "tokens",
        "to",
        "embed",
    ]
    tokenizer = Tokenizer(models.WordPiece({word: index for index, word in enumerate(vocab)}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=64,
    )
    fast.save_pretrained(ROOT)
    # Use the supported BERT tokenizer implementation in Transformers.js.
    tokenizer_config = json.loads((ROOT / "tokenizer_config.json").read_text())
    tokenizer_config["tokenizer_class"] = "BertTokenizer"
    (ROOT / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    config.architectures = ["BertModel"]
    config.save_pretrained(ROOT)
    config._attn_implementation = "eager"
    model = BertModel(config, add_pooling_layer=False).eval()

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).last_hidden_state

    inputs = (torch.tensor([[2, 5, 6, 3]]), torch.ones((1, 4), dtype=torch.long), torch.zeros((1, 4), dtype=torch.long))
    names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(
        Encoder(),
        inputs,
        str(ROOT / "onnx/model.onnx"),
        dynamo=False,
        opset_version=17,
        input_names=names,
        output_names=["last_hidden_state"],
        dynamic_axes={name: {0: "batch", 1: "sequence"} for name in [*names, "last_hidden_state"]},
    )
    quantize_dynamic(
        ROOT / "onnx/model.onnx",
        ROOT / "onnx/model_quantized.onnx",
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul"],
    )
    for name in ["model.onnx", "model_quantized.onnx"]:
        onnx.checker.check_model(onnx.load(ROOT / "onnx" / name))
    quantized_graph = onnx.load(ROOT / "onnx/model_quantized.onnx")
    if not any(node.op_type == "MatMulInteger" for node in quantized_graph.graph.node):
        raise RuntimeError("The q8 fixture must contain quantized matrix operations")
    assets = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "onnx/model.onnx",
        "onnx/model_quantized.onnx",
    ]
    manifest = {
        "license": "Apache-2.0",
        "seed": SEED,
        "hidden_size": 16,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "assets": {
            name: {
                "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                "size": (ROOT / name).stat().st_size,
            }
            for name in assets
        },
    }
    (ROOT / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {sum(asset['size'] for asset in manifest['assets'].values())} bytes in {ROOT}")


if __name__ == "__main__":
    main()
