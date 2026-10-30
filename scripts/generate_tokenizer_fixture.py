# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Generate a tiny Unigram/Metaspace tokenizer and its reference ids from the Rust `tokenizers` library."""

import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/models/lmlite/tiny-unigram"

CORPUS = [
    "sushi right next door to a pharmacy",
    "the bakery is across the street from the station",
    "Café crème brûlée à côté de la pharmacie",
    "la boulangerie est en face de la gare",
    "寿司屋は薬局の隣にあります",
    "パン屋は駅の向かいです",
    "Суши рядом с аптекой",
    "пекарня напротив вокзала",
    "🍣 next to 💊 and 🏥",
]

CASES = [
    "sushi right next door to a pharmacy",
    "Café crème brûlée à côté de la pharmacie",
    "寿司屋は薬局の隣にあります",
    "Суши рядом с аптекой",
    "🍣 next to 💊 🏥 🚲",
    "  leading and   repeated   spaces ",
    "\ttabs\nand newlines\n",
    "unseen words: xylophone, Zürich, 東京",
    "",
    " ".join(CORPUS * 20),
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    trainer = trainers.UnigramTrainer(vocab_size=250, special_tokens=["<pad>", "</s>", "<unk>"], unk_token="<unk>")
    tokenizer.train_from_iterator(CORPUS * 10, trainer)
    # A post-processor so add_special_tokens=True visibly differs from the default False.
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>", pair="$A </s> $B:1 </s>:1", special_tokens=[("</s>", tokenizer.token_to_id("</s>"))]
    )
    tokenizer.save(str(ROOT / "tokenizer.json"))
    # model_max_length is below the long case's length, proving encode() does not truncate.
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 16,
    }
    (ROOT / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")

    reference = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
    cases = [
        {
            "text": text,
            "ids": reference.encode(text, add_special_tokens=False).ids,
            "ids_with_special_tokens": reference.encode(text, add_special_tokens=True).ids,
        }
        for text in CASES
    ]
    manifest = {
        "license": "Apache-2.0",
        "tokenizers_version": tokenizers.__version__,
        "vocab_size": reference.get_vocab_size(),
        "assets": {
            name: {
                "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                "size": (ROOT / name).stat().st_size,
            }
            for name in ["tokenizer.json", "tokenizer_config.json"]
        },
        "cases": cases,
    }
    (ROOT / "fixture.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Generated {len(cases)} cases, vocab size {manifest['vocab_size']}, in {ROOT}")


if __name__ == "__main__":
    main()
