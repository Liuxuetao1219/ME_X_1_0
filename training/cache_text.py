"""Precompute the Clean50 instruction embeddings used during training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

from wan.modules.t5 import T5EncoderModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--wan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    prompts = set()
    pattern = "*/tactile_replay_aloha_clean50_batch/instructions/episode*.json"
    for path in args.data_root.glob(pattern):
        prompts.update(json.loads(path.read_text(encoding="utf-8"))["seen"])
    prompts = sorted(prompts)

    encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device="cuda",
        checkpoint_path=str(args.wan / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(args.wan / "google" / "umt5-xxl"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload_path = args.output.with_suffix(".bin")
    embedding_index = []
    offset = 0
    with payload_path.open("wb") as stream:
        for start in range(0, len(prompts), args.batch_size):
            embeddings = encoder(prompts[start : start + args.batch_size], "cuda")
            for embedding in embeddings:
                embedding = embedding.detach().to("cpu", dtype=torch.bfloat16).contiguous()
                length, width = embedding.shape
                embedding.view(torch.uint16).numpy().tofile(stream)
                embedding_index.append((offset, length, width))
                offset += length * width
    torch.save(
        {
            "cache_format": "variable_length_per_prompt_disk_v2",
            "prompts": prompts,
            "embedding_index": embedding_index,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
