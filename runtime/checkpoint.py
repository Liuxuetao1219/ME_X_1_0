from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


VARIANT = "ME-X-1.0"


def _resolve_asset(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_metadata(export_dir: str | Path) -> dict[str, Any]:
    root = Path(export_dir).expanduser().resolve()
    paths = {
        "manifest": root / "manifest.json",
        "config": root / "model_config.json",
        "model": root / "model.pt",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing ME-X-1.0 {name}: {path}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    if manifest.get("format_version") != 3 or config.get("format_version") != 3:
        raise ValueError("ME-X-1.0 requires export format_version=3")
    if manifest.get("variant") != VARIANT or config.get("variant") != VARIANT:
        raise ValueError("Checkpoint is not the ME-X-1.0 variant")
    if manifest.get("step") != config.get("step") or not isinstance(config.get("step"), int):
        raise ValueError("Manifest/config checkpoint step mismatch")
    if config.get("architecture") != "wam_va" or config.get("joint_attention_mode") != "full_bidirectional":
        raise ValueError("ME-X-1.0 architecture contract mismatch")
    common = config.get("common", {})
    expected_common = {
        "action_dim": 14,
        "state_dim": 14,
        "num_video_frames": 8,
        "video_height": 384,
        "video_width": 320,
        "global_downsample_rate": 3,
        "video_action_freq_ratio": 2,
        "action_chunk_size": 16,
        "temporal_contract_name": "robotwin_save15_sim250_stride3_v1",
    }
    mismatch = {key: common.get(key) for key, value in expected_common.items() if common.get(key) != value}
    if mismatch:
        raise ValueError(f"ME-X-1.0 common contract mismatch: {mismatch}")
    tactile = config.get("model", {}).get("tactile", {})
    if tactile.get("enabled") is not True or tactile.get("codec_type") != "v3_anatomy":
        raise ValueError("ME-X-1.0 requires the frozen V3 anatomy tactile codec")
    expert = config.get("model", {}).get("tactile_expert", {})
    for key, value in {
        "latent_dim": 256,
        "latent_slices": 18,
        "queries_per_slice": 12,
        "condition_slices": 2,
        "hidden_size": 512,
    }.items():
        if expert.get(key) != value:
            raise ValueError(f"ME-X-1.0 tactile expert mismatch: {key}")
    tactile_checkpoint = _resolve_asset(root, str(tactile["checkpoint_path"]))
    if not tactile_checkpoint.is_file():
        raise FileNotFoundError(f"Missing V3 tactile checkpoint: {tactile_checkpoint}")
    return {
        "root": root,
        "manifest": manifest,
        "config": config,
        "model_path": paths["model"],
        "tactile_checkpoint": tactile_checkpoint,
    }


def load_model_state_strictly(model: nn.Module, metadata: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(
        metadata["model_path"], map_location="cpu", weights_only=False, mmap=True
    )
    state = checkpoint.get("module") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise TypeError("ME-X-1.0 checkpoint must contain a module state dictionary")
    model.load_state_dict(state, strict=True)
    return {"state_key_count": len(state)}
