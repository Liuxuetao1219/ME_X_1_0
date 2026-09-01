"""Tactile expert used by ME-X-1.0 joint attention."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wan.modules.model import WanLayerNorm, WanRMSNorm


@contextmanager
def _fork_cpu_seed(seed: int):
    state = torch.get_rng_state()
    torch.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)


def sinusoidal_embedding_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("sin-cos embedding dimension must be even")
    positions = positions.float().reshape(-1, 1)
    omega = torch.arange(dim // 2, device=positions.device, dtype=torch.float32)
    omega = torch.pow(10000.0, -omega / (dim // 2))
    angles = positions * omega.reshape(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


@dataclass(frozen=True)
class UniversalTactileExpertConfig:
    latent_dim: int = 256
    latent_slices: int = 18
    queries_per_slice: int = 12
    condition_slices: int = 2
    hidden_size: int = 512
    ffn_multiplier: int = 4
    num_layers: int = 30
    wan_num_heads: int = 24
    wan_head_dim: int = 128
    norm_eps: float = 1.0e-6
    freq_dim: int = 256
    initialization_seed: int = 42042

    @classmethod
    def from_mapping(
        cls, values: Dict[str, Any] | None, *, num_layers: int
    ) -> "UniversalTactileExpertConfig":
        values = dict(values or {})
        values["num_layers"] = int(num_layers)
        return cls(**values)

    def __post_init__(self) -> None:
        contract = (
            self.latent_dim,
            self.latent_slices,
            self.queries_per_slice,
            self.condition_slices,
        )
        if contract != (256, 18, 12, 2):
            raise ValueError(
                f"ME-X-1.0 requires tactile latent contract (256,18,12,2), got {contract}"
            )
        if self.hidden_size != 512:
            raise ValueError("ME-X-1.0 tactile expert requires hidden_size=512")

    @property
    def future_slices(self) -> int:
        return self.latent_slices - self.condition_slices

    @property
    def sequence_length(self) -> int:
        return self.latent_slices * self.queries_per_slice

    @property
    def wan_dim(self) -> int:
        return self.wan_num_heads * self.wan_head_dim


class UniversalTactileTokenizer(nn.Module):
    def __init__(self, config: UniversalTactileExpertConfig):
        super().__init__()
        self.config = config
        with _fork_cpu_seed(config.initialization_seed):
            self.input_projection = nn.Linear(config.latent_dim, config.hidden_size)
            self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
            self.type_embedding = nn.Parameter(
                torch.randn(2, config.hidden_size) * 0.02
            )
            self.slice_embedding = nn.Parameter(
                torch.randn(config.latent_slices, config.hidden_size) * 0.02
            )
            self.query_embedding = nn.Parameter(
                torch.randn(config.queries_per_slice, config.hidden_size) * 0.02
            )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        expected = (cfg.latent_slices, cfg.queries_per_slice, cfg.latent_dim)
        if tuple(latent.shape[1:]) != expected:
            raise ValueError(f"Expected latent [B,{expected}], got {tuple(latent.shape)}")
        projected = self.input_projection(latent)
        tokens = F.layer_norm(
            projected.float(),
            (cfg.hidden_size,),
            self.input_norm.weight.float(),
            self.input_norm.bias.float(),
            self.input_norm.eps,
        ).to(projected.dtype)
        type_ids = torch.cat(
            (
                torch.zeros(cfg.condition_slices, dtype=torch.long, device=latent.device),
                torch.ones(cfg.future_slices, dtype=torch.long, device=latent.device),
            )
        )
        tokens = (
            tokens
            + self.type_embedding[type_ids][None, :, None]
            + self.slice_embedding[None, :, None]
            + self.query_embedding[None, None]
        )
        return tokens.reshape(latent.shape[0], cfg.sequence_length, cfg.hidden_size)


class TactileExpertBlock(nn.Module):
    def __init__(self, config: UniversalTactileExpertConfig):
        super().__init__()
        self.norm1 = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.norm2 = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.wan_tactile_qkv = nn.Parameter(
            torch.randn(
                3,
                config.wan_num_heads,
                config.hidden_size,
                config.wan_head_dim,
            )
            / (config.hidden_size * config.wan_head_dim) ** 0.5
        )
        self.wan_tactile_norm_q = WanRMSNorm(config.wan_dim, eps=config.norm_eps)
        self.wan_tactile_norm_k = WanRMSNorm(config.wan_dim, eps=config.norm_eps)
        self.wan_tactile_o = nn.Linear(
            config.wan_dim, config.hidden_size, bias=False
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.ffn_multiplier),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_size * config.ffn_multiplier, config.hidden_size),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, config.hidden_size) / config.hidden_size**0.5
        )


class UniversalTactileOutputHead(nn.Module):
    def __init__(self, config: UniversalTactileExpertConfig):
        super().__init__()
        self.config = config
        self.norm = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.modulation = nn.Parameter(
            torch.randn(1, 2, config.hidden_size) / config.hidden_size**0.5
        )
        self.projection = nn.Linear(config.hidden_size, config.latent_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self, tokens: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        shift, scale = (
            self.modulation.unsqueeze(0) + time_embedding.unsqueeze(2)
        ).chunk(2, dim=2)
        values = self.projection(
            self.norm(tokens) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        )
        values = values.reshape(
            tokens.shape[0],
            cfg.latent_slices,
            cfg.queries_per_slice,
            cfg.latent_dim,
        )
        return torch.cat(
            (
                torch.zeros_like(values[:, : cfg.condition_slices]),
                values[:, cfg.condition_slices :],
            ),
            dim=1,
        )


class UniversalTactileExpert(nn.Module):
    def __init__(self, config: UniversalTactileExpertConfig):
        super().__init__()
        self.config = config
        self.tokenizer = UniversalTactileTokenizer(config)
        with _fork_cpu_seed(config.initialization_seed + 2):
            self.time_embedding = nn.Sequential(
                nn.Linear(config.freq_dim, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.time_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size * 6)
            )
            self.blocks = nn.ModuleList(
                [TactileExpertBlock(config) for _ in range(config.num_layers)]
            )
            self.output_head = UniversalTactileOutputHead(config)

    def get_time_embeddings(
        self, timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        slice_t = torch.cat(
            (
                torch.zeros(timestep.shape[0], cfg.condition_slices, device=timestep.device),
                timestep.float()[:, None].expand(-1, cfg.future_slices),
            ),
            dim=1,
        )
        token_t = slice_t.repeat_interleave(cfg.queries_per_slice, dim=1)
        time_features = sinusoidal_embedding_1d(cfg.freq_dim, token_t).reshape(
            timestep.shape[0], cfg.sequence_length, cfg.freq_dim
        )
        time_features = time_features.to(self.time_embedding[0].weight.dtype)
        embedded = self.time_embedding(time_features)
        projected = self.time_projection(embedded).reshape(
            timestep.shape[0], cfg.sequence_length, 6, cfg.hidden_size
        )
        return embedded, projected

    @staticmethod
    def modulation(block: TactileExpertBlock, projected: torch.Tensor):
        return (block.modulation.unsqueeze(0) + projected).chunk(6, dim=2)

    @staticmethod
    def apply_ffn(tokens: torch.Tensor, block: TactileExpertBlock, modulation):
        values = block.norm2(tokens).float() * (
            1 + modulation[4].squeeze(2)
        ) + modulation[3].squeeze(2)
        return tokens + block.ffn(values).to(tokens.dtype) * modulation[5].squeeze(2)
