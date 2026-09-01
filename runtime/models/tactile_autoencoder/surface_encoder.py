from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AnatomyTactileAEV3Config
from .layers import FP32LayerNorm


class ResidualBlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


def surface_type_ids(finger_id: torch.Tensor, segment_id: torch.Tensor) -> torch.Tensor:
    table = finger_id.new_full((7, 5), -1)
    table[0, 0] = 0
    for finger, base in ((1, 1), (2, 4), (3, 7), (4, 10), (6, 13)):
        table[finger, 1] = base
        table[finger, 2] = base + 1
        table[finger, 3] = base + 2
    table[5, 4] = 16
    result = table[finger_id, segment_id]
    if (result < 0).any():
        invalid = torch.stack((finger_id[result < 0], segment_id[result < 0]), dim=-1)
        raise ValueError(f"Invalid finger/segment combination: {invalid[0].tolist()}")
    return result


class LocalSurfaceEncoderV3(nn.Module):
    """Clean six-channel encoder for one complete 10x14 tactile surface."""

    def __init__(self, config: AnatomyTactileAEV3Config):
        super().__init__()
        self.shortcut_projection = nn.Conv2d(4 * config.input_channels, 128, 1)
        self.main_input = nn.Conv2d(config.input_channels, 64, 3, padding=1)
        self.main_norm64 = nn.GroupNorm(8, 64)
        self.main_residual64 = ResidualBlock2d(64)
        self.main_downsample = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.main_norm128 = nn.GroupNorm(8, 128)
        self.main_residual128 = ResidualBlock2d(128)
        self.fusion_residual = ResidualBlock2d(128)
        self.output_projection = nn.Conv2d(128, config.latent_dim, 1)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        shortcut = self.shortcut_projection(F.pixel_unshuffle(values, 2))
        main = self.main_residual64(F.silu(self.main_norm64(self.main_input(values))))
        main = self.main_residual128(F.silu(self.main_norm128(self.main_downsample(main))))
        fused = self.fusion_residual(main + shortcut)
        return {
            "shortcut": shortcut,
            "main": main,
            "feature_map": self.output_projection(fused),
        }


class SurfaceTokenizerV3(nn.Module):
    def __init__(self, config: AnatomyTactileAEV3Config):
        super().__init__()
        self.surface_encoder = LocalSurfaceEncoderV3(config)
        self.hand_embedding = nn.Embedding(config.hand_side_types, config.latent_dim, padding_idx=0)
        self.surface_type_embedding = nn.Embedding(
            config.surface_type_types, config.latent_dim, padding_idx=0
        )
        self.output_norm = FP32LayerNorm(config.latent_dim)
        for embedding in (self.hand_embedding, self.surface_type_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.config = config

    def forward(
        self,
        force: torch.Tensor,
        *,
        region_mask: torch.Tensor,
        grid_mask: torch.Tensor,
        uv_coordinates: torch.Tensor,
        hand_side_id: torch.Tensor,
        finger_id: torch.Tensor,
        segment_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, regions = force.shape[:2]
        geometry = region_mask[:, :, None, None] & grid_mask
        force = force * geometry[:, :, None]
        uv_coordinates = uv_coordinates * geometry[:, :, None]
        values = torch.cat(
            (force, geometry[:, :, None].to(force.dtype), uv_coordinates.to(force.dtype)), dim=2
        )
        encoded = self.surface_encoder(values.flatten(0, 1))
        hidden_mask = F.max_pool2d(
            geometry.flatten(0, 1)[:, None].float(), kernel_size=3, stride=2, padding=1
        )
        feature_map = encoded["feature_map"]
        hidden_mask = hidden_mask.to(feature_map.dtype)
        numerator = F.adaptive_avg_pool2d(feature_map * hidden_mask, (2, 2))
        denominator = F.adaptive_avg_pool2d(hidden_mask, (2, 2))
        surface_tokens = (numerator / denominator.clamp_min(1.0e-6)).flatten(2).transpose(1, 2)
        token_valid = denominator.flatten(1).gt(0)
        surface_tokens = surface_tokens.reshape(batch, regions, 4, -1)
        token_valid = token_valid.reshape(batch, regions, 4)
        surface_ids = surface_type_ids(finger_id, segment_id)
        identity = self.hand_embedding(hand_side_id) + self.surface_type_embedding(surface_ids)
        identity = identity.to(surface_tokens.dtype)
        surface_tokens = self.output_norm(surface_tokens + identity[:, :, None])
        surface_tokens = surface_tokens * token_valid[..., None].to(surface_tokens.dtype)
        return {
            "surface_tokens": surface_tokens,
            "surface_token_valid": token_valid,
            "surface_type_id": surface_ids,
            "surface_feature_map": feature_map.reshape(batch, regions, 256, 5, 7),
        }
