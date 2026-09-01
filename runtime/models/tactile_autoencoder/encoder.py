from __future__ import annotations

import math

import torch
import torch.nn as nn

from .anatomy import anatomy_token_ids, anatomy_token_presence
from .config import AnatomyTactileAEV3Config
from .layers import FP32LayerNorm, LayerScaledSelfAttention
from .surface_encoder import SurfaceTokenizerV3


class AnatomyEncoderV3(nn.Module):
    """Frame-wise 30-region encoder with 12 fixed anatomical output slots."""

    def __init__(self, config: AnatomyTactileAEV3Config):
        super().__init__()
        # The audited Clean6 local DCAE preserves each 10x14 surface. Regions do
        # not communicate across anatomical groups before the 12 named tokens
        # have been formed.
        self.surface_tokenizer = SurfaceTokenizerV3(config)
        region_width = config.tokens_per_region * config.latent_dim
        self.region_projection = nn.Sequential(
            FP32LayerNorm(region_width),
            nn.Linear(region_width, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.region_norm = FP32LayerNorm(config.latent_dim)
        self.pool_hand_embedding = nn.Embedding(
            config.hand_side_types, config.latent_dim, padding_idx=0
        )
        self.pool_part_embedding = nn.Embedding(6, config.latent_dim)
        self.pool_query_projection = nn.Linear(config.latent_dim, config.latent_dim, bias=False)
        self.pool_key_projection = nn.Linear(config.latent_dim, config.latent_dim, bias=False)
        self.pool_value_projection = nn.Linear(config.latent_dim, config.latent_dim, bias=False)
        self.pool_output_norm = FP32LayerNorm(config.latent_dim)
        self.anatomy_layers = nn.ModuleList(
            LayerScaledSelfAttention(
                config.latent_dim,
                config.attention_heads,
                config.ffn_dim,
                config.anatomy_residual_scale_init,
            )
            for _ in range(config.anatomy_layers)
        )
        self.output_norm = FP32LayerNorm(config.latent_dim)
        for embedding in (self.pool_hand_embedding, self.pool_part_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        with torch.no_grad():
            self.pool_hand_embedding.weight[0].zero_()
        self.register_buffer(
            "anatomy_hand_ids",
            torch.tensor([1] * 6 + [2] * 6, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "anatomy_part_ids",
            torch.tensor(list(range(6)) * 2, dtype=torch.long),
            persistent=True,
        )
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
        tokenized = self.surface_tokenizer(
            force,
            region_mask=region_mask,
            grid_mask=grid_mask,
            uv_coordinates=uv_coordinates,
            hand_side_id=hand_side_id,
            finger_id=finger_id,
            segment_id=segment_id,
        )
        batch, regions = region_mask.shape
        local_valid = tokenized["surface_token_valid"]
        ordered_local_tokens = tokenized["surface_tokens"] * local_valid[..., None]
        region_features = self.region_projection(ordered_local_tokens.flatten(2))
        region_features = region_features * region_mask[..., None].to(region_features.dtype)
        region_features = self.region_norm(region_features)

        region_token_id = anatomy_token_ids(hand_side_id, finger_id, region_mask)
        anatomy_valid = anatomy_token_presence(region_token_id)
        token_ids = torch.arange(self.config.anatomy_tokens, device=force.device)
        membership = region_token_id[..., None].eq(token_ids)
        identity = self.pool_hand_embedding(self.anatomy_hand_ids)
        identity = identity + self.pool_part_embedding(self.anatomy_part_ids)
        query = self.pool_query_projection(identity).to(region_features.dtype)
        key = self.pool_key_projection(region_features)
        value = self.pool_value_projection(region_features)
        logits = torch.einsum("bri,ki->brk", key, query) / math.sqrt(self.config.latent_dim)
        logits = logits.masked_fill(~membership, torch.finfo(logits.dtype).min)
        safe_max = torch.where(
            anatomy_valid[:, None],
            logits.amax(dim=1, keepdim=True),
            torch.zeros_like(logits[:, :1]),
        )
        logits = logits - safe_max
        unnormalized = logits.exp() * membership.to(logits.dtype)
        weights = unnormalized / unnormalized.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        anatomy_tokens = torch.einsum("brk,brd->bkd", weights, value)
        anatomy_tokens = self.pool_output_norm(
            anatomy_tokens + identity[None].to(value.dtype)
        )
        anatomy_tokens = anatomy_tokens * anatomy_valid[..., None].to(anatomy_tokens.dtype)
        anatomy_pre_relation = anatomy_tokens
        for layer in self.anatomy_layers:
            anatomy_tokens = layer(anatomy_tokens, anatomy_valid)
        anatomy_tokens = self.output_norm(anatomy_tokens)
        anatomy_tokens = anatomy_tokens * anatomy_valid[..., None].to(anatomy_tokens.dtype)
        return {
            **tokenized,
            "region_features": region_features,
            "region_anatomy_token_id": region_token_id,
            "region_pooling_weights": weights,
            "anatomy_token_valid": anatomy_valid,
            "anatomy_pre_relation": anatomy_pre_relation,
            "frame_tokens": anatomy_tokens,
        }
