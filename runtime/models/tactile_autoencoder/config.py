from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnatomyTactileAEV3Config:
    """Frozen public architecture contract for tactile AE V3."""

    max_regions: int = 30
    input_channels: int = 6
    latent_dim: int = 256
    tokens_per_region: int = 4
    anatomy_tokens: int = 12
    anatomy_layers: int = 2
    anatomy_residual_scale_init: float = 0.1
    decoder_layers: int = 2
    attention_heads: int = 8
    ffn_dim: int = 1024
    hand_side_types: int = 3
    surface_type_types: int = 17
    support_threshold: float = 0.5

    def __post_init__(self) -> None:
        fixed = {
            "max_regions": 30,
            "input_channels": 6,
            "latent_dim": 256,
            "tokens_per_region": 4,
            "anatomy_tokens": 12,
            "hand_side_types": 3,
            "surface_type_types": 17,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Tactile AE V3 fixes {name}={expected}")
        if self.latent_dim % self.attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if not 0.0 < self.support_threshold < 1.0:
            raise ValueError("support_threshold must be in (0,1)")
        if not 0.0 <= self.anatomy_residual_scale_init <= 1.0:
            raise ValueError("anatomy_residual_scale_init must be in [0,1]")

    @property
    def tokens_per_surface(self) -> int:
        """Number of latent tokens produced for each tactile surface."""

        return self.tokens_per_region
