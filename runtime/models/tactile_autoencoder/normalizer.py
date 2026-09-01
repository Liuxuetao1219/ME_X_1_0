from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class RobustChannelStats:
    bias: tuple[float, float, float]
    scale: tuple[float, float, float]
    quantile: float = 0.999


class DomainRobustNormalizer:
    """Zero-preserving, per-domain robust normalization shared by regions."""

    def __init__(
        self,
        groups: Mapping[int, RobustChannelStats],
        eps: float = 1e-6,
        bias_mode: str = "fixed_zero",
    ):
        if not groups:
            raise ValueError("At least one normalization group is required")
        self.groups = dict(groups)
        self.eps = float(eps)
        self.bias_mode = str(bias_mode)
        if self.bias_mode != "fixed_zero":
            raise ValueError(
                "Only fixed_zero is supported without explicit no-contact labels"
            )

    def _parameters(
        self, group_ids: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        biases, scales = [], []
        for group in group_ids.reshape(-1).tolist():
            if int(group) not in self.groups:
                raise KeyError(f"Unknown normalization group {group}")
            stats = self.groups[int(group)]
            biases.append(stats.bias)
            scales.append(stats.scale)
        # Channel is the third dimension from the end: [B,...,3,H,W].
        shape = (len(biases),) + (1,) * (reference.ndim - 4) + (3, 1, 1)
        bias = reference.new_tensor(biases).reshape(shape)
        scale = reference.new_tensor(scales).reshape(shape)
        return bias, scale

    def normalize(
        self, values: torch.Tensor, group_ids: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        bias, scale = self._parameters(group_ids, values)
        result = (values - bias) / scale.clamp_min(self.eps)
        return result * valid.to(result.dtype)

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "DomainRobustNormalizer":
        groups = {
            int(key): RobustChannelStats(
                tuple(value["bias"]), tuple(value["scale"]), float(value["quantile"])
            )
            for key, value in state["groups"].items()
        }
        if int(state.get("version", 0)) != 2:
            raise ValueError("Unsupported tactile normalizer schema; regenerate stats")
        return cls(
            groups,
            eps=float(state.get("eps", 1e-6)),
            bias_mode=str(state.get("bias_mode", "")),
        )
