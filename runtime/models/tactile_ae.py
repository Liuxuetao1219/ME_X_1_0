from __future__ import annotations

from pathlib import Path

import torch

from .tactile_autoencoder import (
    AnatomyEncoderV3,
    AnatomyTactileAEV3Config,
    DomainRobustNormalizer,
)


class TactileAE:
    """Immutable V3 TacAE encoder and the audited RoboTwin physical contract.

    The codec is intentionally not registered as a ME-X-1.0 submodule because
    its frozen weights are loaded from ``tactile_ae.pt``.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        checkpoint = Path(checkpoint_path).resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        schema = state.get("schema", {})
        if schema.get("architecture") != "universal_anatomy_tactile_ae_v3":
            raise ValueError(f"Unexpected tactile codec schema: {schema}")
        if schema.get("latent_frame") != "[B,12,256]" or schema.get("time_compression") is not False:
            raise ValueError("ME-X-1.0 requires the non-temporal [B,12,256] V3 codec")
        config = AnatomyTactileAEV3Config(**schema["config"])
        model = AnatomyEncoderV3(config)
        weights = state.get("ema", {}).get("shadow")
        if weights is None:
            raise ValueError("V3 checkpoint does not contain EMA weights")
        encoder_weights = {
            key.removeprefix("encoder."): value
            for key, value in weights.items()
            if key.startswith("encoder.")
        }
        incompatible = model.load_state_dict(encoder_weights, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict V3 load failed: {incompatible}")
        model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.model = model
        self.normalizer = DomainRobustNormalizer.from_state_dict(state["normalizer"])
        self.device = device
        self.dtype = dtype

    def assert_frozen(self) -> None:
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("V3 tactile encoder must remain eval-only and frozen")

    @staticmethod
    def _metadata(batch: int, time: int, device: torch.device) -> dict[str, torch.Tensor]:
        region_mask = torch.zeros(batch, time, 30, dtype=torch.bool, device=device)
        region_mask[:, :, :4] = True
        grid_mask = torch.zeros(batch, time, 30, 10, 14, dtype=torch.bool, device=device)
        grid_mask[:, :, :4] = True
        u = torch.linspace(-1.0, 1.0, 14, device=device)
        v = torch.linspace(-1.0, 1.0, 10, device=device)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        uv = torch.stack((uu, vv), dim=0)
        uv_coordinates = torch.zeros(batch, time, 30, 2, 10, 14, device=device)
        uv_coordinates[:, :, :4] = uv
        hand = torch.zeros(batch, time, 30, dtype=torch.long, device=device)
        finger = torch.zeros_like(hand)
        segment = torch.zeros_like(hand)
        hand[:, :, :4] = hand.new_tensor((1, 1, 2, 2))
        finger[:, :, :4] = finger.new_tensor((1, 2, 1, 2))
        segment[:, :, :4] = 1
        return {
            "region_mask": region_mask,
            "grid_mask": grid_mask,
            "uv_coordinates": uv_coordinates,
            "hand_side_id": hand,
            "finger_id": finger,
            "segment_id": segment,
        }

    @torch.no_grad()
    def encode_raw(
        self,
        observed_source: torch.Tensor,
        future_source: torch.Tensor,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Dataset layout is [B,R,T,3,10,14]; V3 layout is [B,T,R,3,10,14].
        force = torch.cat((observed_source, future_source), dim=2).permute(0, 2, 1, 3, 4, 5)
        batch, time, regions = force.shape[:3]
        if (time, regions) != (18, 4):
            raise ValueError(f"Expected RoboTwin tactile [B,4,18,3,10,14], got {tuple(force.shape)}")
        metadata = self._metadata(batch, time, force.device)
        canonical = torch.zeros(batch, time, 30, 3, 10, 14, device=force.device, dtype=force.dtype)
        canonical[:, :, :4] = force
        valid = metadata["region_mask"][..., None, None, None] & metadata["grid_mask"][..., None, :, :]
        group_ids = torch.ones(batch, dtype=torch.long, device=force.device)
        normalized = self.normalizer.normalize(canonical.float(), group_ids, valid)
        flat_metadata = {
            key: value.flatten(0, 1) for key, value in metadata.items()
        }
        encoded = self.model(
            normalized.to(dtype=self.dtype).flatten(0, 1), **flat_metadata
        )
        tokens = encoded["frame_tokens"].reshape(batch, time, 12, -1)
        token_valid = encoded["anatomy_token_valid"].reshape(batch, time, 12)
        # Keep the fixed 12-slot anatomy interface. token_valid is returned for
        # auditing; missing anatomical slots are not mistaken for sensor zeros.
        return tokens, token_valid

    @torch.no_grad()
    def encode_condition(self, observed_source: torch.Tensor, **_: torch.Tensor):
        if observed_source.ndim != 6 or observed_source.shape[2] != 2:
            raise ValueError("Observed RoboTwin tactile must be [B,4,2,3,10,14]")
        future = observed_source.new_zeros(
            observed_source.shape[0], 4, 16, 3, 10, 14
        )
        tokens, valid = self.encode_raw(observed_source, future)
        return tokens[:, :2], valid[:, :2]
