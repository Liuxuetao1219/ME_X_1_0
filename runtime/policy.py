from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _install_numpy_pickle_compat() -> None:
    """Load trusted NumPy-2 checkpoints in the reviewed NumPy-1 runtime."""
    if hasattr(np, "_core"):
        return
    sys.modules.setdefault("numpy._core", np.core)
    for child in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        try:
            module = __import__(f"numpy.core.{child}", fromlist=(child,))
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core.{child}", module)


_install_numpy_pickle_compat()

from checkpoint import load_metadata, load_model_state_strictly
from models.me_x import MEXConfig, MEXModel
from utils.image_utils import resize_with_padding
from wan.modules.t5 import T5EncoderModel


logger = logging.getLogger(__name__)


class MEXPolicy:
    """Single-version ME-X-1.0 inference runtime.

    The runtime accepts RGB camera arrays, supports an explicit BGR checkpoint
    compatibility mode, and supplies two physical-zero tactile frames because
    leaderboard observations contain no tactile sensors.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        wan_path: str,
        tactile_frame_interval_seconds: float = 0.06,
        input_color_order: str = "rgb",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("ME-X-1.0 requires CUDA")
        self.device = torch.device("cuda")
        self.wan_path = Path(wan_path).expanduser().resolve()
        self.metadata = load_metadata(checkpoint_path)
        self.config = self.metadata["config"]
        self.tactile_frame_interval_seconds = float(tactile_frame_interval_seconds)
        if not np.isfinite(self.tactile_frame_interval_seconds) or self.tactile_frame_interval_seconds <= 0:
            raise ValueError("tactile_frame_interval_seconds must be finite and positive")
        self.input_color_order = str(input_color_order).lower()
        if self.input_color_order not in {"rgb", "bgr"}:
            raise ValueError("input_color_order must be 'rgb' or 'bgr'")
        self.model = self._build_model()
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device="cuda",
            checkpoint_path=str(self.wan_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(self.wan_path / "google" / "umt5-xxl"),
        )
        self.current_frame: torch.Tensor | None = None
        self.current_state: torch.Tensor | None = None
        self.current_instruction: str | None = None

    def _build_model(self) -> MEXModel:
        cfg = self.config
        common = cfg["common"]
        action = cfg["action_expert"]
        tactile = cfg["model"]["tactile"]
        wan = cfg["model"]["wan"]
        model = MEXModel(
            MEXConfig(
                vae_path=str(self.wan_path / "Wan2.2_VAE.pth"),
                wan_config_path=str(self.wan_path),
                video_precision=str(wan.get("precision", "bfloat16")),
                num_layers=30,
                action_state_dim=int(common["state_dim"]),
                action_dim=int(common["action_dim"]),
                action_expert_dim=int(action["hidden_size"]),
                action_expert_ffn_dim_multiplier=int(action["ffn_dim_multiplier"]),
                action_expert_norm_eps=float(action["norm_eps"]),
                action_chunk_size=int(common["action_chunk_size"]),
                num_video_frames=int(common["num_video_frames"]),
                video_height=int(common["video_height"]),
                video_width=int(common["video_width"]),
                batch_size=1,
                tactile_vae_checkpoint_path=str(self.metadata["tactile_checkpoint"]),
                tactile_expert_config=cfg["model"]["tactile_expert"],
            )
        )
        report = load_model_state_strictly(model, self.metadata)
        model.eval()
        model.tactile_codec.assert_frozen()
        logger.info("Loaded ME-X-1.0 strictly: %s", report)
        return model

    def update_observation(
        self,
        *,
        head_rgb: np.ndarray,
        left_wrist_rgb: np.ndarray,
        right_wrist_rgb: np.ndarray,
        qpos: np.ndarray,
        instruction: str,
    ) -> None:
        # The ME-X-1.0 training/deployment camera mosaic uses a 320x240 head
        # frame above two 160x120 wrist frames.  XPolicyLab observations may
        # expose the native 640x480 head stream, so restore the trained camera
        # contract before assembling the mosaic.
        head = cv2.resize(np.asarray(head_rgb), (320, 240))
        left = cv2.resize(np.asarray(left_wrist_rgb), (160, 120))
        right = cv2.resize(np.asarray(right_wrist_rgb), (160, 120))
        image_rgb = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
        if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB camera image, got {image_rgb.shape}")
        image = image_rgb
        if self.input_color_order == "bgr":
            # XPolicyLab observations are RGB. This explicit opt-in preserves
            # compatibility with checkpoints trained on BGR camera arrays.
            image = image_rgb[..., [2, 1, 0]]
        target = (int(self.config["common"]["video_height"]), int(self.config["common"]["video_width"]))
        if image.shape[:2] != target:
            image = resize_with_padding(image, target)
        image = image.astype(np.float32)
        if image.max(initial=0.0) > 1.0:
            image /= 255.0
        if not np.isfinite(image).all() or image.min(initial=0.0) < 0 or image.max(initial=0.0) > 1:
            raise ValueError("Image must be finite and in [0,1]")
        state = np.asarray(qpos, dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite qpos [14], got {state.shape}")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Instruction must be non-empty")
        self.current_frame = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.current_state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.current_instruction = instruction.strip()

    def _language_embeddings(self) -> list[torch.Tensor]:
        prompt = f"{self.config['prompt']['prefix']}{self.current_instruction}"
        encoded = self.t5_encoder([prompt], self.device)
        if isinstance(encoded, torch.Tensor):
            return [encoded.squeeze(0)] if encoded.ndim == 3 else [encoded]
        if isinstance(encoded, list) and encoded:
            return encoded
        raise TypeError(f"Unexpected T5 output: {type(encoded)}")

    def _zero_tactile(self) -> dict[str, torch.Tensor | float]:
        cadence = self.tactile_frame_interval_seconds
        dtype = self.model.dtype
        return {
            "tactile_observed_source": torch.zeros(
                (1, 4, 2, 3, 10, 14), device=self.device, dtype=torch.float32
            ),
            "tactile_observed_frame_times": torch.tensor(
                [[0.0, cadence]], device=self.device, dtype=dtype
            ),
            "tactile_future_query_times": cadence
            * torch.arange(4, 50, 3, device=self.device, dtype=dtype).unsqueeze(0),
            "tactile_schedule_shift": float(self.config["inference"]["tactile_schedule_shift"]),
        }

    def get_action(self) -> np.ndarray:
        if self.current_frame is None or self.current_state is None or self.current_instruction is None:
            raise RuntimeError("Call update_observation before get_action")
        with torch.inference_mode():
            _, actions = self.model.inference_step(
                first_frame=self.current_frame,
                state=self.current_state,
                num_inference_steps=int(self.config["inference"]["num_inference_timesteps"]),
                language_embeddings=self._language_embeddings(),
                video_schedule_shift=float(self.config["inference"]["video_schedule_shift"]),
                action_schedule_shift=float(self.config["inference"]["action_schedule_shift"]),
                **self._zero_tactile(),
            )
        if tuple(actions.shape) != (1, 16, 14) or not torch.isfinite(actions).all().item():
            raise RuntimeError(f"Invalid ME-X-1.0 action tensor: {tuple(actions.shape)}")
        return actions[0].float().cpu().numpy()

    def reset(self) -> None:
        self.current_frame = None
        self.current_state = None
        self.current_instruction = None
