"""Image preprocessing used by MachEmbodied-Dex1.0 inference."""

from typing import Tuple

import cv2
import numpy as np


def resize_with_padding(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize an HWC image without changing its aspect ratio."""
    target_height, target_width = target_size
    original_height, original_width = frame.shape[:2]
    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    resized = cv2.resize(frame, (new_width, new_height))
    padded = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    padded[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return padded
