"""Clean50 RGB, joint-action, tactile, and cached-text dataset."""

from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


CAMERAS = ("head_camera", "left_camera", "right_camera")
ACTION_OFFSETS = tuple(range(3, 49, 3))
VIDEO_OFFSETS = tuple(range(6, 49, 6))


@dataclass(frozen=True)
class Episode:
    task: str
    number: int
    data: Path
    instructions: Path


class TextEmbeddingCache:
    """Lazy reader for the disk-backed BF16 T5 cache."""

    def __init__(self, metadata_path: str) -> None:
        self.metadata_path = Path(metadata_path)
        metadata = torch.load(self.metadata_path, map_location="cpu", weights_only=False)
        if metadata["cache_format"] != "variable_length_per_prompt_disk_v2":
            raise ValueError("Unsupported text cache format")
        self.prompts = tuple(metadata["prompts"])
        self.index = tuple(tuple(map(int, item)) for item in metadata["embedding_index"])
        self.prompt_to_index = {prompt: index for index, prompt in enumerate(self.prompts)}
        self.payload_path = self.metadata_path.with_suffix(".bin")
        self._payload: np.memmap | None = None

    def get(self, prompt: str) -> torch.Tensor:
        if self._payload is None:
            self._payload = np.memmap(self.payload_path, dtype=np.uint16, mode="c")
        offset, length, width = self.index[self.prompt_to_index[prompt]]
        values = np.asarray(self._payload[offset : offset + length * width])
        return torch.from_numpy(values.reshape(length, width)).view(torch.bfloat16)


class Clean50Dataset(Dataset):
    """Samples the exact c+3..c+48 ME-X-1.0 temporal contract."""

    def __init__(
        self,
        root: str,
        text_cache: str,
        quality_manifest: str,
        *,
        samples_per_episode: int = 10,
        image_size: tuple[int, int] = (384, 320),
    ) -> None:
        self.root = Path(root).resolve()
        self.image_size = image_size
        self.text_cache = TextEmbeddingCache(text_cache)
        opener = gzip.open if quality_manifest.endswith(".gz") else open
        with opener(quality_manifest, "rt", encoding="utf-8") as stream:
            self.quality = json.load(stream)["episodes"]

        episodes: list[Episode] = []
        for task_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            base = task_dir / "tactile_replay_aloha_clean50_batch"
            for path in sorted((base / "data").glob("episode*.hdf5"), key=self._number):
                number = self._number(path)
                episodes.append(
                    Episode(
                        task=task_dir.name,
                        number=number,
                        data=path,
                        instructions=base / "instructions" / f"episode{number}.json",
                    )
                )
        if len(episodes) != 2500:
            raise ValueError(f"Clean50 requires 2,500 episodes, found {len(episodes)}")
        self.episodes = tuple(episodes)
        self.length = len(episodes) * samples_per_episode

    @staticmethod
    def _number(path: Path) -> int:
        match = re.fullmatch(r"episode(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Invalid episode name: {path.name}")
        return int(match.group(1))

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _decode_bgr(dataset: h5py.Dataset, index: int) -> np.ndarray:
        value = dataset[index]
        if not isinstance(value, (bytes, np.bytes_)):
            raise ValueError("Clean50 camera frames must be JPEG bytes")
        frame = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid Clean50 camera frame")
        # Collection encoded RGB buffers with OpenCV's BGR convention. The
        # historical loader then applied this conversion, producing the BGR
        # tensors used by the released checkpoint.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _resize_with_padding(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        target_h, target_w = size
        height, width = frame.shape[:2]
        scale = min(target_h / height, target_w / width)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y = (target_h - resized.shape[0]) // 2
        x = (target_w - resized.shape[1]) // 2
        output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return output

    def _frame(self, handle: h5py.File, index: int) -> torch.Tensor:
        head, left, right = [
            self._decode_bgr(handle["observation"][camera]["rgb"], index)
            for camera in CAMERAS
        ]
        height, width = head.shape[:2]
        left = cv2.resize(left, (width // 2, height // 2))
        right = cv2.resize(right, (width // 2, height // 2))
        frame = np.concatenate((head, np.concatenate((left, right), axis=1)), axis=0)
        frame = self._resize_with_padding(frame, self.image_size)
        return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255

    def _anchors(self, episode: Episode) -> list[int]:
        relative = episode.data.relative_to(self.root).as_posix()
        record = self.quality.get(relative) or self.quality.get(str(episode.data))
        if record is None:
            raise KeyError(f"Episode missing from quality manifest: {relative}")
        return record["training_anchors"]

    def __getitem__(self, _: int) -> dict[str, Any]:
        episode = random.choice(self.episodes)
        anchor = random.choice(self._anchors(episode))
        action_indices = [anchor + offset for offset in ACTION_OFFSETS]
        video_indices = [anchor + offset for offset in VIDEO_OFFSETS]

        with h5py.File(episode.data, "r") as handle:
            qpos = handle["joint_action/vector"]
            state = np.asarray(qpos[anchor], dtype=np.float32)
            actions = np.stack([np.asarray(qpos[index], np.float32) for index in action_indices])
            first_frame = self._frame(handle, anchor)
            video_frames = torch.stack([self._frame(handle, index) for index in video_indices])
            force = handle["tactile_force_field/force_canonical"]
            observed = np.stack([np.asarray(force[index], np.float32) for index in (anchor - 1, anchor)])
            future = np.stack([np.asarray(force[index], np.float32) for index in action_indices])
            intervals = np.asarray(
                handle["tactile_force_field/interval_seconds"][: action_indices[-1] + 1],
                dtype=np.float64,
            )

        absolute_time = np.cumsum(intervals)
        times = absolute_time[np.asarray([anchor - 1, anchor, *action_indices])]
        times = (times - times[0]).astype(np.float32)
        instructions = json.loads(episode.instructions.read_text(encoding="utf-8"))["seen"]
        available = [text for text in instructions if text in self.text_cache.prompt_to_index]
        if not available:
            raise KeyError(f"No cached instruction for {episode.instructions}")
        return {
            "first_frame": first_frame,
            "video_frames": video_frames,
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "language_embeddings": self.text_cache.get(random.choice(available)),
            "tactile_observed_source": torch.from_numpy(np.moveaxis(observed, 1, 0).copy()),
            "tactile_future_source": torch.from_numpy(np.moveaxis(future, 1, 0).copy()),
            "tactile_observed_frame_times": torch.from_numpy(times[:2].copy()),
            "tactile_future_query_times": torch.from_numpy(times[2:].copy()),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    output = {
        key: torch.stack([sample[key] for sample in batch])
        for key in batch[0]
        if key != "language_embeddings"
    }
    embeddings = [sample["language_embeddings"] for sample in batch]
    padded = embeddings[0].new_zeros((len(batch), 512, embeddings[0].shape[-1]))
    for index, embedding in enumerate(embeddings):
        length = min(embedding.shape[0], 512)
        padded[index, :length] = embedding[:length]
    output["language_embeddings"] = padded
    return output


def worker_init(_: int) -> None:
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
