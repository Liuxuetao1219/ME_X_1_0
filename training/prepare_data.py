"""Build the Clean50 training-anchor manifest without modifying source data."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np


ACTION_OFFSETS = np.arange(3, 49, 3, dtype=np.int64)
FORCE_SCALE = np.asarray((12.3728247, 1.1703857, 0.4424788), dtype=np.float32)


def select_anchors(path: Path) -> list[int]:
    with h5py.File(path, "r") as handle:
        qpos = np.asarray(handle["joint_action/vector"], dtype=np.float32)
        force = np.asarray(handle["tactile_force_field/force_canonical"], dtype=np.float32)
    force = force / FORCE_SCALE[None, None, :, None, None]
    valid: list[int] = []
    dynamic: list[int] = []
    low: set[int] = set()
    maximum = len(qpos) - ACTION_OFFSETS[-1] - 1
    for anchor in range(1, maximum + 1):
        indices = anchor + ACTION_OFFSETS
        action = np.concatenate((qpos[anchor][None], qpos[indices]))
        tactile = force[np.concatenate(([anchor - 1, anchor], indices))]
        tactile_jump = np.linalg.norm(np.diff(tactile, axis=0).reshape(17, -1), axis=1).mean()
        action_delta = np.linalg.norm(np.diff(action, axis=0), axis=1)
        is_valid = (
            np.isfinite(tactile).all()
            and np.abs(tactile).max() <= 256
            and tactile_jump <= 100
            and np.isfinite(action).all()
            and np.abs(action).max() <= 20
            and action_delta.max() <= 3
        )
        if not is_valid:
            continue
        occupancy = (np.linalg.norm(tactile, axis=2) > 0.05).mean()
        valid.append(anchor)
        if action_delta.mean() < 0.05 and tactile_jump < 0.01 and occupancy <= 0.001:
            low.add(anchor)
        else:
            dynamic.append(anchor)
    dynamic_array = np.asarray(dynamic)
    selected: list[int] = []
    for anchor in valid:
        if anchor not in low:
            selected.append(anchor)
        elif dynamic_array.size:
            position = np.searchsorted(dynamic_array, anchor, side="right")
            selected.append(int(dynamic_array[position] if position < len(dynamic_array) else dynamic_array[-1]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    episodes = {}
    for path in sorted(root.glob("*/tactile_replay_aloha_clean50_batch/data/episode*.hdf5")):
        episodes[path.relative_to(root).as_posix()] = {"training_anchors": select_anchors(path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt", encoding="utf-8") as stream:
        json.dump({"format": "ME-X-1.0-Clean50", "episodes": episodes}, stream)


if __name__ == "__main__":
    main()
