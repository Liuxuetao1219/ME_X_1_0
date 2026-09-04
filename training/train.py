"""Distributed Clean50 training entry point for MachEmbodied-Dex1.0."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import deepspeed
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

from models.mach_embodied_dex import MachEmbodiedDexConfig, MachEmbodiedDexModel  # noqa: E402
from training.data import Clean50Dataset, collate, worker_init  # noqa: E402


LOG = logging.getLogger("mach_embodied_dex.train")


class LinearSchedule:
    """Linear warmup and decay used by the released 50k run."""

    def __init__(self, optimizer, warmup: int, total: int) -> None:
        self.optimizer = optimizer
        self.warmup = warmup
        self.total = total
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup:
            factor = 1e-6 + (0.99 - 1e-6) * self.step_count / self.warmup
        elif self.step_count < self.total:
            factor = 0.1 + 0.89 * (self.total - self.step_count) / (self.total - self.warmup)
        else:
            factor = 0.1
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = int(state["step_count"])


def model_from_config(config: dict, local_rank: int) -> MachEmbodiedDexModel:
    torch.cuda.set_device(local_rank)
    model_cfg = config["model"]
    paths = config["paths"]
    return MachEmbodiedDexModel(
        MachEmbodiedDexConfig(
            vae_path=str(Path(paths["wan"]) / "Wan2.2_VAE.pth"),
            wan_config_path=paths["wan"],
            video_precision="bfloat16",
            batch_size=int(config["training"]["batch_size"]),
            tactile_vae_checkpoint_path=paths["tactile_ae"],
            tactile_expert_config=model_cfg["tactile_expert"],
            video_loss_weight=float(model_cfg["loss_weights"]["video"]),
            action_loss_weight=float(model_cfg["loss_weights"]["action"]),
            tactile_loss_weight=float(model_cfg["loss_weights"]["tactile"]),
            video_train_schedule_shift=float(model_cfg["schedule_shift"]),
            action_train_schedule_shift=float(model_cfg["schedule_shift"]),
            tactile_train_schedule_shift=float(model_cfg["schedule_shift"]),
        )
    )


def load_initial_weights(model: MachEmbodiedDexModel, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    source = checkpoint.get("module", checkpoint) if isinstance(checkpoint, Mapping) else None
    if not isinstance(source, Mapping):
        raise TypeError("Initial checkpoint must contain a model state dictionary")
    prefixes = (
        "video_model.wan_model.",
        "action_expert.time_embedding.",
        "action_expert.time_projection.",
        "action_expert.blocks.",
    )
    target = model.state_dict()
    selected = {
        key: value.to(dtype=target[key].dtype)
        for key, value in source.items()
        if key.startswith(prefixes) and key in target
    }
    required = {key for key in target if key.startswith(prefixes)}
    missing = required - set(selected)
    if missing:
        raise RuntimeError(f"Initial checkpoint is missing {len(missing)} required tensors")
    model.load_state_dict(selected, strict=False)
    LOG.info("Loaded %d WAN and Action DiT tensors", len(selected))


def build_optimizer(model: MachEmbodiedDexModel, config: dict) -> torch.optim.Optimizer:
    training = config["training"]
    wan = [parameter for parameter in model.video_model.wan_model.parameters() if parameter.requires_grad]
    tactile = [parameter for parameter in model.tactile_expert.parameters() if parameter.requires_grad]
    excluded = {id(parameter) for parameter in (*wan, *tactile)}
    main = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    return torch.optim.AdamW(
        [
            {"name": "main", "params": main, "lr": float(training["learning_rate"])},
            {"name": "wan", "params": wan, "lr": float(training["wan_learning_rate"])},
            {"name": "tactile", "params": tactile, "lr": float(training["tactile_learning_rate"])},
        ],
        betas=(0.9, 0.95),
        weight_decay=float(training["weight_decay"]),
        foreach=False,
    )


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in batch.items():
        dtype = torch.float32 if key.startswith("tactile_") else torch.bfloat16
        output[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_clean50.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    expected_world_size = int(config["training"]["world_size"])
    if world_size != expected_world_size:
        raise ValueError(f"Expected {expected_world_size} training processes, got {world_size}")
    torch.manual_seed(int(config["training"]["seed"]) + rank)

    model = model_from_config(config, args.local_rank)
    if args.resume is None:
        load_initial_weights(model, config["paths"]["initial_checkpoint"])
    optimizer = build_optimizer(model, config)
    schedule = LinearSchedule(
        optimizer,
        warmup=int(config["training"]["warmup_steps"]),
        total=int(config["training"]["max_steps"]),
    )
    engine, optimizer, _, schedule = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=schedule,
        config=config["paths"]["deepspeed_config"],
    )
    start_step = 0
    if args.resume:
        _, client_state = engine.load_checkpoint(args.resume)
        start_step = int(client_state["step"])

    dataset = Clean50Dataset(
        config["data"]["root"],
        config["data"]["text_cache"],
        config["data"]["quality_manifest"],
        samples_per_episode=int(config["data"]["samples_per_episode"]),
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        worker_init_fn=worker_init,
        persistent_workers=True,
        prefetch_factor=4,
    )

    output = Path(config["paths"]["output"])
    max_steps = int(config["training"]["max_steps"])
    save_every = int(config["training"]["save_every"])
    step = start_step
    epoch = 0
    while step < max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            losses = engine(**to_device(batch, engine.device))
            engine.backward(losses["loss"])
            engine.step()
            step += 1
            if rank == 0 and step % 10 == 0:
                LOG.info(
                    "step=%d loss=%.5f video=%.5f action=%.5f tactile=%.5f lr=%.3e",
                    step,
                    losses["loss"].item(),
                    losses["video_loss"].item(),
                    losses["action_loss"].item(),
                    losses["tactile_loss"].item(),
                    schedule.get_last_lr()[0],
                )
            if step % save_every == 0 or step == max_steps:
                engine.save_checkpoint(str(output), tag=f"step_{step}", client_state={"step": step})
            if step == max_steps:
                break
        epoch += 1


if __name__ == "__main__":
    main()
