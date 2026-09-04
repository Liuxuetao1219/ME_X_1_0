#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HOSTFILE:-}" ]]; then
  echo "Set HOSTFILE to a DeepSpeed hostfile containing 32 GPUs." >&2
  exit 1
fi

deepspeed --hostfile "${HOSTFILE}" training/train.py \
  --config "${CONFIG:-configs/train_clean50.yaml}"
