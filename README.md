# ME-X-1.0

ME-X-1.0 is a video-action-tactile policy trained on RoboTwin Clean50.

Model weights are available at [liuxuetao/ME-X-1.0-RoboTwin-Clean2Random-Leaderboard](https://huggingface.co/liuxuetao/ME-X-1.0-RoboTwin-Clean2Random-Leaderboard). The runtime also uses the VAE, T5 encoder, tokenizer, and config from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

XPolicyLab supplies RGB observations. Set `input_color_order: bgr` for this checkpoint so the runtime performs the single RGB-to-BGR conversion expected by training. The runtime does not apply mean/std image normalization.

Clean50 training included an additionally collected three-axis tactile force array. RoboTwin leaderboard evaluation has no tactile observation, so the two observed tactile frames are set to zero.

## Evaluation

ME-X-1.0 is evaluated through the XPolicyLab `ME_X_1_0` adapter. Installation,
checkpoint preparation, and RoboTwin evaluation commands are documented in the
adapter README.

## Training

Install the runtime and training dependencies:

```bash
pip install -r runtime/requirements.txt
pip install -r training/requirements.txt
```

Download [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), the [Motus Stage 2 checkpoint](https://huggingface.co/motus-robotics/Motus), and `tactile_ae.pt` from the ME-X-1.0 checkpoint repository.

Clean50 uses 2,500 HDF5 episodes from 50 tasks. Camera arrays are read in BGR order without mean/std normalization. The action targets are raw joint positions at `t+3, t+6, ..., t+48`; video targets are at `t+6, t+12, ..., t+48`.

Build the quality manifest and cached T5 embeddings:

```bash
python training/prepare_data.py \
  --data-root /path/to/ME-X-1.0-RoboTwin-Clean50 \
  --output /path/to/quality_manifest.json.gz

python training/cache_text.py \
  --data-root /path/to/ME-X-1.0-RoboTwin-Clean50 \
  --wan /path/to/Wan2.2-TI2V-5B \
  --output /path/to/t5_prompt_table.pt
```

Set the paths in `configs/train_clean50.yaml`, prepare a DeepSpeed hostfile for 32 GPUs, and run:

```bash
HOSTFILE=/path/to/hostfile bash train.sh
```

The released run used BF16, a per-GPU batch size of 4, a global batch size of 128, and 50,000 optimizer steps.

## Third-party code

The minimal WAN runtime under `runtime/wan/` is derived from [Wan2.2](https://github.com/Wan-Video/Wan2.2) and retains its original copyright headers.

## License

Apache-2.0.
