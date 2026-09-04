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

The released run used Python 3.10, BF16, and 32 GPUs. Install the dependencies:

```bash
pip install -r runtime/requirements.txt
pip install -r training/requirements.txt
```

Download the Clean50 tactile dataset:

```bash
hf download liuxuetao/ME-X-1.0-RoboTwin-Clean50-Tactile \
  --repo-type dataset \
  --local-dir datasets/ME-X-1.0-RoboTwin-Clean50-Tactile
```

Download the model initialization and WAN assets:

```bash
hf download liuxuetao/ME-X-1.0-RoboTwin-Clean2Random-Leaderboard \
  tactile_ae.pt \
  --local-dir checkpoints/ME-X-1.0-RoboTwin-Clean2Random-Leaderboard

hf download motus-robotics/Motus \
  mp_rank_00_model_states.pt \
  --local-dir checkpoints/Motus

hf download Wan-AI/Wan2.2-TI2V-5B \
  config.json \
  Wan2.2_VAE.pth \
  models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/special_tokens_map.json \
  google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json \
  google/umt5-xxl/tokenizer_config.json \
  --local-dir checkpoints/Wan2.2-TI2V-5B
```

Clean50 uses 2,500 HDF5 episodes from 50 tasks. Camera arrays are read in BGR order without mean/std normalization. The action targets are raw joint positions at `t+3, t+6, ..., t+48`; video targets are at `t+6, t+12, ..., t+48`.

The dataset includes `quality_manifest.json.gz`. Build the cached T5 embeddings:

```bash
python training/cache_text.py \
  --data-root datasets/ME-X-1.0-RoboTwin-Clean50-Tactile \
  --wan checkpoints/Wan2.2-TI2V-5B \
  --output datasets/ME-X-1.0-RoboTwin-Clean50-Tactile/t5_prompt_table.pt
```

The default paths in `configs/train_clean50.yaml` match the commands above. Create a DeepSpeed hostfile for two 16-GPU nodes and run:

```bash
printf 'node0 slots=16\nnode1 slots=16\n' > hostfile
HOSTFILE=hostfile bash train.sh
```

Replace `node0` and `node1` with resolvable hostnames. The per-GPU batch size is 4, the global batch size is 128, and training runs for 50,000 optimizer steps.

## Third-party code

The minimal WAN runtime under `runtime/wan/` is derived from [Wan2.2](https://github.com/Wan-Video/Wan2.2) and retains its original copyright headers.

## License

Apache-2.0.
