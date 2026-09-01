# ME-X-1.0

ME-X-1.0 is an eval-only policy runtime for RoboTwin Clean50-to-Random leaderboard evaluation.

Model weights are available at [liuxuetao/ME-X-1.0-RoboTwin-Clean2Random-Leaderboard](https://huggingface.co/liuxuetao/ME-X-1.0-RoboTwin-Clean2Random-Leaderboard). The runtime also uses the VAE, T5 encoder, tokenizer, and config from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

XPolicyLab supplies RGB observations. Set `input_color_order: bgr` for this checkpoint so the runtime performs the single RGB-to-BGR conversion expected by training. The runtime does not apply mean/std image normalization.

Clean50 training included an additionally collected three-axis tactile force array. RoboTwin leaderboard evaluation has no tactile observation, so the two observed tactile frames are set to zero.

Training support: eval-only (training release ETA: TBD).

## Usage

ME-X-1.0 is evaluated through the XPolicyLab `ME_X_1_0` adapter. Installation,
checkpoint preparation, and RoboTwin evaluation commands are documented in the
adapter README.

## Third-party code

The minimal WAN runtime under `runtime/wan/` is derived from [Wan2.2](https://github.com/Wan-Video/Wan2.2) and retains its original copyright headers.

## License

Apache-2.0.
