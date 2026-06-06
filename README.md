# seedvr2-mlx-swift

Standalone **MLX-Swift** port of **SeedVR2** (ByteDance, ICLR 2026) — one-step
diffusion-based super-resolution — for **MLXEngine** and **ForgeUpscaler** (Export tier,
peer to `OSEDiff_MLX`, alongside the already-ported `EfRLFN`). No runtime dependency on
Python/mflux.

- **Reference (oracle):** [`filipstrand/mflux`](https://github.com/filipstrand/mflux) `src/mflux/models/seedvr2/` (MLX-Python). This port is an MLX-Python → MLX-Swift translation — see `docs/PORT-PLAN.md`.
- **Model/weights license:** Apache-2.0 (ByteDance Seed). Port code: MIT.
- **Weights:** [mlx-community/SeedVR2-3B-mlx](https://huggingface.co/mlx-community/SeedVR2-3B-mlx) (fp16) · [SeedVR2-3B-mlx-int8](https://huggingface.co/mlx-community/SeedVR2-3B-mlx-int8) (near-lossless, ~4 GB on-device) · [collection](https://huggingface.co/collections/mlx-community/seedvr2-mlx-swift-6a23c38505955a29500123b4).
- **Status:** ✅ full inference path ported + parity-verified vs mflux (DiT `t_out` 2.1e-4, VAE 3.5e-3/7.2e-3, RNG/scheduler 0.0, int8 cosine 0.99997) and weights published. Remaining = host preprocess/color-correct + ForgeUpscaler integration (tiling, Export-tier conformer). See `docs/PORT-PLAN.md`.

## Build & test

```bash
# Tests MUST run via xcodebuild (the SPM CLI can't bundle the Metal default.metallib).
xcodebuild test -scheme SeedVR2MLX-Package -destination 'platform=macOS'
swift build         # library + CLI compile fine from the CLI
```

## Weights

Exported from the mflux oracle by `seedvr2-mlx/scripts/prepare_swift.py`:
`transformer.safetensors + vae.safetensors + pos_emb.safetensors + config.json`
(fp16; reload verified bit-exact vs mflux). To be published as `mlx-community/SeedVR2-3B-mlx`
(bf16 + int8 — int8 is near-lossless and the on-device target; int4 degrades this model).

## Layout

```
Sources/SeedVR2MLX/
  Config.swift                  # 3B/7B dims + transformer_overrides
  Models/Transformer/           # RMSNorm ✅, RoPE, SwiGLU, AdaModulation, Attention, PatchIn/Out, ...
  Models/VAE/                   # CausalConv3d, Encoder, Decoder (3D-causal-conv VAE)
  Pipeline/                     # Preprocess, LatentCreator, Scheduler, Upscaler
  Utilities/WeightLoader.swift  # ✅ load exported safetensors + config
Sources/RunUpscale/             # CLI
Tests/SeedVR2MLXTests/          # CPU parity vs goldens_swift_*.npz (xcodebuild)
docs/PORT-PLAN.md               # full module work-list + parity gates
```
