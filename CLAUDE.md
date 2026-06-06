# seedvr2-mlx-swift — agent notes

MLX-Swift port of SeedVR2 (one-step diffusion SR) for MLXEngine / ForgeUpscaler Export tier.

## Hard rules
- **Oracle = mflux** (`filipstrand/mflux` `src/mflux/models/seedvr2/`, MLX-Python). This is an
  MLX-Python → MLX-Swift translation: keep module structure **isomorphic** (same file/class
  names). Numerics/QKV/RoPE/weight-layout decisions are already made there — don't redesign.
- **Tests run via `xcodebuild test -scheme SeedVR2MLX-Package -destination 'platform=macOS'`**,
  NOT `swift test` (the SPM CLI can't bundle `default.metallib` → "Failed to load the default
  metallib"). `swift build` is fine for compile checks.
- **Parity on CPU** (`Device.setDefault(device: Device(.cpu))` in test setUp) — Apple-GPU fp32
  is tf32-like. Gate stage modules `< 1e-2` (fp16) vs the goldens.
- **Constants are not parameters:** sinusoidal embeddings, RoPE freqs, causal masks have no
  checkpoint entry — compute on the fly, don't store as tracked `MLXArray` properties (the
  load would mismatch). See mlx-swift skill porting.md §2.
- Conv weights in the exported safetensors are already MLX layout `(O, *K, I)` — load directly.

## Artifacts (from the Python side, `../seedvr2-mlx/`)
- Weights: `../seedvr2-mlx/dist/SeedVR2-3B-mlx/` (transformer/vae/pos_emb/config).
- Goldens: `../seedvr2-mlx/tests/goldens_swift_seedvr2-3b.npz` (CPU, seed 42, 2x). Copy the
  needed slices into `Tests/.../Resources/` as the modules land. Regenerate via
  `seedvr2-mlx/scripts/prepare_swift.py`.

## Status / next
**Model COMPLETE + published** (`mlx-community/SeedVR2-3B-mlx{,-int8}`). Full inference path
parity-verified vs mflux (DiT 2.1e-4, VAE 3.5e-3/7.2e-3, RNG/sched 0.0, int8 cosine 0.99997).
Remaining is **Forge integration only** → **`docs/FORGE-HANDOFF.md`** (ExportTier conformer
`SeedVR2_MLX` replacing the `OSEDiff_MLX` stub, host preprocess + LAB color-correct, VAE tiling,
Swift HF auto-download). Port detail + parity table: `docs/PORT-PLAN.md`.
