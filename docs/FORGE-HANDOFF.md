# SeedVR2 → ForgeUpscaler — Integration Handoff

**Status (2026-06-05):** the SeedVR2 *model* is **done, parity-verified, and published**. This
doc is the remaining **Forge-side wiring** to surface it as the ForgeUpscaler **Export tier**.
None of the items below are model work — the DiT + 3D-VAE + 1-step diffusion loop are complete.

## What already exists (don't redo)

- **Swift package:** [`xocialize/seedvr2-mlx-swift`](https://github.com/xocialize/seedvr2-mlx-swift) — `import SeedVR2MLX`. Builds + tests via `xcodebuild test -scheme SeedVR2MLX-Package` (NOT `swift test` — SPM CLI can't bundle `default.metallib`).
- **Published weights** (Apache-2.0, MVS provenance):
  - `mlx-community/SeedVR2-3B-mlx` — fp16 (transformer 7.9 GB + VAE fp16 + pos_emb + config)
  - `mlx-community/SeedVR2-3B-mlx-int8` — **on-device target**, ~4.4 GB, near-lossless (cosine 0.99997)
- **Core API** — the entire diffusion path in one call:
  ```swift
  let up = try SeedVR2Upscaler(directory: weightsDir)   // auto-applies int8 from config.json
  let decoded = up.upscale(processedImage: img, seed: 42)   // img: [B,3,H,W] in [-1,1], dims padded to /16
  // decoded: [B,3,1,H*,W*]  (latent-decoded image, pre-crop, pre-color-correct)
  ```
  Internally: `vae.encode → noise+condition (16+17=33 ch) → 1-step transformer → euler step → vae.decode`.
- **Parity vs mflux oracle (CPU):** DiT `t_out` 2.1e-4 · VAE enc/dec 3.5e-3/7.2e-3 · RNG/scheduler 0.0 · decode wiring 6.8e-3 · int8 cosine 0.99997.

## Integration target

`ForgeUpscaler/Sources/ForgeUpscaler/Export/` has an `ExportTier` protocol and a forward-looking
**`OSEDiff_MLX` stub** (one-step diffusion SR placeholder that throws `notYetImplemented`).
**SeedVR2 is the working replacement for that stub** — it *is* one-step diffusion SR, Apache-2.0,
better-licensed than OSEDiff (NC), and the weights ship today. Implement `SeedVR2_MLX: ExportTier`
as a peer to `RealESRGAN_CoreML`, wired into `ExportUpscaler`.

`ExportTier` surface to conform to (see `Export/ExportTier.swift`):
```swift
public protocol ExportTier: Sendable {
    var name: String { get }                 // "seedvr2-mlx"
    var scaleFactor: Int { get }             // 2 or 4
    var inputTileSize: Int { get }
    var inputResolution: (width: Int, height: Int) { get }
    var outputResolution: (width: Int, height: Int) { get }
    var tileOverlap: Int { get }
    func upscale(_ buffer: CVPixelBuffer) async throws -> CVPixelBuffer  // tiles internally
}
```

## Work items

### W1 — `SeedVR2_MLX: ExportTier` conformer (the wrapper)
The glue. `upscale(_ buffer:)` should: `CVPixelBuffer → MLXArray` → **preprocess (W2)** →
**tile (W3)** → per-tile `SeedVR2Upscaler.upscale(processedImage:seed:)` → stitch → **color-correct
(W4)** → `MLXArray → CVPixelBuffer`. Add a `SeedVR2_MLX.swift` next to `OSEDiff_MLX.swift`; register
in `ExportUpscaler`'s tier selection. Run production on the **GPU** stream (the `.cpu` device in the
package tests is for parity only — GPU is ~12 s for a 2× upscale vs minutes on CPU).

### W2 — Host preprocess (port of mflux `seedvr2_util.preprocess_image`)
mflux source: `mflux/models/seedvr2/variants/upscale/seedvr2_util.py::preprocess_image`. Steps:
1. target_res from `ScaleFactor` (factor×min-edge) or absolute; `scale = target_res / min(w,h)`; `true_w/h = round(w/h * scale)` then floored to even.
2. **softness** 0..1 → factor `1 + softness*7`; if >1, bicubic-downscale by `factor` then bicubic-upscale back to `true_w/h` (a controllable pre-blur).
3. **pad to /16:** pad right/bottom with black to make both dims `% 16 == 0`.
4. normalize: `/255 → clip[0,1] → *2-1`; to `CHW`; add batch dim. **Range is `[-1, 1]`.**
Use vImage/CoreImage bicubic (PIL-exact bicubic isn't required — SR is generative, small resampling
differences are fine). `true_h/true_w` are needed later to crop the decoded output back.

### W3 — VAE tiling (use ForgeUpscaler's existing tiler)
The package's `vae.decode` is **non-tiled** (full-resolution). For large images use the host tiler
(`Playback/MLXTileProcessor.swift` / `TileProcessor.swift`). **Important:** mflux's pipeline tiles
*decode* (tile latent ~512 px, overlap); a full non-tiled decode differs from mflux's tiled output by
~20% at tile boundaries (this is tiling-method variance, not a bug — the package decode is the
"correct" full decode). So **match Forge's own tiling conventions** (256/32 per plan §D.2), not
mflux's. Tile in *image* space on the encoder side and/or *latent* space on the decoder side; the
VAE is 8× spatial.

### W4 — LAB-wavelet color correction (optional v1; recommended v2)
mflux `seedvr2_util._lab_color_transfer_exact` (wavelet decomposition 5 levels + RGB↔LAB +
per-channel histogram matching, `luminance_weight=0.8`). Improves color fidelity to the source; it's
**post-process on the final RGB**, doesn't affect structure. Can ship v1 without it, add later (port
to Accelerate/vDSP or a small MLX impl). Numpy-heavy — a host utility, not the model package.

### W5 — Swift HF-hub auto-download in `WeightLoader`
`SeedVR2Weights(directory:)` currently takes a *local* dir. Add an HF-hub fetch (swift-transformers
`Hub`, or the minimal client `longcat-avatar-mlx-swift` ships) so `from_pretrained("mlx-community/SeedVR2-3B-mlx-int8")`
downloads `transformer/vae/pos_emb.safetensors + config.json` on first use. Mirror the
`longcat-avatar-mlx-swift` WeightLoader pattern.

### W6 — Scale ↔ resolution mapping
`ExportTier.scaleFactor` is 2/4; SeedVR2 upscales to a *target resolution*. Map the tier's
`scaleFactor` to the preprocess `ScaleFactor` (W2). SeedVR2 isn't intrinsically tied to a discrete
scale (same note as the `OSEDiff_MLX` stub) — 2×/4× are realized via the target-res computation.

## Gotchas / facts worth knowing

- **vid input is 33 ch = 16 noise + 17 condition** (`concat(latents, concat(encoded_latent, ones_mask))`). The `Upscaler` handles this; don't reshape.
- **RNG composes:** MLX-Swift and MLX-Python share the RNG core, so `MLXRandom.normal(key: key(seed))` matches the Python reference exactly (verified `max_abs 0.0`). Seed is a real knob.
- **int8 load is config-driven:** `config.json` carries `quantization {bits, group_size}`; `SeedVR2Upscaler` applies the same `quantize()` before loading. Nothing extra needed to consume the int8 repo.
- **VAE precision:** the VAE casts activations to **bf16** after each group-norm (mflux `ModelConfig.precision`) — replicated in `VAEBlocks.swift`. Keep VAE fp16 (don't quantize it; int4/int8 VAE degrades decode).
- **int4 is NOT viable** for this model (benchmark 22.7 dB). int8 only for on-device.
- **Benchmarks (M5 Max, 480×360 → 2×):** fp16 12.2 s / 11.5 GB · int8 ~8.9 GB (memory win, ~same speed — the 1-step DiT is VAE/IO-bound). See `seedvr2-mlx/benchmarks/REPORT.md`.

## Reference map
- Model package API: `Sources/SeedVR2MLX/{Pipeline/Upscaler.swift, Utilities/WeightLoader.swift, Utilities/Quantization.swift}`.
- Port detail + parity table: `docs/PORT-PLAN.md`. Provenance: `../seedvr2-mlx/docs/PROVENANCE.md`.
- mflux oracle (preprocess/color-correct/tiling source): `mflux/models/seedvr2/` and `mflux/models/common/vae/vae_tiler.py`.
- Forge target: `ForgeUpscaler/Sources/ForgeUpscaler/Export/{ExportTier.swift, OSEDiff_MLX.swift, ExportUpscaler.swift}`; ADR-0007 (`Docs/ADRs/0007-real-esrgan-export-tier.md`) is the export-tier rationale + the trigger to swap stubs for native MLX.
