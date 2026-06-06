# SeedVR2 → MLX-Swift — Port Plan

**Goal:** standalone MLX-Swift SeedVR2 (one-step diffusion SR) for **MLXEngine / ForgeUpscaler Export tier** (peer to `OSEDiff_MLX`, joining the already-ported `EfRLFN`). No runtime dependency on mflux.

**Oracle:** [`filipstrand/mflux`](https://github.com/filipstrand/mflux) `src/mflux/models/seedvr2/` (MLX-**Python**). This is an MLX-Python → MLX-Swift translation — numerics/layout/QKV/RoPE decisions are already made and parity-tested. Keep module structure **isomorphic** to mflux.

**Artifacts (from `seedvr2-mlx/scripts/prepare_swift.py`, Phase 1 ✅):**
- Weights: `dist/SeedVR2-3B-mlx/{transformer,vae,pos_emb}.safetensors` + `config.json` (fp16; reload self-check **0.00e+00** vs mflux).
- Goldens: `seedvr2-mlx/tests/goldens_swift_seedvr2-3b.npz` (CPU, seed 42, 2x) — stage boundaries:
  `vae_enc_in/out` (1,3,1,272,512)→(1,16,1,34,64) · `t_vid_in/out` (1,33,1,90,120)→(1,16,1,90,120) · `t_timestep` · `t_txt_in` (1,58,5120) · `vae_dec_in/out` · `final_image` (720,960,3).

## 3B dims (Config.swift)
vid_in=33, vid_out=16, vid_dim=2560, txt_in=5120, heads=20, head_dim=128, layers=32, mm_layers=10, rope_dim=128, patch=(1,2,2), window=(4,3,3). 7B: vid_dim=3072, heads=24, layers=36, mm_layers=36, rope_dim=64, rope_on_text=false.

## Module work-list (mflux → Swift), with parity gate

Parity on **CPU** (Apple-GPU fp32 is tf32-like). Leaf modules: random-input self-consistency;
stage modules: vs goldens (`< 1e-2` fp16). Status: ✅ done · ⬜ todo.

### Transformer (`seedvr2_transformer/` → `Sources/SeedVR2MLX/Models/Transformer/`)
| mflux file | Swift file | notes | status |
|---|---|---|---|
| rms_norm.py | RMSNorm.swift | `MLXFast.rmsNorm` | ✅ |
| swiglu_mlp.py | SwiGLU.swift `SwiGLUMLP` | hidden=6912 confirmed; bias=false | ✅ |
| mm_swiglu.py | SwiGLU.swift `MMSwiGLU` | dual-stream (vid+txt), last-layer = vid-only | ✅ |
| ada_modulation.py | AdaModulation.swift | nested `params_vid/txt.*`; emb [B,dim,2,3] | ✅ |
| time_embedding.py | TimeEmbedding.swift | sinusoid (computed, not param) → MLP → 15360 | ✅ |
| patch_in.py | Patch.swift `PatchIn` | patch (1,2,2); 33·4=132→2560; round-trip verified | ✅ |
| patch_out.py | Patch.swift `PatchOut` | unpatch → 16·4=64 channels | ✅ |
| rope.py | RoPE.swift | axial 3D freqs + mm-rope; `freqs` (21,) stored buffer | ✅ (bit-exact in block0) |
| window.py | Window.swift | variable-size partition + **shift path** (odd blocks) | ✅ |
| attention.py | Attention.swift | windowed MM attn; shared blocks reuse vid+txt keys (equal) | ✅ |
| transformer_block.py | TransformerBlock.swift | norm→ada→attn→ada→res→mlp; shared + last variants | ✅ |
| transformer.py | Transformer.swift | full assembly: vid_in, txt_in, emb_in, 32 blocks, out ada, patch_out | ✅ |

**✅ TRANSFORMER COMPLETE & VERIFIED (2026-06-05).** block-0 attn+block parity `max_abs 0.0`
(bit-exact); **full `t_out` gate `max_abs 2.1e-4`** (32 blocks incl. shifted windows +
shared-weight blocks 10–31 + output ada — fp16 accumulation, well within tolerance), on CPU.
Shared blocks differ only in `mlp.all` / `ada.params_all` keys (attn always carries equal
vid+txt keys). Full-transformer CPU run ≈ 3.6 min (window-attn python-style loops; GPU for prod).

### VAE (`seedvr2_vae/` → `Sources/SeedVR2MLX/Models/VAE/`) — ✅ DONE & VERIFIED
| mflux | Swift | notes | status |
|---|---|---|---|
| common/conv3d.py | CausalConv3d.swift | `convGeneral` NDHWC; causal temporal pad; cast input→weight dtype | ✅ |
| common/attention_3d.py | VAEBlocks.swift `Attention3D` | spatial self-attn; groupnorm fp32→bf16 | ✅ |
| encoder/* | VAE.swift `Encoder3D` + VAEBlocks (Resnet/Down/Mid) | 8× downsample → 32ch → mean(16) | ✅ |
| decoder/* | VAE.swift `Decoder3D` + VAEBlocks (Resnet/Up/Mid) | pixelshuffle upsamplers; latent → RGB | ✅ |
| vae.py | VAE.swift `SeedVR2VAE` | encode/decode (scaling_factor 0.9152) | ✅ |

**✅ VAE VERIFIED (2026-06-05):** encode `rel_err 3.5e-3`, decode `7.2e-3` vs mflux goldens (CPU).
Residual is bf16 activation-cast precision (mflux casts to `ModelConfig.precision=bfloat16` after
every group-norm; replicated). Both major components (transformer + VAE) now done.

### Pipeline (`variants/upscale/`, `latent_creator/`, scheduler → `Sources/SeedVR2MLX/Pipeline/`)
| mflux | Swift | notes | status |
|---|---|---|---|
| seedvr2_latent_creator.py | LatentCreator.swift | seeded noise + condition (concat mask) | ✅ |
| schedulers/seedvr2_euler_scheduler.py | Scheduler.swift | 1-step euler | ✅ |
| seedvr2.py (generate_image) | Upscaler.swift | encode → 1-step transformer → step → decode (core path) | ✅ |
| WeightLoader | Utilities/WeightLoader.swift | load exported safetensors + config | ✅ |
| seedvr2_util.py preprocess (bicubic resize/softness) | — | **host/Forge or utility** (PIL-exact bicubic) | ⬜ |
| seedvr2_util.py LAB-wavelet color-correct | — | **host/Forge or utility** (numpy-heavy post-proc) | ⬜ |
| VAE tiling (VAETiler) | — | **host (ForgeUpscaler.MLXTileProcessor)** | ⬜ |

**✅ CORE PIPELINE DONE & VERIFIED (2026-06-05):** seeded-noise RNG match vs Python **max_abs 0.0**
(MLX-Swift/Python share the RNG core — no noise injection needed), scheduler 1-step **max_abs 0.0**,
full-res decode wiring vs non-tiled oracle **rel_err 6.8e-3**. Preprocess/color-correct/tiling are
host concerns (PIL-bicubic + numpy-LAB + tiling don't belong in the model package; Forge supplies them).

### 🎯 FULL INFERENCE PATH VERIFIED — every stage parity-locked vs mflux (CPU)
| stage | metric |
|---|---|
| Transformer `t_out` (32 blocks) | max_abs **2.1e-4** |
| VAE encode / decode | rel_err **3.5e-3 / 7.2e-3** |
| Seeded-noise RNG / scheduler | max_abs **0.0 / 0.0** |
| Decode wiring (non-tiled) | rel_err **6.8e-3** |

**✅ int8 DONE & VERIFIED (2026-06-05):** `Quantization.swift` quantizes transformer Linears
(groupSize 64, bits 8; skips in-dim % 64 ≠ 0 → `vid_in.proj` stays fp16; VAE fp16). Config-driven
load path (WeightLoader reads `quantization`, Upscaler applies before update). GPU-verified
(~3 s): int8 `t_out` cosine vs fp16 **0.9999749** (near-lossless), reload round-trip **1.0**.
Transformer **7.9 GB → 3.9 GB**; self-contained `SeedVR2-3B-mlx-int8/` produced. *(Quant-quality
tests run on GPU — int8-vs-fp16 cosine needs no CPU oracle stream.)*

**Remaining = packaging/integration only:** preprocess + LAB color-correct (host utilities),
VAE tiling (ForgeUpscaler `MLXTileProcessor`), ForgeUpscaler Export-tier conformer, publish
`mlx-community/SeedVR2-3B-mlx{,-int8}`.

## Sequence
1. Leaf parity: RMSNorm ✅ → RoPE → SwiGLU → AdaModulation → Attention (vs random + cross-check vs a small mflux dump).
2. Transformer stage parity: build full transformer, load `transformer.safetensors`, run on `t_vid_in/t_txt_in/t_timestep`, gate vs `t_out` (`< 1e-2`). **This is the make-or-break gate.**
3. VAE stage parity: encode vs `vae_enc_*`, decode vs `vae_dec_*`.
4. Pipeline: full upscale, compare to `final_image` (PSNR; SR is generative so allow trajectory drift, sanity + visual).
5. ForgeUpscaler Export-tier conformer + tiling (reuse `MLXTileProcessor`); int8 on-device; publish `mlx-community/SeedVR2-3B-mlx`.

## Risks / notes
- **int4 degrades** this model (benchmark: 22.7 dB); ship **int8** for on-device (near-lossless 50 dB, ~4 GB).
- 7B is ~2× memory; 3B is the on-device target.
- RNG: capture noise from Python goldens and inject for parity (don't rely on cross-impl seed match).
