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

### VAE (`seedvr2_vae/` → `Sources/SeedVR2MLX/Models/VAE/`)
| mflux | Swift | notes | status |
|---|---|---|---|
| common/conv3d.py | CausalConv3d.swift | `convGeneral` NDHWC; causal temporal pad (temporal=1 for images) | ⬜ |
| common/attention_3d.py | Attention3D.swift | spatial self-attn in VAE mid | ⬜ |
| encoder/{encoder_3d,down_block_3d,downsample_3d} | Encoder.swift | 8× spatial downsample → 16-ch latent | ⬜ |
| decoder/{decoder_3d,decoder_mid_block_3d,decoder_resnet_block_3d,up_block_3d,upsample_3d} | Decoder.swift | latent → RGB | ⬜ |
| vae.py | VAE.swift | encode/decode wrappers | ⬜ |

### Pipeline (`variants/upscale/`, `latent_creator/`, scheduler → `Sources/SeedVR2MLX/Pipeline/`)
| mflux | Swift | notes | status |
|---|---|---|---|
| seedvr2_util.py (preprocess) | Preprocess.swift | resolution/softness pre-downsample, pad to /16 | ⬜ |
| seedvr2_latent_creator.py | LatentCreator.swift | condition (upsample enc to target latent), seeded noise | ⬜ |
| schedulers/seedvr2_euler_scheduler.py | Scheduler.swift | 1-step euler | ⬜ |
| text_embeddings.py | TextEmbeddings.swift | load pos_emb.safetensors | ⬜ |
| seedvr2.py (generate_image) | Upscaler.swift | encode → 1-step transformer → step → decode → color-correct | ⬜ |
| WeightLoader | Utilities/WeightLoader.swift | load exported safetensors + config | ✅ |

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
