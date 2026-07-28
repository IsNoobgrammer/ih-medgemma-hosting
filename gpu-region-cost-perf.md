# GPU availability, cost and performance — India regions (verified 2026-07-28)

> **This file supersedes the GPU/region assumptions in `mumbai-phase1-runbook.md`.** That runbook targets Mumbai + RTX PRO 6000, which is **impossible** — see §1.

---

## 1. ⛔ Mumbai + RTX PRO 6000 is impossible (proven by a failed deploy)

A Model Garden one-click deploy of `medgemma-27b-it` to `asia-south1` with `g4-standard-48` was attempted and **failed**:
```
Deploying model "publishers/google/models/medgemma@medgemma-27b-it"
   Machine type "g4-standard-48" is not supported.
```
**Quota ≠ availability.** The project holds `Custom model serving Nvidia RTX Pro 6000 GPUs per region = 1` in `asia-south1`, but Vertex does not serve the `g4` machine type there. The quota row is meaningless. **Trust the docs' regional accelerator table and a real deploy attempt; never trust a quota row alone.**

Console UI bug that enabled the mistake: changing **Region** does **not** re-filter the **Machine spec** dropdown, so a Singapore-derived `g4-standard-48` persisted after switching to Mumbai and produced a form that looked valid but the API rejects.

## 2. Authoritative regional GPU support (Locations for ML services doc, unmarked = available for online inference)

| GPU | `asia-south1` Mumbai | `asia-south2` Delhi |
|---|---|---|
| H100 80GB | ✅ *(but see §3 — only as 8-GPU A3 Edge/Mega)* | ❌ |
| H200 | ✅ | ✅ |
| L4 | ✅ | ❌ |
| T4 | ✅ | ❌ |
| **RTX PRO 6000** | ❌ | ✅ |
| A100 40/80GB, B200, H100 Mega, P4, P100, V100 | ❌ | ❌ |

Footnotes: `*` = not available for batch/online inference · `⁺` = allowlist-only · `†` = not for training. Mumbai and Delhi rows carry **no markers**.

## 3. ⚠️ A single H100 does not exist in India (verify before filing any QIR)

GCP GPU regions/zones:
- `asia-south1`: A3 **Ultra** (H200, **8-GPU only**), A3 **Edge** (H100 **Mega**, **8-GPU only**), G2+T4, **G4**
- `asia-south2`: **G4**, A3 Ultra

**`a3-highgpu-*` — the only single-H100 family — is in neither Indian region.** The "H100" listed for Mumbai is H100 **Mega** inside 8-GPU-only A3 Edge. Filing an H100-80GB QIR for Mumbai risks the exact §1 failure again.
Additionally `a3-highgpu-1g` is **Spot / Flex-start only** on GCE — not purchasable on-demand at 1-GPU granularity anywhere.

**Verification command (needs Cloud Shell "Authorize"):**
```bash
gcloud compute accelerator-types list --filter="zone~asia-south" --format="value(zone,name)" | sort
gcloud compute machine-types list --filter="name~a3-highgpu AND zone~asia-south"
```
Empty second result ⇒ Mumbai+H100 is dead ⇒ **Delhi + RTX PRO 6000 is the only India-resident single-GPU option.**

## 4. Project serving quota (measured 2026-07-28)

**`asia-south1` standard:** L4 **6** (4 in use) · T4 14 · RTX Pro 6000 1 *(unusable, §1)* · H100 80GB **0** · A100/A100-80GB/H200/H100-Mega/B200 **0**
**`asia-south1` Spot:** B200 115 · H100 Mega 115 · H200 112 · RTX Pro 6000 14 · T4 14
**`asia-south2` standard:** **H200 16** · P4 1 · everything else **0** (incl. RTX Pro 6000, L4, T4, H100)

⚠️ Delhi's H200 = 16 and Mumbai's Spot pool are **unspendable on MedGemma** — Model Garden offers only RTX_PRO_6000, A100_80GB, H100_80GB and L4 specs for this model. No H200/B200/MEGA spec exists.

## 5. MedGemma 27B machine specs offered by Model Garden

| Spec | Context / images | Mumbai | Delhi |
|---|---|---|---|
| 1 × RTX_PRO_6000 `g4-standard-48` | **128K** | ❌ | ✅ *(quota 0 → QIR)* |
| 1 × H100_80GB `a3-highgpu-1g` | 32K / 16 img | ❌ §3 | ❌ |
| 2 × H100_80GB `a3-highgpu-2g` | 90K / 16 img | ❌ §3 | ❌ |
| 8 × H100_80GB `a3-highgpu-8g` | 128K / 16 img | ⚠️ A3 Edge is H100 **Mega** — verify | ❌ |
| 4 × L4 `g2-standard-48` | 24K / 5 img | ✅ quota 6, 4 used | ❌ |
| 8 × L4 `g2-standard-96` | 64K / 5 img | ⚠️ needs L4 6→8 | ❌ |
| 1 × A100_80GB `a2-ultragpu-1g` | 32K / 16 img | ❌ | ❌ |

## 6. Pricing — Vertex list, measured in-console

| Config | Region | $/hour | $/month 24×7 |
|---|---|---|---|
| **1 × RTX PRO 6000** · 128K | **Delhi** | **$6.22** | $4,537 |
| 1 × RTX PRO 6000 · 128K | Singapore | $6.38 | $4,661 |
| 1 × H100 80GB · 32K | Singapore | $16.42 | $11,985 |
| 8 × H100 80GB · 128K | Singapore | $101.01 | $73,735 |
| 1 × H100 80GB | Mumbai | *console suppresses the estimate at 0 quota* | |

Delhi came in **2.5% below** Singapore, so Indian regions carry no premium here.

**Correction to an earlier claim in this repo:** RTX PRO 6000 is *not* meaningfully better value per token. Hourly it is ~2.6× cheaper, but H100 delivers ~2.0–2.1× the throughput, so **per-token cost is a wash** (~$0.30 vs $0.31/Mtok on RunPod; 1.97× price for ~2× perf on GCE asia-south1). **Availability, not price, is the deciding factor.**

## 7. Performance — hardware and measured

**Corrections:** RTX PRO 6000 **Server Edition** bandwidth is **1,597 GB/s** (not the 1,792 Workstation figure used earlier in this repo). NVIDIA's "2 PFLOPS FP8 / 4 PFLOPS FP4" headlines are **with sparsity**; dense FP8 ≈ **960 TFLOPS** vs H100's **1,979**.

| | H100 SXM 80GB | RTX PRO 6000 SE 96GB |
|---|---|---|
| Bandwidth | **3.35 TB/s** | 1,597 GB/s → **H100 2.10×** |
| FP8 dense | 1,979 TFLOPS | ≈960 → **H100 2.06×** |
| FP4 | none (sm_90) | ≈1,920 dense (sm_120) |
| NVLink | 900 GB/s | **none** (PCIe 5 only) |

**Single-stream decode, MedGemma 27B:** measured **27–38 tok/s on 1×H100 via Vertex**; scaled by 2.10× → **RTX PRO 6000 BF16 ≈ 13–18 tok/s** (a 200-token answer ≈ 12–15 s). **FP8 ≈ 40 tok/s.**
→ **Model Garden one-click serves BF16.** So one-click on Delhi is deployable fast but sluggish per-stream; FP8 needs the custom-container path.

**Measured RTX PRO 6000 SE, Qwen3-32B, vLLM, ShareGPT:**
| | @conc 8 | @conc 128 |
|---|---|---|
| BF16 | 165 tok/s | 1,156 tok/s · TTFT 704 ms · TPOT 93.9 ms |
| **FP8** | 287 tok/s | **1,558 tok/s · TTFT 481 ms · TPOT 68.8 ms** |
| NVFP4 | 317 tok/s | 2,050 tok/s · TTFT 288 ms · TPOT 46.7 ms |

**96 GB vs 80 GB** buys 1.87× more KV in BF16 but only 1.34× in FP8 — the extra VRAM matters most exactly where you least want to run.

⚠️ **No public Gemma-3-27B or MedGemma-27B benchmark on RTX PRO 6000 exists** (Google's own Cloud Run RTX PRO 6000 post uses Gemma 3 **1B** with zero numbers). Every 27B figure for that card is a proxy model. A ~2-hour run on a `g4-standard-48` Spot instance with `RedHatAI/gemma-3-27b-it-FP8-dynamic` closes it.

## 8. Quantization verdict

**Use FP8-dynamic or W4A16. Skip NVFP4.**

| Checkpoint | Recovery | Note |
|---|---|---|
| `RedHatAI/gemma-3-27b-it-FP8-dynamic` | **99.73%** | vision 100.11% |
| `RedHatAI/gemma-3-27b-it-quantized.w4a16` | **99.70%** | 348k dl; runs on **both** cards via Marlin |
| `RedHatAI/gemma-3-27b-it-quantized.w8a8` (INT8) | **97.84%** | ❌ **avoid** — GSM8K −6.4 pts |
| any Gemma-3 **NVFP4** | — | ❌ **does not exist** (RedHat/NVIDIA ship Gemma **4** only) |

**Why NVFP4 is not a lever:** only **~1.3×** over FP8 (decode is bandwidth-bound, not compute-bound); sm_120 kernels are unstable (SGLang NaNs, FlashInfer init failures, live silent-KV-corruption issues); **NVFP4 KV cache is unsupported on sm_120**; and NVFP4 accuracy at ~30B drops to **~94% on reasoning** — unacceptable for a clinical pipeline.

## 9. Operational gotchas for whatever we deploy

- **Hybrid sliding-window KV is on by default** (vLLM ≥ 0.9.1) → 5.3× KV saving at 32k. Verify at startup: logs must show **7 KV cache groups** and *"Add 8 padding layers"*. If you see *"Hybrid KV cache manager is disabled for this hybrid model"*, you silently lost ~5× of your KV.
- **On sm_120, FP8 KV is a memory saving only — no attention speedup** (FP8 queries need FA3, which is sm_90-only). It also requires the **`FLASHINFER` or `TRITON_ATTN`** backend, not `FLASH_ATTN`.
- `--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window` is the recommended hybrid-model pairing (flag is churning upstream — pin your vLLM version).
- **fp16 is hard-blocked for Gemma 3 in vLLM** ("numerical instability") — bf16 or fp32 only.
- Exclude `lm_head`, `embed_tokens` (tied, 262k vocab), `vision_tower`, `multi_modal_projector` from quantization.
- vLLM issue **#29531**: FP8 KV + FlashInfer + Gemma 3 fails **on Hopper H100** — closed by stale-bot, not fixed. (Bites H100, not Blackwell.)

## 10. QIR procedure (verbatim from the console)

> *"You have no deployment quota for NVIDIA_H100_80GB in asia-south1. To deploy this model, submit a quota increase request (QIR). **Specify Model Garden and 1 GPU** in the quota increase request. You will be notified of the QIR outcome in **3–5 business days**."*

The "specify Model Garden" phrasing matters — a generic GPU quota grant may not confer Model-Garden deployment rights. Same wording applies to the Delhi RTX PRO 6000 request.

## 11. Where the decision stands

**Delhi + 1 × RTX PRO 6000 is very likely the only India-resident single-GPU option for MedGemma 27B.** Pending the §3 verification, the real choice is not H100-vs-RTX but:

| | BF16 one-click | FP8 custom container |
|---|---|---|
| Single-stream | ~13–18 tok/s | ~40 tok/s |
| Work | none (Model Garden) | build image + stage weights (see runbook §0) |
| Blocker | Delhi RTX quota 0 → QIR | same QIR + custom-container work |

**Unverified / open:** Vertex `a3-highgpu-1g` hourly rate (console suppresses it at 0 quota; ~$16 is inferred, do not quote) · whether RTX PRO 6000 avoids the GeForce FP32-accumulate halving (a 2× swing on the serving path) · any real 27B benchmark on this card.
