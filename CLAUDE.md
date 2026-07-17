# ih-medgemma-hosting

Serving **MedGemma 27B** in production. Prototype/validate on **RTX PRO 6000 Blackwell WS (96 GB)**, ship on **NVIDIA L40 (48 GB)**.

> Working doc + project constitution. Update it as decisions change. Research compiled 2026-07-18; re-verify version-specific claims before relying on them.

---

## Decisions locked

- **Engine: SGLang** (over vLLM). Chosen for RadixAttention prefix reuse — our workload is one big shared clinical system prompt across most requests.
- **Precision: FP8 weights + FP8 KV cache *everywhere*.** Accepting **0.1–0.9% accuracy loss** as fine. (vLLM's clean global-FP8/local-BF16 mixed cache is NOT the plan; we want FP8 KV on all layers via SGLang.)
- **Attention backend: Triton** (`--attention-backend triton`) — arch-generic, JIT-compiles for sm_89, and is the path SGLang uses to serve Gemma FP8-KV. **NOT FlashInfer** (see traps).
- **Prefix caching: ON** (SGLang RadixAttention, default) — biggest safe TTFT win.

## Must validate on the real L40 before prod (non-negotiable)
FP8 KV attention is the **sm_89 (Ada) soft spot**. It works on the Blackwell proto (sm_90+ kernels) and *may fault or silently misbehave on the L40*. This is the #1 thing the prototype **cannot** validate for us.
1. **FP8-KV on L40 runs clean** with SGLang + Triton backend (watch for the SGLang #22277-class Triton dtype-mismatch on Gemma hybrid attention).
2. **Memory fit** at 64k context / target concurrency on 48 GB (96 GB proto hides OOM).
3. **Real TTFT / tok-s** — proto decodes ~2× faster; never set SLAs from it.
4. **Accuracy of the shipped FP8 artifact** on a MedQA-style eval — confirm the 0.1–0.9% loss actually holds for FP8-KV-everywhere (published near-lossless numbers are FP8 *weights*, and NVFP4 numbers are on Qwen, not MedGemma).

---

## Model
- **MedGemma 27B == Gemma 3 27B** architecture, further-trained on medical data. Every serving quirk is a Gemma 3 quirk.
- Use **`google/medgemma-27b-text-it`** (text-only, 87.7% MedQA). `medgemma-27b-it` = multimodal (~29B, +MedSigLIP) — only if we need image/EHR input. Confirm modality of the repo you pull (HF auto-summary is inconsistent).
- License: HAI-DEF — commercial self-host OK. Card forbids direct clinical decisions without our own validation.
- Arch (drives serving): 62 layers, hidden 5376, **32 attn / 16 KV heads (GQA 2:1)**, head_dim **128**, vocab 262,208, **5 local : 1 global** attention, sliding window **1024**, context **128K**. Only the ~10 global layers scale KV with context; the ~52 local layers cache a fixed 1024-token window → KV stays <15% of a dense 27B at 32k.

## Hardware
| | L40 (prod) | RTX PRO 6000 Blackwell WS (proto) |
|---|---|---|
| VRAM | 48 GB GDDR6 | 96 GB GDDR7 |
| Bandwidth | 864 GB/s | ~1,792 GB/s |
| Arch | Ada `sm_89` | Blackwell `sm_120` |
| FP8 / FP4 | ✓ / **✗** | ✓ / ✓ |
| Notes | 300 W, no NVLink | 600 W, no NVLink |

**Bandwidth ratio ~2.07× → proto decodes ~2× faster than prod.** If we can choose the prod card, **L40S** (same 48 GB/864 GB/s, ~2× FP8 compute, better prefill) beats plain L40.

### Blackwell → Ada portability traps
1. Decode SLOs regress ~2× on L40. Never promise SLAs from the proto.
2. **NVFP4 is Blackwell-only** — an NVFP4 artifact won't run on L40. Ship FP8, not FP4.
3. BF16 (~54 GB) fits the proto, never the 48 GB L40.
4. TensorRT engines are arch-locked (`sm_120` ≠ `sm_89`) — build per-arch.

**Rule: pin identical FP8 quant + engine/version on both boxes**, so accuracy validated on the proto transfers 1:1. Cap proto to ~46 GB (`--gpu-memory-utilization ~0.48` on 96 GB) to catch OOM early. Bandwidth can't be faked — perf still needs a real L40.

## Quantization
- **Primary: FP8-dynamic weights** (~27 GB, fits 48 GB, ~99.7% recovery, Ada-native, no calibration).
- **FP8 KV cache** everywhere (our choice) — ~halves KV memory, ~doubles concurrency. sm_89 risk above.
- **Fallback: W4A16 (INT4-AWQ/GPTQ)** ~14 GB, ~99.7% recovery. Frees ~13 GB → more context/batch, and **sidesteps FP8-KV entirely** (INT4 weights + BF16 KV gets similar concurrency on a code path that already works on sm_89). Keep as plan B if FP8-KV misbehaves on L40.
- How labs do it: FP8 weights + dynamic per-token activation scales; GPTQ/AWQ/SmoothQuant for INT4; NVFP4 (2-level micro-scaling) is frontier but Blackwell-only.
- RedHatAI ships ready `gemma-3-27b-it` FP8-dynamic + W4A16 quants — mirror the recipe for MedGemma with `llm-compressor`.

## Batching math (64k context, ~50k shared prefix, FP8 weights + FP8 KV)
- KV pool on L40 ≈ 43 GB usable − 27 GB weights − ~3 GB overhead ≈ **~15 GB**.
- KV/token/layer = 2×16×128 = 4 KB (FP8) / 8 KB (BF16). Global layers (~10) cache full seq; local (~52) cache 1024 only.
- Shared 50k prefix (counted once): ~2 GB. Per unique request tail (14k global + 1024 local window): FP8 ≈ **~0.73 GB**.
- **Concurrency ≈ (15 − 2) / 0.73 ≈ ~18–20 concurrent @ 64k with 50k shared.** (Mixed cache would be ~13; no prefix sharing ~5.) Prefix caching ~2.5–3×'s batch capacity and collapses TTFT (only 14k of each prompt gets prefilled).
- **Roofline decode ceiling** (FP8, batch 1): L40 ≈ 32 tok/s, proto ≈ 66 tok/s (~60–70% real).
- These are estimates — memory is the binding constraint; **benchmark on L40**. Watch prefill queueing when many 14k-token tails hit at once (tune chunked prefill).

## Prefix caching rules
- Prompt must be **token-identical and first**. No per-request timestamps/IDs/whitespace inside the system prompt; identical chat-template rendering every time. One differing token breaks block reuse.
- SGLang RadixAttention (token-level) reuses partial-block prefixes; default on (`--disable-radix-cache` to turn off).

## Speculative decoding
- **No trained EAGLE-3 / Medusa / MTP / DFlash head exists for Gemma 3 27B / MedGemma.** (Those exist for Gemma 4 / Qwen 3.5 — different models.)
- Options: **n-gram / prompt-lookup** (no model, 2–2.8×, ideal for extraction-style medical output; set `prompt_lookup_min≈8` to avoid corrupting structured output) or a `gemma-3-1b`/`270m` **draft model** (vocab matches 262k, but unbenchmarked, ~1.5×).
- Spec decoding helps low-QPS latency; **net-negative (1.4–1.8× slowdown) at high concurrency**. It's a latency lever, not throughput.
- ⚠️ Since we run FP8-KV everywhere: **do NOT combine FP8-KV + spec-decode on sm_89** (crash class, vLLM #44879; SGLang ngram/DFlash also disable overlap scheduler). If we want spec-decode, drop that request path to BF16 KV.

## Known bugs / refs
- vLLM #20865 — FlashInfer silently disables Gemma 3 sliding window, caps context to 1024. → use FlashAttention/Triton.
- vLLM #44879 — FP8-KV + spec-decode CUDA illegal-access on sm_89 (L40). FP8 *weights* fine; FP8-KV *attention* is the fragile piece.
- SGLang #22277 — Gemma FP8-KV Triton `extend_attention` dtype mismatch with shared-KV layers; fixed on recent SGLang. Pin a version that serves Gemma FP8-KV cleanly.

## Dev workflow
1. **Proto (RTX 6000, FP8):** accuracy + functional correctness + app/prompt/cache integration. (Transfers 1:1 to L40 in FP8.)
2. **L40:** memory fit + FP8-KV sm_89 sanity + real TTFT/tok-s + SLOs. (Does NOT transfer from proto — must test here.)
3. Freeze config → optionally TensorRT-LLM for max L40 perf (arch-locked rebuild).

## Reference launch (SGLang, starting point — validate + benchmark)
```bash
python -m sglang.launch_server \
  --model-path google/medgemma-27b-text-it \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \        # NOT flashinfer on Gemma 3
  --context-length 65536 \
  --mem-fraction-static 0.9
  # RadixAttention prefix caching is on by default
  # spec-decode (ngram) NOT combined with fp8 kv on sm_89 — see #44879
```

## Open items (benchmark, don't assume)
- No public rigorous Gemma-3-27B TTFT/tok-s on L40/L40S — measure ourselves.
- Confirm FP8-KV-everywhere accuracy loss (0.1–0.9%?) on a MedQA-style eval for MedGemma.
- Confirm SGLang FP8-KV runs clean on L40 sm_89 with Triton backend.
- head_dim FP8-KV prefill penalty: config says 128; a 256-based ~1.6× TTFT caveat is unverified for this model.
- Verify RTX 6000 bandwidth vs exact SKU (1,792 WS vs 1,597 Server).

## Sources
Full research report (charts + all source links): `S:\Users\shaur\researcher\medgemma-27b-serving_18_JUL_2026\index.html`. Key: Gemma 3 tech report (arXiv 2503.19786), MedGemma tech report (2507.05201), vLLM FP8-KV blog (2026-04-22), SGLang docs + #22277, RedHatAI FP8/W4A16 quants, NVIDIA L40 / RTX PRO 6000 datasheets.
