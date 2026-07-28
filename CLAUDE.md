# ih-medgemma-hosting

Serving **MedGemma 27B** in production. Prototyped on **RTX PRO 6000 Blackwell (96 GB, molab)**, shipping on **1 × NVIDIA L40S (48 GB)** via **Shakti Studio (Yotta)** BYOC containers.

> Working doc + project constitution. Update it as decisions change. Research compiled 2026-07-18; re-verify version-specific claims before relying on them.

---

## ✅ LIVE IN PROD-SHAPE (2026-07-28) — vLLM on 1×L40S, FP8, hybrid SWA confirmed

| | |
|---|---|
| Endpoint | `https://http.riw9sgruhe.shaktistudio.shakticloud.ai` → `/v1/chat/completions` |
| Deployment | `medgemma-vllm-triton-l40s` · `b5793d5b-7bdc-4ea7-8ac4-815a98376840` |
| Model record | `medgemma-27b-vllm-fp8-cu129` · `dc51e970-4245-4e61-8042-088e2c3f39ab` |
| Image | **`vllm/vllm-openai:v0.26.0-cu129`** (NOT `latest` — see trap #1) |
| Weights | **`fhai50032/medgemma-auto-fp8`** (public) — FP8-dynamic compressed-tensors |
| Auth | vLLM's own `VLLM_API_KEY` (masked env var), Shakti `Enable Auth` **OFF** |
| Served name | `medgemma-27b-fp8` (≠ HF repo id — clients must use this) |

```bash
vllm serve fhai50032/medgemma-auto-fp8 --served-model-name medgemma-27b-fp8 \
  --attention-backend TRITON_ATTN --kv-cache-dtype fp8_e4m3 \
  --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 4 \
  --dtype bfloat16 --host 0.0.0.0 --port 8000
```

**MEASURED on the real L40S (supersedes every estimate in this file):**

| | Estimated | **Measured** |
|---|---|---|
| Weights in VRAM | ~26 GiB | **27.42 GiB** ✅ FP8 branch taken (vs 51 GiB BF16 that OOM'd) |
| KV pool | ~12–15 GiB | **10.56 GiB** = **188,347 tokens** |
| Concurrency @ 64K | ~4.4 | **2.87** (vLLM's own `Maximum concurrency` line) |
| Concurrency @ ~4K (real extraction) | — | **~45** |
| Single-stream decode | ~32 tok/s roofline | **~20 tok/s** (~60% of roofline) |
| Cold start | 15–40 min | **~18 min** (888 s of it = HF download) |

**Hybrid SWA IS ACTIVE on vLLM** — two independent proofs: the log emits `Add 8 padding layers, may waste at most 15.38% KV cache memory` (the hybrid-KV-manager marker), and 10.56 GiB → 188,347 tokens = **58.8 KB/token vs 248 KB/token** non-hybrid (62 layers × 4 KB) = **4.2× saving**. Confirmed `Using AttentionBackendEnum.TRITON_ATTN` and `Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8` (real FP8 kernels, not dequantized).

**Verified working:** `/health` 200 · `/v1/models` **401 without token**, 200 with · clinical prompt returns correct reasoning (CAP with pleuritic pain) · OLDCARTS extraction on real NAS_v3 notes **matches Gemini field-for-field**.

### ⚠️ Four traps that cost real deploy cycles — do not repeat
1. **`vllm/vllm-openai:latest` is cu130 and will NOT run on Shakti's L40S nodes** (host driver reports CUDA **12020** ≈ 535). `RuntimeError: The NVIDIA driver on your system is too old`. Containers bundle the CUDA *runtime*, never the *driver* — no image can fix an old host driver. CUDA has **minor**-version compat within 12.x but **no major**-version compat. **Always pin `-cu129`.**
2. **`--disable-log-requests` was REMOVED** from vLLM. It's now `--enable-log-requests` (opt-in), so request logging is **off by default** — which is what we want, since prompts are patient notes. Passing the old flag is a hard argparse failure.
3. **`VLLM_ATTENTION_BACKEND` no longer exists** (absent from `envs.py` in 0.26; `cuda.py` takes `selected_backend` from config). Setting it is a **silent no-op** — you'd keep running FlashInfer while believing otherwise. Use the CLI flag **`--attention-backend TRITON_ATTN`**.
4. **Shakti `Command Override` REPLACES the entrypoint** (proven: only the bad flag was rejected, not a stray `vllm`). So write the full `vllm serve …` command. Also: **model records are immutable** — only Clone/Delete, no edit.

### Structured output: Gemini `response_schema` ≠ vLLM `response_format`
The extractor's schema reached Gemini **out-of-band**; vLLM's `response_format` only *constrains* decoding and never **shows** the model the field names. Worse, every `Extraction` field has a default → pydantic emits `required: []` → guided decoding legally returned `{"patient_history":[],"family_history":[]}` in **17 tokens**: schema-VALID but empty. Two fixes, both in `extract_medgemma.py`:
- `_strict()` walks the schema and marks **every** property required
- `_SCHEMA_BLOCK` injects the schema JSON into the prompt
Bonus: that block is a byte-identical ~1.3k-token prefix on every request → exactly the shared-prefix workload vLLM's prefix caching (on by default) is for.

### Open
- Cold start is **888 s of HF download**; add `HF_TOKEN` env (HF's own log says it lifts rate limits + speed). `hf_transfer` is **not** in vLLM's requirements — setting `HF_HUB_ENABLE_HF_TRANSFER=1` **hard-errors**; `HF_XET_HIGH_PERFORMANCE=1` is safe (ignored if absent).
- `torch.compile` cache (74 s) lives in `/root/.cache` **inside the container** → re-paid every restart unless a volume is mounted. Ask Yotta about a PVC.
- 2.87 concurrent @ 64K is short of the 4 target. Levers: `--gpu-memory-utilization 0.96` (~3.3), `--max-model-len 48K` (~3.8) / 32K (~5.7), or H100. Real extraction traffic (~4K) already gives ~45.
- Stale model records to delete: `1b79f2` (dead flag), `3ebbff` (`latest`→cu130).

## Decisions locked

- ⛔ **REVERSED 2026-07-28 — Engine is now vLLM, not SGLang.** SGLang unconditionally disables hybrid sliding-window KV for **every** Gemma 2/3/3n architecture (`arg_groups/overrides.py`, `@_register_for(...)` → `{"disable_hybrid_swa_memory": True}`, driven by a `FIXME` about a **gemma2** CI failure). It is **not overridable**: `resolvable=True` means the override is *permitted to overwrite the user's value*, and `materialize_declarations()` applies it unconditionally at the end of `__post_init__`. Without hybrid SWA, all 62 layers cache full context → 15.5 GiB per 64K sequence → **64K cannot fit on a 48 GB card at all**. vLLM has hybrid SWA on by default (measured 4.2× above) *and* prefix caching, so nothing was lost.
- ~~**Engine: SGLang** (over vLLM). Chosen for RadixAttention prefix reuse — our workload is one big shared clinical system prompt across most requests.~~ *(vLLM V1 has automatic prefix caching on by default — `enable_prefix_caching=True` confirmed in the live log — so the original reason for SGLang is fully covered.)*
- **Precision: FP8 weights + FP8 KV cache *everywhere*.** Accepting **0.1–0.9% accuracy loss** as fine. (vLLM's clean global-FP8/local-BF16 mixed cache is NOT the plan; we want FP8 KV on all layers via SGLang.)
- **Attention backend: Triton** (`--attention-backend triton`) — arch-generic, JIT-compiles for sm_89, and is the path SGLang uses to serve Gemma FP8-KV. **NOT FlashInfer** (see traps).
- **Prefix caching: ON** (SGLang RadixAttention, default) — biggest safe TTFT win.

## Deployment platform — Shakti Studio (Yotta)  [status: 2026-07-27]

Console `shaktistudio.ai`, org **Intelehealth Inc**. Two separate concepts — do not conflate:
**Models** (register the container: image, port, health, command, GPU count) → **Deployments** (pick that model + **Accelerator Type** + scaling).

**Accelerators offered: `H100` and `L40S` only. No L4** — and L4 (24 GB) could not hold 27B at FP8 (~27 GB) anyway. L40S is Ada `sm_89`, same 48 GB as L40 but ~2× FP8 compute, so all L40 research transfers (including the sm_89 FP8-KV risk).
The accelerator **quantity follows the model's `GPU Per Container`** — ours is 1, so the option is `1 × L40S`. (Marketplace Gemma 3 27B forces `2 ×` because it is BF16 ~54 GB.)

### BYOC model — CREATED & VERIFIED (not yet deployed)
- Name `medgemma-27b-sglang-fp8` · **Model ID `d4e8fa0b-742b-4281-9ce0-6824377c37f6`** · Status Success, image pull "Access verified"
- Image **`lmsysorg/sglang:v0.5.16-cu129`** (public Docker Hub, no registry creds). cu129 over cu130 = tolerates older node drivers.
- GPU Per Container **1** · HTTP **8000** (Public Access on) · Health `/health:8000`, delay 30s, threshold 300, period 10s, timeout 5s (≈50 min grace — needed for the 51 GB download) · Model Endpoint `/v1/chat/completions`
- Env: `HF_TOKEN`, `HF_XET_HIGH_PERFORMANCE=1` (hf_transfer is deprecated → Xet)
- Command override:
  `python3 -m sglang.launch_server --model-path google/medgemma-27b-text-it --quantization fp8 --kv-cache-dtype fp8_e4m3 --attention-backend triton --context-length 65536 --mem-fraction-static 0.85 --host 0.0.0.0 --port 8000`
  (**`--host 0.0.0.0`**, not 127.0.0.1 — must be reachable inside the cluster.)

### ✅ QUOTA GRANTED (2026-07-28) — the zero-quota blocker is cleared
Yotta granted **2 × L40S and 2 × H100**. The accelerator dropdown now offers `1 × H100` and `1 × L40S` with no `Quota Exceeded` label; the form caps **Maximum Pods ≤ 2** and reports `GPU Scaling Capacity: 1 - 2 GPUs`. We only need 1.

### ⛔ FIRST DEPLOYMENT FAILED — two hard blockers found (2026-07-28). Deployment deleted.

**1. Online FP8 quantization cannot work on a 48 GB card. `--quantization fp8` needs ~51 GiB of BF16 headroom first.**
```
Load weight begin. avail mem=44.02 GB
  fp8.py:539 create_weights → data=torch.empty(...)
torch.OutOfMemoryError: Tried to allocate 442.00 MiB.
GPU 0 has a total capacity of 44.53 GiB of which 260.75 MiB is free
```
`create_weights` runs **before any weight is read from disk** — it only allocates empty parameter tensors. SGLang's `Fp8LinearMethod` allocates FP8 **only if the checkpoint is already FP8-serialized**; otherwise it allocates `params_dtype` (BF16), loads BF16, then compresses in `process_weights_after_loading`. `google/medgemma-27b-text-it` is a BF16 checkpoint → peak demand ≈ **51 GiB**, not the ~26 GiB of FP8 we sized for. It died at 43.69 GiB, ~85% through the 62 decoder layers. FP8 was never reached. Crashlooped 3× then was deleted.
→ **A pre-quantized FP8 checkpoint is a PREREQUISITE for L40S, not an optimization.** (Was filed under "Deferred optimizations" — wrong call; the OOM risk noted there is the whole ballgame.) On an 80 GB H100 the BF16 transient fits, so online FP8 works there — L40S is the constrained case.
**Agreed plan (2026-07-28): pre-quantize MedGemma-27B to FP8-dynamic on an RTX 6000, publish to a private HF repo, point the model record's `--model-path` at it.** FP8-dynamic is data-free (no calibration set). Access to the box pending.

**2. Hybrid sliding-window KV is NOT supported for the text-only Gemma 3 class — our KV/concurrency math was ~6× optimistic.**
```
Disable hybrid SWA memory for Gemma3ForCausalLM as it is not yet supported.
→ server_args: disable_hybrid_swa_memory=True
```
All **62** layers cache the full context, not just the ~10 global ones. At 64K with FP8 KV (2 × 16 KV heads × 128 head_dim × 1 B = 4 KB/token/layer):

| | KV for one 64K sequence |
|---|---|
| Assumed (hybrid SWA working) | ~2.8 GB (10 global full + 52 local × 1024 window) |
| **Actual (SWA disabled)** | **~16.6 GB** (62 × 254 KB/token) |

This invalidates the "~18–20 concurrent @ 64k" estimate in *Batching math* below — on a 48 GB L40S it is roughly **one** request at full context. **Open: does `medgemma-27b-it` (multimodal, `Gemma3ForConditionalGeneration`) get hybrid SWA where `Gemma3ForCausalLM` does not?** If so that flips the variant choice, since the KV saving would outweigh the extra vision params. Check before committing to the text-only variant.

### Deployment config that produced the above (recreate from this)
| Field | Value |
|---|---|
| Name | `medgemma-27b-fp8-test` |
| Deployment ID | `5ba4faeb-efae-4ad3-95c4-c402ec5c2665` |
| Short code | `aygjutg9wi` |
| Endpoint | `https://http.aygjutg9wi.shaktistudio.shakticloud.ai` → `/v1/chat/completions` |
| Accelerator | **1 × L40S** |
| Pods | min 1 / max 1 (CPU 80% metric; scale-to-zero OFF, schedule OFF) |
| Auth | **Enabled** (was default OFF — a public unauthenticated 27B endpoint is an abuse/cost risk) |

Container config inherited cleanly from the model record (port 8000, `/health`, 30s delay / 300 threshold, the FP8 command override, `HF_TOKEN`/`HF_XET_HIGH_PERFORMANCE`). Deploy sequence observed in **Events**: pod scheduled on a GPU node → `istio-init` + `istio/proxyv2` sidecar → cert-manager ACME solver issues TLS for the public host → pull `lmsysorg/sglang:v0.5.16-cu129` → (then the 51 GB HF download + FP8 quantize + CUDA-graph capture). Until a backend is healthy the public host returns **nginx 503** — that is normal, and confirms DNS/TLS/routing are already correct.

Notes for next time: the **Region/accelerator dropdown renders in a portal that screenshots miss** — locate options with `find`, then `scroll_to`. `Maximum Pods` defaults to the quota max (2), so **set it to 1 explicitly** or you bill both GPUs.

### Pricing (⚠️ UNVERIFIED — unit ambiguous, confirm with Yotta before quoting)
`shakticloud.ai/pricing` lists, under **Serverless → Dedicated Instance** (the tier our BYOC deployment falls in): `1 × 48 GB L40S` **₹3.17** and `1 × 80 GB H100` **₹7.13**. The page's unit note (*"Per Hour, Per Compute"*) sits under a **different** section and the serverless table carries **no unit of its own**. ₹3.17/hour for an L40S is not credible (the same page's VM tier is **₹137/hr** for 1 × L40S and **₹356/hr** for 1 × H100), so the serverless figures are probably per-minute or per-some-other-unit. **Do not build a cost model on them.** Reliable ratio regardless of unit: **H100 ≈ 2.25× the price of L40S**.

### Scaling: SCHEDULE-BASED, not scale-to-zero  (decided 2026-07-27)
Working days, business window ~08:00–20:00. **Open the schedule window ~07:15–07:30**, because scale-up = image pull + ~51 GB HF download + FP8 quantize + CUDA-graph capture ≈ **15–40 min**; the pod is NOT ready at 08:00. ~60 h/week vs 168 h ≈ **64% cheaper** than always-on. Max pods **1** for tests — the form defaults to **4**, which would bill 4 GPUs.
**Consequence: do NOT bake weights into a custom image.** One cold start per day makes a once-daily download tolerable; a ~60 GB custom image + Depot build pipeline is not worth it. Keep the thin official image.
Verify when quota lands: (1) **schedule timezone** (UTC vs IST — 08:00 IST = 02:30 UTC); (2) whether scale-down **drains in-flight requests**.

### Deferred optimizations
- ~~**Pre-quantize to FP8 offline**~~ → **PROMOTED TO PREREQUISITE, see blocker #1 above.** It was never optional on a 48 GB card. (Bonus once done: payload 51 → ~26 GB, halving the daily cold-start download.)
- **Persistent volume / model cache**: BYOC exposes no volume mount and docs are silent. Ask Yotta whether a PVC at the HF cache path is possible — the actually-correct fix.
- Build infra if we ever do bake weights: **Depot** (a supported BYOC registry, remote builds) or a Yotta VM. Local build/push of 60 GB is impractical; molab cannot run Docker at all (no `CAP_SYS_ADMIN`, read-only cgroups).

### 🔒 Security note
The marketplace model page renders a **live org API JWT** in its sample code (serverless endpoint auth). Treat as a secret; rotate if that page has been screenshotted/shared.

## Must validate on the real L40S before prod (non-negotiable)
FP8 KV attention is the **sm_89 (Ada) soft spot**. It works on the Blackwell proto (sm_90+ kernels) and *may fault or silently misbehave on the L40*. This is the #1 thing the prototype **cannot** validate for us.
0. ⛔ **Weights must LOAD at all** — status 2026-07-28: they don't. Online FP8 quant OOMs (blocker #1); needs a pre-quantized checkpoint first. Items 1–4 are untested because the server never started. *(This is exactly the class of failure the 96 GB proto hides — it validated FP8 end-to-end on Blackwell while the 48 GB target cannot even allocate.)*
1. **FP8-KV on L40 runs clean** with SGLang + Triton backend (watch for the SGLang #22277-class Triton dtype-mismatch on Gemma hybrid attention).
2. **Memory fit** at 64k context / target concurrency on 48 GB (96 GB proto hides OOM) — **now much tighter than planned, see blocker #2.**
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

> ⚠️ **MEASURED WRONG — the whole section below assumes hybrid SWA works. It does not** (`Gemma3ForCausalLM`, SGLang v0.5.16 — see blocker #2 above). Every KV figure here is ~6× too small and the concurrency estimate ~6× too high. Real number at 64k on L40S is ≈1 concurrent request. **Re-derive after resolving the `-it` vs `-text-it` SWA question; do not quote these numbers.**
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
Full research report (charts + all source links): `C:\Users\shaur\researcher\medgemma-27b-serving_18_JUL_2026\index.html`. Key: Gemma 3 tech report (arXiv 2503.19786), MedGemma tech report (2507.05201), vLLM FP8-KV blog (2026-04-22), SGLang docs + #22277, RedHatAI FP8/W4A16 quants, NVIDIA L40 / RTX PRO 6000 datasheets.
