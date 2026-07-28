# GCP Vertex AI ("Agent Platform") — MedGemma deployment audit

Project **IH-MedPalm** (`ih-medpalm`), project number `770546093480`. Audited **2026-07-28** (read-only; nothing changed).
Account: `llm@intelehealth.org` (`authuser=2`).

> **Governing constraint: Indian data residency.** Patient data must remain in India → the production target is **asia-south1 (Mumbai)**. Singapore is staging only.
> **Platform constraint: Agent Console / Vertex AI only.** No Compute Engine, no GKE. All deploys via Model Registry / Model Garden in the console.

---

## A. SINGAPORE deployment — full record (captured before deletion)

**Endpoint**
| Field | Value |
|---|---|
| Name | `google_medgemma-27b-it-mg-one-click-deploy` |
| Endpoint ID | `mg-endpoint-02f73b83-4a1e-49b5-8351-0dbf68543ed9` |
| Region | `asia-southeast1` |
| Dedicated DNS | `mg-endpoint-02f73b83-4a1e-49b5-8351-0dbf68543ed9.asia-southeast1-1084674325994.prediction.vertexai.goog` |
| Status | Active · 1 model · traffic 100% |
| Created / updated | Jul 28 2026, 11:35 / 11:59 |
| Model Monitoring | **Disabled** |
| Replicas | **Auto (1 minimum, 1 maximum)** → always warm, no scale-to-zero |

**Deployed model**
| Field | Value |
|---|---|
| Model | `google_medgemma-27b-it-1785218666525` (Version 1) |
| Deployed model ID | `2589950216761245696` |
| Registry model imported | Jul 28 2026, 11:35:18 |
| Source | Model Garden · Objective Custom · Encryption Google-managed |

**Container spec (verbatim)**
```
image: us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20260127_0916_RC01
command/args:
  python -m vllm.entrypoints.api_server
  --host=0.0.0.0
  --port=8080
  --model=gs://vertex-model-garden-restricted-us/medgemma/medgemma-27b-it
  --tensor-parallel-size=1
  --swap-space=16
  --max-model-len=131072
  --gpu-memory-utilization=0.95
  --max-num-seqs=16
env:
  MODEL_ID=google/medgemma-27b-it
  DEPLOY_SOURCE=API_NATIVE_MODEL
inference route: /generate      health route: /ping
shared memory size: 0 B        startup/custom probes: unset
```

**GPU — derived from vLLM startup logs (not guessed)**
```
Available KV cache memory: 35.31 GiB
GPU KV cache size: 66,112 tokens
WARNING kv_cache_utils: Add 8 padding layers, may waste at most 15.38% KV cache memory
init engine (profile, create kv cache, warmup model) took 50.05 seconds
(EngineCore_DP0 …)  +  "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)"
```
`TP=1` + 27B BF16 weights (~54 GiB) + 35.31 GiB KV + graphs ≈ 92 GiB at util 0.95 → **~97 GiB card = 1 × NVIDIA RTX PRO 6000 Blackwell (96 GB)**. An 80 GB H100/A100 would leave only ~19 GiB KV, not 35.31 — so the 96 GB card is confirmed.
`EngineCore_DP0` + PIECEWISE CUDA graphs = **vLLM V1 engine** (no `VLLM_USE_V1=0`), therefore **automatic prefix caching is ON by default**.

**What is GOOD here (carry forward to Mumbai)**
- vLLM **V1** engine → prefix caching on. This is the single most important setting for our shared-system-prompt workload.
- `--tensor-parallel-size=1` — no sharding, no NCCL, no shared-memory risk.
- Container **20260127** (Jan 2026) — ~9 months newer than Mumbai's.
- `min replicas = 1` → no cold starts.
- No `--disable-log-stats` → throughput metrics available.

**What NEEDS FIXING**
1. **`--max-model-len=131072` is not achievable.** KV holds only **66,112 tokens total** across all concurrent requests. A single 131k request cannot fit. Effective ceiling ≈ 66k, shared across concurrency. Set `--max-model-len` to something honest (e.g. 32k–65k) or raise KV via FP8.
2. **No FP8.** BF16 weights eat 54 GiB of the 96 GiB. `--quantization fp8` → ~27 GiB, freeing ~27 GiB → KV roughly doubles to ~62 GiB (~120k+ tokens). `--kv-cache-dtype fp8` on top roughly doubles token capacity again.
3. **Not a real OpenAI endpoint.** Runs `vllm.entrypoints.api_server` with route `/generate`. Works via Vertex `:predict` + `@requestFormat: chatCompletions` (Google's wrapper routes it to `create_chat_completion`, confirmed in a stack trace), but it is not `/v1/chat/completions`.
4. **Model Monitoring disabled.**
5. **Multimodal `-it` variant** — if the use case is text-only DDx, `medgemma-27b-text-it` is smaller/cheaper.
6. `shared memory size: 0 B` — harmless at TP=1; would matter if TP>1.
7. **Wrong region for production** (data residency).

---

## B. MUMBAI deployment — full record (the one to replace)

| Field | Value |
|---|---|
| Endpoint | `google_medgemma-27b-it-mg-one-click-deploy` |
| Endpoint ID | `mg-endpoint-ae703e35-8c45-40fd-9ee1-6ce7515d3ddf` |
| Dedicated DNS | `mg-endpoint-ae703e35-8c45-40fd-9ee1-6ce7515d3ddf.asia-south1-770546093480.prediction.vertexai.goog` |
| Region | `asia-south1` · Status Active · traffic 100% · Monitoring Disabled |
| Deployed model | `medgemma-27b-it-**s2z**` (Version 1), id `9081493255429816320`, created Jul 24 2026 11:45 |
| Registry model | `google_medgemma-27b-it-1784814129711` (imported Jul 23 2026 19:14) |
| Replicas | **Auto (0 minimum, 1 maximum)** → **scale-to-zero ENABLED** |

```
image: …/pytorch-vllm-serve:20250430_0916_RC00_maas          ← Apr 2025
python -m vllm.entrypoints.api_server --host=0.0.0.0 --port=8080
  --model=gs://vertex-model-garden-restricted-us/medgemma/medgemma-27b-it
  --tensor-parallel-size=4  --swap-space=16  --max-model-len=24000
  --gpu-memory-utilization=0.95  --max-num-seqs=4
  --limit-mm-per-prompt='image=5'  --disable-log-stats
env: MODEL_ID=google/medgemma-27b-it · DEPLOY_SOURCE=API_NATIVE_MODEL · **VLLM_USE_V1=0**
inference route /generate · health /ping · shared memory 0 B
```

**GPU — derived from logs:** 4 worker ranks (pids 311/312/313 + main), each
`model weights 13.17 GiB + non_torch 0.17 + activations 3.18 + KV 4.42 ≈ 20.9 GiB`
→ ~22 GiB usable at util 0.95 → 24 GB card → **4 × NVIDIA L4**, ~`g2-standard-48`.
13.17 × 4 = 52.7 GiB = 27B BF16 sharded across 4.

**Problems**
1. **`VLLM_USE_V1=0` → prefix caching OFF.** Biggest loss for our workload.
2. **KV cache only 4.42 GiB/GPU (~17.7 GiB total)** — this is *why* `max-num-seqs=4` and `max-model-len=24000`. The concurrency cap is a consequence of splitting BF16 27B over small cards, not a tunable.
3. Scale-to-zero → cold starts observed in logs (engine init ~73 s on 07-23, 07-24, and 07-28 13:37, plus GCS weight pull).
4. Container ~15 months old.
5. Monitoring disabled; `--disable-log-stats` hides throughput.
6. Endpoint had ≈0 traffic in 30 days → **provisioned but never validated under load**.

**Also in Mumbai (cleanup candidates)**
- Endpoint `ih-google_medgemma-27b-it` (id `6318669024456605696`) — **0 models** since Aug 2025, dead.
- Registry: `ih-google_medgemma-27b-it-1752667642944` (Jul 2025, unused), `ih_google_medgemma-27b-text-it` (Jun 2025, unused — this is the **text-only** variant).

---

## C. Diagnostic learned while testing (keep)

The console "Test your model" failure was **not** a config problem:
```
aiohttp.client_exceptions.ClientResponseError: 403, message='Forbidden',
url='https://upload.wikimedia.org/wikipedia/commons/…/Chest_Xray_PA_3-8-2010.png'
```
Wikimedia blocks non-browser User-Agents; vLLM's image fetcher gets 403. Container was healthy throughout (`GET /ping 200 OK` every 10 s). **Fix: pass images as base64 `data:` URIs**, or host them somewhere that doesn't block bots. Text-only requests work.

Working payload shape (Vertex `:predict`):
```json
{"instances":[{"@requestFormat":"chatCompletions",
  "messages":[{"role":"user","content":[{"type":"text","text":"..."}]}],
  "max_tokens":200}]}
```

---

## D. Target state for MUMBAI (to be specced/approved)

Requirements from the 2026-07-28 session:
1. **Reachable from a client app with API-key-style auth.** ⚠️ Vertex prediction endpoints authenticate with OAuth2/IAM bearer tokens; **API keys are not a supported auth mode for predict**, and shipping a service-account key into a client app would be a credential leak (and a patient-data incident). Expected correct pattern: a **thin backend proxy** (holds the SA, issues/validates *our own* API keys, rate-limits, audit-logs) in front of the endpoint. To be confirmed by research before committing.
2. **OpenAI-compatible API** — options: keep `:predict` + `@requestFormat: chatCompletions`, or run `vllm.entrypoints.openai.api_server` with predict route `/v1/chat/completions` and call via `:rawPredict`. To be confirmed.
3. **FP8 weights + FP8 KV cache** — `--quantization fp8 --kv-cache-dtype fp8`. Native on Blackwell (sm_120); this is exactly the config validated on the molab RTX PRO 6000 prototype.
4. **1 × RTX PRO 6000 in asia-south1** (org reportedly has 1 in Mumbai, ~10–16 in Singapore).
5. Console-only deploy path: **Model Registry → Import model** with custom container image + args → Deploy to endpoint. (Model Garden one-click fixes the args, so import is likely required.)
6. Keep: `min replicas = 1`, V1 engine, monitoring **enabled**, drop `--disable-log-stats`, honest `--max-model-len`.

**Open question:** the weights come from `gs://vertex-model-garden-restricted-us/...` (a **US** bucket). Fine for residency (weights ≠ patient data), but a custom import may need our own copy staged in an **asia-south1** GCS bucket — which also speeds loads and keeps everything in-region.
