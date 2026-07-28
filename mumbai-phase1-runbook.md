# Mumbai Phase 1 runbook — MedGemma 27B multimodal on Vertex (Agent Console)

> # ⛔ STALE TARGET — READ `gpu-region-cost-perf.md` FIRST
> **Mumbai + RTX PRO 6000 is impossible.** A real deploy attempt failed with
> `Machine type "g4-standard-48" is not supported.` — Vertex does not serve the `g4` family in `asia-south1`.
> The `RTX Pro 6000 = 1` quota row for Mumbai is **meaningless** (quota ≠ availability).
> A single H100 (`a3-highgpu-*`) **does not exist in either Indian region** either.
> **Likely correct target: `asia-south2` (Delhi) + 1 × RTX PRO 6000** (quota 0 → needs a QIR).
> Everything below about the **container spec, args, console fields, proxy and verification steps remains valid** —
> only the region/GPU choice changed. Swap `asia-south1` → `asia-south2` once the QIR lands.

**Goal:** a residency-compliant, OpenAI-compatible, prefix-caching MedGemma endpoint on **1 × RTX PRO 6000**, deployed entirely through the Agent Console, with **scale-to-zero** + a keep-warm schedule.

**Phase 1 is BF16 — deliberately no quantization.** We measured 35.31 GiB of KV cache available for BF16 27B on this exact GPU in Singapore, so BF16 fits fine. FP8 is Phase 2 (a throughput/memory optimization, not a prerequisite). This gets a correct endpoint live without any GPU-side work.

Target project `ih-medpalm`, region `asia-south1`. Quota check (2026-07-28): `Custom model serving Nvidia RTX Pro 6000 GPUs per region` = **1** in asia-south1. Exactly one GPU — do not plan multi-replica.

---

## Step 0 — Two prerequisites

### 0a. Container image must be in Artifact Registry (asia-south1)
The console import screen states it plainly: *"Must be stored in Artifact Registry."* Docker Hub URIs are rejected.

**Lazy path (no build, no 10 GB push from India):** create an **Artifact Registry remote repository**, Docker format, in `asia-south1`, upstream = Docker Hub. Then reference:
```
asia-south1-docker.pkg.dev/ih-medpalm/<remote-repo>/vllm/vllm-openai:<PINNED_TAG>
```
AR pulls through and caches in-region — satisfies both the AR requirement and residency.
⚠️ **Verify** Vertex pulls cleanly through a remote repo (it's a normal AR path, so it should, but this is untested here). Fallback: `gcloud builds submit` / `docker pull && push` into a standard AR repo.

**Use upstream `vllm/vllm-openai`, NOT Google's `pytorch-vllm-serve`.** Google's is a fork behind nginx, its Model Garden configs pin `VLLM_USE_V1=0` (which kills prefix caching), and `--quantization`/`--kv-cache-dtype` are unvalidated on it.

### 0b. Weights must be staged — TESTED 2026-07-28, restricted bucket is NOT readable
Ran in Cloud Shell as `llm@intelehealth.org`:
```
$ gsutil ls gs://vertex-model-garden-restricted-us/medgemma/
AccessDeniedException: 403 llm@intelehealth.org does not have storage.objects.list
access to the Google Cloud Storage bucket 'vertex-model-garden-restricted-us'
```
→ **Only Google's managed one-click deploy can read that bucket. We must host the weights ourselves.**
(Tested with the human identity; the Vertex deploy SA is a different principal, but an SA in this project is not going to have access the project admin lacks.)

**Environment facts confirmed the same session:**
| Fact | Value | Consequence |
|---|---|---|
| Cloud Shell `$HOME` disk | **4.8 GB** | ❌ cannot stage a ~54 GB model in Cloud Shell |
| `gs://ih-training-data` | **multi-region, US** | ❌ unusable — wrong continent *and* multi-region |
| `gs://cloud-ai-platform-c7f8289e-…` | Vertex-managed staging | not for this |
| Artifact Registry API | **DISABLED** (`SERVICE_DISABLED`) | must be enabled before any AR repo exists |

**So Phase 1 needs a staging job.** Recommended: **Cloud Build** — its default worker has ~100 GB disk and fast egress, it's a managed service (no Compute Engine), and it's one command from Cloud Shell.
```yaml
# cloudbuild.yaml  (timeout must be raised; default 10 min is not enough)
timeout: 3600s
steps:
  - name: python:3.12
    entrypoint: bash
    args:
      - -c
      - |
        pip install -q huggingface_hub
        HF_TOKEN=$$HF_TOKEN HF_XET_HIGH_PERFORMANCE=1 \
          hf download google/medgemma-27b-it --local-dir /workspace/m
        gsutil -m cp -r /workspace/m/* gs://$_BUCKET/medgemma-27b-it/
    secretEnv: ['HF_TOKEN']
```
Alternatives: a Vertex AI custom job (also console-allowed, more setup), or the molab RTX PRO 6000 box (fast, but needs a service-account key pushed to a third-party sandbox — **avoid**).

⚠️ The destination **must be a single-region `asia-south1` bucket**. Multi-region `asia` does **not** count as a residency match.

---

## Step 1 — Model Registry → Import (console)

**Models → Import → step 2 → "Import an existing custom container"**

| Field | Value |
|---|---|
| Name | `medgemma-27b-it-vllm-openai` |
| Region | `asia-south1 (Mumbai)` |
| Container image | `asia-south1-docker.pkg.dev/ih-medpalm/<repo>/vllm/vllm-openai:<PINNED_TAG>` |
| **Command** (3 lines) | `python3` / `-m` / `vllm.entrypoints.openai.api_server` |
| Model artifact location | leave empty (weights come via `--model`) |
| **Inference route** | `/v1/chat/completions` |
| **Health route** | `/health` |
| **Port** | `8080` |
| GRPC ports / Predict schemata | empty |

**Arguments** (one per line — the console textarea takes one flag per line, which avoids gcloud's comma-parsing problem):
```
--model=<GS_URI_OR_HF_ID>
--served-model-name=medgemma-27b-it
--tensor-parallel-size=1
--max-model-len=65536
--gpu-memory-utilization=0.90
--max-num-seqs=16
--max-num-batched-tokens=8192
--enable-prefix-caching
--enable-chunked-prefill
--limit-mm-per-prompt={"image":5}
--swap-space=16
--disable-log-requests
--seed=0
```

**Environment variables:** none required. (Do **not** set `VLLM_USE_V1=0` — V1 is what gives prefix caching.)

**Advanced options → Deployment Configuration:**
| Field | Value | Why |
|---|---|---|
| **Shared memory size** | **`16384`** MB | default is 64 MB; vLLM needs GBs |
| **Startup probe** — Command | `/bin/sh`,`-c`,`curl -f http://localhost:8080/health` | stops the health prober killing the container during the slow 27B load |
| Startup probe — Period / Timeout | `30` / `10` | generous; there is no separate deployment-timeout field, the startup probe is the mechanism |
| Custom health probe | leave default (10 s period / 180 s timeout) | |

⚠️ **All container fields are immutable.** Any arg change = new model version + redeploy. Expect to iterate.

---

## Step 2 — Deploy to endpoint (console)

| Setting | Value |
|---|---|
| Accelerator | **1 × NVIDIA RTX PRO 6000** (`g4-standard-48`) |
| Machine region | `asia-south1` |
| **Min replicas** | **0** ← scale-to-zero, per decision below |
| **Max replicas** | **1** (only 1 GPU of quota; more would just fail) |
| **Enable dedicated DNS** | **ON** — needed for 10 MB payloads, SSE streaming, and a configurable timeout |
| **Inference timeout** | **set HIGH at creation** (e.g. 3600 s) |
| Model Monitoring | **ON** (the existing deployments both have it off) |
| Request-response logging | **OFF**, or pin the BigQuery dataset to `asia-south1` |

⚠️ **Timeout trap:** the endpoint inference timeout defaults to 600 s and **must be set at creation** — changing it later needs `EndpointService.UpdateEndpointLongRunning`; a plain `UpdateEndpoint` silently won't do it.
⚠️ **Residency trap:** request-response logging to BigQuery is the one place real patient data can leave India.

### Scale-to-zero decision (2026-07-28)
**Enabled (min replicas = 0)** for cost — a 24/7 RTX PRO 6000 with sparse traffic is wasteful.

Known costs of this choice:
1. **Cold start ≈ 3–8 min** (image pull + GCS weight load + engine init + CUDA-graph capture). Measured on Singapore: engine init alone **50.05 s**; Mumbai 4×L4 showed **~73 s** engine init. First request after idle waits minutes → this is why the endpoint timeout must be high.
2. **The prefix cache is destroyed on every scale-down.** The shared clinical system prompt gets re-prefilled after each wake, which negates our main optimization during sparse traffic.

**Mitigation — keep-warm schedule (do this):** Cloud Scheduler → proxy → cheap 1-token request every ~10 min, **08:00–20:00 IST, Mon–Fri**. Gives daytime warmth + a live prefix cache, and free nights/weekends. This is the same schedule-based pattern chosen for Shakti, implemented on Vertex.
⚠️ Cloud Scheduler cron is **UTC by default** — 08:00 IST = 02:30 UTC. Set the job timezone to `Asia/Kolkata` explicitly.

---

## Step 3 — Call it (OpenAI-shaped)

Self-deployed models are **not** reachable by the OpenAI SDK's `base_url`. Use `:rawPredict`, which forwards the raw HTTP body untouched, so you send/receive **native OpenAI bodies**:

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://<DEDICATED_DNS>/v1/projects/ih-medpalm/locations/asia-south1/endpoints/<ENDPOINT_ID>:rawPredict" \
  -d '{
    "model": "medgemma-27b-it",
    "messages": [
      {"role":"system","content":"You are an experienced physician. Be concise."},
      {"role":"user","content":"3 days fever, productive cough, right pleuritic chest pain. Most likely dx + 2 differentials."}
    ],
    "max_tokens": 256, "temperature": 0
  }'
```
`:streamRawPredict` for SSE streaming. Response is a normal OpenAI body (`choices[0].message.content`).

**Images:** pass **base64 data URIs**, not URLs. A URL fetch already failed once in production logs:
`ClientResponseError: 403 Forbidden — upload.wikimedia.org/...` (Wikimedia blocks non-browser User-Agents; vLLM's fetcher gets 403).

---

## Step 4 — Auth proxy (required)

**Vertex prediction endpoints cannot be made public or API-key authenticated.** Google's auth doc lists API-key support for exactly two methods — `generateContent` and `streamGenerateContent`. `predict`/`rawPredict` are OAuth2/IAM only. There is no unauthenticated mode in any region.

**Cloud Run** service in `asia-south1` (Cloud Run is not Compute Engine — inside the platform constraint):
- holds the service account (`roles/aiplatform.user`); the Google credential never reaches the client
- accepts **our own** `X-API-Key`; validates, rate-limits (spend cap), audit-logs
- forwards to `:rawPredict`, streams back
- exposes an OpenAI-shaped path so the app can use a standard client

Never ship a service-account key to a client app — it's a long-lived, non-scoped bearer credential, and with patient data in scope that's a breach, not just a bill.

> Note: vLLM's own `--api-key` flag exists but is useless here — Vertex's IAM layer rejects requests before they reach vLLM. On **Shakti**, by contrast, `--api-key` gives native API-key auth with no proxy. Worth remembering when weighing platforms.

---

## Step 5 — Verify before declaring done

1. `/health` green; logs show `EngineCore` + `Capturing CUDA graphs … PIECEWISE` → **V1 engine confirmed** (therefore prefix caching active).
2. Logs show `Available KV cache memory` and `GPU KV cache size: N tokens` → **record N.** If `N < max-model-len`, a single max-length request cannot fit — lower `--max-model-len` to be honest. (Singapore claimed 131072 but had only 66,112 tokens of KV.)
3. Text request via `:rawPredict` returns a sane DDx.
4. Multimodal request with a **base64** image returns a sane read.
5. Second identical request is measurably faster → prefix caching working.
6. Cold-start timing: scale to zero, wait, time the first request. Record it; confirm it's inside the endpoint timeout.
7. Confirm monitoring is on and request-response logging is off (or asia-south1-pinned).

---

## Phase 2 (later) — FP8

1. Quantize `google/medgemma-27b-it` with llm-compressor, RedHat recipe: `scheme=FP8_DYNAMIC`, `targets=Linear`, `ignore=["re:.*lm_head","re:vision_tower.*","re:multi_modal_projector.*"]` → **vision tower stays BF16, so multimodal is unaffected**. Reference recovery on Gemma-3-27B: **99.73%**.
2. Stage to the single-region asia-south1 bucket.
3. Add args: `--quantization fp8_per_tensor --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window`.
   ⚠️ The skip-layers flag is **mandatory** with FP8 KV — Gemma 3 is 5-of-6 sliding-window layers and those are the KV-quant-sensitive ones. Also calibrate KV scales; defaults are `1.0` and degrade output.
4. Expect ~28 GB weights (from ~54 GB) → roughly double the KV headroom.
5. Re-run Step 5 verification, especially a clinical eval — FP8 KV on sliding-window layers is the one accuracy risk worth measuring on our own frozen eval.

---

## vLLM flag reference (what we use vs. deliberately skip)

**Using:** `--tensor-parallel-size 1` (TP>1 reintroduces the shared-memory/NCCL risk), `--enable-prefix-caching`, `--enable-chunked-prefill`, `--max-num-batched-tokens`, `--limit-mm-per-prompt`, `--served-model-name`, `--disable-log-requests` (privacy — otherwise prompts land in Cloud Logging), `--seed`.

**Deliberately NOT set:** `--disable-log-stats` (that's what blinded the current Mumbai deployment — we want throughput metrics), `--enforce-eager` (disables CUDA graphs, big slowdown), `--cpu-offload-gb` (latency), `--speculative-config` (no trained Gemma-3 drafter exists, and spec decode can go net-negative at high concurrency), pipeline/data/expert parallel (single GPU).

**Worth considering later:** `--structured-outputs-config` / guided decoding (force JSON-schema DDx output — genuinely useful for our pipeline), `--scheduling-policy priority`, `--long-prefill-token-threshold` (relevant with a large shared prefix), `--enable-auto-tool-choice` + `--tool-call-parser` (function calling), `--otlp-traces-endpoint`.

⚠️ **Version drift is real:** `--limit-mm-per-prompt` syntax changed (`image=5` → JSON), `--quantization fp8` shorthand became `fp8_per_tensor`, `--kv-cache-dtype-skip-layers` is recent. **Pin a tag and run `vllm serve --help` in that exact container** before deploying — don't trust this list against an unpinned image.

---

## Actions needed before deploy (each creates/modifies a resource — needs sign-off)
1. **Enable `artifactregistry.googleapis.com`** (currently disabled).
2. **Create a single-region `asia-south1` GCS bucket** for weights (e.g. `gs://ih-medgemma-weights-asia-south1`). Existing `ih-training-data` is multi-region US and unusable.
3. **Enable Cloud Build API** + run the staging job to copy ~54 GB of `google/medgemma-27b-it` into that bucket.
4. **Create the Artifact Registry repo** in `asia-south1` (remote → Docker Hub, or standard + mirror `vllm/vllm-openai`).

## Open items
- [x] ~~Can the deploy SA read `gs://vertex-model-garden-restricted-us/...`?~~ **No — 403. Weights must be staged.**
- [ ] Does Vertex pull through an AR **remote** repository?
- [ ] Does the console deploy form expose **min replicas = 0** for a custom-container model? (Model Garden set it on the existing deployment, so s2z is available in this project — but confirm it's offered for custom models)
- [ ] Exact `vllm/vllm-openai` tag + `vllm serve --help` flag verification
- [ ] Record actual `GPU KV cache size` and cold-start duration once live
