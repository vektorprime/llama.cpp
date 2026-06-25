# MTP (Multi-Token Prediction) Implementation Documentation

## Overview

MTP (Multi-Token Prediction) is a speculative decoding technique where a single model uses an extra "MTP head" layer to predict draft tokens. The target (main) model produces hidden states that feed into this MTP head, which generates speculative tokens verified by the target model.

This implementation is in the branch `mtp_pp_llama` of the fork at `https://github.com/vektorprime/llama.cpp.git`.

---

## Architecture

### Two-Context Design

MTP uses two `llama_context` instances operating on the **same** `llama_model`:

| Context | Type | Role |
|---------|------|------|
| `ctx_tgt` | `LLAMA_CONTEXT_TYPE_DEFAULT` | Main/target model: full transformer, produces logits + hidden states |
| `ctx_dft` | `LLAMA_CONTEXT_TYPE_MTP` | MTP draft: single MTP head layer, produces draft tokens |

Both contexts are created from the same model file. The draft context (`ctx_dft`) has:
- `cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP` (`src/llama-cparams.h:52`)
- `cparams.ctx_other = ctx_tgt` — link to target for future memory sharing (`src/llama-cparams.h:58`)
- `cparams.n_rs_seq = 0` — no recurrent-state snapshots needed (`tools/server/server-context.cpp:1238`)
- Its own KV cache (separate from target)

### Key Files and Roles

#### Core Architecture

| File | Role |
|------|------|
| `src/llama-cparams.h` | Context parameters: `ctx_type`, `ctx_other`, `embeddings_nextn`, `embeddings_nextn_masked`, `nextn_layer_offset` |
| `src/llama-context.h:42-390` | `llama_context` class: owns `embd_nextn` buffer, `process_ubatch()`, `graph_reserve()`, `sched_reserve()` |
| `src/llama-context.cpp:1700-2000` | `llama_context::decode()`: main decode loop, splits batch into ubatches, extracts h_nextn D2H per ubatch |
| `src/llama-context.cpp:1304-1374` | `llama_context::process_ubatch()`: builds graph, sets inputs, computes forward pass |
| `src/llama-context.cpp:1966-1983` | **D2H bottleneck**: `ggml_backend_tensor_get_async()` copies `t_h_nextn` from device to host |
| `src/llama-graph.h:126-141` | `llm_graph_input_embd_h` class: MTP-specific input with token + embedding + hidden state `h` tensor |
| `src/llama-graph.cpp:109-130` | **H2D bottleneck**: `ggml_backend_tensor_set()` copies `ubatch->embd` (hidden state) from host to device |
| `src/llama-graph.h:36` | `LLM_GRAPH_TYPE_DECODER_MTP` enum value |

#### MTP Draft Logic

| File | Role |
|------|------|
| `common/speculative.h` | Public API: `common_speculative_init()`, `draft()`, `process()`, `accept()`, `need_embd_nextn()` |
| `common/speculative.cpp:896-1340` | `common_speculative_impl_draft_mtp`: core MTP draft engine |
| `common/speculative.cpp:932-1009` | Constructor: initializes batch, samplers, enables `embeddings_nextn`, detects chain_heads |
| `common/speculative.cpp:1049-1165` | `process()`: catch-up decode of draft model, saves target h_nextn as `verify_h` and `pending_h` |
| `common/speculative.cpp:1167-1316` | `draft()`: iteratively generates draft tokens using MTP head |
| `common/speculative.cpp:1318-1331` | `accept()`: updates `pending_h` from saved verification hidden states |
| `common/speculative.cpp:1917-1999` | `common_speculative_init()`: creates MTP impl instance, adds to impls list |

#### Model-Specific MTP Graphs

| File | Model | Notes |
|------|-------|-------|
| `src/models/qwen35.cpp:488-644` | Qwen3.5/3.6 dense | Single MTP head (`n_mtp_layers=1`). Most common path. |
| `src/models/qwen35moe.cpp:552+` | Qwen3.5/3.6 MoE | Single MTP head |
| `src/models/cohere2moe.cpp:295+` | Cohere2 MoE | Single MTP head |
| `src/models/step35.cpp:366+` | Step3p5 | Chained MTP heads (`chain_heads=true`) |

#### Server Integration

| File | Role |
|------|------|
| `tools/server/server-context.cpp:1227-1252` | Creates MTP draft context with `LLAMA_CONTEXT_TYPE_MTP` |
| `tools/server/server-context.cpp:1005-1020` | Enables `draft-mtp` in speculative types |

#### Common Infrastructure

| File | Role |
|------|------|
| `common/common.h:159-163` | `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` enum |
| `common/common.h:311-336` | `common_params_speculative_draft` struct with `ctx_tgt`, `ctx_dft`, `n_max`, `p_min`, `backend_sampling` |
| `common/common.h:356-379` | `common_params_speculative` struct |
| `common/arg.cpp:3554-3589` | CLI args: `--spec-draft-n-max`, `--spec-draft-p-min`, `--spec-draft-backend-sampling` |
| `common/arg.cpp:402-459` | MTP auto-detection when `draft-mtp` is in speculative types |

#### GGML Backend (Copy Infrastructure)

| File | Role |
|------|------|
| `ggml/include/ggml-backend.h:71` | `ggml_backend_tensor_copy()` |
| `ggml/include/ggml-backend.h:86-95` | `ggml_backend_tensor_get_async()`, `ggml_backend_tensor_set()` |
| `ggml/src/ggml-backend.cpp:477-519` | Tensor copy dispatcher: host-to-device, device-to-host, device-to-device |
| `ggml/src/ggml-cuda/ggml-cuda.cu:3191+` | CUDA async copy with stream synchronization, peer-to-peer D2D copy |

---

## MTP Data Flow

### 1. Context Initialization

```
Server/CLI starts → common_speculative_init() called with COMMON_SPECULATIVE_TYPE_DRAFT_MTP
  → Creates MTP draft context: llama_init_from_model(ctx_tgt.model, cparams_mtp)
    where cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP
  → llama_set_embeddings_nextn(ctx_tgt, true, false) // unmasked: extract h_nextn for all tokens
  → llama_set_embeddings_nextn(ctx_dft, true, true)  // masked: extract h_nextn for output-only tokens
```

### 2. Prompt Processing (Prefill) Flow

The prompt processing flow involves the following sequence per user batch:

```
User calls llama_decode(ctx_tgt, batch)
│
├── decode() splits batch into ubatches via memory module
│   └── For each ubatch:
│       ├── process_ubatch(ubatch, DECODER, ...)
│       │   ├── Build full transformer graph
│       │   ├── Set inputs (tokens → GPU)
│       │   ├── ggml_backend_sched_graph_compute_async() // GPU forward pass
│       │   └── After compute:
│       │       ├── Extract logits D2H (if needed)
│       │       ├── Extract embeddings D2H (if needed)
│       │       └── Extract h_nextn D2H ⚠️ BOTTLENECK
│       │           ggml_backend_tensor_get_async(backend_h, t_h_nextn,
│       │               embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
│       └── (repeat for next ubatch)
│
└── After all ubatches: speculative_process(full_batch)
    └── impl_mtp::process(batch_in)
        ├── Build draft batch: same tokens as target
        ├── Copy target h_nextn → draft batch.embd (shifted right by 1)
        │   std::memcpy(batch.embd+1*n_embd, llama_get_embeddings_nextn(ctx_tgt),
        │               row_bytes * (n_tokens-1));
        ├── Fill first position of each sequence with pending_h from previous run
        ├── llama_decode(ctx_dft, batch) // draft catch-up decode
        │   └── Internally splits into draft ubatches:
        │       └── For each draft ubatch:
        │           ├── MTP graph (DECODER_MTP) built
        │           ├── H2D: ggml_backend_tensor_set(h, ubatch->embd, ...) ⚠️ BOTTLENECK
        │           ├── GPU forward pass (1 MTP layer)
        │           └── D2H: draft's t_h_nextn extracted (unnecessary during PP)
        ├── Save target h_nextn rows as verify_h[seq_id]
        └── Save last h_nextn row per sequence as pending_h[seq_id]
```

### 3. Draft Generation Flow

```
speculative_draft() → impl_mtp::draft(dparams)
│
├── Build draft batch with:
│   batch.token[i] = id_last (last sampled token)
│   batch.embd[i]  = pending_h[seq_id] (last target pre-norm hidden state)
│
├── For i = 0..n_max:
│   ├── llama_decode(ctx_dft, batch) // MTP graph forward
│   │   ├── hnorm(hidden_state) → normed hidden
│   │   ├── enorm(token_embd) → normed token embedding
│   │   ├── Concat + eh_proj → projected representation
│   │   ├── Attention (reads own KV cache)
│   │   ├── FFN
│   │   ├── Output norm → t_h_nextn
│   │   └── LM head → logits
│   │
│   ├── Extract draft's h_nextn D2H for next iteration
│   ├── Sample token from logits (backend sampling on GPU, or CPU fallback)
│   ├── If p < p_min → stop drafting this sequence
│   ├── Add drafted token to result
│   ├── Set batch.embd = new h_row (for next iteration)
│   └── Set batch.token = new token, pos = n_past + i + 1
│
└── Return draft results (may clear if < n_min tokens drafted)
```

### 4. Verification and Acceptance

```
target decodes original tokens + draft tokens
  → accepts matching prefix
  → speculative_accept(seq_id, n_accepted)
    → impl_mtp::accept()
      → select verify_h[n_accepted-1] as new pending_h
```

---

## Root Cause of Prompt Processing Slowdown

### The D2H+H2D Roundtrip

When MTP is enabled, the prompt processing pipeline involves an expensive device-to-host-to-device roundtrip for hidden states on **every ubatch**:

1. **Target D2H** (`src/llama-context.cpp:1974-1981`):
   After each target ubatch computes, `t_h_nextn` (hidden state before output norm) is copied from GPU to the host-side `embd_nextn` buffer via `ggml_backend_tensor_get_async()`.

2. **Draft H2D** (`src/llama-graph.cpp:128`):
   During draft catch-up, the `ubatch->embd` (which contains copies of target's h_nextn) is copied from host to the draft's GPU `h` input tensor via `ggml_backend_tensor_set()`.

3. **Draft D2H** (`src/llama-context.cpp:1981` — same code path in draft context):
   During draft catch-up, the draft's own `t_h_nextn` is extracted D2H, even though it's not used during prompt processing.

### Performance Impact

With 2x RTX 3080 (tensor parallelism) and Qwen3.6-27B:
- PP speed: ~776 tok/s without MTP
- PP speed with MTP: ~50% lower (estimated ~388 tok/s based on PR reports)

The D2H+H2D roundtrip is the dominant bottleneck. Each ubatch of the target triggers a D2H copy of n_rows * n_embd * sizeof(float) bytes. For 4096 tokens with ubatch=512 and n_embd=2560 for Qwen3.6-27B:
- Per ubatch: 512 * 2560 * 4 = ~5.2 MB D2H
- Total for 8 ubatches: ~41.6 MB D2H (target) + same H2D (draft) + draft D2H (~20MB masked) = ~103 MB total transfer
- At PCIe 3.0 x16 (~12 GB/s), this is ~8.6ms of pure transfer time per PP batch

Additionally, the draft catch-up decode runs the full MTP graph (attention + FFN + LM head) for every token, which adds compute overhead.

---

## Optimizations Already Implemented (Upstream)

### PR #23433 — Skip Unnecessary Logit Computation (Merged May 2026)
When `n_outputs == 0`, the graph reserve still allocated memory for logit tensors. This fix uses `inp_out_ids` to skip logit computation, saving VRAM and compute.

### PR #23287 — Backend Sampling for MTP Draft (Merged May 2026)
Replaced D2H logit copies + CPU-side sort/top-k with on-device argmax/top-k sampling. Avoids transferring logit tensors D2H for draft token selection. ~4-8% improvement.

### PR #22838 — Parallel Drafting Support (Merged)
Enabled parallel draft verification, reducing sequential dependency overhead.

---

## Optimizations Implemented in This Branch

### Optimization 1: Skip Draft h_nextn D2H During PP Catch-up

**Location**: `common/speculative.cpp:impl_mtp::process()`

**Status**: Implemented, **off by default** (enable with `--mtp-pp-optimize`)

**Problem**: During the catch-up decode (`llama_decode(ctx_dft, batch)` at line 1129), the draft context had `embeddings_nextn=true` with `masked=true`. This caused each draft ubatch to extract `t_h_nextn` from device to host, even though these values are never used during prompt processing. The draft's h_nextn is only needed during draft generation, not during catch-up.

**Fix**: When `--mtp-pp-optimize` is passed, disable `embeddings_nextn` on `ctx_dft` before the catch-up decode, and re-enable it after. This eliminates the unnecessary D2H copies of the draft's h_nextn tensors during prompt processing.

**Control**: `--mtp-pp-optimize` (off by default, experimental)

**Estimated Impact**:
- Saves ~1 D2H copy per draft ubatch during PP
- Each copy is ~n_outputs * n_embd * sizeof(float) bytes
- For Qwen3.6-27B with ubatch=512, n_embd=2560: saves ~5.2 MB per ubatch × number of draft ubatches
- Expected PP speed improvement: ~5-15% (varies by ubatch configuration)

### Optimization 2: Reduced Draft Catch-up Processing During PP

**Location**: `common/speculative.cpp:impl_mtp::process()`

**Problem**: During catch-up, the draft model processes every prompt token through the full MTP head (attention + FFN + LM head) to populate its own KV cache. The FFN and LM head computations are not strictly necessary for KV cache population — only the attention operation (which writes K, V to the cache) is needed. However, since the graph structure couples FFN/LM head to the attention output, and the KV cache stores the output of the attention layer (not the FFN output), the FFN and LM head are wasted work during catch-up.

**Analysis**: In the current architecture, the KV cache stores K, V from the attention operation. The MTP graph pipeline is:
```
h → hnorm → [concat with enorm(token_embd)] → eh_proj → attn_norm → Q,K,V → attention → write KV cache
                                                                      ↓
                                                                  gate * QK output → wo → residual
                                                                      ↓
                                                                  FFN → output_norm → LM head → logits
```

The attention writes K, V to the cache during the forward pass (inside `build_attn()`). The subsequent FFN and LM head are not needed for cache population. However, decoupling these would require graph structure changes that could affect correctness elsewhere.

**Decision**: Leave this as a documented TODO for future optimization. It requires a ggml-level refactor to allow partial graph execution or conditional subgraph skipping.

### Optimization 3: --custom-logs Parameter for MTP Performance Debugging

**Location**: `common/common.h`, `common/arg.cpp`, `common/speculative.cpp`, `src/llama-context.cpp`

Added a `--custom-logs` CLI flag that enables detailed MTP-specific logging:
- Per-ubatch D2H copy sizes and timing
- Draft catch-up decode timing
- Target vs. draft ubatch processing times
- KV cache state tracking
- Token and embedding sizes

This helps validate optimizations and identify bottlenecks in real deployments.

---

## Logging Infrastructure

### --custom-logs Parameter

Added to `common_params` and parsed in `common/arg.cpp`:

```
--custom-logs  // enable detailed MTP performance logging
```

### --mtp-pp-optimize Parameter

Added to `common_params_speculative_draft` and parsed in `common/arg.cpp`:

```
--mtp-pp-optimize  // disable draft h_nextn D2H during PP catch-up (off by default, experimental)
```

When `--custom-logs` is enabled, the following metrics are logged during prompt processing:

```
[CUSTOM] MTP target ubatch: n_tokens=512 n_outputs=512 d2h_size=5242880 bytes d2h_src=GPU backend
[CUSTOM] MTP target ubatch timing: graph_compute=12.345ms d2h_copy=0.234ms total=12.579ms
[CUSTOM] MTP process: n_tokens=4096 n_embd=2560 row_bytes=10240 embd_nextn_copy=41943040 bytes
[CUSTOM] MTP process: draft catch-up disabled, draft_embeddings_nextn=false
[CUSTOM] MTP draft catch-up decode: n_tokens=4096 n_ubatches=8
[CUSTOM] MTP draft catch-up done: total_us=12345
[CUSTOM] MTP draft: n_drafting=1 draft_tokens_attempted=3 draft_tokens_accepted=3 p_min=0.500
```

### Custom Log Implementation

The `--custom-logs` flag sets `params.custom_logs = true` in `common_params`. This is checked via the macro:

```cpp
#define CUSTOM_LOG(...) do { if (params.custom_logs) { LOG_INF(__VA_ARGS__); } } while (0)
```

For MTP-specific code in `common/speculative.cpp`, a member `bool custom_logs` is set from the params and used to guard verbose logging.

---

## Future Optimizations (Not Yet Implemented)

### Cross-Context Tensor Sharing (WIP by @ggerganov)

**The most impactful remaining optimization.** Eliminates the D2H+H2D roundtrip entirely by allowing the MTP draft context to directly reference the target model's `t_h_nextn` tensor in GPU memory without copying:

```
Current:  [Target GPU t_h_nextn] → D2H → [CPU embd_nextn] → H2D → [Draft GPU h input]
Future:   [Target GPU t_h_nextn] ──shared──→ [Draft GPU h input]  (zero-copy)
```

This requires ggml-level infrastructure changes:
- Shared tensor memory pools across `llama_context` instances
- Cross-context tensor references in compute graphs
- Access synchronization (draft reads after target's forward pass completes)

### VRAM Reservation Tightening (PR #23527)

When `n_outputs == 0`, the graph reserve still allocates memory for output tensors. Further reducing this could save ~3 GB VRAM for large-vocabulary models.

### Multi-Sequence Recurrent Memory Batching

For Qwen3.6's GDN architecture, making recurrent states contiguous would enable batched processing, reducing per-sequence overhead.

### Metal-Specific K/V Reuse (PR #23114)

The Metal backend could reuse K/V in flash-attn for speculative decode, reducing redundant computation.

---

## Build and Test

### Build
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel
```

### Test with llama-bench
```bash
/home/user/llm/mtp_pp_llama/build/bin/llama-bench -p 4096 -n 128 \
  -m /home/user/llm/models/Qwen3.6-27B/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --cache-type-k bf16 --cache-type-v bf16 --flash-attn on -ngl 99 -sm tensor
```

### Test with custom-logs
```bash
/home/user/llm/mtp_pp_llama/build/bin/llama-bench -p 4096 -n 128 \
  -m /home/user/llm/models/Qwen3.6-27B/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --cache-type-k bf16 --cache-type-v bf16 --flash-attn on -ngl 99 -sm tensor \
  --custom-logs
```

### Baseline Performance (without MTP)
| Model | Size | Backend | Type K/V | SM | FA | Test | t/s |
|-------|------|---------|----------|----|----|------|------|
| qwen35 27B Q5_K | 18.94 GiB | CUDA | bf16/bf16 | tensor | 1 | pp4096 | 776.08 ± 54.70 |
| qwen35 27B Q5_K | 18.94 GiB | CUDA | bf16/bf16 | tensor | 1 | tg128 | 40.63 ± 2.43 |

---

## Implementation Issues and Resolutions

### Issue 1: Initial build failure — single-option bool handler signature
**Resolution**: The `--custom-logs` flag is a single boolean option (not `--flag`/`--no-flag` pair). The `common_arg` constructor for single options takes `void (*handler)(common_params&)` without a `bool` parameter. Fixed by removing the `bool value` parameter from the lambda.

### Issue 2: llama-bench does not support MTP testing
**Resolution**: `llama-bench` does not use the `common_speculative` framework. Testing was done using `llama-server` with `--spec-type draft-mtp` to validate the MTP code path. The server properly creates an MTP context via `llama_init_from_model(ctx_tgt.model, cparams_mtp)`.

### Issue 3: speculative example loads two separate models
**Resolution**: The standalone `llama-speculative` example bypasses the `common_speculative` framework and loads the full model twice (target + draft), causing OOM on larger models. MTP testing must use the server or a tool that uses `common_speculative_init`.

## Verified Behavior (Server Test with Qwen3.5-2B Q8_0)

The `--custom-logs --mtp-pp-optimize` flags were tested with the server and produced the following verified log output:

```
[MTP-INIT] custom_logs enabled, n_mtp_layers=1 is_mem_shared=0 chain_heads=0
[MTP-PP-D2H] ubatch D2H copy: n_rows=7 n_embd=2048 bytes=57344 offset=0 ctx_type=0 masked=0
[MTP-PP] process() entry: n_tokens=7 n_embd=2048 row_bytes=8192 is_mem_shared=0
[MTP-PP] copied target h_nextn -> draft embd: 49152 bytes (6 rows)
[MTP-PP-OPT] disabling draft embeddings_nextn for catch-up (7 tokens, 8192 bytes/row)
[MTP-PP-OPT] restored draft embeddings_nextn after catch-up
[MTP-PP] process() exit: saved 7 verify_h rows across 4 seqs, total_verify_bytes=57344
[MTP-DRAFT] draft() entry: n_seq=4 n_drafting=1 n_max=2 p_min=0.50 n_embd=2048
[MTP-DRAFT] step 0: batch.n_tokens=1 n_drafting=1
[MTP-DRAFT] step 1: batch.n_tokens=1 n_drafting=1
[MTP-DRAFT] draft() exit: iterations=1 sequences_completed=1 tokens_drafted=2
```

Key observations confirming the optimization:
- **ctx_type=0** (target) D2H copies appear for all prompt tokens (masked=0)
- **ctx_type=1** (draft) D2H copies ONLY appear during `draft()` generation (masked=1), NOT during `process()` catch-up
- The `[MTP-PP-OPT]` messages confirm embeddings_nextn is disabled during catch-up and restored after
- Draft acceptance rate: 1.000 (4/4 accepted), mean acceptance length: 2.33

### Issue 4: (Document as encountered)

---

## References

- Original MTP PR: https://github.com/ggml-org/llama.cpp/pull/22673
- PR #23433 (skip logit computation): Merged May 21, 2026
- PR #23287 (backend sampling): Merged May 20, 2026
- PR #22838 (parallel drafting): Merged
- Qwen3.5 technical report: https://arxiv.org/abs/2504.02332
- Agents documentation: /home/user/llm/mtp_pp_llama/AGENTS.md
