# Q4_K_M_CLONE Auto-Research Idea Ledger

## Experiment Index

| Exp | Description | Outcome |
|-----|-------------|---------|
| EXAMPLE | This is an example entry — format reference only | Example — not a real experiment |
| exp-001 | Remove ATTENTION_QKV Q5_K boost for clone — keep Q4_K for QKV tensors | NULL — dead code, not reached |
| exp-002 | Remove Q6_K boost for ATTENTION_WV and FFN_DOWN from clone | REGRESSION |

## exp-003: Symmetric Sub-block Quantization with 8-bit Scales (SSQ-8)

**Hypothesis:** The Q4_K_M_CLONE block stores 14 bytes of scale metadata (2B d + 2B dmin + 12B 6-bit packed scales+mins) for asymmetric per-sub-block quantization. However, within small sub-blocks of 32 elements, weight distributions are approximately zero-centered, making the min offsets mostly redundant. By switching to symmetric quantization (no mins/dmin) with 8-bit int8 per-sub-block scales, we can: (1) remove dmin (save 2B), (2) replace 12B 6-bit packed scales with 8 plain int8 scales (save 4B), keeping 128B qs unchanged. Total: 144→138 bytes (4.17% reduction ≈ 21 MB for whole model). The 8-bit scales (256 levels vs 64) provide finer granularity to partially compensate for the loss of per-sub-block min offset. Research on K-quant formats shows 8-bit scales work well for Q6_K (which is symmetric), supporting this approach.

**Changes:**
1. `ggml/src/ggml-common.h`: Change `block_q4_K_M_CLONE` struct — remove `dmin`, change scales to `uint8_t scales[8]`. Total: 2+8+128=138 bytes. Update static_assert.
2. `ggml/src/ggml-quants.c`: Write brand-new quantize/dequantize functions with symmetric int8-scale quantization (no wrapper calls to Q4_K).
3. `ggml/src/ggml-cuda/convert.cu`: Write new CUDA dequantize kernel `dequantize_block_q4_K_M_CLONE` for 138-byte block. Register in `ggml_get_to_fp16_cuda` and `ggml_get_to_fp32_cuda`.
4. `ggml/src/ggml-cuda/mmq.cu`: Return false from `ggml_cuda_should_use_mmq` for clone type (forces cublas fallback).
5. `ggml/src/ggml-cuda/mmvq.cu`: Return false from `ggml_cuda_should_use_mmvq` for clone type.
6. `ggml/src/ggml-cuda/common.cuh`: Update `ggml_cuda_type_traits` — qr and qi may change.
7. `ggml/src/ggml-cpu/quants.c`: Update `vec_dot` to handle new scale encoding or add new function.
8. `ggml/src/ggml-cpu/quants.h`: Add new declaration if needed.

**Expected outcome:** GGUF size reduces by ~21 MB (4.17%). KLD may increase slightly due to loss of asymmetric min offsets, but 2x finer scale quantization (8-bit vs 6-bit) should largely compensate. Same top p should stay near baseline.

## NOTE

The entry above is an **example** only. It is not a real experiment.
Use it as a formatting reference for future entries.

---

## exp-001: Remove ATTENTION_QKV Q5_K Boost for Clone

**Hypothesis:** The ATTENTION_QKV tensors are currently boosted to Q5_K (5.5 bpw)
for the clone, same as stock Q4_K_M. Removing this boost and keeping QKV at the
default Q4_K (4.5 bpw) for the clone should reduce GGUF file size while
maintaining KLD ≤ 0.062947 and same top p ≥ 86.387%. QKV tensors are intermediate
in size (not as large as FFN, not as small as output), so the size reduction
should be modest but measurable.

**Changes:**
1. `src/llama-quant.cpp` line 646: Remove `ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE`
   from the condition that boosts QKV tensors to Q5_K, so only stock Q4_K_M gets
   the boost and the clone stays at default Q4_K.

**Expected outcome:** GGUF size reduces by ~2-5 MB. KLD may increase slightly but
should stay below 0.062947. Same top p should remain near baseline.

**Actual outcome:** NULL — size unchanged (529,297,440 bytes = baseline). The ATTENTION_QKV code path at line 642 is dead code: `category_is_attn_v()` (line 162) catches `ATTENTION_QKV` first, so the explicit ATTENTION_QKV block is never reached. The actual QKV boost happens via the ATTENTION_WV path at line 543 which boosts to Q6_K for some layers.

**Lesson:** The `category_is_attn_v()` function includes `ATTENTION_QKV` in its check (line 164), meaning all fused QKV tensors are handled by the V-branch boost logic, not the QKV-specific branch. To affect QKV tensor quantization, changes must target the `category_is_attn_v` path (line 520-557), NOT the ATTENTION_QKV path at line 642-647 (which is unreachable).
---

## exp-002: Remove Q6_K Boost for ATTENTION_WV and FFN_DOWN from Clone

**Hypothesis:** The Q4_K_M_CLONE currently inherits the Q6_K boost for ATTENTION_WV (line 543) and FFN_DOWN (line 599) tensors from stock Q4_K_M. These boosts promote ~50% of WV and FFN_DOWN layers from Q4_K (4.5 bpw) to Q6_K (6.5625 bpw). Removing these boosts should reduce GGUF size by ~10-15 MB while maintaining quality metrics. The stock Q4_K_M boosted these tensors as a grace measure; the clone should survive at pure Q4_K on all tensors.

**Changes:**
1. `src/llama-quant.cpp` line 543: Remove `ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE` from ATTENTION_WV Q6_K boost condition
2. `src/llama-quant.cpp` line 599: Remove `ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE` from FFN_DOWN Q6_K boost condition

**Expected outcome:** GGUF size reduces by 10-15 MB. KLD may increase slightly but should stay ≤ 0.062947. Same top p should remain ≥ 86.387%.

**Actual outcome:** REGRESSION — size reduced from 529,297,440 to 501,452,832 bytes (~27.8 MB, 5.3% reduction), but metrics degraded:
- GGUF size: 501,452,832 bytes (vs baseline 529,297,440 bytes)
- KLD: 0.073436 (vs baseline 0.062947) — exceeded threshold by 16.7%
- Same top p: 85.483% (vs baseline 86.387%) — below threshold by 0.904pp
- RMS Δp: 6.165% (vs baseline 5.753%)

**Lesson:** The Q6_K boost for ATTENTION_WV and FFN_DOWN tensors in Q4_K_M is NOT a "grace" boost — it's essential for maintaining quality. Removing it causes significant KLD increase (+16.7%) and same top p drop. The WV (attention value) and FFN_DOWN tensors are critical to model accuracy. Future experiments should focus on block-level struct compression (reducing qs or scales bytes) rather than removing per-tensor quality boosts, as the latter has too large a quality impact.

---

## exp-E: EXAMPLE — Format Reference

**Hypothesis:** Brief description of what you think will happen and why.
Reference prior findings or code analysis.

**Changes:**
1. Step-by-step list of code changes made
2. Include file paths and approximate line numbers
3. Include key code snippets in ```c blocks

**Expected outcome:** What you predict (e.g., "KL improvement to 0.030"),
with rationale. Include expected quantize time impact.

**Actual outcome:** SUCCESS / FAILED / REGRESSION / NULL — with metrics:
- GGUF size: X MB (vs baseline Y MB)
- KLD: X (vs baseline Y)
- Same top p: X% (vs baseline Y%)

**Lesson:** What was learned. Why did it work or fail? What should be tried
next? What does this reveal about the problem?
