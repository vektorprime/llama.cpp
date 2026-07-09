# Q4_K_M_CLONE Auto-Research Idea Ledger

## Experiment Index

| Exp | Description | Outcome |
|-----|-------------|---------|
| EXAMPLE | This is an example entry — format reference only | Example — not a real experiment |
| exp-001 | Remove ATTENTION_QKV Q5_K boost for clone — keep Q4_K for QKV tensors | NULL — dead code, not reached |
| exp-002 | Remove Q6_K boost for ATTENTION_WV and FFN_DOWN from clone | REGRESSION |
| exp-003 | Symmetric sub-block quantization with 8-bit scales (remove dmin) | REGRESSION |
| exp-004 | Reduce scale/min from 6+6b to 5+3b per sub-block (8-byte scales, 140-byte block) | REGRESSION |
| exp-005 | Dual-anchor DPCM delta encoding of scale-min pairs (10-byte scales, 142B block) | REGRESSION |

## exp-004: Scale-Min Differential Pulse Code Modulation (SM-DPCM)

**Hypothesis:** Within a super-block of 256 elements, adjacent sub-blocks (of 32 elements each) have highly correlated scale and min values because weight statistics change gradually. The current format stores 8 independent 6-bit scale/min pairs packed into 12 bytes. By encoding the first sub-block's scale/min as full 6-bit values (12 bits) and the remaining 7 sub-blocks' scales and mins as 4-bit signed deltas from the base values (-8 to +7 range), we compress the scales[] array from 12 bytes (96 bits) to 9 bytes (72 bits to fit 68 bits). This saves 3 bytes per 144-byte superblock = 2.08% reduction ≈ 11 MB for the whole model.

Key premise: sub-block scale/min values within a super-block typically vary by ≤7 from each other, so 4-bit deltas capture nearly all cases. Clamping overflow introduces minor loss but preserves the full asymmetric quantization framework (d + dmin + per-sub-block mins intact) unlike exp-003. The 4-bit weight data (qs[]) and fp16 superblock scales (d, dmin) remain unchanged.

**Changes:**
1. `ggml/src/ggml-common.h`: Define `K_SCALE_SIZE_CLONE = 9`, change block struct to use it, update static_assert
2. `ggml/src/ggml-quants.c`: Write new quantize/dequantize/quantize_q functions with DPCM scale encoding (no wrapper calls to Q4_K)
3. `ggml/src/ggml.c` + `ggml-cpu/ggml-cpu.c`: Update type_traits type_size (auto-computed from sizeof)
4. `ggml/src/ggml-cpu/quants.c`: Update dispatcher wrapper
5. `ggml/src/ggml-cuda/convert.cu`: Write new dequant kernel with DPCM scale decode; register in dequant dispatch
6. `ggml/src/ggml-cuda/common.cuh`: Update type_traits
7. `ggml/src/ggml-cuda/mmq.cu`: Return false from should_use_mmq for clone type
8. `ggml/src/ggml-cuda/mmvq.cu`: Return false from should_use_mmvq for clone type

**Expected outcome:** GGUF size reduces by ~11 MB (2.08%). The delta encoding may introduce small sub-block scale errors from clamping but should be acceptable. KLD should stay near baseline; same top p should remain ≥ 86.387%.

**Actual outcome:** REGRESSION — size reduced from 529,297,440 to 523,209,760 bytes (~6.1 MB, 1.15% reduction), but quality degraded:
- KLD: 0.077500 (vs baseline 0.062947) — 23.1% increase, far above threshold
- Same top p: 85.427% (vs baseline 86.387%) — below threshold by 0.96pp
- RMS Δp: 6.380% (vs baseline 5.753%)
- PPL: 23.568 (vs baseline 22.450) — 5.0% worse

**Lesson:** Reducing sub-block scale precision from 6 to 5 bits and min precision from 6 to 3 bits causes significant quality loss despite preserving the asymmetric framework. The 3-bit mins (8 levels vs 64) are too coarse to provide adequate sub-block grid centering. While not as catastrophic as removing mins entirely (exp-003), the precision loss is still unacceptable. The scales[] compression approach is valid but needs a different encoding scheme that better preserves precision — e.g., delta encoding (DPCM) that keeps the range but reduces entropy, or vector quantization of the scale/min pairs.

---
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

**Actual outcome:** REGRESSION — size reduced from 529,297,440 to 520,165,920 bytes (~9 MB, 1.72% reduction), but quality degraded severely:
- KLD: 0.114585 (vs baseline 0.062947) — 82% increase, far above threshold
- Same top p: 82.501% (vs baseline 86.387%) — below threshold by 3.89pp
- RMS Δp: 7.885% (vs baseline 5.753%)
- PPL: 24.893 (vs baseline 22.450) — 10.9% worse

**Lesson:** Removing per-sub-block min offsets (going symmetric) causes catastrophic quality loss. Asymmetry in Q4_K is essential at 4 bits — the mins provide sub-block grid centering that cannot be recovered by finer scale granularity. Scale precision (8-bit vs 6-bit) doesn't compensate for loss of grid offset freedom. Future experiments should preserve dmin + per-sub-block mins while finding other compression angles (scale-min correlation encoding, fewer bits per min scale, or mixed sub-block precision).

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

## exp-005: Dual-Anchor DPCM Delta Encoding of Scale-Min Pairs

**Hypothesis:** The Q4_K_M_CLONE block stores 8 independent (6-bit scale, 6-bit min) pairs in 12 bytes (96 bits). Adjacent sub-blocks within a 256-element superblock have correlated scale and min values because weight statistics change gradually across rows. Instead of storing all 8 pairs independently, we use Dual-Anchor Differential Pulse Code Modulation (DPCM):

- **Two anchors** at sub-blocks 0 and 4 (full 6+6 bits each = 24 bits)
- **Six deltas** for sub-blocks 1-3 and 5-7 (4-bit signed scale delta + 4-bit signed min delta = 8 bits each, 48 bits total)
- **Total: 72 bits = 9 bytes** (3-byte savings, 25% compression of scales[], 2.08% of block)

Dual anchors halve the maximum DPCM chain length from 7 to 3 steps, bounding cumulative delta error to ≤21 per variable (3 × ±7). Anchor reinjection at the midpoint prevents error cascading across the full superblock.

**Recovery method:** During dequant, unpack the two anchor bases and six deltas. Reconstruct each sub-block's scale and min via DPCM accumulation from the nearest anchor, with clamping to [0,63]. When the actual delta ≤ ±7, the encoding is lossless for that pair. Only outlier sub-blocks with larger inter-sub-block changes experience clamping.

**Key difference from exp-004's 5+3 approach:** This preserves 6-bit precision for ALL non-clamped sub-blocks. The delta width (4-bit = ±7) is chosen to match expected inter-sub-block variation. Unlike simple precision reduction, most information is preserved through reconstruction.

**Changes:**
1. `ggml/src/ggml-common.h`: Add `K_SCALE_SIZE_CLONE = 9`, change scales[] size, update static_assert
2. `ggml/src/ggml-quants.c`: Write new quantize/dequantize/quantize_q functions with DPCM scale encoding (no more thin wrappers)
3. `ggml/src/ggml-cuda/convert.cu`: Write new CUDA dequant kernel with DPCM scale decode; register in dispatch
4. `ggml/src/ggml-cuda/mmq.cu`: Return false from should_use_mmq for clone type
5. `ggml/src/ggml-cuda/mmvq.cu`: Return false from should_use_mmvq for clone type
6. `ggml/src/ggml-cpu/ggml-cpu.c`: type_traits auto-update from sizeof
7. `ggml/src/ggml.c`: type_traits auto-update from sizeof

**Expected outcome:** GGUF size reduces by ~11 MB (2.08%). KLD should stay near baseline 0.062947; same top p ≥ 86.387%. Delta clamping errors should be minimal because adjacent sub-blocks within a superblock have highly correlated statistics.

**Actual outcome:** REGRESSION — size reduced from 529,297,440 to 526,253,600 bytes (~3 MB, 0.57% reduction), but quality degraded:
- KLD mean: 0.180095 (vs baseline 0.062947) — 186% increase, far above threshold
- Same top p: 78.472% (vs baseline 86.387%) — below threshold by 7.92pp
- RMS Δp: 10.060% (vs baseline 5.753%)
- PPL: 25.380 (vs baseline 22.450) — 13.1% worse

**Lesson:** DPCM delta encoding compounds errors across chains, even with dual anchors (max 3-step chain). The inter-sub-block scale/min deltas within a superblock are sometimes larger than the 4-bit range (-8 to +7), causing clamping. Furthermore, the error accumulates: if subblock 1's delta is clamped, subblocks 2-3 are decoded from an incorrect base, amplifying the error. The correlation between adjacent sub-blocks is not strong enough to overcome the precision loss from delta encoding. For scale/min compression to work with DPCM, delta ranges of at least 5-6 bits are needed, which limits the compression achievable. Future directions should explore non-DPCM compression methods: codebook-based vector quantization of scale-min pairs, or exploiting the scale-min correlation within the SAME sub-block (delta-encode m from sc rather than from previous m).

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
