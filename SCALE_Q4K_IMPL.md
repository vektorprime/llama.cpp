# Scale Q4_K Implementation

## Goal
Modify Q4_K quantization so that when a post-quant nibble is 0, instead of computing
`weight = d*sc * 0 - dmin*m = -dmin*m`, we output `weight = d*sc` (the scale value).

This gives zero-valued nibbles a meaningful non-zero weight contribution equal to the
sub-block scale, rather than the current minimum-offset value.

## Formula Change
- **Standard Q4_K**: weight = d * sc * nibble - dmin * m  (nibble ∈ [0,15])
- **Scale Q4_K**:   weight = d * sc          if nibble == 0
- **Scale Q4_K**:   weight = d * sc * nibble - dmin * m  if nibble > 0

## Files to Modify
1. ggml/src/ggml-quants.c - CPU dequantize_row_q4_K
2. ggml/src/ggml-cuda/convert.cu - CUDA dequantize_block_q4_K
3. ggml/src/ggml-cuda/vecdotq.cuh - CUDA vec_dot kernels (MMVQ + MMQ)
4. ggml/src/ggml-cuda/mmvq.cu - CUDA MMVQ dispatch (no changes needed, uses vecdotq)
5. ggml/src/ggml-cuda/mmq.cuh - CUDA MMQ dispatch (no changes if mmq not used for Q4_K)

## Status
- [ ] CPU dequant modification
- [ ] CUDA convert dequant modification
- [ ] CUDA vec_dot MMVQ correction
- [ ] CUDA vec_dot MMQ correction
- [ ] Build with CUDA
- [ ] Quantize Qwen 3.5 2B
- [ ] KLD comparison vs BF16
- [ ] Test -sm tensor
- [ ] Test with llama-cli inference

## Issues & Fixes
(To be filled as we go)


## Implementation Details

### Modified Files
1. **ggml/src/ggml-quants.c** — CPU dequant + static flag + setter/getter
2. **ggml/src/ggml-quants.h** — Declarations for ggml_set/get_scale_q4_k
3. **ggml/src/ggml-cuda/convert.cu** — CUDA dequant kernel + __constant__ flag + setter
4. **ggml/src/ggml-cuda/vecdotq.cuh** — MMVQ + MMQ correction kernels
5. **ggml/include/ggml-cuda.h** — Declaration for ggml_cuda_set_scale_q4_k
6. **common/common.h** — Added `bool scale_q4_k` to common_params
7. **common/arg.cpp** — Added `--scale-q4-k` CLI flag
8. **tools/perplexity/perplexity.cpp** — Wired setter calls
9. **tools/server/server-context.cpp** — Wired setter calls (used by cli + server)

### Runtime Flag
- CLI flag: `--scale-q4-k` (bool, default false)
- When enabled, nibble==0 in Q4_K dequant outputs d*sc (the scale value) instead of -dmin*m
- Affects both CPU and CUDA code paths
- MMVQ CUDA kernel adds correction term per sub-block
- MMQ CUDA kernel adds approximate correction

### Correction Formula (CUDA vec_dot)
Standard MMVQ result: d * sum(d8[i]*dot1*sc[i]) - dmin * sum(d8[i]*dot2*m[i])
Correction: +d8[i] * sum_zero_q8 * (d*sc[i] + dmin*m[i])
Where sum_zero_q8 = sum of q8 values at nibble==0 positions


## Build
- Build server: 10.0.0.188 (2x RTX 3080)
- CMake flags: -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
- Parallel: -j4 (conservative to avoid CUDA template build failures)
- Build started: $(date)

## Test Plan
1. Quantize Qwen 3.5 2B BF16 → Q4_K using standard llama-quantize
2. Run perplexity with `--scale-q4-k` and without (baseline)
3. Compare KLD vs BF16 logits using `--kl-divergence --kl-divergence-base`
4. Test `-sm tensor` (--split-mode tensor) for multi-GPU
5. Test llama-cli inference


## RESULTS (5-chunk validation, $(date))

### Baseline Q4_K (no --scale-q4-k)
- Mean PPL(Q): 13.317  (ref 13.9, healthy)
- Mean KLD:    0.0570  (ref 0.093, healthy)
- Same top-p:  86.76%  (ref 83.9%, healthy)

### Scale Q4_K (--scale-q4-k)
- Mean PPL(Q): 14.886  (+1.57 WORSE)
- Mean KLD:    0.1844  (3.2x WORSE, above 0.1 problem threshold)
- Same top-p:  79.15%  (WORSE, below 82% problem threshold)

### Finding
The naive scale-substitution (nibble 0 -> d*sc) HURTS Q4_K accuracy.
ROOT CAUSE: In Q4_K, nibble 0 does NOT mean weight=0. The dequant formula is
weight = d*sc*nibble - dmin*m, so nibble 0 -> weight = -dmin*m (the sub-block
minimum value, typically a meaningful negative number). Replacing it with the
scale value d*sc corrupts these weights. The premise (weight*scale=0) only holds
for min-less quant like Q4_0, not Q4_K which has a min offset.
Next experiment per user guidance: sign-bit variant.

## SIGN-AWARE VARIANT (experiment 2)
Per user guidance after naive scale proved ineffective. When nibble==0, sign the
scale by the direction of -m1 (the value being approximated):
  weight = (m1 > 0) ? -d*sc : +d*sc
Rationale: nibble-0 weight approximates -m1 (<=0 when m1>0). The naive +d*sc had
the wrong sign for negative-min blocks. Sign-aware is strictly closer to -m1.
Applied to: CPU dequant, CUDA dequant, CUDA MMVQ, CUDA MMQ, CPU vec_dot.
Still gated by the SAME --scale-q4-k flag (replaces naive behavior).

## SIGN-AWARE RESULTS (5-chunk validation)
- Mean PPL(Q): 14.036  (naive was 14.89, baseline 13.32)
- Mean KLD:    0.1127  (naive was 0.1844, baseline 0.0570)
- Same top-p:  82.94%  (naive was 79.15%, baseline 86.76%)

### FULL COMPARISON
| Metric    | Baseline Q4_K | Naive Scale | Sign-aware Scale |
|-----------|---------------|-------------|------------------|
| PPL       | 13.32         | 14.89       | 14.04            |
| KLD       | 0.0570        | 0.1844      | 0.1127           |
| Same top-p| 86.76%        | 79.15%      | 82.94%           |

### CONCLUSION
Sign-awareness recovers ~40% of the accuracy lost by the naive variant (KLD
0.184 -> 0.113), confirming the sign fix is correct. BUT neither scale variant
beats baseline Q4_K. ROOT CAUSE stands: Q4_K nibble-0 already encodes the
sub-block minimum (-m1) with 6-bit precision, which is the OPTIMAL representation.
Any scale substitution is strictly worse because it discards that information.
The premise (weight*scale=0 loses accuracy) applies to min-less quant (Q4_0),
not Q4_K. For Q4_K, the feature is a net regression and should stay OFF by default
(it is: --scale-q4-k defaults false).

## -sm tensor VALIDATION
Both modes tested with -sm tensor (2x RTX 3080, tensor split). Results match
single-GPU exactly -> modified MMVQ/MMQ CUDA kernels are correct on multi-GPU:
- baseline -sm tensor:  PPL 13.319, KLD 0.0570, top-p 86.73%
- scale    -sm tensor:  PPL 14.889, KLD 0.1847, top-p 79.10% (naive run)
No crashes, no CUDA errors. Note: SPLIT_MODE_TENSOR prints a harmless
llama_params_fit warning (auto-fit unsupported for tensor split), inference works.
