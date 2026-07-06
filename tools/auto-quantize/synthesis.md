# IQ4_XS Auto-Research Synthesis — Session 2026-07-06

## Baseline Established: KL 0.024916 (stock IQ4_XS vs Q8_0 reference)

First experiment (Bartowski stock IQ4_XS quantize) gives us the baseline:

| Metric | Q8_0 Reference | IQ4_XS (stock) |
|--------|---------------|----------------|
| PPL | 6.7917 | 6.8952 |
| KL divergence | 0.0 | **0.0249** |
| Same top p | 100% | **94.17%** |
| RMS Δp | 0.0% | 4.245% |

KL percentile breakdown:
- 99.9%: 2.27 (rare catastrophic outliers)
- 99.0%: 0.227 (1% worst-case)
- Median: 0.007 (typical token)
- 90% of tokens have KL < 0.0365

## Transferable Improvements from IQ2_XXS

| Improvement | Transfers? | Notes |
|-------------|-----------|-------|
| Weight exponent 0.5→0.30 | ✅ Yes | Same `qw * powf(sigma2 + xb^2, p)` formula in IQ4_XS |
| Per-sub-block sigma2 | ✅ Yes | Same 4-sub-block structure |
| Superblock d optimization | ✅ Yes | Same `d` + 4-bit sub-block level encoding |
| Post-d refinement | ⚠️ Partial | Re-evaluate 4-bit codebook entry with quantized scale |
| waux softening | ⚠️ Uncertain | IQ4_XS may have different neighbor/off-map patterns |
| K-means codebook learning | ❌ No | IQ4_XS uses fixed `kvalues_iq4nl`, no learned grid |
| nwant/kmaps | ❌ No | No neighbor search in IQ4_XS |

## Next Steps

1. Establish quantize time baseline (stock IQ4_XS on 27B model)
2. Apply weight exponent 0.30 with per-sub-block sigma2
3. Apply superblock d optimization
4. Apply post-d refinement
