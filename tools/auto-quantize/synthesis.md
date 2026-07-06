# IQ4_XS Auto-Research Synthesis — Session 2026-07-06

## Status: Research started. IQ2_XXS research completed at best KL 0.662001 (archived in git history).

## Baseline (to establish)

First experiment: quantize Qwen3.5-2B with stock IQ4_XS, measure KL, establish baseline.

## Known Improvements from IQ2_XXS

The following are proven improvements from the IQ2_XXS research that should
transfer to IQ4_XS since they target shared algorithmic patterns:

| Improvement | Transfers? | Notes |
|-------------|-----------|-------|
| Weight exponent 0.5→0.30 | ✅ Yes | Same `qw * powf(sigma2 + xb^2, p)` formula in IQ4_XS |
| Per-sub-block sigma2 | ✅ Yes | Same 4-sub-block structure |
| Superblock d optimization | ✅ Yes | Same `d` + 4-bit sub-block level encoding |
| Post-d refinement | ⚠️ Partial | Re-evaluate 4-bit codebook entry with quantized scale |
| waux softening | ⚠️ Uncertain | IQ4_XS may have different neighbor/off-map pattern |
| K-means codebook learning | ❌ No | IQ4_XS uses fixed `kvalues_iq4nl`, no learned grid |
| nwant/kmaps | ❌ No | No neighbor search in IQ4_XS |
