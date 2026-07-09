# Q4_K_M_CLONE Auto-Research Synthesis

## Baseline: Q4_K_M (stock) vs BF16 reference

| Metric | BF16 Reference | Q4_K_M (stock) |
|--------|---------------|----------------|
| PPL | 21.5386 | 22.4499 |
| KL divergence | 0.0 | 0.062947 |
| Same top p | 100% | 86.387% |
| RMS Δp | 0.0% | 5.753% |
| GGUF Size | ~1.41 GB | 529,297,440 bytes (~505 MB) |

## Research Objective

Reduce GGUF file size below 505 MB while maintaining:
- KLD ≤ 0.062947
- Same top p ≥ 86.387%

## Results Summary

| Exp | Description | Size (bytes) | KLD | Same top p | Status |
|-----|-------------|-------------|-----|------------|--------|
| — | Q4_K_M baseline | 529,297,440 | 0.062947 | 86.387% | Baseline |
| exp-001 | Remove ATTENTION_QKV Q5_K boost (dead code) | 529,297,440 | 0.062947 | 86.387% | NULL |
| exp-002 | Remove Q6_K boost for ATTENTION_WV and FFN_DOWN | 501,452,832 | 0.073436 | 85.483% | REGRESSION |
| exp-003 | Symmetric sub-block quant with 8-bit scales (no dmin) | 520,165,920 | 0.114585 | 82.501% | REGRESSION |
| exp-004 | 5+3b scale/min per sub-block (8-byte scales, 140B block) | 523,209,760 | 0.077500 | 85.427% | REGRESSION |
| exp-005 | Dual-anchor DPCM delta encoding (10-byte scales, 142B block) | 526,253,600 | 0.180095 | 78.472% | REGRESSION |

## Key Insights (as of exp-004)

1. **Asymmetric quantization is essential at 4 bits**: exp-003 removed per-sub-block mins/dmin (going symmetric). KLD increased 82%. Per-sub-block min offsets provide grid centering for skewed weight distributions within sub-blocks.

2. **Per-tensor mixing boosts are essential**: exp-002 showed Q6_K upgrades for WV and FFN_DOWN in stock Q4_K_M are not cosmetic — removing them degrades KLD by 16.7%.

3. **Scale and min precision are both critical**: exp-004 reduced scale precision 6→5 bits and min precision 6→3 bits while keeping the asymmetric framework intact. Even this caused KLD increase of 23.1%. Mins especially need more than 3 bits — 8 levels is too coarse for proper sub-block grid centering. Scales at 5 bits also degrade.

4. **Size-quality tradeoff is steep**: All 4 experiments have either been null (dead code), or achieved size reduction at unacceptable quality cost. The KLD threshold (0.062947) is narrow — even 1.15% size reduction can push KLD beyond it.

5. **Future direction**: Focus on lossless-within-margin compression. DPCM/delta encoding of scale-min pairs (keeping same 6+6 bit precision but exploiting inter-sub-block correlation to reduce total bits) is the most promising approach. Mixed precision using importance-weighted sub-blocks is another. Simple precision reduction won't work — need encoding schemes that preserve effective precision.

6. **DPCM delta encoding fails due to error accumulation (exp-005)**: Encoding scale-min differences across sub-blocks using 4-bit signed deltas (±8) compounds errors. With dual anchors (max 3-step chain), error still accumulates to significant levels. Inter-sub-block scale deltas sometimes exceed the delta range, and clamped values cascade. The correlation between adjacent sub-blocks is insufficient to justify DPCM's precision trade-off. Codebook-based vector quantization or intra-sub-block correlation (delta-encoding m from sc rather than from previous m) may be more promising.
