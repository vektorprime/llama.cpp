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

## Key Insights (as of exp-003)

1. **Asymmetric quantization is essential at 4 bits**: exp-003 removed per-sub-block mins/dmin (going symmetric) and replaced 6-bit packed scales with 8-bit int8 per-sub-block scales. KLD increased 82% (0.063→0.115) — far beyond the threshold. The per-sub-block min offsets provide grid centering for skewed weight distributions within sub-blocks. Finer scale quantization (8-bit vs 6-bit) cannot compensate for loss of grid offset freedom.

2. **Per-tensor mixing boosts are essential**: exp-002 showed that Q6_K upgrades for WV and FFN_DOWN in stock Q4_K_M are not cosmetic — removing them degrades KLD by 16.7%. These tensors are high-importance.

3. **Dead code traps exist**: exp-001 found the ATTENTION_QKV boost code path is unreachable. Always verify code paths before targeting them.

4. **Future direction**: Block-level compression should preserve asymmetry (dmin + per-sub-block mins). Promising angles: (a) scale-min correlation encoding — since scale and min tend to be proportional within a super-block, encode mins as deltas from scales using fewer bits, (b) reduce per-sub-block scale/min precision from 6-bit to 5-bit while keeping both, (c) mixed sub-block precision — use fewer bits for low-variance sub-blocks.
