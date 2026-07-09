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

## Key Insights (as of exp-002)

1. **Per-tensor mixing boosts are essential**: The Q6_K upgrades for WV and FFN_DOWN
   tensors in stock Q4_K_M are not cosmetic — removing them degrades KLD by 16.7%
   and drops same top p by 0.9pp. These tensors are high-importance. Size savings
   (~27.8 MB) come at unacceptable quality cost.

2. **Dead code traps exist**: The ATTENTION_QKV boost code path in llama-quant.cpp
   is unreachable for Qwen models because `category_is_attn_v()` catches QKV tensors
   first. Always verify code paths are actually executed before targeting them.

3. **Future direction**: Block-level struct compression (reducing scales bytes or
   qs bytes) is the more promising path. Per-tensor demotion trades too much quality
   for the size saved. Focus on making the quantization block itself more compact
   while preserving the same quantization algorithm quality.
