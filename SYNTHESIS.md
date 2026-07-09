# Q4_K_M_CLONE Auto-Research Synthesis — EXAMPLE

This is a **template/example**. Replace when real experiments are run.

## Baseline: Q4_K_M (stock) vs BF16 reference

| Metric | BF16 Reference | Q4_K_M (stock) |
|--------|---------------|----------------|
| PPL | 21.5386 | 22.4499 |
| KL divergence | 0.0 | 0.062947 |
| Same top p | 100% | 86.387% |
| RMS Δp | 0.0% | 5.753% |
| GGUF Size | ~1.41 GB | ~505 MB |

## Research Objective

Reduce GGUF file size below 508 MB while maintaining:
- KLD ≤ 0.035490
- Same top p ≥ 89.613%

## Results Summary

| Exp | Description | Size (MB) | KLD | Same top p | Status |
|-----|-------------|-----------|-----|------------|--------|
| — | Q4_K_M baseline | 508 | 0.035490 | 89.613% | Baseline |
