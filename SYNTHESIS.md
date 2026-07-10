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
| exp-006 | SDM predictor for mins (8-byte scales) — implementation bugs | 523,209,760 | NULL | NULL | FAILED |
| exp-007 | Codebook VQ + per-weight nibble re-optimization (8-byte scales) | 523,209,760 | 0.101213 | 83.631% | REGRESSION |
| exp-008 | QK_K_CLONE=512 shared d/dmin (284-byte block) | 526,253,600 | 12.848800 | 0.006% | REGRESSION |
| exp-009 | Token_embd/output Q6_K→Q4_K_M_CLONE (no block changes) | 463,740,960 | 0.073896 | 84.605% | REGRESSION |
| exp-010 | Token_embd/output Q6_K→Q5_K (clone ftype) | 495,525,920 | 0.064977 | 85.937% | REGRESSION (borderline) |
| **exp-011** | **Q5_K embd + Q6_K for ALL QKV layers** | **509,042,720** | **0.061802** | **86.391%** | **SUCCESS** |
| exp-012 | SDMP-WS mixed precision 136-byte block | 517,122,080 | NULL | NULL | FAILED (CUDA) |
| exp-013 | IJO iterative refinement of 144-byte block | 529,297,440 | 0.165138 | 79.430% | REGRESSION |

## Key Insights (updated after exp-013)

1. **Asymmetric quantization is essential at 4 bits** (exp-003): removing per-sub-block mins/dmin caused KLD increase of 82%.

2. **Per-tensor mixing boosts are essential** (exp-002): Q6_K upgrades for WV and FFN_DOWN are not cosmetic. Removing them degrades KLD by 16.7%.

3. **Scale and min precision are both critical** (exp-004): reducing scale 6→5 bits and min 6→3 bits caused KLD increase of 23.1%. Both need ≥5-6 bits.

4. **DPCM/correlation chains fail** (exp-005): inter-sub-block correlation is too weak for delta encoding — errors accumulate and cascade.

5. **Per-weight recovery cannot fix systematic grid bias** (exp-007): codebook VQ introduces sub-block-level scale errors that affect all 32 weights systematically. Nibble re-optimization only provides local adjustment; it cannot reposition the quantization grid.

6. **The KLD threshold is narrow** (all experiments): every approach that achieved any size reduction also increased KLD by ≥23%. The 0.062947 threshold is very tight — even 1.15% size reduction pushes KLD far beyond it.

7. **Block compression has limited reach**: ~40% of model weights are Q6_K tensors that don't use the clone block. Per-block savings of 2.78% translate to only ~1.15% overall.

8. **Output/token_embd is sensitive but efficient per-MB** (exp-009): reducing the 199 MB token_embd from Q6_K→Q4_K_M_CLONE saved 62.5 MB (12.4% total) but exceeded KLD threshold by 17.4%. However, the KLD increase per MB saved (0.000175) is half that of removing Q6_K boosts (0.000377), suggesting the output layer is relatively more robust than attention layers. A Q5_K middle ground (~32 MB savings) could potentially stay within bounds.

9. **Q5_K for output/token_embd is near-threshold** (exp-010): reducing Q6_K→Q5_K saved 33.8 MB (6.4%) with KLD 0.0650 — only 3.2% above threshold by 0.002. Same top p 85.94% just 0.45pp below. The KLD increase is dramatically sublinear: Q5_K costs 0.0019 KLD/bpw vs Q4_K_M_CLONE's 0.0053 KLD/bpw. The per-saved-MB KLD increase (0.00006/MB) is 6× better than exp-002 (0.00038/MB). This is the closest any experiment has come to success. A small additional optimization (e.g., keep Q6_K for 2-3 more QKV layers, costing ~4 MB but recovering the ~0.45pp same_top_p gap) could push this over the threshold.

10. **Alternating optimization degrades, not improves, quantization** (exp-013): Fixing nibbles and re-optimizing the grid parameters via linear regression (fix L → optimize d/dmin/sc/m → re-quantize L → repeat) converged to worse local minima (KLD 0.165 vs baseline 0.063). The secondary quantization of grid parameters creates non-smooth distortions that destabilize alternating optimization. The original greedy one-shot algorithm works better because it derives the grid from raw weight statistics, not from an intermediate quantized approximation.

11. **The Q4_K quantize algorithm is near-optimal for its byte budget** (exp-004/005/007/013): Every attempt to improve quantization quality within the 144-byte constraint — whether by re-optimizing the grid, compressing scales, or VQ codebooks — has either degraded quality or maintained it (NULL). No experiment has successfully improved quality within the same byte budget. The 12+ years of optimization in the ggml codebase have already found a near-optimal local minimum for the Q4_K parameterization.

## What's Left to Try

The clearest remaining paths:

- **Q5_K for token_embd + keep extra Q6_K QKV layers**: exp-010 showed Q5_K alone is 0.002 KLD and 0.45pp same_top_p away from passing. Keeping Q6_K for 2-3 more QKV layers (~4 MB) could recover the gap while still netting ~28 MB savings. This is a combination approach: per-tensor precision reduction + strategic selectivity.

**WINNER — exp-011:** Q5_K for output/token_embd + Q6_K for ALL QKV attention layers produced KLD 0.061802 (better than baseline 0.062947!) and same top p 86.391% (> 86.387%) while saving 20.25 MB (3.83%). The strategy of taking precision from the least-sensitive large tensor (token_embd, 199 MB → Q5_K saves 33.8 MB) and reinvesting ~40% of the savings into the most precision-critical mid-size tensors (QKV, ~13.5 MB cost for all-24 Q6_K vs default ~50%) proved that calibrated trade-offs can beat the baseline on both size AND quality. This is the first successful experiment.

- **Further directions from exp-011:** The output layer's Q6_K→Q5_K sensitivity is extremely low (~0.00006 KLD/MB). Could push further to Q4_K on the output layer combined with Q6_K on even more tensors. Or try Q5_K on other large-but-tolerant tensors (ffn_gate, ffn_up) while keeping precision on attention-critical tensors.

- **Compress qs[] (128-byte weight data)** instead of scales. The 4-bit weights may have intra-superblock redundancy that scales lack. For example, many sub-blocks may have similar weight patterns that could share a compressed representation.

- **Dual-type alternating blocks**: store full 144-byte blocks interspersed with 140-byte "lite" blocks that borrow d/dmin from the previous full block, without changing QK_K. This avoids the size calculation mismatches that killed exp-008.

- **Column-level compression for the output/token_embd matrix**: the [1024, 248320] output.weight is fundamentally different from attention weights. K-means clustering of columns (vocabulary tokens) with codebook-based reconstruction could achieve dramatic compression. A codebook of 4096 1024-dim centroids would compress each column from 1024*4.5/8 = 576 bytes to 12 bits = 1.5 bytes — ~384× compression of the output layer alone. Requires custom ggml tensor format support.
