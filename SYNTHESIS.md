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

## Key Insights (updated after exp-009)

1. **Asymmetric quantization is essential at 4 bits** (exp-003): removing per-sub-block mins/dmin caused KLD increase of 82%.

2. **Per-tensor mixing boosts are essential** (exp-002): Q6_K upgrades for WV and FFN_DOWN are not cosmetic. Removing them degrades KLD by 16.7%.

3. **Scale and min precision are both critical** (exp-004): reducing scale 6→5 bits and min 6→3 bits caused KLD increase of 23.1%. Both need ≥5-6 bits.

4. **DPCM/correlation chains fail** (exp-005): inter-sub-block correlation is too weak for delta encoding — errors accumulate and cascade.

5. **Per-weight recovery cannot fix systematic grid bias** (exp-007): codebook VQ introduces sub-block-level scale errors that affect all 32 weights systematically. Nibble re-optimization only provides local adjustment; it cannot reposition the quantization grid.

6. **The KLD threshold is narrow** (all experiments): every approach that achieved any size reduction also increased KLD by ≥23%. The 0.062947 threshold is very tight — even 1.15% size reduction pushes KLD far beyond it.

7. **Block compression has limited reach**: ~40% of model weights are Q6_K tensors that don't use the clone block. Per-block savings of 2.78% translate to only ~1.15% overall.

8. **Output/token_embd is sensitive but efficient per-MB** (exp-009): reducing the 199 MB token_embd from Q6_K→Q4_K_M_CLONE saved 62.5 MB (12.4% total) but exceeded KLD threshold by 17.4%. However, the KLD increase per MB saved (0.000175) is half that of removing Q6_K boosts (0.000377), suggesting the output layer is relatively more robust than attention layers. A Q5_K middle ground (~32 MB savings) could potentially stay within bounds.

## What's Left to Try

The pattern is clear: any compression of scales or mins degrades quality. The remaining angles:
- **Compress qs[] (128-byte weight data)** instead of scales. The 4-bit weights may have spatial redundancy (similarity across adjacent weights or sub-blocks) that could be exploited — e.g., delta encoding across columns, or pattern-based compression.
- **Secondary quantization with importance weighting**: apply coarser quantization to less important sub-blocks while keeping full precision for important ones.
- **Tighter scale encoding without information loss**: e.g., Huffman/arithmetic coding of scale-min pairs exploiting the non-uniform distribution of values (but this is variable-length, hard to implement in fixed-size blocks).
- **Keep QK_K fixed at 256.** exp-008 showed that changing the superblock size breaks tensor dimension math across the ggml stack. All experiments should preserve block size invariants and find compression within the 144-byte envelope.
- **Dual-type alternating blocks**: store full 144-byte blocks interspersed with 140-byte "lite" blocks that borrow d/dmin from the previous full block, without changing QK_K. This avoids the size calculation mismatches that killed exp-008.
- **Q5_K for token_embd**: the Q6_K→Q4_K_M_CLONE jump was too aggressive (0.0739 KLD), but Q5_K (5.5 bpw) might save ~32 MB while staying within the 0.0629 KLD threshold. Only a 1-line change in llama-quant.cpp.
