# IQ2_XXS Auto-Research Synthesis — Session 2026-07-06

## Status: Best KL = 0.716172 (exp-034, nwant=8). All experiments since (exp-035 through exp-047) null or regression.

## BREAKTHROUGH (exp-033): Neighbor search depth nwant=2→4 — KL=0.7166, surpasses Unsloth!

**Increasing the neighbor search depth from `nwant=2` to `nwant=4` in `iq2xxs_rebuild_map_and_neighbours()` improved KL from 0.7238 to 0.7166 — a 1.0% relative improvement that BEATS the Unsloth target (0.721).**

This is the first experiment since exp-007 to produce a genuine KL improvement beyond the 0.7238 plateau.

### Why it works

The kmap neighbor precomputation uses Euclidean distance to find candidate grid points for 16-bit 2-bit patterns not directly in the codebook. `nwant=2` means only 2 unique distance levels are included. For the E8 lattice (highly regular), this captures all needed neighbors. But learned K-means grids lack E8's symmetry — nearby points are spread across more distance levels. Increasing to `nwant=4` casts a wider net, finding better centroid matches during quantization.

| Experiment | KL Divergence | PPL | Same Top P | Q-Time | Key Technique |
|-----------|--------------|-----|------------|--------|---------------|
| **Unsloth target** | 0.721 | 26.44 | 60.25% | — | Reference |
| **exp-033 (new best)** | **0.7166** | **26.24** | **60.56%** | **2.4 min** | **nwant=4 neighbor search** |
| Best learned (exp-020) | 0.7238 | 26.46 | 60.34% | 9.6 min | K-means++ learned grid |
| E8 lattice (no learn) | 0.7248 | 26.55 | 60.02% | 4.6 min | Pure E8 lattice |

### Why this wasn't tried before
All previous experiments focused on the codebook VALUES (K-means variants, init schemes, distance metrics). The neighbor search depth was never considered — it was assumed E8's nwant=2 setting was optimal. With learned grids, the distance structure is fundamentally different, making wider neighbor search beneficial.

## Key Findings

### 1. Neighbor search quality matters as much as codebook quality
The quantization quality depends on both (a) what grid points exist and (b) how well the quantizer finds the best grid point for each 2-bit pattern. Learned grids change the distance structure, requiring wider neighbor search.

### 2. K-means codebook learning plateau was NOT fundamental
The previous plateau at 0.7238 was not a theoretical limit — the neighbor search was undershooting for learned grids. With nwant=4, learned grids can now achieve KL below the E8 baseline and beat Unsloth.

### 3. 2.4 minute quantize is well under the 5-minute limit
The nwant=4 change adds negligible compute cost. Quantize time was 142 seconds.

## What Worked
- **nwant=2→4** (exp-033): KL improved from 0.7238 to 0.7166 (−1.0%, beats Unsloth)

## What Didn't Work (exp-035 through exp-038)

| Exp | KL | Description | Verdict |
|-----|-----|-------------|---------|
| 035 | 0.7162 | 5 KMPP trials nwant=8 | Null (identical) |
| 036 | 0.723 | Cross-tensor accumulation (4 tensors) | Regression |
| 037 | 2.379 | Cross-tensor + L1 normalization | Catastrophic |
| 038 | 0.720 | Odd-only snap + 3-round ±2 refinement | Regression |

### exp-038: Odd-only centroid enforcement — grid over-constrained
Forcing centroid values to odd [1,3,5,...,127] to prevent kmap collisions caused REGRESSION (0.716→0.720). The kmap collision issue was theoretical but in practice the neighbor search path
(fallback for non-direct-mapped patterns) already covers all 256 entries. Even-valued centroids
still participate in the neighbor list. The odd-only constraint removed valid centroid positions
without proportional benefit. Multi-round refinement with ±2 steps also failed to improve.

**Lesson**: The error-aware snap's even-valued centroids are not harmful — they contribute through
the neighbor search path. Constraining to odd values reduces codebook expressiveness.

## Current Code State (post exp-047 revert — back to exp-034 best)
- `nwant`: 8 (was 2) — the key change from exp-033/034
- `kmeans_iters`: 20
- `num_trials`: 1 (E8 warm-start only)
- `max_samples`: 16384
- Single global grid with K-means (E8 warm-start)
- Float-space L1-weighted K-means (standard, no trimming — exp-047 reverted)
- Error-aware int8 snap (allows any value 0-127, no odd constraint)
- 0 rounds of multi-round refinement

## Recent Additions (exp-042 through exp-047)

| Exp | KL | Technique | Verdict |
|-----|-----|-----------|---------|
| 042 | 0.722 | Post-quantization grid refinement with quantized sub-block scales | Regression |
| 043 | 0.717 | Wider quantizer scale search (±12 step 0.2) | Regression (marginal) |
| 045 | 0.719 | 2-pass iterative grid-scale refinement during quantization | Regression |
| 046 | 0.717 | Greedy per-element level perturbation after LS refinement | Regression |
| 047 | 0.716 | Inlier-Focus K-means (Trimmed K-means, 10% outlier discarding) | Null (identical KL) |

### exp-047: Inlier-Focus K-means null — trimming doesn't help when centroids barely move
Discarding the top 10% of furthest samples per centroid during each K-means update produced identical KL (0.716172). The centroids start from E8 and barely drift in 20 iterations, so the "outliers" are the same samples each iteration and trimming their contribution doesn't change the centroid trajectory. This further confirms the K-means training is at a fixed point — the grid is essentially E8 with minimal data-adaptive adjustments, and training tweaks don't affect the outcome.

**Lesson**: All K-means training variants (sample selection, weighted updates, outlier trimming, regularization) converge to the same near-E8 attractor. The remaining headroom for KL improvement must come from changes to the quantization SEARCH (nwant, neighbor structure) or the QUANTIZER ITSELF (scale handling, residual coding), not the grid values.

## Earlier: exp-039 through exp-041

| Exp | KL | Technique | Verdict |
|-----|-----|-----------|---------|
| 039 | 1.238 | Density-aware centroid splitting (LBG VQ) | Catastrophic regression |
| 040 | 0.7162 | E8-regularized K-means (15% blend) | Null (identical) |
| 041 | 0.7162 | Importance-weighted sample selection | Null (identical) |

### exp-039: Density-aware centroid splitting destroyed diversity
Over-utilized centroids split into under-utilized slots created near-duplicates, reducing grid diversity. Under-utilized centroids apparently provide unique patterns during quantization — removing them loses information.

### exp-040: E8 regularization is redundant
Blending centroids 15% toward E8 after each iteration had no effect because K-means starting from E8 barely drifts in 20 iterations. The centroid positions are already near-E8.

### exp-041: Importance-weighted sampling doesn't change grid
Changing sample selection from uniform stride to imatrix-proportional CDF sampling produced identical KL (0.7162). The K-means converges to the same attractor regardless of sampling distribution — the E8 warm-start dominates the trajectory.

**Lesson**: With 1-tensor E8-warm-start K-means, the grid is already near its fixed point after 20 iterations. Sample selection doesn't matter. The codebook is essentially E8 with minor data-adaptive adjustments. Breakthroughs require structural changes (like nwant adjustment, exp-033) rather than training tweaks.

## Gap Analysis
- **Best KL**: 0.716172 (exp-034)
- **Unsloth target**: 0.721 (surpassed!)
- **Remaining gap**: None to Unsloth; new benchmark set
- **Grid training is saturated**: 13 consecutive experiments (exp-035 through exp-047) have produced either null or regression results. Every K-means variant (weighted sampling, outlier trimming, E8 regularization, cross-tensor, more iterations, etc.) converges to the same near-E8 grid. Neighbor search (nwant) was the only lever that ever improved KL.
- **Next directions must change strategy**: Grid training space is exhausted. The quantizer search (kmap neighbor structure, distance metrics during quantization) and structural format changes are the remaining unexplored levers.

## Remaining Research Directions
1. **Cross-tensor accumulation WITH nwant=8**: Previously tried with nwant=4 (exp-036) and regressed. nwant=8 might compensate for the reduced per-tensor focus. (Untested combination)
2. **Hybrid E8+K-means grid**: Replace low-count centroids with their E8 equivalents to maintain coverage diversity. (Untested)
3. **Quantization-aware neighbor distance**: Weight the kmap neighbor distance by imatrix importance during kmap construction, not just during quantization. (Untested)
4. **Multi-codebook quantization**: Store 2+ grids per tensor, select best per superblock with 1-bit selector. (Requires format change — speculative)
5. **Learn the quantizer scale `d` per superblock**: Instead of selecting from discrete candidates, learn the optimal scale per block via gradient descent on quantization loss. (Computational cost concern)
6. **Beyond K-means: Low-bitwidth quantization using the fixed E8 grid with learned scale per centroid**: Each centroid gets its own learned scale factor, effectively creating a scaled codebook. (Requires format change)
