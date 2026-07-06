# IQ2_XXS Auto-Research Synthesis — Session 2026-07-06

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

## Current Code State
- `nwant`: 8 (was 2) — the key change from exp-033/034
- `kmeans_iters`: 20
- `num_trials`: 1 (E8 warm-start only)
- `max_samples`: 16384
- Single global grid with K-means (E8 warm-start)
- Float-space L1-weighted K-means
- Error-aware int8 snap (allows any value 0-127, no odd constraint)
- 0 rounds of multi-round refinement

## Gap Analysis
- **Best KL**: 0.716172 (exp-034)
- **Unsloth target**: 0.721 (surpassed!)
- **Remaining gap**: None to Unsloth; new benchmark set
- **Next directions**: Still open — neighbor search is maxed at nwant=8, grid training is
  1-tensor E8-warm-start only. Cross-tensor and refinement approaches consistently regress.

## Remaining Research Directions
1. **Even wider neighbor search**: Already at nwant=8; nwant=10+ may add more noise than signal
2. **Per-tensor neighbor lists**: Different nwant per tensor category — untried
3. **More K-means iterations with nwant=8**: 20 iters is conservative; 40-60 iters untested at nwant=8
4. **Cross-tensor accumulation WITHOUT normalization**: Raw pooling from 2-3 tensors with nwant=8 — previously tried with nwant=4 and regressed; worth re-testing with nwant=8
5. **Grid pruning**: Remove redundant centroids (kmap collisions) post-training to ensure all 256 entries are unique at the 2-bit level — prune then re-fill with split of highest-error centroid
