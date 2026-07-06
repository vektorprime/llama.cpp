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

## Current Code State
- `nwant`: 4 (was 2) — the key change
- `kmeans_iters`: 20
- `num_trials`: 1 (E8 warm-start only)
- `max_samples`: 16384
- Single global grid with K-means (E8 warm-start)
- Float-space L1-weighted K-means
- Error-aware int8 snap
- 0 rounds of multi-round refinement

## Gap Analysis
- **Best KL**: 0.7166 (exp-033)
- **Unsloth target**: 0.721 (surpassed!)
- **Remaining gap**: None to Unsloth; new benchmark set
- **Next directions**: Even wider neighbor search (nwant=6? 8?), per-tensor neighbor lists, or revisit K-means refinement now that search is better

## Remaining Research Directions
1. **Even wider neighbor search**: Try nwant=6, nwant=8 to see if further gains possible
2. **Per-tensor neighbor lists**: Different nwant per tensor category
3. **Revisit K-means refinement**: With better neighbor search, grid values may now benefit from more K-means iterations/trials
4. **More aggressive multi-round refinement**: Now that search is better, ±1 GD might help
5. **Combine best-learned grid with wider search**: Current code already does this
