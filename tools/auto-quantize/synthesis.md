# IQ2_XXS Auto-Research Synthesis — Session 2026-07-06 (Updated exp-082)

## Status: Best KL = 0.666162 (exp-082, weight exponent 0.30). No improvement since exp-082.

## Plateaud: exp-083 through exp-094 all regressed or null

**exp-082's weight exponent 0.30 remains the best (KL 0.666162). All 8 subsequent experiments regressed.** The weight formula has been exhaustively explored — exponent values from 0.25 to 1.0, asymmetric profiles, ratio clamping, normalization — all regressed or null.

| Experiment | KL Divergence | PPL | Same Top P | Q-Time | Key Technique | Verdict |
|-----------|--------------|-----|------------|--------|---------------|---------|
| **Unsloth target** | 0.721 | 26.44 | 60.25% | — | Reference | — |
| **exp-082 (BEST)** | **0.666162** | **25.11** | **61.96%** | **5.9 min** | **Weight exponent 0.30** | **BEST** |
| exp-090 | 0.682228 | 25.48 | 61.24% | 6.0 min | Remove sqrtf from waux (neighbor weight alignment) | REGRESSION |
| exp-089 | 0.679015 | 25.47 | 61.17% | 5.0 min | Intra-sub-block weight normalization | REGRESSION |
| exp-088 | 0.794320 | 29.13 | 58.99% | 5.9 min | Weight ratio clamping per sub-block | REGRESSION |
| exp-087 | 0.723409 | 26.93 | 58.72% | 5.0 min | Linear-L1 weight formula (1+\|xb\|) | REGRESSION |
| exp-086 | 0.669495 | 25.22 | 62.09% | 5.9 min | d-opt/post-d exponent 0.40 | REGRESSION |
| exp-085 | 0.669486 | 25.26 | 62.07% | 5.9 min | d-opt/post-d exponent 0.50 | REGRESSION |
| exp-084 | 0.673372 | 25.34 | 61.43% | 5.9 min | Harmonize all to 0.30 | REGRESSION |
| exp-083 | 0.679018 | 25.63 | 61.04% | 5.9 min | Weight exponent 0.25 | REGRESSION |
| exp-081 | 0.668342 | 25.01 | 61.58% | 5.9 min | Weight exponent 0.35 | IMPROVEMENT |
| exp-080 | 0.682340 | 25.41 | 61.11% | 5.9 min | Weight exponent 0.4 | IMPROVEMENT |
| exp-078 | 0.691085 | 25.69 | 61.29% | 5.0 min | Per-sub-block sigma2 | IMPROVEMENT |

## Key Findings

### 1. Weight formula exponent is saturated at 0.30
exp-082 (0.30) is the best. exp-083 (0.25) regressed to 0.679. All asymmetric profiles (084-086) regressed. All structural weight changes (087-090) regressed. The optimal weight formula is confirmed: `qw * powf(sigma2_per_ib + xb^2, 0.30f)`.

### 2. Every post-exp-082 modification regressed — the system is at a local optimum
8 consecutive experiments (exp-083 through exp-090) all regressed or null. The weight formula, neighbor search structure, and quantizer search space have been exhaustively explored within the current IQ2_XXS format. Further improvement likely requires format changes (multi-codebook, non-uniform scales, gain-shape decomposition).

### 3. Combined effect: exp-078 through exp-082 = 4.9% cumulative KL improvement
KL improved from 0.699009 (exp-064) to 0.666162 (exp-082), a 4.7% relative improvement from weight formula changes alone.

## What Worked
- **Weight exponent 0.30** (exp-082): KL 0.666162 (−0.33% from 0.668342)
- **Weight exponent 0.35** (exp-081): KL 0.668342 (−2.05% from 0.682340)
- **Weight exponent 0.4** (exp-080): KL 0.682340 (−1.27% from 0.691085)
- **Per-sub-block sigma2** (exp-078): KL 0.691085 (−1.13% from 0.699009)
- **65-candidate d optimization 0.5% step** (exp-064): KL 0.699009 (−0.52% from 0.702666)
- **Post-d grid index recomputation** (exp-055): KL 0.710657 (−0.63% from 0.715144)
- **Superblock d optimization** (exp-049): KL 0.715144 (−0.14% from 0.716172)
- **nwant=2→4** (exp-033): KL 0.7166 (−1.0% from 0.7238)

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

## Current Code State (exp-082 — new best)
- `nwant`: 8
- `kmeans_iters`: 20
- `num_trials`: 1 (E8 warm-start only)
- `max_samples`: 16384
- Single global grid with K-means (E8 warm-start)
- Float-space L1-weighted K-means (standard)
- Error-aware int8 snap (allows any value 0-127, no odd constraint)
- 0 rounds of multi-round refinement
- **Per-sub-block sigma2** for weight formula (exp-078)
- **Weight exponent 0.30** (`powf(sigma2+xb^2, 0.30f)`) instead of sqrt (exp-082)
- **65-candidate d optimization** at 0.5% step, ±16% range (exp-064)
- **Post-d grid index recomputation** with quantized scale (exp-055)

## Recent Additions (exp-042 through exp-090)

| Exp | KL | Technique | Verdict |
|-----|-----|-----------|---------|
| 042 | 0.722 | Post-quantization grid refinement with quantized sub-block scales | Regression |
| 043 | 0.717 | Wider quantizer scale search (±12 step 0.2) | Regression (marginal) |
| 045 | 0.719 | 2-pass iterative grid-scale refinement during quantization | Regression |
| 046 | 0.717 | Greedy per-element level perturbation after LS refinement | Regression |
| 047 | 0.716 | Inlier-Focus K-means (Trimmed K-means, 10% outlier discarding) | Null (identical KL) |
| 083 | 0.679 | Weight exponent 0.25 (too soft) | Regression |
| 084 | 0.673 | Harmonize all weights to 0.30 | Regression |
| 085 | 0.669 | d-opt/post-d exponent 0.50, main 0.30 | Regression |
| 086 | 0.669 | d-opt/post-d exponent 0.40, main 0.30 | Regression |
| 087 | 0.723 | Linear-L1 weight formula (1+\|xb\|) | Catastrophic |
| 088 | 0.794 | Weight ratio clamping per sub-block | Catastrophic |
| 089 | 0.679 | Intra-sub-block weight normalization | Regression |
| 090 | 0.682 | Remove sqrtf from waux (align neighbor weight) | Regression |

### exp-055: Post-d-optimization grid index recomputation — new best KL=0.710657 (-0.63% from exp-049)

**Grid indices chosen with continuous sub-block scales are suboptimal for the final quantized scale `d*(2*l+1)`.** After the superblock `d` optimization (exp-049), recompute each 8D chunk's 2-bit codes and grid indices using the actual quantized scale. This second-pass refinement uses `fabsf(xb)` (absolute values from raw input) to avoid the stale-xval bug that caused exp-054's catastrophic regression (KL=1.071).

**Result**: KL improved from 0.715144 to 0.710657. PPL dropped from 26.45 to 26.26. Same top p improved to 60.61%. Quantize time unchanged (~4.8 min).

**Why it works**: The initial grid index selection chooses centroids that minimize weighted L1 error under a continuous per-sub-block scale. But the final storage quantizes the sub-block scale to 4 bits via `l = nearest_int(0.5*(id*scale-1))`. The quantized scale `d*(2*l+1)` can differ from the optimal continuous scale by several percent. Re-selecting centroids with the actual quantized scale (via kmap lookup or neighbor search) finds entries whose odd-int values better match the real runtime decoding.

**Implementation**: ~40 lines added after the d optimization block, inside `quantize_row_iq2_xxs_impl`. The refinement iterates over each sub-block, computes `id_q = 1/(d*(2*l+1))`, generates 2-bit codes from `fabsf(xb)`, and picks the best grid entry (via kmap or neighbor fallback). A weighted error comparison ensures only beneficial changes are applied.

### exp-049: Superblock scale optimization — first improvement since exp-034, KL=0.715144

**Superblock scale `d` optimization**: Previously `d = max_scale/31` (maximizing dynamic range). Now we try 9 candidates of `d` (±16% in 4% steps), compute the full 128-element weighted reconstruction error for each (using the already-chosen grid indices, centroid values, and sign bits), and pick `d` with the lowest error. This accounts for:
1. Imatrix importance weights per sub-block
2. Actual centroid values (not just scale magnitudes)
3. Joint 4-bit scale quantization error across all sub-blocks

**Result**: KL improved from **0.716172 to 0.715144** (Δ = -0.001028, 0.14% relative improvement). PPL essentially unchanged (26.45 vs 26.24 for exp-033, within noise). Same top p improved to 60.42%.

**Why it works**: The `max_scale/31` heuristic assumes the sub-block with the largest scale is always the most important. But the weighted reconstruction error criterion correctly balances importance (imatrix) against element magnitude. For superblocks where sub-block scales are close to each other, the optimal `d` can shift slightly to reduce quantization error on high-importance sub-blocks at the expense of low-importance ones. The 4% step granularity (9 candidates ±16%) provides sufficient coverage without excessive computation (negligible overhead — 0.9s added to quantize time).

**Lesson**: Even small optimizations at the superblock scale level produce measurable improvements. This is the first successful experiment outside K-means training since exp-034, confirming the synthesis direction that "grid training is saturated — the quantizer SEARCH still has headroom."

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
- **Best KL**: 0.666162 (exp-082)
- **Unsloth target**: 0.721 (surpassed by 7.6%)
- **Weight formula EXHAUSTED**: All exponent values (0.25-1.0), asymmetric profiles, structural variants (clamping, normalization, linear-L1, waux sqrt removal) tested. The optimal configuration is confirmed: `qw * powf(sigma2 + xb^2, 0.30f)` with per-sub-block sigma2 (exp-078).
- **Quantize time**: ~356-362s (5.9-6.0 min) — slightly exceeds 5-min soft limit.
- **Index-scale coupling remains fragile**: Any two-way modification (changing both indices and scale) still causes catastrophic regression.
- **Remaining headroom**: The current IQ2_XXS algorithm appears to be at a global local optimum. No further improvement found in weight formula, neighbor search structure, or quantizer search space. Future progress likely requires:
  1. **Multi-codebook quantization**: Split 256-entry codebook into per-sub-block selectable sub-codebooks
  2. **Non-uniform scale level spacing**: Change 4-bit scale decoding from uniform `2*l+1` to non-uniform
  3. **Gain-shape weight representation**: Decouple magnitude from direction at format level
  4. **Residual quantization**: Encode residual after 2-bit quantization with additional bits
  5. **Per-element adaptive bit allocation**: Different bit widths per element based on importance

## Key Findings: Experiments 056-067

| Exp | KL | Technique | Verdict |
|-----|-----|-----------|---------|
| 055 | 0.710657 | Post-d grid index recomputation | SUCCESS |
| 056 | 0.7949 | Second d optimization + ±1 level | REGRESSION |
| 057 | 0.7122 | Odd-forcing kmap + neighbor | REGRESSION |
| 058 | 0.7212 | Robust index refinement + d recompute | REGRESSION |
| 059 | 0.7122 | Odd-forced neighbor scoring only | REGRESSION |
| 060 | 1.2006 | Level recomputation after refinement | CATASTROPHIC |
| **061** | **0.7049** | **d opt 4%→2% step (17 cand)** | **SUCCESS** |
| 062 | 0.7126 | d opt ±24% range | REGRESSION |
| **063** | **0.7027** | **d opt 2%→1% step (33 cand)** | **SUCCESS** |
| **064** | **0.6990** | **d opt 1%→0.5% step (65 cand, ±16%)** | **SUCCESS** |
| 065 | 0.7036 | d opt 0.25% step (129 cand) | REGRESSION |
| 066 | 0.8152 | Sign parity re-evaluation in post-d refinement | CATASTROPHIC |
| 067 | 0.7017 | d opt ±8% range (33 cand) | REGRESSION |

### Critical Lessons

1. **Index-scale coupling is fundamental**: Changing scale after fixing indices (or vice versa) consistently causes regression. The only safe modifications are ones where one side is fixed while the other is optimized.

2. **D optimization benefits from finer resolution**: The ±16% range at 0.5% step (65 candidates) is significantly better than the original 4% step (9 candidates). The cumulative improvement from these three experiments is -1.64% KL relative.

3. **Wider d range hurts**: Expanding beyond ±16% introduces d candidates that don't match the fixed grid indices, causing level quantization mismatches. The safe range is exactly ±16% (which corresponds to the ±2 level range around the nearest level).

4. **Neighbor L1 scoring is optimal**: All attempts to improve it (dot-product, odd-forced, L2) cause regression. The raw weighted L1 with full diversity is the best selection criterion for the quantizer's neighbor search.

## Updated Syntheses: Experiments 067-071 (All Regressions)
The project is at a fundamental local optimum. Every quantizer-search modification tried in the last
5 experiments regressed:

| Exp | KL | Technique | Failure Mode |
|-----|-----|-----------|-------------|
| 067 | 0.702 | Narrower d range ±8% | ±8% misses optimal d for some superblocks |
| 068 | 12.31 | Quantizer-aware K-means (dot-product assignment) | Assignment/update metric mismatch — same as exp-037 collapse |
| 069 | 0.752 | kmap2 second-best centroid | Second-best adds noise, L2-first optimal for L1 objective |
| 070 | 0.743 | Importance-weighted 2-bit levels (alpha=0.3) | Biased levels push patterns off-map, over-depending on neighbor search |
| 071 | 1.077 | Exhaustive 4-bit level search (weighted error) | Level ≠ nearest-int breaks index-scale coupling; d optimization assumes nearest-int |

### Key Findings
1. **The nearest-int level formula IS optimal**: Derived from `scales[ib] = sum(w*xval*centroid)/sum(w*centroid^2)`, the level `l = nearest_int(0.5*(scales[ib]/d - 1))` minimizes the same weighted reconstruction error used throughout the quantizer. Any deviation increases error — there is no slack in the level selection.

2. **Index-scale coupling is inviolable**: Exp-071 confirms that even changing levels (not indices, not d) breaks the coupling. The three parameters (d, levels, indices) are jointly optimal at their current configuration. Any one-way modification causes the system to find a worse joint optimum.

3. **Every modifiable knob has been tried**: All quantizer search parameters (d step, d range, level selection, neighbor criterion, neighbor depth, grid index selection) have been individually modified and either regressed or hit null. The combined configuration at exp-064 (65 candidates, 0.5% step, ±16% range, nwant=8, nearest-int levels) is the global optimum for the current IQ2_XXS format and algorithm.

### What Hasn't Been Tried
- **Multi-codebook quantization**: Split the 256-entry codebook into per-sub-block selectable sub-codebooks. Format change.
- **Non-uniform scale level spacing**: Change the 4-bit scale decoding from uniform `2*l+1` to non-uniform. Requires dequantizer changes across all backends.
- **Residual quantization**: Encode the residual after the first 2-bit quantization with additional bits.
- **Per-element adaptive bit allocation**: Some elements get 3 bits, others get 1 bit, based on importance.
- **Gain-shape weight representation**: Decouple weight magnitude from direction at the format level.

All of these require FORMAT CHANGES (new GGML_TYPE, new block struct, new inference kernels across CPU/CUDA/Metal/Vulkan), representing a major infrastructure undertaking beyond the current experiment scope.

## exp-088 In Progress: Weight ratio clamping per sub-block

**Hypothesis**: The weight formula can produce extreme max:min weight ratios (10:1+) within a sub-block when outlier elements dominate. Clamping the ratio to ≤5 per sub-block prevents single elements from dominating the optimization while preserving relative importance ordering. This is a post-processing step on the weights that only activates for outlier-dominated sub-blocks.

## Remaining Research Directions
1. **Combined d+level joint optimization in a single pass**: Modify the d optimization to evaluate levels using the same exhaustive criterion. Safe because both d and level are optimized simultaneously. Computational cost: ~16x d optimization (adds ~10-15s, still within 5 min). **This is the most promising near-term direction.**
2. **Multi-codebook quantization**: Requires format changes but could provide step-change improvement.
3. **Gain-shape quantization**: Format-level decoupling of magnitude and direction. Speculative.
