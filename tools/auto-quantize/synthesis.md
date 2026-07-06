# IQ2_XXS Auto-Research Synthesis — Session 2026-07-06 (Updated exp-081)

## Status: Best KL = 0.668342 (exp-081, weight exponent 0.35). Previous best was 0.682340 (exp-080, weight exponent 0.4).

## BREAKTHROUGH (exp-081): Weight exponent 0.35 — KL breaks below 0.67!

**Reducing the weight exponent from 0.4 to 0.35 improves KL from 0.682340 to 0.668342 — a 2.05% relative improvement.** The softer exponent continues to improve results, suggesting the optimal may be even lower. PPL dropped from 25.41 to 25.01. Same top p improved from 61.11% to 61.58%.

| Experiment | KL Divergence | PPL | Same Top P | Q-Time | Key Technique |
|-----------|--------------|-----|------------|--------|---------------|
| **Unsloth target** | 0.721 | 26.44 | 60.25% | — | Reference |
| **exp-081 (NEW BEST)** | **0.668342** | **25.01** | **61.58%** | **5.9 min** | **Weight exponent 0.35 (was 0.4)** |
| **exp-080** | 0.682340 | 25.41 | 61.11% | 5.9 min | Weight exponent 0.4 (was 0.5) |
| **exp-078** | 0.691085 | 25.69 | 61.29% | 5.0 min | Per-sub-block sigma2 |
| exp-064 | 0.699009 | 25.90 | 61.11% | 5.0 min | 0.5% d step (65 candidates) |
| exp-063 | 0.702666 | 26.02 | 60.98% | 4.9 min | 1% d step (33 candidates) |
| exp-061 | 0.704868 | 26.12 | 60.97% | 4.8 min | 2% d step (17 candidates) |
| exp-055 | 0.710657 | 26.26 | 60.61% | 4.8 min | Post-d grid index recomputation |
| exp-049 | 0.715144 | 26.45 | 60.42% | 4.6 min | Superblock scale optimization |
| exp-033 | 0.7166 | 26.24 | 60.56% | 2.4 min | nwant=4 neighbor search |

## Key Findings

### 1. Weight formula exponent has significant headroom below 0.4 (exp-081)
Reducing exponent from 0.4 to 0.35 improves KL by 2.05% (0.682340 → 0.668342). This is LARGER than the 1.27% improvement from 0.5→0.4, suggesting the optimal exponent may be well below 0.35. The softer weighting continues to benefit the quantizer by preventing outlier elements from dominating.

### 2. Sigma2 localization improves weight selectivity (exp-078)
Per-sub-block sigma2 (32-element) is strictly better than per-superblock sigma2 (128-element). Further localization to per-chunk (8-element) regresses — the 32-element sub-block is the optimal granularity for the joint d optimization.

### 3. Combined effect: exp-078 + exp-080 + exp-081 = 4.5% cumulative KL improvement
The weight exponent experiments compound with per-sub-block sigma2: KL improved from 0.699009 (exp-064) to 0.668342 (exp-081), a 4.5% relative improvement.

## What Worked
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

## Current Code State (exp-081 — new best)
- `nwant`: 8
- `kmeans_iters`: 20
- `num_trials`: 1 (E8 warm-start only)
- `max_samples`: 16384
- Single global grid with K-means (E8 warm-start)
- Float-space L1-weighted K-means (standard)
- Error-aware int8 snap (allows any value 0-127, no odd constraint)
- 0 rounds of multi-round refinement
- **Per-sub-block sigma2** for weight formula (exp-078)
- **Weight exponent 0.35** (`powf(sigma2+xb^2, 0.35f)`) instead of sqrt (exp-081)
- **65-candidate d optimization** at 0.5% step, ±16% range (exp-064)
- **Post-d grid index recomputation** with quantized scale (exp-055)

## Recent Additions (exp-042 through exp-047)

| Exp | KL | Technique | Verdict |
|-----|-----|-----------|---------|
| 042 | 0.722 | Post-quantization grid refinement with quantized sub-block scales | Regression |
| 043 | 0.717 | Wider quantizer scale search (±12 step 0.2) | Regression (marginal) |
| 045 | 0.719 | 2-pass iterative grid-scale refinement during quantization | Regression |
| 046 | 0.717 | Greedy per-element level perturbation after LS refinement | Regression |
| 047 | 0.716 | Inlier-Focus K-means (Trimmed K-means, 10% outlier discarding) | Null (identical KL) |

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
- **Best KL**: 0.668342 (exp-081)
- **Unsloth target**: 0.721 (surpassed by 7.3%)
- **New directions opened**: Weight exponent tuning continues to deliver improvements. Moving from 0.5→0.4→0.35 gave cumulative 3.3% improvement. The optimal exponent may be even lower (0.3, 0.25) or adaptive per-sub-block.
- **Quantize time**: 356s (5.9 min) — exceeds 5-min soft limit due to powf vs sqrtf overhead. Could be optimized back to sqrtf + rescaling or using a fast pow approximation.
- **Index-scale coupling remains fragile**: Any two-way modification (changing both indices and scale) still causes catastrophic regression.
- **Remaining headroom**: Weight exponent 0.3 or 0.25 is the most promising next step. The improvement has been accelerating (0.5→0.4: -1.27%, 0.4→0.35: -2.05%), suggesting the true optimal may be well below 0.35.

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

## Remaining Research Directions
1. **Combined d+level joint optimization in a single pass**: Modify the d optimization to evaluate levels using the same exhaustive criterion. Safe because both d and level are optimized simultaneously. Computational cost: ~16x d optimization (adds ~10-15s, still within 5 min). **This is the most promising near-term direction.**
2. **Multi-codebook quantization**: Requires format changes but could provide step-change improvement.
3. **Gain-shape quantization**: Format-level decoupling of magnitude and direction. Speculative.
