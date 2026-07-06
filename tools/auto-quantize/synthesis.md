# IQ2_XXS Auto-Research Synthesis — Session 2026-07-06 (Final: exp-093 best KL 0.662001)

## Status: Best KL = 0.662001 (exp-093, waux=powf(weight,0.20)). System is at a genuine local optimum.

## Plateau broken by waux softening

After 12 consecutive null/regression experiments (exp-083 through exp-095), **exp-093 broke the plateau** by softening the waux weight from `sqrtf(weight)` (effective exponent 0.15) to `powf(weight, 0.20)` (effective exponent 0.06). The neighbor search for off-map patterns (~5% of chunks) benefits from more uniform weighting, letting the global grid structure dominate centroid selection over noisy per-element importance estimates.

| Experiment | KL Divergence | PPL | Same Top P | Q-Time | Key Technique | Verdict |
|-----------|--------------|-----|------------|--------|---------------|---------|
| **Unsloth target** | 0.721 | 26.44 | 60.25% | — | Reference | — |
| **exp-093 (BEST)** | **0.662001** | **25.25** | **62.00%** | **5.9 min** | **waux=powf(weight,0.20)** | **BEST** |
| exp-092 | 0.680119 | 25.37 | 61.01% | 6.0 min | Adaptive weight exponent per sub-block via CV | REGRESSION |
| exp-091 | 0.681633 | 25.38 | 61.68% | 5.9 min | Outlier-robust sigma2 using trimmed mean | REGRESSION |
| exp-090 | 0.682228 | 25.48 | 61.24% | 6.0 min | Remove sqrtf from waux (neighbor weight alignment) | REGRESSION |
| exp-089 | 0.679015 | 25.47 | 61.17% | 5.0 min | Intra-sub-block weight normalization | REGRESSION |
| exp-088 | 0.794320 | 29.13 | 58.99% | 5.9 min | Weight ratio clamping per sub-block | REGRESSION |
| exp-087 | 0.723409 | 26.93 | 58.72% | 5.0 min | Linear-L1 weight formula (1+\|xb\|) | REGRESSION |
| exp-086 | 0.669495 | 25.22 | 62.09% | 5.9 min | d-opt/post-d exponent 0.40 | REGRESSION |
| exp-085 | 0.669486 | 25.26 | 62.07% | 5.9 min | d-opt/post-d exponent 0.50 | REGRESSION |
| exp-084 | 0.673372 | 25.34 | 61.43% | 5.9 min | Harmonize all to 0.30 | REGRESSION |
| exp-083 | 0.679018 | 25.63 | 61.04% | 5.9 min | Weight exponent 0.25 | REGRESSION |
| exp-082 | 0.666162 | 25.11 | 61.96% | 5.9 min | Weight exponent 0.30 | PREV BEST |
| exp-081 | 0.668342 | 25.01 | 61.58% | 5.9 min | Weight exponent 0.35 | IMPROVEMENT |
| exp-080 | 0.682340 | 25.41 | 61.11% | 5.9 min | Weight exponent 0.4 | IMPROVEMENT |
| exp-078 | 0.691085 | 25.69 | 61.29% | 5.0 min | Per-sub-block sigma2 | IMPROVEMENT |

## Key Finding: waux asymmetry at optimal separation

Exp-093's success reveals a new dimension — **waux softening**. The waux effective exponent was tuned through 4 experiments:
| waux formula | Eff. exp | KL | Verdict |
|-------------|---------|-----|---------|
| weight (no sqrt) | 0.30 | 0.682 | REGRESSION (exp-090) |
| sqrt(weight) | 0.15 | 0.666 | (exp-082 baseline) |
| powf(weight, 0.20) | 0.06 | 0.662 | BEST (exp-093) |
| powf(weight, 0.10) | 0.03 | 0.668 | REGRESSION (exp-098) |
| 1.0 (uniform) | 0.00 | 0.675 | REGRESSION (exp-096) |

The optimum at effective exponent 0.06 is sharp — neighbors at 0.03 and 0.15 both regress. The three-stage weight profile is now:
- **Pre-d neighbor search (waux)**: eff. exp 0.06 — grid-structure-dominated centroid selection
- **Main quantization (weight)**: exp 0.30 — balanced importance-magnitude trade-off
- **Post-d neighbor (wtmp)**: eff. exp 0.35 — sharp importance-focused index refinement

The widening of the pre-d/post-d asymmetry (0.06 vs 0.35) helps: pre-d casts a wide net with soft weights, post-d selects precisely with sharp weights.

## What Worked
- **waux=powf(weight,0.20)** (exp-093): KL 0.662001 (−0.62% from 0.666162)
- **Weight exponent 0.30 main / 0.35 d-opt/post-d** (exp-082): KL 0.666162
- **Weight exponent 0.35 / 0.35** (exp-081): KL 0.668342
- **Weight exponent 0.4** (exp-080): KL 0.682340 (−1.27% from 0.691085)
- **Per-sub-block sigma2** (exp-078): KL 0.691085
- **65-candidate d optimization 0.5% step** (exp-064): KL 0.699009
- **Post-d grid index recomputation** (exp-055): KL 0.710657
- **Superblock d optimization** (exp-049): KL 0.715144
- **nwant=2→4** (exp-033): KL 0.7166

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

## Current Code State (exp-093 — best KL 0.662001)
- `nwant`: 8
- `kmeans_iters`: 20
- `num_trials`: 1 (E8 warm-start only)
- `max_samples`: 16384
- Single global grid with K-means (E8 warm-start)
- Float-space L1-weighted K-means (standard)
- Error-aware int8 snap (allows any value 0-127, no odd constraint)
- 0 rounds of multi-round refinement
- **Per-sub-block sigma2** for weight formula (exp-078)
- **Weight exponent 0.30** main (`powf(sigma2+xb^2, 0.30f)`), **0.35** d-opt/post-d (exp-082)
- **waux = powf(weight, 0.20f)** — effective exp 0.06 (exp-093)
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

## Recent Experiments 094-099 (All Null/Regressions)

| Exp | KL | Technique | Verdict |
|-----|-----|-----------|---------|
| 094 | 1.0401 | Post-refinement level recomputation with index re-verification | CATASTROPHIC |
| 095 | 0.6662 | Align K-means weights with quantizer formula | Null (identical to 082) |
| 096 | 0.6754 | Uniform waux=1.0 (zero importance) | REGRESSION (+2.0%) |
| 097 | 0.6873 | Soften post-d wtmp with powf 0.20 | REGRESSION (+3.8%) |
| 098 | 0.6679 | Finer waux powf(weight,0.10) | REGRESSION (+0.88%) |
| **099** | **0.6799** | **L1-based weight formula (mean_abs+|xb| instead of sigma2+xb²)** | **REGRESSION (+2.7%)** |

### exp-099: L1-based weight formula — L2 metric confirmed essential
The weight formula's L2-based magnitude `sigma2 + xb²` (mean of squares) is structurally necessary. Replacing with L1-based `mean_abs + |xb|` regressed KL from 0.662 to 0.680 (+2.7%). The L2 baseline amplifies large-magnitude elements more aggressively than L1, providing essential per-element selectivity for the d optimization and post-d refinement. The sigma2 baseline's amplification is exactly what makes the weight formula effective — the L2-based form `(sigma2 + xb²)^p` with p=0.30 provides the right compromise between uniformity and selectivity.

## Gap Analysis
- **Best KL**: 0.662001 (exp-093, waux=powf(weight,0.20))
- **Unsloth target**: 0.721 (surpassed by 8.2%)
- **Weight formula EXHAUSTED**: All exponent values (0.25-1.0), asymmetric profiles, structural variants (L1-based, 1+|xb| linear) tested. Optimal: `qw * powf(sigma2 + xb^2, 0.30f)` main, 0.35 d-opt/post-d. **L2-based sigma2 confirmed essential.**
- **waux dimension EXHAUSTED**: powf(weight, 0.20) (eff 0.06) is optimal; softer/harder both regress.
- **Quantize time**: ~355-358s (~5.9 min) — slightly exceeds 5-min soft limit.
- **Index-scale coupling remains fragile**: Any two-way modification (changing both indices and scale) still causes catastrophic regression.
- **System is at a genuine local optimum**: All accessible knobs (weight exponent, sigma2 granularity, sigma2 metric L2 vs L1, waux, d step/range, neighbor depth, post-d refinement, K-means training) have been optimally tuned. No further improvement possible within current IQ2_XXS format without structural changes.

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
