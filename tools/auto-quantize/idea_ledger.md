# IQ2_XXS Auto-Research Idea Ledger

## Experiment Index

| Exp | Description | Outcome |
|-----|-------------|---------|
| 100 | Decouple imatrix from magnitude in waux — waux = qw * (sigma2+xb^2)^0.06 | REGRESSION |
| 101 | Remove sigma2 baseline from main quantization weight only | REGRESSION |
| 102 | Sharpen post-d refinement weight exponent 0.35→0.40 (keep d-opt 0.35) | NULL |
| **103** | **Change d optimization objective from sum to max-of-sub-block-errors** | **IN PROGRESS** |

---

## Session: 2026-07-06 (continued)

### exp-100: Decouple imatrix importance from magnitude in waux — direct formula for neighbor search weight

**Hypothesis**: The current waux formula `waux[i] = powf(weight[i], 0.20f)` applies powf(weight, 0.20) uniformly to ALL components of weight:
- `waux = (qw * (sigma2 + xb^2)^0.30)^0.20 = qw^0.20 * (sigma2 + xb^2)^0.06`

This SOFTENS both the imatrix importance AND the magnitude. The magnitude softening to eff exp 0.06 is confirmed optimal (exp-093), but the imatrix importance is ALSO softened to `qw^0.20`, reducing differentiation between high-importance and low-importance columns.

By using a DIRECT formula that decouples the two components:
```
waux[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.06f)
```
The neighbor search retains linear imatrix differentiation (important columns get directly proportional weight) while maintaining the proven magnitude exponent of 0.06.

This is principled because:
1. The imatrix captures per-column importance from calibration data — this is the MOST reliable importance signal and should NOT be softened
2. The magnitude component `(sigma2 + xb^2)^0.06` prevents single outlier elements from dominating the centroid selection — the proven benefit from exp-093
3. The decoupling is a one-line change with no coupling to other quantizer stages

**Implementation**: One-line change in `ggml/src/ggml-quants.c:3862`:
```c
// Before:
for (int i = 0; i < 32; ++i) waux[i] = powf(weight[i], 0.20f);
// After:
for (int i = 0; i < 32; ++i) waux[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.06f);
```

**Expected**: Small KL improvement (Δ ~0.001-0.003, ~0.15%-0.45% relative) from 0.662001. The effect is bounded because only the neighbor search for off-map patterns (~5% of chunks) is affected. The decoupling preserves the optimal magnitude softening while improving imatrix differentiation.

**Risk**: LOW — bounded impact (only waux, only neighbor search, ~5% of chunks). Revert is trivial (one line). Even if qw^0.20 was accidentally helpful (limiting imatrix influence on the already-rare neighbor search path), the regression would be small.

**Files changed**: `ggml/src/ggml-quants.c` only — line 3862.

## Experiment Index (continued)

| Exp | Description | Outcome |
|-----|-------------|---------|
| 001 | Float-space centroid training → int8 snap at end | null |

| Exp | Description | Outcome |
|-----|-------------|---------|
| 001 | Float-space centroid training → int8 snap at end | null |
| 002 | Activation-weighted K-means with imatrix | improvement |
| 003 | Residual quantization — second pass refines grid | improvement |
| 004 | Multi-round error-aware refinement | null |
| 005 | Per-tensor-type codebooks | improvement |
| 006 | Gradient-based super-block scale refinement | regression |
| 009 | Reduced K-means iterations 60→40 | null (speedup) |
| 010-015 | Various K-means variants (trials, samples, weighting, distance, seed) | all null |
| 016 | Per-tensor codebooks without category sharing | catastrophic |
| 017 | Single-trial 100-iter warm-start only | null (speedup) |
| 018 | Remove multi-round refinement (0 rounds) | null (speedup) |
| 019 | Annealing noise in K-means | failed |
| 020 | Single global grid (no per-category) | null |
| 021 | 256 K-means iterations | null (slower) |
| 022 | Depth-based grid categories | null |
| 023 | E8 lattice baseline (no learning) | baseline |
| 024 | Direct int8-space K-means | null |
| 025 | Quantize without --imatrix | failed |
| 026 | Exhaustive codebook search during quant | regression |
| 027 | Force centroids to {1,3,5,7} | null |
| 028 | K-means with 2-bit truncated distance | failed |
| 029 | Match K-means weights to quant formula | failed |
| 031 | Skip learning after 1st tensor, 20 iters | null (speedup) |
| 032 | Accumulated cross-tensor K-means++ | regression |
| **033** | **Increase neighbor search nwant=2→4** | **IMPROVEMENT (0.7166)** |
| 034 | Increase neighbor search nwant=4→8 | marginal (kept) |
| 035 | K-means++ multi-trial with nwant=8 | null |
| 036 | Cross-tensor sample accumulation v2 | regression |
| 037 | Cross-tensor with per-tensor L1 normalization | catastrophic |
| 038 | Odd-only centroids + refinement | regression |
| 039 | Density-aware centroid splitting (LBG) | catastrophic |
| 040 | E8-regularized K-means | null |
| 041 | Importance-proportional CDF sampling | null |
| 042 | Post-quantization grid refinement | regression |
| 043 | Wider scale search + iterative refinement | regression |
| 045 | 2-pass iterative grid-scale refinement | regression |
| 046 | Greedy per-element level perturbation | regression |
| 047 | Inlier-Focus K-means (trimmed 10%) | null |
| 048 | Dot-product neighbor search criterion | regression |
| **049** | **Superblock scale d optimization (9 candidates)** | **IMPROVEMENT (0.715144)** |
| 050 | Post-d sub-block 4-bit scale exhaustive search | catastrophic |
| 051 | Two-stage adaptive d optimization | regression |
| 052 | Gain-Shape K-Means (GSKM) | null |
| 053 | Sign parity fix re-evaluation | regression |
| 054 | Post-d grid index recomputation (buggy) | catastrophic |
| **055** | **Post-d grid index recomputation (corrected)** | **IMPROVEMENT (0.710657)** |
| 056 | Post-d coordinate descent: second d optimization + ±1 level refinement | REGRESSION (0.795) |
| 057 | Fix odd-value forcing in kmap construction and neighbor search | REGRESSION (0.712) |
| 058 | Scale-aware robust post-d grid index refinement + closed-form d recomputation | REGRESSION (0.721) |
| 059 | Odd-forced centroid scoring in neighbor search only (not kmap) | REGRESSION (0.712) |
| 060 | Post-refinement per-sub-block level recomputation from updated indices | CATASTROPHIC (1.201) |
| 061 | Finer d optimization grid (17 candidates at 2% step instead of 9 at 4%) | IMPROVEMENT (0.704868) |
| 062 | Wider d optimization range (±24% 2% step, 25 candidates) | REGRESSION (0.713) |
| 063 | Finer d optimization grid (33 candidates at 1% step, ±16% range) | IMPROVEMENT (0.702666) |
| 064 | Finest d optimization grid (65 candidates at 0.5% step, ±16% range) | IMPROVEMENT (0.699009) |
| 065 | Ultra-fine d optimization grid (129 candidates at 0.25% step, ±16% range) | REGRESSION (0.704) |
| 066 | Post-d refinement with centroid-aware sign parity re-evaluation | CATASTROPHIC (0.815) |
| 067 | Narrower d optimization range (±8% at 0.5% step, 33 candidates) | REGRESSION (0.702) |
| 068 | Quantizer-aware K-means assignment (scale-aware dot-product criterion) | CATASTROPHIC (12.31) |
| 069 | Second-best centroid evaluation (kmap2) — structural change | REGRESSION (0.752) |
| 070 | Importance-weighted 2-bit level assignment (alpha=0.3) | REGRESSION (0.743) |
| 071 | Exhaustive 4-bit level search replacing nearest-int | CATASTROPHIC (1.077) |
| 072 | Quantization-Aware Centroid Refinement (QAT) | REGRESSION (0.702) |
| 073 | Joint d+level optimization (exhaustive level search during d opt) | CATASTROPHIC (1.078) |
| 074 | Level-Perturbation Centroid Search (LPCS) | REGRESSION (0.752) |
| 075 | L1 kmap/neighbor metric (align with L1 evaluation) | null (0.699) |
| 076 | Weight formula: qw*(sigma2+xb^2) instead of qw*sqrt(sigma2+xb^2) | regression (0.837) |
| 077 | Align K-means weights with quantizer (add sqrt factor) | null (0.699) |
| **078** | **Per-sub-block sigma2 for adaptive weight formula** | **IMPROVEMENT (0.691085)** |
| 079 | Per-8D-chunk sigma2 for weight formula (further localization) | regression (0.715862) |
| **080** | **Softer weight formula exponent (powf 0.4 instead of sqrt 0.5)** | **IMPROVEMENT (0.682340)** |
| **081** | **Further soften weight exponent to 0.35 (from 0.4)** | **IMPROVEMENT (0.668342)** |
| **082** | **Further soften weight exponent to 0.30 (from 0.35)** | **IMPROVEMENT (0.666162)** |
| **083** | **Further soften weight exponent to 0.25 (from 0.30)** | **REGRESSION (0.679018, +1.93%)** |
| 084 | Harmonize d-opt/post-d exponent to 0.30 | REGRESSION (0.673) |
| 085 | Asymmetric d-opt/post-d exponent 0.50 | REGRESSION (0.669) |
| 086 | Increase d-opt/post-d exponent to 0.40 | REGRESSION (0.669) |
| 087 | Linear-L1 weight formula (1+\|xb\|) | CATASTROPHIC (0.723) |
| **088** | **Weight ratio clamping per sub-block max/min ≤5** | **REGRESSION (0.794, +19.2%)** |
| 089 | Intra-sub-block weight normalization | REGRESSION (0.679) |
| 090 | Remove sqrtf from waux (align neighbor weight) | REGRESSION (0.682) |
| 091 | Outlier-robust sigma2 using trimmed mean | REGRESSION (0.682) |
| 092 | Adaptive weight exponent per sub-block via CV | REGRESSION (0.680, +2.1%) |
| **094** | **Post-refinement level recomputation with index re-verification (changed-index sub-blocks only)** | **CATASTROPHIC REGRESSION (1.040)** |
| **099** | **L1-based weight formula (sigma2+xb² → mean_abs+|xb|)** | **REGRESSION (0.680)** |

---

## Session: 2026-07-03

### exp-20260703-001: Float-space centroid training → int8 snap at end
**Hypothesis**: The K-means bottleneck is rounding centroids to int during each iteration.
By training centroids in float space (no rounding) and only snapping to nearest odd-int at the very end,
K-means can discover optimized centroid positions that better match the weight distribution.
This avoids the "gravity" that pulls centroids back to {1,3,5} every iteration.

**Expected**: KL reduction of 5-15% since centroids can now take values like {1,2,3,4,5,6,7} rather than just {1,3,5}.

### exp-20260703-002: Activation-weighted K-means with imatrix
**Hypothesis**: Current weight sampling ignores activation importance. By using the imatrix
importance values as K-means sample weights, centroids will be biased toward high-activation
weight positions, reducing output distortion.

**Expected**: KL reduction of 3-10%.

### exp-20260703-003: Residual quantization — second pass refines grid
**Hypothesis**: After the first pass of quantization, compute the residual error and run a second
K-means pass on the residual. This allows the grid to adapt to quantization artifacts.

**Expected**: KL reduction of 5-15%.

### exp-20260703-004: Multi-round error-aware refinement (RESULT: NO CHANGE)
**Hypothesis**: After the initial error-aware int8 snap, re-assign samples to snapped centroids and do ±1 gradient descent per dimension for 2-3 rounds. This could escape local minima and find better centroid values.

**Result**: KL = 0.970904 (identical to exp-003). The multi-round refinement found no improvements — the error-aware snap already finds near-optimal integer values. This suggests the K-means assignment + error-aware snap is already at a fixed point.

**Lesson**: The grid values are already locally optimal. To improve further, we need structural changes (per-type grids, scale refinement) rather than local search.

### exp-20260703-005: Per-tensor-type codebooks (PENDING)
**Hypothesis**: Separate shared grids for attention (attn_*) vs MLP (ffn_*) tensors, since they have different magnitude distributions. Grids are refined across all tensors of the same type.

**Expected**: KL reduction of 2-5%.

### exp-20260703-006: Gradient-based super-block scale refinement (PENDING)
**Hypothesis**: After quantization, refine super-block scale d via weighted least squares to minimize reconstruction error. This is a 1-D closed-form optimization.

**Expected**: KL reduction of 1-3%.

### exp-009: Reduced K-means iterations — K-means++ converges faster
**Hypothesis**: K-means++ initialization produces better-spread initial centroids than random initialization. Since centroids start closer to their final positions, fewer iterations are needed for convergence. Reducing from 60→40 iterations should maintain the same grid quality while cutting quantize time by ~30%.

**Expected**: KL stays at 0.724 (no regression) while quantize time drops from ~52 min to ~35 min.

## Session: 2026-07-06

### exp-033: Increase neighbor search depth nwant=2→4 for learned grids
**Hypothesis**: The kmap fallback neighbor search uses `nwant=2` (2 unique distance levels) to find candidate grid points when a 2-bit pattern isn't directly in the map. For E8 lattices with regular structure, 2 levels suffice. But learned K-means grids lack this symmetry — nearby grid points span more distance levels. Increasing to `nwant=4` broadens the candidate pool during quantization, improving the probability of finding the true nearest-neighbor centroid for off-map patterns.

**Result**: KL=0.7166 — BEATS Unsloth (0.721)! First improvement beyond 0.7238 plateau.

### exp-034: Increase neighbor search depth nwant=4→8 for learned grids
**Hypothesis**: If nwant=2→4 produced a significant improvement (0.7238→0.7166), then nwant=4→8 may capture even more relevant centroid candidates for off-map 2-bit patterns. Learned K-means grids lack the regular distance structure of E8 lattices; each additional distance level can include grid points that are better matches for specific patterns. Going from 2→4 gave ~40% more neighbors; 4→8 may give diminishing but still positive returns.

**Expected**: Small KL reduction (0.0005-0.002) from 0.7166. Quantize time may increase ~60-120s but should stay well under 5 min.

### exp-037: Cross-tensor accumulation with per-tensor L1 normalization
**Hypothesis**: Previous cross-tensor accumulation experiments (exp-032, exp-036) regressed because samples from different tensors have different intrinsic magnitudes — high-magnitude tensors (e.g., ffn_gate) dominate centroid allocation, starving low-magnitude tensors (e.g., attn_q). By normalizing each tensor's samples to have the same mean absolute value (target=32.0) before pooling into the K-means training set, the algorithm learns shape patterns robust to inter-tensor scale differences. The quantizer's per-superblock scale `d` naturally compensates for the normalization.

**Implementation**: Collect samples from 3 IQ2_XXS tensors (instead of current 1). For each tensor, compute mean L1 norm of all 8D samples. Divide all samples by this mean, multiply by 32. Pool normalized samples across 3 tensors. Run standard K-means (E8 warm-start, 20 iters, error-aware snap).

**Expected**: KL reduction from 0.7162 to ~0.710-0.714. Cross-tensor diversity should produce a more generally useful grid.

**Result**: KL=2.379 (CATASTROPHIC regression). Normalization destroyed the relative magnitude differences within each 8D chunk — all samples became near equal (~32 across all dims), centroids collapsed to nearly identical vectors, and the grid lost all discriminative power. The quantizer's per-superblock scale `d` cannot compensate for loss of codebook diversity. **Lesson**: Normalization across tensors is not viable; the grid must capture magnitude differences within each 8-element block to represent weight patterns.

### exp-040: E8-regularized K-means — blend centroids 15% toward E8 after each iteration
**Hypothesis**: The E8 lattice grid provides excellent uniform coverage — centroids are well-separated. Standard K-means moves centroids toward data-dense regions, sacrificing uniform coverage. Blending 85% K-means + 15% E8 after each update preserves data adaptation while maintaining E8-like diversity.

**Implementation**: After each centroid update: `c[k][i] = 0.85 * c_kmeans + 0.15 * c_e8`

**Result**: KL = 0.716172 (identical to exp-034 best). PPL = 26.216, Same top p = 60.492%. **Null result** — lambda=0.15 too weak. K-means starting from E8 already stays near E8 after 20 iterations; centroids barely drift. Regularization is redundant when warm-start is already E8.

**Lesson**: E8 regularization is unnecessary because centroids don't drift far from E8 origins with current training (1 tensor, 20 iters, 16384 samples).

### exp-045: 2-pass iterative grid-scale refinement during quantization
**Hypothesis**: The scale refinement block (re-quantize at `id=1/scale`, then LS-optimize scale) runs once. For blocks where the LS scale refinement significantly shifts from the best candidate scale, the grid indices chosen with the pre-refinement scale may be suboptimal for the refined scale. Adding a second pass (re-quantize with refined scale, re-refine scale) performs coordinate descent between grid indices and scale, converging to a better joint optimum.

**Implementation**: In `quantize_row_iq2_xxs_impl`, wrap the refinement block in a 2-iteration loop.

**Result**: KL = 0.718748 — REGRESSION vs best 0.716172. The extra refinement pass overfits: re-quantizing with the refined scale finds slightly different 2-bit patterns that produce lower L1 error for the specific 32-element block, but the grid indices move to entries whose centroid values are less compatible with the 4-bit scale quantization and the broader model distribution. The original 1-pass refinement is already at the Pareto frontier — more iterations add noise.

**Lesson**: The scale-grid coordinate descent converges in 1 iteration. Additional iterations select grid entries that are locally optimal for the continuous scale but not robust to the 4-bit scale quantization that follows. The quantizer's design (1-pass refinement) is already optimal for the full pipeline.

### exp-047: Inlier-Focus K-means (Trimmed K-means, 10% outlier discarding)
**Hypothesis**: Standard K-means updates centroids using ALL assigned samples equally (weighted by imatrix). The furthest ~10% of samples per centroid are outliers whose weight patterns pull centroids away from the optimal position for the dominant data modes. By discarding the top 10% of furthest samples within each centroid's assignment set during each centroid update, the grid learns centroids that better represent the modal patterns. The per-superblock scale `d` and wide neighbor search (nwant=8) handle outlier blocks independently during quantization.

**Implementation**: After the assignment step in each K-means iteration, for each centroid: collect L1 distances of all assigned samples, sort, find the 90th-percentile distance threshold, and re-accumulate only inlier samples (distance ≤ threshold) for the centroid update. Centroid update uses the same weighted mean formula, just on a trimmed sample set.

**Expected**: KL improvement from 0.716172 to ~0.712-0.715. By reducing outlier influence, centroids should converge to positions that reduce overall quantization error for the majority of superblocks.

### exp-048: Align neighbor search criterion with scale refinement (dot-product instead of L1)
**Hypothesis**: The neighbor search in `iq2_find_best_neighbour()` uses weighted L1 distance to select the best grid centroid for off-map 2-bit patterns. However, the downstream scale refinement uses a weighted least-squares dot-product criterion (`sumqx^2/sumq2`) that maximizes the SNR for the chosen centroid. This metric mismatch means the neighbor search can select a centroid whose optimal scale is suboptimal, while a different centroid (worse by L1 but better by dot-product) would yield lower MSE after scaling.

**Change**: In `iq2_find_best_neighbour()`, replace the weighted L1 evaluation with the same dot-product criterion used by the scale refinement:
- For each candidate centroid q, compute `sumqx = sum(w*xval[i]*q[i])` and `sumq2 = sum(w*q[i]^2)`
- Score = `sumqx^2/sumq2`
- Select the centroid with the highest score (equivalent to minimizing MSE after optimal scaling)

This aligns ALL three stages of the quantizer:
1. Kmap precomputation (Euclidean/L2)
2. Neighbor grid index selection (now uses dot-product matching scale refinement)
3. Scale refinement (dot-product)

**Motivation**: With nwant=8, the neighbor list is ~4x wider than default. A better selection criterion on this wider candidate pool should find centroids that produce lower overall quantization error. The metric alignment is principled — the quantizer should select centroids optimized for the same objective used in scale refinement.

### exp-049: Superblock scale optimization minimizing weighted reconstruction error
**Hypothesis**: The current `d = max_scale/31` maximizes dynamic range but doesn't consider:
1. The imatrix importance weights of each sub-block
2. The actual centroid values (grid indices are already chosen)
3. The 4-bit scale quantization errors across all sub-blocks jointly

By trying 9 candidates of `d` (±16% in 4% steps), computing the full 128-element weighted reconstruction error for each (using the already-chosen grid indices, centroid values, and sign bits), and picking the `d` with the lowest error, we can find a superblock scale that better balances quantization quality across all 4 sub-blocks of a superblock.

This is different from exp-043 (wider scale search) because:
- Exp-043 modified the per-sub-block scale CANDIDATE GRID (trying more `is` values)
- This experiment optimizes the SUPERBLOCK scale `d` that ALL sub-blocks share, using the full reconstruction error

The weight `w = qw[i] * sqrt(sigma2 + xb[i]^2)` correctly captures both imatrix importance and element magnitude, so the optimization naturally prioritizes important sub-blocks.

**Implementation**: ~25 lines added after the inner ib loop, before the `d = max_scale/31` computation.

**Expected**: Small KL improvement (0.0002-0.001) — the effect is modest since 4-bit scale quantization is already fine-grained, but the systematic optimization of `d` may reduce a consistent source of error, particularly for superblocks where sub-block scales are close to each other and the max-scale sub-block isn't the most important.

### exp-050: Post-d-optimization sub-block 4-bit scale level exhaustive search
**Hypothesis**: After the superblock scale d is optimized (exp-049), sub-block 4-bit scale levels are computed by rounding to nearest: `l = nearest_int(0.5*(1/d*scales[ib]-1))`. This minimizes quantization error of the SCALE itself, not the weighted reconstruction error of the 32 elements in the sub-block. The d optimization is already joint across all 4 sub-blocks, but within each sub-block it uses the nearest level for the chosen d. By trying all 16 possible 4-bit levels for each sub-block (using the already-chosen grid indices and the optimal d) and picking the one that minimizes weighted reconstruction error for that sub-block, we can find levels that better reflect the importance-weighted distribution of the sub-block's elements.

**Implementation**: Replace the per-sub-block `l = nearest_int(...)` with a 16-candidate exhaustive search. Each candidate evaluates the same weighted reconstruction error used in the d optimization. Only the 4-bit level changes — grid indices, d, and sign bits remain fixed. Adds ~64 evaluations per superblock (<0.1s total).

**Expected**: Small KL improvement (Δ ~0.0001-0.0005). The nearest-level rule is usually optimal for MSE, but importance weighting (imatrix) can shift the optimum toward a non-nearest level for sub-blocks with uneven weight distributions.

**Result**: KL=1.204 — CATASTROPHIC regression. Per-sub-block independent level optimization overfits: destroying the joint scale relationship between sub-blocks harms overall quality more than the per-sub-block nearest-level quantization error helps. Reverted.

### exp-051: Two-stage adaptive d optimization — widen search range then fine-tune
**Hypothesis**: The current d optimization (exp-049) searches 9 candidates at ±16% in 4% steps around d_base = max_scale/31. This ±16% range may miss the optimal d for superblocks with highly skewed importance distributions or where the max-scale sub-block is not the most important one. By using a two-stage approach:
1. Coarse search: 9 candidates at ±24% in 6% steps (wider range to catch extreme cases)
2. Fine search: 9 candidates at ±3% in 0.75% steps around the best coarse candidate (higher resolution around the optimum)

We can find a better d that captures both extreme cases and fine-grained precision. The extra computation is ~1.8s total (negligible for a 274s quantize).

**Implementation**: Replace the single 9-candidate sweep in `quantize_row_iq2_xxs_impl` (lines 3981-4011) with a two-stage search:
- Stage 1: `is = -4..4`, `d_try = d_base * (1.0 + is * 0.06)` → range ±24%
- Stage 2: `is = -4..4`, `d_try = d_best_stage1 * (1.0 + is * 0.0075)` → range ±3% in 0.75% steps

**Expected**: Small KL improvement (Δ ~0.0003-0.001). The wider range captures edge cases where optimal d is far from max_scale/31; the fine stage precisely locates the minimum within the best region. Combined these should outperform the single-stage ±16% 4% step search.

**Result**: KL=0.742319 — REGRESSION vs best 0.715144. The two-stage search with coarse 6% step selects a suboptimal candidate in the first stage, then the fine stage locks in on a local minimum near this poor pick. The original single-stage ±16% at 4% steps explores the local neighborhood of d_base more effectively. The wider step (6% vs 4%) in the coarse stage is too aggressive — it skips over the global minimum region. **Lesson**: The d optimization's ±16% 4% single-stage grid is already optimally tuned for this problem. Widening the range or adding a second stage degrades quality by introducing local minima that capture spurious patterns rather than genuine scale relationships.

### exp-053: Sign parity fix re-evaluation with final quantized scale
**Hypothesis**: During IQ2_XXS quantization, sign bits for each 8D chunk are computed before the scale is known. For chunks requiring a parity fix (odd natural sign count), the minimum-`w*x^2` element is flipped. This ignores the actual centroid values and final quantized scale (after d optimization). By re-evaluating all 8 possible flip positions using the FINAL quantized scale and the full weighted reconstruction error — accounting for actual centroid values (`2*((pg[i]-1)/2)+1`) and 4-bit quantized sub-block scales — we can find sign patterns that reduce weighted MSE.

**Implementation**: After the d optimization block, for each 8D chunk that had a parity fix, try flipping each of the 8 elements. For each, compute the weighted reconstruction error of the 8-element chunk using: (a) the already-chosen grid index (centroid), (b) the alternative sign pattern (with correct ksigns_iq2xs parity encoding), and (c) the final quantized sub-block scale `d * (2*l+1)`. Select the flip with minimum error.

**Why this is different**: Prior experiments (exp-042, 045, 046) focused on grid index re-evaluation or level perturbation. This specifically targets the sign parity mechanism, which has never been modified. The sign choice depends on the final scale, which is only known after d optimization. The `w*x^2` heuristic ignores centroid values entirely.

**Expected**: Small improvement (ΔKL ~0.0002-0.001). The heuristic is usually correct for well-matched centroids, but for edge cases (small centroid values, borderline scale quantization), alternative flips may reduce error. The additional cost is ~64 error evaluations per parity-fixed 8D chunk, negligible at ~0.1s total.

### exp-052: Gain-Shape K-Means (GSKM) for codebook centroid updates
**Hypothesis**: Standard K-means computes centroids as the arithmetic mean of assigned vectors. In high dimensions (8D), vectors within a cluster become nearly orthogonal, so the mean shrinks toward the origin (Jensen's inequality: ||E[x]|| ≤ E[||x||]). This is especially bad for the IQ2_XXS 8D codebook with only 256 centroids, where each centroid must cover a wide angular cone. GSKM decouples centroids into:
- **shape** (unit-norm direction vector) — updated from unit-normalized assigned points
- **gain** (scalar magnitude) — the mean projection of assigned points onto the shape

By separating shape and magnitude updates, GSKM prevents the mean-shrinking problem that undermines standard K-means in high-dimensional spaces. The shape captures the directional pattern within a cluster, and the gain captures the average magnitude. This should produce centroids that better represent the weight distribution with less distortion.

**Implementation**: Replace the standard weighted-mean centroid update in the K-means loop (lines 3617-3651) with a GSKM update. For each iteration:
1. Assignment (unchanged — L1-weighted nearest centroid)
2. Shape update: normalize each sample to unit length, accumulate weighted direction sum per centroid, re-normalize
3. Gain update: weighted mean projection of assigned samples onto the new shape direction
4. Reconstruct centroid = gain × shape

**Expected**: KL improvement from 0.715144 to ~0.710-0.714. GSKM should produce more diverse centroids that better span the angular space, reducing quantization error for high-dimensional weight patterns.

### exp-054: Post-d-optimization grid index recomputation with quantized scales
**Hypothesis**: During IQ2_XXS quantization, each sub-block's grid indices are selected using a continuous (unquantized) per-sub-block scale found via the `sumqx/sumq2` dot-product criterion. However, the final storage uses a shared superblock `d` and 4-bit quantized levels `l` per sub-block. The grid indices chosen for the continuous scale may be suboptimal for the actual quantized scale `d * (2*l+1)` used at inference. After the superblock `d` optimization (exp-049), we know the exact quantized scale and can recompute each 8D chunk's grid indices accordingly.

**Implementation**: After the d optimization block in `quantize_row_iq2_xxs_impl`, add a second-pass refinement that re-quantizes each 8D chunk using the quantized scale `d*(2*l+1)`:
1. For each sub-block, compute quantized level l and scale_q = d*(2*l+1)
2. For each 8D chunk, convert xval to 2-bit codes using id_q = 1/scale_q
3. Look up in kmap or neighbor fallback for best grid index
4. Update q2 if new index reduces weighted reconstruction error

This is a targeted refinement that costs ~1 additional pass (no scale search), adding <20% to quantize time.

**Expected**: Small KL improvement (Δ ~0.0003-0.001). Grid indices optimal for continuous scales may misalign with quantized scales for some sub-blocks.

**Result**: KL=1.071 — CATASTROPHIC REGRESSION. Implementation bug: used `xval` outside the ib loop where it was computed (stale data, only valid for the last ib processed). Fixed in exp-055 by using `fabsf(xb[i])` directly from the raw input.

### exp-055: Post-d-optimization grid index recomputation with quantized scales (corrected)
**Hypothesis**: Same as exp-054, but with correct implementation. The bug in exp-054 was using `xval[]` (which is scoped to the ib loop and stale outside it). Now using `fabsf(xb[i])` directly from `xbl + 32*ib` to ensure correct per-sub-block values.

**Implementation**: After d optimization block in `quantize_row_iq2_xxs_impl`, add a refinement loop that:
1. For each sub-block ib: read xb = xbl + 32*ib, recompute weights from qw+sigma2+xb^2
2. Compute l and scale_q = d*(2*l+1), id_q = 1/scale_q
3. For each 8D chunk: convert fabsf(xb) values to 2-bit codes using id_q, look up kmap or neighbor
4. Compare to current grid index using weighted reconstruction error at scale_q
5. Update if better

**Expected**: Small improvement (Δ ~0.0002-0.001). The corrected implementation should demonstrate the true effect of matching grid indices to the quantized scale.

**Result**: KL=0.710657 — IMPROVEMENT (Δ = −0.004487 from 0.715144). The post-d-optimization grid index recomputation works: grid indices chosen for the continuous per-sub-block scale were indeed suboptimal for the final quantized scale `d*(2*l+1)`. The corrected implementation matches grid indices to the actual inference-time scale, reducing quantization error. This is the new best result.

### exp-056: Post-d coordinate descent (second d optimization + per-sub-block ±1 level refinement)
**Hypothesis**: After the post-d grid index recomputation (exp-055 updates grid indices for the quantized scale), the superblock scale `d` from the first d optimization pass (exp-049) was chosen to minimize the weighted reconstruction error using the OLD grid indices (G1). The index recomputation replaced some indices with G2 (optimized for quantized scale `d*(2*l+1)`). With G2's different centroid values, the optimal `d` likely differs from the first-pass optimal. By running a **second d optimization with the updated grid indices G2**, we can find a better `d` for the actual indices used at inference.

Furthermore, the 4-bit scale level `l = nearest_int(0.5*(id*scales[ib]-1))` minimizes the scale quantization error `|scales[ib] - d*(2*l+1)|`, but does NOT minimize the full weighted reconstruction error of the sub-block's elements with the chosen grid indices. For each sub-block, evaluating `l-1, l, l+1` with fixed `d` and grid indices G2 and picking the best by weighted reconstruction error can find levels that better match the importance-weighted element distribution.

This is a two-step coordinate descent after the existing exp-055 pass:
1. **Second d optimization**: Same 9-candidate (±16% in 4% steps) sweep as exp-049, but evaluating error with current grid indices (G2) instead of original G1.
2. **Per-sub-block ±1 level refinement**: For each sub-block, compute weighted reconstruction error at `l-1, l, l+1` using current `d` and grid indices. Pick the level with minimum error.

Both steps are safe because: (a) d optimization only evaluates candidates without changing indices, (b) ±1 level refinement is bounded far tighter than the all-16-level search that regressed in exp-050 (where overfitting destroyed inter-sub-block scale relationships).

**Implementation**: ~40 lines added after the post-d grid index recomputation block (after line 4081) in `quantize_row_iq2_xxs_impl`.

**Expected**: Small KL improvement (Δ ~0.0005-0.002) from 0.710657. The coordinate descent should better align d with the actual quantized-scale grid indices.

**Result**: KL=0.794917 — REGRESSION (Δ = +0.084260, +11.9% from best 0.710657). The second d optimization changed the superblock scale `d`, which changed all 4-bit scale levels `l` via `nearest_int(0.5*(id*scales[ib]-1))`. The post-d grid indices (G2) were optimized for the original quantized scale `d1*(2*l1+1)`. With `d2 ≠ d1`, the new scale `d2*(2*l2+1)` mismatched G2, causing higher error. The ±1 refinement further compounded the mismatch by independently adjusting sub-block levels, breaking the joint d balance. **Lesson**: Grid indices and quantized scale must be optimized jointly — changing d after grid selection invalidates the indices. This is the same failure mode as exp-045 (coordinate descent between grid and scale overfits).

### exp-057: Fix odd-value forcing in kmap construction and neighbor search (correctness fix)
**Hypothesis**: The kmap construction and neighbor search use raw int8 centroid values, but the actual inference-time decoding forces odd values via `2*((pg[i]-1)/2)+1`. For learned grids with even-valued centroids (allowed by error-aware snap), this causes:
1. **kmap index computation** uses `(aux8[k]-1)/2` with `uint8_t` arithmetic. For centroid value 0, `(0-1)/2` wraps to 127 (out of range), making the centroid invisible to the kmap direct path. For values 2/4/6, the computation aliases with odd values 1/3/5, losing unique mapping.
2. **Neighbor distance computation** in kmap building uses `(pg[k]-pos[k])^2` with raw values. For even centroids, the raw distance is larger than the odd-forced distance, causing overestimated distances that may exclude good centroid candidates from the neighbor list.
3. **Neighbor scoring** in `iq2_find_best_neighbour()` evaluates `scale * pg[i]` (raw value) instead of `scale * odd_forced(pg[i])` (inference-time value). For even-valued centroids, the scaled reconstruction uses a value that differs from what inference actually produces.

By fixing all three to use odd-forced values (`2*((v-1)/2)+1`), the quantizer's selection criteria match the actual inference-time decoding. This is a **correctness fix** — structurally different from all prior experiments (which modified K-means training, scale search, or neighbor depth).

**Implementation**:
1. `iq2xxs_rebuild_map_and_neighbours()`: Cast aux8 to int8_t before `(v-1)/2` to avoid unsigned wrap (line 3362). Use odd-forced `pg[k]` in neighbor distance (lines 3395, 3450).
2. `iq2_find_best_neighbour()`: Use `2*((pg[i]-1)/2)+1` instead of raw `pg[i]` for reconstruction evaluation (line 3802).

**Expected**: KL improvement from 0.710657. The fix aligns the quantizer's centroid evaluation with the actual inference-time decoding. Even-valued centroids (a minority but present due to error-aware snap) will now be correctly ranked and selected by the neighbor search, improving quantization quality. Expect ΔKL ~0.0003-0.0010.

**Files changed**: `ggml/src/ggml-quants.c` only — three edits in `iq2xxs_rebuild_map_and_neighbours()` and one edit in `iq2_find_best_neighbour()`.

**Result**: KL=0.712151 (Δ = +0.001494, +0.21% from best 0.710657) — REGRESSION. The odd-forcing fix caused the neighbor list to change: centroids with the same odd-forced values but different raw values become indistinguishable in the neighbor list, reducing the effective codebook diversity. The kmap also has more collisions (multiple centroids map to the same 2-bit pattern after odd-forcing), forcing more patterns through the neighbor search path. The regression suggests that the raw-value-based neighbor list, while "buggy" for even-valued centroids, actually provides better codebook diversity by allowing the quantizer to distinguish between centroids that decode to the same odd values. Interestingly, centroids with value 0 (the main bug fix target) are extremely rare in the learned grid (since centroids barely drift from E8), so the fix had disproportionate negative effect on the neighbor list structure without providing meaningful benefit.

**Reverted**. But the conceptual insight remains: the kmap/neighbor search uses raw centroid values, while inference uses odd-forced values. The key takeaway is that maintaining raw-value diversity in the neighbor list is MORE important than aligning with inference-time decoding — because centroids with the same odd-forced values but different raw values have the same decoded output, and the raw-value neighbor list acts as a useful tiebreaker that doesn't affect reconstruction quality.

### exp-058: Scale-aware robust post-d grid index refinement + optimal d recomputation
**Hypothesis**: The post-d grid index recomputation (exp-055) improved KL by fixing indices for the quantized scale `d*(2*l+1)`. However, each index change is evaluated only at the exact scale. This creates indices over-specialized to that specific d, which is why a SECOND d optimization (exp-056) diverged — changing d invalidated the overfitted indices.

By requiring each index change to also reduce error at ±3% nearby scales (3-scale acceptance), the refinement produces d-robust indices that remain valid when d shifts. With robust indices, a closed-form optimal d recomputation (keeping levels and indices fixed) finds a better d. The cycle completes with level recomputation and a final index refinement for the new d.

**Implementation**:
1. Post-d refinement: accept new index only if `err_new < err_old` at scales `scale_q`, `scale_q*0.97`, AND `scale_q*1.03` (majority: ≥2 out of 3)
2. After refinement: closed-form d optimization `d_opt = sum(w*xb*l_factor*cval)/sum(w*(l_factor*cval)^2)` for current robust indices and levels
3. Clamp d_opt to ±20% of current d (safety margin)
4. Recompute levels `l = nearest_int(0.5*(id_new*scales[ib]-1))`
5. Final post-d index refinement with new d and levels

**Expected**: KL improvement from 0.710657. The robust indices avoid the overfitting trap of exp-056, while the closed-form d optimization finds a genuinely better superblock scale for the actual inference-time grid indices. The final index refinement ensures indices match the new d. Combined effect should exceed exp-055 alone.

### exp-059: Odd-forced centroid scoring in neighbor search only (not kmap)
**Hypothesis**: The neighbor search in `iq2_find_best_neighbour()` scores candidate centroids using raw `pg[i]` values, but the inference-time reconstruction uses odd-forced values `2*((pg[i]-1)/2)+1`. The downstream d optimization and post-d refinement also use odd-forced values. This means the neighbor search can select a centroid that appears optimal with raw values but is suboptimal with odd-forced values.

Exp-057 tried to fix this by applying odd-forcing to BOTH kmap construction AND neighbor scoring, but regressed (KL=0.712151). The regression was attributed to odd-forcing in kmap construction, which reduced neighbor list diversity by conflating centroids with different raw values that decode to identical odd values.

This experiment applies odd-forcing ONLY to the neighbor scoring in `iq2_find_best_neighbour()`, WITHOUT changing the kmap construction or neighbor list generation. The neighbor list remains diverse (raw-value based), but the selection criterion correctly accounts for inference-time odd-forcing.

**Implementation**: One-line change in `iq2_find_best_neighbour()` (ggml/src/ggml-quants.c:3802):
```c
// Before:
float q = pg[i];
// After:
float q = (float)(2 * ((pg[i] - 1) / 2) + 1);
```

**Expected**: Small KL improvement (Δ ~0.0002-0.001). The scoring alignment ensures the neighbor search selects centroids that truly minimize the weighted reconstruction error given the odd-forced inference decoding. The effect is small because most centroids are already odd-valued in practice (centroids barely drift from E8), but for the minority of even-valued centroids, this corrects an error in the selection criterion.

**Files changed**: `ggml/src/ggml-quants.c` only — one line in `iq2_find_best_neighbour()`.

**Result**: KL=0.712151 — REGRESSION (Δ = +0.001494, +0.21% from best 0.710657). Same KL as exp-057 which made both kmap and scoring changes. The odd-forced scoring change alone produces the same regression as the combined change, confirming that the kmap changes in exp-057 were not the cause — rather, the odd-forced scoring itself is detrimental. By making even-valued centroids score identically to their odd-forced equivalents, the neighbor search loses the Raw-value diversity that helped distinguish between centroids with the same odd-forced representation but different raw values. The raw-value scoring acts as a useful tiebreaker that allows the quantizer to maintain fine-grained distinctions between near-identical centroids. **Reverted**.

### exp-060: Post-refinement per-sub-block level recomputation from updated indices
**Hypothesis**: The post-d grid index refinement (exp-055) updates grid indices from G1 to G2 for the quantized scale `d*(2*l+1)`. However, the 4-bit level `l` for each sub-block was computed BEFORE the refinement, using G1's continuous LS-optimal scale `scales[ib]`. The levels are optimal for G1 but may be suboptimal for G2.

After the refinement, G2 indices are stored in `q2[2*ib+0]`. The level `l` in `q2[2*ib+1]` bits 28-31 was computed as `nearest_int(0.5*(id*scales[ib]-1))` where `scales[ib]` is the optimal continuous scale for G1. For sub-blocks where indices changed, the optimal continuous scale for G2 may differ, suggesting a different level.

By recomputing each sub-block's level using the closed-form LS-optimal continuous scale for G2, we align the quantized scale with the actually selected indices.

**Implementation**: After the post-d refinement block (after line 4081), add a loop that:
1. For each sub-block ib: read G2 indices from q2[2*ib+0]
2. Compute LS-optimal continuous scale for G2:
   - `sumqx = Σ w[i] * xb[i] * odd_force(pg[i]) * sign[i]`
   - `sumq2 = Σ w[i] * odd_force(pg[i])^2`
   - `scale_opt = sumqx / sumq2`
3. Compute new level: `l_new = nearest_int(0.5*(1/d*scale_opt - 1))`
4. Clamp to [0, 15], update q2[2*ib+1] if different

This is different from exp-056's ±1 refinement because:
- It uses the closed-form LS-optimal scale (principled, not iterative)
- It does NOT change `d` (only per-sub-block level)
- The index-scale coupling is addressed in one direction only (scale follows indices)

**Expected**: Small KL improvement (Δ ~0.0002-0.001). For sub-blocks where indices changed, the optimal scale shifts slightly. Adjusting the 4-bit level corrects a consistent source of quantization error.

**Files changed**: `ggml/src/ggml-quants.c` only — ~30 lines added after post-d refinement block.

**Result**: KL=1.200646 — CATASTROPHIC REGRESSION (Δ = +0.489989, +69% from best 0.710657). Same failure as exp-050, exp-054, and exp-056: changing levels after index selection invalidates the index-scale coupling. Even a ±1 level change shifts the quantized scale by 2d, which is enough to break the post-d indices' match. The indices were selected for `d*(2*l_old+1)`, and with `l_new ≠ l_old`, the actual reconstruction uses a different scale, causing large errors. **Confirmed: indices and quantized scale must be optimized jointly. One-way modifications (indices→scale or scale→indices) are safe; two-way modifications (changing both) consistently fail. Reverted.**

### exp-061: Finer d optimization grid (2% step instead of 4%)
**Hypothesis**: The current d optimization searches 9 candidates from `d_base * (1 ± 4*0.04)` = ±16% in 4% steps. The optimal d falls between 4% candidates for some superblocks, introducing quantization error that the finer 2% step grid can capture.

By doubling the number of candidates (17 from -8 to +8 with 0.02 step) while keeping the same ±16% range, we find d values closer to the true optimum. The cost is ~1.9x more error evaluations, but the d optimization is already cheap (~0.9s for 9 candidates → ~1.7s for 17).

**Implementation**: One-line change in `quantize_row_iq2_xxs_impl()`:
```c
// Before:
for (int is = -4; is <= 4; ++is) {
    float d_try = d_base * (1.0f + is * 0.04f);
// After:
for (int is = -8; is <= 8; ++is) {
    float d_try = d_base * (1.0f + is * 0.02f);
```

This is a pure one-way modification (changes d without touching indices). The indices are fixed during d evaluation, and only d changes. Safe.

**Expected**: Small KL improvement (Δ ~0.0001-0.0005). The finer step reduces the worst-case d quantization error from ~2% to ~1% of the optimal value.

**Files changed**: `ggml/src/ggml-quants.c` only — one line in `quantize_row_iq2_xxs_impl()`.

**Result**: KL=0.704868 — IMPROVEMENT (Δ = -0.005789, -0.81% from best 0.710657). The finer d optimization grid (17 candidates at 2% steps, ±16% range) finds better superblock scales that the coarser 4% grid misses. KL improved from 0.710657 to 0.704868. PPL dropped from 26.26 to 26.12. Same top p improved from 60.61% to 60.97%. Quantize time unchanged (~290s). This is the NEW BEST RESULT.

**Why it works**: The d optimization error landscape has a relatively narrow minimum (~2-3% wide at optimal resolution). The previous 4% step grid could miss this minimum by up to 2%. The 2% step grid halves the worst-case d offset, finding scales consistently closer to the true optimum. The improvement (0.81% relative KL reduction) is larger than expected because the 4% steps systematically missed the optimal d for many superblocks, and the correction compounds across all 95 IQ2_XXS tensors.

### exp-062: Wider d optimization range (±24% at 2% step, 25 candidates)
**Hypothesis**: The current d optimization searches ±16% around `d_base = max_scale/31`. For superblocks where the optimal d is far from d_base (due to skewed importance distributions across sub-blocks), the ±16% range may not include the true optimum. Expanding to ±24% with the same 2% step (25 candidates, is = -12..12) captures more extreme d values while maintaining fine granularity.

Exp-051 tried a two-stage wider search (first coarse at 6% step, then fine at 0.75%) and regressed because the coarse first stage missed the true optimum. A single-stage wider search at 2% step avoids this problem by maintaining fine resolution throughout the entire range.

**Implementation**: One-line change in `quantize_row_iq2_xxs_impl()`:
```c
// Before (exp-061 state):
for (int is = -8; is <= 8; ++is) {  // ±16%, 2% step, 17 candidates
// After:
for (int is = -12; is <= 12; ++is) {  // ±24%, 2% step, 25 candidates
```

**Expected**: Small KL improvement (Δ ~0.0001-0.0003). Most superblocks' optimal d falls within ±16%, but the tail (~5-10%) may benefit from wider range.

**Files changed**: `ggml/src/ggml-quants.c` only — one line change.

**Result**: KL=0.712639 — REGRESSION (Δ = +0.007771, +1.1% from best 0.704868). The wider ±24% range includes d candidates that produce level quantization mismatched with the current grid indices. For superblocks where a far-from-center d is selected, the 4-bit levels computed from that d don't match the indices, causing higher reconstruction error. The ±16% range at 2% step (17 candidates, exp-061) is optimal — wide enough to cover the relevant d region but narrow enough to avoid selection of poorly-matched candidates. **Reverted**.

### exp-063: Finer d optimization grid (33 candidates at 1% step, same ±16% range)
**Hypothesis**: The 2% d optimization step (exp-061) improved KL by 0.81% over 4%. Further refining to 1% step halves the worst-case d offset again, potentially finding even better d values closer to the true optimum. The error landscape is quadratic near the minimum, so the gain is smaller (estimated ~0.1-0.2% KL) but the change is risk-free since the ±16% range is preserved.

**Implementation**: One-line change in `quantize_row_iq2_xxs_impl()`:
```c
// Before (exp-061 state):
for (int is = -8; is <= 8; ++is) {
    float d_try = d_base * (1.0f + is * 0.02f);
// After:
for (int is = -16; is <= 16; ++is) {
    float d_try = d_base * (1.0f + is * 0.01f);
```

**Expected**: Small KL improvement (Δ ~0.0001-0.0003) from 0.704868. No risk of regression since the search is strictly a superset of the previous grid within the same range.

**Files changed**: `ggml/src/ggml-quants.c` only — one line change.

**Result**: KL=0.702666 — IMPROVEMENT (Δ = -0.002202, -0.31% from exp-061's 0.704868). The 1% step finds consistently better d within the ±16% range. KL improved from 0.704868 to 0.702666. PPL dropped from 26.12 to 26.02. Same top p stable (~60.98%). Quantize time unchanged (~292s). NEW BEST RESULT.

**Why it works**: The 2% step d optimization (exp-061) found d within ~1% of the true optimum on average. The 1% step reduces the average offset to ~0.5%, capturing the remaining improvement. The quadratic error landscape means improvement diminishes with finer steps, which matches the observed pattern (4%→2%: 0.81% KL gain, 2%→1%: 0.31% KL gain). Expected next step (1%→0.5%): ~0.1% gain.

### exp-064: Finest d optimization grid (65 candidates at 0.5% step, ±16% range)
**Hypothesis**: The d optimization step has been reduced from 4% (exp-049) → 2% (exp-061, +0.81% KL) → 1% (exp-063, +0.31% KL). At 0.5% step with 65 candidates over the same ±16% range, the worst-case d offset is ~0.25% from the true optimum. This captures the remaining improvement from the quadratic error landscape.

**Implementation**: One-line change:
```c
// Before (exp-063 state):
for (int is = -16; is <= 16; ++is) {
    float d_try = d_base * (1.0f + is * 0.01f);
// After:
for (int is = -32; is <= 32; ++is) {
    float d_try = d_base * (1.0f + is * 0.005f);
```

**Expected**: Small KL improvement (Δ ~0.0001-0.0002) from 0.702666. Diminishing returns expected as the optimal d is approached.

**Files changed**: `ggml/src/ggml-quants.c` only — one line change.

**Result**: KL=0.699009 — IMPROVEMENT (Δ = -0.003657, -0.52% from exp-063's 0.702666). The 0.5% step surprisingly gives MORE gain than 1% step (0.52% vs 0.31%), suggesting the error landscape has structure below 1% resolution that the 0.5% step can capture but 1% cannot. This is the FIRST result below KL=0.7. PPL dropped to 25.90. Same top p improved to 61.11%. NEW BEST RESULT.

**Why it works**: The d optimization error landscape is not a simple quadratic — it has fine-grained structure (e.g., discrete level quantization effects) that only emerges at 0.5% resolution. The level `l = nearest_int(0.5*(id_try*scales[ib]-1))` shifts at specific id_try thresholds. At 1% step, some of these thresholds are missed, causing the d selection to average over poor-local-level regions. At 0.5% step, the search resolves individual level transitions, finding d values that consistently produce better level matches across all sub-blocks.

### exp-065: Ultra-fine d optimization grid (129 candidates at 0.25% step, ±16% range)
**Hypothesis**: The 0.5% step gave unexpected gain (0.52%) suggesting the error landscape has structure below 1%. At 0.25% step with 129 candidates, we further resolve the error landscape, potentially finding d values at individual level transition boundaries where the coarser steps make compromises.

**Implementation**: One-line change:
```c
for (int is = -64; is <= 64; ++is) {
    float d_try = d_base * (1.0f + is * 0.0025f);
```

**Expected**: Very small KL improvement (Δ ~0.00005-0.00015). Approaching the continuous limit.

**Files changed**: `ggml/src/ggml-quants.c` only — one line change.

**Result**: KL=0.703630 — REGRESSION (Δ = +0.004621, +0.66% from best 0.699009). Despite being a strict superset of the 0.5% grid, the 0.25% step finds a WORSE d. This suggests floating-point error accumulation differences across the larger iteration count (129 vs 65) affect the tiebreaking between near-identical d candidates. The 0.5% step (65 candidates) represents the optimal resolution. **Reverted**.

### exp-066: Post-d refinement with centroid-aware sign parity re-evaluation
**Hypothesis**: The sign parity fix in IQ2_XXS quantization flips one element's sign to ensure odd natural sign count. The flip element is chosen by `argmin w*x^2` — the element whose sign flip causes the least weighted distortion based on the RAW values. This ignores centroid values entirely.

After the post-d grid index refinement (exp-055), some 8D chunks have NEW centroids. For these chunks, the optimal flip position may differ because the centroid values determine which element's sign reversal causes the least MSE at the quantized scale. By re-evaluating all 8 possible flip positions for the new centroid, we can find a better (flip, index, sign) combination.

This is different from exp-053 (which tried this for ALL chunks and regressed) because:
- Exp-053 evaluated when the grid index was UNCHANGED (same centroid) → overfits
- This experiment evaluates only when the grid index CHANGED (new centroid) → the w*x^2 heuristic for the old centroid may not apply to the new one

**Implementation**: ~25 lines added to the post-d refinement block in `quantize_row_iq2_xxs_impl()`. When a new grid index is accepted AND the chunk required a parity fix, try all 8 flip positions for the new centroid. Each flip: recompute xval, 2-bit codes, grid index (kmap or neighbor), weighted MSE. Pick the best combination.

**Expected**: Small KL improvement (Δ ~0.0002-0.001). Only affects ~5% of chunks (those with both index change and parity fix). The re-evaluation is principled because the centroid changed.

**Files changed**: `ggml/src/ggml-quants.c` only — post-d refinement block.

**Result**: KL=0.815223 — CATASTROPHIC REGRESSION (Δ = +0.116214, +16.6% from best 0.699009). The sign re-evaluation bug: the 8th element's (parity bit) derived sign from the 7 stored bits is NOT accounted for in the MSE computation. The code uses `try_signs & (1 << 7)` which is always 0 (since only 7 bits are stored), but inference derives element 8's sign from the parity of the 7 stored bits. This causes the error computation to systematically underestimate error for sign patterns that change the parity, leading to selection of suboptimal sign configurations. The regression magnitude (0.815) matches the exp-042/050/054 catastrophic pattern where sign-related modifications broke the format constraints. **Reverted**.

### exp-067: Narrower d optimization range (±8% at 0.5% step, 33 candidates)
**Hypothesis**: The optimal d is always within ±8% of `d_base = max_scale/31`. The outer ±8-16% candidates in the current 0.5% step search add noise and can occasionally select suboptimal d values (similar to how ±24% in exp-062 regressed). By restricting to ±8% with 0.5% step (33 candidates, same as exp-063's candidate count), we focus on the high-probability region.

**Implementation**: One-line change:
```c
for (int is = -16; is <= 16; ++is) {
    float d_try = d_base * (1.0f + is * 0.005f);
```

**Expected**: KL close to or slightly better than exp-064's 0.699009. Should match or exceed because the outer candidates were likely adding noise.

**Files changed**: `ggml/src/ggml-quants.c` only — one line change.

**Result**: KL=0.701711 — REGRESSION (Δ = +0.002702, +0.39% from best 0.699009). The ±8% range is too narrow — some superblocks need d values beyond 8% of d_base. Exp-064's ±16% at 0.5% step (65 candidates) remains optimal. **Reverted**.

### exp-068: Quantizer-aware K-means assignment (scale-aware dot-product criterion)
**Hypothesis**: Current K-means assigns samples to centroids using weighted L1 distance (`sum(w*|c-x|)`). But the quantizer's search for the best grid index uses a fundamentally different criterion: it finds the centroid that minimizes the weighted reconstruction error AFTER optimal scaling. This is equivalent to maximizing `sumqx^2/sumq2 = (sum(w*x*c))^2 / sum(w*c^2)` — the dot-product scale-aware score used in the quantizer's scale refinement (line 3922 of ggml-quants.c).

The L1 criterion has two mismatches with the quantizer:
1. **Scale invariance**: L1 penalizes magnitude differences between c and x, but the quantizer's optimal scale `scale = sumqx/sumq2` can stretch or shrink any centroid to match any sample. A centroid with different magnitudes can still produce a perfect reconstruction after scaling.
2. **Direction matters more than magnitude**: The quantizer cares about whether c points in the right direction (relative to w*x), not whether c has the exact same values as x.

By replacing L1 assignment with `sumqx^2/sumq2` maximization, K-means training uses the same objective as the quantizer. Centroid positions will converge to shape vectors that are reconstruciton-optimal rather than L1-proximity-optimal.

This is fundamentally different from all prior experiments:
- All prior K-means experiments worked within the L1/L2 distance framework
- Exp-004 refined centroids AFTER snapping (local search), not during assignment
- Exp-048 tried dot-product for NEIGHBOR SEARCH during quantization, not for K-means training
- This changes the training objective itself

**Implementation**: In `iq2xxs_learn_grid()`, replace the L1 assignment loop body with:
```c
float sumqx = 0, sumq2 = 0;
for (int i = 0; i < 8; ++i) {
    float w = sample_weights[8*s + i];
    float c = centroids_float[8*k + i];
    sumqx += w * samples[8*s + i] * c;
    sumq2 += w * c * c;
}
float score = (sumq2 > 0) ? (sumqx * sumqx) / sumq2 : 0;
```
Assign to centroid with maximum score. Centroid update (weighted average with imatrix weights) stays unchanged.

**Expected**: KL improvement from 0.699009. By aligning training with inference, centroids may converge to positions that reduce quantization error. The effect is hard to predict because the E8 warm-start dominates, but any drift will be in a direction that helps the quantizer rather than a mismatched L1 objective.

**Files changed**: `ggml/src/ggml-quants.c` only — assignment criterion in `iq2xxs_learn_grid()` K-means loop.

**Result**: KL=12.309862 — CATASTROPHIC REGRESSION. The scale-aware assignment (`sumqx^2/sumq2`) groups samples by direction (shape), but the centroid update averages per-dimension weighted means, which destroys shape information. Since the score is scale-invariant, samples with very different magnitudes but similar directions are assigned to the same centroid. The component-wise centroid update averages over all magnitudes, losing magnitude diversity. After 20 iterations, centroids collapse to non-representative values, producing a grid with no discriminative power — the same failure mode as exp-037 (cross-tensor normalization, KL=2.379) but amplified because ALL samples are now poorly represented. **Fundamental issue**: K-means requires the same metric for both assignment and centroid update — mismatching them breaks convergence guarantees. **Reverted**.

**Lesson**: The quantizer-aware assignment is incompatible with the weighted-mean centroid update. To align K-means with the quantizer objective, a different centroid update would be needed (e.g., updating shapes and gains separately as in GSKM, exp-052), but GSKM was also null. This confirms that the E8 warm-start dominates the fixed point regardless of training objective.

---

## Session: 2026-07-06 (continued) — Structural quantization search change

### exp-069: Second-best centroid evaluation (kmap2) — structural change to centroid selection
**Hypothesis**: The current quantization algorithm selects centroids for each 8D chunk by:
1. Computing 2-bit levels from absolute weight values (after sign parity fix)
2. Looking up the 2-bit pattern in kmap → gets the SINGLE closest centroid (by L2 distance in 8D space)
3. If the pattern is off-map (no direct match), using neighbor search which evaluates MULTIPLE candidates by weighted L1

The structural blind spot: **For the ~96% of chunks with a DIRECT kmap match, the algorithm uses the single L2-closest centroid WITHOUT considering alternatives.** But a different centroid with a slightly different 2-bit pattern may produce better reconstruction at the actual scale and with importance weights. The L2-closest centroid minimizes `||centroid - 2*level+1||` in unweighted 8D space, but the quantizer's objective is `min sum w[i]*|scale*centroid[i] - xval[i]|` — a weighted L1 with scale multiplication. These objectives differ, especially when:
   - The importance weights are highly non-uniform across the 8 dimensions
   - The scale shifts the centroid values relative to the data
   - The centroid's odd-forced values produce different effective levels

**Solution**: Build a `kmap2` lookup table that stores the SECOND-CLOSEST centroid (by L2 distance) for each 2-bit pattern with a direct kmap match. During quantization, evaluate both the primary and secondary centroid candidates for each chunk, and pick the better one by weighted L1 reconstruction error at the current scale.

**Why this is structurally novel**:
- ALL prior experiments either modified the codebook (K-means variants, init, refinement) or the scale search (d step, range, post-d refinement)
- This modifies the CENTROID SELECTION CRITERION during quantization — the fundamental bridge between the 2-bit pattern and the grid entry
- It's a one-way modification (centroid lookup changes, inference unchanged — same grid, same 8-bit index stored)
- It directly addresses the mismatch between L2-based kmap and L1-weighted quantizer objective

**Implementation**:
1. **ggml/src/ggml-quants.c**: Add static `g_iq2_xxs_kmap2` array
2. **iq2xxs_rebuild_map_and_neighbours()**: For each 2-bit pattern with a direct kmap match, compute L2 distances to all 256 centroids, find the second-closest (excluding the direct match), store in kmap2
3. **quantize_row_iq2_xxs_impl()**: In the initial scale search loop AND in the final encoding, when kmap returns a direct match, also check kmap2 for the second-best candidate. Evaluate both using weighted L1 at the current scale. Pick the better one.

**Expected**: KL improvement from 0.699009. The effect may be small (~0.1-0.5%) because most 2-bit patterns have their closest centroid well-separated from the second-closest. But for patterns with near-collisions (two centroids mapping to nearly the same 2-bit pattern), the second-best centroid may be significantly better for the weighted L1 objective. The improvement compounds across ~95 IQ2_XXS tensors.

**Files changed**: `ggml/src/ggml-quants.c` only — ~40 lines added.

### exp-070: Importance-weighted 2-bit level assignment during quantization
**Hypothesis**: The 2-bit level `l_i = nearest_int(0.5*(id*xval-1))` that forms the kmap lookup pattern treats all 8 dimensions equally. However, per-dimension imatrix importance weights vary significantly (often 10x+ range within an 8D chunk). By boosting the effective value for high-importance dimensions before level assignment, the kmap centroid selection is biased toward centroids that better reconstruct important dimensions.

**Implementation**: Multiply `xval[i]` by `(weight[i]/avg_w)^alpha` before computing the 2-bit level, where `weight[i]` is the full per-element weight (imatrix * sqrt(sigma2+x^2)) and `avg_w` is the arithmetic mean across the 8D chunk. alpha=0.3 for moderate effect. Applied in 3 places:
1. Initial scale search level computation (line 3904)
2. Final encoding level computation (line 3932)
3. Post-d refinement level computation (line 4049)

The neighbor search uses `waux[i] = sqrt(weight[i])` independently, preserving the weighted-L1 criterion. This is a one-way modification — the grid, kmap, inferface are unchanged.

**Why this is different from all prior experiments**:
- No experiment has modified the 2-bit LEVEL COMPUTATION to incorporate per-dimension importance
- This changes centroid SELECTION during quantization, not grid training
- One-way modification (only changes which centroid is selected for each 8D chunk)

**Expected**: Small KL improvement (Δ ~0.001-0.003). By biasing centroid selection toward important dimensions, the quantizer should better preserve the most critical weight values. alpha=0.3 is conservative enough to avoid excessive off-map patterns while providing meaningful weighting.

**Files changed**: `ggml/src/ggml-quants.c` only — ~10 lines added across 3 locations.

---

## Session: 2026-07-06 (continued) — Structural level encoding scheme

### exp-072: Quantization-Aware Centroid Refinement (QAT) — blend K-means centroids with scale-aware targets
**Hypothesis**: K-means training minimizes `sum w * |x - c|` — raw weighted L1 distance between weight values and centroids. But the quantizer's actual objective is `sum w * (x - scale * odd(c))^2` — weighted MSE with LS-optimal scaling and odd-forcing. These differ because the scale can stretch/compress centroids to match samples.

After K-means + error-aware snap, for each training sample assigned to centroid k, compute the LS-optimal scale `scale_opt = sumqx/sumq2` where q is the odd-forced centroid. The "target" centroid value that gives zero error at this scale is `target[i] = x[i] * sumq2/sumqx`. Blend: `c_new = (1-alpha) * c_kmeans + alpha * mean(target_assigned)`. alpha=0.5.

This adjusts centroids toward values that minimize the quantizer's actual objective (scale-compensated MSE), not the K-means proxy (raw L1). For centroids with diverse sample magnitudes, the scalar factor `sumq2/sumqx` per sample shifts the centroid toward the scale-aware optimum without requiring L1-to-objective alignment.

This is genuinely different from all prior experiments:
- No experiment has blended K-means centroids with quantizer-derived targets
- It operates on the K-means objective itself (raw L1 → scale-compensated MSE alignment)
- It's a one-shot refinement after training (not iterative, no new loop)
- Computational cost: negligible (~0.2s for 16384 samples × 256 centroids)

**Implementation**: ~60 lines added to `iq2xxs_learn_grid()` after the trial loop and before grid storage. Reassign samples to snapped best grid, compute scale-aware targets per centroid, blend with alpha=0.5, snap to int8.

**Expected**: Small KL improvement (Δ ~0.001-0.005). The effect is per-centroid value shifts of ±1-2 in a few dimensions, compoundable across 256 centroids × 95 tensors.

**Files changed**: `ggml/src/ggml-quants.c` only.

---

### exp-071: Exhaustive 4-bit level search (replace nearest-int with weighted-reconstruction-error minimization)
**Hypothesis**: The current sub-block scale level is selected by `l = nearest_int(0.5*(id*scales[ib]-1))` (line 4024), which minimizes the Euclidean distance between the continuous scale and the quantized scale `d*(2*l+1)`. However, the quantizer's TRUE objective is weighted reconstruction error across the sub-block's 32 elements:

```
min_l  sum_{k=0..3} sum_{i=0..7} w[8k+i] * (xb[8k+i] - d*(2*l+1) * centroid_{gidx(k)}[i] * sign[8k+i])^2
```

The nearest-level rule minimizes `|scales[ib] - d*(2*l+1)|`, which is only optimal when:
1. All elements have equal weights (w[i] = const)
2. All centroid values are equal (centroid[i] = const)
3. The grid index is independent of the level

None of these hold in practice. By exhaustively evaluating all 16 possible 4-bit levels for each sub-block and picking the one with minimum weighted reconstruction error (using the same error computation as the d optimization), we can find a level that genuinely minimizes the quantizer's objective.

**Why this is structurally novel**:
- Changes the LEVEL ENCODING SCHEME — how the 4-bit scale value is selected from the continuous scale
- Unlike exp-050 (which changed levels AFTER quantization and broke the index-scale coupling), this changes level selection DURING the initial encoding, BEFORE the post-d refinement. The grid indices were chosen at the optimal continuous scale, so re-evaluating levels with the chosen indices is safe (same as how the d optimization evaluates d candidates with fixed indices)
- Unlike exp-070 (which biased level computation toward important dimensions via alpha-scaling), this directly minimizes the actual objective without heuristics
- It's a one-way modification: levels → d → indices (fixed during level evaluation). No feedback loop.

**Implementation**: Replace the nearest-int level selection (lines 4023-4027) with a 16-candidate exhaustive search using the weighted reconstruction error computation from the d optimization (lines 3992-4005). The cost is negligible: 16 × 32 = 512 multiply-adds per sub-block.

**Expected**: Small KL improvement (Δ ~0.001-0.003). For sub-blocks with uneven importance weights or large centroid value variation, the optimal level may be ±1 from the nearest-int value. The effect compounds across ~8 sub-blocks × 95 tensors.

**Files changed**: `ggml/src/ggml-quants.c` only — ~35 lines replaced (lines 4022-4027).

---

### exp-073: Joint d+level optimization in a single pass (exhaustive level search during d optimization)
**Hypothesis**: The current d optimization (65 candidates, 0.5% step, ±16% range) picks the best `d` but computes 4-bit levels via `l = nearest_int(0.5*(id_try*scales[ib]-1))` which minimizes scale quantization error, NOT weighted reconstruction error. The d optimization evaluates each `d_try` with suboptimal levels, potentially selecting a wrong `d` because the levels for that `d` were poorly chosen.

By trying ALL 16 possible 4-bit levels for each sub-block during the d optimization, we find the jointly optimal (d, level) combination for the fixed grid indices.

**Key difference from exp-050/056/060** (which all regressed):
- Exp-050 changed levels AFTER quantization (broke index-scale coupling)
- Exp-056 changed d AFTER post-d refinement (indices selected for old d)
- Exp-060 changed levels AFTER post-d refinement (indices selected for old levels)
- **This optimizes d AND levels SIMULTANEOUSLY, BEFORE post-d refinement**. The post-d refinement then optimizes indices for the jointly-optimal (d, level) — a proper one-way chain.

**Implementation** (ggml/src/ggml-quants.c):
1. **d optimization loop** (lines 3985-4006): Replace `nearest_int(0.5*(id_try*scales[ib]-1))` with exhaustive search over levels 0..15 per sub-block. Each `ll` computes `scale_q = d_try*(2*ll+1)` and weighted reconstruction error. Pick min-error level per sub-block, accumulate.
2. **Final encoding** (lines 4023-4027): Same exhaustive level search with best `d`.

Computational cost: ~16x inner loop (~15-20s added, well under 5-min limit).

**Expected**: KL improvement from 0.699009 (Δ ~0.001-0.005). The nearest-int level is only optimal for uniform weights; with imatrix weighting, the true optimal level may be ±1-2 from nearest_int.

### exp-074: Level-Perturbation Centroid Search (LPCS) — evaluate alternative centroids via perturbed 8-bit patterns
**Hypothesis**: The kmap maps 8-bit patterns (2-bit levels for 4 elements in an 8D chunk) to the L2-closest centroid. But the quantization objective is weighted L1/L2 reconstruction error at the final scale (which depends on d, sub-block level, and centroid). The L2-closest centroid for a given pattern may not be the best by weighted reconstruction error — a DIFFERENT 8-bit pattern (off by ±1 in one element) might map to a centroid that produces lower overall weighted error.

For each 8D chunk, after the primary centroid is selected via kmap, try perturbing each element's 2-bit level by ±1 (if within [0,3]). For each perturbation, compute the new 8-bit pattern, look up the kmap for an alternative centroid, and evaluate weighted reconstruction error at the current scale. Pick the best centroid.

This is fundamentally different from exp-069 (kmap2) because:
- Exp-069 tried the SECOND-CLOSEST centroid by L2 for the SAME 8-bit pattern
- This tries centroids from DIFFERENT 8-bit patterns (nearby patterns)
- The level perturbation directly reflects the trade-off: slightly worsen one element's quantization for a centroid with better overall match
- The kmap's L2 criterion selects centroids by 8D Euclidean distance to the pattern, but the actual objective is weighted reconstruction at the scale — nearby patterns can produce better centroids for the weighted objective

**Implementation**: Modify `quantize_row_iq2_xxs_impl()` in two places:
1. **Final encoding** (after line 3941): After setting L from the kmap centroid, try alternative centroids via level perturbation. Evaluate weighted error at the sub-block scale. Update L if better.
2. **Post-d refinement** (after line 4077): Similarly, after finding a new centroid via kmap/neighbor at the quantized scale, try alternatives via level perturbation.

Cost: ~4-8 alternative centroid evaluations per 8D chunk via fast kmap lookup (O(1)). Total: ~0.5s added per tensor, negligible.

**Expected**: Small KL improvement (Δ ~0.001-0.003). The kmap L2 criterion is usually close to optimal, but for edge cases with non-uniform importance weights and the quantized scale mismatch, the perturbed-pattern centroid may reduce error. Effect compounds across 95 IQ2_XXS tensors × ~100 superblocks each.

**Result**: KL=0.752031 — REGRESSION (Δ = +0.053022, +7.6% from best 0.699009). The level-perturbation search introduces noise into centroid selection. For each chunk, trying alternative 8-bit patterns (via ±1 level perturbation) finds centroids that are better by L2 distance to the perturbed pattern, but worse by weighted reconstruction error at the actual scale. The kmap's L2 criterion, despite being an imperfect proxy, is more robust than the perturbed-pattern approach because it directly selects the centroid that minimizes the mapping from the ACTUAL (not perturbed) pattern. The perturbation outer loop (p over 0..7, delta over ±1) generates patterns that are geometrically close to the correct pattern but map to centroids with mismatched magnitude profiles, increasing reconstruction error. **Reverted**.

### exp-075: L1 distance for kmap and neighbor list construction (align with quantizer evaluation criterion)
**Hypothesis**: The kmap/neighbor list for off-map 2-bit patterns is built using L2 (squared Euclidean) distance in `iq2xxs_rebuild_map_and_neighbours()`:
```c
d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
```
But the quantizer's `iq2_find_best_neighbour()` evaluates candidates by **weighted L1**:
```c
float diff = scale*q - xval[i];
d1 += weight[i]*fabsf(diff);
```

This metric mismatch means the neighbor list is sorted by a criterion that differs from the evaluation criterion. With `nwant=8` truncation (8 unique distance levels), the L2-sorted list may exclude centroids that are closer by L1 and would score better in the evaluation. By switching to L1 distance for building the neighbor list, the sort order matches the evaluation criterion, ensuring that the first `nwant` L1-distance-levels contain the most promising candidates.

**Why this hasn't been tried**: All prior experiments focused on:
- Codebook training (K-means variants, init, distance metrics for assignment)
- Scale optimization (d step/range, level selection, post-d refinement)
- Neighbor search depth (nwant) and scoring (dot-product, odd-forced)
- Alternative centroid selection (kmap2, level perturbation)

The kmap/neighbor metric (L2) has been unchanged since the original E8 implementation. It was assumed optimal for the E8 lattice where L2 and L1 are highly correlated for centroid-on-grid patterns. With learned K-means grids where centroids can be at ANY int8 value (not just odd), L1 and L2 may disagree more frequently.

**Implementation**: Two lines changed in `iq2xxs_rebuild_map_and_neighbours()`:
```c
// Before:
d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
// After:
d2 += abs((int)pg[k] - (int)pos[k]);
```

**Expected**: Marginal KL improvement (Δ ~0.001-0.003) from 0.699009. The effect is limited because L1 and L2 rankings are correlated for integer-valued centroids, and most neighbors within nwant=8 distance levels are similar. But for edge cases where L1 and L2 disagree, the corrected sorting could find better centroids for off-map patterns.

**Files changed**: `ggml/src/ggml-quants.c` only — two lines in `iq2xxs_rebuild_map_and_neighbours()`.

---

## Session: 2026-07-06 (continued) — Quantizer weight formula change

### exp-076: Weight formula change — `qw * (sigma2 + xb^2)` instead of `qw * sqrt(sigma2 + xb^2)`
**Hypothesis**: The quantizer's weight formula `weight[i] = qw[i] * sqrt(sigma2 + xb[i]^2)` combines imatrix importance with element magnitude via the RMS. This formula has been unchanged since the original implementation and has never been experimented with. By removing the `sqrtf()` and using `w[i] = qw[i] * (sigma2 + xb[i]^2)`, high-magnitude elements get proportionally MORE weight relative to low-magnitude elements. This sharpens the quantizer's focus on preserving the largest weight values, which contribute most to the model's output.

This is fundamentally different from all prior experiments:
- **No d optimization change**: The d loop (65 candidates, 0.5% step, ±16% range) stays unchanged
- **No kmap change**: kmap construction and lookup stay unchanged  
- **No K-means change**: Grid training (sample selection, assignment, centroid update) stays unchanged
- **No level/scale change**: Level selection stays as `nearest_int`
- It's a **coordinated change to the weight formula** that affects ALL stages of the quantizer uniformly (scale selection, neighbor search, d optimization, post-d refinement) — every component uses the same new formula via `weight[]` and `waux[]` derived from it.

The change is minimal (remove `sqrtf(` in 4 places) but affects the fundamental trade-off: elements with larger `|xb|` contribute more to the weighted error, so the quantizer prioritizes them. Large weight values dominate the model's output, so better preserving them should reduce KL divergence.

**Implementation**: Change 4 lines in `quantize_row_iq2_xxs_impl()`:
```c
// Before:
weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
// After:
weight[i] = qw[i] * (sigma2 + xb[i]*xb[i]);
```
And the 3 other occurrences (d optimization loop, post-d refinement).

**Expected**: KL improvement from 0.699009. The effect is uncertain but this is a genuinely unexplored dimension. If it works, it suggests the sqrt was overly smoothing the weight distribution and the sharper formula better targets quantization effort. If it regresses, it confirms the RMS weight heuristic is optimal.

**Files changed**: `ggml/src/ggml-quants.c` only — 4 lines in `quantize_row_iq2_xxs_impl()`.

---

### exp-077: Align K-means training weights with quantizer weight formula (add sqrt(sigma2 + x^2) factor)
**Hypothesis**: The K-means training weights currently use only the imatrix importance values (`sample_weights[i] = qw[i]`). The quantizer uses `weight[i] = qw[i] * sqrt(sigma2 + xb[i]^2)` which includes a magnitude-dependent factor. This mismatch means K-means treats all elements equally (scaled by imatrix) while the quantizer prioritizes high-magnitude elements. By adding the `sqrt(sigma2 + x^2)` factor to K-means training weights, the centroid positions will shift to better represent patterns that contribute most to quantization error, aligning the training objective with the actual quantizer evaluation.

Exp-029 tried this approach and crashed (likely a code bug, not fundamental). This implementation computes `sigma2_train = mean(x^2)` across all training samples and multiplies sample_weights by `sqrt(sigma2_train + x[i]^2)` — matching the quantizer's formula exactly.

**Implementation**: ~8 lines added to `iq2xxs_learn_grid()` in `ggml/src/ggml-quants.c` after sample collection, before the K-means loop. Compute `sigma2_train` from all sample data, then multiply each `sample_weights` entry by `sqrtf(sigma2_train + samples[i]^2)`.

**Expected**: Small KL improvement (Δ ~0.0005-0.002) from 0.699009. By aligning K-means training with the quantizer's weight criterion, centroids should better represent high-magnitude-high-importance patterns, reducing quantization error for the most impactful elements. The effect may be modest because centroids barely drift from E8, but even a small systematic shift toward magnitude-weighted patterns could compound across 256 centroids × 95 tensors.

**Files changed**: `ggml/src/ggml-quants.c` only — `iq2xxs_learn_grid()`.

---

## Session: 2026-07-06 (continued) — Per-sub-block sigma2 for adaptive weight formula

### exp-078: Per-sub-block sigma2 for adaptive weight computation in the quantizer
**Hypothesis**: The current weight formula `weight[i] = qw[i] * sqrt(sigma2 + xb[i]^2)` uses `sigma2 = mean(xb^2)` computed once per 128-element superblock and shared across all 4 sub-blocks. This GLOBAL sigma2 is dominated by the largest-magnitude sub-block and provides a poor magnitude baseline for sub-blocks with different intrinsic scales.

For sub-blocks with values much smaller than the superblock average, the global sigma2 term artificially inflates the weight baseline, making the weight formula LESS sensitive to per-element magnitude variations within that sub-block. The sqrt(sigma2) term overpowers the sqrt(xb²) term, reducing the formula's ability to prioritize important large elements within low-magnitude sub-blocks.

By computing sigma2 independently per 32-element sub-block (`sigma2_ib = mean_32(xb^2)`), each sub-block gets the correct magnitude baseline for its own intrinsic scale:
- Low-magnitude sub-blocks: small sigma2 → sqrt(xb²) dominates → finer per-element weight differentiation
- High-magnitude sub-blocks: large sigma2 (matches their own scale) → weight profile matches actual importance

This is fundamentally different from ALL prior experiments:
- NO experiment has modified the sigma2 locality (all uses the same global superblock sigma2)
- It changes the WEIGHT COMPUTATION — which affects scale selection, centroid selection, d optimization, and post-d refinement UNIFORMLY
- It's a ONE-WAY modification: all downstream stages use the same weight formula, just with better-localized sigma2
- The change is PRINCIPLED: any statistic should be computed at the level of the data it describes (sub-block-level weights should use sub-block-level statistics)

**Implementation**: ~10 lines changed in `ggml/src/ggml-quants.c`:
1. Remove global sigma2 computation (line 3853-3855)
2. Add per-sub-block sigma2 computation inside the sub-block loop (after line 3857)
3. Store `sigma2_ib` in an array for later use in d optimization and post-d refinement
4. Use `sigma2_ib` instead of `sigma2` in all weight computations (lines 3860, 4002, 4045, 4071)

**Expected**: KL improvement from 0.699009 (Δ ~0.001-0.005). The effect should be largest for tensors with high inter-sub-block variance (where global sigma2 poorly represents individual sub-block magnitudes). The weight formula becomes more selective, directing quantization effort toward elements that matter most within each sub-block.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()`.

### exp-079: Per-8D-chunk sigma2 for weight formula (further localization)
**Hypothesis**: exp-078 localized sigma2 from superblock (128 elements) to sub-block (32 elements), improving KL from 0.699009 to 0.691085 (-1.13%). The natural next step is to localize sigma2 to the 8D chunk — the fundamental quantization unit in IQ2_XXS.

Within a 32-element sub-block, the 4 × 8D chunks can have very different magnitude distributions (e.g., chunk 0 has large weights, chunk 3 is near-zero). A single sub-block sigma2 still inflates the weight baseline for low-magnitude chunks:
- Low-magnitude chunk: sigma2 dominated by other chunks → `sqrt(sigma2)` floor too high → weight profile less selective
- High-magnitude chunk: sigma2 diluted by other chunks → `sqrt(sigma2)` floor too low → less magnitude normalization benefit

By computing sigma2 per 8D chunk (`sigma2_ch = mean_8(xb^2)`), each chunk's weight formula uses its true local magnitude baseline. This is the finest meaningful granularity — the 8D chunk is the atomic unit of quantization (kmap lookup, neighbor search, centroid selection).

**Implementation**: ~15 lines changed in `quantize_row_iq2_xxs_impl()`:
1. Replace `sigma2_per_ib[QK_K/32]` with `sigma2_ch[QK_K/8]` array
2. Change weight computation loop: compute sigma2 per 8D chunk (inside k loop) instead of per 32-element sub-block
3. Update d optimization (line 4003) to use per-chunk sigma2
4. Update post-d refinement (lines 4046, 4072) to use per-chunk sigma2

All 4 weight computation sites are changed uniformly — no metric mismatch.

**Expected**: KL improvement from 0.691085 (Δ ~0.001-0.003, ~0.15-0.45% relative). The effect may be smaller than exp-078 (1.13%) because sub-block sigma2 already captures most of the localization benefit. But 8D-chunk sigma2 is strictly more local, and the change is risk-free (uniform across all stages).

**Result**: KL=0.715862 — REGRESSION (Δ = +0.024777, +3.6% from best 0.691085). Per-chunk sigma2 overfits to local 8-element fluctuations. Within a 32-element sub-block, the 4 × 8D chunks share a joint d optimization and level quantization — they NEED a shared sigma2 baseline to properly weight inter-chunk trade-offs. Computing sigma2 per chunk makes the weight formula too sensitive to each chunk's local magnitude, biasing the optimization toward chunk-level characteristics at the expense of the sub-block's overall error profile. The 32-element sub-block (exp-078) is the optimal granularity for the weight baseline. **Reverted**.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (Reverted).

### exp-080: Softer weight formula exponent (0.4 instead of 0.5) with per-sub-block sigma2
**Hypothesis**: The weight formula `weight[i] = qw[i] * sqrt(sigma2_per_ib + xb[i]^2)` uses an exponent of 0.5 (via `sqrtf`). This gives large elements disproportionately high weight, potentially over-optimizing the quantizer for outlier elements at the expense of the sub-block's overall distribution.

With per-sub-block sigma2 (exp-078), the baseline is already well-localized. The `sqrt(sigma2 + xb^2)` factor may still give too much weight to elements where |xb| >> sqrt(sigma2). Reducing the exponent from 0.5 to 0.4 (`powf(sigma2 + xb^2, 0.4f)`) softens the magnitude sensitivity, making the weight profile more uniform within each sub-block.

This is the OPPOSITE direction from exp-076 (which removed sqrt entirely → exponent 1.0 → catastrophic regression 0.837). Softer weighting may better balance the optimization across all elements, preventing the largest element from dominating the sub-block's scale and index selection.

Exp-076 showed that SHARPENING the weight (higher exponent) hurts badly. The optimal exponent likely lies between 0.3 and 0.5 — this experiment tests 0.4 as a moderate reduction.

**Implementation**: 4 lines changed in `quantize_row_iq2_xxs_impl()`:
```c
// Before:
weight[i] = qw[i] * sqrtf(sigma2_per_ib[ib] + xb[i]*xb[i]);
// After:
weight[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.4f);
```
Same change at lines 4003, 4046, 4072.

**Expected**: Small KL improvement (Δ ~0.001-0.003). If the sqrt is over-sharp, softening improves the balance. The change is uniform across all stages — no coupling issues.

**Result**: KL=0.682340 — IMPROVEMENT (Δ = -0.008745, -1.27% from best 0.691085). NEW BEST RESULT. The softer exponent (0.4) outperforms the original sqrt (exponent 0.5). Reducing the magnitude weight exponent softens the dynamic range, preventing outlier elements from dominating the sub-block optimization. PPL dropped from 25.69 to 25.41. Quantize time increased from ~299s to ~355s due to powf being slower than sqrtf.

**Why it works**: The weight formula `weight = qw * sqrt(sigma2 + xb^2)` = `qw * (sigma2 + xb^2)^0.5`. The exponent 0.5 means elements with |xb| >> sqrt(sigma2) get weight qw*|xb| — proportional to their magnitude. With exponent 0.4, they get weight qw*|xb|^0.8 — still magnitude-sensitive but less extreme. This better balances the optimization across all elements within a sub-block, allowing the d optimization and scale selection to find a configuration that minimizes total weighted error without over-prioritizing the largest element.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (4 lines: sqrtf → powf(..., 0.4f)). **KEPT** (new best).

### exp-081: Further soften weight exponent to 0.35 (from 0.4)

**Hypothesis**: The weight exponent reduction from 0.5 (sqrtf) to 0.4 gave a 1.27% KL improvement (0.691085 → 0.682340). This suggests the optimal exponent lies below 0.5. If the relationship is roughly monotonic in this region, further reducing to exponent 0.35 may continue the improvement — reducing the magnitude-weighting even more, giving the quantizer a flatter optimization landscape where small-magnitude elements are not neglected.

The risk is that going too soft under-weights large elements that contribute most to model output quality. Exp-076 (exponent 1.0, no sqrt) was catastrophic (KL=0.837), so the penalty for going too sharp is severe. Going softer is safer — the worst case is the weight formula becomes too uniform, similar to not having the sqrt factor at all (effectively exponent 0 → just qw). Exponent 0.35 is a moderate 12.5% reduction from 0.4.

**Implementation**: Change 4 lines in `quantize_row_iq2_xxs_impl()`:
```c
// Before:
weight[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.4f);
// After:
weight[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.35f);
```

**Expected**: Small KL improvement (Δ ~0.002-0.005) from 0.682340. Diminishing returns expected as exponent approaches the optimal value around 0.3-0.35.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (4 lines: 0.4f → 0.35f).

---

### exp-082: Further soften weight exponent to 0.30 (from 0.35)

**Hypothesis**: The weight exponent reduction from 0.5 (sqrtf) → 0.4 → 0.35 has given accelerating improvements: 0.5→0.4 = -1.27%, 0.4→0.35 = -2.05%. This suggests the true optimal exponent lies well below 0.35. Reducing further to exponent 0.30 should continue the trend.

The softer exponent reduces the dominance of outlier elements in the weight formula, giving the quantizer a more balanced optimization landscape. Each reduction has produced larger relative improvement than the previous one, possibly because the weight formula is approaching the point where magnitude-weighting just compensates for the L1 quantizer's tendency to over-preserve large values.

**Implementation**: Change 4 lines in `quantize_row_iq2_xxs_impl()`:
```c
// Before:
weight[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.35f);
// After:
weight[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.30f);
```

**Expected**: KL improvement (Δ ~0.003-0.006) from 0.668342. If the accelerating improvement trend continues, the gain could be comparable to or exceed the 2.05% from 0.4→0.35.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (4 lines: 0.35f → 0.30f).

---

### exp-083: Further soften weight exponent to 0.25 (from 0.30)

**Hypothesis**: The weight exponent reduction from 0.5→0.4→0.35→0.30 has produced consistent KL improvements:
- 0.5→0.4: -1.27% (0.691085 → 0.682340)
- 0.4→0.35: -2.05% (0.682340 → 0.668342)
- 0.35→0.30: -0.33% (0.668342 → 0.666162)

The diminishing returns from 0.35→0.30 (-0.33%) suggest the optimal exponent is near 0.30 or slightly below. Reducing further to exponent 0.25 should verify whether the curve has flattened or if there is still marginal headroom.

The softer exponent continues to reduce the dominance of outlier elements in the weight formula. At 0.25, the magnitude-weighting is significantly softer than the original sqrt (0.5), giving the quantizer a very flat optimization landscape where per-element magnitude differences matter less. This may marginally improve the inter-element balance, but the effect is expected to be small or null if the true optimum is near 0.30.

**Implementation**: Change up to 4 lines in `quantize_row_iq2_xxs_impl()`, all 0.30f/0.35f → 0.25f:
- Line ~3861: `powf(..., 0.30f)` (weight computation)
- Line ~4003: `powf(..., 0.35f)` → 0.25f (d optimization)
- Line ~4046: `powf(..., 0.35f)` → 0.25f (post-d refinement)
- Line ~4072: `powf(..., 0.35f)` → 0.25f (post-d refinement)

**Expected**: Small KL improvement or null (Δ ~0.0001-0.001) from 0.666162. The diminishing returns make further improvement unlikely beyond 0.25. If null, the optimal exponent is confirmed at 0.30.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (4 lines: 0.30f/0.35f → 0.25f).

**Result**: KL=0.679018 — REGRESSION (Δ = +0.012856, +1.93% from best 0.666162). Exponent 0.25 overshoots the optimal. The weight formula becomes too uniform, under-weighting large elements that contribute most to model output quality. The optimal exponent is confirmed near 0.30. **Reverted**.

### exp-084: Harmonize weight exponent to 0.30 everywhere (fix d opt + post-d refinement from 0.35 to 0.30)

**Hypothesis**: Exp-082 reduced the main weight exponent from 0.35 to 0.30 and improved KL from 0.668342 to 0.666162 (-0.33%). However, the implementation only changed line 3861 (main weight computation) and left 3 other lines at 0.35:

---

### exp-085: Asymmetric weight profile — sharpen d optimization and post-d refinement to 0.50 while keeping main at 0.30

**Hypothesis**: Exp-082 (main=0.30, d opt/post-d=0.35) outperforms both exp-081 (all=0.35, KL=0.668342) and exp-084 (all=0.30, KL=0.673372). This reveals that **asymmetric weight profiles across quantizer stages are beneficial** — the main quantization benefits from softer weights (balanced per-element optimization) while the d optimization and post-d refinement benefit from sharper weights (prioritizing preservation of high-magnitude elements during scale selection and index refinement).

**Result**: KL=0.669486 — REGRESSION (Δ = +0.003324, +0.50% from best 0.666162). The asymmetric weight profile with d opt/post-d at 0.50 (original sqrt) regresses compared to the 0.30/0.35 configuration. This shows that the optimal asymmetry gap is narrower than tested here. The widening from 0.35 to 0.50 was too aggressive — the d optimization and post-d refinement with sqrt (0.50) weights over-prioritize large elements during scale selection, pushing d away from the global optimum for the 0.30-weighted main quantization. The 0.30/0.35 split (exp-082) provides the right balance: slightly softer main for diverse centroid selection, slightly sharper refinement for focused scale tuning. **Reverted.**

Exp-084 showed harmonizing d opt and post-d DOWN to 0.30 regresses (+1.08% from best). The natural next direction is to increase d opt and post-d UP to 0.50 (original sqrt, before exp-080). This widens the asymmetry gap:
- **Main quantization (exp=0.30)**: Softer weight → balanced per-element level computation, fair kmap/neighbor centroid selection across all 8 elements. No single outlier dominates.
- **d optimization (exp=0.50)**: Sharper weight → scale selection prioritizes preserving high-magnitude elements, which contribute most to model output. The superblock scale is chosen to minimize weighted reconstruction error with sharper emphasis on large elements.
- **Post-d refinement (exp=0.50)**: Sharper weight → grid index recomputation at quantized scale focuses on best-preserving the most impactful elements.

This is fundamentally novel because:
- ALL prior weight-exponent experiments (exp-076 through exp-084) either changed ALL exponents uniformly or harmonized an existing inconsistency toward uniformity
- No experiment has deliberately increased the asymmetry gap
- It's a one-way modification (weights only, no index/scale change) — safe against overfitting

**Implementation**: Change 3 lines in `quantize_row_iq2_xxs_impl()`:
```c
// Line 4003: d optimization
powf(..., 0.50f)  // was 0.35f
// Line 4046: Post-d refinement candidate computation
powf(..., 0.50f)  // was 0.35f
// Line 4072: Post-d refinement comparison
powf(..., 0.50f)  // was 0.35f
```

**Expected**: KL improvement from 0.666162. If the asymmetry is beneficial, widening the gap from (0.30 vs 0.35) to (0.30 vs 0.50) could give ~0.3-1.0% additional improvement. The original sqrt (0.50) was optimal for the quantizer before exp-080 — reverting d opt and post-d refinement to 0.50 while keeping main at 0.30 combines the best of both regimes.
- Line 4003: d optimization loop weight
- Line 4046: Post-d refinement candidate computation weight
- Line 4072: Post-d refinement comparison weight

This inconsistency means:
1. The **d optimization** uses a sharper weight profile (0.35) than the main quantization (0.30) — selecting d values for a different trade-off
2. The **post-d refinement** evaluates grid index changes with different weights (0.35) than the initial selection (0.30) — potentially rejecting beneficial index changes or accepting harmful ones

By harmonizing ALL 4 weight exponent sites to 0.30, the quantizer becomes self-consistent. All stages evaluate the same weighted objective.

**Expected**: KL improvement from 0.666162. The effect direction is uncertain because:
- If 0.35 was inadvertently helping the d optimization resist noise, harmonizing to 0.30 could weaken d selection → mild regression
- If the inconsistency was causing suboptimal d/index choices, harmonizing could improve KL by 0.1-0.5%

**Implementation**: 3 lines changed in `quantize_row_iq2_xxs_impl()` (ggml/src/ggml-quants.c):
```c
// Line 4003: d optimization
float w = qw[8*k + i] * powf(sigma2_per_ib[ib] + xb[8*k + i] * xb[8*k + i], 0.30f);
// Line 4046: Post-d refinement candidate
wtmp[i] = qw[8*k+i] * powf(sigma2_per_ib[ib] + xb[8*k+i]*xb[8*k+i], 0.30f);
// Line 4072: Post-d refinement comparison
float w = qw[8*k+i] * powf(sigma2_per_ib[ib] + xb[8*k+i]*xb[8*k+i], 0.30f);
```

**Files changed**: `ggml/src/ggml-quants.c` only.

---

## Session: 2026-07-06 (continued) — D-opt/Post-d weight exponent 0.40

### exp-086: Increase d-opt and post-d refinement weight exponent from 0.35 to 0.40, main stays at 0.30

**Hypothesis**: The asymmetric weight profile (main=0.30, d opt/post-d=0.35) outperforms both uniform (all=0.30, KL=0.673) and wider asymmetry (main=0.30, d opt/post-d=0.50, KL=0.669). This suggests the optimal d opt/post-d exponent lies between 0.35 and 0.50.

Increasing d opt/post-d from 0.35 to 0.40 widens the asymmetry gap from 0.05 to 0.10. The sharper weights for scale selection and index refinement may better prioritize high-magnitude elements during these critical steps, while maintaining the softer main quantization that gives balanced centroid selection across all elements.

**Implementation**: 3 lines changed in `quantize_row_iq2_xxs_impl()` (ggml/src/ggml-quants.c):
- Line 4003: d optimization weight: 0.35f → 0.40f
- Line 4046: Post-d refinement candidate weight: 0.35f → 0.40f
- Line 4072: Post-d refinement comparison weight: 0.35f → 0.40f

**Expected**: Small KL improvement (Δ ~0.0003-0.001) from 0.666162.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()`.

---

## Session: 2026-07-06 (continued) — Complete functional change to weight formula

### exp-087: Linear-L1 magnitude weight formula (replace powf with 1+|xb|) — REGRESSION

**Hypothesis**: All prior weight formula experiments explored the power-law family `qw * (sigma2 + xb^2)^p`, varying exponent p from 1.0 down to 0.25. The optimal was p≈0.30. A structurally different approach using linear-L1 magnitude `qw * (1 + |xb|)` could outperform the tuned power-law.

**Implementation**: Replaced `powf(sigma2_per_ib + xb^2, 0.30f)` with `(1.0f + fabsf(xb[i]))` at all 4 weight formula locations. Removed `sigma2_per_ib` array and `s2` computation.

**Result**: KL=0.723409 — CATASTROPHIC REGRESSION back to E8 baseline levels (0.7248). The linear-L1 form effectively disables all the per-sub-block sigma2 and exponent tuning benefits (exp-078 through exp-082). The power-law L2 form with tuned exponent is structurally necessary — the per-element adaptive weighting from `(sigma2_per_ib + xb^2)^p` provides essential selectivity that `(1 + |xb|)` completely lacks.

**Lesson**: The sigma2 baseline is critical — it provides a per-sub-block floor that prevents near-zero elements from getting zero weight (which `1+|xb|` does not adequately address because the `1+` constant is sub-block-agnostic). The power-law form with per-sub-block sigma2 is the correct family for this quantizer. **Reverted.**

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()`. (Reverted)

---

### exp-088: Weight ratio clamping per sub-block to prevent extreme outlier dominance

**Hypothesis**: The weight formula `weight[i] = qw[i] * (sigma2_ib + xb[i]^2)^0.30` can produce extreme weight ratios within a sub-block — if one element has |xb| = 10x sigma, its weight is ~4x higher than the sigma-baseline weight. For 100x outliers, the ratio reaches ~16x. This causes the d optimization, scale selection, and grid index selection to focus almost exclusively on the single outlier element at the expense of all other 31 elements in the sub-block.

By clamping the max:min weight ratio to ≤5 within each sub-block, we preserve the intELEMENT importance ranking (same element still has highest weight) while preventing single outliers from completely dominating the sub-block's optimization. For the majority of sub-blocks with weight ratio ≤5, the behavior is identical to exp-082.

This is fundamentally different from all prior weight experiments:
- exp-076: Removed sqrt (exponent 1.0) — made extreme weights MORE extreme → catastrophic
- exp-078: Per-sub-block sigma2 — changed sigma2 granularity, not dynamic range
- exp-080/081/082: Exponent reduction (0.5→0.4→0.35→0.30) — flattened the slope of weight-vs-magnitude
- **exp-088: Clamps the dynamic range directly** without changing the exponent or the per-element weight formula shape. This is a post-processing step that only activates for outlier-dominated sub-blocks.

The clamping limits the effective ratio to 5:1:
```
max_w / min_w ≤ 5
```
Achieved by linearly compressing weights toward min_w:
```
compressed[i] = min_w + (weight[i] - min_w) * (5*min_w - min_w) / (max_w - min_w)
```
This preserves the ranking and relative spacing between elements while capping the extreme ratio.

**Implementation**: ~15 lines added after line 3861 in `quantize_row_iq2_xxs_impl()`:
1. After computing all 32 weights, find max_w and min_w
2. If max_w / min_w > 5.0f, compress weights linearly toward min_w so the compressed max/min ratio = 5.0f

**Expected**: KL improvement from 0.666162 (Δ ~0.002-0.005, ~0.3-0.8% relative). The effect may be modest because exp-082's exponent 0.30 already softens the weight profile considerably. But for the tail of sub-blocks with extreme outliers (common in attention tensors where a few elements dominate), clamping should improve overall balance.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (~15 lines added after line 3861).

**Result**: KL=0.794320 — REGRESSION (Δ = +0.128158, +19.2% from best 0.666162). Weight clamping per sub-block is actively harmful — it compresses the weight dynamic range, making the quantizer LESS selective. The exponent 0.30 already provides sufficient softening; adding clamping on top over-softens the weights, causing the d optimization and index selection to deprioritize important elements. The compression formula `min_w + (weight[i] - min_w) * scale` preserves ranking but reduces the absolute weight ratios that the quantizer's optimization relies on. Reducing a 10:1 ratio to 5:1 means the d optimization gives 2x more error tolerance to high-weight elements than it should. **Reverted**.

---

### exp-089: Intra-sub-block weight normalization — equalize total weight per sub-block

**Hypothesis**: The weight formula `weight[i] = qw[i] * powf(sigma2_ib + xb[i]^2, 0.30)` produces weights whose SUM varies hugely across sub-blocks within a superblock. A sub-block with large values (|xb| ~ 10) has weights ~|xb|^0.6 ~ 4x larger per element than a sub-block with small values (|xb| ~ 1). This means in the d optimization, which minimizes weighted error across ALL 4 sub-blocks simultaneously, high-magnitude sub-blocks have ~4x more influence on which `d` is selected.

This is a structural issue because:
1. The sub-block magnitude is already captured by the level `l` (which scales the centroid output by `d*(2*l+1)`)
2. The imatrix `qw[i]` already captures cross-element importance WITHIN a tensor
3. The per-sub-block sigma2 (exp-078) captures the sub-block's relative magnitude, which then feeds back into the weight formula — doubling the magnitude signal

By normalizing each sub-block's total weight to 32 (equal per-element mean weight = 1.0), the d optimization treats ALL sub-blocks equally when selecting the best `d`. The per-element RELATIVE importance (captured by qw and per-element magnitude variation within the sub-block) is preserved, but the cross-sub-block total weight imbalance is removed.

This is genuinely different from all prior experiments:
- exp-078: Per-sub-block sigma2 — changed the sigma2 baseline granularity
- exp-079: Per-8D-chunk sigma2 — overfitted to local chunk
- exp-080/081/082: Exponent tuning — changed the magnitude-to-weight slope
- exp-088: Weight ratio clamping — clipped extreme within-sub-block ratios
- **exp-089: Normalizes each sub-block's TOTAL weight to the same value** — eliminating cross-sub-block magnitude bias while preserving per-element relative importance

The key insight: the normalization applies uniformly to ALL four weight computation sites (main, d opt, post-d refinement stages 1 and 2). I store the normalized weights in a `weight_norm[4][32]` array and use the same values everywhere — no stage inconsistency.

**Implementation**: In `quantize_row_iq2_xxs_impl()` (ggml/src/ggml-quants.c):
1. Add `float weight_norm[QK_K/32][32];` declaration
2. After line 3861-3862 (weight computation): normalize weights per sub-block to sum=32, store in `weight_norm[ib]`
3. Recompute `waux` from normalized weights
4. Lines 4003, 4046, 4072: replace `qw[8*k+i] * powf(..., 0.35f)` with `weight_norm[ib][8*k+i]`

**Expected**: KL improvement from 0.666162. The effect should be modest (Δ ~0.001-0.004) but principled. The sub-block level `l` already handles magnitude; the weight should only capture per-element importance deviation from the sub-block mean. Normalizing removes the double-counting of magnitude in the d optimization.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (~10 lines added, 3 lines modified).

---

### exp-090: Remove sqrtf from waux — align neighbor search weight with rest of quantizer

**Hypothesis**: The `waux` array is used exclusively in `iq2_find_best_neighbour()` for the off-map fallback neighbor search during grid index selection. Currently `waux[i] = sqrtf(weight[i])`, which gives the neighbor search HALF the weight exponent (0.15) of the level assignment and scale evaluation stages (0.30). This is an inconsistency:

| Stage | Weight formula | Effective exponent |
|-------|---------------|-------------------|
| Level assignment (line 3892: `make_qp_quants`) | `weight[i]` | 0.30 |
| Scale evaluation (line 3918-3949: `sumqx/sumq2`) | `weight[i]` | 0.30 |
| **Neighbor search** (lines 3913, 3940) | `waux[i] = sqrtf(weight[i])` | **0.15** |
| D optimization (line 4003) | original weight | 0.35 |
| Post-d neighbor search (line 4058) | `wtmp[i]` | **0.35** (no sqrt) |

The sqrt in waux is a historical artifact — the original `sqrtf(sigma2 + x^2)` time (exp-080 era) used sqrt for the weight itself, and the additional sqrtf on waux was carried forward without reconsideration. Now that we use `powf(..., 0.30f)` for `weight[i]`, the sqrt reduces the effective exponent to 0.15 — making the neighbor search nearly uniform in weighting, barely distinguishing between large and small elements.

Critically, the **post-d refinement neighbor search (line 4058)** uses `wtmp[i] = qw[i] * powf(..., 0.35f)` DIRECTLY (no sqrt). So the same operation (neighbor search for grid index) uses different weighting in the main pass (sqrt) vs post-d (direct). This is structurally inconsistent.

**Change**: Remove the sqrtf:
```c
// Before (line 3862):
for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
// After:
for (int i = 0; i < 32; ++i) waux[i] = weight[i];
```

**Expected**: The neighbor search now uses the same weight profile as the level assignment and scale stages. This should improve grid index selection for off-map patterns (which go through the neighbor search path), reducing the frequency of suboptimal index choices. KL improvement Δ ~0.0005-0.002.

**Files changed**: `ggml/src/ggml-quants.c` only — line 3862.

---

### exp-091: Outlier-robust sigma2 using trimmed mean (single-largest-xb² removal)

**Hypothesis**: The sigma2 baseline `sigma2_per_ib = mean(xb²)` per 32-element sub-block is inflated by outlier elements. A single massive element (|xb| >> RMS) raises the sigma2 floor, which the weight formula `weight[i] = qw[i] * powf(sigma2 + xb², 0.30)` uses as an additive baseline. This inflated sigma2 creates a high weight floor for small elements, REDUCING per-element weight selectivity:
- For the outlier: `weight ≈ qw * (xb²)^0.30` — sigma2 barely matters (xb² >> sigma2)
- For typical elements: `weight ≈ qw * (2*sigma2)^0.30` — sigma2 inflates weight by ~23%
- For small elements: `weight ≈ qw * sigma2^0.30` — weight is entirely sigma2-driven

When sigma2 is inflated by an outlier, the weight floor for small elements rises, collapsing the effective dynamic range of weights within the sub-block. The quantizer no longer distinguishes well between important and less-important elements.

By removing the single largest xb² before computing sigma2, we get a sigma2 that better represents the "typical" element magnitude. This is a trimmed mean that excludes exactly one outlier per sub-block, which is principled because:
1. The xb² distribution is roughly chi-squared-ish; the max element in 32 draws is likely an outlier
2. The single-outlier exclusion is robust and doesn't over-trim (unlike a fixed threshold or percentage)
3. For uniform sub-blocks (all xb ≈ same), removing any one element barely changes sigma2

**Implementation**: One-line change in `quantize_row_iq2_xxs_impl()` (ggml/src/ggml-quants.c, lines 3858-3860):
```c
// Before:
float s2 = 0.0f;
for (int i = 0; i < 32; ++i) s2 += xb[i]*xb[i];
sigma2_per_ib[ib] = s2 / 32.0f;
// After:
float s2 = 0.0f; float max_x2 = 0.0f;
for (int i = 0; i < 32; ++i) {
    float x2 = xb[i]*xb[i];
    s2 += x2;
    if (x2 > max_x2) max_x2 = x2;
}
sigma2_per_ib[ib] = (s2 - max_x2) / 31.0f;
```

All 4 weight computation sites (main quantization, d optimization, post-d refinement stages 1 and 2) use the same `sigma2_per_ib` array, so this change is uniform across all stages — no coupling issues. The change is additive (only sigma2 changes) and doesn't touch d, levels, indices, or any quantizer state.

**Expected**: Small KL improvement (Δ ~0.001-0.004, ~0.15-0.6% relative) from 0.666162. The effect should be largest for sub-blocks in attention tensors where a few elements dominate (common in Q/K projections). No risk of regression since the change is a minute shift in the sigma2 baseline only.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (3 lines modified).

---

## Session: 2026-07-06 (continued) — Adaptive weight exponent per sub-block

### exp-092: Adaptive weight exponent based on per-sub-block magnitude distribution shape

**Hypothesis**: All prior weight-exponent experiments (exp-076 through exp-091) used a FIXED exponent per tensor — every sub-block in every tensor used the same exponent (0.30 main, 0.35 d-opt/post-d). But the optimal weight exponent depends on the sub-block's magnitude distribution:

- **Heavy-tailed sub-blocks** (one or two elements dominate, others are near-zero): A softer exponent (~0.30) prevents the dominant element from completely controlling the optimization. The sigma2 baseline is inflated by the outlier, so the `powf(sigma2 + xb², 0.30)` formula already provides a high weight floor for small elements.
- **Near-uniform sub-blocks** (all 32 elements have similar magnitude): A sharper exponent (~0.50) provides better per-element differentiation. Since all elements are similar, the sigma2 baseline is small, and a sharper exponent amplifies subtle magnitude differences that help the quantizer prioritize.

The coefficient of variation (CV) `sqrt(sigma2_per_ib) / mean_abs(xb)` captures this shape:
- Uniform (all ≈ equal): CV ≈ 1.0 → sharper exponent (0.50)
- Gaussian-like: CV ≈ 1.25 → moderate exponent (~0.42)
- Heavy-tailed: CV > 1.5 → softer exponent (0.30)

**Implementation**: For each sub-block `ib`, compute `mean_abs = mean(|xb|)` in the sigma2 computation loop. Then compute:
```c
float shape_ratio = sigma2_per_ib[ib] / (mean_abs * mean_abs + 1e-10f);
float alpha = min(shape_ratio - 1.0f, 1.0f);  // 0 (uniform) to 1 (heavy-tailed)
float exp_adaptive = 0.50f - 0.20f * alpha;    // 0.50 for uniform, 0.30 for heavy-tailed
```

Also store `exp_adaptive` in an `exp_per_ib[4]` array alongside `sigma2_per_ib[ib]`.

Replace all 4 `powf(..., 0.30f/0.35f)` calls with `powf(..., exp_per_ib[ib])`:
- Line 3861: `0.30f` → `exp_per_ib[ib]`
- Line 4003: `0.35f` → `exp_per_ib[ib]`
- Line 4046: `0.35f` → `exp_per_ib[ib]`
- Line 4072: `0.35f` → `exp_per_ib[ib]`

This is a **one-way modification** (weights only, no index/scale/level coupling). All 4 weight sites use the SAME adaptive exponent per sub-block — no stage inconsistency.

**Why this is genuinely novel**:
- No prior experiment has varied the exponent WITHIN a tensor (per sub-block)
- The adaptive rule is based on a principled statistic (CV / shape ratio)
- It uses existing per-sub-block infrastructure (sigma2_per_ib array) with negligible computational overhead
- The change is uniform across all 4 weight sites — no coupling issues
- If the hypothesis is wrong, the worst case is a small regression (bounded by the 0.30-0.50 range, which are both previously-tested safe values)

**Expected**: KL improvement from 0.666162 (Δ ~0.002-0.006, ~0.3-0.9% relative). The effect should be largest for tensors with diverse sub-block distributions (e.g., attention tensors where some sub-blocks have dominant outlier elements while others are uniform). Heavy-tailed sub-blocks get the proven 0.30 exponent (exp-082), uniform sub-blocks get the sharper 0.50 (original sqrt). The combination should outperform any single fixed exponent.

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (~10 lines changed: add mean_abs computation, exp_per_ib array, 4 exponent substitutions).

### exp-093: Softer neighbor search weighting — `waux = powf(weight, 0.20)` instead of `sqrtf(weight)`

**Hypothesis**: The neighbor search in `iq2_find_best_neighbour()` handles off-map 2-bit patterns (rare patterns without a direct kmap match). It uses `waux[i] = sqrtf(weight[i])` as the per-element weight, which gives an effective exponent of 0.15 (from `weight = qw * (sigma2+xb^2)^0.30`). 

Exp-090 proved that REMOVING the sqrt (using `weight[i]` directly, effective exponent 0.30) causes regression (KL 0.682 vs best 0.666). The neighbor search needs LOWER effective weighting than the main path. This makes intuitive sense: for rare off-map patterns, per-element importance ratios are noisy — the kmap L2 distance should dominate the centroid selection, not the importance-weighted error.

By reducing `waux = powf(weight, 0.20)` (effective exponent 0.06), the neighbor search becomes more uniform, letting the global grid structure dominate centroid selection for rare patterns. The main quantization path still uses the full weight (exp 0.30), and the post-d refinement uses the sharper wtmp (exp 0.35). This widens the existing asymmetry:
- **Pre-d neighbor (waux, exp 0.15 → 0.06)**: Nearly uniform, grid-structure-dominated centroid selection
- **Main weight (exp 0.30)**: Balanced importance-magnitude trade-off
- **Post-d neighbor (wtmp, exp 0.35)**: Sharper, importance-focused index refinement

The widening of the asymmetry between pre-d and post-d neighbor search is principled: the initial search uses a broad objective (good enough centroid for rare patterns), the post-d refinement uses a sharp objective (best centroid with quantized scale). The two-stage approach benefits from wider separation.

**Implementation**: One-line change in `ggml/src/ggml-quants.c:3862`:
```c
// Before:
for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
// After:
for (int i = 0; i < 32; ++i) waux[i] = powf(weight[i], 0.20f);
```

**Expected**: Small KL improvement (Δ ~0.001-0.003, ~0.15-0.45% relative) from 0.666162. The effect is limited because most chunks (≥95%) use the DIRECT kmap path and don't invoke the neighbor search. But for the 5% of off-map patterns, better centroid selection compounds across 95 IQ2_XXS tensors.

**Risk**: LOW — bounded impact because only the neighbor search (fallback path for ~5% of chunks) is affected. The main weight, d optimization, scale selection, and post-d refinement are unchanged. Revert is trivial (one line).

**Files changed**: `ggml/src/ggml-quants.c` only — line 3862.

**Result**: KL=0.662001 — IMPROVEMENT (Δ = -0.004161, -0.62% from best 0.666162). NEW BEST RESULT. The softer waux (powf 0.20, effective exponent 0.06) significantly improves neighbor search for off-map patterns. The hypothesis is confirmed: the neighbor search benefits from more uniform weighting (effective exp 0.06) than the main quantization path (exp 0.30). Widening the asymmetry from pre-d neighbor (0.06) to main (0.30) to post-d refinement (0.35) provides better centroid selection at each stage for their respective roles. PPL slightly increased from 25.11 to 25.25, but KL improved and same-top-p improved from 61.96% to 62.00%. **KEPT — new best.**

---

## Session: 2026-07-06 (continued) — Quantizer indexing change

### exp-094: Post-refinement level recomputation with index re-verification (changed-index sub-blocks only)

**Hypothesis**: The post-d grid index refinement (lines 4034-4082, from exp-055) updates grid indices for the quantized scale `d*(2*l+1)`. However, the 4-bit scale levels `l` were computed BEFORE this refinement (line 4025) using the OLD grid indices' continuous scales. For sub-blocks where indices changed, the level is stale — the optimal 4-bit scale for the new centroid values may differ from the level computed for the old centroids.

Previous experiments that tried recomputing levels after post-d refinement (exp-050, exp-060) caused catastrophic regression because they recomputed levels for ALL sub-blocks, including those WITHOUT index changes. For unchanged-index sub-blocks, the old level was optimal; changing it broke the index-scale coupling.

**This experiment restricts level recomputation to sub-blocks where at least one of the 4 grid indices changed.** For these sub-blocks, the level is genuinely stale (designed for the old centroids), and a new level is required. After recomputing the level, we RE-VERIFY the indices for the new quantized scale, ensuring the index-level pair is self-consistent.

**Why this is different from exp-060**:
- exp-060: recomputed levels for ALL sub-blocks → catastrophic
- exp-056: changed d (not levels) after post-d → catastrophic (different mechanism)
- exp-094: recomputes levels ONLY for changed-index sub-blocks, AND re-verifies indices afterward → maintains index-scale coupling

**Implementation** (~35 lines added after line 4082):
1. Add a `int index_changed[QK_K/32]` flag array, set to 0 initially
2. In the post-d refinement loop (line 4061), when a new index is accepted (line 4076-4078), set `index_changed[ib] = 1`
3. After the post-d refinement loop, for each sub-block with `index_changed[ib]`:
   a. Compute LS-optimal continuous scale for the current (updated) grid indices:
      - Loop over 4 chunks, compute `sumqx += w[i]*xb[i]*odd_centroid[i]*sign[i]`, `sumq2 += w[i]*odd_centroid[i]^2`
      - `scale_opt = sumqx / sumq2`
   b. Compute new level: `l_new = nearest_int(0.5*(1/d*scale_opt - 1))`, clamp to [0, 15]
   c. If `l_new != l_old`:
      - Update `q2[2*ib+1]` bits 28-31 with `l_new`
      - Re-run the post-d index refinement for all 4 chunks of this sub-block with the new `scale_q = d*(2*l_new+1)`
   d. This is a single update (no cascading)

**Expected**: Small KL improvement (Δ ~0.001-0.004). The effect compounds across all superblocks where post-d refinement changed indices AND the level needs updating. The re-verification ensures robustness.

**Result**: KL=1.040147 — CATASTROPHIC REGRESSION (Δ = +0.373985, +56% from best 0.666162). Same failure mode as exp-060. Even though level recomputation was restricted to changed-index sub-blocks only, changing the 4-bit level (even by 1) shifts the quantized scale `d*(2*l+1)` by `2*d`, which is enough to invalidate the post-d indices selected for the old quantized scale. The re-verification step cannot fix this: when the level changes, the scale changes, and the indices selected for the new scale are simply different from those selected for the old scale — the index-scale coupling is broken regardless of re-verification. **Confirmed: any modification that changes the 4-bit level after grid index selection causes catastrophic regression. The index-scale coupling is inviolable — indices must be selected for the final quantized scale, and the final quantized scale must be derived from the selected indices. No post-hoc adjustment works.**

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()` (~35 lines added after line 4082). **Reverted**.

---

## Session: 2026-07-06 (continued) — K-means training weight alignment with quantizer formula

### exp-095: Align K-means training weights with full quantizer weight formula (per-sub-block sigma2 + powf exponent 0.30)

**Hypothesis**: The K-means training in `iq2xxs_learn_grid()` uses only imatrix importance values for per-sample weights:
```c
sample_weights[8*s + k] = weights ? weights[(idx*8 + k) % n_per_row] : 1.0f;
```

The quantizer's four weight-computation sites all use the augmented formula:
```c
weight[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.30f);
```

This misalignment means:
1. **K-means centroid update** treats all samples equally (scaled by imatrix), while the quantizer's error evaluation prioritizes elements with large |xb| within high-variance sub-blocks.
2. **K-means assignment** uses `imatrix * L1 distance`, but the quantizer's grid index selection uses `imatrix * powf(sigma2 + xb^2, 0.30) * (scale*c - x)^2` — a fundamentally different objective.
3. The E8 warm-start dominates the K-means attractor, so imatrix-only weights barely move centroids. Adding the per-sub-block sigma2 + powf factor creates per-sample weight variation that depends on the LOCAL sub-block variance, not just the global column importance. This may nudge centroids toward positions that better serve the quantizer's actual objective.

Exp-077 tried this approach with the OLD quantizer weight formula (`sqrt(sigma2 + xb^2)` with global sigma2) and got null (KL identical to baseline). Now that the quantizer uses per-sub-block sigma2 + exponent 0.30, the per-sample weight variation is larger and more localized — sub-blocks with different variance levels now get different weight multipliers, creating meaningful weight diversity that the old global-sigma2 approach lacked.

**Implementation**: In `iq2xxs_learn_grid()` (ggml/src/ggml-quants.c):
1. After computing step/n_samples, precompute `sigma2` per 32-element sub-block for the entire tensor.
2. Modify the sample/weight collection loop to use the full weight formula: `sample_weights[k] = imatrix * powf(sigma2[sb] + val*val, 0.30f)`.
3. The per-element weight now varies with both global importance (imatrix) AND local sub-block magnitude (sigma2 + val^2).

**Expected**: Small KL improvement (Δ ~0.001-0.005, 0.15-0.75% relative) from 0.666162. Even a modest centroid shift could compound across 256 centroids × 95 tensors.

**Files changed**: `ggml/src/ggml-quants.c` only — `iq2xxs_learn_grid()` (~15 lines added/modified).

**Risk**: LOW. The change only affects K-means training (not the quantizer search). Even if the centroids move in an unhelpful direction, the post-d refinement and d optimization still use the correct quantizer weight formula. The effect is bounded by the extent of centroid drift from E8, which is small (20 iterations from warm-start).

### exp-096: Uniform neighbor search weighting — `waux = 1.0f` (zero importance weighting, pure L2-driven selection)

**Hypothesis**: The neighbor search weighting trend shows: softer waux → better KL.
- exp-090: waux=weight (effective 0.30) → REGRESSION (KL 0.682)
- exp-082: waux=sqrtf(weight) (effective 0.15) → KL 0.666
- exp-093: waux=powf(weight,0.20) (effective 0.06) → KL 0.662 (IMPROVEMENT)

If the optimal is effectively zero weighting (neighbor search should be purely L2-driven), then waux=1.0f should give the best KL. For rare off-map patterns (~5% of chunks), the importance weights are noisy — the L2 distance between the centroid values and the 2-bit pattern is a more robust selection criterion.

The post-d refinement still uses wtmp with exponent 0.35 for importance-weighted index refinement after scale quantization. So importance is still applied — just in the right stage (post-d, not pre-d).

**Implementation**: One-line change in `ggml/src/ggml-quants.c:3862`:
```c
// Before:
for (int i = 0; i < 32; ++i) waux[i] = powf(weight[i], 0.20f);
// After:
for (int i = 0; i < 32; ++i) waux[i] = 1.0f;
```

**Expected**: KL improvement (Δ ~0.001-0.003 from 0.662001). The extreme case tests whether zero effective weighting is optimal.

**Risk**: LOW — only affects neighbor search (~5% of chunks). Post-d refinement still uses importance-aware wtmp.

**Result**: KL=0.675383 — REGRESSION (Δ = +0.013382, +2.0% from best 0.662001). Uniform waux (effective exponent 0.0) overshoots the optimal. The waux effective exponent sweet spot is confirmed at 0.06 (powf 0.20). Going either softer (0.0, this experiment) or sharper (0.15 sqrt, 0.30 weight) from the sweet spot degrades KL. **Reverted**.

### exp-097: Soften post-d refinement neighbor search weight — `wtmp = powf(wtmp, 0.20)` (effective 0.07, matching main waux)

**Hypothesis**: The post-d refinement uses `wtmp[i] = qw[i] * powf(sigma2_ib + xb[i]^2, 0.35f)` for its neighbor search, giving effective exponent 0.35. This is much sharper than the main quantization's waux (effective 0.06). The asymmetry argument from exp-093 applies equally here: centroid selection (both pre-d and post-d) benefits from softer weighting that lets the grid structure dominate, while only the final error COMPARISON needs sharp weights.

The post-d refinement has TWO weight uses:
1. **Neighbor search** (line 4058): `iq2_find_best_neighbour(..., wtmp, ...)` — selects candidate centroids by weighted L1
2. **Error comparison** (lines 4072-4074): `w *= diff^2` — decides whether the new centroid is better

By softening ONLY the neighbor search weight (keeping the comparison sharp), the post-d refinement casts a wider net for candidate centroids while making the final decision with importance-aware precision.

**Implementation**: One line change in `ggml/src/ggml-quants.c:4046`:
```c
// Before:
wtmp[i] = qw[8*k+i] * powf(sigma2_per_ib[ib] + xb[8*k+i]*xb[8*k+i], 0.35f);
// After:
{ float w_ = qw[8*k+i] * powf(sigma2_per_ib[ib] + xb[8*k+i]*xb[8*k+i], 0.35f);
   wtmp[i] = powf(w_, 0.20f); }
```

**Expected**: Small KL improvement (Δ ~0.001-0.003 from 0.662001). The post-d neighbor search shares the same role as the pre-d neighbor search (centroid selection), so it should benefit from the same soft-weighting treatment.

**Risk**: LOW — the error comparison still uses the sharp weight (0.35). Only the neighbor list centroid selection is softened. Revert is one line.

**Result**: KL=0.687299 — REGRESSION (Δ = +0.025298, +3.8% from best 0.662001). The softened post-d wtmp degrades the refinement's ability to select the correct centroid at the quantized scale. Unlike the pre-d search (which has a correction step via post-d refinement), the post-d refinement is the FINAL selection — it needs sharp weighting to properly discriminate between near-identical centroids. **Reverted**.

### exp-098: Finer waux — `powf(weight, 0.10)` (effective exponent 0.03, between sweet spot 0.06 and uniform 0.0)

**Hypothesis**: The waux effective exponent has the trend: 0.15→0.06 improves (-0.62%), 0.0→regresses (+2.0%). The true optimum lies between 0.06 and 0.0. Trying powf(weight, 0.10) (eff 0.03) probes the midpoint between the sweet spot (0.06) and the overshoot (0.0).

**Implementation**: One-line change in `ggml/src/ggml-quants.c:3862`:
```c
// Before:
for (int i = 0; i < 32; ++i) waux[i] = powf(weight[i], 0.20f);
// After:
for (int i = 0; i < 32; ++i) waux[i] = powf(weight[i], 0.10f);
```

**Expected**: Small KL improvement (Δ ~0.0005-0.002 from 0.662001) or null if the optimum is a broad plateau around 0.06.

### exp-099: L1-based weight formula — replace sigma2 (mean of squares) with mean_abs (mean of |xb|)

**Hypothesis**: The weight formula `weight[i] = qw[i] * powf(sigma2 + xb², exponent)` uses an L2-based magnitude baseline `sigma2 = mean(xb²)`. The quantizer's evaluation uses L1 distance (weighted L1 in neighbor search, L1-weighted error). The L2-based baseline is the only L2 component in the entire pipeline; the L2 mean-of-squares is more sensitive to outliers than L1 mean-of-abs.

By changing to an L1-based baseline `mean_abs = mean(|xb|)` with `powf(mean_abs + |xb|, exponent)`, the weight formula aligns with the quantizer's native L1 metric. For sub-blocks with outlier elements, the L1 baseline is less inflated, preserving better per-element weight discrimination.

This is structurally different from:
- **exp-078**: Changed sigma2 granularity superblock→sub-block (still L2-based)
- **exp-079**: Per-chunk sigma2 (still L2, regressed from over-localization)  
- **exp-087**: Linear-L1 `1+|xb|` (no per-sub-block baseline, no powf — catastrophic)
- **exp-091**: Trimmed-mean sigma2 (still L2-based, regressed)
- **This changes the fundamental metric from L2 (sum of squares) to L1 (sum of abs)**

**Implementation**: 4 lines changed in `quantize_row_iq2_xxs_impl()`:
1. Line 3858-3860: Replace `s2 += xb[i]*xb[i]` and `sigma2_per_ib[ib] = s2/32` with `s1 += fabsf(xb[i])` and `abs_mean_per_ib[ib] = s1/32`
2. Lines 3861, 4003, 4046, 4072: Replace `sigma2_per_ib[ib] + xb[...]*xb[...]` with `abs_mean_per_ib[ib] + fabsf(xb[...])`

**Files changed**: `ggml/src/ggml-quants.c` only — `quantize_row_iq2_xxs_impl()`.

**Expected**: Small KL improvement (Δ ~0.001-0.004, ~0.15-0.6% relative) from 0.662001. The L1-based baseline should be particularly beneficial for attention tensors with outlier-heavy sub-blocks. The effect compounds across all 95 IQ2_XXS tensors.

---

### exp-100: Decouple imatrix importance from magnitude in waux — direct formula for neighbor search weight (REGRESSION)

**Hypothesis**: Replace `waux[i] = powf(weight[i], 0.20f)` with `waux[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.06f)` to use linear imatrix instead of qw^0.20.

**Result**: KL=0.690359 — REGRESSION (+4.3% from best 0.662001). The qw^0.20 softening is necessary — linear imatrix overweights high-importance elements in the neighbor search for off-map patterns. Both imatrix and magnitude must be softened equally via a single powf(weight, 0.20). **Reverted**.

### exp-102: Sharpen post-d refinement weight exponent from 0.35 to 0.40 (keep d-opt at 0.35)

**Hypothesis**: The three-stage weight profile has the asymmetry:
- **main (0.30)**: Softest — balanced per-element optimization
- **d-opt (0.35)**: Sharper — prioritizes high-magnitude elements during scale selection
- **post-d (0.35)**: Same as d-opt — BUT the post-d refinement is the FINAL index selection and may benefit from even sharper weighting to better discriminate between near-identical centroids at the quantized scale

Exp-085 (d-opt/post-d = 0.50, main = 0.30) regressed because 0.50 was too sharp for both. Exp-086 (d-opt/post-d = 0.40, main = 0.30) also regressed, changing BOTH d-opt AND post-d together.

But the post-d stage is fundamentally different from d-opt:
- d-opt evaluates which superblock scale `d` minimizes total error — softer weighting gives better cross-sub-block balance
- post-d selects the BEST grid index for each 8D chunk at the quantized scale — sharper weighting gives better per-element discrimination

By sharpening ONLY post-d (from 0.35 to 0.40) while keeping d-opt at 0.35, we widen the post-d vs d-opt asymmetry. This is principled: the final index selection needs sharper importance weights than the scale selection.

**Implementation**: 2 lines changed in `ggml/src/ggml-quants.c`:
```c
// Line 4046: Post-d refinement candidate weight (0.35 → 0.40)
wtmp[i] = qw[8*k+i] * powf(sigma2_per_ib[ib] + xb[8*k+i]*xb[8*k+i], 0.40f);
// Line 4072: Post-d refinement comparison weight (0.35 → 0.40)
float w = qw[8*k+i] * powf(sigma2_per_ib[ib] + xb[8*k+i]*xb[8*k+i], 0.40f);
```

Line 4003 (d-opt) stays at 0.35. Lines 3861 (main) stays at 0.30. Line 3862 (waux) stays at powf(weight, 0.20).

**Expected**: Small KL improvement (Δ ~0.001-0.004, ~0.15-0.6% relative) from 0.662001. The effect is bounded because only post-d refinement (which only changes indices for some chunks) is affected.

**Risk**: MODERATE — the post-d uses sharper weights for both the neighbor search AND the error comparison. If the sharper weight makes the neighbor search too selective (narrowing the candidate pool), it could miss good centroids. However, the neighbor list is from kmap (precomputed), so the candidate pool is fixed — only the scoring criterion changes.

**Files changed**: `ggml/src/ggml-quants.c` only — lines 4046, 4072.

---

### exp-103: Change d optimization objective from sum-of-errors to max-of-sub-block-errors

**Hypothesis**: The current d optimization minimizes the TOTAL weighted quantization error across all 4 sub-blocks (128 elements). This sum objective can select a d that works well for 3 sub-blocks but poorly for 1 — the high-error sub-block is "tolerated" if the other 3 sub-blocks have low error.

By using a MAX objective (minimize the worst sub-block error), the d optimization must find a d that balances ALL sub-blocks. This is principled because:
1. The superblock structure shares one scale `d` across all 4 sub-blocks — the scale should be chosen to accommodate the MOST demanding sub-block
2. A sub-block with high error after quantization contributes disproportionately to the model's KL divergence, so preventing one sub-block from being very bad is more important than making three sub-blocks slightly better
3. The max objective naturally handles cases where `d_base = max_scale/31` overestimates the optimal d (because the max-scale sub-block isn't the most quantization-sensitive)

**Implementation**: In `ggml/src/ggml-quants.c`, lines 3985-4011 (d optimization loop):
```c
// Before:
float err = 0.0f;
for (int ib = 0; ib < QK_K/32; ++ib) {
    ... compute per-sub-block error ...
    err += w * diff * diff;       // sum across sub-blocks
}
if (err < best_err) { best_err = err; ... }

// After:
float err_max = 0.0f;
for (int ib = 0; ib < QK_K/32; ++ib) {
    float err_ib = 0.0f;
    ... compute per-sub-block error ...
    err_ib += w * diff * diff;    // sum within sub-block only
    if (err_ib > err_max) err_max = err_ib;  // track max
}
if (err_max < best_err) { best_err = err_max; ... }
```

This is a one-way modification (only changes which d is selected, doesn't touch indices/levels). Safe to revert.

**Expected**: Small KL improvement (Δ ~0.001-0.004) from 0.662001. The effect is bounded because d is already well-tuned by the sum objective. The max objective should help only when sub-block errors are imbalanced — a subset of superblocks.

**Risk**: LOW — the d optimization is a one-way modification. Even if the max objective selects a worse d on average (regression), the change is easily revertible. No coupling with other quantizer stages.

**Files changed**: `ggml/src/ggml-quants.c` only — lines 3985-4011.

---

### exp-101: Remove sigma2 baseline from main quantization weight only (keep for d-opt and post-d)

**Hypothesis**: The sigma2 baseline `sigma2_per_ib[ib]` in the main weight formula `weight[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.30f)` provides a cross-element floor that inflates weights for near-zero elements in high-variance sub-blocks. This dilutes the weight differentiation within the sub-block during:
1. **2-bit level assignment** — near-zero elements get artificially elevated weight, distorting the kmap pattern
2. **Initial scale computation** (sumqx/sumq2) — scale is pulled toward zero-element accommodation
3. **Sign parity fix** — the w*x^2 flip criterion is distorted by inflated near-zero weights

For the main quantization, near-zero elements contribute nothing to model output and should not influence level/scale/index selection. Removing sigma2 gives `weight[i] = qw[i] * powf(xb[i]*xb[i], 0.30f)` — purely per-element magnitude weighting.

The d optimization AND post-d refinement KEEP the sigma2 baseline (exponent 0.35) to ensure the shared superblock scale and final index refinement account for the full sub-block distribution.

This is a principled three-stage decoupling:
- **Main quantization (line 3861)**: Per-element only — `qw * (xb^2)^0.30`
- **d optimization (line 4003)**: Full formula with sigma2 — unchanged
- **Post-d refinement (lines 4046, 4072)**: Full formula with sigma2 — unchanged
- **waux (line 3862)**: `powf(weight, 0.20)` using the new per-element weight — naturally gives `qw^0.20 * |xb|^0.12`

**Implementation**: One-line change in `ggml/src/ggml-quants.c:3861`:
```c
// Before:
weight[i] = qw[i] * powf(sigma2_per_ib[ib] + xb[i]*xb[i], 0.30f);
// After:
weight[i] = qw[i] * powf(xb[i]*xb[i], 0.30f);
```

The d optimization and post-d refinement lines (4003, 4046, 4072) are UNCHANGED, keeping sigma2_per_ib.

**Why this is different from exp-079/089/091**:
- exp-079: Changed sigma2 granularity (per-chunk) — overfitted
- exp-089: Normalized total weight per sub-block — destructively equalized magnitude signal
- exp-091: Trimmed mean sigma2 — reduced baseline in a different way
- **This removes sigma2 completely from main quantization — a structural change to how weights are computed, not how sigma2 is estimated**

**Risk**: MODERATE. If the main quantization becomes too sensitive to element-level fluctuations without the sigma2 floor, level assignment may become unstable, especially for sub-blocks where most elements are near zero. However, for such sub-blocks, the per-superblock scale and d optimization (with sigma2) will still find appropriate values.

**Expected**: KL improvement (Δ ~0.002-0.008, ~0.3-1.2% relative) from 0.662001. The effect should be largest for sub-blocks with mixed magnitudes (few large, many near-zero), common in attention tensors.

**Files changed**: `ggml/src/ggml-quants.c` only — line 3861.
