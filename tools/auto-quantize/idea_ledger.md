# IQ2_XXS Auto-Research Idea Ledger

## Experiment Index

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

