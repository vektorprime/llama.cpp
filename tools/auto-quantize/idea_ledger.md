# IQ2_XXS Auto-Research Idea Ledger

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

**Implementation**: In `quantize_row_iq2_xxs_impl`, wrap the refinement block (lines 3927-3952) in a 2-iteration loop.

**Expected**: Marginal KL improvement (0.0001-0.0005) for blocks with unusual weight distributions where scale refinement is large.

Note: exp-043 tried a combined change (wider search ±12 step 0.2 + 2 iteration refinement) and regressed. This experiment keeps the search range unchanged and only adds the second refinement iteration — isolating the effect of iterative refinement.
