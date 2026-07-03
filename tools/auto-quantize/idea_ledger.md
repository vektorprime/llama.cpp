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

### exp-20260703-006: Gradient-based super-block scale refinement (REGRESSION)
**Hypothesis**: After quantization, refine super-block scale d via weighted least squares to minimize reconstruction error. This is a 1-D closed-form optimization.

**Result**: KL = 1.066 (regression). Post-hoc d refinement is fundamentally flawed because sub-block scale indices are already locked in.

### exp-20260703-007: K-means++ initialization + more thorough convergence (RESULT: MAJOR WIN — NEAR-UNSLOTH)
**Hypothesis**: The previous K-means initialization (random sampling) combined with only 40 iterations and 5 trials insufficiently explores the grid space. K-means++ initialization (probabilistic farthest-first seeding) ensures initial centroids span the weight distribution, reducing poor local minima convergence. Increasing to 60 iterations and 7 trials provides more thorough optimization.

**Result**: KL = **0.723834** (PPL=26.455, Same top p=60.339%). **25% reduction from 0.971 to 0.724!** Nearly matches Unsloth reference (0.721, PPL=26.44, Same top p=60.19%). The K-means++ initialization was the critical change — it produces grids that actually outperform the E8 lattice by a wide margin where previous approaches couldn't.

**Lesson**: The E8 lattice warm-start and random initialization both bias the optimization toward suboptimal grid configurations. K-means++ with more thorough convergence discovers fundamentally better codebook arrangements. The synthesis conclusions that "all three top hypotheses are exhausted" and "more radical structural changes are needed" were premature — the solution was a better initialization, not a different algorithm.

### exp-20260703-008: Per-tensor codebooks with K-means++ init
**Hypothesis**: The per-category codebook sharing (ATTN/MLP/OTHER) was designed before K-means++ init existed. Now that K-means++ consistently produces good grids (exp-007), different tensors should produce genuinely different grids specialized to their own weight distributions. Removing the category sharing should allow each tensor to learn its own optimal grid independently.

**Expected**: KL improvement from 0.724 → 0.722 or better. Per-tensor specialization should beat per-category aggregation.

**Risk**: The per-category approach might provide beneficial data aggregation (averaging noise across multiple tensors). If KL regresses, revert.
