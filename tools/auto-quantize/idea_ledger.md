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

### exp-035: Re-enable K-means++ multi-trial init with nwant=8 neighbor search
**Hypothesis**: K-means++ trials (exp-017) showed no benefit with nwant=2 because the narrow neighbor search (2 distance levels) limited the quantizer's ability to exploit better grid values. With nwant=8's wider search, grid quality should now matter more — a grid with better-distributed centroids (found through K-means++ init + E8 warm-start) should enable the quantizer to find superior matches for each 2-bit pattern.

The current grid uses single-trial E8 warm-start only (20 iters, `num_trials=1`). Activate the full 5-trial pipeline (index 0 = E8 warm-start, indices 1-4 = K-means++ init) so the best of 5 initializations is picked. Learning time increases ~20-40s but quantize time stays at 272s (nwant=8).

**Expected**: KL reduction from 0.7162 to ~0.7155-0.7158. The wider search window (nwant=8) finally allows better-initialized grids to realize their quality advantage during quantization.
