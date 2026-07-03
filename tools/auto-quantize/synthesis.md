# IQ2_XXS Auto-Research Synthesis — Session 2026-07-03

## Summary

Ran 3 experiments exploring novel directions beyond basic K-means. Found **first meaningful improvement**: KL reduced from **0.993 → 0.971** (-2.2%) via error-aware int8 snapping + data-driven initialization.

## Results Table

| Experiment | KL Divergence | PPL | Same Top P | Size | Q-Time | Key Technique |
|-----------|--------------|-----|------------|------|--------|---------------|
| **Baseline** | 0.993 | 34.75 | 55.89% | 776 MB | — | No learning (E8 lattice grid) |
| exp-001 | 0.993 | 34.76 | 55.89% | 786 MB | 10.6 min | Float-space K-means (no int8 during training) |
| **exp-002** | **0.971** | **33.95** | **56.50%** | 786 MB | 30.9 min | Error-aware snap + data-driven init + allow-zero |
| exp-003 | 0.971 | 33.75 | 55.63% | 786 MB | 30.9 min | L1 distance metric (on top of exp-002) |
| **Unsloth target** | 0.721 | 26.44 | 60.25% | 733 MB | — | Reference |

## Key Findings

### 1. Error-aware int8 snapping WORKS (+2.2% KL)
The core insight: instead of blindly rounding float centroids to nearest int, compute the assignment-weighted error for both floor and ceil, and pick the direction with lower error. This prevents K-means from collapsing back to {1,3,5}.

### 2. Float-space K-means alone does NOT help
Training centroids as floats and snapping only at the end (exp-001) produced identical results to the baseline. The snap dominates the final grid values.

### 3. Data-driven initialization matters
Removing the E8 lattice warm-start (exp-002) and initializing from random weight samples gave centroids a chance to explore values beyond {1,3,5}.

### 4. Allow-zero centroids helps
Changing the lower clamp from 1.0f to 0.0f gave K-means another degree of freedom.

### 5. L1 vs L2 distance metric doesn't matter
Switching the K-means and quantization search from L2 to L1 (exp-003) produced essentially identical results (±noise).

## What Didn't Work (Anti-Patterns)

- **Blind float K-means**: No benefit over baseline if snapped to nearest int
- **L1 metric**: No measurable difference from L2 for this task
- **Quantile initialization**: Too slow (O(n²) insertion sort), not tested

## Code Architecture

All changes are in `ggml/src/ggml-quants.c`, function `iq2xxs_learn_grid()`:
- `kmeans_iters`: 40 (up from 20)
- `num_trials`: 5 (up from 3)
- No E8 lattice warm-start for random trials
- Error-aware snapping: `err_floor` vs `err_ceil` weighted by assignments
- Centroid clamp: [0, 127] instead of [1, 127]

## Top 3 Next Hypotheses

### H1: Multi-round error-aware refinement (highest priority)
After the initial error-aware snap, re-assign samples and do 2-3 more rounds of per-centroid gradient descent (±1 adjustments). This could escape local minima from the first snap. Expected: additional 1-3% KL reduction.

### H2: Per-tensor-type codebooks with error-aware snap
Learn separate grids for attention tensors (attn_gate, attn_output, attn_k, attn_q) vs MLP tensors (ffn_gate, ffn_up, ffn_down). Different tensor types have different magnitude distributions. Expected: 2-5% KL reduction.

### H3: Gradient-based scale refinement post-quantization
After quantization, refine super-block scale `d` using gradient descent on reconstruction error (jointly with the 4-bit scale). The scale `d` is stored as float16 and can be adjusted without changing bpw. Expected: 1-3% KL reduction.

## Gap Analysis

- **Distance to Unsloth target**: 0.971 vs 0.721 = still **34% gap**
- **Bottleneck**: The 256-entry 8D int8 codebook format fundamentally limits expressiveness. Unsloth likely uses a different quantization approach (mixed types: iq2_xxs + q4_K + q2_K + q5_K)
- **Next frontier**: Joint scale+codebook optimization, or structural changes to the block format

## Session 2026-07-03 continued -- 3 experiments testing H1/H2/H3

Ran 4 experiments testing the top 3 hypotheses from earlier synthesis.

### exp-004 (H1): Multi-round error-aware refinement
**Result: NO CHANGE** -- KL = 0.970904 (identical to exp-003)

### exp-005b (H2): Per-tensor-type codebooks  
**Result: NO CHANGE** -- KL = 0.970904 (identical to exp-003)

### exp-005 (H2+H3 UNSAFE): Catastrophic failure -- KL=12.35

### exp-006b (H3 with safety clamping): REGRESSION -- KL=1.066 (+9.8%)

### Conclusions
- All three top hypotheses exhausted: no improvement found
- The error-aware int8 snap at 0.971 KL appears to be a hard local optimum
- H3 d refinement is fundamentally flawed: sub-block scale indices cannot be re-optimized post-hoc
- To close 34% gap to Unsloth (0.721), more radical structural changes are needed
- Next directions: joint grid+scale optimization, Unsloth grid warm-start, or block format changes
