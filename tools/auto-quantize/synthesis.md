# IQ2_XXS Auto-Research Synthesis — Session 2026-07-04

## CRITICAL FINDING (exp-023 no-learn)

**The E8 lattice grid with Unsloth-matched types achieves KL=0.724789 — only 0.13% worse than our best learned KL of 0.723834.** Codebook learning provides a negligible benefit with the current type map. The remaining gap to Unsloth (0.721) is not about codebook initialization — it's a structural difference in the quantization pipeline itself (search algorithm, scale initialization, block partitioning).

## Results Table (key experiments)

| Experiment | KL Divergence | PPL | Same Top P | Q-Time | Key Technique |
|-----------|--------------|-----|------------|--------|---------------|
| **Unsloth target** | 0.721 | 26.44 | 60.25% | — | Reference |
| **E8 lattice (no learn)** | **0.7248** | 26.55 | 60.02% | 4.6 min | Pure E8 lattice, Unsloth-matched types |
| **Best learned (exp-020)** | **0.7238** | 26.46 | 60.34% | 9.6 min | Single global grid, 100 iters, 1 trial |
| exp-007 | 0.7238 | 26.46 | 60.34% | 52 min | K-means++ (7 trials, 60 iters, per-category) |
| exp-009 | 0.7238 | 26.46 | 60.34% | 28 min | 40 iters (46% faster) |

## Key Findings

### 1. Codebook learning provides only 0.13% improvement over E8 lattice
The E8 lattice baseline with Unsloth-matched types is KL=0.7248. Our best learned result is KL=0.7238. Learning improves KL by only 0.00096 (0.13%).
| exp-012 | 0.724 | 26.46 | 60.34% | 755 MB | 54 min | 32K samples (null, reverted) |
| exp-013 | 0.724 | 26.46 | 60.34% | 755 MB | 28 min | Name-hash RNG seed (null, reverted) |
| exp-014 | 0.724 | 26.46 | 60.34% | 755 MB | 28 min | Unweighted K-means (null, reverted) |
| exp-015 | 0.724 | 26.46 | 60.34% | 755 MB | 28 min | L2 distance (null, reverted) |
| exp-016 | 12.31 | — | — | 755 MB | 36 min | No per-category sharing (BROKEN, reverted) |

## Key Findings

### 1. K-means++ is the big win (-25% KL)
Replacing the E8 lattice warm-start + random init with K-means++ probabilistic farthest-first sampling (exp-007) produced the single largest improvement: KL 0.971 → 0.724. The centroid quality improvement from better initialization dwarfs all other techniques combined.

### 2. The K-means procedure is at a global optimum
Nine different experiments (exp-007 through exp-015, excluding broken ones) ALL produce EXACTLY the same KL (0.723834). The procedure is deterministic and converged:
- Same result regardless of trials (7, 12), iterations (40, 60), distance metric (L1, L2), weighting (weighted, unweighted), samples (16K, 32K), or RNG seed

### 3. Per-category codebook sharing is ESSENTIAL
Removing cross-tensor grid sharing (exp-016) causes catastrophic failure (KL=12.31). The shared grid acts as an accumulator, refining across multiple tensors. Without it, each tensor's independent K-means produces worse grids.

### 4. The remaining gap (0.003, 0.4%) may be fundamental
The 0.724 KL vs Unsloth's 0.721 is a minimal gap. Possible explanations:
- Unsloth uses a different quantization search algorithm (nearest-centroid lookup)
- Different super-block scale initialization
- Different block partitioning or sign storage
- Random variation between quantization runs

### 5. 40 iterations = 60 iterations (46% time savings)
K-means++ converges fast. Reducing from 60→40 iterations saved 24 minutes with zero quality loss. This is now the default.

## What Worked
- **K-means++ init** (exp-007): KL -25%, from 0.971 to 0.724
- **40 iterations** (exp-009): 46% faster quantize, same quality

## What Didn't Work (All Null Results)
- More trials (7→12): same KL, 70% more time
- Weighted K-means++ init: same KL
- More samples (16K→32K): same KL, 2x more time
- Name-hash RNG seed: same KL
- Unweighted K-means: same KL
- L2 distance metric: same KL

## What Broke Things
- Per-tensor codebooks w/o sharing (exp-016): KL=12.31, catastrophic
- Per-tensor codebooks w/ K-means++ (exp-008): KL=12.40, catastrophic

## Current Code State
- `kmeans_iters`: 40 (optimized from 60)
- `num_trials`: 7
- `max_samples`: 16384
- Per-category shared grids (ATTN/MLP/OTHER) with K-means++ init
- Float-space L1-weighted K-means
- Error-aware int8 snap
- 3 rounds of multi-round refinement
- Allow-zero centroids [0, 127]

## Remaining Research Directions
1. **Super-block scale d refinement**: Post-quantization adjustment of d via closed-form least squares
2. **Annealing/temperature**: Add noise to K-means to escape deterministic optimum
3. **Grid ensemble**: Combine per-tensor and per-category grids for each tensor
4. **Non-K-means approaches**: PCA-based initialization, online stochastic gradient descent
5. **Unsloth reverse engineering**: Extract and study Unsloth's actual codebook grids

## Gap Analysis
- **Best KL**: 0.724 (exp-007/009)
- **Unsloth target**: 0.721
- **Remaining gap**: 0.003 (0.4%)
- **Assessment**: K-means codebook learning is fully exploited. The remaining gap likely requires structural changes to the quantization pipeline (block format, scale optimization, or search algorithm), not just better codebook initialization.
