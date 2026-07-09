# IQ4_XS Auto-Research Synthesis — Session 2026-07-09

## Baseline: KL 0.024916 (stock IQ4_XS vs Q8_0 reference)

| Metric | Q8_0 Reference | IQ4_XS (stock) |
|--------|---------------|----------------|
| PPL | 6.7917 | 6.8952 |
| KL divergence | 0.0 | **0.0249** |
| Same top p | 100% | **94.17%** |
| RMS Δp | 0.0% | 4.245% |

## Best Result: exp-007 — KL 0.024811 (marginal improvement)

Only 1 of 10 experiments improved over stock: per-sub-block sigma2 with sqrtf (exp-007).
All other experiments regressed or failed.

## Results Summary (Experiments 001-010)

| Exp | Description | KL | Δ from baseline | Status |
|-----|-------------|-----|---------|--------|
| 001 | Weight exponent 0.30 + per-sub-block sigma2 | — | — | Failed (timeout) |
| 002 | exp2f(0.30*log2f(x)) + per-sub-block sigma2 | — | — | Failed (timeout) |
| 003 | Weight exponent 0.30 + per-sub-block sigma2 (20 min) | 0.025029 | +0.5% | Regression |
| 004 | Superblock d candidate search (±8%, 33 candidates) | 0.031630 | +26.9% | Regression |
| 005 | Post-d level perturbation (±1 level) | 0.025166 | +1.0% | Regression |
| 006 | K-means learned 16-entry codebook | 0.027259 | +9.4% | Regression |
| **007** | **Per-sub-block sigma2 with sqrtf** | **0.024811** | **-0.4%** | **BEST** |
| 008 | Reduce ntry from 7 to 3 | 0.026841 | +7.7% | Regression |
| 009 | Superblock d divisor 32→28 | 0.025410 | +2.0% | Regression |
| 010 | Remove sigma2 from weight formula | 0.027991 | +12.3% | Regression |

## Key Findings

### 1. IQ4_XS is near a Pareto optimum
The stock parameters are remarkably well-tuned. Of 10 experiments, 9 regressed and only 1 showed marginal improvement (-0.4% KL). IQ4_XS quantization is fundamentally different from IQ2_XXS which had large optimization headroom (final KL improved 35% over stock).

### 2. Weight formula sensitivity
IQ4_XS is very sensitive to the importance weight formula:
- The sigma2 floor is critical (exp-010: removing it causes 12.3% regression)
- The 0.50 exponent (sqrtf) is optimal (exp-003: 0.30 regresses)
- Per-sub-block sigma2 marginally helps (exp-007: -0.4%)

### 3. Non-transferability from IQ2_XXS
Most improvements that worked for IQ2_XXS fail for IQ4_XS:
- Weight exponent 0.30: **regression** (not improvement)
- Superblock d optimization: **regression** (was second-best for IQ2_XXS)
- Post-d refinement: **null** (small benefit in IQ2_XXS)
- K-means codebook: **regression** (IQ2_XXS has learned grid, not applicable)

The fundamental difference: IQ4_XS uses a fixed 16-entry non-uniform codebook with 4 sub-blocks × 32 elements, while IQ2_XXS uses a learned 8D grid with neighbor search. The fixed codebook makes IQ4_XS less adaptable to per-weight optimization strategies.

### 4. Codebook is near-optimal
K-means learning from weight samples produced worse KL (exp-006). The hand-tuned `kvalues_iq4nl` outperforms data-driven clustering, likely because it was optimized for downstream task performance, not weight-space MSE.

### 5. ntry=7 is optimal for per-sub-block d refinement
Reducing ntry to 3 (exp-008) caused significant regression, confirming that 15 d-candidates are necessary for quality. Increasing ntry (e.g., 10 or 15) may help but exceeds the 20-minute quantize limit without ≥10% KL gain.

## Remaining Unexplored Directions

| Direction | Rationale | Risk |
|-----------|-----------|------|
| Increase ntry to 10 | More d refinement may help | Higher quantize time |
| Global sigma2 + per-sub-block sigma2 blend | Hybrid weighting | Likely neutral/regression |
| Codebook per weight type (FFN/attn/norm) | Different weight distributions | Complex implementation |
| Importance-weighted K-means (using imatrix) | Better than uniform K-means | Long implementation |
| Two-pass quantization with residual | Second pass captures residuals | Doubles quantize time |
| Different level encoding (e.g. 5-bit per level) | Would require format change | Block size change |
