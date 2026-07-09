# IQ4_XS Auto-Research Idea Ledger

## Experiment Index

| Exp | Description | Outcome |
|-----|-------------|---------|
| 001 | Weight exponent 0.30 + per-sub-block sigma2 | Failed — quantize timed out (>7 min) |
| 002 | Weight exponent 0.30 + per-sub-block sigma2 (FAST pow: exp2f) | Failed — quantize timed out (>7 min) |
| 003 | Weight exponent 0.30 + per-sub-block sigma2 (20 min timeout) | Regression — KL 0.025029 vs 0.024916 baseline (slightly worse) |
| 004 | Superblock d candidate search (±8%, 33 candidates, 0.5% step) | Regression — KL 0.031630 vs 0.024916 baseline (worse) |
| 005 | Post-d level perturbation (±1 around chosen l) | Regression — KL 0.025166 vs 0.024916 baseline |
| 006 | K-means learned 16-entry codebook from weight samples | Regression — KL 0.027259 vs 0.024916 baseline |
| 007 | Per-sub-block sigma2 with sqrtf (isolated from exp-003's powf) | Improvement — KL 0.024811 vs 0.024916 baseline (marginal) |
| 008 | Reduce ntry from 7 to 3 (less per-sub-block d overfitting) | Regression — KL 0.026841 vs 0.024811 best |
| 009 | Superblock d divisor 32→28 (finer sub-block scale quantization) | Regression — KL 0.025410 vs 0.024811 best |
| 010 | Remove sigma2 from weight formula (qw * xb^2 only) | Regression — KL 0.027991 vs 0.024811 best |
| 011 | Increase ntry from 7 to 10 (more per-sub-block d refinement) | Regression — KL 0.025423 vs 0.024811 best |

---

> **IQ2_XXS ledger archived in git history.** The IQ2_XXS research (exp-001 through exp-107)
> achieved best KL 0.662001 on Qwen3.5-2B. See the `auto_research_llama` branch or
> earlier commits in `auto_research_iq4xs_llama` for the full IQ2_XXS experiment index
> and detailed hypotheses.
>
> Key transferable findings:
> - Weight exponent 0.30 (main), 0.35 (d-opt/post-d) with per-sub-block sigma2
> - Superblock d optimization: 65 candidates, 0.5% step, ±16% range
> - Post-d grid index recomputation with quantized scale
> - waux = powf(weight, 0.20) for neighbor search softening
>
> These should be validated on IQ4_XS.

---

## exp-001: Weight exponent 0.30 + per-sub-block sigma2

**Hypothesis:** The single biggest improvement in IQ2_XXS research was changing the weight formula from `qw * sqrt(sigma2 + x^2)` (exponent 0.50) to `qw * pow(sigma2_ib + x^2, 0.30)` with per-sub-block sigma2 instead of global superblock sigma2. This gave ~5% cumulative KL gain on IQ2_XXS.

**Changes:**
1. Remove global sigma2 computation (lines 5589-5591)
2. Inside the ib loop, compute per-sub-block sigma2:
   ```c
   float sigma2_ib = 0;
   for (int j = 0; j < block_size; ++j) sigma2_ib += xb[j]*xb[j];
   sigma2_ib *= 2.f/block_size;
   ```
3. Change weight formula from `sqrtf(sigma2 + xb[j]*xb[j])` to `powf(sigma2_ib + xb[j]*xb[j], 0.30f)`

**Expected outcome:** KL improvement from ~0.0249 baseline.

**Actual outcome:** FAILED — quantize timed out at 7 min HARD limit. Only ~480/866 tensors completed (blk.35/64). `powf()` is ~5-10x slower than `sqrtf()`, making full quantize impossible within the 7-min budget. Code discarded. Hypothesis and results remain documented for reference.

**Lesson:** `powf(x, 0.30f)` is too slow for production. Alternative: use `sqrtf(sqrtf(sqrtf(x)))` ~ x^0.25 as a cheap approximation, or precompute via lookup table, or use a faster exponent like 0.50 (sqrtf) or 0.25 (double sqrtf).

---

## exp-002: Weight exponent 0.30 + per-sub-block sigma2 (FAST pow: exp2f)

**Hypothesis:** Exp-001 failed because `powf(x, 0.30f)` is too slow for 27B elements. Using `exp2f(0.30f * log2f(x))` instead — which is ~3-5x faster than `powf()` — should complete quantization within the 7-min budget while delivering the same KL improvement.

**Changes:**
1. Remove global sigma2 computation (lines 5589-5591)
2. Inside the ib loop, compute per-sub-block sigma2:
   ```c
   float sigma2_ib = 0;
   for (int j = 0; j < block_size; ++j) sigma2_ib += xb[j]*xb[j];
   sigma2_ib *= 2.f/block_size;
   ```
3. Change weight formula from `sqrtf(sigma2 + xb[j]*xb[j])` to `exp2f(0.30f * log2f(sigma2_ib + xb[j]*xb[j]))`

**Expected outcome:** KL improvement from ~0.0249 baseline, quantize completes within 7 min.

**Actual outcome:** FAILED — quantize timed out at 7 min HARD limit. Only ~485/866 tensors completed (blk.36/63). `exp2f(0.30*log2f(x))` is still too slow — `log2f` is also expensive per element. The combined `log2f` + `exp2f` approach is not significantly faster than `powf`.

**Lesson:** Any per-element transcendental math in the weight formula is too slow for 27B elements. The weight formula must use cheap operations only: `sqrtf`, multiplication, addition. Exponents other than 0.50 (sqrtf) or 0.25 (double sqrtf) require a different approach — precomputed LUT, or vectorized approximation.

---

## exp-003: Weight exponent 0.30 + per-sub-block sigma2 (20 min timeout)

**Hypothesis:** Exp-001 and exp-002 failed because the 7-minute limit was too tight. The timeout has now been increased to 20 minutes (1200s) per quantize. `powf()` on 866 tensors × 27B elements should complete within the 20-minute window. This is the same code change as exp-001.

**Changes:**
1. Remove global sigma2 computation (lines 5589-5591)
2. Inside the ib loop, compute per-sub-block sigma2:
   ```c
   float sigma2_ib = 0;
   for (int j = 0; j < block_size; ++j) sigma2_ib += xb[j]*xb[j];
   sigma2_ib *= 2.f/block_size;
   ```
3. Change weight formula to: `weight[j] = qw[j] * powf(sigma2_ib + xb[j]*xb[j], 0.30f)`

**Expected outcome:** KL improvement from 0.024916 baseline (stock). Quantize expected to complete within 20 min given the increased timeout. Improvement expected: ~30% KL reduction to ~0.017 based on IQ2_XXS transfer results.

**Actual outcome:** Regression — KL 0.025029 ± 0.001012 vs baseline 0.024916. Quantize completed in 700.25s (11.7 min), well within timeout. Eval completed on device 1 (RTX 3080, device 0 had CUDA error). KL slightly increased, PPL 6.8974 vs baseline 6.8952, Same top p 94.194% vs 94.17% (marginally better). Net result: no improvement, minor regression.

**Lesson:** Weight exponent 0.30 + per-sub-block sigma2 does NOT transfer to IQ4_XS the same way it did for IQ2_XXS. Possible reasons:
1. IQ4_XS has a 16-entry non-uniform codebook vs IQ2_XXS learned grid — weight exponent affects very different quantization dynamics
2. The baseline weight formula `qw * sqrtf(sigma2 + x^2)` with global sigma2 may already be near-optimal for IQ4_XS
3. Per-sub-block sigma2 may cause overfitting to local statistics, increasing variance without improving accuracy

---

## exp-004: Superblock d candidate search (±8%, 33 candidates, 0.5% step)

**Hypothesis:** The current superblock d is computed as `-max_scale/32` — picking maximum dynamic range for the 4-bit sub-block levels. This ignores the weighted reconstruction error across all sub-blocks. In IQ2_XXS, a candidate search over d (~65 candidates, ±16% range, 0.5% step) was the second-biggest improvement (~2% KL gain). The same approach should transfer to IQ4_XS: search 33 candidates (±8%, 0.5% steps) around `d_base = -max_scale/32`, pick the one minimizing `sum(sub-block weighted MSE)`. This is cheap (33 candidates × 4 sub-blocks × 32 elements = 4224 extra MACs per superblock — negligible vs the existing algorithm).

**Changes:**
1. Replace lines 5653-5675: instead of `float d = -max_scale/32;` directly, try candidates `d_try = d_base * (1.0 + is * 0.005)` for `is = -16..16`
2. For each candidate, compute weighted reconstruction error across all non-zero sub-blocks
3. Pick the d with minimum error, then recompute L with quantized sub-block scales
4. Skip `scales[ib] == 0` sub-blocks (all-zeros)

**Expected outcome:** KL improvement from 0.024916 baseline, targeting 0.0245 or better. Quantize time should increase by <5% (negligible overhead). Should complete well within 20 min.

**Transfer note:** This same idea was the second-biggest improvement in IQ2_XXS. The d-candidate search doesn't depend on the codebook structure — it's purely about better superblock scale selection given sub-block scale levels.

**Actual outcome:** Regression — KL 0.031630 ± 0.001150 vs baseline 0.024916. PPL improved slightly (6.8660 vs 6.8952) but KL is significantly worse, especially in the median percentile (0.009577 vs ~0.007 baseline). Same top p 93.76% vs 94.17% (worse). The candidate search increased median KL substantially, suggesting that picking d purely by minimizing weighted reconstruction error over the initial L assignments overfits to sub-optimal codebook indices. The initial L assignments were made with per-block d, not superblock d, so the reconstruction error proxy used in the search is mismatched. Quantize time increased from 700s → 782s as expected.

---

## exp-005: Post-d level perturbation (±1 level around chosen l)

**Hypothesis:** After the superblock d and per-sub-block levels `l` are finalized, the codebook entries are chosen with `idl = 1/(d*l)`. While `l` is optimal in isolation (nearest integer to `id*scales[ib]`), the interaction between `l` and the 32 codebook entries may be suboptimal. Trying `l-1` and `l+1` as alternatives — re-evaluating all 32 codebook entries with the new quantized scale `d * l_try` — may find a better combination. This is analogous to the post-d refinement from IQ2_XXS.

**Changes:** After the existing superblock d quantization loop (lines 5659-5675), add a level perturbation pass:
1. Save per-sub-block levels `l` before packing
2. For each sub-block, try `l-1`, `l`, `l+1`
3. For each candidate level, re-evaluate ALL 32 codebook entries via `best_index_int8()`
4. Compute weighted reconstruction error for each candidate
5. Pick the level + codebook indices combination with minimum error
6. If level changed, repack the scale bits in `scales_l` and `scales_h`

**Expected outcome:** Small KL improvement (~0.2-0.5% reduction to ~0.0248-0.02485). The level perturbation is cheap (3× re-evaluation of 32 elements per sub-block = 384 extra `best_index_int8` calls per superblock, negligible overhead). Quantize time should increase by <1%.

**Actual outcome:** Regression/null — KL 0.025166 ± 0.001036 vs baseline 0.024916. Mean KL slightly higher (0.025166 vs 0.024916), well within 1σ noise (σ = 0.001036). PPL 6.8939 vs baseline 6.8952 (negligibly better). Same top p 94.159% vs 94.17% (negligibly worse). Quantize time 760s, comparable to baseline (~700s) and exp-003 (700s) but faster than exp-004 (782s). The level perturbation had essentially no effect — likely because `best_index_int8` already finds the optimal codebook entry for each level, so trying adjacent levels rarely finds a better combination. Code discarded.

**Lesson:** Unlike IQ2_XXS, where post-d refinement provided ~0.6% KL gain, IQ4_XS's fixed 16-entry codebook means `best_index_int8` already exhaustively checks all entries. The quantized scale `idl` is already used in the main quantization loop (line 5667). Level perturbation adds nothing because `l` was already chosen as the optimal integer nearest to `id*scales[ib]`, and the codebook entries for `l-1`/`l+1` are already evaluated with the correct scale during the perturbation — but `best_index_int8` with a slightly different scale finds the same or worse entries. The original `l` is already optimal.

---

## exp-006: K-means learned 16-entry codebook from weight samples

**Hypothesis:** The current `kvalues_iq4nl` table (`-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113`) was designed by hand (empirically tuned). K-means clustering on actual weight samples from the BF16 model should produce a better codebook that minimizes reconstruction error for this specific model's weight distribution. By collecting weight samples from the GGUF, running K-means with K=16 on absolute values, scaling to match the magnitude range (~127), and creating a symmetric codebook sorted in non-decreasing order, we get data-driven quantization levels optimized for this model.

**Changes:**
1. Collect weight samples from the BF16 GGUF (first 30 tensors, up to 500K weights)
2. Run K-means (K=16) on absolute weight values
3. Replace `kvalues_iq4nl` in `ggml/src/ggml-common.h:1110-1112` with the learned sorted values

**Expected outcome:** KL improvement from 0.024916 baseline. K-means codebook should reduce quantization error by aligning the quantization levels with the actual weight distribution. No change to quantize time (codebook is just a compile-time constant).

**Actual outcome:** Regression — KL 0.027259 ± 0.001135 vs baseline 0.024916. PPL 6.8903 vs baseline 6.8952 (marginally better). Same top p 94.076% vs 94.17% (worse). Quantize time 719.42s (unchanged from stock ~700s, as expected). The K-means learned codebook `[-109, -81, -62, -47, -35, -24, -15, -5, 4, 13, 23, 34, 47, 62, 80, 109]` is more symmetric and concentrated near zero than the hand-tuned codebook `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`. Code discarded.

**Lesson:** The hand-tuned `kvalues_iq4nl` codebook was probably designed with knowledge of downstream task performance, not just reconstruction MSE. K-means on weight samples optimizes for per-weight MSE but ignores:
1. The codebook interacts with the superblock scale `d` and sub-block level `l` encoding scheme — values that cluster near zero may cause level underflow 
2. The codebook is fixed across ALL weight types (attention Q, K, V, FFN up, down, gate, norm) which may have very different distributions — a unified codebook trained on diverse tensors may be worse than specialization
3. The existing codebook's wider range (|-127| to |113| = 240 span vs learned 109-(-109) = 218) provides more dynamic range headroom
4. K-means on BF16 weight samples targets weight-space reconstruction, not output-space KL divergence

**Next directions:** Instead of changing the codebook globally, consider learning per-tensor-type codebooks, or using the importance matrix to weight K-means. Alternatively, keep the existing codebook but improve the scale/level assignment (exp-003, 004, 005 already explored this path).

---

## exp-007: Per-sub-block sigma2 with sqrtf (isolated from exp-003's powf)

**Hypothesis:** exp-003 changed BOTH the sigma2 scope (global→per-sub-block) AND the weight exponent (0.50→0.30), producing a regression (KL 0.025029 vs 0.024916). It is unclear whether the regression was caused by the exponent change or the sigma2 scope change. By keeping the original exponent (0.50, sqrtf) but using per-sub-block sigma2 instead of global superblock sigma2, we isolate the sigma2 effect. Per-sub-block sigma2 provides a more locally accurate variance estimate for the importance weight formula `qw[j] * sqrtf(sigma2 + xb[j]^2)`, since each 32-element sub-block may have different scale characteristics. This is cheap: only 32 multiply-adds per sub-block (vs 256 for global sigma2, actually fewer total ops).

**Changes:** In `quantize_row_iq4_nl_impl()`:
1. Remove global sigma2 computation at lines 5589-5591
2. Inside the `ib` loop, add per-sub-block sigma2 computation:
   ```c
   float sigma2_ib = 0;
   for (int j = 0; j < block_size; ++j) sigma2_ib += xb[j]*xb[j];
   sigma2_ib *= 2.f/block_size;
   ```
3. Change weight formula from `sqrtf(sigma2 + xb[j]*xb[j])` to `sqrtf(sigma2_ib + xb[j]*xb[j])`

**Expected outcome:** If the exp-003 regression was solely from the exponent change, per-sub-block sigma2 with sqrtf should improve KL. If the regression was from per-sub-block sigma2 causing overfitting, KL will regress. Quantize time unchanged.

**Actual outcome:** Marginal improvement — KL 0.024811 ± 0.000926 vs baseline 0.024916. This is within 1σ noise (baseline noise ~0.001) but technically an improvement. PPL worsened slightly (6.9131 vs 6.8952). Same top p 94.018% vs 94.17% (worse). Quantize time 648.93s (faster than stock ~700s — removal of global sigma2 loop may help slightly). The isolated per-sub-block sigma2 provides marginally more accurate importance weights at no computational cost.

**Lesson:** The exp-003 regression was likely caused by the exponent change (0.50→0.30), NOT by per-sub-block sigma2. Per-sub-block sigma2 alone is neutral-to-slightly-beneficial. The 0.30 exponent introduced by powf() likely distorted the importance weight distribution in a way harmful to IQ4_XS's fixed codebook. IQ4_XS is more sensitive to weight formula changes than IQ2_XXS because its 16-entry codebook doesn't have the adaptive capacity of a learned grid.

---

## exp-008: Reduce ntry from 7 to 3 (less per-sub-block d overfitting)

**Hypothesis:** The `ntry=7` parameter in `quantize_row_iq4_nl_impl()` causes the inner loop to try 15 different d values per sub-block (lines 5631-5645). This refines the initial d estimate to maximize weighted reconstruction quality for that specific sub-block. However, these per-sub-block d values are later quantized against a shared superblock d (`d_super = -max_scale/32`). A more aggressively optimized per-sub-block d may have values that are harder to quantize against the superblock d (they are more spread out, leading to larger level quantization error). By reducing ntry from 7 to 3 (7 candidates instead of 15), the per-sub-block d values stay closer to their initial estimate, making them more consistent across sub-blocks and easier to quantize to the shared superblock d. This also reduces quantize time by ~50% in the inner loop.

**Changes:** In `quantize_iq4_xs()` at `ggml/quants.c:5742`, change `ntry` from 7 to 3.

**Expected outcome:** KL improvement from the current best 0.024811. Quantize time should decrease from ~649s to ~350-400s.

**Actual outcome:** Regression — KL 0.026841 ± 0.001174 vs best 0.024811. PPL 6.9243 vs 6.9131 (worse). Same top p 94.108% vs 94.018% (slightly better). Quantize time 386.14s vs 648.93s (as predicted). ntry=3 completed 2x faster but at significant KL cost.

**Lesson:** The hypothesis that per-sub-block d overfits with ntry=7 was wrong. The inner loop refinement (lines 5631-5645) is critical for finding good per-sub-block d values. 15 candidates (ntry=7) is not overfitting — it's necessary for the basic quantization quality. The ~90s saved in quantize time is not worth the 8% KL regression. This also implies that INCREASING ntry might help further (e.g., ntry=15).

---

## exp-009: Superblock d divisor 32→28 (finer sub-block scale quantization)

**Hypothesis:** The superblock d is computed as `d = -max_scale/32` (line 5656). The divisor 32 means the largest sub-block scale maps to level magnitude 32, and d = max_scale/32. With 64 quantization levels (l ∈ [-32,31]), the quantization step is d. If we use divisor 28 instead, the largest sub-block maps to level ~28 (still within [-32,31] range), and the quantization step becomes d = max_scale/28, which is 14% finer (32/28 = 1.14x). This provides finer quantization of sub-block scales, which should improve reconstruction quality. The largest sub-block scale still has headroom (level 28, ceiling at 31), so no clipping occurs. Quantize time is unchanged (divisor is a compile-time constant).

**Changes:** In `quantize_row_iq4_nl_impl()` line 5656, change `-max_scale/32` to `-max_scale/28`.

**Expected outcome:** Small KL improvement from current best 0.024811, targeting ~0.0246-0.0247. Same quantize time.

**Actual outcome:** Regression — KL 0.025410 ± 0.001052 vs best 0.024811. PPL 6.8955 vs 6.9131 (slightly better!). Same top p 94.073% vs 94.018% (marginally better). Quantize time 649.98s (unchanged). The finer scale quantization (divisor 28 vs 32) gave better PPL but worse KL — interesting tradeoff. PPL improved because sub-block scales are quantized more precisely, but KL worsened because the coarser d range (larger d) creates larger quantization steps for codebook entries.

**Lesson:** The divisor 32 gives the optimal balance between sub-block scale precision and codebook entry quantization step. Decreasing the divisor makes d smaller, giving finer sub-block scales but larger `d*l` values (more dynamic range for codebook entries). This shows the tight coupling between d and the codebook values — changing d affects both sub-block scale quantization AND the effective codebook range. The current divisor=32 is optimal for this model.

---

## exp-010: Remove sigma2 from weight formula (qw * xb^2 only)

**Hypothesis:** The importance weight formula currently uses `qw[j] * sqrtf(sigma2_ib + xb[j]^2)`. The sigma2 term acts as a floor to prevent near-zero weights for small-magnitude coefficients. But exp-003 (powf 0.30) and exp-007 (per-sub-block sqrtf) show that IQ4_XS is very sensitive to the weight formula. Removing the sigma2 term entirely — using just `qw[j] * xb[j]*xb[j]` (importance × squared magnitude) — might eliminate a source of distortion. Small weights near zero would be deprioritized, focusing the quantizer's attention on large-magnitude weights. Without the sigma2 floor and sqrt, the weight is purely importance-scaled squared magnitude, which may better align with the imatrix-based `qw` values. This also removes the per-sub-block sigma2 computation, speeding up quantize slightly.

**Changes:** In `quantize_row_iq4_nl_impl()`, remove the sigma2_ib computation and sqrtf, replacing with `weight[j] = qw[j] * xb[j]*xb[j]`.

**Expected outcome:** Unknown — could improve KL by removing unnecessary sigma2 distortion, or regress by removing beneficial floor. Quantize time should decrease by ~5-10s.

**Actual outcome:** Regression — KL 0.027991 ± 0.001172 vs best 0.024811. PPL 6.9329 (worst of all experiments). Same top p 93.841% (worst of all). Quantize time 645.66s (only ~3s faster than exp-007's 648.93s — sigma2 computation was negligible). Removing the sigma2 floor caused a 13% degradation in KL.

**Lesson:** The sigma2 floor in the weight formula is critical for IQ4_XS. Without it, small-magnitude weights get near-zero importance, allowing the quantizer to make large relative errors on them. The sigma2 term ensures that ALL weights matter, regardless of magnitude. This is especially important for IQ4_XS because the fixed 16-entry codebook has limited resolution — errors on small weights compound through downstream layers. The sigma2 floor acts as a regularization that distributes quantization error evenly across all weight magnitudes.

---

## exp-011: Increase ntry from 7 to 10 (more per-sub-block d refinement)

**Hypothesis:** exp-008 showed that reducing ntry from 7 to 3 caused a 7.7% KL regression. This strongly suggests that more d refinement iterations improve quantization quality. Increasing ntry from 7 to 10 gives 21 candidate d values per sub-block (vs 15 for ntry=7), providing finer search for the optimal per-sub-block d. The lesson from exp-008 is that the inner loop refinement is critical for finding good d values. Quantize time should increase proportionally: ~649s * (21/15) = 909s, within the 20-min limit.

**Changes:** In `quantize_iq4_xs()` line 5742, change ntry from 7 to 10.

**Expected outcome:** KL improvement from 0.024811 to ~0.0245-0.0247. Quantize time ~900s.

**Actual outcome:** Regression — KL 0.025423 ± 0.001054 vs best 0.024811. PPL 6.8969 vs 6.9131 (slightly better). Same top p 94.127% vs 94.018% (slightly better). Quantize time 843.07s (as predicted).

**Lesson:** There's an optimal ntry around 7. Both lower (ntry=3, KL 0.026841) and higher (ntry=10, KL 0.025423) are worse. The sweet spot at ntry=7 provides exactly the right amount of per-sub-block d refinement. Too few iterations (ntry=3) underfits the d estimate. Too many (ntry=10) may overfit the d to the specific codebook index assignment, making the sub-block d less compatible with the shared superblock d. ntry=7 appears optimal for this model/codebook combination.
