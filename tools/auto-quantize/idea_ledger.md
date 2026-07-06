# IQ4_XS Auto-Research Idea Ledger

## Experiment Index

| Exp | Description | Outcome |
|-----|-------------|---------|
| 001 | Weight exponent 0.30 + per-sub-block sigma2 | Failed — quantize timed out (>7 min) |
| 002 | Weight exponent 0.30 + per-sub-block sigma2 (FAST pow: exp2f) | Failed — quantize timed out (>7 min) |
| 003 | Weight exponent 0.30 + per-sub-block sigma2 (20 min timeout) | Regression — KL 0.025029 vs 0.024916 baseline (slightly worse) |

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
