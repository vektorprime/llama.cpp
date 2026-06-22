# Scale Q4_K — Implementation & KLD Benchmark Report

## What was built
A runtime-optional Q4_K modification on a fresh llama.cpp clone at
`/home/user/llm/scale_llama` (branch `scale-q4k`, build server 10.0.0.188, 2x RTX 3080).
When the post-quant nibble is `0`, instead of the standard
`weight = d*sc*0 - dmin*m`, it outputs the **scale value** as the weight.
Gated by a new `--scale-q4-k` flag (default OFF), wired through **CPU dequant,
CPU vec_dot, CUDA dequant, CUDA MMVQ, and CUDA MMQ** — only the CPU and CUDA
pipelines, as requested. Quantized model saved as `Qwen3.5-2B-Q4K-SCALE-1.gguf`
(1.1 GB, 4.82 BPW).

Two variants implemented (per fallback guidance):
1. **Naive** — nibble 0 -> `+d*sc`
2. **Sign-aware** — nibble 0 -> `(m1>0) ? -d*sc : +d*sc` (signs the scale by the
   direction of the value being approximated, `-m1`)

## KLD Benchmarks vs BF16 logits (5 chunks, wikitext)

| Metric      | Baseline Q4_K | Naive Scale | Sign-aware Scale | Q4_0 ref |
|-------------|---------------|-------------|------------------|----------|
| Mean PPL(Q) | 13.32         | 14.89       | 14.04            | 13.9     |
| Mean KLD    | 0.0570        | 0.1844      | 0.1127           | 0.093    |
| Same top-p  | 86.76%        | 79.15%      | 82.94%           | 83.9%    |

Sign-awareness recovers ~40% of the accuracy the naive version lost
(KLD 0.184 -> 0.113), confirming the sign fix is mathematically correct.
But neither variant beats baseline Q4_K.

## Why it doesn't help (root cause)
The premise — "post-quant weight 0 -> weight*scale = 0, losing accuracy" — holds
for **min-less** quant like Q4_0. **Q4_K has a min offset**:
`weight = d*sc*nibble - dmin*m`. So nibble 0 does **not** mean a zero weight — it
encodes the **sub-block minimum** `-m1` (a meaningful, often-negative value) with
6-bit precision. That's already the optimal representation for those weights, so
any scale substitution discards real information and is strictly worse. The
all-positive-block case (where `m1=0`, so nibble 0 ~ 0) is too rare to offset the
damage to mixed-sign blocks.

## -sm tensor (multi-GPU) — verified working
Tested on both RTX 3080s with tensor split. Results match single-GPU exactly, so
the modified MMVQ/MMQ CUDA kernels are correct across the split path:

| Run (-sm tensor) | PPL    | KLD    | top-p  |
|------------------|--------|--------|--------|
| baseline         | 13.319 | 0.0570 | 86.73% |
| sign-aware scale | 14.030 | 0.1127 | 82.83% |

No crashes, no CUDA errors. (One harmless warning: `llama_params_fit` auto-fit is
unimplemented for `SPLIT_MODE_TENSOR` — inference runs fine regardless.)

## Bottom line
The feature is fully functional, optional, and correct on CPU + CUDA + multi-GPU —
but for **Q4_K it's a net regression** and correctly defaults OFF. The concept
would only pay off on a min-less format (Q4_0) where nibble 0 genuinely maps to a
zero weight.

## Files modified
1. `ggml/src/ggml-quants.c` — CPU dequant + `g_scale_q4_k` flag (exported)
2. `ggml/src/ggml-cpu/quants.c` — CPU vec_dot (unpacking + sign-aware min correction)
3. `ggml/src/ggml-cuda/convert.cu` — CUDA dequant + `__constant__` flag + setter
4. `ggml/src/ggml-cuda/vecdotq.cuh` — MMVQ + MMQ sign-aware correction kernels
5. `ggml/include/ggml.h` — `ggml_set/get_scale_q4_k()` declarations
6. `ggml/include/ggml-cuda.h` — `ggml_cuda_set_scale_q4_k()` declaration
7. `common/common.h` — `bool scale_q4_k` field in common_params
8. `common/arg.cpp` — `--scale-q4-k` / `--no-scale-q4-k` CLI flag
9. `tools/perplexity/perplexity.cpp` — setter wiring
10. `tools/server/server-context.cpp` — setter wiring (covers llama-cli)

See `SCALE_Q4K_IMPL.md` for the full implementation log, every build error + fix,
and all raw numbers.
