# Q4_K_INFO — Wiring Guide for Cloning Q4_K_M

This document describes every file, line, and pattern needed to clone the Q4_K_M
quantization type into a new `GGML_TYPE_Q4_K_M_CLONE`. It covers both layers:
the GGML-level type registration and the LLaMA-level ftype/per-tensor mixing.

## Two-Layer Architecture

In this codebase, K-quant types exist at two layers:

1. **GGML layer:** `GGML_TYPE_Q4_K` (value 12) — a single quantized type. There
   is no Q4_K_M vs Q4_K_S distinction at the GGML level.
2. **LLaMA layer:** `LLAMA_FTYPE_MOSTLY_Q4_K_M` (value 15) and
   `LLAMA_FTYPE_MOSTLY_Q4_K_S` (value 14) — "mostly quant" file types that
   describe a mixing strategy (which tensors get which GGML types). Both Q4_K_S
   and Q4_K_M ultimately use `GGML_TYPE_Q4_K` as their primary quantization
   type, with different per-tensor override rules.

To clone Q4_K_M, you need a new GGML type AND a new LLAMA ftype that mirrors the
Q4_K_M mixing strategy.

## Block Structure

`block_q4_K` is 144 bytes per 256 elements = 4.5 bpw:

```
ggml_half    d;                     // 2 bytes — super-block scale
ggml_half    dmin;                  // 2 bytes — super-block scale for mins
uint8_t      scales[K_SCALE_SIZE];  // 12 bytes — 6-bit quantized scales/mins
uint8_t      qs[QK_K/2];            // 128 bytes — 4-bit half-byte quants
```

Key constants (from `ggml/src/ggml-common.h`):
- `QK_K` = 256
- `K_SCALE_SIZE` = 12
- `QR4_K` = 2
- `QI4_K` = `QK_K / (4 * QR4_K)` = 32

## Files to Touch

### GGML Core (7 files)

| # | File | Change |
|---|------|--------|
| 1 | `ggml/include/ggml.h` (~L431) | Add `GGML_TYPE_Q4_K_M_CLONE = 42`, bump `GGML_TYPE_COUNT` to 43 |
| 2 | `ggml/src/ggml-common.h` (~L328) | Define `block_q4_K_M_CLONE` struct (identical to `block_q4_K`) |
| 3 | `ggml/src/ggml-quants.h` (~L30,63,97) | Declare: `quantize_row_q4_K_M_CLONE_ref`, `dequantize_row_q4_K_M_CLONE`, `quantize_q4_K_M_CLONE` |
| 4 | `ggml/src/ggml-quants.c` | Implement thin wrappers calling stock Q4_K internals; add validation switch case (~L6190) |
| 5 | `ggml/src/ggml.c` (~L765,7741) | Add `type_traits[]` entry + `ggml_quantize_chunk` dispatch case |
| 6 | `ggml/src/ggml-cpu/ggml-cpu.c` (~L305) | Add `cpu_type_traits[]` entry |
| 7 | `ggml/src/ggml-cpu/quants.c` + `quants.h` | CPU dispatcher wrapper `quantize_row_q4_K_M_CLONE` |

### CPU Ops Dispatch (2 files)

| # | File | Change |
|---|------|--------|
| 8 | `ggml/src/ggml-cpu/ops.cpp` (7 locations) | Add `case GGML_TYPE_Q4_K_M_CLONE:` fallthrough after each `case GGML_TYPE_Q4_K:` |
| 9 | `ggml/src/ggml-cpu/repack.cpp` (3 locations) | Add clone to type-check assertions and dispatch condition |

### CUDA Backend (5 files)

| # | File | Change |
|---|------|--------|
| 10 | `ggml/src/ggml-cuda/ggml-cuda.cu` (~L5183) | Add `case GGML_TYPE_Q4_K_M_CLONE:` fallthrough |
| 11 | `ggml/src/ggml-cuda/mmq.cu` (3 locations) | Add clone to switch chains and MMQ eligibility check |
| 12 | `ggml/src/ggml-cuda/mmvq.cu` (~13 locations) | Add `case GGML_TYPE_Q4_K_M_CLONE:` fallthrough (replaceAll) |
| 13 | `ggml/src/ggml-cuda/mmq.cuh` (3 locations) | Add clone to layout, DP4A, and MMA tile-size switches |
| 14 | `ggml/src/ggml-cuda/common.cuh` (~L1032) | Add `ggml_cuda_type_traits<GGML_TYPE_Q4_K_M_CLONE>` template specialization |
| 15 | `ggml/src/ggml-cuda/convert.cu` (2 locations) | Add clone fallthrough to dequant function pointer returns |

### LLaMA Layer (4 files)

| # | File | Change |
|---|------|--------|
| 16 | `include/llama.h` (~L157) | Add `LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE = 41` |
| 17 | `src/llama-quant.cpp` | Add ftype→ggml_type mapping; add `\|\| LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE` to every `LLAMA_FTYPE_MOSTLY_Q4_K_M` condition (5 locations) |
| 18 | `src/llama-model-loader.cpp` (~L53,752) | Ftype name string + reverse ggml_type→ftype mapping |
| 19 | `tools/quantize/quantize.cpp` (~L71) | CLI string-to-ftype mapping: `"Q4_K_M_CLONE"` |

### Other Backends (not needed for CPU+CUDA)

The clone can be omitted from Vulkan, Metal, SYCL, OpenCL, WebGPU, and OpenVINO
backends if those are not enabled at build time. If they are enabled, add
`case GGML_TYPE_Q4_K_M_CLONE:` fallthrough or assert-not-supported in each
backend's dispatch chain.

## Implementation Pattern

### Thin wrapper functions (ggml-quants.c)

Since `block_q4_K_M_CLONE` has identical memory layout to `block_q4_K`, the
clone functions simply cast and call through:

```c
void quantize_row_q4_K_M_CLONE_ref(const float * GGML_RESTRICT x, block_q4_K_M_CLONE * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_K_ref(x, (block_q4_K *)y, k);
}

void dequantize_row_q4_K_M_CLONE(const block_q4_K_M_CLONE * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    dequantize_row_q4_K((const block_q4_K *)x, y, k);
}

size_t quantize_q4_K_M_CLONE(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    return quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}
```

### Per-tensor mixing (llama-quant.cpp)

The clone ftype must produce identical tensor type assignments as stock Q4_K_M.
Add `|| ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE` to every condition that
checks `ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M`. The conditions are in these
tensor category sections:

1. `ATTENTION_WV` (~L543): Q4_K_M || Q5_K_M → Q6_K for attention heads
2. `FFN_DOWN` (~L599): Q4_K_M → Q6_K for FFN down projections
3. `ATTENTION_OUTPUT` (~L627): Q4_K_M in expert-model condition list
4. `ATTENTION_QKV` (~L646): Q4_K_M → Q5_K for QKV tensors
5. `ftype_to_ggml_type_default` (~L828): Q4_K_M_CLONE → GGML_TYPE_Q4_K_M_CLONE

### Type traits (ggml.c, ggml-cpu.c, common.cuh)

All three backends need type traits entries pointing to the clone functions:

```
.type_name     = "q4_K_M_CLONE"
.blck_size     = QK_K
.type_size     = sizeof(block_q4_K_M_CLONE)
.is_quantized  = true
.to_float      = dequantize_row_q4_K_M_CLONE
.from_float_ref = quantize_row_q4_K_M_CLONE_ref
.from_float    = quantize_row_q4_K_M_CLONE  (CPU dispatcher)
.vec_dot       = ggml_vec_dot_q4_K_q8_K     (same as Q4_K)
.vec_dot_type  = GGML_TYPE_Q8_K
```

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF
timeout 1200 cmake --build build -j16
```

## Quantize and Verify

```bash
rm -f /tmp/qwen3.5-0.8b-q4km-clone-exp.gguf
timeout 1200 ./build/bin/llama-quantize \
  /home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  /tmp/qwen3.5-0.8b-q4km-clone-exp.gguf Q4_K_M_CLONE

CUDA_VISIBLE_DEVICES=3 timeout 1200 build/bin/llama-perplexity \
  -m /tmp/qwen3.5-0.8b-q4km-clone-exp.gguf \
  -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
  -t 8 -c 256 --chunks 250 -fa on \
  --cache-type-k bf16 --cache-type-v bf16 \
  --no-mmap -ngl 999 -np 1 \
  --kl-divergence --kl-divergence-base /home/user/llm/models/Qwen3.5-0.8B-BF16.logits
```

Compare with stock Q4_K_M quantized by the same build — GGUF size and all eval
metrics must match exactly.

## Current Baselines (Qwen3.5-0.8B, this build)

| Quant | GGUF Size | PPL | KLD | Same top p | RMS Δp |
|-------|-----------|-----|-----|------------|--------|
| BF16 | ~1.41 GB | 21.5386 | 0.0 | 100% | 0.0% |
| Q4_K_M / Q4_K_M_CLONE | ~505 MB | 22.4499 | 0.062947 | 86.387% | 5.753% |
