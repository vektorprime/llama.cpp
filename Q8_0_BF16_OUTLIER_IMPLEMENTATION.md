# Q8_0 BF16 Outlier — Implementation Summary

## Goal

Implement a llama.cpp quantization mode where a tensor is mostly stored as `Q8_0`, but selected 32-weight blocks are excluded from Q8_0 quantization and preserved as BF16 when an outlier value would distort the block scale.

**Scope**: CUDA-only runtime. Other backends (CPU, Metal, Vulkan) are not yet wired.

## Architecture

```
base tensor:  normal Q8_0, protected blocks zeroed
sidecar idx:  [2, n_blocks] I32  — (row, block_col) per protected block
sidecar bf16: [32, n_blocks] BF16 — original values for protected blocks

Runtime:
  output = ggml_mul_mat(w_q8_zeroed, x)           // normal Q8_0 matmul
  output += ggml_mul_mat_outlier_blocks(idx, bf16, x)  // sparse BF16 correction
```

## Files Changed

### GGML Level (new op infrastructure)

| File | Change |
|------|--------|
| `ggml/include/ggml.h:586` | Added `GGML_OP_MUL_MAT_OUTLIER_BLOCKS` enum value (97th op, before `GGML_OP_COUNT`) |
| `ggml/include/ggml.h:1434` | Added `ggml_mul_mat_outlier_blocks()` function declaration |
| `ggml/src/ggml.c:1081` | Added `"MUL_MAT_OUTLIER_BLOCKS"` to `GGML_OP_NAME[]` |
| `ggml/src/ggml.c:1083` | Updated `static_assert(GGML_OP_COUNT == 97)` |
| `ggml/src/ggml.c:1191` | Added `"mul_mat_outlier_blocks(idx,values,x)"` to `GGML_OP_SYMBOL[]` |
| `ggml/src/ggml.c:1193` | Updated second `static_assert(GGML_OP_COUNT == 97)` |
| `ggml/src/ggml.c:3279` | Added `ggml_mul_mat_outlier_blocks()` tensor creation function — validates shapes, stores `n_rows_out`/`n_cols` in `op_params[0..1]` as i32 |
| `ggml/src/ggml.c:6875` | Added backward pass no-op case for the new op |
| `ggml/include/ggml-rpc.h:14` | Updated `static_assert(GGML_OP_COUNT == 97)` |

### Model/Loader Level

| File | Change |
|------|--------|
| `src/llama-model.h:579` | Added `llama_outlier_block_info` struct with CSR layout (`row_ptr`, `block_col`), `outlier_info` map, `build_outlier_info()`, `has_outlier_blocks()`, `get_outlier_info()` |
| `src/llama-model.cpp:979` | Implemented `build_outlier_info()` — scans `tensors_by_name` for `.outlier_idx`/`.outlier_bf16` sidecars, validates shapes, builds CSR `row_ptr`/`block_col` from idx data |
| `src/llama-model.cpp:1545` | Creates sidecar tensors in ggml CPU context before `done_getting_tensors()` — scans `weights_map` for sidecar tensor names, creates via `ggml_new_tensor_2d`, increments `n_created` |
| `src/llama.cpp:334` | Calls `model_ptr->build_outlier_info()` after `load_tensors()` |

### Graph Builder Level

| File | Change |
|------|--------|
| `src/llama-graph.h:587` | Added `const llama_model * model = nullptr;` at end of `llm_graph_params` (after `res`, so brace-init still works) |
| `src/llama-graph.h:777` | Added `const llama_model & model;` as first member of `llm_graph_context` |
| `src/llama-graph.h:19` | Added forward declaration `struct llama_model;` |
| `src/llama-graph.cpp:1002` | Updated constructor to initialize `model(*params.model)` |
| `src/llama-graph.cpp:1065` | Wired outlier correction into `build_lora_mm()` — checks `model.has_outlier_blocks(w)`, creates `ggml_mul_mat_outlier_blocks()` correction, adds via `ggml_add()` |
| `src/llama-model.cpp:2248` | Sets `params.model = this` in `build_graph()` before graph construction |

### Quantizer Level (Phase A — pre-existing, with fixes)

| File | Change |
|------|--------|
| `src/llama-quant.cpp:1683` | **FIX**: Added `q8_outlier_zero_protected_blocks()` call after quantization — zeros protected blocks in base Q8_0 tensor before writing |
| `src/llama-quant.cpp:1707` | **FIX**: Added sidecar data write loop after main quantization loop — writes `outlier_idx` (int32 pairs) and `outlier_bf16` (bf16 values) to GGUF file with alignment padding |

### CUDA Kernel Level

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/outlier.cuh` | **NEW** — Kernel header declaring `ggml_cuda_op_mul_mat_outlier_blocks()` |
| `ggml/src/ggml-cuda/outlier.cu` | **NEW** — CUDA kernel: one thread block per (outlier_block, token), warp-reduced dot product of 32 BF16 weights × 32 float activations, atomicAdd into output. Uses `nv_bfloat16*` + `__bfloat162float()` for correct BF16→float conversion |
| `ggml/src/ggml-cuda/ggml-cuda.cu:66` | Added `#include "ggml-cuda/outlier.cuh"` |
| `ggml/src/ggml-cuda/ggml-cuda.cu:2983` | Added compute dispatch case: `ggml_cuda_op_mul_mat_outlier_blocks(ctx, dst)` |
| `ggml/src/ggml-cuda/ggml-cuda.cu:5429` | Added `GGML_OP_MUL_MAT_OUTLIER_BLOCKS` to `supports_op` (returns true) |
| `ggml/src/ggml-cuda/ggml-cuda.cu:5448` | Added to `get_op_batch_size` (returns `op->ne[2]`) |


## Bugs Found and Fixed

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Sidecar tensor data never written to GGUF | `data is not within the file bounds` | Added write loop in `llama-quant.cpp:1707` |
| 2 | Protected blocks never zeroed in base Q8_0 | Double-counting (base + correction) | Added `q8_outlier_zero_protected_blocks()` call in `llama-quant.cpp:1683` |
| 3 | `model` field at position 1 broke brace-init | Compile error in `llama-context.cpp:2310` | Moved `model` to end of `llm_graph_params` |
| 4 | `n_created`/`n_tensors` mismatch | `too many tensors created; expected 866, got 1612` | Create sidecar tensors with `n_created++` so counts match |
| 5 | BF16 values converted with `__half2float` (F16) | Silent wrong results | Changed to `nv_bfloat16*` + `__bfloat162float()` |
| 6 | No split state handler for new op | `GGML_BACKEND_SPLIT_AXIS_UNKNOWN` crash during warmup | Added `GGML_OP_MUL_MAT_OUTLIER_BLOCKS` case returning `MIRRORED` |
| 7 | `ggml_add` can't combine MIRRORED + non-MIRRORED sources | Same UNKNOWN crash (persisted after fix 6) | Modified `handle_bin_bcast` to use non-MIRRORED axis when one source is MIRRORED |

## Usage

### Quantization

```bash
./llama-quantize --outlier-blocks bf16 \
  --outlier-ratio 100 \
  --outlier-nonmax-rel-rmse 0.0 \
  --outlier-max-frac 0.5 \
  model-bf16.gguf model-q8-outlier.gguf Q8_0_BF16_OUTLIER
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--outlier-blocks bf16` | off | Enable sparse BF16 outlier sidecars |
| `--outlier-ratio N` | 16 | Protect block when `max_abs / second_abs ≥ N` |
| `--outlier-nonmax-rel-rmse X` | 0.01 | Minimum `rel_l2` on 31 non-outlier values (set to 0.0 to disable) |
| `--outlier-max-frac F` | 0.02 | Maximum protected block fraction per tensor |
| `--outlier-report PATH` | — | Write JSON report |

### Inference (CUDA only)

```bash
./llama-cli -m model-q8-outlier.gguf -ngl 99 -p "Hello"
./llama-server -m model-q8-outlier.gguf -ngl 99
```

Must use `-ngl 99` to offload all layers to GPU. CPU-only layers will fail since only CUDA backend supports the correction op.

## Detection Algorithm

For each 32-weight block:
1. Find `max_abs` and `second_abs`
2. If `max_abs / second_abs < ratio_threshold` → skip
3. Compute Q8_0 quantization with `d = max_abs / 127`
4. Compute `rel_l2` on the 31 non-outlier values
5. If `rel_l2 < rel_rmse_threshold` → skip
6. Protect block: store original BF16 values, zero the Q8_0 block

## Limitations

- **CUDA only** — CPU/Metal/Vulkan backends not wired
- **2D tensors only** — 3D+ tensors (experts) skipped
- **No split GGUF support** — `--keep-split` throws error
- **Full storage only** — delta mode not implemented
- **`int32` op_params** — `n_rows_out`/`n_cols` limited to INT32_MAX