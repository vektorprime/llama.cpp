# Q8_0 BF16 Outlier — Implementation Summary

## Goal

Implement a llama.cpp quantization mode where a tensor is mostly stored as `Q8_0`, but selected 32-weight blocks are excluded from Q8_0 quantization and preserved as BF16 when an outlier value would distort the block scale.

**Scope**: CUDA-only runtime. Other backends (CPU, Metal, Vulkan) are not yet wired.

## Architecture (Current: GPU-Only Runtime Correction)

```
base tensor:  normal Q8_0, protected blocks zeroed
sidecar idx:  [2, n_blocks] I32  — (row, block_col) per protected block
sidecar bf16: [32, n_blocks] BF16 — original values for protected blocks

Runtime:
  Y = X @ W_q8_zeroed          (standard Q8_0 matmul, zeroed blocks → 0)
  Y += X_sparse @ W_bf16       (sparse BF16 correction via MUL_MAT_OUTLIER_BLOCKS)

Loading:
  Sidecar tensors created in GPU contexts (same as parent weight).
  load_all_data() uploads directly from GGUF to GPU — no CPU→GPU copies.
  ggml_cpy NOT used in graph builder — sidecars already on GPU.
```

### Why Not Pre-Patch?

The pre-patch approach (re-quantizing BF16→Q8_0 during loading) was abandoned because:
- Re-quantizing BF16→Q8_0 produces **identical** Q8_0 blocks to the original quantization
- The outlier protection provides **zero benefit** — same perplexity as plain Q8_0
- The BF16 values MUST be kept as BF16 and applied as a separate sparse matmul

## Files Changed

### Build System

| File | Change |
|------|--------|
| `src/CMakeLists.txt:51` | Added `../ggml/src` to `llama` target PRIVATE include dirs |

### GGML Level (new op)

| File | Change |
|------|--------|
| `ggml/include/ggml.h:586` | Added `GGML_OP_MUL_MAT_OUTLIER_BLOCKS` enum value |
| `ggml/include/ggml.h:1434` | Added `ggml_mul_mat_outlier_blocks()` function declaration |
| `ggml/src/ggml.c:1081` | Added `"MUL_MAT_OUTLIER_BLOCKS"` to `GGML_OP_NAME[]` |
| `ggml/src/ggml.c:3279` | Added `ggml_mul_mat_outlier_blocks()` — creates tensor, stores `n_rows_out`/`n_cols` in op_params |
| `ggml/src/ggml.c:3306` | **FIX**: `x->ne[0] <= n_cols` (was `==`) — shard-local column count may be smaller with tensor parallelism |
| `ggml/src/ggml.c:6875` | Added backward pass no-op case |
| `ggml/include/ggml-rpc.h:14` | Updated `static_assert(GGML_OP_COUNT)` |

### CPU Backend (required for meta backend — MIRRORED ops run on all backends)

| File | Change |
|------|--------|
| `ggml/src/ggml-cpu/ggml-cpu.c` | Added `GGML_OP_MUL_MAT_OUTLIER_BLOCKS` to `ggml_get_n_tasks`, `ggml_compute_forward` dispatch, and work buffer estimation |
| `ggml/src/ggml-cpu/ops.h` | Added `ggml_compute_forward_mul_mat_outlier_blocks()` declaration |
| `ggml/src/ggml-cpu/ops.cpp` | Added CPU kernel implementation (zeros output, computes sparse correction) |

### CUDA Backend

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/outlier.cuh` | **NEW** — Kernel header |
| `ggml/src/ggml-cuda/outlier.cu` | **NEW** — CUDA kernel with shard-aware column offset |
| `ggml/src/ggml-cuda/ggml-cuda.cu:66` | Added `#include "ggml-cuda/outlier.cuh"` |
| `ggml/src/ggml-cuda/ggml-cuda.cu:2983` | Added compute dispatch case |
| `ggml/src/ggml-cuda/ggml-cuda.cu:5429` | Added to `supports_op` |
| `ggml/src/ggml-cuda/ggml-cuda.cu:5448` | Added to `get_op_batch_size` |

### Meta Backend

| File | Change |
|------|--------|
| `ggml/src/ggml-backend-meta.cpp:883` | **FIX**: Split state mirrors `handle_mul_mat` logic — uses src0 (idx) as "weight" and src2 (activation) as "activation". Converts axis splits to PARTIAL to prevent downstream `handle_per_row` assertion failures. |
| `ggml/src/ggml-backend-meta.cpp:546` | Modified `handle_bin_bcast` for MIRRORED + non-MIRRORED source combination |

### Model/Loader Level

| File | Change |
|------|--------|
| `src/llama-model.h:579` | Added `llama_outlier_block_info` struct with CSR layout, `outlier_info` map, `build_outlier_info()`, `has_outlier_blocks()`, `get_outlier_info()` |
| `src/llama-model.cpp:979` | Implemented `build_outlier_info()` — scans `tensors_by_name` for sidecars, validates shapes, builds CSR layout. **FIX**: Uses `ggml_backend_tensor_get()` for GPU-side idx tensors instead of direct `data` deref. |
| `src/llama-model.cpp:1579` | **FIX**: Creates sidecar tensors in each GPU context containing the parent weight (not forced to CPU). Adjusts `n_tensors` for duplicates so `done_getting_tensors()` passes. |
| `src/llama.cpp:334` | Calls `build_outlier_info()` after `load_tensors()`. **Removed** explicit CPU→GPU `ggml_backend_tensor_set` copy loop. |

### Graph Builder Level

| File | Change |
|------|--------|
| `src/llama-graph.h:587` | Added `const llama_model * model = nullptr;` to `llm_graph_params` |
| `src/llama-graph.h:777` | Added `const llama_model & model;` to `llm_graph_context` |
| `src/llama-graph.cpp:1065` | `build_lora_mm()` — uses `ob->idx`/`ob->values` directly (already on GPU). No `ggml_cpy`. |
| `src/llama-model.cpp:2248` | Sets `params.model = this` in `build_graph()` |

### Quantizer Level

| File | Change |
|------|--------|
| `src/llama-quant.cpp:840` | `q8_outlier_block_is_candidate()` — outlier detection algorithm |
| `src/llama-quant.cpp:906` | `q8_outlier_analyze_tensor()` — per-tensor analysis with logging |
| `src/llama-quant.cpp:985` | `q8_outlier_zero_protected_blocks()` — zeros protected blocks in base Q8_0 tensor |
| `src/llama-quant.cpp:1707` | Sidecar data write loop — writes `outlier_idx` and `outlier_bf16` to GGUF |

## Key Architectural Decisions

### 1. GPU-Only Sidecar Loading (No CPU→GPU Copies)

Sidecar tensors are created in the same ggml_context as their parent weight tensor. During `load_all_data()`, the GGUF data is uploaded directly to GPU via `ggml_backend_tensor_set()`. No explicit CPU→GPU copy loop in `llama.cpp`. No `ggml_cpy` in the graph builder.

**Rationale**: The user rejected CPU→GPU copies as "too slow." The meta backend's `ggml_cpy` approach also had issues with tensor parallelism.

### 2. Shard-Aware Kernel with `col_offset`

The CUDA kernel computes `col_offset` from `ctx.device * n_cols_x` when the activation is split across GPUs but not a view. This correctly maps global column indices (from idx data) to shard-local positions.

**Without this fix**: GPU1 would skip blocks in its column range because `col_global < n_cols_x` check assumed shard starts at column 0.

### 3. Split State Mirrors `handle_mul_mat`

The meta backend split state for `MUL_MAT_OUTLIER_BLOCKS` mirrors `handle_mul_mat` logic:
- Both MIRRORED → MIRRORED
- Sidecar axis 1 + activation MIRRORED → axis 0
- Activation axis 1 + sidecar MIRRORED → follows activation
- Both axis 0 → PARTIAL
- Fallback → MIRRORED

This ensures the correction output has the same split state as the Q8_0 matmul result, so `ggml_add` combines them cleanly.

### 4. Sidecar Tensors in All GPU Contexts (Tensor Parallelism)

With tensor parallelism (`-sm tensor`), weights are split across GPUs. Sidecar tensors are created in EACH GPU context containing the parent weight. Each GPU loads its own copy from GGUF. The `n_tensors` count is adjusted for duplicates.

## Bugs Found and Fixed

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Sidecar tensor data never written to GGUF | `data is not within the file bounds` | Added write loop in `llama-quant.cpp:1707` |
| 2 | Protected blocks never zeroed in base Q8_0 | Double-counting (base + correction) | Added `q8_outlier_zero_protected_blocks()` |
| 3 | `model` field at position 1 broke brace-init | Compile error | Moved `model` to end of `llm_graph_params` |
| 4 | `n_created`/`n_tensors` mismatch | `too many tensors created` | Create sidecar tensors with `n_created++` |
| 5 | BF16 values converted with `__half2float` (F16) | Silent wrong results | Changed to `nv_bfloat16*` + `__bfloat162float()` |
| 6 | No split state handler for new op | `GGML_BACKEND_SPLIT_AXIS_UNKNOWN` crash | Added split state handler |
| 7 | `ggml_add` can't combine MIRRORED + non-MIRRORED | Same UNKNOWN crash | Modified `handle_bin_bcast` |
| 8 | CPU backend `ggml_get_n_tasks` crash | `op not implemented` | Added CPU backend op handlers |
| 9 | CUDA kernel never dispatched | Sidecar tensors on CPU, GPU can't access | Create sidecars in GPU contexts |
| 10 | NaN activations in CPU compute | CPU reading GPU memory | Sidecars on GPU; CPU kernel zeros output |
| 11 | Segfault accessing GPU weight tensor from CPU | Direct `weight->data` deref on device tensor | Use `ggml_backend_tensor_get/set` |
| 12 | `ggml-impl.h` not found | Missing include path | Added `../ggml/src` to include dirs |
| 13 | Shard-split assertion `x->ne[0] == n_cols` | Crash with tensor parallelism | Changed to `<=` |
| 14 | `handle_per_row` assertion `axis != 0` | Crash from explicit axis split propagation | Convert axis splits to PARTIAL in split state handler |
| 15 | `handle_mul_mat` GGML_ABORT at line 600 | PARTIAL input to downstream matmul | Mirror `handle_mul_mat` logic exactly |
| 16 | `col_offset` always 0 for split activations | GPU1 skips blocks in its column range | Compute `col_offset = ctx.device * n_cols_x` |
| 17 | Debug segfault reading `values_tensor->data[0]` | GPU pointer deref from CPU | Guard with `ggml_backend_buffer_is_host()` check |
| 18 | Sidecar tensors only on one GPU with tensor split | Other GPUs can't access sidecar data | Create sidecars in all GPU contexts; adjust `n_tensors` |

## Open Issue: 53% Top-p Agreement (vs 98% for plain Q8_0)

With aggressive outlier parameters (`--outlier-ratio 2 --outlier-max-frac 0.9`), the model achieves only 53% same top-p vs BF16 baseline, while plain Q8_0 achieves 98%.

**Hypothesis**: The outlier detection may not be finding enough candidates despite aggressive parameters, OR the correction kernel is producing incorrect values.

**Debug logging added** to `q8_outlier_analyze_tensor` and `q8_outlier_block_is_candidate` to trace candidate discovery.

## Usage

### Quantization

```bash
./llama-quantize --outlier-blocks bf16 \
  --outlier-ratio 2 \
  --outlier-nonmax-rel-rmse 0.0001 \
  --outlier-max-frac 0.9 \
  model-bf16.gguf model-q8-outlier.gguf Q8_0_BF16_OUTLIER
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--outlier-blocks bf16` | off | Enable sparse BF16 outlier sidecars |
| `--outlier-ratio N` | 16 | Protect block when `max_abs / second_abs ≥ N` |
| `--outlier-nonmax-rel-rmse X` | 0.01 | Minimum `rel_l2` on 31 non-outlier values |
| `--outlier-max-frac F` | 0.02 | Maximum protected block fraction per tensor |
| `--outlier-report PATH` | — | Write JSON report |

### Inference

```bash
./llama-cli -m model-q8-outlier.gguf -ngl 99 -p "Hello"
./llama-server -m model-q8-outlier.gguf -ngl 99
```

Works with tensor parallelism (`-sm tensor`). GPU-only — no CPU→GPU copies.

## Detection Algorithm

For each 32-weight block:
1. Find `max_abs` and `second_abs`
2. If `max_abs / second_abs < ratio_threshold` → skip
3. Compute Q8_0 quantization with `d = max_abs / 127`
4. Compute `rel_l2` on the 31 non-outlier values
5. If `rel_l2 < rel_rmse_threshold` → skip
6. Protect block: store original BF16 values, zero the Q8_0 block

## Limitations

- **2D tensors only** — 3D+ tensors (experts) skipped
- **No split GGUF support** — `--keep-split` throws error
- **Full storage only** — delta mode not implemented
- **`int32` op_params** — `n_rows_out`/`n_cols` limited to INT32_MAX
- **CPU backend required** — MIRRORED ops run on all backends; CPU kernel zeros output
- **Accuracy issue** — 53% top-p vs 98% for plain Q8_0 (under investigation)