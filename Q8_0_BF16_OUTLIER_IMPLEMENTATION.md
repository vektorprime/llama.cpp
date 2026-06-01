# Q8_0 BF16 Outlier — Implementation Summary

## Goal

Implement a llama.cpp quantization mode where a tensor is mostly stored as `Q8_0`, but selected 32-weight blocks are excluded from Q8_0 quantization and preserved as BF16 when an outlier value would distort the block scale.

**Scope**: CUDA-only runtime. Other backends (CPU, Metal, Vulkan) are not yet wired.

## Architecture

```
base tensor:  normal Q8_0, protected blocks zeroed
sidecar idx:  [2, n_blocks] I32  — (row, block_col) per protected block
sidecar bf16: [32, n_blocks] BF16 — original values for protected blocks

Loading (final approach):
  During model load, BF16 sidecar values are re-quantized to Q8_0
  and patched directly into the zeroed blocks of the base Q8_0 tensor.
  No runtime correction op needed — standard Q8_0 matmul just works.
```

## Final Approach: Pre-Patch Weights During Loading

The original design used a runtime `MUL_MAT_OUTLIER_BLOCKS` op to apply the BF16 correction at inference time. This proved problematic because:

1. Sidecar tensors (idx, bf16) live on CPU but the correction must be applied on GPU
2. The meta backend's MIRRORED split state caused the op to run on all backends, but GPU couldn't access CPU sidecar data
3. Cross-backend data transfer for dynamically-computed tensors is not automatic

**Solution**: During model loading, after `build_outlier_info()`, iterate over all protected blocks and re-quantize the BF16 values to Q8_0 format, writing them directly into the zeroed blocks of the base Q8_0 weight tensor. The standard Q8_0 matmul then uses the patched weights with no runtime overhead.

### Weight Patching (`src/llama.cpp`)

For each weight tensor with outlier blocks:
1. Read 32 BF16 values from the sidecar tensor
2. Compute `amax = max(|vals|)`, scale `d = amax / 127`
3. Quantize: `qs[j] = round(vals[j] / d)`
4. Write `d` (FP16) and `qs[32]` (int8) into the Q8_0 block

For device tensors (GPU), uses `ggml_backend_tensor_get/set` to transfer data between host and device.

## Files Changed

### Build System

| File | Change |
|------|--------|
| `src/CMakeLists.txt:51` | Added `../ggml/src` to `llama` target PRIVATE include dirs (for `ggml-impl.h`, `ggml-common.h`) |

### GGML Level (new op infrastructure — kept for safety, not used at runtime)

| File | Change |
|------|--------|
| `ggml/include/ggml.h:586` | Added `GGML_OP_MUL_MAT_OUTLIER_BLOCKS` enum value (97th op, before `GGML_OP_COUNT`) |
| `ggml/include/ggml.h:1434` | Added `ggml_mul_mat_outlier_blocks()` function declaration |
| `ggml/src/ggml.c:1081` | Added `"MUL_MAT_OUTLIER_BLOCKS"` to `GGML_OP_NAME[]` |
| `ggml/src/ggml.c:1083` | Updated `static_assert(GGML_OP_COUNT == 97)` |
| `ggml/src/ggml.c:1191` | Added `"mul_mat_outlier_blocks(idx,values,x)"` to `GGML_OP_SYMBOL[]` |
| `ggml/src/ggml.c:1193` | Updated second `static_assert(GGML_OP_COUNT == 97)` |
| `ggml/src/ggml.c:3279` | Added `ggml_mul_mat_outlier_blocks()` tensor creation function |
| `ggml/src/ggml.c:6875` | Added backward pass no-op case for the new op |
| `ggml/include/ggml-rpc.h:14` | Updated `static_assert(GGML_OP_COUNT == 97)` |

### CPU Backend (op handlers — kept for safety)

| File | Change |
|------|--------|
| `ggml/src/ggml-cpu/ggml-cpu.c` | Added `GGML_OP_MUL_MAT_OUTLIER_BLOCKS` to `ggml_get_n_tasks`, `ggml_compute_forward` dispatch, and work buffer estimation |
| `ggml/src/ggml-cpu/ops.h` | Added `ggml_compute_forward_mul_mat_outlier_blocks()` declaration |
| `ggml/src/ggml-cpu/ops.cpp` | Added CPU kernel implementation |

### CUDA Backend (kernel kept for safety)

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/outlier.cuh` | **NEW** — Kernel header |
| `ggml/src/ggml-cuda/outlier.cu` | **NEW** — CUDA kernel |
| `ggml/src/ggml-cuda/ggml-cuda.cu:66` | Added `#include "ggml-cuda/outlier.cuh"` |
| `ggml/src/ggml-cuda/ggml-cuda.cu:2983` | Added compute dispatch case |
| `ggml/src/ggml-cuda/ggml-cuda.cu:5429` | Added to `supports_op` |
| `ggml/src/ggml-cuda/ggml-cuda.cu:5448` | Added to `get_op_batch_size` |

### Meta Backend

| File | Change |
|------|--------|
| `ggml/src/ggml-backend-meta.cpp:876` | Added `GGML_OP_MUL_MAT_OUTLIER_BLOCKS` split state handler (MIRRORED) |
| `ggml/src/ggml-backend-meta.cpp:546` | Modified `handle_bin_bcast` for MIRRORED + non-MIRRORED source combination |

### Model/Loader Level

| File | Change |
|------|--------|
| `src/llama-model.h:579` | Added `llama_outlier_block_info` struct with CSR layout, `outlier_info` map, `build_outlier_info()`, `has_outlier_blocks()`, `get_outlier_info()` |
| `src/llama-model.cpp:979` | Implemented `build_outlier_info()` — scans `tensors_by_name` for sidecars, validates shapes, builds CSR layout |
| `src/llama-model.cpp:1545` | Creates sidecar tensors in CPU ggml context before `done_getting_tensors()` |
| `src/llama.cpp:334` | Calls `build_outlier_info()` after `load_tensors()` |
| `src/llama.cpp:339` | **Weight patching**: BF16→Q8_0 conversion for protected blocks, uses `ggml_backend_tensor_get/set` for device tensors |

### Graph Builder Level

| File | Change |
|------|--------|
| `src/llama-graph.h:587` | Added `const llama_model * model = nullptr;` to `llm_graph_params` |
| `src/llama-graph.h:777` | Added `const llama_model & model;` to `llm_graph_context` |
| `src/llama-graph.h:19` | Added forward declaration `struct llama_model;` |
| `src/llama-graph.cpp:1002` | Updated constructor to initialize `model(*params.model)` |
| `src/llama-graph.cpp:1065` | `build_lora_mm()` — outlier correction is pre-patched, no runtime op needed |
| `src/llama-model.cpp:2248` | Sets `params.model = this` in `build_graph()` |

### Quantizer Level

| File | Change |
|------|--------|
| `src/llama-quant.cpp:1683` | Added `q8_outlier_zero_protected_blocks()` — zeros protected blocks in base Q8_0 tensor |
| `src/llama-quant.cpp:1707` | Added sidecar data write loop — writes `outlier_idx` and `outlier_bf16` to GGUF |

## Bugs Found and Fixed

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Sidecar tensor data never written to GGUF | `data is not within the file bounds` | Added write loop in `llama-quant.cpp:1707` |
| 2 | Protected blocks never zeroed in base Q8_0 | Double-counting (base + correction) | Added `q8_outlier_zero_protected_blocks()` in `llama-quant.cpp:1683` |
| 3 | `model` field at position 1 broke brace-init | Compile error in `llama-context.cpp:2310` | Moved `model` to end of `llm_graph_params` |
| 4 | `n_created`/`n_tensors` mismatch | `too many tensors created` | Create sidecar tensors with `n_created++` |
| 5 | BF16 values converted with `__half2float` (F16) | Silent wrong results | Changed to `nv_bfloat16*` + `__bfloat162float()` |
| 6 | No split state handler for new op | `GGML_BACKEND_SPLIT_AXIS_UNKNOWN` crash | Added MIRRORED split state |
| 7 | `ggml_add` can't combine MIRRORED + non-MIRRORED | Same UNKNOWN crash | Modified `handle_bin_bcast` |
| 8 | CPU backend `ggml_get_n_tasks` crash | `op not implemented: MUL_MAT_OUTLIER_BLOCKS` | Added CPU backend op handlers |
| 9 | CUDA kernel never dispatched | Sidecar tensors on CPU, GPU can't access | Changed to pre-patch approach (no runtime op) |
| 10 | NaN activations in CPU compute | CPU reading uninitialized GPU memory | Removed runtime op from graph |
| 11 | Segfault accessing GPU weight tensor from CPU | Direct `weight->data` deref on device tensor | Use `ggml_backend_tensor_get/set` for device tensors |
| 12 | `ggml-impl.h` not found | Missing include path | Added `../ggml/src` to `llama` target include dirs |

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

### Inference

```bash
./llama-cli -m model-q8-outlier.gguf -ngl 99 -p "Hello"
./llama-server -m model-q8-outlier.gguf -ngl 99
```

Works with any `-ngl` setting. No runtime correction op needed — weights are pre-patched during loading.

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
- **Pre-patch approach** — BF16 values are re-quantized to Q8_0 during loading (minor precision loss vs true BF16 matmul, but avoids cross-backend data transfer issues)