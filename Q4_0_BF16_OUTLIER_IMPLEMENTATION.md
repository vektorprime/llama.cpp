# Q4_0_BF16_OUTLIER Implementation

## Overview

Q4_0_BF16_OUTLIER extends the Q8_0_BF16_OUTLIER pattern to Q4_0 base quantization: base Q4_0 matmul + sparse residual correction for high-error blocks. The residual delta values can be stored as BF16, Q8_0, or Q4_0 (selectable via `--outlier-value-type`).

**Architecture:** Separate `q4_outlier_*` functions mirror `q8_outlier_*` — no parameterization of existing Q8 code.

## File Type

- **Enum:** `LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER = 42`
- **Default base type:** `GGML_TYPE_Q4_0`
- **CLI string:** `Q4_0_BF16_OUTLIER`

## Q4_0 Block Layout

| Field | Type | Offset | Size |
|-------|------|--------|------|
| `d` (scale) | `ggml_fp16_t` | 0 | 2 bytes |
| `qs` (nibbles) | `uint8_t[16]` | 2 | 16 bytes |
| **Total** | | | **18 bytes** |

Each byte holds two 4-bit values (lower nibble: index j, upper nibble: index j+1). Dequantized value range: -8..7 (biased from stored 0..15).

### Quantization Formula (reference `quantize_row_q4_0_ref`)

```
m = W[j*] where j* = argmax_j |W[j]|     // signed value with largest abs
d = m / -8                                // scale (can be negative)
q[j] = clamp(0, 15)(round(W[j] / d) + 8) // quantize, bias to [0,15]
W_hat[j] = d * (q[j] - 8)                // dequantize
```

**Key difference from Q8_0:** Q4_0 scale is `max / -8` (sign-aware), not `max_abs / 127` (always positive). The representable range is asymmetric: `[-8d, 7d]`, where `|m| = -8d` is exactly representable.

## Selection Scoring

Q4_0 uses **residual-energy scoring** (not the Q8 ratio×RMSE heuristic):

```
d = m / -8  where m = W[j*], j* = argmax|W[j]|

sqerr = 0, energy = 0
for j = 0..31:                          // All 32 elements
    q = clamp(-8, 7)(round(W[j] / d))
    sqerr  += α[j] · (W[j] - q·d)²
    energy += α[j] · W[j]²

score = sqerr / (energy + ε)            // Normalized residual energy
```

Blocks ranked by score, top `max_frac` fraction selected. Minimum score floor: 1e-6 (skip blocks where Q4_0 already quantizes well).

**Why not ratio×RMSE?** Q4_0 rounding error is inherent to 4-bit representation — not just an "outlier dominates scale" problem. The ratio filter rejects blocks where Q4 error is large but no single value dominates (e.g., uniform-distribution blocks).

## Delta Computation

```
Δ[r,c,j] = W_fp32[r,c,j] - W_q4_dequant[r,c,j]
V[k,j] = bf16(Δ[r_k, c_k, j])          // BF16 delta for selected block k
I[k] = (r_k, c_k)                       // I32 index: (row, block_col)
```

Dequantization in `q4_outlier_compute_deltas()`:
```c
d = fp16_to_fp32(block_ptr[0..1])
q = (qs[j/2] & 0x0F) - 8    // lower nibble, j even
q = (qs[j/2] >> 4) - 8      // upper nibble, j odd
q4_val = q * d
delta = orig[j] - q4_val
```

## Inference Path

Identical to Q8_0 — generic kernels handle any base type:

```
Y_base[r,t] = ggml_mul_mat(W_q4, X)                    // Standard Q4_0 matmul
Y_corr[r,t] = ggml_mul_mat_outlier_blocks(I, V, X)      // Sparse BF16 correction
Y[r,t] = Y_base + Y_corr                                   // ggml_add
```

**Kernels:** `ggml_compute_forward_mul_mat_outlier_blocks()` (CPU) and `outlier_blocks_kernel()` (CUDA) handle BF16, Q8_0, and Q4_0 delta values. The kernel dispatch selects the correct dequant function based on `values->type`.

**Graph builder unchanged:** `build_lora_mm()` wraps `ggml_mul_mat` with correction + `ggml_add` for any tensor with outlier info.

## Token Embedding Correction

Token embeddings (`token_embd.weight`) use `ggml_get_rows` for lookup, which has no outlier correction op. Instead, `patch_embedding_outliers()` pre-patches the embedding tensor at load time:

1. **Dequantize** each row from Q4_0 → FP32
2. **Add** outlier block delta values to the corresponding 32-element slots
3. **Re-quantize** the patched row back to Q4_0
4. **Write** the patched row back to GPU/CPU memory — the embedding tensor is permanently corrected

This works for all delta types (BF16, Q8_0, Q4_0). In streaming mode, delta values are read from the `outlier_cache` CPU-side copy rather than the (null) GPU tensor pointers.

| Delta type | values_element_size | Dequant function |
|-----------|---------------------|------------------|
| BF16 | 64 (32×bf16) | `ggml_bf16_to_fp32` |
| Q8_0 | 34 (sizeof block_q8_0) | `dequantize_row_q8_0` |
| Q4_0 | 18 (sizeof block_q4_0) | `dequantize_row_q4_0` |

## Storage Cost

| Component | Size per block |
|-----------|---------------|
| Q4_0 base | 4.5 bpw (18 bytes / 32 values) |
| I32 index sidecar | 2 bpw (8 bytes / 32 values) |
| BF16 delta sidecar | 18 bpw (64 bytes / 32 values) |
| Q8_0 delta sidecar | 8.5 bpw (34 bytes / 32 values) |
| **Q4_0 delta sidecar** | **4.5 bpw (18 bytes / 32 values)** |

Effective bpw with fraction `f` protected and delta type:
```
bpw ≈ 4.5 + (delta_bpw × f)
```

| f | BF16 (18 bpw) | Q8_0 (8.5 bpw) | **Q4_0 (4.5 bpw)** |
|---|--------------|----------------|---------------------|
| 1% | 4.68 | 4.59 | **4.55** |
| 5% | 5.40 | 4.93 | **4.73** |
| 10% | 6.30 | 5.35 | **4.95** |

### Measured (Qwen3.5-2B, f≈67% via max-abs-error=0.002)

| Delta type | Sidecar size | Total model | Same top P |
|-----------|-------------|-------------|------------|
| BF16 | ~1500 MB | ~2.6 GB | ~89-91% |
| Q8_0 | ~800 MB (est.) | ~1.9 GB (est.) | TBD |
| **Q4_0** | **572 MB** | **1.6 GB** | **~84%** |

**Recommendation:** BF16 for maximum quality, Q4_0 for maximum compression. Q8_0 offers a middle ground (not yet benchmarked).

## Metadata

**GGUF keys** (separate namespace from Q8):
- `llama.q4_outlier.version` = 1
- `llama.q4_outlier.block_size` = 32
- `llama.q4_outlier.base_type` = "q4_0"
- `llama.q4_outlier.value_type` = "bf16", "q8_0", or "q4_0" (delta storage format)
- `llama.q4_outlier.index_encoding` = "row_block_col"
- `llama.q4_outlier.store` = "delta"
- `llama.q4_outlier.tensor_count`
- Per-tensor: `llama.q4_outlier.tensor.{i}.name`, `.index`, `.values`, `.n_blocks`

**Sidecar tensor names:** `{base_name}.outlier_idx` (I32, [2, n_blocks]), `{base_name}.outlier_bf16` (type varies: BF16/Q8_0/Q4_0, [32, n_blocks])

## CLI Usage

```bash
# Basic: use ftype string
llama-quantize --outlier-max-frac 0.05 model.gguf out.gguf Q4_0_BF16_OUTLIER

# With Q8_0 delta sidecar (34 bytes/block, 47% smaller than BF16)
llama-quantize --outlier-base q4_0 --outlier-value-type q8_0 \
  --outlier-max-abs-error 0.002 --outlier-max-frac 0.99 \
  model-bf16.gguf model-q8d.gguf Q4_0_BF16_OUTLIER

# With Q4_0 delta sidecar (18 bytes/block, 72% smaller than BF16)
llama-quantize --outlier-base q4_0 --outlier-value-type q4_0 \
  --outlier-max-abs-error 0.002 --outlier-max-frac 0.99 \
  model-bf16.gguf model-q4d.gguf Q4_0_BF16_OUTLIER

# With imatrix for weighted scoring
llama-quantize --outlier-base q4_0 --outlier-max-frac 0.03 \
  --imatrix imatrix.bin model.gguf model-q4-outlier.gguf
```

Shared `--outlier-*` flags set both Q8 and Q4 params; `--outlier-base q4_0` sets ftype and enables Q4 mode.
`--outlier-value-type` selects delta storage: `bf16` (default), `q8_0`, or `q4_0`.

## Bugs Fixed

| Bug | Fix | Commit |
|-----|-----|--------|
| `q8_outlier_reconstruct_tensor` pointer overload: `idx[b*1+1]` → `idx[b*2+1]` | Stride fix | `d2abd74f2` |
| Q4_0 scale formula: `max_abs/7` → `max_val/-8` | Match reference quantization | `61831e2fd` |
| `Q4_0_BF16_OUTLIER` ftype overwrites CLI `--outlier-*` args | Propagate q8→q4 params | `1cac80879` |
| Tied embeddings (`token_embd`/`output`) upgrade to Q6_K | Exclude Q4_0_BF16_OUTLIER from upgrade | `2e40e112b` |
| Q4 local params not created when ftype triggers outlier mode | Add local params copy with defaults | `28346fcec` |

## Implementation Files

| File | Role |
|------|------|
| `include/llama.h` | `LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER`, `q4_outlier_*` params, `LLAMA_OUTLIER_VALUE_TYPE_{BF16,Q8_0,Q4_0}` enum |
| `src/llama-q8-outlier.h` | Q4 metadata keys (`LLAMA_Q4_OUTLIER_*`) |
| `src/llama-quant.cpp` | `q4_outlier_*` functions: candidate detection, analysis, deltas (BF16/Q8_0/Q4_0), reconstruction, metadata |
| `src/llama-model-loader.{h,cpp}` | `has_q4_outlier_metadata()`, `read_q4_outlier_metadata()`, Q4 sidecar validation (accepts BF16, Q8_0, Q4_0) |
| `src/llama-model.cpp` | `build_outlier_info` (non-streaming + streaming cache registration), `patch_embedding_outliers` (token_embd pre-patch with Q4_0 delta support) |
| `src/llama-outlier-stream.{h,cpp}` | Streaming outlier cache: `upload_entry`, `add_entry`, Q4_0 element size (18 bytes) |
| `src/llama-context.cpp` | `graph_compute` dispatches to outlier correction via `build_lora_mm` |
| `src/llama-graph.cpp` | `build_lora_mm` — wraps matmul with outlier correction |
| `ggml/src/ggml-cuda/outlier.cu` | CUDA kernels: `outlier_blocks_kernel_{bf16,q8_0,q4_0}`, dispatch on values_type |
| `ggml/src/ggml-cpu/ops.cpp` | CPU kernel: `ggml_compute_forward_mul_mat_outlier_blocks` with Q4_0 dequant |
| `tools/quantize/quantize.cpp` | `--outlier-base q4_0`, `--outlier-value-type bf16|q8_0|q4_0` CLI, ftype string, param propagation |

## Bugs Fixed (continued)

| Bug | Fix | Commit |
|-----|-----|--------|
| CPU context sidecar skip | Removed skip, sidecars created for all non-Host contexts | `5714857df` |
| CUDA_Host sidecar crash | Skip `_Host` buffer types in sidecar creation | `5714857df` |
| `cpu_ctx` out of scope | Use `ctx` in 0-byte buffer check | `a9c6acb11` |
| `tensor->data` NULL crash | Added `&& tensor->data` guards in `build_outlier_info` | `e61111a5d` |
| O(N*M) sidecar lookup | Hash map for O(1) lookup in `build_outlier_info` | (recent) |
| `--keep-split` restriction | Removed throw for Q8 and Q4 outlier | `65ea7de88` |
| Scoring FP16 round-trip mismatch | Simulate FP16 scale round-trip in scoring | `db3c7f804` |
| Zero-L2 tensors | GPU idx sampling diagnostic, bounds checks | `c43ca78a3` |
| Sidecar size not reported | Show sidecar size and total size in quantize output | `b4007e788` |
| Ratio pre-filter too aggressive (max-abs-error) | Remove ratio gate from max-abs-error path | `948e01a06` |

### Q4_0 Delta Sidecar Bugs (2026-06-17)

| Bug | Fix | Commit |
|-----|-----|--------|
| `build_outlier_info` rejected Q4_0 values tensor | Added `GGML_TYPE_Q4_0` to type check | `23e1ccf59` |
| Model loader rejected Q4_0 values tensor | Added `GGML_TYPE_Q4_0` to validation | `5a9c72552` |
| `patch_embedding_outliers` missing Q4_0 branch | Added Q4_0 values_element_size and dequant branch | `63937f399` |
| Debug logging OOB read for Q4_0 values | Added Q4_0 branch, fixed BF16 hardcoded sizes → `ggml_nbytes` | `b45675791` |
| Phase 1b `val_elem` defaulted to 64 bytes for Q4_0 | Added Q4_0 branch (18 bytes) | `16a45e84e` |
| Streaming GGUF read `val_nbytes` defaulted to 64 bytes | Added Q4_0 branch (18 bytes) in Phase 1b streaming path | `03ab5fe5c` |

## Selection Modes

Two block selection modes are available:

### Residual-Energy Scoring (default)

```
score = Σ α[j] · (W[j] - W_q4[j])² / Σ α[j] · W[j]²
```

Blocks ranked by score, top `max_frac` selected. Requires `--outlier-ratio >= 4.0` (pre-filter: `max_abs / second_abs`).

### Max Absolute Error (alternative)

```
--outlier-max-abs-error 0.0005
```

Protects any block where `|W[j] - W_q4[j]| > threshold` for any element. No ratio pre-filter. Useful when absolute precision matters more than relative energy.

## Runtime Status

- **Layer-split mode:** Works without `--sm tensor` on single GPU
- **Tensor-split mode:** Works with `--sm tensor`
- **`--stream-outliers`:** Works correctly with all delta types (BF16, Q8_0, Q4_0). Sidecars streamed from CPU to GPU on-demand.
- **Non-streaming:** Works with BF16 and Q8_0 deltas. **Q4_0 deltas currently crash** in non-streaming mode (ggml backend interaction with Q4_0 tensors in permanent GPU buffers). Use `--stream-outliers` with Q4_0 deltas.
- **Q4_0 delta quality (Qwen3.5-2B):** Same top P ~84% (vs ~89% BF16 baseline), KLD ~0.093 (vs ~0.04-0.06)
- **Q4_0 delta compression:** 62% smaller sidecar vs BF16, 38% smaller total model

## Known Limitations

- **MoE multi-expert delta computation:** Only first expert handled (same as Q8)
- **Token embedding correction:** `patch_embedding_outliers()` pre-patches token embeddings at load time by dequantizing Q4_0 rows, adding delta values, and re-quantizing. Only supports Q4_0 base type. Works with BF16, Q8_0, and Q4_0 delta types.
- **Non-streaming Q4_0 deltas crash:** Loading Q4_0 sidecar tensors as permanent GPU tensors causes a segfault during inference. Root cause likely in ggml backend buffer allocation for Q4_0 tensors with shape [32, n_blocks]. Workaround: use `--stream-outliers`.
- **Block size hardcoded to 32** in kernels (matches QK4_0 = 32, but not parametric)
- **Double-quantization quality cost:** Q4_0 deltas on Q4_0 base means deltas are quantized twice. Each 4-bit quantization step adds ~5-6% same-top-P loss.

## Streaming Outlier Cache Lessons Learned

The `--stream-outliers` feature streams outlier sidecar data from RAM to VRAM with a sliding window. Here are the bugs we hit and what would have been nice to know:

### 1. Off-by-one in suffix length (subtle, catastrophic)

The streaming registration code in `build_outlier_info` Phase 1b uses `rfind(".outlier_idx")` with `idx_suffix_len`. The original code had `idx_suffix_len = 13` (copy-pasted from `.outlier_bf16` which IS 13 chars). But `.outlier_idx` is only 12 chars. This made the rfind check ALWAYS fail, so NO entries were registered. Result: streaming mode was silently equivalent to base model with no outliers (83% top-p agreement, same as baseline).

**Moral:** When using rfind-based suffix matching, verify the string lengths. Test with one known-good entry to confirm the pipeline works end-to-end.

### 2. Null `info.values` in `patch_embedding_outliers`

In streaming mode, Phase 1b sets `info.values = nullptr` and `info.idx = nullptr` (since sidecar tensors aren't loaded in GPU memory). But `patch_embedding_outliers()` unconditionally dereferenced `info.values->buffer`. Any model with a Q4_0 token_embd tensor and outlier blocks would segfault.

**Fix:** Guard with `if (info.values)` and fall back to reading from `outlier_cache.entries[name].values_data` (CPU-side copy populated during model load).

**Moral:** When adding a mode that changes pointer semantics, audit ALL consumers of those pointers.

### 3. `weight_tensor->data` can be NULL even when buffer exists

With `--no-mmap`, backend buffer allocation sets up the buffer (`tensor->buffer`) but `tensor->data` can be NULL for some reasons (GPU allocation nuances). The original code called `ggml_backend_tensor_get()` unconditionally when `weight_on_gpu` was true, which asserts `tensor->data != NULL`.

**Fix:** Add `&& weight_tensor->data` guard to both the read and write-back paths in `patch_embedding_outliers`.

**Moral:** GPU tensor data pointers are not guaranteed after allocation — always check `.data != NULL` before calling `ggml_backend_tensor_get/set`.

### 4. `--custom-logs` is indispensable

The codebase uses `ggml_custom_logs_enabled()` to gate verbose debug output. Without it, silent failures like the off-by-one bug go undetected. Always add `--custom-logs` instrumentation at key decision points: entry registration, cache population source, value data source, upload sizes, etc.

### 5. Two-phase loading must be consistent

Phase 1a (non-streaming) looks for sidecar tensors in `tensors_by_name`. Phase 1b (streaming) looks for them in `ml->weights_map`. When streaming is enabled, sidecar tensors are NOT created in GPU contexts, so Phase 1a finds nothing and Phase 1b must handle everything. Phase 1b then uses `tensors_map.find(parent_name)` to get the tensor pointer — this is different from Phase 1a's iteration over `tensors_by_name`. Ensure the tensor pointers match between phases.

### 6. Use-after-free from sliding window eviction during graph build

The CUDA kernel failed with `cudaMemcpy: invalid argument` because the graph builder's `build_lora_mm()` creates graph nodes referencing cached GPU tensor pointers, but eviction can free those tensors before graph compute runs.

**Root cause:** A single transformer layer has ~7-8 weight tensors. With `window=6`, the 7th `ensure_gpu()` evicts the 1st tensor (now LRU). But graph nodes built from the 1st tensor still hold dangling pointers to that freed GPU buffer. When the CUDA kernel executes, `idx->data` is freed memory → `cudaMemcpy` fails.

**Fix:** Two-part solution:
1. **Latched entries:** `llama_outlier_cache_entry.latched` flag. Set to `true` after `ensure_gpu()` uploads data. `evict_lru()` skips entries with `latched=true`.
2. **Release between passes:** `release_all_latches()` clears all latches. Called at the start of `build_graph()` (before each graph build). This ensures entries from the *previous* forward pass can be evicted, but entries used in the *current* graph build cannot.
3. **Window overflow guard:** When all loaded entries are latched and a new upload is needed, `evict_lru()` returns without evicting anything. The `while (loaded_count >= window_size)` loop breaks when `loaded_count` doesn't decrease — the window temporarily exceeds the limit. The excess entries are cleaned up on the next `release_all_latches()` + eviction cycle.

**Moral:** Any cache that evicts entries during graph building must track which entries have graph nodes referencing them. A simple "latch" mechanism (mark in-use) prevents use-after-free without complex reference counting.

**Tradeoff:** Latching means entries loaded during a forward pass stay in VRAM until the pass completes. The window grows monotonically during graph building — every weight tensor touched in that pass is latched and cannot be evicted. For a 24-layer model with ~8 weight tensors per layer, all ~192 sidecar tensors end up in VRAM during the pass (same peak as non-streaming). VRAM savings only materialize *between* passes: `release_all_latches()` at the next `build_graph()` call makes all entries eligible for eviction, and the cache compacts back to `window_size` (12) on subsequent uploads. To get intra-pass VRAM savings, entries would need reference counting from graph nodes — evict only entries whose graph nodes have already been computed.

### 7. Allocated buffer can be 32x larger than data for Q8_0 values

`upload_entry()` creates `val_t = [32, n_blocks]` of type Q8_0. `ggml_nbytes()` for a Q8_0 tensor considers each "element" as a 34-byte Q8_0 block, so the total is `32 * n_blocks * 34` bytes. But the actual data uploaded is only `n_blocks * 34` bytes (one Q8_0 block per outlier block). The GPU buffer is 32x larger than the data.

For BF16 values, there is no waste: `[32, n_blocks]` of BF16 = `64 * n_blocks` bytes, which exactly matches the data.

**Fix candidate:** Use shape `[1, n_blocks]` for Q8_0 values, or use a raw buffer allocation instead of tensor shapes.
