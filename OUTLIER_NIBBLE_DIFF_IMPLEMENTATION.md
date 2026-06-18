# Outlier Block Nibble-Diff Implementation

## Overview

Outlier protection stores sparse sidecar deltas for high-error blocks in Q4_0 or Q8_0 base quantization. The base tensor remains intact (no zeroing), and a sparse matmul correction is added at inference time.

**Delta storage uses a custom 4-bit nibble encoding** that replaces the older BF16/Q8_0 delta formats, reducing sidecar size from 64 bytes/block (BF16) or 34 bytes/block (Q8_0) down to **16 bytes/block**.

---

## Architecture

```
base tensor:  normal Q4_0 or Q8_0 (blocks NOT zeroed)
sidecar idx:  [2, n_blocks] I32  — (row, block_col) per protected block
sidecar values: [16, n_blocks] I8  — packed nibble-diff deltas

Inference:
  Y = X @ W_base              (standard Q4_0/Q8_0 matmul)
  Y += sparse_correction(idx, values, X)    (ggml_mul_mat_outlier_blocks)

Correction:
  For each protected block at (row, block_col):
    Y[row, token] += sum_{j=0..31} nibble_decode(values[block, j]) * X[col0+j, token]
```

The delta approach ensures operations that support correction (matmul via `build_lora_mm`) get exact reconstruction within nibble precision, while operations that don't (`ggml_get_rows` for embeddings) fall back to standard Q4_0/Q8_0 quality.

---

## Nibble-Diff Encoding (4 bits per weight)

Each weight's delta is encoded as a 4-bit nibble. 32 weights = 16 bytes per block.

### Bit Layout (MSB to LSB)

| Bit | Name | 0 | 1 |
|-----|------|---|---|
| 3 | enable | skip (diff ≈ 0) | apply diff |
| 2 | sign | negative | positive |
| 1 | zero count | 0.0X (no extra zero) | 0.00X (one extra zero) |
| 0 | digit | X = 1 | X = 2 |

"0.0" is always implied.

### Value Table

| Nibble | Enable | Sign | Zeros | Digit | Value |
|--------|--------|------|-------|-------|-------|
| 0x0_ | 0 | — | — | — | 0 (disabled) |
| 0x8 | 1 | neg | 0 | 0 | -0.01 |
| 0x9 | 1 | neg | 0 | 1 | -0.02 |
| 0xA | 1 | neg | 1 | 0 | -0.001 |
| 0xB | 1 | neg | 1 | 1 | -0.002 |
| 0xC | 1 | pos | 0 | 0 | +0.01 |
| 0xD | 1 | pos | 0 | 1 | +0.02 |
| 0xE | 1 | pos | 1 | 0 | +0.001 |
| 0xF | 1 | pos | 1 | 1 | +0.002 |

### Packing

Two nibbles per byte: lower nibble = weight j, upper nibble = weight j+1.

### Encoding Rules

Diffs outside the representable range (`|diff| < 0.0005`) are encoded as disabled (0x0). The encoding picks the closest representable value:
- `|diff| ≥ 0.015` → ±0.02
- `|diff| ≥ 0.005` → ±0.01
- `|diff| ≥ 0.0015` → ±0.002
- otherwise → ±0.001

### Example

```
1110:
  1 = enable diff math for this weight
  1 = sign is positive
  1 = add one extra zero → 0.00X
  0 = digit is 1 → 0.001
Final diff: +0.001
```

---

## Storage Cost

| Component | BF16 delta | Q8_0 delta | Nibble-diff |
|-----------|-----------|-----------|-------------|
| Per-block values | 64 bytes | 34 bytes | 16 bytes |
| Per-block idx | 8 bytes | 8 bytes | 8 bytes |

Effective bpw for Q4_0 base with fraction `f` protected:
```
bpw ≈ 4.5 + 6f    (nibble)  vs  4.5 + 18f  (BF16)
```

Effective bpw for Q8_0 base with fraction `f` protected:
```
bpw ≈ 8.5 + 6f    (nibble)  vs  8.5 + 18f  (BF16)
```

| f | Q4+BF16 | Q4+Nibble | Q8+BF16 | Q8+Nibble |
|---|---------|-----------|---------|-----------|
| 1% | 4.68 | 4.56 | 8.68 | 8.56 |
| 5% | 5.40 | 4.80 | 9.40 | 8.80 |
| 10% | 6.30 | 5.10 | 10.30 | 9.10 |

---

## File Type

- **Enum:** `LLAMA_FTYPE_MOSTLY_Q8_0_BF16_OUTLIER = 41`
- **Enum:** `LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER = 42`
- **Value type:** `LLAMA_OUTLIER_VALUE_TYPE_NIBBLE_DIFF = 2` (default for new quantizations)

Backward compatible: `LLAMA_OUTLIER_VALUE_TYPE_BF16 = 0` and `LLAMA_OUTLIER_VALUE_TYPE_Q8_0 = 1` are still supported for reading old models.

---

## Outlier Detection

### Q8_0: Ratio + Non-Max RMSE

For each 32-weight block:
1. Find `max_abs` (largest absolute value) and `second_abs` (second largest)
2. Ratio filter: `max_abs / second_abs ≥ ratio_threshold`
3. Compute Q8_0 quantization with `d = max_abs / 127`
4. Compute relative RMSE on the 31 non-max values
5. Score = `ratio * rel_rmse`

Blocks ranked by score, top `max_frac` fraction selected.

### Q4_0: Residual-Energy Scoring

For each 32-weight block:
1. Ratio pre-filter: `max_abs / second_abs ≥ ratio_threshold`
2. Compute Q4_0 quantization with `d = max_val / -8` (FP16 round-tripped)
3. Score = `sum(α·error²) / sum(α·weight²)` — normalized residual energy

Blocks ranked by score, top `max_frac` selected. Minimum score floor: 1e-6.

### Q4_0: Max Absolute Error (Alternative)

`--outlier-max-abs-error 0.0005`: protects any block where `|W[j] - W_q4[j]| > threshold` for any element. No ratio pre-filter. Caps by row-major order.

---

## Implementation Files

| File | Role |
|------|------|
| `include/llama.h` | Enums: `LLAMA_OUTLIER_VALUE_TYPE_NIBBLE_DIFF`, `LLAMA_FTYPE_MOSTLY_Q*_*_OUTLIER`; quantize params |
| `src/llama-q8-outlier.h` | Metadata key constants, nibble encode/decode/pack/unpack functions |
| `src/llama-quant.cpp` | Outlier detection, delta computation (nibble/BF16/Q8), reconstruction, GGUF write |
| `src/llama-model.cpp` | `build_outlier_info()` (CSR layout), `patch_embedding_outliers()`, streaming cache integration |
| `src/llama-model-loader.h` | `has_q*_outlier_metadata()`, `read_q*_outlier_metadata()`, sidecar validation |
| `src/llama-graph.cpp` | `build_lora_mm()` — wraps matmul with `ggml_mul_mat_outlier_blocks` + `ggml_add` |
| `src/llama-outlier-stream.h` | Streaming cache: `llama_outlier_cache_entry`, `llama_outlier_stream_cache` |
| `src/llama-outlier-stream.cpp` | LRU eviction, GPU upload/download for streaming mode |
| `ggml/include/ggml.h` | `GGML_OP_MUL_MAT_OUTLIER_BLOCKS`, `ggml_mul_mat_outlier_blocks()` declaration |
| `ggml/src/ggml.c` | `ggml_mul_mat_outlier_blocks()` — creates tensor/op node, validates I8 type |
| `ggml/src/ggml-cpu/ops.cpp` | CPU kernel: row-partitioned sparse correction, nibble decoding |
| `ggml/src/ggml-cuda/outlier.cu` | CUDA kernel: one thread block per (outlier_block, token) pair, warp reduction, nibble/BF16/Q8 decode |
| `ggml/src/ggml-backend-meta.cpp` | Meta backend split-state handler for tensor parallelism |
| `tools/quantize/quantize.cpp` | CLI parsing for `--outlier-*` flags |

---

## CLI Usage

```bash
# Basic: Q4_0 base + nibble-diff outliers (default)
llama-quantize --outlier-base q4_0 --outlier-max-frac 0.05 \
  model.gguf model-q4-outlier.gguf

# Q8_0 base + nibble-diff outliers
llama-quantize --outlier-blocks bf16 --outlier-max-frac 0.02 \
  model.gguf model-q8-outlier.gguf Q8_0_BF16_OUTLIER

# With importance matrix
llama-quantize --outlier-base q4_0 --outlier-max-frac 0.03 \
  --imatrix imatrix.bin model.gguf model-q4-outlier.gguf

# Max absolute error mode (Q4_0)
llama-quantize --outlier-base q4_0 --outlier-max-abs-error 0.0005 \
  model.gguf model-q4-outlier.gguf

# Use BF16 deltas instead of nibble-diff (legacy)
llama-quantize --outlier-base q4_0 --outlier-value-type bf16 \
  model.gguf model-q4-outlier.gguf
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--outlier-base q4_0\|q8_0` | off | Enable outlier sidecars for Q4_0 or Q8_0 |
| `--outlier-blocks bf16` | off | Legacy: enable Q8_0 outlier sidecars |
| `--outlier-ratio N` | 16 (Q8) / 4 (Q4) | Ratio pre-filter threshold |
| `--outlier-max-frac F` | 0.02 | Maximum protected block fraction |
| `--outlier-score S` | 0.01 | Minimum residual-energy score (Q4) |
| `--outlier-nonmax-rel-rmse X` | 0.01 | Minimum non-outlier RMSE (Q8) |
| `--outlier-max-abs-error T` | 0.0 | Max absolute error threshold (Q4 alt mode) |
| `--outlier-value-type bf16\|q8_0\|nibble_diff` | nibble_diff | Delta storage type |
| `--outlier-report PATH` | — | Write JSON report |
| `--outlier-include-weights REGEX` | — | Only protect matching tensors |
| `--outlier-exclude-weights REGEX` | — | Skip matching tensors |
| `--stream-outliers` | off | Stream sidecars with sliding window (VRAM saving) |

---

## Data Flow

### Quantization (write path)

```
F32 weight → [detect outlier blocks] → [idx: (row, block_col)]
                                       → [base quantize Q4_0/Q8_0]
                                       → [compute delta = F32 - base_dequant]
                                       → [encode deltas as nibble-diff]
                                       → [write base + idx + values to GGUF]
```

### Model load (read path)

```
GGUF → [load base tensors to GPU]
     → [load sidecar tensors to GPU (or CPU in streaming mode)]
     → build_outlier_info() builds CSR from idx data
```

### Inference (every matmul)

```
Y = Q4_0/Q8_0_matmul(W_base, X)              // fast, continuous
Y += sparse_correction(idx, nibble_values, X) // small overhead (~1-5% blocks)
```

---

## GGUF Metadata

Separate namespaces for Q8 and Q4:

```
llama.q8_outlier.version = 1
llama.q8_outlier.block_size = 32
llama.q8_outlier.base_type = "q8_0"
llama.q8_outlier.value_type = "nibble_diff"   // or "bf16", "q8_0"
llama.q8_outlier.index_encoding = "row_block_col"
llama.q8_outlier.store = "delta"
llama.q8_outlier.tensor_count = N
llama.q8_outlier.tensor.{i}.name = "blk.0.attn_q.weight"
llama.q8_outlier.tensor.{i}.index = "blk.0.attn_q.weight.outlier_idx"
llama.q8_outlier.tensor.{i}.values = "blk.0.attn_q.weight.outlier_bf16"
llama.q8_outlier.tensor.{i}.n_blocks = 1234
```

Same pattern with `q4_outlier` prefix for Q4.

Sidecar tensor name suffixes: `.outlier_idx` (12 chars), `.outlier_bf16` (13 chars).

---

## Kernels

### CPU (`ggml-cpu/ops.cpp`)

Row-partitioned to eliminate thread races. Each thread owns a range of output rows, iterates all blocks but only accumulates into its rows. No atomics needed.

For nibble-diff: unpacks 16 bytes → 32 nibbles, decodes each to float, computes dot product with activation slice.

### CUDA (`ggml-cuda/outlier.cu`)

Grid: `(n_blocks, n_tokens)` — one thread block per (outlier block, token) pair.
32 threads per block, warp reduction, `atomicAdd` to output.

For nibble-diff: each thread reads one nibble from packed bytes, decodes via `nibble_diff_decode()` device function.

Shard-aware: computes `col_offset` for tensor parallelism.

---

## Streaming Outlier Cache

The `--stream-outliers` feature streams sidecar data from RAM to VRAM with a sliding window.

### Key design points

- **Sliding window:** `window_size = 12` (covers one full layer + prefetch headroom)
- **Latched entries:** After `ensure_gpu()`, entry is latched so it cannot be evicted during the current graph build
- **Release between passes:** `release_all_latches()` clears all latches at start of each `build_graph()` call
- **Use-after-free prevention:** Graph nodes reference cached GPU tensor pointers; latches prevent eviction while nodes exist
- **Window overflow:** When all loaded entries are latched, the window temporarily exceeds the limit. Excess entries are cleaned up on next release + eviction cycle.

### VRAM tradeoff

Latching means entries loaded during a forward pass stay in VRAM until the pass completes. Peak VRAM equals non-streaming mode. Savings only materialize *between* passes when latches are released and the cache compacts.

---

## Bugs Fixed

| Bug | Symptom | Fix |
|-----|---------|-----|
| Meta backend split-state misconfig | Wrong output with `-sm tensor` | Added `AXIS_1 + AXIS_0` → `PARTIAL` case |
| CPU kernel thread race condition | Corrupt results with `-t > 1` | Row-partitioned output, zero atomics |
| CPU kernel activation stride | Wrong column mapping | Use `x->nb[1]` instead of `n_cols` |
| Embedding table bypasses correction | Corrupted input embeddings | Delta approach: base intact, zeroing removed |
| Q4_0 scale formula: `max_abs/7` | Wrong Q4_0 dequant | Fixed to `max_val/-8` matching reference |
| Q4_0 scale FP16 mismatch | Scoring mismatch at small scales | Simulate FP16 round-trip in scoring |
| Off-by-one suffix length in streaming | 0 entries registered silently | `.outlier_idx` = 12 chars, not 13 |
| Null `info.values` in patch_embedding | Segfault in streaming mode | Guard + fallback to `outlier_cache.entries` |
| GPU tensor `data` can be NULL | Assertion failure with `--no-mmap` | Add `&& tensor->data` guards |
| Allocated buffer 32x larger for Q8_0 | Wasteful VRAM on streaming upload | Fixed: `ne[0]=16` for nibble, `ne[0]=32` for BF16 |

---

## Known Limitations

- **2D tensors only** — 3D+ tensors (MoE experts) skipped
- **MoE multi-expert delta:** Only first expert handled in delta computation
- **Block size hardcoded to 32** in kernels
- **Nibble precision:** ±0.001 step for small diffs. Diffs outside representable range (< 0.0005) are treated as zero. Real-world diffs from Q4_0 quantization cluster around ±0.01-0.02 (from nibble rounding) and this encoding matches the "significant digits" pattern well.
- **CPU backend required** — MIRRORED ops run on all backends; CPU kernel zeros output for unsupported backends

---

## Lessons Learned

1. **Trace actual data paths:** The embedding table uses `ggml_get_rows`, not `ggml_mul_mat`. Graph builder only wraps matmul operations.
2. **Scheduler metadata matters:** Meta backend split-state misconfiguration silently produces wrong results in tensor parallelism mode.
3. **Delta > zeroing:** Keeping base tensors intact ensures graceful degradation (Q4_0/Q8_0 approximation) for operations without correction support.
4. **Use-after-free is subtle:** Streaming cache eviction during graph build requires latching to prevent dangling GPU pointers.
5. **Validate string lengths:** Off-by-one errors in suffix-based string matching cause silent failures.
6. **Always include debug logging:** `ggml_custom_logs_enabled()` gates verbose output; without it, silent failures go undetected.
