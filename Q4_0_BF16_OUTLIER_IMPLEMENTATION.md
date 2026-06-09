# Q4_0_BF16_OUTLIER Implementation

## Overview

Q4_0_BF16_OUTLIER extends the Q8_0_BF16_OUTLIER pattern to Q4_0 base quantization: base Q4_0 matmul + sparse BF16 residual correction for high-error blocks.

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

**Kernels unchanged:** `ggml_compute_forward_mul_mat_outlier_blocks()` (CPU) and `outlier_blocks_kernel()` (CUDA) operate purely on BF16 values + I32 indices — base type agnostic.

**Graph builder unchanged:** `build_lora_mm()` wraps `ggml_mul_mat` with correction + `ggml_add` for any tensor with outlier info.

## Storage Cost

| Component | Size per block |
|-----------|---------------|
| Q4_0 base | 4.5 bpw (18 bytes / 32 values) |
| BF16 delta sidecar | 18 bpw (64 bytes / 32 values) |
| I32 index sidecar | 2 bpw (8 bytes / 32 values) |

Effective bpw with fraction `f` protected:
```
bpw ≈ 4.5 + 18f
```

| f | bpw |
|---|-----|
| 1% | 4.68 |
| 5% | 5.40 |
| 10% | 6.30 |

**Recommendation:** Keep `f` small (1-5%). Beyond ~10%, sidecar cost pushes into Q5/Q6 territory with a slower sparse correction path.

## Metadata

**GGUF keys** (separate namespace from Q8):
- `llama.q4_outlier.version` = 1
- `llama.q4_outlier.block_size` = 32
- `llama.q4_outlier.base_type` = "q4_0"
- `llama.q4_outlier.value_type` = "bf16"
- `llama.q4_outlier.index_encoding` = "row_block_col"
- `llama.q4_outlier.store` = "delta"
- `llama.q4_outlier.tensor_count`
- Per-tensor: `llama.q4_outlier.tensor.{i}.name`, `.index`, `.values`, `.n_blocks`

**Sidecar tensor names:** `{base_name}.outlier_idx` (I32, [2, n_blocks]), `{base_name}.outlier_bf16` (BF16, [32, n_blocks])

## CLI Usage

```bash
# Basic: use ftype string (thresholds from --outlier-* flags)
llama-quantize --outlier-blocks bf16 --outlier-max-frac 0.05 \
  model.gguf model-q4-outlier.gguf Q4_0_BF16_OUTLIER

# Alternative: explicit base selection
llama-quantize --outlier-base q4_0 --outlier-max-frac 0.05 \
  model.gguf model-q4-outlier.gguf

# With imatrix for weighted scoring
llama-quantize --outlier-base q4_0 --outlier-max-frac 0.03 \
  --imatrix imatrix.bin model.gguf model-q4-outlier.gguf
```

Shared `--outlier-*` flags set both Q8 and Q4 params; `--outlier-base q4_0` sets ftype and enables Q4 mode.

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
| `include/llama.h` | `LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER`, `q4_outlier_*` params |
| `src/llama-q8-outlier.h` | Q4 metadata keys (`LLAMA_Q4_OUTLIER_*`) |
| `src/llama-quant.cpp` | `q4_outlier_*` functions: candidate detection, analysis, deltas, reconstruction, metadata |
| `src/llama-model-loader.{h,cpp}` | `has_q4_outlier_metadata()`, `read_q4_outlier_metadata()`, Q4 sidecar validation |
| `tools/quantize/quantize.cpp` | `--outlier-base q4_0` CLI, ftype string, param propagation |

## Bugs Fixed (continued)

| Bug | Fix | Commit |
|-----|-----|--------|
| CPU context sidecar skip | Removed skip, sidecars created for all non-Host contexts | `5714857df` |
| CUDA_Host sidecar crash | Skip `_Host` buffer types in sidecar creation | `5714857df` |
| `cpu_ctx` out of scope | Use `ctx` in 0-byte buffer check | `a9c6acb11` |
| `tensor->data` NULL crash | Added `&& tensor->data` guards in `build_outlier_info` | `e61111a5d` |
| O(N*M) sidecar lookup | Hash map for O(1) lookup in `build_outlier_info` | (recent) |
| `--keep-split` restriction | Removed throw for Q8 and Q4 outlier | `65ea7de88` |

## Runtime Status

- **Layer-split mode:** Works without `--sm tensor` on single GPU
- **Tensor-split mode:** Works with `--sm tensor`
- **Same top P:** 2-3% (horrible) — indicates fundamental numerical issue
- **Zero-L2 tensors:** Two tensors per layer produce L2=0 despite non-zero n_blocks — suspected idx coordinate mismatch
- **PPL:** ~2M (broken)

## Known Limitations

- **MoE multi-expert delta computation:** Only first expert handled (same as Q8)
- **`ggml_get_rows` (embedding lookup):** No correction applied — graceful degradation to plain Q4_0
- **Block size hardcoded to 32** in kernels (matches QK4_0 = 32, but not parametric)
