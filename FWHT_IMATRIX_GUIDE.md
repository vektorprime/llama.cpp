# FWHT-Compat Imatrix Guide

How to collect and use an imatrix that is compatible with the FWHT-based
Q4_K_M_CLONE quantizer. This solves the coordinate mismatch between
original-space importance and FWHT-rotated weights.

## Theory

The imatrix is a per-input-channel vector `q_i = E[x_i^2]` — activation
variance at each column. After FWHT rotates the weight columns, the
original per-channel importance no longer aligns with the transformed
coordinates. The correct statistic is `q'_l = E[(R x)_l^2]` where R is the
fwht_256 transform.

**Critical:** The imatrix GGUF stores per-tensor named entries. One file can
contain FWHT-space entries for CLONE-tensors and original-space entries for
non-FWHT-tensors. The quantizer looks up entries by tensor name, so each
tensor gets the appropriately-collected imatrix.

## Code Changes (from stock llama.cpp)

### ggml/src/ggml-quants.c

1. `fwht_256` made non-static (exposed for llama-imatrix to call):

```c
// was: static void fwht_256(float * x) {
void fwht_256(float * x) {
```

2. `quantize_row_q4_K_M_CLONE_ref` and `dequantize_row_q4_K_M_CLONE`:
   cleaned of debug fprintf. Keep them as thin wrappers: FWHT then call
   stock Q4_K quant/dequant.

3. `quantize_q4_K_M_CLONE` — FWHT weights, pass imatrix through unchanged:
   (the imatrix was collected in FWHT space for CLONE-tensors by the collector)

```c
size_t quantize_q4_K_M_CLONE(..., const float * quant_weights) {
    for each row:
        copy weights, fwht_256 per 256-block
        if (quant_weights) quantize_row_q4_K_impl(rotated, ..., quant_weights);
        else              quantize_row_q4_K_ref(rotated, ...);
}
```

### ggml/src/ggml-quants.h

```c
GGML_API void fwht_256(float * x);
```

### tools/imatrix/imatrix.cpp

1. `#include <unordered_set>`

2. Forward-declare the FWHT function:
```cpp
extern "C" void fwht_256(float * x);
```

3. `accumulate_importance()` — optional FWHT path:
```cpp
static void accumulate_importance(float * dst, const float * src,
        int64_t n, bool use_fwht) {
    if (!use_fwht) {
        for (int64_t j = 0; j < n; ++j) dst[j] += src[j] * src[j];
        return;
    }
    std::vector<float> tmp(256);
    for (int64_t block = 0; block < n; block += 256) {
        memcpy(tmp.data(), src + block, 256 * sizeof(float));
        fwht_256(tmp.data());
        for (int j = 0; j < 256; ++j) {
            if (!std::isfinite(tmp[j])) { /* abort */ }
            dst[block + j] += tmp[j] * tmp[j];
        }
    }
}
```

4. Added to `IMatrixCollector`:

```cpp
class IMatrixCollector {
public:
    void load_fwht_tensor_file(const std::string & file_path);
    bool tensor_uses_fwht(const std::string & tensor_name) const;
private:
    std::unordered_set<std::string> m_fwht_tensor_names;
};
```

- `load_fwht_tensor_file`: reads `name=type` lines, marks `q4_k_m_clone` tensors
- `tensor_uses_fwht`: returns true if tensor name is in the set

5. In `collect_imatrix`, both the dense MUL_MAT and MoE MUL_MAT_ID paths
   call `tensor_uses_fwht(wname)` to decide whether to FWHT the activation
   before accumulating squares:

```cpp
const bool use_fwht = tensor_uses_fwht(wname);
// ...
if (use_fwht) {
    accumulate_importance(e.values.data() + mat_start, x, ne0, true);
} else {
    accumulate_importance(e.values.data() + mat_start, x, ne0, false);
    // (or keep the original loop with isfinite check)
}
```

6. CLI argument `--fwht-tensor-file <path>` extracted before
   `common_params_parse`:

```cpp
for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--fwht-tensor-file") == 0 && i + 1 < argc) {
        g_collector.load_fwht_tensor_file(argv[i + 1]);
        // shift args down, argc -= 2
    }
}
```

## Procedure — New Model

### 0. Prerequisites

- `llama-imatrix` and `llama-quantize` built with the changes above
- BF16/F16 full-precision GGUF of the model
- **Calibration text file**: use `bartowski_calibration_data_v5.txt` (NOT wikitext —
  wikitext is the eval set and would taint the evaluation)
- A tensor-type file mapping tensors → quantization types

### 1. Create the tensor-type file

**The easiest way**: run a throwaway quantize and extract the tensor→type mapping:

```bash
build/bin/llama-quantize "$MODEL_BF16" /tmp/tmp.gguf Q4_K_M_CLONE 2>&1 \
  | grep "converting to" \
  | sed 's/^.*\] //' \
  | sed 's/ -.*converting to / /' \
  | awk '{print $1 "=" tolower($2)}' \
  > tensor_map.txt
rm -f /tmp/tmp.gguf
```

This captures exactly which tensors the clone quantizer will target (only Q4_K_M →
Q4_K_M_CLONE; boosted tensors keep their Q5_K/Q6_K types). The collector uses
only the `q4_k_m_clone` entries to decide where to apply FWHT during activation
accumulation.

Format: one `tensor_name=type` per line (same format as `--tensor-type-file`
accepted by `llama-quantize`).

Example (`tensor_map.txt`):
```
token_embd.weight=q6_k
blk.0.attn_gate.weight=q5_k
blk.0.ffn_gate.weight=q4_k_m_clone
blk.0.ffn_up.weight=q4_k_m_clone
blk.0.ssm_out.weight=q8_0
...
```

Rules:
- Any tensor assigned `q4_k_m_clone` will receive FWHT-transformed
  importance during imatrix collection AND FWHT during quantization.
- All other tensors get normal (original-space) importance and the stock
  quantization type specified.
- The `--tensor-type-file` passed to `llama-imatrix` (via `--fwht-tensor-file`)
  and to `llama-quantize` (via `--tensor-type-file`) should be the SAME file.
  The collector only looks for `q4_k_m_clone` entries; other types are ignored
  during imatrix collection but used during quantization.

### 2. Generate the imatrix

```bash
# Replace paths as needed
CALIB_FILE=/path/to/calibration.txt
MODEL_F16=/path/to/model-f16.gguf
TENSOR_MAP=/path/to/tensor_map.txt
IMATRIX_OUT=/path/to/imatrix_fwht.gguf

build/bin/llama-imatrix \
    -m "$MODEL_F16" \
    -f "$CALIB_FILE" \
    -o "$IMATRIX_OUT" \
    -ngl 99 \
    --no-ppl \
    -c 512 \
    -b 4096 \
    -ub 512 \
    --chunks 200 \
    --output-frequency 50 \
    --fwht-tensor-file "$TENSOR_MAP"
```

Key flags:
- `-c 512`: context window per chunk
- `-b 4096`: batch size (higher = more GPU utilization)
- `--chunks 200`: process 200 chunks of calibration data (~100k tokens)
- `--fwht-tensor-file`: the SAME tensor-type file used during quantization
- The collector parses `q4_k_m_clone` entries from this file and applies
  FWHT to those tensors' activations. Other tensors get normal accumulation.

### 3. Verify the imatrix (optional)

```bash
build/bin/llama-imatrix \
    -m "$MODEL_F16" \
    --in-file "$IMATRIX_OUT" \
    --show-statistics
```

Check:
- All expected tensors are present (186+ for Qwen3.5-0.8B)
- FWHT tensors (those assigned `q4_k_m_clone`) have higher total activation
  energy (Σ(Act²)) than non-FWHT tensors of similar size (due to 256x energy
  spread of unnormalized FWHT)
- No NaNs, infinities, or zero-count entries

### 4. Quantize with the imatrix

```bash
build/bin/llama-quantize \
    --tensor-type-file "$TENSOR_MAP" \
    --imatrix "$IMATRIX_OUT" \
    "$MODEL_F16" \
    /path/to/output.gguf \
    Q4_K_M_CLONE
```

The `--tensor-type-file` sets per-tensor types; the `--imatrix` provides
importance data. For CLONE-tensors, the imatrix entries were collected in
FWHT space and the quantizer applies FWHT to weights, so coordinates match.

To quantize WITHOUT imatrix (for comparison):
```bash
build/bin/llama-quantize \
    --tensor-type-file "$TENSOR_MAP" \
    "$MODEL_F16" \
    /path/to/output-noimat.gguf \
    Q4_K_M_CLONE
```

### 5. Evaluate

```bash
build/bin/llama-perplexity \
    -m /path/to/output.gguf \
    -f wiki.test.raw \
    -t 8 -c 256 --chunks 250 \
    -fa on --cache-type-k bf16 --cache-type-v bf16 \
    --no-mmap -ngl 999 -np 1 \
    --kl-divergence \
    --kl-divergence-base /path/to/reference.logits
```

Compare KLD, same-top-p, and PPL between the imatrix and no-imatrix variants.

### Expected Results

For a 0.8B model with ~48 CLONE-tensors (Qwen3.5-0.8B):

| Variant | PPL Δ | KLD Δ | Same top p Δ |
|---------|-------|-------|-------------|
| imatrix vs no imatrix | -0.25 | -0.0028 | +0.21% |

The recollected imatrix (E[(Rx)^2]) should improve every metric compared
to no imatrix. If it doesn't:
- Verify the tensor-type file paths match between imatrix collection and quantization
- Check that FWHT tensors show higher Σ(Act²) in the imatrix statistics
- Ensure the same `fwht_256` implementation is used in collector and quantizer
- Run with `-v` to verify `loaded N FWHT tensor names` appears in the log

## Why fabs(FWHT(old_imatrix)) is Wrong (Do Not Use)

An earlier attempt FWHT-transformed the imatrix itself inside the quantizer:

```c
// WRONG — do not use:
fwht_256(imat_block);
for (int i = 0; i < QK_K; i++) imat_block[i] = fabsf(imat_block[i]);
```

This is mathematically invalid:
1. Importance transforms as `R C R^T` (covariance), not as `R q` (vector)
2. `fabs(FWHT(q))` turns a flat imatrix `[1,1,...,1]` into `[256,0,...,0]`
   — incorrectly saying 255 coordinates have zero importance
3. The correct diagonal approximation from old imatrix is the flat block mean,
   not the Hadamard spectrum

The only correct approaches are:
- **Recollect** `E[(R x)^2]` directly from FWHT-transformed activations (this guide)
- **Flat block mean**: `q' = mean(q) * [1,...,1]` per 256-group (loses information
  but is at least mathematically sound)
- **Full covariance**: use GPTQ/QuIP-style Hessian from RCR^T (high effort)

## Diagnostic Checks

### Energy conservation

For unnormalized FWHT (entries ±1), each 256-block's squared-sum is scaled by 256:
```
sum(q_fwht_block) ≈ 256 * sum(q_original_block)
```

### Flat-input test

If all activations have the same variance, the FWHT-space imatrix should be
flat (all entries equal), not spiky:
```
Input:  q = [1, 1, ..., 1]   (all entries = 1.0)
Output: q' should be [256, 256, ..., 256]  (all equal)
NOT:    [65536, 0, 0, ..., 0]  (this would be fabs(FWHT(q)))
```
