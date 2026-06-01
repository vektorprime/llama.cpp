# Implementation Plan: Q8_0 with BF16 Protected Outlier Blocks in llama.cpp

## Goal

Implement a llama.cpp quantization mode where a tensor is mostly stored as `Q8_0`, but selected 32-weight blocks are excluded from Q8_0 quantization and preserved as BF16 when an outlier value would distort the block scale.

The target behavior is:

```text
ordinary blocks:
    stored and executed as normal Q8_0

protected outlier blocks:
    excluded from Q8_0 quantization
    stored separately as BF16
    applied during inference as a sparse correction
```

This feature should work through:

```bash
./llama-quantize
./llama-cli
./llama-server
```

The plan intentionally avoids changing the existing `GGML_TYPE_Q8_0` binary layout, because standard `Q8_0` assumes every block has the same fixed structure.

---

## Current llama.cpp behavior

Current `Q8_0` is a fixed 32-weight block format. The reference quantizer processes one block of 32 values at a time, finds the maximum absolute value, sets the block scale to `amax / 127`, and stores each value as an int8 value relative to that scale.

Conceptually:

```c
amax = max(abs(x[0..31]));
d = amax / 127.0f;
for j in 0..31:
    q[j] = round(x[j] / d);
```

The dequantized value is then approximately:

```text
x_hat[j] = q[j] * d
```

This means one very large value can increase the scale for the whole 32-value block and reduce precision for the other 31 values.

Relevant source areas:

- `ggml/src/ggml-quants.c`
- `ggml/include/ggml.h`
- `src/llama-quant.cpp`
- `tools/quantize/README.md`

Useful references:

- <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c>
- <https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h>
- <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp>
- <https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md>

---

## Key design decision

Do **not** modify the existing `Q8_0` format.

A standard `Q8_0` tensor is expected to be a contiguous array of fixed-size `block_q8_0` records. Existing loaders, matmul kernels, dequantizers, and GPU backends assume that every block has the same layout.

A bad design would be:

```c
struct block_q8_0_or_bf16 {
    uint16_t tag;
    union {
        block_q8_0 q8;
        ggml_bf16_t bf16[32];
    } data;
};
```

This would technically allow per-block fallback, but it would also make every block as large as the largest variant. Since BF16 for 32 values already costs 64 bytes, this tagged union would cost more than simply storing the whole tensor as BF16.

Instead, use this design:

```text
base tensor:
    normal Q8_0 tensor
    protected blocks are zeroed

sidecar tensors:
    sparse list of protected block coordinates
    BF16 values for those protected blocks
```

Runtime then computes:

```text
Y = X @ W_q8_zeroed
Y += X @ W_outlier_bf16_sparse
```

This gives block-level protection while keeping ordinary Q8_0 compatibility internally.

---

## Proposed storage format

For each tensor using protected outlier blocks, store three tensors in GGUF.

### 1. Base weight tensor

```text
name:  blk.N.some_weight
 type: GGML_TYPE_Q8_0
 data: normal Q8_0 blocks, except protected blocks are zeroed
```

For a protected block:

```text
d  = 0
qs = all 0
```

This ensures the base Q8_0 matmul contributes nothing for that block.

### 2. Outlier index tensor

```text
name:  blk.N.some_weight.outlier_idx
 type: GGML_TYPE_I32
shape: [2, n_outlier_blocks]
```

Each entry stores:

```text
outlier_idx[0, k] = row_index
outlier_idx[1, k] = q8_block_col_index
```

Where:

```text
row_index          = output row of the weight matrix
q8_block_col_index = floor(column_index / 32)
```

### 3. Outlier BF16 value tensor

```text
name:  blk.N.some_weight.outlier_bf16
 type: GGML_TYPE_BF16
shape: [32, n_outlier_blocks]
```

Each column stores the original 32 BF16 values for one protected block.

---

## GGUF metadata

Add explicit metadata so the loader can discover and validate the sidecar tensors.

Suggested global metadata:

```text
llama.q8_outlier.version = 1
llama.q8_outlier.block_size = 32
llama.q8_outlier.base_type = "q8_0"
llama.q8_outlier.value_type = "bf16"
llama.q8_outlier.index_encoding = "row_block_col"
```

Suggested per-tensor metadata:

```text
llama.q8_outlier.tensor_count = N
llama.q8_outlier.tensor.0.name = "blk.0.attn_q.weight"
llama.q8_outlier.tensor.0.index = "blk.0.attn_q.weight.outlier_idx"
llama.q8_outlier.tensor.0.values = "blk.0.attn_q.weight.outlier_bf16"
llama.q8_outlier.tensor.0.n_blocks = 1234
```

Add a new file type or quantization label at the llama layer:

```text
MOSTLY_Q8_0_BF16_OUTLIER
```

Add a command-line quantization alias:

```text
Q8_0_BF16_OUTLIER
```

This alias should mean:

```text
base quantization: Q8_0
protected block value type: BF16
protected block storage: sparse sidecar tensors
```

Older llama.cpp versions should fail clearly when encountering this metadata, rather than silently loading the model and ignoring the BF16 sidecars.

---

## Size and overhead

Standard `Q8_0` stores:

```text
FP16 scale: 2 bytes
int8 values: 32 bytes
---------------------
total: 34 bytes per 32 weights
```

That is:

```text
34 bytes * 8 / 32 weights = 8.5 bits per weight
```

Each protected BF16 block adds approximately:

```text
BF16 values: 32 * 2 bytes = 64 bytes
index data: 2 * 4 bytes = 8 bytes, if using int32 row/block indices
```

So each protected block adds about 72 bytes on top of the zeroed Q8_0 block that is still present in the base tensor.

Approximate total bits per weight:

```text
8.5 + protected_fraction * 18
```

Examples:

```text
0.5% protected blocks  -> about 8.59 bpw
1.0% protected blocks  -> about 8.68 bpw
2.0% protected blocks  -> about 8.86 bpw
5.0% protected blocks  -> about 9.40 bpw
10.0% protected blocks -> about 10.30 bpw
```

This format is useful only when protected blocks are rare.

---

## Quantizer changes

### Files likely to modify

```text
tools/quantize/quantize.cpp
tools/quantize/README.md
src/llama-quant.cpp
include/llama.h
src/llama.cpp
src/llama-model-loader.cpp
gguf-py/gguf/constants.py, if Python GGUF tooling should understand the new metadata
```

The exact file boundaries may shift as llama.cpp evolves, but the core logic belongs near the existing quantization policy and GGUF writing code.

### New CLI options

Add options such as:

```bash
./llama-quantize \
  --outlier-blocks bf16 \
  --outlier-ratio 16 \
  --outlier-nonmax-rel-rmse 0.01 \
  --outlier-max-frac 0.02 \
  --outlier-report outliers.json \
  input-bf16.gguf output-q8-outlier.gguf Q8_0_BF16_OUTLIER
```

Suggested options:

```text
--outlier-blocks bf16
    Enable protected BF16 outlier blocks.

--outlier-ratio N
    Protect a block when the largest absolute value is at least N times larger
    than the second-largest or percentile-based reference value.

--outlier-nonmax-rel-rmse X
    Protect a block when Q8_0 quantization causes at least this much relative
    RMSE on the non-outlier values.

--outlier-max-frac F
    Maximum fraction of protected blocks per tensor.

--outlier-include-weights REGEX
    Only apply outlier protection to matching tensors.

--outlier-exclude-weights REGEX
    Do not apply outlier protection to matching tensors.

--outlier-report PATH
    Write a JSON report with counts, thresholds, and estimated bpw.

--outlier-store full|delta
    full: store exact BF16 block and zero the Q8_0 base block.
    delta: store correction over a normally quantized Q8_0 block.
```

For the requested behavior, the default should be:

```text
--outlier-store full
```

This means protected blocks are truly excluded from Q8_0 quantization.

---

## Outlier block detection

For every 32-weight block, evaluate whether the largest value is distorting the Q8_0 scale enough to justify BF16 protection.

### Basic detection algorithm

For each block:

```text
x[0..31] = original source values converted to float
```

Compute normal Q8_0 quantization:

```text
amax = max(abs(x[j]))
d = amax / 127
q[j] = round(x[j] / d)
x_hat[j] = q[j] * d
```

Find the largest absolute value:

```text
j_max = argmax(abs(x[j]))
max_abs = abs(x[j_max])
```

Find a non-outlier reference value:

```text
second_abs = second_largest(abs(x[j]))
```

Compute dominance ratio:

```text
ratio = max_abs / max(second_abs, eps)
```

Compute error on the non-largest values:

```text
nonmax_rmse = sqrt(mean((x[j] - x_hat[j])^2 for j != j_max))
nonmax_rms  = sqrt(mean(x[j]^2 for j != j_max))
nonmax_rel_rmse = nonmax_rmse / max(nonmax_rms, eps)
```

Protect the block if:

```text
ratio >= outlier_ratio
AND nonmax_rel_rmse >= outlier_nonmax_rel_rmse
```

This avoids protecting blocks where all 32 values are simply large. It only protects blocks where one value dominates the scale and harms the quantization of the rest of the block.

### Percentile-based alternative

Instead of comparing the largest value to the second-largest value, compare it to a percentile value:

```text
reference_abs = percentile(abs(x), 90th or 95th percentile)
ratio = max_abs / max(reference_abs, eps)
```

This can be more stable when a block contains more than one large value.

### Multi-outlier detection

If several values dominate the block, define the outlier set as:

```text
O = { j | abs(x[j]) >= max_abs / outlier_ratio }
```

Then compute error on the complement:

```text
nonoutlier_rel_rmse = rmse(x[j] - x_hat[j], j not in O) / rms(x[j], j not in O)
```

Protect if:

```text
len(O) is small
AND nonoutlier_rel_rmse is high
```

A reasonable initial rule:

```text
len(O) <= 4
```

### Importance-matrix-aware detection

When `--imatrix` is supplied, use weighted error instead of plain RMSE.

```text
weighted_sqerr  += importance[j] * (x[j] - x_hat[j])^2
weighted_energy += importance[j] * x[j]^2
weighted_rel_rmse = sqrt(weighted_sqerr / max(weighted_energy, eps))
```

Then protect based on weighted relative RMSE.

This is useful because the same numeric quantization error may matter more in some weights than others.

### Candidate scoring

If too many blocks qualify, rank candidates by score and keep the most important ones.

Suggested score:

```text
score = ratio * nonmax_rel_rmse
```

With imatrix:

```text
score = ratio * weighted_rel_rmse
```

Apply the per-tensor cap:

```text
max_protected_blocks = floor(total_blocks * outlier_max_frac)
```

If candidates exceed the cap, keep only the highest-scoring candidates.

---

## Quantization write path

For each tensor selected for `Q8_0_BF16_OUTLIER`:

```text
for each row:
    for each 32-value block in that row:
        if block is protected:
            write zero Q8_0 block into base tensor
            append [row, block_col] to outlier_idx
            append original BF16[32] values to outlier_bf16
        else:
            write normal Q8_0 block
```

The quantizer should print a report like:

```text
tensor blk.0.attn_q.weight:
    q8_0 blocks: 262144
    protected bf16 blocks: 912
    protected fraction: 0.348%
    estimated bpw: 8.563
```

The optional JSON report should look like:

```json
{
  "format": "Q8_0_BF16_OUTLIER",
  "block_size": 32,
  "tensors": [
    {
      "name": "blk.0.attn_q.weight",
      "blocks_total": 262144,
      "blocks_protected": 912,
      "protected_fraction": 0.00348,
      "estimated_bpw": 8.563,
      "thresholds": {
        "ratio": 16,
        "nonmax_rel_rmse": 0.01,
        "max_frac": 0.02
      }
    }
  ]
}
```

---

## Loader changes

### Files likely to modify

```text
src/llama-model-loader.cpp
src/llama-model.cpp
src/llama.cpp
src/llama-impl.h
```

### Internal model structure

Add something like:

```cpp
struct llama_outlier_block_info {
    ggml_tensor * idx;       // [2, n_blocks], i32
    ggml_tensor * values;    // [32, n_blocks], bf16

    int64_t n_blocks;
    int64_t block_size;      // 32

    // Runtime-optimized layout:
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> block_col;
};
```

In the model object:

```cpp
std::unordered_map<ggml_tensor *, llama_outlier_block_info> outlier_by_tensor;
std::unordered_map<std::string, llama_outlier_block_info> outlier_by_name;
```

### Loader validation

When loading a model:

1. Read normal Q8_0 base tensors.
2. Read sidecar metadata.
3. Locate sidecar tensors by metadata or suffix.
4. Validate:

```text
base tensor type == GGML_TYPE_Q8_0
index tensor type == GGML_TYPE_I32 or GGML_TYPE_I64
values tensor type == GGML_TYPE_BF16
values first dimension == 32
index shape == [2, n_outlier_blocks]
index count == values block count
row indices are in range
block column indices are in range
protected base blocks are zero, if using full mode
```

5. Sort sidecar blocks by row and block column.
6. Build CSR-style row pointers for faster matmul.

Suggested CSR representation:

```text
row_ptr:   [n_rows + 1]
block_col: [n_outlier_blocks]
values:    [n_outlier_blocks, 32]
```

---

## Runtime graph changes

Centralize weight matmul calls behind a helper.

Example:

```cpp
static ggml_tensor * llama_mul_mat_weight(
    llama_context_build & lctx,
    ggml_tensor * w,
    ggml_tensor * x
) {
    ggml_tensor * y = ggml_mul_mat(lctx.ctx0, w, x);

    if (lctx.model.has_outlier_blocks(w)) {
        const auto & ob = lctx.model.get_outlier_blocks(w);

        ggml_tensor * corr = ggml_mul_mat_outlier_blocks(
            lctx.ctx0,
            ob.idx,
            ob.values,
            x,
            w->ne[1],
            w->ne[0]
        );

        y = ggml_add(lctx.ctx0, y, corr);
    }

    return y;
}
```

Then replace direct weight matmuls in attention and feed-forward layers with this helper.

This is important because `llama-cli` and `llama-server` both rely on the same libllama graph-building path. If the feature is implemented in libllama, both frontends inherit support.

---

## New GGML operation

Add a new operation:

```c
GGML_OP_MUL_MAT_OUTLIER_BLOCKS
```

Conceptual signature:

```text
out = mul_mat_outlier_blocks(idx, values, x, n_rows, n_cols)
```

Where:

```text
idx:    [2, n_outlier_blocks]
values: [32, n_outlier_blocks]
x:      [n_cols, n_tokens]
out:    [n_rows, n_tokens]
```

For every protected block:

```text
row = idx[0, k]
block_col = idx[1, k]
col0 = block_col * 32

for token in tokens:
    out[row, token] += dot(values[:, k], x[col0:col0+32, token])
```

### CPU reference kernel

Implement CPU first.

Use CSR layout to avoid atomics:

```text
row_ptr[row]..row_ptr[row + 1] gives protected blocks for that output row
```

Pseudo-code:

```cpp
for row in parallel range(n_rows):
    for token in range(n_tokens):
        float acc = 0.0f;

        for p in range(row_ptr[row], row_ptr[row + 1]):
            int col0 = block_col[p] * 32;
            const bf16 * w = &values[p * 32];
            const float_or_f16 * xv = &x[col0, token];

            acc += dot_32_bf16_activation(w, xv);

        out[row, token] = acc;
```

Then the graph does:

```text
normal_q8_output + sparse_bf16_correction
```

### Initial CPU implementation constraints

Start simple:

```text
support contiguous tensors first
support common activation layouts first
assert unsupported layouts clearly
add general strided support after correctness is proven
```

---

## Backend support strategy

llama.cpp supports many backends. Do not try to implement all GPU kernels in the first patch.

### Phase 1: CPU only

Implement:

```text
CPU quantized base matmul
CPU sparse BF16 outlier correction
```

Make `llama-cli` and `llama-server` work on CPU.

### Phase 2: safe GPU offload behavior

If a GPU backend does not support the sparse outlier correction op, do **not** silently ignore the correction.

Possible behavior:

```text
Option A:
    keep affected layers on CPU

Option B:
    refuse the offload configuration with a clear error
```

A warning should look like:

```text
warning: tensor blk.12.ffn_down.weight uses Q8_0_BF16_OUTLIER;
         selected backend does not support sparse outlier correction;
         keeping this layer on CPU
```

### Phase 3: CUDA and HIP

Add kernels using CSR sidecar layout:

```text
row_ptr:   [n_rows + 1]
block_col: [n_outlier_blocks]
values:    [n_outlier_blocks, 32]
```

Launch over:

```text
rows x token tiles
```

For each row/token tile:

```text
acc[token] = 0
for each outlier block in row:
    load 32 BF16 weights
    load corresponding 32 activations for token tile
    accumulate
write correction output
```

Initially keep this as a separate correction kernel. Fuse it with Q8_0 matmul later only if benchmarks show launch overhead is significant.

### Phase 4: Metal

Implement a Metal correction kernel for Apple Silicon.

### Phase 5: Vulkan, SYCL, and others

Add additional kernels only after the CPU/CUDA/HIP/Metal path proves useful.

---

## Requantization and conversion behavior

### Input is BF16, F16, or F32

Normal path:

```text
source tensor -> Q8_0 base tensor + BF16 outlier sidecars
```

### Input already has outlier sidecars

For `COPY`:

```text
preserve base tensor, metadata, and sidecars exactly
```

For requantization to plain `Q8_0`:

```text
reconstruct protected blocks from BF16 sidecars
then quantize all blocks normally to Q8_0
```

For requantization to another outlier format:

```text
reconstruct full source values
rerun outlier detection
write new base + new sidecars
```

### Split GGUF files

Ensure `gguf-split` keeps sidecar tensors in the same shard as the base tensor when possible.

If that is not possible, metadata must be sufficient for the loader to find sidecars across split files.

---

## API changes

Extend `llama_model_quantize_params` by appending fields at the end to preserve ABI as much as possible:

```c
typedef enum llama_q8_outlier_store {
    LLAMA_Q8_OUTLIER_STORE_FULL,
    LLAMA_Q8_OUTLIER_STORE_DELTA,
} llama_q8_outlier_store;

struct llama_model_quantize_params {
    // existing fields...

    bool q8_outlier_enable;
    float q8_outlier_ratio;
    float q8_outlier_nonmax_rel_rmse;
    float q8_outlier_max_frac;
    llama_q8_outlier_store q8_outlier_store;
    const char * q8_outlier_report_path;
};
```

Update:

```c
llama_model_quantize_default_params()
```

Suggested defaults:

```text
q8_outlier_enable = false
q8_outlier_ratio = 16.0
q8_outlier_nonmax_rel_rmse = 0.01
q8_outlier_max_frac = 0.02
q8_outlier_store = LLAMA_Q8_OUTLIER_STORE_FULL
q8_outlier_report_path = nullptr
```

Only activate the feature when the user explicitly asks for:

```text
Q8_0_BF16_OUTLIER
```

or passes:

```text
--outlier-blocks bf16
```

---

## Testing plan

### Unit tests for outlier detection

Test a block with one extreme value:

```text
x[0..30] = values around 0.01
x[31]    = 100.0
```

Expected:

```text
block is protected
base Q8_0 block is zeroed
sidecar contains original BF16 values
```

Test a block where all values are large:

```text
x[0..31] = values around 100.0
```

Expected:

```text
block is not protected
```

Test a block where one value is large but the non-outlier RMSE is low:

```text
ratio high
nonmax_rel_rmse below threshold
```

Expected:

```text
block is not protected
```

Test cap behavior:

```text
candidates exceed outlier_max_frac
```

Expected:

```text
only highest-scoring candidates are protected
```

### GGUF write/read tests

After quantization, inspect the generated GGUF:

```text
base tensor exists and is Q8_0
outlier_idx exists and has expected shape
outlier_bf16 exists and has expected shape
metadata references the sidecars
protected base blocks are zeroed
```

### Matmul correctness tests

For random tensors:

```text
dense_reference = matmul using reconstructed mixed tensor
hybrid_output = Q8_0 base matmul + sparse BF16 correction
```

Expected:

```text
hybrid_output matches dense_reference within tolerance
```

Test batch sizes:

```text
n_tokens = 1
n_tokens = 8
n_tokens = 128
```

Test shapes representative of:

```text
attention q/k/v/o weights
ffn gate/up/down weights
embedding-like matrices
output tensor
```

### Runtime smoke tests

Run:

```bash
./llama-cli -m model-Q8_0_BF16_OUTLIER.gguf -p "Hello"
```

Run server:

```bash
./llama-server -m model-Q8_0_BF16_OUTLIER.gguf
```

Send a small request to the server and verify generation succeeds.

### Backend tests

CPU:

```text
must pass correctness and smoke tests
```

GPU backend without native outlier support:

```text
must keep affected layers on CPU or fail clearly
must not silently ignore sidecars
```

GPU backend with native outlier support:

```text
must match CPU output within tolerance
```

### Quality tests

Compare:

```text
BF16 baseline
plain Q8_0
Q8_0_BF16_OUTLIER at 0.1%, 0.5%, 1%, 2%, 5% protected blocks
```

Report:

```text
perplexity
model size
estimated bpw
prompt eval tokens/s
generation tokens/s
load time
```

---

## Documentation updates

Update `tools/quantize/README.md` with a new quantization type:

```text
Q8_0_BF16_OUTLIER
    Mostly Q8_0, but selected 32-weight blocks are stored as BF16 sidecars
    when an outlier value would distort the Q8_0 block scale.
```

Example:

```bash
./llama-quantize \
  --outlier-blocks bf16 \
  --outlier-ratio 16 \
  --outlier-nonmax-rel-rmse 0.01 \
  --outlier-max-frac 0.02 \
  model-bf16.gguf model-q8-outlier.gguf Q8_0_BF16_OUTLIER
```

Warnings to document:

```text
Older llama.cpp builds cannot load this format.
Some GPU backends may fall back to CPU for affected tensors until native kernels are implemented.
This format is useful only when protected blocks are rare.
```

---

## Implementation phases

### Phase A: quantizer-only prototype

Goal: produce GGUF files with Q8_0 base tensors and BF16 sidecars.

Tasks:

```text
1. Add CLI flags.
2. Add outlier detection.
3. Add protected block candidate scoring.
4. Apply max protected fraction cap.
5. Write zeroed Q8_0 blocks for protected blocks.
6. Write outlier_idx sidecar tensors.
7. Write outlier_bf16 sidecar tensors.
8. Write GGUF metadata.
9. Emit JSON report.
10. Add tests that inspect GGUF contents.
```

### Phase B: CPU runtime support

Goal: `llama-cli` and `llama-server` work on CPU.

Tasks:

```text
1. Loader detects outlier sidecars.
2. Loader validates sidecar metadata and shapes.
3. Internal model maps base tensor to outlier sidecars.
4. Build CSR-style sidecar layout.
5. Add GGML sparse BF16 outlier matmul operation.
6. Add CPU reference kernel.
7. Route weight matmuls through a helper.
8. Add correctness tests.
9. Add llama-cli smoke test.
10. Add llama-server smoke test.
```

### Phase C: offload-safe behavior

Goal: models never silently produce wrong output with GPU offload.

Tasks:

```text
1. Add backend capability check for sparse outlier correction.
2. If unsupported, keep affected tensors or layers on CPU.
3. Alternatively, fail with a clear error if mixed placement is not safe.
4. Print clear warning.
5. Add tests for -ngl behavior.
```

### Phase D: CUDA and HIP support

Goal: practical GPU performance on NVIDIA and AMD.

Tasks:

```text
1. Upload CSR sidecar metadata to GPU.
2. Upload BF16 sidecar values to GPU.
3. Add sparse block correction kernel.
4. Integrate the correction kernel into graph execution.
5. Benchmark fused vs unfused correction.
6. Add correctness tests against CPU.
```

### Phase E: Metal support

Goal: practical support on Apple Silicon.

Tasks:

```text
1. Add Metal sparse block correction kernel.
2. Validate BF16 support or choose an internal F16 fallback.
3. Add tests against CPU.
4. Benchmark with different protected block fractions.
```

### Phase F: upstream hardening

Goal: make the change reviewable and maintainable.

Tasks:

```text
1. Split into small PRs:
   - quantizer detection and reporting
   - GGUF sidecar metadata
   - CPU runtime support
   - backend support
2. Mark the format experimental at first.
3. Keep standard Q8_0 untouched.
4. Add docs and migration notes.
5. Add compatibility errors for older or unsupported paths.
```

---

## Main risks and mitigations

### Risk 1: storage overhead

Protected blocks are expensive because the base Q8_0 block still exists and the BF16 sidecar is additional data.

Mitigation:

```text
keep default max protected fraction low
print estimated bpw
allow user-defined bpw or fraction caps
```

### Risk 2: sparse correction performance

A separate sparse correction matmul may add overhead, especially for small batch sizes or GPU inference.

Mitigation:

```text
start with correctness
sort sidecars by row
use CSR format
cap protected block density
fuse correction with Q8_0 matmul only after profiling
```

### Risk 3: backend complexity

Every backend has its own matmul machinery.

Mitigation:

```text
CPU first
safe fallback or clear error for unsupported GPU backends
add CUDA/HIP/Metal incrementally
```

### Risk 4: graph integration bugs

llama.cpp supports many architectures and many weight tensors.

Mitigation:

```text
centralize all weight matmul through one helper
add model-architecture coverage tests
validate that every sidecar-backed tensor is actually used with correction
```

### Risk 5: compatibility

Older readers will not understand the sidecars.

Mitigation:

```text
new quantization/file type
explicit metadata version
clear loader error
never overload standard Q8_0 semantics
```

---

## Final intended user experience

Quantize:

```bash
./llama-quantize \
  --outlier-blocks bf16 \
  --outlier-ratio 16 \
  --outlier-nonmax-rel-rmse 0.01 \
  --outlier-max-frac 0.02 \
  model-bf16.gguf model-q8-outlier.gguf Q8_0_BF16_OUTLIER
```

Run CLI:

```bash
./llama-cli -m model-q8-outlier.gguf -p "Explain block quantization."
```

Run server:

```bash
./llama-server -m model-q8-outlier.gguf
```

The resulting model should execute as:

```text
output = normal_q8_0_matmul_with_zeroed_protected_blocks
       + sparse_bf16_matmul_for_protected_blocks
```

This implements block-level exclusion from quantization while keeping ordinary `Q8_0` unchanged.
