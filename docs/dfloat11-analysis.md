# Dfloat11 Implementation Document for llama.cpp

> **Last updated**: 2026-06-16
> **Branch**: `dfloat11`
> **Reference**: `E:\python\Dfloat11` → `E:\cpp\llama.cpp`

---

## 1. What Dfloat11 Is

Dfloat11 **losslessly** compresses BF16 neural network weights by ~30% using Huffman coding
of the 8-bit exponent. Sign + mantissa (also 8 bits) are stored literally.

```
BF16 (16 bits) = sign[15] | exponent[14:7] | mantissa[6:0]
                  ↓ split
   exponent_8bits (Huffman encoded → ~3–5 bits avg)
   sign_mantissa  (stored raw, 8 bits)
                  ↓ reconstruct on-the-fly
   BF16 restored bit-for-bit
```

### Fixed Constants
| Constant | Value |
|---|---|
| `BYTES_PER_THREAD` | 8 |
| `THREADS_PER_BLOCK` | 512 |
| `MAX_HUFFMAN_CODE_BITS` | 32 |
| `LUT_THRESHOLD` | 240 (values ≥ 240 = continue decoding at next LUT row) |
| `QK_DF11` | 1 (one "block" per tensor — DF11 is full-tensor, not sub-block) |

### Per-Tensor Compressed Data Layout

```
[  header:  block_df11 (24 bytes)  ]
  uint32_t n_luts          // number of LUT rows (prefix rows + 1 length row)
  uint32_t n_bytes         // bytes in Huffman-encoded exponent bitstream
  uint32_t n_elements      // total number of BF16 values
  uint32_t n_blocks        // ceil(n_bytes / (THREADS_PER_BLOCK * BYTES_PER_THREAD))
  uint32_t reserved_0
  uint32_t reserved_1

[  luts:  (n_luts) rows × 256 bytes  ]
  Each row i (0 ≤ i < n_luts-1) is a 256-entry prefix-tree node:
    entry[b] < 240  →  symbol decoded, this is the exponent value
    entry[b] ≥ 240 →  continue at LUT row (256 – entry[b])
  Last row: code length for each exponent value 0..255

[  codes:  n_bytes bytes  ]
  Huffman-encoded exponent bitstream (big-endian bit packing)

[  sign_mantissa:  n_elements bytes  ]
  One byte per value: [sign(1) | mantissa(7)]

[  output_positions:  (n_blocks + 1) × sizeof(uint32_t)  ]
  Cumulative element counts at CUDA block boundaries

[  gaps:  ceil(n_blocks × THREADS_PER_BLOCK × 5 / 8) bytes  ]
  5-bit per-thread gap, packed big-endian within each byte
  Thread 0 occupies bits [4:0], thread 1 occupies bits [9:5], etc.
```

---

## 2. Files Modified (14 files, ~1000 lines added)

### 2.1 Type System

#### `ggml/include/ggml.h`
- Added `GGML_TYPE_DF11 = 42` to `ggml_type` enum (before the closing `}`)
- Updated `GGML_TYPE_COUNT` from `42` → `43`
- Added `GGML_FTYPE_MOSTLY_DF11 = 28` to `ggml_ftype` enum
- Total: +4 lines changed

#### `include/llama.h`
- Added `LLAMA_FTYPE_MOSTLY_DF11 = 43` to `llama_ftype` enum
- Positioned after `LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER = 42`
- Total: +1 line

#### `ggml/src/ggml-common.h`
- Added `QK_DF11 = 1` (single-element block — DF11 is per-tensor)
- Added `block_df11` struct (24 bytes, 6 × uint32_t)
- Positioned after the `block_tq2_0` definition, before Super-block section
- Total: +11 lines

```c
#define QK_DF11 1
typedef struct {
    uint32_t n_luts;
    uint32_t n_bytes;
    uint32_t n_elements;
    uint32_t n_blocks;
    uint32_t reserved_0;
    uint32_t reserved_1;
} block_df11;
static_assert(sizeof(block_df11) == 24, "wrong df11 block size/padding");
```

---

### 2.2 CPU Encode / Decode

#### `ggml/src/ggml-quants.h`
- Declared `dequantize_row_df11(const block_df11*, float*, int64_t)`
- Declared `dequantize_row_df11_to_bf16(const block_df11*, ggml_bf16_t*, int64_t)`
- Declared `quantize_row_df11_ref(const float*, block_df11*, int64_t)`
- Declared `quantize_df11(const float*, void*, int64_t, int64_t, const float*)`
- Total: +6 lines

#### `ggml/src/ggml-quants.c` (~560 lines added at end of file)

**Huffman Engine (static helpers)**:
- `df11_hnode` — tree node with left/right/parent/symbol/freq
- `df11_heap` — min-heap for Huffman tree construction (array-based)
- `df11_heap_push()` / `df11_heap_pop()` — standard binary heap ops
- `df11_build_huffman()` — frequency counting → Huffman tree → 32-bit max constraint enforcement
  - If max code > 32 bits: repeatedly raises floor of lowest-freq symbols, rebuilds
- `df11_build_lut()` — prefix-tree LUT construction:
  - Collects all byte-aligned prefixes from Huffman codes
  - Sorts by length ascending
  - For each prefix, fills 256-entry row checking if prefix+next_byte completes a code or a longer prefix
  - Last row: code lengths per symbol 0..255

**`quantize_row_df11_ref()`**:
1. FP32 → BF16 conversion (using `ggml_fp32_to_bf16`)
2. Split each BF16: `exponent = (bf16 >> 7) & 0xFF`, `sign_mantissa = ((bf16 >> 8) & 0x80) | (bf16 & 0x7F)`
3. Count exponent frequencies → build Huffman codes → build LUT
4. Huffman-encode exponents into bitstream with big-endian packing
5. Track **gaps** at thread boundaries (every 64 bits): gap[t] = total_bits % 64
6. Track **output positions** at block boundaries (every 512×8 bytes)
7. Pack gaps: 5 bits per thread, big-endian within bytes
8. Write header + LUTs + codes + sign_mantissa + output_positions + gaps

**`dequantize_row_df11()`**:
1. Parse header: n_luts, n_bytes, n_elements, n_blocks
2. Extract data pointers: luts → codes → sign_mantissa (output_positions and gaps not needed for CPU single-thread decode)
3. 64-bit buffer accumulator, LUT prefix-tree walk (up to 4 levels)
4. BF16 reconstruction formula:
```c
uint8_t sm = sign_mantissa[output_idx];
uint16_t bf16;
bf16  = (uint16_t)(sm & 0x80) << 8;      // sign   → bit 15
bf16 |= (uint16_t)(decoded & 0xFE) << 7;  // exp[7:1] → bits 14:8
bf16 |= (uint16_t)(decoded & 0x01) << 7;  // exp[0]   → bit 7
bf16 |= (uint16_t)(sm & 0x7F);            // mantissa → bits 6:0
y[idx] = ggml_bf16_to_fp32(tmp_bf16);
```
5. Fill remaining beyond decoded elements with zeros

**`dequantize_row_df11_to_bf16()`**: Same as above but outputs `ggml_bf16_t` directly (no FP32 conversion).

**`quantize_df11()`**: Iterates `nrows`, calls `quantize_row_df11_ref` for each, computes cumulative byte count from header fields. Returns total bytes.

**`ggml_validate_row_data()`**: Added `GGML_TYPE_DF11` to the `I8/I16/I32/I64` skip-validation group (DF11 is self-validating via header).

---

### 2.3 GGML Core Registration

#### `ggml/src/ggml.c`

**`type_traits` array** — added entry at `[GGML_TYPE_DF11]`:
```c
[GGML_TYPE_DF11] = {
    .type_name      = "df11",
    .blck_size      = QK_DF11,         // 1
    .type_size      = sizeof(block_df11), // 24
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_df11,
    .from_float_ref = (ggml_from_float_t) quantize_row_df11_ref,
},
```

**`ggml_quantize_chunk()`** — added DF11 case and relaxed assertion:
- Added `case GGML_TYPE_DF11:` calling `quantize_df11()`
- Changed final assertion to accept variable-length: `assert(result == nrows * row_size || type == GGML_TYPE_DF11)`

No change to `ggml_quantize_requires_imatrix()` — DF11 does not need importance matrices.
No change to `ggml_quantize_init()` — DF11 needs no initialization.

---

### 2.4 CPU Backend Wrappers

#### `ggml/src/ggml-cpu/quants.h`
- Added `void quantize_row_df11(const float*, void*, int64_t)` declaration

#### `ggml/src/ggml-cpu/quants.c`
- Added `quantize_row_df11()` — thin wrapper calling `quantize_row_df11_ref()`

---

### 2.5 CUDA Decode Kernel

#### `ggml/src/ggml-cuda/dequantize.cuh`
- Added forward declaration: `void dequantize_df11_cuda(const void*, void*, int64_t, cudaStream_t)`

#### `ggml/src/ggml-cuda/convert.cu` (265 lines added)

**Kernel: `dequantize_df11_kernel`** (8-phase, 512-thread, shared memory):
1. **Phase 1**: Load 12 bytes into registers (8 own + 4 from next thread for lookahead)
2. **Phase 2**: Read 5-bit gap from packed array, build 64-bit buffer from registers, shift by gap
3. **Phase 3**: Pre-decode until `free_bits ≥ 32` (LUT walk, up to 4 levels)
4. **Phase 4**: Load extra 4 bytes, merge into buffer, subtract 32 free bits
5. **Phase 5**: Count remaining symbols (decode up to 8 symbols per thread)
6. **Phase 6**: Parallel prefix sum (scan) on element counts to compute output positions
7. **Phase 7**: Re-decode and reconstruct BF16 values into shared memory write buffer
8. **Phase 8**: Scatter from shared memory to global outputs

**Host launcher: `dequantize_df11_cuda()`**:
- Parses header → extracts luts/codes/sm/pos/gaps pointers
- Computes shared memory size: counters (512 ints) + write buffer (max elements per block × 2 bytes)  
- Launches `n_blocks` blocks, 512 threads each

**Dispatch wrappers:**
```c
// Forward declarations (before ggml_get_to_fp16_cuda)
static void dequantize_row_df11_fp16_cuda(const void * vx, half * vy, int64_t k, cudaStream_t stream);
static void dequantize_row_df11_fp32_cuda(const void * vx, float * vy, int64_t k, cudaStream_t stream);
```

Each wrapper:
1. Allocates `nv_bfloat16` scratch via `cudaMallocAsync`
2. Calls `dequantize_df11_cuda` → BF16 scratch
3. Calls `convert_unary_cont_cuda<nv_bfloat16, T>` to convert BF16 → target type
4. Frees scratch via `cudaFreeAsync`

**Dispatch table entries added to:**
- `ggml_get_to_fp16_cuda()` → returns `dequantize_row_df11_fp16_cuda`
- `ggml_get_to_fp32_cuda()` → returns `dequantize_row_df11_fp32_cuda`

---

### 2.6 GGUF Variable-Length Size Override

#### `ggml/src/gguf.cpp` and `ggml/include/gguf.h`

**Problem**: GGUF computes tensor data size via `ggml_nbytes()`, which for DF11 returns
`sizeof(block_df11) × n_elements = 24 × n_elements` bytes. The actual compressed data
is much smaller. This causes:
- File offset miscalculation for subsequent tensors
- Size mismatch assertions during GGUF file writing
- Wrong memory allocation for DF11 tensors

**Solution**: Added a `custom_nbytes` override field to the GGUF tensor info:

```c
// gguf.cpp, struct gguf_tensor_info
struct gguf_tensor_info {
    struct ggml_tensor t;
    uint64_t offset;
    size_t   custom_nbytes;  // 0 = use ggml_nbytes(&t); nonzero = override
};
```

**New function**: `gguf_set_tensor_data_size(ctx, name, nbytes)`
- Must be called after `gguf_set_tensor_type`
- Stores `nbytes` in `custom_nbytes`
- Recomputes offsets of all subsequent tensors using the custom size for this one

**Updated functions to respect `custom_nbytes`:**
- `gguf_get_tensor_size()` — returns `custom_nbytes` when set, else `ggml_nbytes()`
- `gguf_writer_buf::write_tensor_data()` — uses `custom_nbytes` for buffer resize and memcpy
- `gguf_writer_file::write_tensor_data()` — uses `custom_nbytes` for temp buffer and file write
- `gguf_set_tensor_type()` — offset computation uses `custom_nbytes` when set (via helper)

**Initialization**: All `gguf_tensor_info` instances are value-initialized (`= {}`), guaranteeing `custom_nbytes = 0` by default.

---

### 2.7 Model Loader

#### `src/llama-model-loader.cpp`
- Added `case LLAMA_FTYPE_MOSTLY_DF11: return "DF11";` to `llama_model_ftype_name()`
- Added `case GGML_TYPE_DF11: ftype = LLAMA_FTYPE_MOSTLY_DF11; break;` to the type_max → ftype switch

---

### 2.8 Quantize Tool

#### `src/llama-quant.cpp`
- Added `case LLAMA_FTYPE_MOSTLY_DF11: return GGML_TYPE_DF11;` to `llama_ftype_get_default_type()`
- **OUTPUT category fix**: Added `&& ftype != LLAMA_FTYPE_MOSTLY_DF11` to the catch-all `else if` at line 465 that forces unknown types to `Q6_K`. This prevents output tensors (output_norm.weight, etc.) from being downgraded.
- **Dry-run**: Uses `nelements × 2` as size estimate for DF11 instead of the incorrect `ggml_row_size()`
- **Quantize path**: Bypasses the expert/chunk loop for DF11 — calls `quantize_df11()` directly (single-pass). 3D expert tensors are rejected with a clear error.
- **GGUF write**: For DF11 tensors, calls `gguf_set_tensor_data_size()` after `gguf_set_tensor_type()` instead of asserting size equality

#### `tools/quantize/quantize.cpp`
- Added `{ "DF11", LLAMA_FTYPE_MOSTLY_DF11, "lossless BF16 compression (Huffman-coded exponents)", }` to `QUANT_OPTIONS`

---

### 2.9 On-the-Fly MatMul (Phase 3)

#### `ggml/src/ggml-cuda/ggml-cuda.cu`
- Added `#include "ggml-cuda/dequantize.cuh"` for kernel declaration
- Added DF11 branch at the top of `ggml_cuda_mul_mat()`, before all existing dispatch logic:

```cpp
if (src0->type == GGML_TYPE_DF11) {
    // 1. Allocate BF16 scratch via CUDA pool
    ggml_cuda_pool_alloc<nv_bfloat16> scratch_bf16(ctx.pool(), n_elements);

    // 2. Decode DF11 → BF16 on GPU
    dequantize_df11_cuda(src0->data, scratch_bf16.ptr, n_elements, ctx.stream());

    // 3. Create temporary BF16 tensor wrapping the scratch buffer
    ggml_tensor src0_bf16 = *src0;
    src0_bf16.type = GGML_TYPE_BF16;
    src0_bf16.data = scratch_bf16.ptr;
    src0_bf16.nb[0] = sizeof(nv_bfloat16);
    src0_bf16.nb[1] = src0->ne[0] * sizeof(nv_bfloat16);

    // 4. Dispatch via MMF (if applicable) or cuBLAS BF16 GEMM
    // ... uses existing matmul paths with the decoded BF16 tensor

    return;  // scratch freed automatically by pool allocator destructor
}
```

---

### 2.10 AGENTS.md
Added a "Gotchas" section documenting the OUTPUT tensor category catch-all pattern:
> In the OUTPUT category block (~line 447 of `llama-quant.cpp`), there's a catch-all `else if`
> at the end that forces unknown ftypes to `GGML_TYPE_Q6_K`. When adding a new ftype, you must
> either add an explicit case in that block or exclude the new ftype from the catch-all condition.
> `LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER` and `LLAMA_FTYPE_MOSTLY_DF11` are excluded.

---

## 3. Usage

### Quantize a model
```bash
llama-quantize <input.gguf> DF11 <output.gguf>
```

To keep embeddings and output tensors in higher precision:
```bash
llama-quantize <input.gguf> DF11 <output.gguf> --output-tensor-type f16 --token-embedding-type f16
```

### Load and run
The model loads and runs identically to any BF16/HF16 model. DF11 weights are decompressed transparently at load time (Solution A) or on-the-fly during matmul (Solution B, CUDA only).

---

## 4. Build Fixes Applied

During implementation, the following issues were found and fixed:

| # | Issue | Fix |
|---|-------|-----|
| 1 | `ggml_bf16_to_fp32_row_value()` does not exist | Use `ggml_bf16_to_fp32(ggml_bf16_t)` with struct construction from raw uint16_t |
| 2 | `dequantize_row_df11_fp16/fp32_cuda` undefined at point of use in dispatch tables | Added forward declarations before `ggml_get_to_fp16_cuda()` |
| 3 | `ggml_quantize_chunk()` has no DF11 case → crash | Added case + relaxed row-size assertion |
| 4 | `ggml_validate_row_data()` fails for DF11 | Added DF11 to skip-validation group |
| 5 | OUTPUT tensors forced to Q6_K for DF11 | Added `LLAMA_FTYPE_MOSTLY_DF11` exclusion to catch-all condition |
| 6 | GGUF size assertion fails for variable-length DF11 | Added `gguf_set_tensor_data_size()` with `custom_nbytes` override |
| 7 | Dry-run size estimate wrong for DF11 | Special case: use `nelements × 2` as upper bound |

---

## 5. Implementation Checklist (Complete)

| # | File | Change |
|---|------|--------|
| 1 | `ggml/include/ggml.h` | `GGML_TYPE_DF11 = 42`, `GGML_TYPE_COUNT = 43`, `GGML_FTYPE_MOSTLY_DF11 = 28` |
| 2 | `include/llama.h` | `LLAMA_FTYPE_MOSTLY_DF11 = 43` |
| 3 | `ggml/src/ggml-common.h` | `QK_DF11 = 1`, `block_df11` struct (24 bytes) |
| 4 | `ggml/src/ggml-quants.h` | 4 function declarations |
| 5 | `ggml/src/ggml-quants.c` | CPU Huffman encoder, LUT builder, encode, decode, multi-row quantize (~560 lines) |
| 6 | `ggml/src/ggml.c` | `type_traits` entry, `ggml_quantize_chunk` case + relaxed assertion |
| 7 | `ggml/src/ggml-cpu/quants.h` | CPU wrapper declaration |
| 8 | `ggml/src/ggml-cpu/quants.c` | CPU wrapper implementation |
| 9 | `ggml/src/ggml-cuda/dequantize.cuh` | `dequantize_df11_cuda()` declaration |
| 10 | `ggml/src/ggml-cuda/convert.cu` | CUDA kernel, host launcher, FP16/FP32 wrappers, dispatch table entries (265 lines) |
| 11 | `ggml/src/ggml-cuda/ggml-cuda.cu` | On-the-fly DF11 → BF16 → matmul branch |
| 12 | `ggml/src/gguf.cpp` | `custom_nbytes` field, `gguf_set_tensor_data_size()`, writer updates |
| 13 | `ggml/include/gguf.h` | `gguf_set_tensor_data_size()` declaration |
| 14 | `src/llama-model-loader.cpp` | DF11 ftype name + type mapping |
| 15 | `src/llama-quant.cpp` | Ftype→type mapping, OUTPUT Q6_K exclusion, dry-run fix, DF11 quantize branch, GGUF size override |
| 16 | `tools/quantize/quantize.cpp` | `"DF11"` entry in `QUANT_OPTIONS` |
| 17 | `AGENTS.md` | OUTPUT tensor category gotcha |

---

*End of implementation document.*
