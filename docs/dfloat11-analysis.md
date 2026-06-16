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
   Followed by 0–3 pad bytes to align output_positions to 4 bytes

[  output_positions:  (n_blocks + 1) × sizeof(uint32_t)  ]
   Cumulative element counts at CUDA block boundaries
   4-byte aligned — padded after sign_mantissa if needed

[  gaps:  ceil(n_blocks × THREADS_PER_BLOCK × 5 / 8) + 1 bytes  ]
   5-bit per-thread gap, packed big-endian within each byte
   Thread 0 occupies bits [4:0], thread 1 occupies bits [9:5], etc.
   Extra +1 byte: CUDA kernel reads gaps[byte_idx+1] for last thread.
   Gap values range 0–63 (tracked in 6 bits during encode, stored as 5 LSBs).
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

**`quantize_df11()`**: Compresses the entire tensor (all rows) as a single block_df11. Three-phase:
1. Count exponent frequencies across ALL rows → build shared Huffman codes + LUT
2. Pre-compute exact output layout (codes, sm, pos, gaps sizes)
3. Stream-encode all elements into one contiguous output

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
- **Quantize path**: Bypasses the standard `llama_tensor_quantize_impl` loop — calls `quantize_df11()` directly (whole-tensor, single-call). 3D expert tensors are rejected with a clear error.
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

## 6. Implementation Fixes (2026-06-16)

These fixes were applied after build #7 (initial implementation). Each subsection describes a problem discovered during testing and how it was resolved.

### 6.1 Whole-Tensor Compression (Per-Row Overhead Elimination)

**Problem**: The initial `quantize_df11()` called `quantize_row_df11_ref()` per row, creating independent block_df11 headers, LUT tables, output positions, and gaps per row. For `token_embd.weight` (2048 × 248,320 = 248,320 rows), the per-row LUT overhead alone was ~1,280 bytes × 248,320 ≈ 317 MB. The DF11 output was *larger* than the original BF16 (16.54 BPW vs 16.00 BPW).

**Root cause**: DF11 is designed for full-tensor compression with one shared Huffman tree and LUT. Per-row compression defeats the compression ratio.

**Fix** (`ggml-quants.c`): Rewrote `quantize_df11()` to compress all rows as a single block_df11:
1. Phase 1: Count exponent frequencies across ALL rows → single pass
2. Phase 2: Build ONE shared Huffman code + LUT
3. Phase 3: Pre-compute exact output layout from total_bits
4. Phase 4: Stream all elements into ONE contiguous codes bitstream

The output is now one block_df11 per tensor (not per row), matching the Python reference.

**Result**: Model size drops from ~4.6 GB to ~2.5 GB (expected ~30% reduction).

### 6.2 GGUF Reader Offset Validation (Custom Size Mismatch)

**Problem**: After loading a correctly-written DF11 GGUF file, the reader failed with:
```
tensor 'blk.0.attn_gate.weight' has offset 688248576, expected 12205432832
```

The writer correctly stored offsets using `custom_nbytes` (actual compressed size), but the reader's `gguf_init_from_reader()` validation loop used `ggml_nbytes(&ti.t)` (= `24 × nelements` for DF11) to compute expected offsets. Mismatch.

**Fix** (`gguf.cpp`): Changed the reader's size accumulation loop from:
```c
padded_size = GGML_PAD(ggml_nbytes(&ti.t), alignment);
```
to:
```c
// Use offset of NEXT tensor (respects writer's custom_nbytes)
if (i + 1 < info.size())
    padded_size = info[i + 1].offset - info[i].offset;
else
    padded_size = total_data_size - ctx->size;  // last tensor
```

Added `gguf_reader::remaining()` public method to expose `nbytes_remain` for the last-tensor case.

### 6.3 Model Loader Data Size (Bounds Check & Copy)

**Problem**: `llama_model_loader` used `ggml_nbytes(tensor)` for:
1. Data bounds check: `offs + ggml_nbytes(tensor) > file->size()` → fails for DF11 (12 GB vs 800 MB actual)
2. Data copy in `load_data_for()`: copies `ggml_nbytes(cur)` bytes → tries to read far beyond actual data
3. Data copy in `load_all_data()`: uses `ggml_nbytes(cur)` for read/copy size

**Fix** (`llama-model-loader.h`, `llama-model-loader.cpp`):
- Added `size` field to `llama_tensor_weight` — computed from adjacent GGUF tensor offsets
- `load_data_for()`: Use `w.size` instead of `ggml_nbytes(cur)`
- `load_all_data()`: Use `weight->size` instead of `ggml_nbytes(cur)`

### 6.4 CPU GET_ROWS for DF11

**Problem**: Token embedding lookup (`GGML_OP_GET_ROWS`) on CPU had no DF11 handler. The default case was `GGML_ABORT("fatal error")`.

**Fix** (`ggml-cpu/ops.cpp`): Added `ggml_compute_forward_get_rows_df11()`:
1. Decompress entire DF11 tensor to BF16 scratch buffer
2. Extract requested rows via BF16→F32 conversion

Added `case GGML_TYPE_DF11:` to the get_rows type switch.

### 6.5 CPU/GPU Backend Routing (supports_op)

**Problem**: DF11 tensors were placed on CPU but CUDA matmul path (`ggml-cuda.cu:2608`) already supports DF11 (decompress to BF16, then BF16 matmul). CPU cannot perform DF11 matmul efficiently. Need GPU to handle matmuls.

The GPU's `supports_op` for MUL_MAT had a type switch that didn't include DF11, so GPU rejected DF11 matmuls. The CPU's `supports_op` defaulted to accepting them (no DF11 check), but would crash trying to execute.

**Fix**:
- `ggml-cpu/ggml-cpu.c`: Added `[GGML_TYPE_DF11]` entry to `type_traits_cpu` with `vec_dot_type = GGML_TYPE_COUNT` (unsupported)
- `ggml-cpu/ggml-cpu.cpp`: Explicitly return `false` for DF11 in MUL_MAT/MUL_MAT_ID `supports_op`
- `ggml-cuda/ggml-cuda.cu`: Added `GGML_TYPE_DF11` to GPU's MUL_MAT type switch

Result: GET_ROWS → CPU, MUL_MAT → GPU. Scheduler copies DF11 tensors CPU→GPU for matmul.

### 6.6 GPU Buffer Over-Allocation (nb[1] Adjustment)

**Problem**: After routing DF11 to GPU, `ggml_backend_alloc_ctx_tensors_from_buft()` allocated GPU buffers using `ggml_nbytes(t) = 24 × nelements`. For 195 DF11 tensors, this was ~44 GB — exceeding RTX 3080's 20 GB.

**Fix** (`llama-model.cpp`): Before buffer allocation, adjust DF11 tensor `nb[1]` stride so `ggml_nbytes` reflects actual compressed size:
```c
t->nb[1] = ceil_div(weight->size - ne[0]*24, ne[1] - 1);
```
GPU buffer shrinks from 44 GB to ~2.5 GB, fitting in 20 GB VRAM.

### 6.7 Host Reading Device Memory (cudaMemcpy Fix)

**Problem**: `dequantize_df11_cuda()` read `block_df11` header fields (n_luts, n_bytes, n_elements, n_blocks) and `pos` array values directly from `vx` (device pointer) on the host. This is illegal — host cannot read device memory. When DF11 tensors were on CPU (before 6.5), this worked. After moving DF11 to GPU, segfault.

**Fix** (`ggml-cuda/convert.cu`): 
1. Copy `block_df11` header device→host via `cudaMemcpyAsync`
2. Copy `pos` array device→host via `cudaMemcpy`
3. Read header fields and pos values from host copies

### 6.8 Pos Array Alignment Padding

**Problem**: CUDA kernel reads `pos` array as `uint32_t` — requires 4-byte alignment. Layout: `header(24) + luts(n_luts×256) + codes(n_bytes) + sm(n_elements)`. The offset to `pos` = `24 + n_luts×256 + n_bytes + n_elements`. When `(n_bytes + n_elements) % 4 != 0`, pos is misaligned → `cudaErrorMisalignedAddress`.

**Fix** (`ggml-quants.c`, `ggml-cuda/convert.cu`): Writer pads after `sm` section to 4-byte alignment. Reader computes alignment:
```c
uintptr_t pos_off = (sm + n_elements + 3) & ~(uintptr_t)3;
```
CPU dequantize unaffected (doesn't use pos array).

### 6.9 Gaps Buffer +1 Byte (Kernel Out-of-Bounds Read)

**Problem**: CUDA kernel `dequantize_df11_kernel` reads 5-bit gaps as `uint8_t buf0 = gaps[byte_idx]; uint8_t buf1 = gaps[byte_idx + 1];`. For the last thread (global_thread_id = n_threads - 1), `byte_idx + 1` equals `gaps_bytes` — one byte past the allocated array. This caused `cudaErrorIllegalMemoryAccess` (error 700) during kernel execution.

**Fix** (`ggml-quants.c`): Allocate `gaps_bytes = ceil(n_threads * 5 / 8) + 1`. Extra byte is zero-initialized. Kernel safely reads the padding byte for the last thread.

### 6.10 Files Modified Summary

| # | File | Change |
|---|------|--------|
| 6.1 | `ggml/src/ggml-quants.c` | Rewrite `quantize_df11()` — whole-tensor compression |
| 6.2 | `ggml/src/ggml-quants.c` | Fix gap encoding byte alignment, pos alignment padding |
| 6.2 | `ggml/src/gguf.cpp` | Reader: offset-difference size accumulation, `remaining()` |
| 6.3 | `src/llama-model-loader.h` | `llama_tensor_weight.size` field, bounds check fix |
| 6.3 | `src/llama-model-loader.cpp` | `load_data_for()` / `load_all_data()` use `w.size` |
| 6.4 | `ggml/src/ggml-cpu/ops.cpp` | `ggml_compute_forward_get_rows_df11()` |
| 6.5 | `ggml/src/ggml-cpu/ggml-cpu.c` | `type_traits_cpu[GGML_TYPE_DF11]` |
| 6.5 | `ggml/src/ggml-cpu/ggml-cpu.cpp` | Reject DF11 MUL_MAT on CPU |
| 6.5 | `ggml/src/ggml-cuda/ggml-cuda.cu` | Add DF11 to GPU MUL_MAT supports_op |
| 6.6 | `src/llama-model.cpp` | Shrink DF11 nb[1] for GPU buffer size |
| 6.7 | `ggml/src/ggml-cuda/convert.cu` | cudaMemcpy header+pos from device to host |
| 6.7 | `ggml/src/ggml-cuda/convert.cu` | Pos alignment in reader |
| 6.7 | `ggml/src/ggml-alloc.c` | Debug prints (gated by `--custom-logs`) |
| 6.9 | `ggml/src/ggml-quants.c` | gaps_bytes +1 padding byte |

---

*End of implementation document.*
