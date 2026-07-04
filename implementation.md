# IQ2_XXS_V2: Deduplicated Scales + Extended Sub-Block Precision

## Summary

Current IQ2_XXS stores a 16-bit FP16 super-block scale `d` for every block of 256 weights. Analysis of the Qwen3.5-2B IQ2_XXS model shows only 0.14% of scale values are unique (5,546 unique across 4.1M blocks, per tensor <= 3,000 unique). The 16-bit scale wastes 4 bits per block.

Repurpose `d` as a 12-bit LUT index (0-4095, covering all observed per-tensor unique scales) plus a 4-bit extension nibble. The extension nibble provides 1 extra precision bit to up to 4 sub-blocks whose 4-bit scale saturates at `scale_4bit == 15`, extending their effective range from `[0.125d, 4.0d]` to `[0.125d, 8.0d]`.

This is a new ggml type: `GGML_TYPE_IQ2_XXS_V2`. The block struct layout, block size (256 weights), and type size (66 bytes) are identical to IQ2_XXS. Only the interpretation of the `d` field changes, plus a separate per-tensor scale Look Up Table (LUT).

- **CPU and CUDA only.** Other backends (Metal, Vulkan, SYCL, WebGPU) are left unimplemented.
- **No format change to GGUF.** The new type is used only at the ggml tensor level.
- **Existing IQ2_XXS is unchanged.** V2 is an additive type.

### Baseline: IQ2_XXS (Unsloth) on Qwen3.5-2B

Results from the Unsloth UD IQ2_XXS benchmark. Used as quality baseline for V2 comparison.

| Metric | Value |
|---|---|
| KL Divergence | 0.7207 |
| PPL | 26.44 |
| Same top P | 60.19% |
| Model size | 733 MB |

---

## 1. New Type Declaration

### 1.1 Enum entry

**File:** `ggml/include/ggml.h` (line ~430)

```c
// after GGML_TYPE_NVFP4 = 40,
GGML_TYPE_IQ2_XXS_V2 = 42,
GGML_TYPE_COUNT   = 43,  // was 42
```

### 1.2 Block struct (identical to V1, `d` reinterpreted)

**File:** `ggml/src/ggml-common.h` (after line 375)

```c
// V2: same layout as block_iq2_xxs, d is repurposed:
//    bits 0-11:  LUT index into per-tensor scale table
//    bits 12-15: extension nibble (1 extra scale bit per saturated sub-block)
typedef block_iq2_xxs block_iq2_xxs_v2;
```

No new struct needed. Same 66 bytes. The `d` field is accessed as `uint16_t` instead of `ggml_half` in V2 kernels.

### 1.3 Scale LUT storage

Per-tensor, stored in a separate small buffer. At quantization time, the quantizer produces:
- The compressed block data (same layout as IQ2_XXS)
- A scale LUT: `(uint32_t count, ggml_half values[count])`

The LUT is attached to the ggml tensor via `tensor->extra`. During model loading (or quantization), the LUT buffer is malloc'd and stored in `tensor->extra`. All dequant, vec_dot, and mmq kernels read the LUT from `src0->extra`.

**LUT buffer layout (in memory pointed to by `tensor->extra`):**

```
offset 0:  uint16_t n_lut      (number of LUT entries, 16-bit)
offset 2:  uint16_t padding    (alignment)
offset 4:  ggml_half lut[n_lut] (FP16 scale values, 2 bytes each)
```

Maximum LUT size for any observed tensor: ~3,000 entries = 6,008 bytes + 4 header = ~6 KB.

### 1.4 Type registration

**File:** `ggml/src/ggml.c` (after line 796)

```c
[GGML_TYPE_IQ2_XXS_V2] = {
    .type_name                = "iq2_xxs_v2",
    .blck_size                = QK_K,        // 256
    .type_size                = sizeof(block_iq2_xxs),  // 66
    .is_quantized             = true,
    .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs_v2,
    .from_float_ref           = NULL,
},
```

---

## 2. Extension Nibble Protocol

The 4-bit extension nibble (bits 12-15 of `d`) is consumed by sub-blocks whose `scale_4bit == 15` (saturated). During dequant:

```
Let ext = (d >> 12) & 0xF         // extension nibble
Let saturated_index = 0            // count of saturated sub-blocks seen so far

For each sub-block ib in 0..7:
    scale_4bit = (qs[4*ib+3] >> 28) & 0xF   // as in V1

    if scale_4bit == 15:
        ext_bit = (ext >> saturated_index) & 1
        scale_4bit = 16 + ext_bit            // 5-bit: 16 or 17
        saturated_index += 1

    db = lut[d_lut_idx] * (0.5f + scale_4bit) * 0.25f
```

The quantizer guarantees that at most 4 sub-blocks per super-block have `scale_4bit == 15`. If a 5th sub-block would saturate, the quantizer increases `d` (the super-block scale via LUT entry selection) to bring it within range. The extension nibble is populated left-to-right in sub-block order (ib=0 first).

**Dequant formula (unchanged from V1 after scale lookup):**

```
weight = lut[d_idx] * (0.5 + scale_value) * 0.25 * codebook[lattice_idx][j] * sign
```

---

## 3. CPU Pipeline Changes

### 3.1 Dequant kernel

**File:** `ggml/src/ggml-quants.c` (new function, modeled after `dequantize_row_iq2_xxs` at line 2416)

```c
void dequantize_row_iq2_xxs_v2(const void * GGML_RESTRICT x,
                                float * GGML_RESTRICT y, int64_t k, 
                                const void * lut_extra) {
    const uint16_t * header = (const uint16_t *)lut_extra;
    uint16_t n_lut = header[0];
    const ggml_half * lut = (const ggml_half *)(header + 2);

    const int64_t nb = k / QK_K;
    const block_iq2_xxs * blocks = (const block_iq2_xxs *)x;

    // ... same loop structure as V1 dequant ...
    for (int i = 0; i < nb; i++) {
        uint16_t d_raw = blocks[i].d;
        uint16_t d_idx = d_raw & 0x0FFF;
        uint32_t d_ext = d_raw >> 12;

        float d_fp16 = GGML_FP16_TO_FP32(lut[d_idx]);

        int saturated_count = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            // ... read aux32, determine scale_4bit ...
            int scale_4bit = aux32[1] >> 28;
            if (scale_4bit == 15) {
                int ext_bit = (d_ext >> saturated_count) & 1;
                scale_4bit = 16 + ext_bit;
                saturated_count++;
            }

            float db = d_fp16 * (0.5f + scale_4bit) * 0.25f;
            // ... rest identical to V1 (codebook lookup, sign decode) ...
        }
    }
}
```

### 3.2 Dispatch hook in dequant path

**File:** `ggml/src/ggml-cpu/ops.cpp` (in `ggml_compute_forward_dup_from_q`, line ~524)

The `to_float` function pointer for IQ2_XXS_V2 is `dequantize_row_iq2_xxs_v2`. However, the current dispatch signature is:

```c
typedef void (*ggml_to_float_t)(const void * x, float * y, int64_t k);
```

This takes 3 args. The V2 dequant needs 4 args (the LUT). We have two options:

**Option A:** Change `ggml_to_float_t` to 4 args and update ALL types. Too invasive (touches every quant type's dequant function, every backend).

**Option B:** Special-case IQ2_XXS_V2 in `ggml_compute_forward_dup_from_q`. Jump table skips the standard dispatch and calls `dequantize_row_iq2_xxs_v2` directly with the extra arg. This is a 3-line addition:

```c
// in ggml_compute_forward_dup_from_q(), after the type lookup:
if (type == GGML_TYPE_IQ2_XXS_V2) {
    dequantize_row_iq2_xxs_v2(
        (const void *) ((char *) src0->data + x_offset),
        (float *) ((char *) dst->data + dst_offset),
        qk,
        src0->extra);  // LUT buffer
} else {
    dequantize_row_q(...);  // standard path for all other types
}
```

**Choose Option B.**

### 3.3 CPU type traits dispatch

**File:** `ggml/src/ggml-cpu/ggml-cpu.c` (line ~327, after IQ2_XXS entry)

```c
[GGML_TYPE_IQ2_XXS_V2] = {
    .vec_dot      = ggml_vec_dot_iq2_xxs_v2_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
```

The `.vec_dot` function signature is:

```c
typedef void (*ggml_vec_dot_t)(
    int n, float * s, size_t bs, 
    const void * vx, size_t bx, 
    const void * vy, size_t by, int nrc);
```

Same problem as dequant - no `tensor->extra` pointer. Solution: the dispatch code in `ggml_compute_forward_mul_mat` has access to `src0->extra` via the tensor pointer. Special-case IQ2_XXS_V2 in the mul_mat dispatch:

**File:** `ggml/src/ggml-cpu/ops.cpp` (in `ggml_compute_forward_mul_mat`)

```c
if (src0->type == GGML_TYPE_IQ2_XXS_V2) {
    ggml_vec_dot_iq2_xxs_v2_q8_K_generic(
        max_n_threads ? ... : nrc,
        ...,
        src0->extra);  // additional extra arg
} else {
    // standard vec_dot dispatch
}
```

### 3.4 CPU vec_dot implementation

**File:** `ggml/src/ggml-cpu/quants.c` (new function, modeled after `ggml_vec_dot_iq2_xxs_q8_K_generic` at line 855)

Differs from V1 in two places:
1. Super-block scale: `d = GGML_CPU_FP16_TO_FP32(lut[d_idx])` instead of `GGML_CPU_FP16_TO_FP32(x[i].d)`
2. Sub-block scale with saturation extension

```c
void ggml_vec_dot_iq2_xxs_v2_q8_K_generic(
    int n, float * s, size_t bs, 
    const void * vx, size_t bx,
    const void * vy, size_t by, int nrc,
    const void * lut_extra) {
    
    const uint16_t * header = (const uint16_t *)lut_extra;
    uint16_t n_lut = header[0];
    const ggml_half * lut = (const ggml_half *)(header + 2);

    // Extract d_idx, lookup lut[d_idx] instead of x[i].d
    uint16_t d_raw = x[i].d;
    float d = GGML_CPU_FP16_TO_FP32(lut[d_raw & 0x0FFF]);
    uint32_t d_ext = d_raw >> 12;

    // In the inner loop, after reading ls = 2*(aux32[1]>>28) + 1:
    int scale_4bit = (aux32[1] >> 28);
    if (scale_4bit == 15) {
        int ext_bit = (d_ext >> saturated_count) & 1;
        scale_4bit = 16 + ext_bit;
        // ls = 2*scale_4bit + 1  (unchanged formula)
        saturated_count++;
    }
}
```

### 3.5 x86 AVX2 vec_dot

**File:** `ggml/src/ggml-cpu/arch/x86/quants.c` (new function near line 2522)

Copy `ggml_vec_dot_iq2_xxs_q8_K` and modify:
- Super-block scale: LUT lookup instead of `_mm256_cvtph_ps(_mm_set1_epi16(x[i].d))`
- Sub-block scale extension: read `d_ext` from the block, use `scale_4bit > 15` as extension indicator, extract ext_bit

The AVX2 implementation uses intrinsics. Key change in the inner loop:

```c
// V1:
const __m256i m_ls = _mm256_set1_epi32(2*(aux32[1] >> 28) + 1);

// V2:
int scale_4bit = (aux32[1] >> 28);
int ls_val = 2*scale_4bit + 1;
if (scale_4bit == 15) {
    int ext = (d_ext >> saturated_count) & 1;
    ls_val = 2*(16+ext) + 1;
    saturated_count++;
}
const __m256i m_ls = _mm256_set1_epi32(ls_val);
```

### 3.6 ARM NEON vec_dot

**File:** `ggml/src/ggml-cpu/arch/arm/quants.c` (new function near line 3557)

Same pattern as AVX2 but with NEON intrinsics. The scale value is computed via the NEON multiply intrinsics rather than AVX2 equivalents.

---

## 4. CUDA Pipeline Changes

### 4.1 Dequant kernel

**File:** `ggml/src/ggml-cuda/convert.cu` (new kernel near line 311)

Copy `dequantize_block_iq2_xxs` and modify:

```c
template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs_v2(
    const void * __restrict__ vx, 
    dst_t * __restrict__ yy,
    const void * __restrict__ lut_extra) {

    const uint16_t * header = (const uint16_t *)lut_extra;
    uint16_t n_lut = header[0];
    const half * lut = (const half *)(header + 2);

    const int64_t i = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs *)vx;
    const uint16_t d_raw = x[i].d;
    const uint16_t d_idx = d_raw & 0x0FFF;
    const uint32_t d_ext = d_raw >> 12;
    // read d from LUT
    // compute saturated_count using shared memory or per-thread logic
}
```

**Key CUDA-specific challenge: saturated_count.** In the CPU implementation, `saturated_count` is a simple local variable that increments per sub-block. In the CUDA kernel, threads (tid = il + ib*4) process different sub-blocks concurrently. Thread `tid` corresponds to sub-block `ib = tid/4` with lattice index `l = tid%4`. All 4 threads for the same sub-block need the same saturated_count value.

**Solution:** Compute `saturated_count` cooperatively. Each sub-block has 4 threads (one per lattice group). Threads first load their `scale_4bit` from `qs`, then use a warp-level prefix-sum of the saturation flags to compute `saturated_count` for their sub-block.

Alternative: serialize sub-block processing within each thread. Instead of each thread handling 1/4 of all sub-blocks, each thread handles ALL of 1-2 sub-blocks. This avoids the cooperative count but changes the thread grid layout.

Simpler approach: Pre-compute `saturated_count` for each sub-block in shared memory. Each thread loads `scale_4bit` for its sub-block, writes 1 to shared memory if saturated, 0 otherwise. Then barrier and a prefix sum in shared memory. Then each thread reads `saturated_count[ib]` from shared memory.

```c
__shared__ int saturated_counts[8];

int scale_4bit = (aux32 >> 28) & 0xF;
saturated_counts[ib] = (scale_4bit == 15) ? 1 : 0;
__syncthreads();

// Parallel prefix sum (single warp across 8 elements)
if (threadIdx.x < 8) {
    // tree reduction for prefix sum
    // ...
}

int my_saturated_count = saturated_counts[ib] > 0 ? 
    saturated_counts[ib] - 1 : 0;  // count of prior saturated sub-blocks

int ext_bit = (d_ext >> my_saturated_count) & 1;
int scale_eff = (scale_4bit == 15) ? 16 + ext_bit : scale_4bit;
float db = __half2float(lut[d_idx]) * (0.5f + scale_eff) * 0.25f;
```

**CUDA dequant dispatch:**

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` or equivalent dispatch

Add a case for `GGML_TYPE_IQ2_XXS_V2` that calls the V2 dequant kernel with the `src0->extra` pointer as an additional kernel argument. The tensor's `extra` buffer must be on the device (cudaMemcpy'd during setup).

If `src0->extra` is a host pointer, the kernel needs to follow a device pointer indirection or the LUT must be copied to device memory first.

**LUT GPU transfer:** During tensor initialization/setup, the CPU-side `extra` buffer is copied to device memory. A `ggml_tensor_extra_gpu` or similar struct stores the device pointer. The dequant kernel receives this device pointer, not the host pointer.

### 4.2 vec_dot kernel

**File:** `ggml/src/ggml-cuda/vecdotq.cuh` (new function near line 1015)

New function `vec_dot_iq2_xxs_v2_q8_1`:

```c
static __device__ __forceinline__ float vec_dot_iq2_xxs_v2_q8_1(
    const void * __restrict__ vbq, 
    const block_q8_1 * __restrict__ bq8_1, 
    const int & kbx, const int & iqs,
    const half * __restrict__ lut) {  // LUT passed as extra arg

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *)vbq + kbx;
    uint16_t d_raw = bq2->d;
    uint16_t d_idx = d_raw & 0x0FFF;
    
    // ... same lattice/sign/scale logic as V1 ...
    
    int ls;
    int scale_4bit = aux32 >> 28;
    if (scale_4bit == 15) {
        // need saturated_count -- same challenge as dequant
        // since vec_dot processes one 32-weight group at a time (via iqs),
        // we can pass saturated_count as a precomputed value
    }
    ls = 2*scale_4bit + 1;  // unchanged from V1
    
    float d = __half2float(lut[d_idx]) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}
```

### 4.3 mmq (matrix-matrix quant) kernel

**File:** `ggml/src/ggml-cuda/mmq.cuh`

**`load_tiles_iq2_xxs` (line 2741):** Add a V2 variant that:
- Reads `d_idx` from `x[i].d & 0x0FFF`
- Looks up scale from LUT
- Handles extension nibble for scale computation

**`mmq_type_traits` instantiation (line 3383):** Add a V2 template specialization with `VDR_IQ2_XXS_V2_Q8_1_MMQ`.

**`generate_cu_files.py` (line 41):** Add `"GGML_TYPE_IQ2_XXS_V2"` to the type list to auto-generate template instantiations.

### 4.4 CUDA dispatch tables

**File:** `ggml/src/ggml-cuda/mmvq.cu` (lines 25, 53, 118, 158, 179, 190, 208, 226, 303, 853, 1063)

Each switch case that dispatches on `GGML_TYPE_IQ2_XXS` needs a corresponding V2 case. The V2 vector dot function takes additional `lut` parameter, so the dispatch wrapper must pass `src0->extra` (or its device equivalent).

**File:** `ggml/src/ggml-cuda/vecdotq.cuh` (lines 982-983)

Add `#define VDR_IQ2_XXS_V2_Q8_1_MMVQ 2` and `#define VDR_IQ2_XXS_V2_Q8_1_MMQ 2`.

---

## 5. Quantizer Changes

### 5.1 Quantize function

**File:** `ggml/src/ggml-quants.c` (new function, modeled after `quantize_iq2_xxs` at line 3580)

```c
// Returns (see below): LUT on success, plus fills dst with compressed blocks.
// The heap-allocated LUT buffer must be freed by the caller and attached
// to the tensor via tensor->extra.
size_t quantize_iq2_xxs_v2(
    const float * src, void * dst, 
    int64_t nrow, int64_t n_per_row, 
    const float * quant_weights,
    void ** lut_out, size_t * lut_size) {

    // Phase 1: Quantize all rows using EXISTING V1 quantize logic.
    // The existing quantize_row_iq2_xxs_impl computes:
    //   - max_scale = max of all 8 sub-block scales per super-block
    //   - d = max_scale / 31
    //   - scale_4bit = nearest_int(0.5 * (scales[ib] / d - 1)), clamped to [0,15]
    // We modify this to support scale_5bit extension:

    // Phase 2: Per-block, compute V2 d encoding.
    // For each super-block:
    //   1. Compute 8 sub-block scales as in V1.
    //   2. For each sub-block where scale_4bit == 15:
    //      - Try extending to scale_5bit = 16 or 17 (the two candidates)
    //      - Choose the one that better fits the actual scale value
    //      - If scale would still saturate at 17, increase d instead.
    //   3. After all sub-blocks processed, look up d in the LUT.
    //      If d is not in LUT, add it.
    //   4. Write d_idx (LUT index) + extension nibble into block->d.

    // Phase 3: Build per-tensor LUT.
    // The LUT is a sorted unique set of all d values across the tensor.
    // Write the LUT to *lut_out, returning the size in *lut_size.

    // Phase 4: Return total bytes written (same as V1: nrow * nblock * 66).
}
```

### 5.2 Sub-block scale extension logic

The extension phase runs inside the V1 quantize loop at line 3388-3396:

```c
// V1 (current):
float d = max_scale/31;
y[ibl].d = GGML_FP32_TO_FP16(d);
for (int ib = 0; ib < QK_K/32; ++ib) {
    int l = nearest_int(0.5f * (scales[ib] / d - 1));
    l = MAX(0, MIN(15, l));
    q2[2*ib+1] |= ((uint32_t)l << 28);
}

// V2 (modified):
// Step 1: Try with d = max_scale/31 (same as V1).
// Step 2: Count saturated sub-blocks.
// Step 3: If > 4 saturate, increase d iteratively until <= 4 saturate.
// Step 4: For saturated sub-blocks, compute the best 5-bit extension.
// Step 5: Build ext_nibble from saturated sub-block extension bits.
// Step 6: Look up d in LUT, get index.
// Step 7: Write d_idx | (ext_nibble << 12) into block->d.
```

**Iterative d adjustment (step 3):**

```c
float d = max_scale / 31;
int max_iters = 0;
while (max_iters < 10) {
    int saturated = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int l = nearest_int(0.5f * (scales[ib] / d - 1));
        if (l > 15) saturated++;
    }
    if (saturated <= 4) break;
    d *= 1.125f;  // increase by 1/8
    max_iters++;
}
```

The 5th-bit extension for saturated sub-blocks is simply: if `nearest_int(0.5*(scale/d - 1)) == 16`, ext_bit = 0; if 17, ext_bit = 1. The clamped value stored in qs is always 15 (the 4-bit saturated marker). The kernel reads the ext_bit from the nibble at runtime.

### 5.3 LUT building and deduplication

After all blocks in a row are quantized, collect all `d` values into a hash set, sort, and write to the LUT buffer:

```c
// Phase 3: after all rows quantized
std::unordered_set<uint16_t> d_set;  // or a C hash table
for each block: d_set.insert(GGML_FP32_TO_FP16(d_value));

// Sort and build LUT
std::vector<uint16_t> lut_vec(d_set.begin(), d_set.end());
std::sort(lut_vec.begin(), lut_vec.end());

// Allocate and fill output
size_t lut_size = 4 + lut_vec.size() * 2;
uint8_t * lut_buf = (uint8_t *)malloc(lut_size);
*(uint16_t *)lut_buf = (uint16_t)lut_vec.size();
memcpy(lut_buf + 4, lut_vec.data(), lut_vec.size() * 2);
*lut_out = lut_buf;
*lut_size = lut_size;
```

### 5.4 Quantize dispatch

**File:** `ggml/src/ggml.c` (line ~7746, in the quantize dispatch switch)

```c
case GGML_TYPE_IQ2_XXS_V2: 
    result = quantize_iq2_xxs_v2(src + start, (char *)dst + start_row * row_size, 
                                  nrows, n_per_row, imatrix, &lut_out, &lut_size); 
    break;
```

### 5.5 Free LUT on tensor teardown

When a ggml tensor is freed and has `extra != NULL` for IQ2_XXS_V2, the LUT buffer must be freed. This is handled in `ggml_free` or the owning backend's tensor deallocation.

---

## 6. Model Saver / Loader Changes

### 6.1 GGUF quantization

**File:** `ggml/src/llama.cpp` or equivalent quantization entry point

When quantizing a model to IQ2_XXS_V2 via `llama_model_quantize_internal`:
- The quantizer is called per tensor (row group).
- After quantization, `lut_out` is attached to the ggml tensor via `tensor->extra`.
- The GGUF writer must serialize the LUT alongside the quantized tensor data.

**GGUF approach:** Store the LUT as the first bytes of each tensor's compressed data:

```
Tensor data layout for IQ2_XXS_V2:
  [2 bytes: n_lut (uint16 LE)]
  [2 bytes: padding (zeros)]
  [n_lut * 2 bytes: LUT values (FP16 LE)]
  [n_blocks * 66 bytes: compressed blocks]
```

The ggml `row_size` and `nbytes` calculations must account for the 4 + n_lut*2 byte header. Since the LUT is per-tensor (not per-row), the header is included in the first row's data only. Wait -- `ggml_row_size` is per-row. If the header spans multiple rows, the calculation changes.

**Option A (per-row LUT header):** Duplicate the LUT header at the start of every row. Wasteful (~6KB * n_rows overhead). But dead simple: `row_size = 4 + n_lut*2 + (ne[0]/QK_K)*66`.

**Option B (single header, first row):** The LUT lives at offset 0 of the tensor data. Row 0's data starts at offset 4 + n_lut*2. All other rows start at that offset + row*row_size_no_header. But this breaks the uniform `row_size` assumption in ggml_data layout. The stride `nb[1]` would be set differently, requiring changes to all row-iteration code.

**Choose Option A for simplicity.** The overhead for the Qwen2B model: ~6000 bytes per row * ~2,000 rows per tensor = ~12 MB overhead per tensor -- unacceptable!

**Choose Option C (per-tensor header in GGUF metadata):** Store the LUT as a custom GGUF key-value pair per tensor. During model loading, the KV is parsed and attached to the tensor's `extra` pointer. The compressed block data has the same layout as V1 (66 bytes per block, no header inside the data). This requires changes to:

- `gguf_writer`: when writing an IQ2_XXS_V2 tensor, also write its LUT as a KV.
- `llama-model-loader.cpp`: when loading, look up the LUT KV for each IQ2_XXS_V2 tensor and attach it.

**GGUF KV key format:** `{tensor_name}.iq2_xxs_v2_lut` with value type `GGUFValueType.ARRAY` of `uint8_t` bytes containing `(uint16_t)n_lut, (uint16_t)padding, (half[n_lut])values`.

**Choose Option C.**

### 6.2 `llama_model_saver`

**File:** `src/llama-model-saver.cpp`

When saving a model with IQ2_XXS_V2 tensors (after quantization), the saver must:
1. Write each tensor's block data (same as V1).
2. Write the LUT for each IQ2_XXS_V2 tensor as a custom GGUF KV.

The saver needs access to `tensor->extra` for IQ2_XXS_V2 tensors to extract the LUT.

### 6.3 `llama-model-loader`

**File:** `src/llama-model-loader.cpp`

During `create_tensor`, after `ggml_dup_tensor`, for IQ2_XXS_V2 tensors:
1. Look up the GGUF metadata key `{tensor_name}.iq2_xxs_v2_lut`.
2. Parse the LUT buffer from the KV value.
3. Allocate a heap buffer and copy the LUT.
4. Set `tensor->extra = lut_buffer`.

The LUT buffer must outlive the tensor. It should be freed when the tensor is freed.

---

## 7. Expected Results

### 7.1 Space

Zero change in model size. Same 66 bytes per block, 2.0625 bpw.

### 7.2 Quality

The sub-block scale doubles dynamic range from 4.0d to 8.0d for saturated sub-blocks. Super-blocks containing outlier weights (the ones with high inter-sub-block variance) benefit most. The super-block scale `d` no longer needs to be set artificially high to accommodate one extreme sub-block -- the extreme sub-block can use the extended 5-bit range while others keep their normal range.

Expected improvement: lower MSE on weight reconstruction, particularly in layers with high variance across weight blocks (FFN down projection, attention output). Quantization error at extreme weight values is halved.

### 7.3 Performance

- **CPU:** One extra branch per sub-block (`scale_4bit == 15` check), negligible. One LUT lookup per block (`lut[d_idx]`) replacing an FP16-to-FP32 conversion, same cost.
- **CUDA:** Shared memory for the saturated_count prefix sum adds ~10 instructions per warp per block. The LUT lookup replaces a register half2float conversion. Net cost: estimated <0.5% slowdown.
- **Memory:** ~6 KB per IQ2_XXS_V2 tensor for the LUT. For a 96-layer model with ~200 IQ2_XXS_V2 tensors: ~1.2 MB total LUT overhead.

### 7.4 Backward Compatibility

Existing IQ2_XXS models are unaffected. IQ2_XXS_V2 is a separate type. No GGUF format version bump needed.

---

## 8. Implementation Order

The changes are listed in dependency order. Each phase results in a compilable codebase.

### Phase 1: Type scaffolding

| # | File | Change |
|---|---|---|
| 1 | `ggml/include/ggml.h` | Add `GGML_TYPE_IQ2_XXS_V2 = 42`, bump `GGML_TYPE_COUNT` to 43 |
| 2 | `ggml/src/ggml-common.h` | Add `typedef block_iq2_xxs block_iq2_xxs_v2` |
| 3 | `ggml/src/ggml.c` | Add type traits entry for IQ2_XXS_V2 |
| 4 | `gguf-py/gguf/constants.py` | Add `IQ2_XXS_V2 = 42` to both enums |

### Phase 2: CPU pipeline

| # | File | Change |
|---|---|---|
| 5 | `ggml/src/ggml-quants.h` | Declare `dequantize_row_iq2_xxs_v2()` and `quantize_iq2_xxs_v2()` |
| 6 | `ggml/src/ggml-quants.c` | Implement `dequantize_row_iq2_xxs_v2()` (ref: line 2416 V1) |
| 7 | `ggml/src/ggml-quants.c` | Implement `quantize_iq2_xxs_v2()` with LUT building |
| 8 | `ggml/src/ggml-cpu/ops.cpp` | Special-case V2 in `ggml_compute_forward_dup_from_q` |
| 9 | `ggml/src/ggml-cpu/ggml-cpu.c` | Add type traits for V2 (vec_dot, vec_dot_type) |
| 10 | `ggml/src/ggml-cpu/quants.h` | Declare `ggml_vec_dot_iq2_xxs_v2_q8_K_generic` |
| 11 | `ggml/src/ggml-cpu/quants.c` | Implement generic vec_dot V2 |
| 12 | `ggml/src/ggml-cpu/ops.cpp` | Special-case V2 in mul_mat dispatch (pass `src0->extra`) |
| 13 | `ggml/src/ggml-cpu/arch/x86/quants.c` | Implement AVX2 vec_dot V2 |
| 14 | `ggml/src/ggml-cpu/arch/arm/quants.c` | Implement NEON vec_dot V2 |

### Phase 3: CUDA pipeline

| # | File | Change |
|---|---|---|
| 15 | `ggml/src/ggml-cuda/convert.cu` | Implement `dequantize_block_iq2_xxs_v2` kernel |
| 16 | `ggml/src/ggml-cuda/vecdotq.cuh` | Implement `vec_dot_iq2_xxs_v2_q8_1` |
| 17 | `ggml/src/ggml-cuda/mmvq.cu` | Add V2 cases in all dispatch switches |
| 18 | `ggml/src/ggml-cuda/mmq.cuh` | Add V2 `load_tiles`, `mmq_type_traits` |
| 19 | `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` | Add V2 to type list |

### Phase 4: Model I/O

| # | File | Change |
|---|---|---|
| 20 | `ggml/src/ggml.c` | Add V2 case to quantize dispatch |
| 21 | `src/llama-model-saver.cpp` | Write LUT as GGUF KV for V2 tensors |
| 22 | `src/llama-model-loader.cpp` | Read LUT from GGUF KV, attach to tensor->extra |
| 23 | `gguf-py/gguf/quants.py` | Add Python `IQ2_XXS_V2` class for dequant/quant |

### Phase 5: Verification

#### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_BACKEND_DL=OFF
cmake --build build --target test-quantize-fns -j$(nproc)
cmake --build build --target test-backend-ops -j$(nproc)
cmake --build build --target test-quantize-perf -j$(nproc)
cmake --build build --target llama-quantize -j$(nproc)
```

#### Unit tests

```bash
./build/bin/test-quantize-fns -v           # quantize/dequant/dot-product for all types
./build/bin/test-backend-ops               # dequant, mul_mat across CPU backend
./build/bin/test-quantize-perf             # quantization benchmarks
```

#### End-to-end: Quantize BF16 -> IQ2_XXS_V2

```bash
llama-quantize \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-USE-A-UNIQUE-NAME.gguf \
  IQ2_XXS_V2
```

Replace `USE-A-UNIQUE-NAME` with a descriptive name.

#### End-to-end: KLD perplexity (CUDA)

Must use CUDA device 1 and the exact command below. Only change the model filename.

```bash
CUDA_VISIBLE_DEVICES=1 /home/user/llm/outlier_llama/llama.cpp/build/bin/llama-perplexity \
  -m /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-USE-A-UNIQUE-NAME.gguf \
  -f /home/user/llm/wikitext-2-raw/wiki.test.raw -t 8 -c 512 --chunks 200 \
  -fa on --cache-type-k bf16 --cache-type-v bf16 --no-mmap -ngl 999 -np 1 \
  --kl-divergence --kl-divergence-base /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits
```

Compare KLD and perplexity against the IQ2_XXS V1 baseline (generated with the same command but with the V1-quantized model).

#### Profile CUDA kernel timing

Compare V2 kernel times against V1 baseline in `llama-perplexity` output (token probabilities include timing breakdowns).

---

## 9. Files NOT Modified (Out of Scope)

Per "CPU and CUDA only" constraint, these backends are left with default/unsupported behavior for V2:

- `ggml/src/ggml-metal/` -- Metal backend
- `ggml/src/ggml-vulkan/` -- Vulkan backend
- `ggml/src/ggml-sycl/` -- SYCL backend
- `ggml/src/ggml-webgpu/` -- WebGPU backend
- `ggml/src/ggml-opencl/` -- OpenCL backend
- `ggml/src/ggml-cpu/arch/riscv/`, `loongarch/`, `powerpc/`, `s390/` -- niche CPU architectures
