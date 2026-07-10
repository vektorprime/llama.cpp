# Q4_K_CLONE_INFO — Complete Reference for Q4_K_M_CLONE

**Read this before modifying any code.** This document describes the exact
state of the Q4_K_M_CLONE quantization type: what it is, where it lives in the
codebase, and which files you must touch to modify it.

## Architecture Overview

The clone exists at two layers:

| Layer | Enum | Value | Purpose |
|-------|------|-------|---------|
| GGML type | `GGML_TYPE_Q4_K_M_CLONE` | 42 | Block struct, quant/dequant functions, backend dispatch |
| LLaMA ftype | `LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE` | 41 | Per-tensor mixing strategy, CLI name, file metadata |

The clone is a **structural copy of Q4_K_M**. It uses the same block layout
(144 bytes per 256 elements = 4.5 bpw), same quantization algorithm, and same
per-tensor mixing rules. The quant/dequant functions are **thin wrappers** that
cast to `block_q4_K` and call through to the stock Q4_K implementations.

## Block Structure

```
block_q4_K_M_CLONE  (144 bytes, identical to block_q4_K):
  ggml_half d;                      // 2 bytes
  ggml_half dmin;                   // 2 bytes
  uint8_t  scales[K_SCALE_SIZE];    // 12 bytes  (K_SCALE_SIZE = 12)
  uint8_t  qs[QK_K/2];             // 128 bytes (QK_K = 256)
```

Constants: `QK_K = 256`, `K_SCALE_SIZE = 12`, `QR4_K = 2`, `QI4_K = 32`
(defined in `ggml/src/ggml-common.h`)

## Dequantization Formula

Each 256-element superblock is split into 8 sub-blocks of 32 elements each.
The dequant formula for a single weight:

```
x = d * sc * q - min * m
```

Where:
- `d` = super-block scale (`GGML_FP16_TO_FP32(block.d)`, a float16 converted to float32)
- `min` = super-block minimum offset (`GGML_FP16_TO_FP32(block.dmin)`)
- `sc` = sub-block scale (6-bit unsigned, 0..63, unpacked from `scales[]`)
- `m` = sub-block minimum multiplier (6-bit unsigned, 0..63, unpacked from `scales[]`)
- `q` = 4-bit quantized value (nibble from `qs[]`: low nibble for first 32 elements, high nibble for next 32)

The dequant loop (from `dequantize_row_q4_K` in `ggml/src/ggml-quants.c:1582`):
```c
for each of 8 sub-blocks:
    get_scale_min_k4(i, scales, &sc, &m);  // unpack 6-bit sc and m from scales[]
    d1 = d * sc;   m1 = min * m;
    for 32 elements:  y[l] = d1 * (q[l] & 0xF)      - m1;   // low nibble
    for 32 elements:  y[l] = d1 * (q[l] >> 4)       - m1;   // high nibble
```

### Scale/Min Packing in the 12-byte `scales[]` Field

The 12 bytes store **8 pairs** of `(6-bit scale, 6-bit min)` — one pair per
sub-block of 64 elements. The packing function `get_scale_min_k4`
(`ggml/src/ggml-quants.c:862`) unpacks them:

```
Bytes 0-3:    scale[i] = q[i] & 63        // bottom 6 bits
              min[i]   = q[i+4] & 63      // bottom 6 bits

Bytes 4-11:   scale[i] = (q[i+4] & 0xF) | ((q[i-4] >> 6) << 4)
              min[i]   = (q[i+4] >> 4)   | ((q[i-0] >> 6) << 4)
```

The 96 bits of scale data (8 pairs × 12 bits each) pack into 12 bytes.
Each scale/min is 6 bits, so 8×2×6 = 96 bits = 12 bytes.

## FWHT and Importance Matrix Incompatibility

**The imatrix is NOT per-weight importance.** It is a per-input-channel vector
`q_i = E[x_i^2]` — the variance of activations at each input channel, averaged
over calibration data. The same channel-importance vector is reused for every
output row of a weight matrix.

**After FWHT rotates the weight columns**, the original per-channel importance
no longer aligns with the transformed coordinates. For a 256-element FWHT group,
every transformed coordinate receives the SAME importance of `(1/256) * sum(q)`.

**Direct FWHT on the importance vector is WRONG.** Importance transforms
bilinearly as `RCR^T` (like a covariance), not linearly as `Rx` (like an
activation vector). FWHT would also produce negative values, which are invalid
as squared-error weights in the quantization search.

**Best solutions (in order of preference):**

1. **Recollect imatrix in FWHT space.** During calibration, apply FWHT to
   activations and collect `E[(Rx)_l^2]`. This recovers off-diagonal correlation
   information lost in the original diagonal imatrix.

2. **Flat fallback: average each 256-entry imatrix group.** Mathematically
   consistent with the diagonal-only approximation, but the imatrix becomes
   largely inert — similar to no imatrix at all.

3. **No imatrix (accept it).** FWHT intentionally makes coordinate directions
   more uniform. If the recollected imatrix is nearly flat, that is EXPECTED
   behavior — the rotation has removed the anisotropy that the imatrix was
   designed to exploit. Useful importance information may now live in off-diagonal
   covariance terms that a diagonal imatrix cannot capture.

**Validation experiment: generate these 6 variants:**

| # | FWHT | Imatrix | Expected |
|---|------|---------|----------|
| 1 | No | None | Baseline |
| 2 | No | Original | Better than 1 |
| 3 | Yes | None | May beat 1 via better distribution |
| 4 | Yes | Original (unchanged) | Expected REGRESSION vs 3 — coordinates misaligned |
| 5 | Yes | Block-averaged | Removes regression but ~= 3 |
| 6 | Yes | Recollected in FWHT space | Only variant with chance to beat 3 |

If variant 6 adds nothing over variant 3, the remaining importance information
is off-diagonal and would require a covariance/Hessian-aware quantizer (GPTQ/
QuIP-style), not llama.cpp's current coordinatewise imatrix heuristic.

## Known Constraints

Do NOT rediscover these:

1. **Bit removal without recovery is not a valid experiment.** Simply shrinking
   `qs[]`, `scales[]`, or removing `dmin` discards information without any
   method to reconstruct it. Every byte removed from the block must be paired
   with a technique that preserves the lost information: encoding, correlation
   exploitation, secondary quantization, adaptive bit allocation, etc. If your
   hypothesis is "make X smaller", it's incomplete. The hypothesis must be
   "make X smaller BY encoding/compressing/correlating it with Y".

2. **Scale precision matters.** The 6-bit scale quantization (0..63) in
   `scales[]` is already aggressive for 4-bit weights. Reducing scale precision
   makes `d * sc` too coarse to track sub-block variance, causing rapid
   quality erosion.

3. **Promising directions:** scale-min correlation encoding (mins often
   correlate with scales, enabling delta encoding), mixed sub-block precision
   (fewer bits for low-variance sub-blocks), and secondary quantization
   (codebook-based compression of scales or qs).

## Complete File Inventory

These are ALL the files that contain clone-related code. When making changes to
the clone (new quant algorithms, different structs, etc.), review every file
below for necessary updates.

### Part 1: GGML Type Registration

**`ggml/include/ggml.h`** — Type enum
- `GGML_TYPE_Q4_K_M_CLONE = 42` (line ~433)
- `GGML_TYPE_COUNT = 43` (bumped from 42)

**`ggml/src/ggml-common.h`** — Block struct
- `block_q4_K_M_CLONE` struct (after `block_q4_K`)
- `static_assert` verifying 144-byte size

**`ggml/src/ggml-quants.h`** — Function declarations
- `quantize_row_q4_K_M_CLONE_ref()` — reference quantize
- `dequantize_row_q4_K_M_CLONE()` — dequantize
- `quantize_q4_K_M_CLONE()` — dispatcher (used by ggml_quantize_chunk)

**`ggml/src/ggml-quants.c`** — Function implementations + validation
- Three thin wrapper functions (~L1695-1716): `_ref`, `dequantize_`, `quantize_`
- Validation switch case (~L6210): `VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_K_M_CLONE, ...)`

**`ggml/src/ggml.c`** — Type traits + quantize_chunk dispatch
- `type_traits[GGML_TYPE_Q4_K_M_CLONE]` entry (~L776)
- `ggml_quantize_chunk` switch case (~L7752)

**`ggml/src/ggml-cpu/ggml-cpu.c`** — CPU backend traits
- `cpu_type_traits[GGML_TYPE_Q4_K_M_CLONE]` entry (~L317)
- Uses `quantize_row_q4_K_M_CLONE` (CPU dispatcher wrapper)
- `vec_dot = ggml_vec_dot_q4_K_q8_K` (reuses Q4_K kernel)

**`ggml/src/ggml-cpu/quants.c`** — CPU dispatcher wrapper
- `quantize_row_q4_K_M_CLONE()` — casts void* → block_q4_K_M_CLONE* → calls ref

**`ggml/src/ggml-cpu/quants.h`** — CPU dispatcher declaration

### Part 2: CPU Ops Dispatch (7 switch chains)

**`ggml/src/ggml-cpu/ops.cpp`** — Seven locations, all using same pattern:
- `ggml_compute_forward_add` → type check chain
- `ggml_compute_forward_add1` → type check chain
- `ggml_compute_forward_acc` → type check chain
- `ggml_compute_forward_out_prod` → type check chain
- `ggml_compute_forward_set` → type check chain
- `ggml_compute_forward_cpy` → type check chain
- `ggml_compute_forward_clamp` → type check chain

Each adds `case GGML_TYPE_Q4_K_M_CLONE:` after the corresponding `GGML_TYPE_Q4_K:`.

### Part 3: CPU Repack

**`ggml/src/ggml-cpu/repack.cpp`** — 3 locations:
- `repack_q4_K_to_q4_K_8_bl` type assertion: `|| t->type == GGML_TYPE_Q4_K_M_CLONE`
- `repack_q4_K_to_q4_K_16_bl` type assertion: same
- Dispatch condition: `cur->type == GGML_TYPE_Q4_K_M_CLONE`

### Part 4: CUDA Backend

**`ggml/src/ggml-cuda/ggml-cuda.cu`** — Main CUDA dispatch, 1 location:
- Add `case GGML_TYPE_Q4_K_M_CLONE:` fallthrough to Q4_K handling

**`ggml/src/ggml-cuda/mmq.cu`** — MMQ dispatch, 3 locations:
- `mul_mat_q_case` dispatch: fallthrough to `GGML_TYPE_Q4_K` template
- MMQ supported types: add clone to the supported list
- MMQ eligibility check: `type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q4_K_M_CLONE`

**`ggml/src/ggml-cuda/mmvq.cu`** — MMVQ dispatch, ~13 locations:
- `case GGML_TYPE_Q4_K_M_CLONE:` fallthrough after every `GGML_TYPE_Q4_K:`

**`ggml/src/ggml-cuda/mmq.cuh`** — MMQ type traits, 4 locations:
- DS layout switch: clone falls through to DS4 layout
- DP4A tile sizes: `MMQ_DP4A_TXS_Q4_K` for clone
- MMA tile sizes: `MMQ_MMA_TILE_X_K_Q8_1` for clone
- Template specialization `mmq_type_traits<..., GGML_TYPE_Q4_K_M_CLONE>` (NOT needed — clone falls through to Q4_K at call site)

**`ggml/src/ggml-cuda/common.cuh`** — CUDA type traits template:
- `ggml_cuda_type_traits<GGML_TYPE_Q4_K_M_CLONE>` — qk=QK_K, qr=QR4_K, qi=QI4_K

**`ggml/src/ggml-cuda/convert.cu`** — Dequant function dispatch, 2 locations:
- `dequantize_row_q4_K_cuda` for clone type

### Part 5: LLaMA Layer

**`include/llama.h`** — Ftype enum:
- `LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE = 41`

**`src/llama-quant.cpp`** — Per-tensor mixing, 6 locations:
- Ftype→ggml_type mapping: `LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE → GGML_TYPE_Q4_K_M_CLONE`
- `ATTENTION_WV` condition (~L543): clone gets same Q6_K boost as Q4_K_M
- `FFN_DOWN` condition (~L599): clone uses same ffn_down rules as Q4_K_M
- `ATTENTION_OUTPUT` condition (~L627): clone in the expert-model list
- `ATTENTION_QKV` condition (~L646): clone gets Q5_K for QKV tensors

**`src/llama-model-loader.cpp`** — Metadata, 2 locations:
- Ftype name string: `"Q4_K - Medium Clone"`
- Reverse mapping: `GGML_TYPE_Q4_K_M_CLONE → LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE`

**`tools/quantize/quantize.cpp`** — CLI name:
- `{ "Q4_K_M_CLONE", LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE, "clone of Q4_K_M for research" }`

### Part 6: Other Backends (NOT wired)

The clone is NOT wired in Vulkan, Metal, SYCL, OpenCL, WebGPU, or OpenVINO.
Only GPU device 3 (RTX 3050) with CUDA is used for evaluation. If you need
other backends, add fallthrough cases in each backend's dispatch chain.

## How to Modify the Clone

### Changing the quantization algorithm

1. Edit the clone functions in `ggml/src/ggml-quants.c` — replace the thin wrappers with new quantize/dequantize logic
2. Update the CPU dispatcher in `ggml/src/ggml-cpu/quants.c` if needed
3. Build and test

### Changing the per-tensor mixing

1. Edit conditions in `src/llama-quant.cpp` — change which GGML types are assigned to which tensor categories
2. Add `|| ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE` to new conditions if they should apply to the clone
3. Build, quantize, eval

### Changing the block structure

1. Modify `block_q4_K_M_CLONE` in `ggml/src/ggml-common.h`
2. Update the static_assert
3. Update quant/dequant functions in `ggml/src/ggml-quants.c`
4. Update validation in `ggml/src/ggml-quants.c`
5. Update type_traits (`.type_size`) in `ggml/src/ggml.c` and `ggml-cpu/ggml-cpu.c`
6. Update CUDA type traits in `ggml/src/ggml-cuda/common.cuh`
7. Build and test

### Critical rule: the clone must always use `GGML_TYPE_Q4_K_M_CLONE` and
### `LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE`, never `GGML_TYPE_Q4_K` or the stock ftypes.

## Current Baseline

| Quant | Type Value | Ftype Value | Size | PPL | KLD | Same top p |
|-------|-----------|-------------|------|-----|-----|------------|
| BF16 | — | — | ~1.41 GB | 21.5386 | 0.0 | 100% |
| Q4_K_M | 12 | 15 | ~505 MB | 22.4499 | 0.062947 | 86.387% |
| Q4_K_M_CLONE | 42 | 41 | ~505 MB | 22.4499 | 0.062947 | 86.387% |
