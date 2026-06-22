# Scale Q4_K — Architecture & Implementation Reference

> Self-contained reference for the `scale-q4k` branch. Detailed enough to fully
> reconstruct the implementation, the math, the build, and the results without
> re-deriving anything. Repo: `/home/user/llm/scale_llama` (build server
> `10.0.0.188`, 2x RTX 3080). Fork remote: `github.com/vektorprime/llama.cpp`.

---

## 0. TL;DR

A **runtime-optional** Q4_K dequant modification: when a post-quant 4-bit nibble
is `0`, output the **scale value** as the weight instead of the standard
`-dmin*m`. Gated by `--scale-q4-k` (default OFF). Two variants:

- **Naive**: nibble 0 -> `+d*sc`
- **Sign-aware**: nibble 0 -> `(m1>0) ? -d*sc : +d*sc`  (m1 = dmin*m)

**Result: net regression on Q4_K** (baseline KLD 0.057 -> sign-aware 0.113).
Root cause: Q4_K nibble 0 is NOT a zero weight — it already encodes the sub-block
minimum `-m1` optimally. The premise only holds for min-less quant (Q4_0).
Feature is correct, builds clean, works on CPU + CUDA + `-sm tensor`, defaults OFF.

---

## 1. Q4_K format primer (the foundation)

### Block struct (`ggml/src/ggml-common.h`)
```c
#define QK_K 256          // weights per super-block
#define K_SCALE_SIZE 12   // bytes of packed 6-bit scales+mins
#define QR4_K 2           // Q8_1 blocks per Q4_K weight block (CUDA)
#define QI4_K 32          // = QK_K/(4*QR4_K)

typedef struct {
    union {
        struct { ggml_half d;    // super-block scale for the 6-bit SCALES
                 ggml_half dmin;  // super-block scale for the 6-bit MINS
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // 8 sub-block (scale, min) pairs, 6-bit each
    uint8_t qs[QK_K/2];           // 128 bytes, two 4-bit nibbles per byte
} block_q4_K;                     // 256 weights = 8 sub-blocks of 32
```

### Dequant math (THE key equation)
Each super-block has 8 sub-blocks of 32 weights. Sub-block `j` has a 6-bit scale
`sc` and 6-bit min `m` (unpacked via `get_scale_min_k4`). For a weight with
nibble value `q` (0..15):

```
weight = (d * sc) * q  -  (dmin * m)
       =      d1  * q  -        m1
```
where `d1 = d*sc` (the per-sub-block SCALE) and `m1 = dmin*m` (the per-sub-block
MIN, always >= 0 because the quantizer forces min<=0 and stores `-min`).

**What nibble 0 means**: `weight = -m1`. This is the *sub-block minimum* — a
real, usually-negative value, stored with 6-bit precision. It is NOT zero
(except in all-positive blocks where the quantizer forces m=0). **This single
fact is why the feature fails for Q4_K** — see Section 9.

---

## 2. The transformation

| Case (nibble q) | Standard Q4_K | Naive Scale | Sign-aware Scale |
|-----------------|---------------|-------------|------------------|
| q > 0           | d1*q - m1     | d1*q - m1   | d1*q - m1        |
| q == 0          | -m1           | +d1         | (m1>0) ? -d1 : +d1 |

Sign-aware rationale: the value being replaced is `-m1` (<=0 when m1>0). The
naive `+d1` has the wrong sign for negative-min blocks; signing by `-m1`'s
direction is strictly closer to the true value. Both gated by the same
`--scale-q4-k` flag (sign-aware replaced naive in the final build; naive numbers
preserved in Section 9).

---

## 3. System architecture: 2 pipelines x 2 paths

Inference touches Q4_K weights through **two distinct code paths**, and each
exists in **both** the CPU and CUDA backends. ALL FOUR were patched.

```
                    +-------------------- DEQUANT PATH ---------------------+
                    |  Q4_K block  ->  fp32 row   (GET_ROWS / full dequant) |
                    +------------------------------------------------------+
                    +------------------ QUANTIZED MATMUL PATH --------------+
                    |  Q4_K . Q8 dot product directly (no full dequant)     |
                    +------------------------------------------------------+

  CPU backend                              CUDA backend
  -----------                              ------------
  Dequant : dequantize_row_q4_K()          Dequant : dequantize_block_q4_K()
            ggml-quants.c:1471                       convert.cu:~205
  Matmul  : ggml_vec_dot_q4_K_q8_K_generic Matmul  : vec_dot_q4_K_q8_1_impl_vmmq (single-token / MMVQ)
            ggml-cpu/quants.c:645                    vec_dot_q4_K_q8_1_impl_mmq  (batch / MMQ)
                                                     vecdotq.cuh
```

- **MMVQ** (mat-vec, `VDR_Q4_K_Q8_1_MMVQ=2`): used for single-token generation.
- **MMQ** (mat-mat, `VDR_Q4_K_Q8_1_MMQ=8`): used for batch/prompt processing and
  is also exercised by `-sm tensor` (split-mode tensor) across GPUs.
- The MMVQ/MMQ paths do NOT fully dequantize — they compute the dot product
  directly from packed nibbles using `dp4a`. This is why the correction (Section
  7) cannot just "change the dequant"; it must be folded into the dot product.

---

## 4. Runtime flag plumbing (full chain)

The flag must reach BOTH a host-side global (CPU paths read it directly) AND a
CUDA `__constant__` (device kernels read it). Two separate setters.

```
CLI: --scale-q4-k / --no-scale-q4-k          [common/arg.cpp, BOOL-pair pattern]
        |
        v
common_params.scale_q4_k  (bool)             [common/common.h]
        |
        +--> ggml_set_scale_q4_k(bool)       [ggml-quants.c]  -> bool g_scale_q4_k   (host global, NON-static, exported)
        |        consumed by: dequantize_row_q4_K (ggml-quants.c)
        |                     ggml_vec_dot_q4_K_q8_K_generic (ggml-cpu/quants.c, via `extern bool g_scale_q4_k`)
        |
        +--> ggml_cuda_set_scale_q4_k(bool)  [convert.cu] -> cudaMemcpyToSymbol(g_cuda_scale_q4_k)
                 g_cuda_scale_q4_k : __device__ __constant__ bool   (defined convert.cu, `extern` in vecdotq.cuh)
                 consumed by: dequantize_block_q4_K (convert.cu)
                              vec_dot_q4_K_q8_1_impl_vmmq / _mmq (vecdotq.cuh)

Setter call sites (run right after model load, before any inference):
  - tools/perplexity/perplexity.cpp   (after common_init_from_params)
  - tools/server/server-context.cpp   (after common_init_from_params; this file
                                        backs BOTH llama-server AND llama-cli)
Both call sites guard the CUDA setter with #ifdef GGML_USE_CUDA.
```

**Declarations** live in PUBLIC headers so tools can call them:
- `ggml/include/ggml.h`      : `ggml_set_scale_q4_k`, `ggml_get_scale_q4_k`
- `ggml/include/ggml-cuda.h` : `ggml_cuda_set_scale_q4_k`

> Pitfall fixed: originally declared in `ggml/src/ggml-quants.h`, which is NOT on
> the tools' include path -> `fatal error: ggml-quants.h: No such file`. Moved to
> `ggml.h` (public). See Section 8.

---

## 5. Per-file change catalog

| # | File | Change |
|---|------|--------|
| 1 | `ggml/src/ggml-quants.c` | CPU dequant branch; defines `bool g_scale_q4_k` (NON-static) + `ggml_set/get_scale_q4_k` |
| 2 | `ggml/src/ggml-quants.h` | (early) extern decls — superseded by ggml.h |
| 3 | `ggml/src/ggml-cpu/quants.c` | CPU vec_dot: `extern bool g_scale_q4_k`, unpack tweak + min/sign correction |
| 4 | `ggml/src/ggml-cuda/convert.cu` | CUDA dequant branch; defines `__constant__ g_cuda_scale_q4_k` + `ggml_cuda_set_scale_q4_k` |
| 5 | `ggml/src/ggml-cuda/vecdotq.cuh` | `sum_q8_at_zero_nibbles()` helper; MMVQ + MMQ correction; `extern` constant |
| 6 | `ggml/include/ggml.h` | public decls for host setter/getter |
| 7 | `ggml/include/ggml-cuda.h` | public decl for CUDA setter |
| 8 | `common/common.h` | `bool scale_q4_k = false;` in `common_params` |
| 9 | `common/arg.cpp` | `--scale-q4-k` / `--no-scale-q4-k` (BOOL-pair add_opt) |
| 10 | `tools/perplexity/perplexity.cpp` | setter wiring |
| 11 | `tools/server/server-context.cpp` | setter wiring (covers llama-cli) |

### 5.1 CPU dequant (`ggml-quants.c`, dequantize_row_q4_K)
```c
if (g_scale_q4_k) {
    const float s1 = (m1 > 0.0f) ? -d1 : d1;   // sign-aware scale, low nibbles
    const float s2 = (m2 > 0.0f) ? -d2 : d2;   // sign-aware scale, high nibbles
    for (int l = 0; l < 32; ++l) { uint8_t nib = q[l] & 0xF; *y++ = nib ? (d1*nib - m1) : s1; }
    for (int l = 0; l < 32; ++l) { uint8_t nib = q[l] >> 4;  *y++ = nib ? (d2*nib - m2) : s2; }
} else {
    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;
}
```

### 5.2 CUDA dequant (`convert.cu`, dequantize_block_q4_K)
```c
if (g_cuda_scale_q4_k) {
    y[l + 0] = n0 ? (d1*n0 - m1) : ((m1 > 0.0f) ? -d1 : d1);
    y[l +32] = n1 ? (d2*n1 - m2) : ((m2 > 0.0f) ? -d2 : d2);
} else {
    y[l + 0] = d1*n0 - m1;
    y[l +32] = d2*n1 - m2;
}
```

---

## 6. CPU vec_dot correction (`ggml-cpu/quants.c`)

`ggml_vec_dot_q4_K_q8_K_generic` unpacks nibbles into `int8 aux8[256]`, then
accumulates `aux32[l] += scale * q8[l] * aux8[l]` and finally
`sumf = d*sum(aux32) - dmin*sumi` where `sumi = sum(bsums[j]*mins[j/2])`.

Approach (sign-aware):
1. **Unpack**: in scale mode, map nibble `0 -> 1` so it contributes `+d*sc*q8`
   to the dot product (instead of 0). Save `q4_orig = x[i].qs` for the
   correction pass.
2. **Min correction** (cancel the min term wrongly applied to zero nibbles):
   `sumf += dmin * mins[is] * sum_zero_q8`.
3. **Sign flip** (sign-aware only): if `mins[is] > 0`, flip the contribution from
   `+d*sc` to `-d*sc`: `sumf -= 2 * d * scales[is] * sum_zero_q8`.

`sum_zero_q8` = sum of int8 q8 values at positions where the Q4 nibble is 0, per
sub-block `s` (q4 byte offset `(s/2)*32`, low/high nibble by `s&1`, q8 at
`s*32 + l`). The correction loop runs only `if (g_scale_q4_k)`.

> CRITICAL pitfall: the `sumf -= dmin*sumi;` string also exists in the q5_K and
> q6_K functions. A naive string-replace inserted the correction (which uses
> `q4_orig`) into q5_K too -> `q4_orig undeclared`. Fix: ensure exactly ONE
> insertion, in q4_K only. See Section 8.

---

## 7. CUDA vec_dot correction (`vecdotq.cuh`) — the core trick

The MMVQ/MMQ kernels use `dp4a` over packed nibbles, so we CANNOT special-case
zero nibbles inside the dot product cheaply. Instead we compute the STANDARD
result and add a **post-hoc correction** for zero-nibble lanes.

### Helper
```c
static __device__ int sum_q8_at_zero_nibbles(int nib_packed, int q8_packed) {
    int s = 0;                                   // nibbles packed as 0x0N0N0N0N
    if ((nib_packed       & 0xF)==0) s += (q8_packed       & 0xFF);
    if (((nib_packed>>8)  & 0xF)==0) s += ((q8_packed>>8)  & 0xFF);
    if (((nib_packed>>16) & 0xF)==0) s += ((q8_packed>>16) & 0xFF);
    if (((nib_packed>>24) & 0xF)==0) s += ((q8_packed>>24) & 0xFF);
    return s;
}
```

### Derivation (per sub-block i)
Standard MMVQ contribution: `d*sc*sum(q*q8) - dmin*m*sum(q8)`, scaled by the
Q8_1 scale `d8[i]`. For a zero-nibble lane, the standard contributes
`-dmin*m*q8` (dp4a gives 0 for q=0). We WANT:
- Naive: `+d*sc*q8`     -> correction `(d*sc + dmin*m)*q8`
- Sign : `sign*d*sc*q8` -> correction `(sign*d*sc + dmin*m)*q8`, `sign=(m>0)?-1:+1`

So, summing `sum_zero` = sum of q8 at zero-nibble lanes:
```c
// MMVQ (d8[ci] is the Q8_1 scale; dm4f.x=d, dm4f.y=dmin)
if (sum_zero > 0) {
    const float sscale = (m[ci] > 0) ? -dm4f.x*sc[ci] : dm4f.x*sc[ci];
    result += d8[ci] * (float)sum_zero * (sscale + dm4f.y*m[ci]);
}
// MMQ (ds8f.x is the Q8_1 scale from ds8[ci])
if (sum_zero > 0) {
    const float sscale = (m[ci] > 0) ? -dm4f.x*sc[ci] : dm4f.x*sc[ci];
    result += ds8f.x * (float)sum_zero * (sscale + dm4f.y*m[ci]);
}
```
Both gated by `if (g_cuda_scale_q4_k)`. MMVQ loops `ci < QR4_K (=2)`; MMQ loops
`ci < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1` with an inner `cj < QI8_1` over `v[cj]`.

> Pitfall fixed: the MMVQ correction (which references `d8`) was accidentally
> pasted into the MMQ function (which has `ds8`, not `d8`) ->
> `identifier "d8" is undefined`. MMVQ must use `d8[ci]`, MMQ must use `ds8f.x`.

---

## 8. Build system notes + every error encountered & fix

**Build cmd** (server 10.0.0.188, RTX 3080 = sm_86):
```bash
cd /home/user/llm/scale_llama
export LLAMA_USE_PREBUILT_UI=OFF
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel -j4
# First full CUDA build ~15+ min (normal). vecdotq.cuh flows through common.cuh
# into ~all CUDA files, so editing it triggers a near-full CUDA recompile.
# Faster iteration after first build: cd build && make llama-quantize llama-perplexity llama-cli -j4
```

| Error | Cause | Fix |
|-------|-------|-----|
| `identifier "d8" is undefined` (vecdotq.cuh) | MMVQ correction pasted into MMQ fn | MMVQ uses `d8[ci]`, MMQ uses `ds8f.x` |
| `q4_orig undeclared` (ggml-cpu/quants.c) | correction leaked into q5_K fn | insert correction in q4_K only |
| `add_opt is not captured` / wrong ctor (arg.cpp) | insertion landed inside a lambda; BOOL needs pair form | place after last add_opt, use `{"--x"},{"--no-x"}` 4-arg ctor |
| `undefined reference to g_scale_q4_k` (link) | `static` gave internal linkage | drop `static` -> exported symbol |
| `ggml-quants.h: No such file` (tools) | src-only header not on include path | declare setters in public `ggml.h` |
| conflicting C/C++ linkage decl (ggml.h) | inserted before BOTH extern-C guards | insert ONCE, after `ggml_quantize_free` |
| `final link failed: bad value` (libggml-cuda.so) | `static __device__ __constant__` vs `extern` mismatch | drop `static` on the `__constant__` def |
| `test-reasoning-budget` link fail | unrelated test target | build only the 3 needed binaries |

> Editing rule: NEVER `sed -i` C++ on the build server (quote mangling). Edit via
> python string-replace with assertions, or `patch`. Always grep the changed
> region afterward.

---

## 9. Results & root-cause analysis

### Validation (Qwen3.5-2B, wikitext, 5 chunks, KLD vs `Qwen3.5-2B-BF16.logits`)
| Metric    | Baseline Q4_K | Naive Scale | Sign-aware Scale | Q4_0 ref |
|-----------|---------------|-------------|------------------|----------|
| Mean PPL  | 13.32         | 14.89       | 14.04            | 13.9     |
| Mean KLD  | 0.0570        | 0.1844      | 0.1127           | 0.093    |
| Same top-p| 86.76%        | 79.15%      | 82.94%           | 83.9%    |

### -sm tensor (2x RTX 3080, tensor split) — verified, matches single-GPU
| Run (-sm tensor) | PPL    | KLD    | top-p  |
|------------------|--------|--------|--------|
| baseline         | 13.319 | 0.0570 | 86.73% |
| sign-aware scale | 14.030 | 0.1127 | 82.83% |
No crashes/CUDA errors. Harmless warning: `llama_params_fit` auto-fit
unimplemented for `SPLIT_MODE_TENSOR` (inference runs fine).

### Root cause (definitive)
Q4_K nibble 0 already encodes the sub-block minimum `-m1` with 6-bit precision —
the OPTIMAL representation for those weights. Replacing it with `+/-d*sc`
discards that info, so every scale variant is strictly worse. Sign-awareness
recovers ~40% of the loss (KLD 0.184 -> 0.113) by fixing the sign, but cannot
beat baseline. The premise "weight*scale=0 loses accuracy" is true only for
**min-less** quant (Q4_0: `weight = d*q`, q signed, q=0 -> weight=0). Q4_K's min
offset breaks the premise. Feature correctly defaults OFF.

---

## 10. Reproduce / extend

### Quantize (standard Q4_K; the feature is inference-time, not quant-time)
```bash
./build/bin/llama-quantize --token-embedding-type q4_K \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-Q4K-SCALE-1.gguf Q4_K
```

### KLD test (add `--scale-q4-k` to enable; add `-sm tensor` for multi-GPU)
```bash
CUDA_VISIBLE_DEVICES=0 timeout 120 ./build/bin/llama-perplexity \
  -m .../Qwen3.5-2B-Q4K-SCALE-1.gguf -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
  -t 8 -c 512 --chunks 5 -fa on --cache-type-k bf16 --cache-type-v bf16 \
  --no-mmap -ngl 999 -np 1 [--scale-q4-k] [-sm tensor] \
  --kl-divergence --kl-divergence-base .../Qwen3.5-2B-BF16.logits
```

### Inference smoke test
```bash
./build/bin/llama-cli -m .../Qwen3.5-2B-Q4K-SCALE-1.gguf -ngl 999 [--scale-q4-k] -p "..."
```

### To port the idea to Q4_0 (where the premise DOES hold)
nibble q is signed [-8,7]; q=0 -> weight=0. Patch the dequant + the
`vec_dot_q4_0_q8_1` MMVQ/MMQ + `dequantize_row_q4_0` to emit `+/-d` when q==0.
Same flag plumbing. Sign can come from a spare bit or the block's d sign.

### To add a true sign-bit encoding (quantizer-side, not heuristic)
Q4_0/Q4_K blocks are tightly packed (no free bits). Would need a sidecar bitmask
(1 bit per zero-nibble weight) written in `src/llama-quant.cpp` and read at
dequant. Larger change; the runtime heuristic (sign of `-m1`) avoids it.

---

## 11. Coordinates
- Branch: `scale-q4k`  (commit `700b097fc` = base feature)
- Remotes: `origin` = ggerganov (upstream, pull only); `fork` = vektorprime (push)
- Model: `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-Q4K-SCALE-1.gguf` (1.1G, 4.82 BPW)
- BF16 logits: `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits`
- Test data: `/home/user/llm/wikitext-2-raw/wiki.test.raw`
- Report: `results.md` (summary) | this file (architecture)
