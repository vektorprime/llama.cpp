# CUDA and CPU backend behavior

The branch adds **no new kernels** to `ggml-cuda` or `ggml-cpu`. Every op the
glm5next graph emits already has CUDA and CPU implementations in-tree (most
were added recently for DeepSeek-V4 / kimi-k3 and are reused here). What this
document maps is which op each piece of the model lowers to, on each
backend, and the precision/type rules that differ between them.

## Op map

| model piece | ggml ops | CUDA | CPU |
|---|---|---|---|
| all GEMMs (proj, Wo, experts, lm_head) | `ggml_mul_mat` | mmq / wmma kernels, quantized weight types incl. MXFP4 experts, fp16/bf16 activations | quantized dot ops per type |
| mHC pre/comb/post | `ggml_dsv4_hc_pre/comb/post` (fused, on by default) | fused dsv4_hc kernels | same fused ops, CPU impl |
| KDA conv | `ggml_ssm_conv` + `ggml_silu` | `ssm-conv.cu` | ssm-conv in ops |
| KDA recurrence (chunked) | `ggml_gated_delta_net` fused, or explicit chain | `gated_delta_net.cu` | CPU fused or ops chain |
| KDA recurrence (AR) | `ggml_gated_delta_net` K=1 fused, or explicit | same kernel, AR mode | same |
| DSA dense/sparse attention | `ggml_flash_attn_ext` (FA on) or KQ/softmax/V path (FA off) | FA kernel / unfused | FA kernel (CPU has flash attn) / unfused |
| indexer scoring | `ggml_lightning_indexer` (fused, on by default) or mul_mat+relu+sum | `lightning-indexer.cu` | `ops.cpp` loop |
| pool member gather / top-k expand | `ggml_get_rows` | get_rows kernel | ops |
| top-k over pools | `ggml_top_k` | `top-k.cu` | ops (deliberately non-deterministic among ties) |
| MoE routing + expert mix | `ggml_topk_moe` (+ `mul_mat` experts) | `topk-moe.cu`, mmq | ops |
| masks/selection | `ggml_set_rows`, `ggml_add`, `ggml_dup`, `ggml_fill`, `ggml_cast` | kernels; SET_ROWS advertised for f32 values | ops |
| caches | `cpy_k` into host/GPU buffers | offloaded KV when `-otk`/offload on | KV on host by default |

Fused-op defaults are set in `llama_context` ctor: `fused_lid`, `fused_gdn_ar`,
`fused_gdn_ch`, `fused_dsv4_hc_*` all start true and are **probed** against
the actual backends (a small 1-token test graph per op, `llm_fused_op_*_probe`
in `llama-context.cpp`); if a backend lacks the op, the flag flips to false
and the unfused graph spelling is used. So the same GGUF runs identically
modulo summation order on a CPU-only build.

## The fused lightning indexer (`ggml_lightning_indexer`)

Signature (from `ggml.c`): `q [d, n_head, n_tokens, n_stream]` F32,
`k [d, 1, n_kv, n_stream]` (F32 or quantizable), `w [n_head, n_tokens, 1,
n_stream]` F32 (prescaled), `m [n_kv, n_tokens, 1, n_stream]` **F16**; result
`[n_kv, n_tokens, 1, n_stream]` F32 = `sum_h relu(q_h . k) * w_h + m`. The
mask add is free inside the kernel - on the graph side the unfused path needs
a separate `+ pool_bias` node.

CUDA (`ggml/src/ggml-cuda/lightning-indexer.cu`):

- Two kernel families, dispatched by compute capability:
  - **wmma** (Turing and up, `TURING_MMA_AVAILABLE`): 8-head tiles x
    16-embd inner loops, Q packed to half in shared memory, K as a wmma
    matrix_b fragment. K types: F16 and the quant types Q4_0/Q4_1/Q5_0/Q5_1/
    Q8_0 (dequantized into half in shared memory). n_embd/n_head instantiated
    for (128, 64) and (128, 32) - glm5next is (128, 32).
  - **vec** (pre-Turing): scalar/vector path, K types F16, Q4_0..Q8_0, BF16,
    F32.
- Layout: one block per (K-vector chunk, batch row, stream); W row in shared
  mem, Q tile streamed, partials reduced per head chunk, mask added on store.
- The f32 `pool_k` makes the kernel take the f32/vector path for K rather than
  the f16 wmma path - deliberate, so `GGML_PREC_F32` on the `w` GEMM is not
  undone by an f16 K accumulation.

CPU (`ggml/src/ggml-cpu/ops.cpp:11941`):

- Threads split the **K axis** (one row range per thread); per (stream,
  token, k-row): dequant K row to F32 scratch, loop over the 32 heads doing
  `ggml_vec_dot_f32` (SIMD), `score += max(qk, 0) * w[h]`, then add the f16
  mask entry. Straightforward and exact in F32 - it is the reference
  arithmetic the CUDA kernel is measured against.

Determinism note: the fused kernel's head-summation order differs from the
unfused graph, so near-tied pool scores can rank differently and a top-k
boundary pool can flip. `LLAMA_FUSED_LID_DISABLE=1` (llama-context.cpp)
forces the unfused path for comparison runs. This only matters at pool-score
ties - the selection is still whole-pool on either path.

## top-k and tie behavior

`ggml_top_k` (pools axis, width `select_k` = 512):

- CPU: explicitly **unordered among equal scores** (the op deliberately swaps
  its first two results to keep backends honest).
- CUDA: `determinism::not_guaranteed` for the same reason.

Because the cut is over **pools** and pools enter/leave the tie group as
units, CPU and CUDA can disagree *which* equal pools are returned but never
*whether a pool is taken whole* - that is how the branch removes the
CPU/CUDA divergence structurally instead of by luck. (A cell-level top-k would
expose it: 7.5% of rows on the test fixture.)

## Mask and type rules

- `sel_mask`/`cand_mask` are **F16** graph inputs (0/-inf only, lossless).
  - Flash attention on: the KQ mask is F16, the adds need no cast.
  - Flash attention off: the KQ mask is F32; `ggml_add` yields src0's type,
    and F16+F32 -> F16 is a supported bin_bcast on **both** CUDA and CPU, so
    the sparse mask stays F16 and `ggml_soft_max` takes an F16 mask as
    readily as F32.
  - The one direction needing an explicit cast (F32 mask meeting F16 KQ mask
    for `ggml_flash_attn_ext`) is handled in `build_attn_sparse`.
- `pool_bias_f16` is a `ggml_cast` node built **once per graph** (not per DSA
  layer) because all 11 indexer layers share the same mask; the cast is exact
  (0/-inf).
- `ggml_set_rows` (the scatter) converts values into the destination type;
  CUDA advertises SET_ROWS only for **F32 values**, so the scatter source is
  an F32 zeros tensor regardless of the mask type.
- `ggml_get_rows` (pool member gather, top-k expand) always yields F32 - the
  pooling math is F32 even though the indexer cache is F16, and the expanded
  indices stay I32 end to end.

## Where the caches live

- KV offload follows the standard flags (`-otk` / `kv_offload`): attn and
  indexer caches are ordinary `llama_kv_cache` buffers, so they offload to the
  GPU like any other KV. The pool map tensors (`pool_cells`, `pool_bias`,
  `sel_mask`, `cand_mask`) are **host** inputs (asserted host-backed in
  `llama_kv_cache_set_input_kpool`); they are small relative to the caches
  (`n_kv x n_tps` F16 entries each) and are written by the CPU before every
  compute.
- Recurrent state (KDA) lives where the recurrent memory puts it (host by
  default, offloadable).
- The indexer cache is F16 regardless of `-ctk` (quantized types are refused
  with a warning), which keeps the compressor softmax and the f16->f32 gather
  cheap on both backends.

## Offload split

Standard split: weights to GPU (or CPU) per `--n-gpu-layers`/tensor-type
rules; the compute graph is a single graph - ops land on the backend of their
tensors. The mHC state is explicitly `ggml_build_forward_expand`ed before the
FFN so offload does not pull it onto the expert weights' backend (same
pattern as deepseek4). On a CPU-only build every fused op above falls back to
its CPU implementation or its unfused graph spelling via the probes, so the
model runs identically (modulo F32 summation order) with zero GPU.

## Practical differences to expect, CUDA vs CPU

1. Decode is bandwidth-bound on both; CUDA wins on the MoE GEMMs and the
   fused KDA/indexer kernels. CPU decode cost is dominated by the 34 KDA
   layers' state traffic (~270 MB/token) + 9 expert GEMMs x 42 layers.
2. Prefill is FLOPs-bound; the indexer's full-cache member gather (11x per
   ubatch) is a memory term that scales with n_kv on both backends.
3. The fused lightning indexer changes near-tie top-k outcomes vs CPU's order;
   if a parity test disagrees on *which* pool (not how many), that is the
   first suspect, and `LLAMA_FUSED_LID_DISABLE=1` confirms it.
4. Quantization: MXFP4 experts + q8_0 MLA projections + F16 indexer cache is
   the measured release configuration; the pinned full-precision tensors
   (01-components.md, precision policy) apply identically on both backends.
