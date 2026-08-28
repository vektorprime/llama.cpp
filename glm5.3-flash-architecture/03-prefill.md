# Prefill

Prefill = any ubatch with `n_seq_tokens > 1` (default `n_ubatch` 512 tokens,
chunked over the prompt). Everything below runs identically on GPU and CPU
except where noted; op-level backend behavior is in 05-cuda-cpu.md.

## Before the graph (host side, once per ubatch)

1. The ubatch is prepared against `llama_memory_hybrid_context`, which carries
   the slot info of all three caches (attn/idx share it, recr has its own
   state slots). `set_input` on the graph inputs writes:
   - `k_idxs` for the attn cache (MLA latent cells),
   - `k_idxs` for the **indexer** cache - **unconditionally**, see below,
   - the KQ causal mask(s),
   - the whole **pool map** (`pool_cells`, `pool_bias`, `sel_mask`,
     `cand_mask`) - O(n_kv x n_tokens) host stores, the single biggest host
     cost of the model, paid **once** and shared by all 11 DSA layers.
2. Recurrent state for the KDA layers is read at the sequence's current
   position; conv state likewise.

## Topology decision (once per context, not per ubatch)

In the graph ctor (`src/models/glm5next.cpp:767`):

```
scoring  = cparams.n_ctx > n_select        # n_select = 2048 + 4 - 1 = 2051
inp_kp   = build_inp_kpool(...)  (always, if the model has an idx cache)
```

- Gated on **`n_ctx`, not on the current n_kv**. n_kv grows as the cache
  fills, so gating on it would flip the graph topology mid-run (a new graph
  per crossing ubatch). n_ctx is fixed for the context's lifetime, so the
  topology is decided once. The cost: a context configured above 2051 runs the
  indexer scoring even while the cache is still short, where it selects every
  visible pool - wasted work, never a wrong answer;
  `llama_kpool_select_k` clamps the budget to the pools that exist.
- Below n_select the reference selects every visible position, so the **dense
  path is not an approximation of the sparse one - it is the same function**.
  `scoring` only gates the selection; the indexer key/gate **store is not
  gated**: skipping it would leave the first 2051 positions of every sequence
  with no indexer state, and the first ubatch to cross the threshold would
  pool cells that were never written.

## Per-layer work, in graph order

For each of the 45 layers (graph nodes are built for the whole ubatch at
once; the graph is a single ggml computation graph executed per backend):

### KDA layers (34)

`build_kda_layer` over `n_seq_tokens` tokens:

- 3 projection GEMMs (`[8192, 4096] x [4096, n_tokens]`) - these dominate the
  KDA layer's FLOPs; on CUDA they are `mul_mat` with quantized weights (mmq
  kernels), on CPU the quantized dot ops.
- One depthwise conv over `[24576, n_seq_tokens]` (`ggml_ssm_conv`, kernel
 4) + SiLU. Conv state snapshot for rollback.
- L2 norms, gate/beta GEMMs (small).
- The recurrence: `build_delta_net` takes the **chunked** path for
  `n_seq_tokens > 1` - `build_delta_net_chunking` (chunk size 64: intra-chunk
  matmul-based scan + inter-chunk state carry), or the single fused
  `ggml_gated_delta_net` node in chunked mode when `fused_gdn_ch` probed on
  (CUDA has the kernel; see 05). The KDA-specific fix in this branch
  (reshape of `g` to `[S_k, 1, H_v, n_seqs]`) lives in the shared
  `build_delta_net` path and applies here.

> **Correction (2026-08-28 validation):** Chunk size is **16 for KDA**, 64 for GDA. Code is `const int CS = kda ? 16 : 64` (`delta-net-base.cpp:61`). The "64" above is the GDA (non-KDA) size; the 34 glm5next KDA layers use 16. Also line reference `src/models/glm5next.cpp:767` in Topology section is actually at `:23` (definition `glm5next_n_select`) and ctor use at `:770-773`; `O(n_kv x n_tokens)` for the pool map is only the two F16 masks (`sel/cand [n_kv,n_tps]`); `pool_bias [n_pools,n_tps]` is `O(n_kv/kpool * n_tokens)` and `pool_cells [kpool*n_pools]` is `O(n_kv)` (`llama-graph.cpp:3603-3637`, `kpool.h:96-103`). The `hc_pre [24,16384]` below is stored as `[16384,24]` (`hc_dim x hc_mix_dim` in `deepseek4.cpp:359`).
- Output gate GEMM, RMS norm, sigmoid gate, `Wo` GEMM.

Node budget note: a KDA layer costs 182 nodes + ~16/token, hence the raised
`graph_max_nodes` for this arch.

### DSA layers (11)

`build_dsa_layer`:

- `qr = rms(Wq_a x)` (shared with the indexer), `q = Wq_b(qr)`, absorbed
  through `Wk_b`; `kv = rms(Wkv_a x)` - four GEMMs, all standard mul_mat.
- `build_indexer`:
  - store: key + gate projections (2 GEMMs `[128, 4096]`), pack,
    `cpy_k` into the indexer cache at the ubatch's cells. F16 cache.
  - score (when `scoring`):
    - `ggml_get_rows` over `[256, n_kv, n_stream]` gathers
      `kpool x n_pools` pool-member rows per stream into F32 - this is an
      O(n_kv)-ish read of the whole indexer cache **per DSA layer** (11x per
      ubatch), the main reason the map (not the cache) is what gets shared.
    - pool softmax: permute, `+ ape`, `ggml_soft_max` over 4 slots,
      `sum_rows(keys * probs)` -> `[128, n_pools]` pooled keys.
    - `iq = W_attn_q_b(qr)`, `w = W_proj(x)` (F32-prec GEMM).
    - scores: fused `ggml_lightning_indexer` (one node; the CUDA wmma/vec
      kernel or the CPU loop) or the unfused mul_mat/relu/sum chain with
      `+ pool_bias`.
    - `ggml_top_k(pool_score, select_k)` over the pool axis, then
      `ggml_get_rows(pool_cells, sel)` to expand pools to 2048 cell indices.
- `build_attn_sparse`: dup the sel_mask, `set_rows` the 2048 cells to 0,
  `+ cand_mask`, `+ kq_mask`, then `build_attn_mha` - full-cache MLA
  attention `[512-latent K, 64 heads of q]` with the sparse mask, V-absorbed
  through `Wv_b`, then `Wo`.
  - With flash attention on, this is `ggml_flash_attn_ext` under the F16 mask
    (the mask type path that exists precisely so no cast is needed).
  - With flash attention off, it is the unfused KQ-masked attention
    (softmax over `[n_kv, n_tokens]` per head after the KQ GEMM) - the path
    the branch's measurements were taken on.
- Dense alternative (short context or `scoring` off): the same attention with
  the plain causal KQ mask and no indexer scoring. Same FLOPs for the
  attention, minus all the pooling machinery.

### FFN

- Layers 0-2: dense SwiGLU (clamped).
- Layers 3-44: `build_moe_ffn` - router GEMM `[288, 4096]`, sigmoid +
  noaux_tc top-8 (the topk-moe op; `exp_probs_b` biases selection only),
  expert GEMMs over the stacked expert tensors (8 of 288 experts per token,
  `n_expert_used` 8), norm_topk_prob, x2.5 routing scale, clamped SwiGLU -
  plus the shared expert, added. On CUDA the MoE GEMMs are the
  `topk-moe`/mmq path; on CPU the same ops with CPU dot kernels.

### mHC glue

Around every sublayer: the hc_pre GEMM (`[24, 16384]`) + Sinkhorn (fused
`ggml_dsv4_hc_comb` or 20-iter row/col softmax chain) + stream mixing; after
each sublayer the hc_post (fused or explicit). Small GEMMs, but 2x45x2
instances.

## End of prefill

After layer 44: optional layer-embedding extraction (`build_hc_mean` of the
wide state), row-gather for the requested output positions, `build_hc_mean`
(unweighted mean over the 4 streams), output RMS norm, lm_head GEMM
`[n_vocab=154880, 4096]`. Logits out.

## Cost shape of a prefill ubatch (where the time goes)

1. MoE expert GEMMs (42 sparse layers x 8+1 experts) - largest FLOPs term.
2. MLA attention over the full cache with sparse mask - O(n_tokens x n_kv)
   reads of the 512-wide latent, 11 layers.
3. KDA: projections + chunked recurrence, 34 layers.
4. Indexer: 2 projection GEMMs + full-cache member gather + scoring + top-k,
   11 layers. The member gather reads the whole F16 indexer cache each time.
5. Host: the pool map (O(n_kv x n_tokens) stores) - why it is built once per
   ubatch and shared across layers.
6. mHC glue: many small ops, mostly fused.
