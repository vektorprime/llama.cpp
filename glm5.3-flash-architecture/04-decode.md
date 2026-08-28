# Token generation (decode)

Decode = one ubatch with `n_seq_tokens == 1` (or a few tokens for MTP-style
batching, which this model does not expose yet). One graph build, one compute,
one new token. The graph is rebuilt per token (the KQ mask and the pool map
change every step) but has the **same topology** as prefill when `n_ctx >
2051` - the sparse path stays sparse; nothing flips at the decode boundary.

## What changes between prefill and decode

| piece | prefill | decode |
|-------|---------|--------|
| KDA recurrence | chunked scan (chunk 64) or fused chunked | **AR fused** `ggml_gated_delta_net` (K=1) or explicit AR op chain |

> **Correction (2026-08-28 validation):** Chunked scan size is **16 for KDA**, 64 for GDA (`delta-net-base.cpp:61` `CS = kda ? 16 : 64`). Prefill for the 34 glm5next KDA layers uses 16; "chunk 64" above is the GDA size.
| KDA conv | full conv over the chunk | conv with state carry (same `ggml_ssm_conv`, n_tokens=1) |
| DSA attention | n_tps queries x n_kv keys | 1 query x n_kv keys |
| indexer scoring | n_tps queries over all pools | 1 query over all pools - same ops, n_tps = 1 |
| top-k | `select_k` pools per query row | `select_k` = 512 pools for the single query |
| host pool map | O(n_kv x n_tps) | O(n_kv) - cheap |

## Per-token walk

1. **Host side.** New cell assigned for the new token in the attn and indexer
   caches (same slot layout). `k_idxs` written. Pool map refilled:
   - the new token's pool is now one member fuller; when it completes (every
     4th token), it becomes a selectable pool and enters `pool_cells` with
     finite `pool_bias` for future queries.
   - the query's own trailing pool is its own incomplete pool (its own token
     included) - `sel_mask` forces it.
   - This is O(n_kv) per token (two mask rows + pool bias row), negligible
     next to the GPU work until very long context.

2. **Embedding + mHC init.** Token embedding repeated into 4 streams.

3. **Layers 0-44, in order.**

   - **KDA layers (34).** The recurrence is the point of decode: each layer
     reads its `[128,128,64,1]` state (1.05M floats), updates it in place

> **Correction (2026-08-28 validation):** State width `64` is the KDA-head count notation for `d_inner 8192`; in-memory `n_embd_s` for `n_embd 4096` is `128*128*32=524288` floats per seq/layer (`llama-hparams.cpp:217`) depending on which `n_head` alias is used. Doc's 1.05M (=64*128*128) and ~8 MB/layer (~270 MB for 34 layers) are correct only under the 64-head counting; under the 32-head `n_embd_s` it is ~4 MB/layer (~136 MB total). Both are data-driven from GGUF, not hard-coded.
     (fused kernel or op chain), writes the new state back, and produces
     `o = x * s`-style output for the single token.
     - Fused (`fused_gdn_ar` probed on, CUDA): one `ggml_gated_delta_net`
       node, K=1; the output is a view of the result tensor and the new state
       is a second view - the state update and the readout share the kernel.
     - Unfused: the explicit AR chain (state *= decay broadcast, delta rule
       `s += beta (v - s k^T) k^T` per key channel, readout `o = s^T q`).
     - The per-key-channel decay broadcast fix (`g` reshaped to
       `[S_k, 1, H_v, n_seqs]`) matters here exactly as in prefill.
     - Memory traffic per KDA layer per token: 2 x 1.05M floats state
       read+write + weights. 34 layers x ~8 MB = ~270 MB of state traffic per
       token on top of the weights - this is what makes the KDA/MLA hybrid
       fast at long context: the 34 layers' cost is **constant in context
       length**.

   - **DSA layers (11).** Each:
     - store the new key latent into the MLA cache (1 cell, 512-wide) and the
       new indexer key+gate into the indexer cache (1 cell, 256-wide F16).
     - indexer scoring for the single query:
       - gather all `n_kv/4` pools' members (full-cache F32 gather - the
         dominant memory term of the DSA layer at long context; note it grows
         linearly in n_kv, unlike the 34 KDA layers),
       - pool softmax + pooled keys,
       - `iq` (1 query, 32 heads) and `w` (F32 GEMM),
       - scores: fused `ggml_lightning_indexer` (CUDA wmma/vec or CPU loop;
         with n_tps = 1 this is the probe's native case) or the unfused chain,
       - `ggml_top_k` picks 512 of the ~n_kv/4 pools; expand to 2048 cells.
     - `build_attn_sparse`: mask = sel (tail) + scatter(2048) + cand + causal,
       then attention of the single query against the whole cache under the
       sparse mask, Wv_b absorption, Wo.
       - Flash-attention path: `ggml_flash_attn_ext` with the F16 mask.
       - Unfused path: the KQ GEMM for one query is 512 x n_kv per head.
     - Net effect: each DSA query touches **2048 of n_kv keys** (plus its own
       tail) instead of all n_kv - that is the whole point of DSA, and why
       this model's serving cost at 1M context is roughly 2048/n_kv of dense.
     - Near-tie caveat: the fused kernel sums the 32 head terms in a different
       order than the unfused graph, so a near-tied pool top-k can come out
       different. Escape hatch: `LLAMA_FUSED_LID_DISABLE=1` forces the
       unfused graph (and the f32 mask path).

   - **FFN.** Same as prefill: router, top-8 experts + shared, clamped
     SwiGLU. Decode MoE is latency-bound on the 9 expert GEMMs per layer
     (CUDA: mmq/topk-moe kernels; CPU: quantized dots).

4. **Head.** 4-stream mean, RMS norm, lm_head GEMM `[154880, 4096]`, logits
   for the one token; sampler picks the next token, which becomes step 1's
   input.

## State that persists across tokens

- MLA K cache: +1 cell per token (512-wide latent per DSA layer, 11 layers).
- Indexer cache: +1 cell per token (2 x 128 F16 per DSA layer, 11 layers).
- KDA recurrent state: overwritten in place (per KDA layer, per seq).
- KDA conv state: rolled forward by 1 (per KDA layer, per seq).
- Pool map: rebuilt from the cell table every step (no persistent state of its
  own - it is a pure function of the attn cache's cells + positions + the
  ubatch, which is what makes rollback/shift/seq_rm correct for free: rewind
  the cells, and the map rebuilds itself).

## MTP / NextN

`num_nextn_predict_layers` = 1; the tensors load with `--mtp`, but
`build_arch_graph` refuses `LLM_GRAPH_TYPE_DECODER_MTP` - speculation is not
wired up in this branch yet. The NextN block would be a 46th DSA layer
(consuming the trunk's wide state via `eh_proj`) with its own indexer;
`index_share_for_mtp_iteration` in the HF config hints at sharing the
indexer cache across the MTP iteration.

## Practical notes

- Graph reuse: the KQ mask and pool-map tensors are per-ubatch inputs, so the
  graph is rebuilt per token, but the ubatch shape (1 token) is stable, so the
  rebuild is a cheap node re-emission, not a re-plan.
- `n_ctx` must exceed 2051 for the sparse path to exist at all; with a small
  context the model silently runs dense MLA everywhere (still correct - it
  is the identical function below the selection width).
- Context shift: supported only if the indexer cache can shift (it can - its
  keys carry no positional encoding; the shift bookkeeping exists only to
  clear the pending per-cell delta).
- Save/restore sessions include the indexer cache; restoring without it
  selects wrong cells (see 02-memory.md).
