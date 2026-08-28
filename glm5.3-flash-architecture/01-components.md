# Components

Everything is in `src/models/glm5next.cpp` unless noted. The graph struct
derives from `llama_model_deepseek4::graph`, which it inherits for two reasons:
the mHC mixer code (Sinkhorn, hc_pre/hc_post, `build_hc_mean`) and access to
`build_delta_net` for the KDA layers. Only the final residual collapse differs
from DeepSeek-V4: glm5next uses an **unweighted mean** of the streams, not a
learned gated head (there is no `hc_head` tensor).

## Graph shape (one pass over the ubatch)

```
inpL = repeat(tok_embd(x), 4)                      [4096, 4, n_tokens]  mHC streams
for il in 0..44:
    cur  = hc_pre(inpL)                             mix 4 streams -> 1
    cur  = rms(cur);  cur = KDA(il) or DSA(il)
    inpL = hc_post(cur, inpL)
    cur  = hc_pre(inpL);  cur = rms(cur)
    cur  = dense-FFN(il<3) or MoE+shared
    inpL = hc_post(cur, inpL)
out  = mean(inpL over 4 streams) -> rms -> lm_head  [n_vocab, n_tokens]
```

`build_arch_graph` asserts `gtype != LLM_GRAPH_TYPE_DECODER_MTP`: the NextN
tensors load (with `--mtp`), but the MTP graph itself is not implemented yet.

## mHC wide residual

`llama_model_deepseek4::graph::build_hc_pre` / `build_hc_post`
(`src/models/deepseek4.cpp`).

State is 4 exact copies of the hidden state (`[4096, 4, n_tokens]`, no
scaling, no one-hot into stream 0). Before each sublayer:

1. RMS-norm the flattened `[16384, n_tokens]` (model rms eps).
2. One GEMM `hc_*_fn` `[16384 -> 24]` (hc_mix_dim = (2+4)*4 = 24 rows).
3. The 24 rows split into `pre[4]`, `post[4]`, `comb[16]`; each gets its
   affine (scale row + base chunk), `pre`/`post` go through sigmoid
   (`pre` scaled +bias by `hc_eps`, `post` scaled by 2).
4. `comb[16]` reshaped to `[4,4]` passes through **Sinkhorn**: row softmax,
   then 20 row/column normalizations with eps 1e-6 - a doubly-stochastic
   mixing matrix, which is the "manifold constraint" that keeps the wide
   residual from exploding.

> **Correction (2026-08-28 validation):** "20 row/column normalizations" is imprecise. Code does `soft_max` then `add eps` then `norm_cols` once, then 19 iterations of `norm_rows`+`norm_cols` (`deepseek4.cpp:320-347`, `iters=hparams.dsv4_hc_sinkhorn_iters=20`). Total is 20 column normalizations + 19 row normalizations (+ initial softmax), not 20+20.
5. Sublayer input = sum of `pre[i] * stream[i]` (elementwise, per stream).

After the sublayer, `hc_post` writes the new streams as
`comb @ old_streams + post * sublayer_out` (fused form:
`ggml_dsv4_hc_post`; the unfused fallback is the same ops spelled out).

Fused forms exist for `hc_pre`, `hc_comb` (the Sinkhorn) and `hc_post`
(`ggml_dsv4_hc_*`, `LLM_FUSED_OP_DSV4_HC_*`); they are on by default and
probed per backend at context creation. The graph calls
`ggml_build_forward_expand` on the residual/post/comb tensors before the FFN
so op offload does not pull the mHC state onto the expert weights' backend.

## KDA layer (34 layers)

`build_kda_layer`. KDA = Gated DeltaNet with **per-key-channel** decay (the
"kda" in kimi-linear/kimi-k3). 64 heads, head dim 128, `d_inner` 8192.

> **Note (2026-08-28 validation):** Head count/dims (64/128, d_inner 8192) are GGUF values for the released `zai-org/GLM-5.3-Flash` checkpoint (`conversion/glm5next.py:120-124` from `linear_cfg`). Code derives them (`glm5next.cpp:150-152` `d_inner=head_dim*n_head`, `hparams.n_embd_head_kda`), not compile-time constants. Other widths would load if GGUF provides them.

Per token sequence, in order:

1. `Q, K, V = {Wq, Wk, Wv} x inp` (three `[8192, n_tokens]` GEMMs; the f/g
   gates below read `inp`, **not** the projected values).
2. Concat to `qkv [24576, n_seq_tokens, n_seqs]` and run **one depthwise
   conv1d** (kernel 4, `ggml_ssm_conv`) over the concatenated channels, then
   SiLU on the conv output. The conv weights are three separate checkpoint
   tensors concatenated at graph build. One conv (instead of three) leaves the
   conv state as one contiguous block, which is what recurrent-state rollback
   (`build_conv_state`) snapshots.
3. View back into Q/K/V; L2-normalize Q and K per head with the reference's
   own constant 1e-6 (`ggml_l2_norm`, divides by `max(norm, eps)`; the
   reference uses `sqrt(sum + eps)` - close, not bit-exact).
4. Decay gate `g`: `g = -5.0 * sigmoid( exp(A_log) * (f_b(f_a(x)) + dt_bias) )`,
   `[128, 64, n_tokens]`. `ssm_a` stores **-exp(A_log)** (the kimi-k3 sign
   convention; the converter verifies the sign at export), so the graph does
   `g *= ssm_a; g = sigmoid(-g); g *= gate_lower_bound`. `gate_lower_bound`
   (-5.0) is a multiplicative scale on the sigmoid, **not** a clamp - a
   missing key would silently select the softplus branch instead.
5. Update gate `beta = sigmoid(W_beta x)`, one per head.
6. Recurrence. State is `[128, 128, 64, n_seqs]` per KDA layer (key x value
   per head; `n_embd_s` = 128*128*64 = 1.05M floats/seq/layer).
   `build_recurrent_attn` -> `build_delta_net` dispatches on ubatch size:
   - `n_seq_tokens == 1` (decode): fused `ggml_gated_delta_net` (AR mode) when
     `fused_gdn_ar` is probed on, else the explicit autoregressive op chain.
   - `n_seq_tokens > 1` (prefill): fused (chunked mode) when `fused_gdn_ch`
     is on, else `build_delta_net_chunking` (inter-chunk scan + intra-chunk
     recurrence, chunk 64).

> **Correction (2026-08-28 validation):** Chunk size is **16 for KDA**, 64 for non-KDA (GDA). Code is `const int CS = kda ? 16 : 64` (`delta-net-base.cpp:61`). The "chunk 64" above applies to GDA models (qwen3next); glm5next's 34 KDA layers use 16. State shape `64` heads is correct for glm5next (`d_inner 8192=64*128` via `llama-hparams.cpp:213-217`), but the value is data-driven from GGUF (`n_head`/`n_embd_head_kda`), not a hard-coded assert.
   The branch's only delta-net change: the decay `g` is reshaped to
   `[S_k, 1, H_v, n_seqs]` so the per-key-channel decay broadcasts over the
   **key** axis. For GDA models (qwen3next) `g->ne[0]` is 1 and both spellings
   agree; for KDA it is load-bearing.
7. Output gate is low-rank: `gate = g_b(g_a(x))` `[128, 64, n_tokens]`
   (kimi-k3 has a single full-rank `ssm_g` instead).
8. `o` is RMS-normalized over head dim with one shared weight per layer,
   multiplied by `sigmoid(gate)` (a plain sigmoid gate, not the SiLU default
   of FusedRMSNormGated), then `Wo` projects `[8192 -> 4096]`.

## DSA layer (11 layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)

`build_dsa_layer`. DeepSeek-style sparse attention over an **absorbed,
nope-only MLA** cache.

- `qr = rms(Wq_a x)` - the 1536-wide q LoRA residual. The indexer reuses this
  (see below), so it is computed once in the DSA layer.
- `q = Wq_b qr` -> `[256, 64, n_tokens]`.
- `kv = rms(Wkv_a x)` -> `[512, n_tokens]` - the single MQA latent.
- Absorption: `q = Wk_b * q` so q lives in the 512-wide latent space, and
  attention is computed against the latent directly. V is a **view** of the
  same latent rows; `Wv_b` expands 512 -> 256 per head **after** softmax. This
  is what `is_mla()` means for the cache: one K row of width
  `kv_lora_rank` (512) per cell, no separate V, no rope half anywhere
  (`n_rot() == 0`; the reference hardcodes `qk_rope_head_dim = 0`).
- Scale is `1/sqrt(qk_head_dim)` = 1/sqrt(256), the pre-absorption head size;
  scaling by sqrt(512) would be a different model.
- `Wo` projects `[16384 -> 4096]`.
- The attention itself is either `build_attn` (dense: causal mask over the
  whole cache) or `build_attn_sparse` (selected cells; see below). Both end in
  `build_attn_mha` with `v_mla = Wv_b`.

## Pooled lightning indexer (per DSA layer)

`build_indexer` + `src/llama-kv-cache-kpool.{h,cpp}`. This is the part GLM-5.3
does not have in DeepSeek-V3.2/DSV4: the indexer does not score individual
keys, it scores **pools of `kpool` = 4 consecutive positions** and top-k is
taken over pools, then expanded to members.

Checkpoint-side tensors: `indexer_attn_k` (key projection, `[4096, 128]`),
`indexer_comp_wgate` (pool gate, a **second independent** projection of the
hidden state), `indexer_comp_ape` (learned per-slot position table
`[128, 4]`), `indexer_attn_q_b` (query from the shared q LoRA residual,
`[1536, 32*128]`), `indexer_proj` (head weights, `[4096, 32]`),
`indexer_k_norm` (a LayerNorm **with bias**, eps 1e-6).

Per ubatch, per DSA layer:

1. **Store (always, even on the dense path).** `ik = LN(W_attn_k x)` and
   `gate = W_wgate x` are packed as two "heads" in one row,
   `[128, 2, n_tokens]`, and copied into the **indexer cache** at the same
   cell ids as the attention cache (the hybrid memory forces the slot layout
   to match). The gate must be cached beside the key: a pool is rebuilt from
   its members' keys+gates, and a pool is only rebuilt once its member tokens
   have left the batch.
2. **Score (only when `scoring`).**
   - Gather each pool's 4 member rows (`ggml_get_rows` over
     `[256, n_kv, n_stream]`, driven by the host-side `pool_cells` map; the
     gather yields F32 even though the cache is F16).
   - Pool key = weighted average of member keys, weights = softmax over the
     slot axis of (member gates + `ape[slot]`), where slot m = position p % 4.
     `ape` is added **pre-softmax**, indexed by logical slot. Incomplete pools
     are neutralized by `pool_bias` (-inf), never by a NaN - unlike the
     reference, no -inf ever enters the compressor softmax.
   - `iq = W_attn_q_b(qr)` -> `[128, 32, n_tps, n_stream]` (MQA: 32 heads
     share the pooled keys). No RoPE.
   - Head weights `w = W_proj(x)`, computed at **F32 precision**
     (`GGML_PREC_F32`) - on vLLM a bf16 head gate moves a logit by ~1e-2,
     enough to swap two near-tied pools. Both scale constants
     (d_idx^-0.5 and n_heads^-0.5) are folded in as one 1/sqrt(128*32) scale.
   - Pool score = sum over heads of `relu(q.k) * w` + `pool_bias`. The ReLU
     sits between the per-head dot and the head weighting (the head weights
     are sign-unconstrained and the sum is not a convex combination, so
     moving the ReLU is a different function). No Hadamard rotation: the
     engines rotate q/k only to spread magnitude ahead of fp8; the semantic
     reference has none, and (Hq).(Hk) == q.k anyway.
     - Fused: one `ggml_lightning_indexer(iq, pool_k, w, pool_bias_f16)` node
       doing the dot -> relu -> head-sum -> mask-add, mask add for free.
     - Unfused: `mul_mat` -> permute -> `relu` -> `* w` -> `sum_rows` ->
       `+ pool_bias`.
   - **Top-k over POOLS**: `select_k = min(n_pools, 2048/4 = 512)`. This is
     the only correct spelling - a cell-level top-k of width 2048 is wrong
     because the ReLU drives most pool scores to exactly 0.0, tie groups span
     pools, and `ggml_top_k` is explicitly unordered among equals (the CPU op
     deliberately swaps its first two results; the CUDA op declares
     `determinism::not_guaranteed`). Measured on the TinySparse fixture, the
     cell-level form leaves a partial pool on 7.5% of query rows; the pool
     form leaves none. Pool-level selection also removes the CPU/CUDA
     tie-break divergence structurally: backends may disagree which equal
     pools come out, never whether a pool is taken whole.
   - Expand each selected pool to its 4 member **cells** by gathering whole
     rows of `pool_cells` - the reference's `pool_indices[batch_idx, selected]`.
     Result: I32 `[2048, n_tps, n_stream]` attention-cache cell indices.

### Why the top-k indices are cell indices, and how they become a mask

The attention cache is addressed by **cells** (slot ids), not positions.
Under eviction/shifting/seq_cp, cells of one pool are not adjacent, not
ordered, and (under a unified cache) not even owned by one sequence. The
position -> pool -> cell mapping therefore cannot be derived in the graph; it
is built host-side per ubatch (see 02-memory.md) and enters the graph as plain
input tensors (`pool_cells`, `pool_bias`, `sel_mask`, `cand_mask`).

`build_attn_sparse` (`src/llama-graph.cpp:3643`) turns the I32 cell list into
a KQ mask:

1. Start from a per-layer **copy** of `sel_mask` (F16, `[n_kv, n_tps, 1,
   n_stream]`). It holds 0.0 only on the query's own incomplete trailing pool
   (GLM always attends to its tail - `index_kpool_always_select_tail`) and
   -inf elsewhere. Starting from sel_mask (not a fresh -inf fill) is what
   keeps the top-k budget a whole number of pools. The copy matters:
   `ggml_set_rows` writes through, and the mask input is shared by all 11 DSA
   layers - scattering into it directly would let layer 7 inherit layer 3's
   selections.
2. `ggml_set_rows(mask, 0.0f, top_k)` - unmask exactly the selected cells.
   The value is a constant 0, never the cell's own bias, so the scatter can
   not erase a tail cell that top-k also named.
3. `+ cand_mask` - additively re-masks any over-budget selection that fell
   outside the reference's candidate set (complete visible pools union tail).
4. `+ kq_mask` - re-apply causality/occupancy/padding. Load bearing: it keeps
   an empty, future, or foreign-sequence cell masked no matter what top-k
   returned (ggml_top_k returns `select_k` ordinals even when fewer pools have
   finite scores, which is the NORMAL state during prefill).
5. Full-cache MLA attention (`build_attn_mha`) under the resulting sparse
   mask, with `Wv_b` absorption as in the dense path.

So sparse DSA is implemented as **dense attention over the whole cache with a
scattered mask**, not a gathered KV. The gather happens on 4-member pools for
the indexer only; the attention itself reads the standard MLA K cache.

## MoE FFN (layers 3..45; layers 0-2 dense)

`build_layer_ffn`.

- Dense lead (first `first_k_dense_replace` = 3 layers): plain parallel
  SwiGLU FFN, `n_ff` = 12288. The same SwiGLU **clamp** applies (the
  reference routes dense and MoE through one MLP class).
- Sparse layers: `build_moe_ffn` with 288 experts (`[4096, 2048, 288]`
  stacked gate/up/down), top-8 (`num_experts_per_tok`), sigmoid scoring with
  `noaux_tc` (`exp_probs_b` biases the top-k **selection** only; the weights
  are the unbiased sigmoid scores; `n_group` = 1 so group masking is a no-op),
  `norm_topk_prob` on, `routed_scaling_factor` 2.5 applied to the routed
  weights only. Plus one shared expert (`[4096, 2048]`, unscaled), added to
  the routed output.
- Clamp: with `swiglu_clamp` set (10.0 for every layer), the up projection is
  clamped to `[-10, 10]` and the gate to `[-inf, 10]` **before** the
  activation, then a plain `silu(gate) * up` (the dsv4/glm5next spelling of
  clamped SwiGLU in `build_ffn`/`build_moe_ffn`). This differs from
  `ggml_swiglu_oai`, which clamps the same way but then adds 1 to the up
  branch (a gpt-oss detail this model does not share).

## NextN (MTP) block

`num_nextn_predict_layers` = 1. Layer 45 is a full DSA decoder layer (own
attn/FFN/indexer tensors) plus an `eh_proj` `[8192 -> 4096]` that fuses the
previous token's embedding with the trunk hidden state, two norms
(`enorm`, `hnorm`) and a `shared_head_norm`. It keeps the plain residual (no
mHC mixer) and shares the trunk's embeddings and lm_head (the checkpoint has
no separate copies; the loader accepts them if an export adds them). The
converter supports `--mtp-only`/trunk-only GGUF splits. **Running** it is not
implemented: `build_arch_graph` asserts the gtype is not MTP.

## Vision (mtmd)

`tools/mtmd/models/glm5next-vision.cpp` + `clip.cpp`
(`PROJECTOR_TYPE_GLM5NEXT`). The tower is the GLM-OCR ViT unchanged: 24
blocks, hidden 1024, 16 heads, patch 14 x temporal 2, spatial merge 2, out
4096, attention bias on, q/k norms, no learned position embeddings, no
post-conv patch norm, SwiGLU MLPs. The only structural difference from GLM-4V
is the clamp: gate/up are clamped before the SwiGLU in **both** the per-block
MLP and the merger (`FFN_SILU_CLAMP`, `clip.vision.swiglu_limit` = 10.0).
`clip_graph_glm5next::build()` asserts its preconditions and delegates to
`clip_graph_glm4v::build()`.

Preprocessing: `mtmd_image_preprocessor_glm5next` implements the reference's
smart resize (pixel budget between `image_min_pixels` and
`image_max_pixels`, both sides of the comparison divided by
`patch*merge`^2, so the temporal factor cancels for still images; dimensions
rounded to the patch multiple), with **bicubic** resampling to match the
reference (PIL BICUBIC; neither mtmd filter matches torchvision exactly).
Token limits 16..8000.

Converter: `Glm5NextVisionModel` derives from `Glm4VVisionModel` (every tensor
already maps through the GLM-4V entries) and adds `spatial_merge_size` +
`swiglu_limit`. The `fe95953cc` fix makes `clip_graph_glm4v` skip the
post-conv norm when `norm_embd` is absent (GLM-OCR has none; `build_norm`
would still normalize with a null weight).

## Tokenizer

glm4 pre-tokenizer (chatglm-bpe family) with `ignore_merges = true`
(`81e3e6716`). Without it, greedy BPE cannot reach some vocab entries: e.g.
`" 王"` needs the merge (Ġ, çİĭ) but greedy order stops three tokens short.
It is triggered by whitespace before CJK and inflates mixed Chinese-English
text by ~13%. Not applied to chatglm-bpe (ChatGLM3), which shares the pre
type but is a different tokenizer.

## Precision policy

`src/llama-quant.cpp` (`869e87879`): these tensors are excluded from
quantization and stay at source precision (~1 GiB total, noise against a
100-240 GB quant):

- mHC mixers `hc_attn_fn`, `hc_ffn_fn`
- indexer `indexer_compressor_ape`, `indexer_compressor_gate` (underscore
  spelling), `indexer.proj`, `indexer.attn_k`, `indexer.attn_q_b` (dot
  spelling) - both spellings are required; a single "indexer." test misses the
  compressor pair
- KDA recurrence gates `ssm_f_a/b`, `ssm_g_a/b`, `ssm_beta`

Rationale: quantizing these perturbs *which pools the indexer selects* and
*how much state each KDA step retains* - errors that compound over a sequence
instead of averaging out. Deliberately NOT pinned: the MLA projections
(`attn_q_a`, `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`) - precision-sensitive,
but the release recipe pins them to q8_0 via `--tensor-type`, and the shipped
quants were measured in that configuration. The indexer **cache** is likewise
kept float (F16) at runtime: the gate head feeds a softmax, so `-ctk q8_0`
downgrades it to F16 with a warning (`llama-model.cpp`).

Loader guards (`load_arch_hparams`): requires q LoRA, `n_rot() == 0`, KDA head
dim, negative `kda_gate_lower_bound`, `indexer_top_k % kpool == 0`, per-layer
`head_count_kv` array (0 = KDA), at least one KDA and one DSA layer, and
types the model (45 layers + 4096 + 288 experts -> `313B_A17B`).
