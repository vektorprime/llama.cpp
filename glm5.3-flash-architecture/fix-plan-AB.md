# Plan: fix GLM-5.3-Flash prefill (issues A and B)

Branch: `glm5next-fast-prefill`. Status: VALIDATED (V1-V5 complete, results
folded into the design below; see "Validation outcomes" at the end).
Nothing here is implemented yet.

## Problem restatement

A. The 11 DSA layers do **dense** O(n_tokens x n_kv) flash attention over the
   whole cache with a scattered -inf mask; the FA kernel has no
   fully-masked-tile skip. Sparsity (2048 of n_kv) saves nothing.
   Evidence: `src/llama-graph.cpp:3743-3746` (K = full cache),
   `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1802-1873` (no tile skip).

B. The 34 KDA layers run prefill through the **single-kernel serial** CUDA
   recurrence (`ggml/src/ggml-cuda/gated_delta_net.cu:63`, loop over all
   n_tokens; line 180: `//TODO: Add chunked kernel for even faster pre-fill`).
   The parallel chunked path that exists on CPU (`build_delta_net_chunking`,
   `src/models/delta-net-base.cpp:16`) has no CUDA kernel, and the
   `fused_gdn_ch` probe (16 tokens) cannot distinguish the two.

Both fixes are env-gated with the existing mask/serial path kept as fallback,
and both reuse existing ggml ops. No new model arch changes.

---

## Fix A: compact-gather sparse attention for DSA

### Design

Replace "dense FA + full-cache mask" with "gather the selected keys into a
compact per-query KV, FA over that with a tiny per-entry mask".

Facts the design relies on (each mapped to a validation item):

1. The top-k output already exists as I32 cell indices
   `[select_k=512, n_tps, n_stream]` (pools), expanded to
   `[2048, n_tps, n_stream]` cells in `build_indexer`
   (`src/models/glm5next.cpp:545-560`).

> **Correction (2026-08-28 validation):** Line range has drifted to `582-600` (`select_k=llama_kpool_select_k` and `get_rows(pc3,sel_flat)` -> `r*select_k=2048`); same block, arithmetic `r=4, select_k=512=>2048` holds.
2. A selected pool with **finite** pool_bias is fully resident and all its
   members are causally visible (visibility is tested at the pool's last
   member). So per-entry validity = (pool score finite). No full-cache KQ
   mask is needed in the gather path. (V2 re-verifies against the kpool
   filler, incl. unified-cache runs, seq_rm holes, seq_cp sharing.)
3. The query's own tail pool is always attended and is fully host-derivable
   (cells at positions `[tail_start, tail_start + r)` of the query's
   sequence). The host already computes `tail_start` per query. (V2)
4. `ggml_get_rows` can do both gathers with existing 2-D/3-D contracts.
   Result type/stride behavior and quantized-src support on CUDA+CPU must be
   confirmed. (V1)
5. FA accepts the per-query-batch shape: K `[512, n_kv_sel, 1, B]`,
   Q `[512, 1, 64, B]`, mask `[n_kv_sel, 1, 1, B]` F16 (this is exactly the
   decode shape with batch B). `n_kv_sel` alignment against
   `FATTN_KQ_STRIDE` and CPU FA support must be confirmed. (V1)

Shapes (unified cache, n_stream = 1, B = n_tps; the non-unified multi-seq
case falls back to the existing mask path in v1 - see Risks). Padding: the
CUDA FA selector requires `K->ne[1] % FATTN_KQ_STRIDE == 0` with
FATTN_KQ_STRIDE = 256 (ggml-cuda/fattn-common.cuh:9), so the gathered KV
width is 2048 + kpool + 252 = **2304** (9 x 256), not 2052. The 252 pad
entries are -inf-masked.

> **Correction (2026-08-28 validation):** Width `2304` pads `2052` (2048+4). `n_select` is `2048+4-1=2051` (`glm5next.cpp:28`), so tail width in the plan is `kpool` (4) vs HF's `kpool-1` (3) - off by one entry of pad, same final 2304 (extra pad masked with -inf). No functional impact. Also `K_cache` zero-stride view `nb[2]=nb[3]=0` as written violates `ggml_get_rows` contract when `B>1` (`ggml.c:3895` asserts `a->ne[2]==b->ne[1]` etc. on shape, not stride). In-tree never uses this; first implementation must runtime-check and fallback to flat `[512,2048*B]` gather + view back (feasible, same bytes) or iterate per batch. See R-A7 below, which correctly flags this as untested.

```
P        = top_k pool indices            [512, B]      (existing)
scores_g = pool scores                   [n_pools, B]  (existing, GPU)
K_cache  = MLA K cache, viewed [512, n_kv, 1, B] with nb[2]=nb[3]=0
          (zero-stride broadcast over the B batches - get_rows rows are
          addressed as i01*nb01 + i11*nb02 + i12*nb03, so the zero strides
          are harmless; V1 flagged this as untested in-tree, first
          implementation step is a runtime check of it)

K_sel    = get_rows(K_cache, top_k_cells [2048, B, 1, 1])  [512, 2048, B]  F32
K_tail   = get_rows(K_cache, tail_cells  [kpool, B, 1, 1]) [512, kpool, B] F32 (new host input)
K        = concat(K_sel, K_tail, pad) along ne[1]          [512, 2304, B]
K16      = cast(K, F16)  (CUDA FA has no F32-K kernel; see R-A1)

# per-entry mask: 0 where the pool score was finite, -inf otherwise
sb       = get_rows(permute(scores_g) [B, n_pools], P [512, B, 1, 1]) [B, 512] F32
m_sel    = ternary(sb > -inf, 0.0f, -inf)                     [B, 512]
m_tail   = from host tail_cells validity (0 / -inf)           [B, kpool]
mask     = concat(m_sel, m_tail, -inf pad) -> [2304, B] -> view [2304, 1, 1, B]  F16

Q        = view(q [512, 64, B]) as [512, 1, 64, B]
out      = ggml_flash_attn_ext(Q, K16, K16, mask, kq_scale, 0, 0)
out      = permute -> mul_mat(v_mla = Wv_b) -> permute          (as in build_attn_mha #if 1 branch)
out      = reshape -> Wo
```

Duplicates: **none possible** (V2). The query's own tail pool is never in the
top-k set: while incomplete its pool_bias is -inf (the last-member visibility
test drops it whole), and when it is complete the tail set is empty
(`(q+1) % kpool == 0`), i.e. it competes in top-k normally. This matches the
reference exactly: HF `modeling_glm5_next.py` does
`cat([topk_indices, tail_indices])` with `tail = arange(kpool-1)` masked by
visibility and `output_width = index_topk + kpool - 1` (the -1 is the maximum
tail width, not a dedup allowance), padding the rest with -1. vLLM (PR
#53906, `expand_pools_and_append_tail` into a -1-prefilled buffer consumed by
a sparse-MLA gather) confirms gather is the reference execution model.

Why the mask comes from gathered scores (not from the full-cache mask
pipeline): the per-entry value is `mask_all[cell[e,b], b]` - a diagonal
gather that `ggml_get_rows` cannot express. The score-finiteness derivation
avoids it entirely, and the full-cache `sel_mask`/`cand_mask`/`set_rows`
graph ops are skipped in the gather path (host tensors unchanged; they are
still built for the mask fallback path).

### Steps

- A1. Host input `tail_cells` `[kpool, B]` I32 + `tail_valid` `[kpool, B]`
      F16 (0/-inf) in `llm_graph_input_kpool`
      (`src/llama-kv-cache-kpool.h:180-208`, created in `build_inp_kpool`,
      `src/llama-graph.cpp:3556-3641` - mirror the existing new_tensor +
      set_input lines). Filled in `llama_kv_cache_set_input_kpool`
      (`src/llama-kv-cache-kpool.cpp:71-419`), in the per-query loop where
      `tail_start` is computed (line 360). Tail definition (V2): resident
      cells at positions `[tail_start, q]`; when `(q+1) % kpool == 0` the
      tail is EMPTY (pool complete, competes in top-k). Missing members get
      cell 0 + -inf. There is no `cell_at(pos)` helper: build the inverse
      map from `pos_at` (lines 273-276), hoisted once per sequence, O(n_kv).
- A2. New graph path in `build_dsa_layer`
      (`src/models/glm5next.cpp`): when `scoring` and
      `cparams.glm5next_sparse_gather` is enabled and the shape preconditions
      hold (unified or single-seq, FA on, K type supported by get_rows),
      emit the compact-gather graph above; else the existing
      `build_attn_sparse` mask path. Implement the FA+v_mla tail as a small
      helper next to `build_attn_mha` (reuse the `#if 1` v_mla spelling).
- A3. cparam + env gate: `glm5next_sparse_gather` default AUTO: on when
      flash_attn is on and the preconditions hold; env
      `LLAMA_GLM5NEXT_SPARSE=gather|mask` forces either (pattern:
      `LLAMA_FUSED_LID_DISABLE` in `src/llama-context.cpp:239-247`).
- A4. Parity test: extend `tests/test-glm5next-memory.cpp` (or the existing
      arch test fixture) to run the DSA layer with `mask` vs `gather` on CPU
      and assert logits are identical or within a fixed eps, for: short
      context (dense path, both modes must agree exactly), context just
      above 2051 (under-full top-k, i.e. invalid entries present), and a
      seq_rm-hole context (cand_mask case).
- A5. Measurement: llama-bench prefill sweep (2K/8K/32K/64K) mask vs gather
      on the user's box; nsys to confirm the FA kernel time drops to the
      2048/ n_kv fraction and no new bottleneck appears (K_sel gather
      traffic, compute buffer).

### Risks / resolved items (validation outcomes)

- R-A1 (RESOLVED, now the top risk): `ggml_get_rows` result is **F32 forced**
      (ggml.c:3900-3905, "TODO: implement non F32 return"), regardless of src
      type. K_sel = 512 x 2304 x 512 x 4B = 2.4 GB, plus the F16 cast K16
      (1.2 GB) CUDA FA requires (no F32-K FA kernel). V5 compute-buffer
      estimate: gather peak ~4.4 GB per DSA layer vs the mask path's
      0.10/0.39/3.1 GB at n_kv = 32K/128K/1M - the gather path uses MORE
      compute buffer at short/medium context. Mitigations, in order:
      (a) the gather path pays for itself only at long context (FA FLOPs drop
      2048/n_kv), so gate AUTO on n_kv (e.g. only above ~16K cells);
      (b) the buffer is reserved once at `sched_reserve` and auto re-reserved
      if a graph no longer fits (ggml-alloc.c:1052-1067), so OOM risk is
      contained - but A5 must measure it on the user's box before default-on;
      (c) later: F16 dst for get_rows (the CUDA kernel already can write
      F16 dst, getrows.cu:413-433; only the ggml.c result-type pin blocks
      it) as a small follow-up if headroom fails.
- R-A2 (RESOLVED): get_rows src types on CUDA include F16, BF16, F32 and
      Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 plus the k-quants and MXFP4
      (getrows.cu:302-408); CPU is a superset (ops.cpp:5019-5061). A q8_0 K
      cache works; the result is F32 either way (R-A1).
- R-A3 (RESOLVED): pad to 2304 (FATTN_KQ_STRIDE = 256, fattn-common.cuh:9);
      2052/2056/2080/2176 all fail the selector and ABORT on CUDA
      (fattn.cu:573-574).

> **Correction (2026-08-28 validation):** Line `fattn.cu:573-574` has drifted; selector check is now `fattn.cu:422-439` (`gqa_opt_applies = mask && max_bias==0 && K->ne[1]%256==0`) which returns `BEST_FATTN_KERNEL_NONE -> GGML_ABORT` at `fattn.cu:238`. Substance unchanged.
- R-A4 (RESOLVED): CPU FA exists (ggml-cpu ops.cpp:9209) and accepts
      mask ne[1]==1, batch B, K head dim 512, incl. -inf skipping; no
      fallback needed.
- R-A5 (KEPT): multi-seq non-unified prefill falls back to the mask path in
      v1. The get_rows contract (ggml.c:3895-3905) ties idx's trailing dims
      to the src slice, so per-stream gathers would need the zero-stride view
      trick per stream - do it only if llama-embedding-style workloads need
      it.
- R-A6 (RESOLVED): reference masks invalid top-k entries by masking the pool
      scores before topk and padding the output with -1 (HF
      `modeling_glm5_next.py`); our score-finiteness per-entry mask is the
      same semantics executed after the gather.
- R-A7 (NEW from V1): the zero-stride broadcast view of the K cache
      (nb[2]=nb[3]=0) is not used anywhere in-tree. First implementation
      step: a 20-line standalone check (or a test-backend-ops get_rows case)
      that a zero-strided-batch get_rows matches the naive gather on both
      backends. If it misbehaves, fallback: gather [512, 2048*B] in one pass
      over the 2-D cache view and view it back to [512, 2048, B] (same bytes,
      one extra view, no copy - the idx would be a flat [2048*B, 1, 1, 1]...
      NO: idx must index per-batch; the flat form needs idx [2048*B] with
      batch baked into the index values, which is exactly what the per-batch
      idx encodes - verify feasibility during A2, flag if it needs a cont).

---

## Fix B: fast KDA prefill recurrence

Ordered: measure first (B0), dispatch fix (B1) only if it wins, kernel
(B2) only if still needed. The unfused chunked **op chain** already exists
and is backend-agnostic; the fused single kernel may simply be the wrong
choice for n_tokens > 1.

### B0 - Measurement (needs a small env override added first)

V3 confirmed: **no** env override exists for the fused GDN flags (only
`LLAMA_FUSED_LID_DISABLE`, llama-context.cpp:242-246), and the
`fused_gdn_ch` probe tests at n_tokens_per_seq = 16 - exactly the size where
serial-vs-chunked cannot be distinguished.

- B0.1. Add `LLAMA_FUSED_GDN_DISABLE` in the llama-context.cpp:239-247
      pattern block (set `fused_gdn_ar`/`fused_gdn_ch` and their auto flags
      to false; cparams are internal-only, `src/llama-cparams.h`, no public
      API change).
- B0.2. Measure prefill t/s (test-llama-archs glm5next fixture, fixed seed,
      or the real model) with the flag on vs off at ubatch = 64 / 256 / 512
      / 1024 tokens, CUDA and CPU. nsys: time of the `gated_delta_net`
      kernel vs the op-chain kernels - the chain is **solve_tri (n=16 fast
      path), cumsum (F32), tri, diag, mul_mat, ...** (there is no ssm_scan in
      this chain; V3 corrected the plan). This decides B1.
- B0.3. The fused kernel's actual grid for GLM5NEXT KDA (H=64, S_v=128,
      n_seqs=1) is (64, 1, 32) = 2048 blocks, block (32, 4, 1),
      launch_bounds(128, 2); one warp per state column, rows_per_lane = 4,
      single serial loop over all n_tokens (gated_delta_net.cu:5-184).

### B1 - Dispatch fix (minimal, only if B0 shows the op chain wins)

- B1.1. In `llm_build_delta_net_base::build_delta_net`
      (`src/models/delta-net-base.cpp:428-449`): for `n_seq_tokens > 1`,
      use the fused op only when `n_seq_tokens <= 16` (one KDA chunk; CS =
      `kda ? 16 : 64`, delta-net-base.cpp:61 - V3 corrected the plan's "64"),
      else use `build_delta_net_chunking` (the op chain). Keep `fused_gdn_ar`
      for n_tokens == 1 unchanged.
- B1.2. HARD CONSTRAINT (V3): when `n_rs_seq > 0` (recurrent-state rollback
      mode) `build_recurrent_attn` bypasses `build_delta_net` entirely and
      ALWAYS emits the fused op with K snapshots (delta-net-base.cpp:549-607),
      and the op chain writes **no** snapshot slots. So B1.1's dispatch must
      be unreachable in rollback mode (it is, structurally - the bypass is
      upstream of build_delta_net) and the B1 parity test must cover
      n_rs_seq = 0 only; `test-recurrent-state-rollback` (runs on the qwen35
      fixture, exercises the shared op's snapshot path) must still pass.
- B1.3. Node budget (V5): the chain costs ~510 nodes/KDA layer at 512 tokens
      (32 chunks x ~14 ops + ~55 one-time) vs ~2 fused; 34 layers adds ~17k
      to the ~25-30k current graph -> ~45k, well under the 91,840 budget
      (`max(n_tokens*160, 64*n_tensors)`, n_tensors ~= 1435).
- B1.4. Snapshot fusion (`ggml_cuda_try_gdn_cache_fusion`, ggml-cuda.cu:2742)
      is fused-op-only; the chain path writes state via a plain `ggml_cpy`
      (~4 MiB/KDA layer/ubatch extra). Irrelevant in rollback mode (B1.2) -
      record the delta in the B0.2 nsys run anyway.
- B1.5. Parity test: home is `test-llama-archs` (self-contained synthetic
      GLM5NEXT fixture at :219-233, decodes 128 tokens per device, NMSE vs
      CPU at 1e-4). Pattern: two contexts from the same model, one with
      `LLAMA_FUSED_GDN_DISABLE` set, fixed seed, compare logits externally
      (the cparams-mutation two-ctx pattern exists in
      test-recurrent-state-rollback.cpp:12-17). Also add the one-line
      `test_gated_delta_net(GGML_TYPE_F32, 64, 128, 128, 1, 1, false, true)`
      KDA-shape case at test-backend-ops.cpp:~10500 (op-level, fused only).

### B2 - Chunked CUDA kernel (only if B0/B1 leaves KDA a top cost)

Implements the `//TODO: Add chunked kernel for even faster pre-fill`
(`gated_delta_net.cu:180`). V4 extracted the exact algorithm and cross-checked
it element-wise against the published chunked delta rule (Yang et al.,
arXiv:2406.06484, eqs 7-9; flash-linear-attention `chunk_delta`) - the CPU
chain is the standard UT/WY transform; a B2 kernel must match **the code, not
the paper**, with these deltas:

- B2.1. Per chunk (KDA: CS = 16; GDA: CS = 64), per (head, seq), with
      alpha_u = exp(g_u), prefix G(t) = cumsum(g), D(a,b) = exp(G(b)-G(a))
      for b>=a, Dbar(t) = exp(G(t)), S = state at chunk start ([key x value]):
        A[t,r]  = beta_t * (k_t . k_r weighted by D(r,t))  (strict lower,
                  KDA builds it via the permuted [S_k, CS, CS, CHB] decay mask
                  and transpose(mul_mat(mask*k_b, k)));
        U      = (I + A)^-1  as  X = solve_tri(I+A, -A); U = X + I
                  (the solve_tri identity, not in-place forward substitution);
        v_tilt = U (beta * v);
        W      = U (beta * Dbar * k);
        v_u    = v_tilt - W S;
        o      = (Dbar * q) S^T  +  lower(D * (q k^T)) v_u;
        S'     = Dbar(last) S  +  (k * D(->last))^T v_u.
      Partial chunks: zero-padded (g=0), output truncated by view.
      GOTCHA (V4): ggml's ne0 axis is the display COLUMN (M[row,col] =
      T[ne0=col, ne1=row]) - mixing this up makes A look transposed.
- B2.2. Invariants to preserve: state layout `[S_v, S_v, H, n_seqs]`
      (ne0 = key axis); q/k already L2-normalized by the layer, never
      re-normalized in the op; the 1/sqrt(S_k) scale (pre-multiplied into q
      on CPU, post-multiplied onto o in the existing CUDA kernel - pick one
      and match it in tests); decay convention: the chain does cumsum-then-
      single-exp vs the existing CUDA kernel's per-token expf - bit parity
      with both is impossible, target the CPU chain (cumsum+exp); KDA g
      broadcast on the KEY axis (the fix at delta-net-base.cpp:330-338; the
      old reshape decayed the value axis - invisible for GDA where
      g->ne[0]==1); snapshot slots: slot 0 = state after the LAST token,
      slot s = after token (n_tokens-1-s), n_written = min(n_tokens, K),
      `state_slot_stride` = S_v^2*H*n_seqs (or the fused-cache view stride);
      GDA mode (g->ne[0] == 1, CS = 64, scalar decay, A = plain
      mul_mat(k, k_b) x mask) must keep working - the op is shared with
      qwen3next and friends.
- B2.3. Parity vs the CPU chain (primary target) and vs the fused AR kernel
      (looser eps, different rounding) at n_tokens in {16, 32, 64, 128, 512,
      1024}, random states, KDA + GDA modes, K in {1, 4} (snapshot slots).
- B2.4. test-backend-ops already parameterizes the op (PP 64/256/512/1024,
      K>1 cases at test-backend-ops.cpp:10090-10493); add the GLM5NEXT KDA
      shape line (B1.5) so the new kernel is covered by the standard
      per-backend run.

### B risks

- R-B1: the op chain may be launch-overhead-bound on CUDA for small chunks;
      B0 measures before B1 commits.
- R-B2: the snapshot/rollback feature is the subtlety; any recurrence
      change (B1 or B2) must keep the snapshot slots bit-compatible with the
      fused kernel (tests exist).
- R-B3: B2 is the only step with real new kernel code; it is deliberately
      conditional.

---

## Cross-cutting (V5-validated)

- Node budget (91,840 at n_tokens=512): A is net ~0 nodes/DSA layer
  (replaces the mask path's 8-9 nodes with 7-8); B1 brings the total to
  ~45k. Headroom 3x. `src/llama-context.cpp:2304-2310`.
- Compute buffer: largest new live tensor is K_sel + K16 (~3.6 GB at B=512,
  see R-A1); reserved once at `sched_reserve`, auto re-reserve on graph
  mismatch (ggml-alloc.c:1052-1067). The gather path's AUTO mode therefore
  gates on n_kv (the FA-FLOP win must exceed the buffer cost): off below
  ~16K cells, on above; `LLAMA_GLM5NEXT_SPARSE` overrides.
- Env surface: `LLAMA_GLM5NEXT_SPARSE` (new, A3), `LLAMA_FUSED_GDN_DISABLE`
  (new, B0.1; the only existing pattern is `LLAMA_FUSED_LID_DISABLE`,
  llama-context.cpp:239-247). cparams are internal-only
  (`src/llama-cparams.h`) - zero public API change.
- Build: no new source files, no CMake changes (V5.6).
- Rollout order: A first (bigger win, no kernel code; R-A7 check -> A1 ->
  A2 -> A3 -> A4 parity -> A5 measurement, and A5's numbers decide whether
  AUTO turns on by default), then B0 (env var + measurement), B1 only if the
  op chain wins, B2 only if KDA is still a top nsys cost. Each lands as its
  own commit set with the parity tests green and the user's
  llama-bench/nsys numbers recorded. Commit messages written by the human.

## Validation outcomes (V1-V5, completed)

| track | verdict | material corrections folded into this plan |
|-------|---------|--------------------------------------------|
| V1 get_rows + FA feasibility | PASS with 3 design changes | (1) K cache must be viewed with a zero-stride broadcast over the batch for get_rows (untested in-tree -> new R-A7, runtime check first); (2) K/mask width pads to 2304 (FATTN_KQ_STRIDE = 256), not 2052; (3) get_rows result is F32-forced -> 2.4 GB + 1.2 GB F16 cast, ~4.4 GB peak/layer -> new gating + measurement gate (R-A1). FA shape [512, 2304, 1, B] + [512, 1, 64, B] + mask [2304, 1, 1, B] selects the mma_f16 gqa64 kernel on GA102; CPU FA supports the shape; v_mla tail reuses build_attn_mha's #if 1 block verbatim; cell id == cache row confirmed |
| V2 kpool invariants + reference | PASS | finite pool_bias => fully resident AND all members <= query position (incl. tail pool); **no top_k/tail duplicates possible** (tail pool -inf while incomplete, empty tail when complete) - the -1 in n_select is the max tail width, matching HF `modeling_glm5_next.py` (`cat([topk, tail])`, pad -1 to topk+kpool-1); vLLM PR #53906 confirms gather is the reference execution model; tail cells host-derivable in the existing per-query loop (no cell_at helper - build inverse from pos_at); unified-cache foreign-pool/empty-member edges closed |
| V3 delta-net paths + tests | PASS | KDA chunk size is 16 (not 64); chain ops all have CUDA kernels (solve_tri n=16 fast path, cumsum F32-only, tri, diag - no ssm_scan); **no** fused-GDN env override exists (must add); rollback mode (n_rs_seq>0) bypasses build_delta_net and always uses the fused op with snapshots - the chain writes none, so B1 is structurally safe there but parity tests must use n_rs_seq=0; snapshot cpy fusion is fused-op-only; test homes identified (test-llama-archs for parity, one-line test-backend-ops KDA case, test-glm5next-memory is NOT ctest-registered - manual with the tiny GGUF from tests/glm5next_make_tiny_gguf.py) |
| V4 chunked math | PASS | exact UT/WY equations extracted (above, B2.1), element-wise match with the published chunked delta rule (arXiv:2406.06484) with 6 code-specific deltas to preserve; the KDA g-axis bug (pre-fix reshape decayed the value axis) explained; bit-parity target = CPU chain (cumsum+single-exp), not the CUDA kernel's per-token expf |
| V5 budget + buffer + infra | PASS | node budget 91,840 vs ~45k after B1 (fits, 3x headroom); Fix A net node count ~0 (replaces, not adds); compute buffer is interval-allocated per graph and reserved once at sched_reserve with auto re-reserve - OOM contained but must be measured (A5); cparams are internal-only (no public API change); no CMake changes needed for any step |

Remaining unknowns (resolve during implementation, not by more planning):
R-A7 zero-stride get_rows behavior on both backends (20-line check); the
A5 compute-buffer + t/s numbers on the user's 6-GPU box; B0.2 fused-vs-chain
verdict (decides whether B1/B2 happen at all).
