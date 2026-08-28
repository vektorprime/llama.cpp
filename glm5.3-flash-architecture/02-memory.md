# Memory: three caches + the host-side pool map

## The hybrid memory

`llama_memory_hybrid` (extended in `2db8295ee`, per-sequence pooling in
`204fa7003`) now owns up to three memories, wired up for GLM5NEXT in
`llama_model::create_memory` (`src/llama-model.cpp`):

| memory | layers | contents | type |
|--------|--------|----------|------|
| `mem_attn` | DSA layers (`!is_recr`) | MLA latent, one K row of width `kv_lora_rank` (512) per cell. `is_mla()` suppresses the V allocation; V is a graph-level view of K | `params.type_k` (e.g. q8_0) |
| `mem_recr` | KDA layers (`is_recr`) | recurrent state `[128, 128, 64, n_seqs]` per layer + conv state (one contiguous 24576-wide block per layer) | f32 states |

> **Correction (2026-08-28 validation):** Recurrent state shape is `[128, 128, 32, n_seqs]` for the released checkpoint, not 64. `n_embd_s = n_embd_head_kda^2 * n_head` (`llama-hparams.cpp:213-217`); with `n_embd 4096`, `head_dim 128`, `n_head = 32` (since `d_inner 4096 = 32*128` would be the per-layer? Actually glm5next uses `n_head 64` for `d_inner 8192`, but `n_embd_s` is `128*128*32=524288` because `n_head` in that formula is the *indexer* head count alias - see `llama-hparams.cpp:217` and `tests/test-glm5next-memory.cpp:379`. Doc's `64` would give 1.05M and matches `01-components.md:81` 64-head notation, but the in-memory `n_embd_s` for `n_embd 4096` is 524K (128*128*32). The `24576` conv block (`3*(3-1)*4096`) is correct.
| `mem_idx` | DSA layers (`!is_recr`) | indexer key **and** pool gate, two heads of width `indexer_head_size` (128) per cell: cell row = `[key(128) | gate(128)]` | F16 forced (see below) |

`mem_idx` only exists when `indexer_head_size > 0`. It is shaped as an MQA
`llama_kv_cache` with `n_head_kv = 2` (1 for every other arch; 0/absent for
non-indexer archs) and `n_embd_head_k_full = 128`. Because the hparams view
still looks MLA-ish, no V is allocated.

Type rule: the gate head feeds a softmax (the pool compressor), so quantizing
it would be far more sensitive than quantizing a key. `create_memory` keeps
the indexer cache at `params.type_k` **unless that type is quantized**, in
which case it drops to F16 with a warning. Nothing in the graph may see a
quantized indexer row: `build_indexer` asserts the cell holds a key head and a
gate head and that they are adjacent in memory.

### Slot layout sharing

The indexer cache is a **side buffer addressed by the attention cache's
cells**. `llama_memory_hybrid::init_batch` copies the attention cache's
`slot_info` into the indexer context (`heads_idx = heads_attn`) instead of
letting it allocate its own slots. If the two allocated independently they
drift apart when the context is rewritten between turns, and the top-k indices
- which are read against the attention mask - would point at the wrong cells.
`llama_memory_hybrid_context::apply()` asserts both caches cover the same
window (`get_n_kv()` and `get_n_stream()` equal).

Everything that touches a sequence touches all three caches: `clear`,
`seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`, `memory_breakdown`, and
state save/restore. State save/restore includes the indexer cache **on
purpose**: indexer keys are not recomputable from the attention cache (the
pool gate is a separate projection, and a pool is only rebuilt from cached
members), so a restored session that skipped them would select the wrong
cells. Shifting: `get_can_shift()` is false if the indexer cache cannot shift;
the indexer keys carry no positional encoding, so a shift has nothing to
correct in them, but the pending per-cell delta must still be cleared.

> **Correction (2026-08-28 validation):** `state_write/read` skips `mem_attn`/`mem_idx` when `flags & PARTIAL_ONLY` but always includes `mem_recr` (`llama-memory-hybrid.cpp:241-257`). Also `llama_memory_hybrid_context::init_full/update` (`:277,292`) ignores `ctx_idx` status (only combines `ctx_attn+ctx_recr`); indexer init failure would be silently ignored.

### Unified KV cache

`kv_unified` is supported and worth understanding, because pooling is defined
on positions and positions are only unambiguous **within one sequence**:

- Non-unified cache: each sequence has its own stream and its own cells array;
  one stream == one sequence, and the pool table is trivial.
- Unified cache: every sequence of the ubatch lives in **one** stream and one
  cells array. The stream's pool table is therefore **partitioned**: each
  sequence gets a contiguous run of pool slots and rebases inside it.
  `pool_bias` is -inf outside a query's own run, which stops a query from
  selecting a foreign sequence's pools. `seq_has` membership (not just
  occupancy) is what keeps another sequence's keys out of a pooled run.

The runs are **packed**, not one full-width table per sequence. A full-width
table would multiply the pool axis (and hence the whole score tensor, since
every pool is scored against every query) by the sequence count -
`llama-embedding`, which asks for `n_seq_max` 256 with a unified cache,
reserved 286 GB of compute buffer that way before the fix.

## The pool map (host-side, per ubatch)

`src/llama-kv-cache-kpool.cpp` fills five graph inputs, once per ubatch, via
`llm_graph_input_kpool::set_input` (which `llama_context` calls for every
graph build; the tensors are plain `ggml_set_input` host buffers):

```
k_idxs     I32  [n_tokens]                       indexer-cache cell per token (store)
pool_cells I32  [kpool*n_pools, n_stream]        pool slot, member -> cell (0 if absent)
pool_bias  F32  [n_pools, n_tps, n_stream]       0 / -inf per (pool, query)
sel_mask   F16  [n_kv, n_tps, 1, n_stream]       0 on the query's own tail pool, else -inf
cand_mask  F16  [n_kv, n_tps, 1, n_stream]       0 on complete visible pools OR tail
```

> **Correction (2026-08-28 validation):** `k_idxs` is **I64**, not I32. Code is `GGML_TYPE_I64` (`llama-kv-cache.cpp:1397` `n_tokens` int64, `llama-graph.cpp:3570` `build_input_k_idxs`, `llama-kv-cache.cpp:1465` `int64_t*` data). The other four types/shapes are correct. Also `bo_vis` in step 4 below is `max(0, tail_start/r - b_base)` (`kpool.cpp:365`) - doc omits the `max(0,)` clamp (behavior identical for `q>=0`).

`n_pools = llama_kpool_n_pools(n_kv, kpool, n_ps) = n_kv/kpool + 2*n_ps`: the
shared budget plus 2 slots of rebasing slack per sequence. `n_stream` is 1
for unified, `n_seqs_unq` otherwise. `n_tps` = tokens per stream. The map is
sized on the **ubatch**, not on `n_seq_max` (`n_seqs_unq` is already part of
the graph-reuse key, so shapes hold while a graph is reused).

Filling (`llama_kv_cache_set_input_kpool`), per stream, per sequence:

1. **Partition** `[TAG_KPOOL_PACK]`: for each sequence, find the min/max
   resident pool (`pos/r`) of its cells; that range is the sequence's run. If
   all runs together exceed `n_pools` (only possible when `seq_cp` shared a
   prefix, so one cell is pooled by several sequences), the runs are cut from
   the top: each sequence keeps its **newest** pools - the same cut a large
   hole from `seq_rm` forces.
2. **Rebase**: pool ordinal = absolute `p/r`, which can far exceed
   `n_kv/kpool`, so each sequence rebases on `b_base = max(b_min, b_max -
   (n_run-1))`. Grouping is untouched (every member shifts together). Anchoring
   at `p/r` (not at the first resident key, as HF does) follows vLLM/SGLang and
   is the only anchor that keeps a pool's identity stable between the prefill
   that built it and the decodes that read it.
3. **Fill**: `pool_cells[bo*r + p%r] = cell`. A pool is usable only if all `r`
   members are resident (and exactly once each - two cells claiming one
   position overwrite each other, so over-filled pools are unusable too).
   Incomplete pools point their slots at **cell 0** (a clamp, since
   `ggml_get_rows` has no negative sentinel) and are neutralized by the masks,
   never by a NaN.
4. **Per query**: `tail_start = (q+1)/r*r` (the query's own incomplete pool,
   including its own token, is always attended); `bo_vis = tail_start/r -
   b_base` (visibility is tested at a pool's LAST member, so a pool the query
   straddles is dropped whole; position alignment reduces the test to
   `bo < bo_vis`). Then:
   - `sel_mask[q]`: 0 where `visible && pos >= tail_start`, else -inf.
   - `cand_mask[q]`: 0 where `visible && (pooled || tail)`, else -inf. This is
     the reference's candidate set: what the top-k budget may overrun but must
     never escape. It exists because `ggml_top_k` always returns `select_k`
     ordinals - during prefill that means arbitrary -inf-tie ordinals (on the
     TinySparse fixture, 20 of 20 query rows are under budget). The causal KQ
     mask kills the spills that are empty/foreign/future; `cand_mask` kills
     the one the KQ mask cannot: a resident, causally visible cell in an
     *incomplete* pool below the tail (unreachable while positions are
     contiguous, reachable after a `seq_rm` hole).
   - `pool_bias[q]`: 0 where the pool is completely resident and visible, else
     -inf; the query's own trailing pool included, so no budget is spent on
     it. Derived directly, NOT gathered at the pool's last member cell: an
     incomplete pool has no resident last member, its slot points at cell 0,
     and gathering would let the pool inherit cell 0's validity and compete
     for budget with a finite score.

`sel_mask`/`cand_mask` are **F16** (`e88c92d02`): they only ever hold 0.0 and
-inf, both exact in f16, and the masks are two `[n_kv, n_tps, n_stream]`
inputs that live for the whole ubatch - 2 GiB each at n_ctx 1M, ubatch 512 in
f32. Every consumer takes f16: with flash attention it is the KQ mask's own
type; without, `ggml_add` gives the result src0's type and f16+f32 -> f16 is a
supported bcast on CUDA and CPU.

Per-ubatch cost: O(n_kv x n_tokens) host stores for the two masks plus
O(n_pools x n_tokens) for pool_bias. This is shared by all 11 DSA layers -
rebuilding per layer would cost ~11x67M float stores per ubatch at 128K cells
and 512 tokens and dominate prefill on its own. Sharing `pool_bias`/masks
across layers is correct only while every indexer layer sees the same
candidate set; glm5next's `indexer_types` are all "full", a mixed
(windowed) model would need one map per window.

## What the graph reads from it

- Store: `mctx_idx->cpy_k(packed, k_idxs, il)` in `build_indexer` - runs on
  the dense path too. Gating the store on the sparse path would leave the
  first `n_select` (2051) positions of every sequence with no indexer state,
  and the first ubatch to cross the threshold would pool cells that were never
  written.
- Score: `mctx_idx->get_k` (the whole indexer cache), `pool_cells`,
  `pool_bias` (or its f16 cast), then in `build_attn_sparse` the masks and the
  attention cache `k_idxs`/KQ mask.

## Capacity / node budget

`llama_context::graph_max_nodes` gives GLM5NEXT the same elevated budget as
kimi-k3 (`max(n_tokens*160, 64*n_tensors)` instead of `n_tokens*40`): each KDA
layer costs 182 nodes + ~16/token, so the 34 KDA layers alone need
6.2k + 31.9*n_tokens before DSA or MoE is counted.
