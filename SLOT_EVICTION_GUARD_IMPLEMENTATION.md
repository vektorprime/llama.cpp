# Slot Eviction Guard - Implementation

Branch: `multiple_slots_llama`
Scope: `llama-server` automatic slot selection
Status: implemented, NOT yet compiled or benchmarked (this fork's AGENTS.md says analyze/edit only)

--------------------------------------------------------------------------------

## Summary

### The issue

When several agents share one `llama-server` started with `--parallel N`, the
server's prefix cache ping-pongs between them and triggers full re-prefills.

Concrete trigger (the opencode case): a main coding session runs on a slot with
a large warm KV cache (tens of thousands of tokens). A sub-agent (general,
explore, summary, title, ...) fires with a different system prompt. The server
picks a slot by Longest Common Prefix (LCP). The sub-agent request partially
matches the main session's slot (they share some prefix), so the LCP picker
selects the main slot and evicts most of its cache to serve the sub-agent. When
the main agent resumes, its cache is gone and the whole context is re-prefilled.
The result is two full prefills where there should have been zero.

The known workaround is to pin `id_slot` per agent from the client side
(opencode model config). That works but is manual, client-specific, and easy to
get wrong. We want the server to handle it automatically.

### The solution

Make automatic slot selection eviction-aware. The selection code already
computes `f_keep` (the fraction of the chosen slot's cache that would survive
the match) but only uses it to decide whether to save the evicted cache, never
to avoid the eviction. The fix adds a guard: when the best LCP match would
discard most of a warm slot (`f_keep` below a threshold) and a free cold slot is
available, route the request to the cold slot instead. The warm slot stays
cached for its owner; the diverging request lands on its own slot. No client
changes, no `id_slot` needed.

The behavior is gated behind a new opt-in flag `--slot-eviction-guard`
(default off, so stock behavior is unchanged) with a tunable
`--slot-keep-threshold` (default 0.50). Explicit `id_slot` requests are never
affected.

--------------------------------------------------------------------------------

## Background: how slot selection works today

The selector is `server_context::get_available_slot(const server_task & task)`
in `tools/server/server-context.cpp`. Order of preference:

1. If `task.id_slot != -1`, use that exact slot (explicit pin).
2. Else, if `slot_prompt_similarity != 0.0f` (default 0.1, so on by default),
   pick the non-processing slot whose cache best matches the incoming prompt.
3. Else fall back to the least-recently-used non-processing slot.

The match metric in step 2, per candidate slot:

    sim_cur = LCP(slot.cached_tokens, task.tokens) / task.tokens.size()

That is the fraction of the INCOMING prompt covered by the slot's cache. The
slot with the highest `sim_cur` (above the `slot_prompt_similarity` threshold)
wins. After a winner is chosen, the code computes:

    f_keep = (sim_best * task.tokens.size()) / winner.cached_tokens.size()
           = LCP / winner.cached_tokens.size()

`f_keep` is the fraction of the WINNER slot's existing cache that the match
preserves. Today `f_keep` is used only to set `update_cache` (save the
about-to-be-lost cache into the prompt-cache pool). It does not influence which
slot is chosen.

--------------------------------------------------------------------------------

## Root cause

`sim_cur` maximizes coverage of the incoming prompt and is blind to how much of
the target slot's cache is destroyed. A small diverging request (sub-agent) can
out-score an empty slot against a big warm slot because it shares a short common
prefix, then evict the warm slot. `f_keep` already measures that damage but is
computed too late to matter for routing.

Bounce sequence (today):

    main turn    -> slot0 [main 50k cache]            warm, correct
    sub fires    -> LCP picks slot0 (partial match, low f_keep)
                    slot0 cache evicted, sub prefilled
    main resumes -> slot0 cache gone -> re-prefill 50k    (the cost)
    next sub     -> may bounce again

--------------------------------------------------------------------------------

## The fix: how it works

Inside the `slot_prompt_similarity` branch of `get_available_slot`:

1. While scanning slots, remember the first free (empty-cache) slot as
   `cold_slot`. This is a candidate home for a diverging request.
2. After the best LCP slot `ret` is chosen, compute `f_keep` as before.
3. Eviction guard. If ALL of the following hold:
     - `slot_eviction_guard` is enabled, and
     - `task.id_slot == -1` (automatic selection only; never override a pin), and
     - `f_keep < slot_keep_thold` (the match would discard most of the warm
       slot's cache), and
     - `cold_slot != nullptr` (a free slot exists),
   then set `ret = cold_slot` and `update_cache = true`. The request is served
   on the cold slot; the warm slot is left intact for its owner. Setting
   `update_cache = true` lets the prompt-cache pool restore this request's own
   prior cache into the cold slot if it was paged out earlier.
4. Otherwise, behavior is exactly as before (reuse `ret`, set `update_cache`
   when `f_keep < 0.5`).

The downstream LRU fallback runs only when `ret == nullptr`, so a guard reroute
(which sets `ret`) correctly skips it. The cold slot flows through the existing
`update_cache` save/restore block the same way the LRU path already does, so no
new cache-handling code is introduced.

Routed behavior (with guard, `--parallel 2`):

    main turn    -> slot0 [main 50k cache]            warm, preserved
    sub fires    -> f_keep low + slot1 cold -> route to slot1, prefill sub
    main resumes -> slot0 still warm -> 0 re-prefill      (fixed)
    next sub     -> slot1 now warm -> LCP keeps sub on slot1

### Formulas

    sim_cur = LCP(slot.cached_tokens, request.tokens) / request.tokens.size()
    f_keep  = LCP(winner.cached_tokens, request.tokens) / winner.cached_tokens.size()

`sim_cur` in [0,1] = how much of the request is already cached.
`f_keep`  in [0,1] = how much of the slot's cache survives reuse.
Low `f_keep` means high eviction cost; that is what the guard avoids.

### Invariants and limits

- Default off: with `--slot-eviction-guard` absent, output is byte-identical to
  upstream. The only added cost is tracking one pointer in the scan loop.
- Explicit `id_slot` requests are never rerouted.
- Needs `--parallel >= 2`: with one slot there is no cold slot to fall back to,
  and the guard is a no-op.
- If every slot is warm (no cold slot), the guard backs off and the original
  best-LCP slot is used; the eviction is then unavoidable and that slot is still
  the least-bad choice.
- Only active when `slot_prompt_similarity != 0.0f` (the default path). If the
  user disables similarity selection (`-sps 0`), selection is pure LRU and the
  guard does not apply.

--------------------------------------------------------------------------------

## Configuration

New server flags (examples group `LLAMA_EXAMPLE_SERVER`):

- `--slot-eviction-guard`
  Enable the guard. Off by default.
- `--slot-keep-threshold F`
  Minimum fraction of a slot's cached tokens that must survive a match before
  the slot is reused; below this a free cold slot is preferred. Default 0.50.

Example:

    llama-server -m model.gguf --parallel 2 --slot-eviction-guard

Optionally tune the threshold:

    llama-server -m model.gguf --parallel 2 --slot-eviction-guard --slot-keep-threshold 0.4

With this, opencode (or any client) no longer needs per-agent `id_slot`. The
main session keeps its slot; sub-agents segregate onto the free slot
automatically.

--------------------------------------------------------------------------------

## Build and validation status

Not compiled in this fork (AGENTS.md: analyze and edit only). When validating on
the build server (10.0.0.188, RTX 3080), the server target is enough since the
change is server-side only:

    cd /home/user/llm/multiple_slots_llama/llama.cpp
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_UI=OFF
    cmake --build build --target llama-server -j

Quick checks after build:

- `llama-server --help | grep slot-eviction-guard` should list the flag.
- Start with `--parallel 2 --slot-eviction-guard`, drive a main + sub-agent
  request pattern, and watch the server log for
  `eviction guard: not evicting slot N (f_keep = ... < ...), routing to cold slot`.
- Benchmark: compare prompt tokens processed (prefill count) for the
  main-resume turn with the guard off vs on. The guard should drop the
  main-resume prefill from full-context to ~0.

(Build error log to be appended here when the first compile is run.)

--------------------------------------------------------------------------------

## Files touched

All changes are additive and gated; none alter default behavior.

| File | Change |
| --- | --- |
| `common/common.h` | Add two fields to `common_params` after `slot_prompt_similarity`: `bool slot_eviction_guard = false;` and `float slot_keep_thold = 0.5f;` |
| `common/arg.cpp` | Add two server CLI options after `--slot-prompt-similarity`: `--slot-eviction-guard` (no-value, sets the bool) and `--slot-keep-threshold F` (parses a float via `std::stof`). |
| `tools/server/server-context.cpp` | (1) Add matching members to `server_context` after its `slot_prompt_similarity`. (2) Propagate both from `params_base` in the init path next to `slot_prompt_similarity = params_base.slot_prompt_similarity;`. (3) In `get_available_slot()`, track a `cold_slot` in the scan loop and add the eviction guard around the `f_keep` check. |

Identifier anchors (robust against line drift):

- `common_params::slot_eviction_guard`, `common_params::slot_keep_thold`
- CLI: `--slot-eviction-guard`, `--slot-keep-threshold`
- `server_context::slot_eviction_guard`, `server_context::slot_keep_thold`
- `server_context::get_available_slot` (the guard block, keyed on `cold_slot`)

Feature commit on this branch: `2cec863ca server : add --slot-eviction-guard for automatic slot selection`.
