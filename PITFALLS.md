# PITFALLS — Avoid These When Experimenting on Q4_K_M_CLONE

Populated by sub-agents at the end of each experiment. Read before starting.

## Code Traps

- **`ATTENTION_QKV` handler in `llama-quant.cpp` is dead code.** QKV tensors are
  caught by `category_is_attn_v()` (line ~544) before reaching the QKV handler.
  Modifying the QKV handler has zero effect. Modify the WV block instead.
- **When changing `block_q4_K_M_CLONE` struct size**, update ALL of:
  `static_assert` in `ggml-common.h`, `type_size` in `ggml.c`, `type_size` in
  `ggml-cpu/ggml-cpu.c`, `sizeof` references in `ggml-cuda/common.cuh`, and
  the `qs[]`/`scales[]` buffer sizes in `ggml-quants.c`. Missing any causes
  silent misquantization or segfaults.

- **Changing block_q4_K_M_CLONE struct size breaks ALL thin-wrappers.** The
  clone's quantize/dequantize functions cast to `block_q4_K *` which expects
  12-byte K_SCALE_SIZE. If scales[] is resized (e.g., to 8 bytes), the cast
  pointer has wrong layout and writing 12 bytes to an 8-byte array corrupts
  qs[]. All quant/dequant functions must be self-contained for the new layout.
- **Struct padding from ggml_half2 union enforces 4-byte alignment.** The
  union (4 bytes, align 4) forces trailing padding. Sizes that are NOT
  multiples of 4 waste bytes: 10-byte scales → 142→144, 9-byte→141→144.
  Only 8, 12, 16-byte scales yield actual savings. Plan your compression
   around these alignment constraints.
- **Per-weight re-optimization cannot fix systematic scale errors.** When a
  sub-block's (sc, m) pair is approximated (e.g., via VQ codebook), all 32
  weights in the sub-block share the same wrong grid (center + scale). Re-tuning
  individual 4-bit nibbles within that grid cannot correct the grid position
  itself. Scale compression needs ≤1-2 bits of error, or a grid-level recovery
  method (e.g., re-optimizing d or dmin to partially absorb sub-block bias).
- **Block compression only affects non-Q6_K tensors.** ~40% of model size comes
  from Q6_K tensors (token_embd, QKV, ffn_down in some layers) which don't use
  the clone block. The effective size reduction from block compression is much
  smaller than theoretical: a 2.78% per-block saving yields ~1.15% overall.
