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

## Experiment Design Traps

- **Bit removal without recovery is not valid.** "Make X smaller" always fails.
  Proposal must be "make X smaller BY encoding it as Y."
- **Per-tensor mixing changes are off-limits.** Tweaking which tensors get Q6_K
  vs Q4_K is not the goal. Focus on the block struct and quant/dequant functions.
- **Scale-min correlation within a superblock is weak.** Adjacent sub-block
  scales are not strongly correlated, so DPCM/predictive coding of scales
  accumulates drift that dominates the error budget.
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
