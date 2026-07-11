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

- **QK_K must not change from 256.** Changing QK_K_CLONE to 512 (exp-008) breaks
  tensor size calculations across the entire ggml stack. The doubled block size
  halves the number of blocks per row, which ripples through ggml_row_size,
  ggml_nbytes, CUDA tensor allocation, and internal dispatch assumptions.
  The model quantizes correctly but dequantizes into garbage on GPU and
  segfaults on CPU. All future experiments MUST keep QK_K=256.

- **vec_dot = NULL can cause CPU segfaults.** Setting vec_dot to NULL with
  vec_dot_type = GGML_TYPE_COUNT causes out-of-bounds access to type_traits_cpu
  in the llamafile gemm fallback path at ggml-cpu.c:1278. Use a dummy vec_dot
  or ensure the CPU matmul never reaches that code path (e.g., by making
  llamafile handle the clone type).

- **Output/token_embd tensor IS highly quality-sensitive.** Reducing it from
  Q6_K to Q4_K_M_CLONE (exp-009) saved 62.5 MB but increased KLD by 0.010949
  (17.4% above threshold). The degradation is comparable to removing all
  attn_v/ffn_down Q6_K boosts (exp-002: +0.010489). Despite the output layer
  being "last mile" to token prediction, its per-MB sensitivity is actually
  lower than attention layers. Q5_K (saving ~32 MB) might be tolerable.

- **Q5_K for output/token_embd is borderline** (exp-010): KLD 0.0650 (+3.2% above threshold, 0.002 KLD). Same top p 85.94% (0.45pp below). This saves 33.8 MB (6.4% of model). Combining Q5_K with keeping Q6_K for 2-3 extra attention QKV layers (~4 MB) could push it across the threshold to success.

- **Changing the block_q4_K_M_CLONE struct size breaks ALL CUDA-accelerated paths** (exp-012): Reducing the block from 144 to 136 bytes caused cublas init to crash during GPU eval warmup. The MMQ/MMVQ kernels, repack functions, and vec_dot all assume the block has the same layout as block_q4_K. Even with new CUDA dequant kernels and MMQ/MMVQ disabled, the fallback to dequant+cublasSgemm path triggers cublasCreate failures during graph computation. The root cause is that when the CUDA backend's supports_op returns true for MUL_MAT with clone type, the dispatch tries the MMQ path first (disabled), then MMVQ (disabled), then falls to ggml_cuda_op_mul_mat_cublas which calls cublas_handle before the CUDA context is fully initialized. **For block compression to work, either keep the struct size identical (find encoding compression within the same byte budget) OR rewrite ALL CUDA dispatch paths (MMQ, MMVQ, repack, vec_dot, convert, type traits) simultaneously.**

- **Alternating optimization degrades quality** (exp-013): Iterative refinement (fix nibbles → re-optimize grid → re-quantize nibbles) converges to WORSE local minima than the greedy one-shot approach in Q4_K. The secondary quantization of grid parameters (fp16 d/dmin, 6-bit sc/m) creates non-smooth distortions between iterations, breaking convergence. The original one-shot algorithm succeeds because it derives grid parameters directly from raw weight statistics, not from an intermediate quantized approximation. Post-hoc refinement of already-quantized blocks is counterproductive.

- **zlib level 1 achieves almost zero compression on 4-bit data** (exp-014): The 128-byte qs[] field (packed 4-bit nibbles) is near maximum entropy, and zlib's deflate algorithm at Z_BEST_SPEED saves only ~1.2% on the tensor data. zstd at default level achieves ~2.65% on the ENTIRE GGUF file (including compressible metadata). For post-quantization compression, zstd is significantly better than zlib for this use case.

- **GGUF metadata is loaded separately from tensor data** (exp-014): `gguf_init_from_file` opens the file independently (not through `llama_file`), so adding decompression to `llama_file` alone doesn't work. Decompression must happen BEFORE `gguf_init_from_file` reads the file header. The correct approach is to decompress the entire file to a temp file and redirect the model loader to the temp file.

- **Post-quantization GGUF compression plateaus at ~3.06%** (exp-015): After exhaustively testing zstd (levels 3-22), xz/lzma, bzip2, delta encoding, split metadata/tensor compression, and dictionary training — the hard ceiling is ~3.06% (zstd -19). The 4-bit/6-bit quantized tensor data (97.93% of the file) is near-entropy and fundamentally resists generic compression. Metadata (11 MB of tokenizer strings) compresses ~78% regardless of compressor. Byte-level delta encoding (XOR) makes compression WORSE. Split metadata/tensor compression saves only ~96 KB more than whole-file zstd -19. Going from zstd -19 to --ultra -22 saves only 15 KB more but takes 5x longer. Any further size reduction must come from changes to the quantization itself (at quality cost) or GGUF format changes (redundant metadata removal, variable-length encoding).

- **CUDA shared-memory FWHT kernel with 32 threads** (exp-016): The CUDA dequant kernel for clone type must fit within 32-thread block size (matching original Q4_K dequant launch params). Each thread processes 8 elements, stores to __shared__ float[256] (1KB), then participates in 8-stage FWHT via butterfly operations. The FWHT loop uses `for (int j = tid; j < 128; j += 32)` to distribute 128 pair operations across 32 threads per stage. Each stage requires __syncthreads() barrier. The inverse scaling (1/256) is applied during final write-back to global memory.

- **Scale encoding changes break CUDA dequant even with same struct size** (exp-017): Simply changing the scale/min encoding from 6+6 bits to 6+2 bits (keeping 144-byte struct, using 8 active + 4 padding bytes) breaks GPU dequant catastrophically (PPL ~465 vs expected ~22). The CUDA kernel must use the EXACT same scale unpacking as the CPU dequant, and any discrepancy between the two produces silent corruption. The CUDA kernel's `x[i].scales[is+N]` access pattern must match the CPU's byte-level scale read order. Even with correct-looking kernel code, subtle mismatches in byte ordering or min expansion formulas are essentially undebuggable at the 250-chunk eval level. Future experiments must avoid ANY change to the scale encoding in the dequant path — focus ONLY on the quantize side (preprocessing like FWHT) or accept that scale format changes require complete CUDA kernel verification at the single-block level before full eval.

- **GGUF byte reordering breaks across multiple load paths** (exp-018): Permuting quantized block data within the GGUF file (e.g., row-major to column-major) and adding an inverse permutation during loading fails because `load_all_data` has multiple complex code paths (host buffer, GPU async upload w/pinned memory, simple read_buf) and the retry mechanism (`common_fit_params`) causes a second load with different parameters. Any data layout change requires the inverse to be implemented in ALL paths, which is fragile. Additionally, `ggml_type_size()` varies per type (144, 176, 210 for Q4_K, Q5_K, Q6_K), requiring size-aware permutation across mixed-type files. GGUF format-level modifications are fundamentally harder than quantize-side-only changes.

- **zstd fails on /tmp with "Disk quota exceeded"** (exp-029/030/031): The zstd CLI tool's temp file handling fails on tmpfs /tmp even with 12G free space. zstd compression via `system("zstd -19 ...")` writes a `.zst` temp in the same directory as the output. for quantize, use an output path NOT on /tmp (e.g., workspace dir). for eval of zstd-compressed GGUF files, the model loader's transparent decompression at line 296 creates a temp file hardcoded to `/tmp/llama-decomp-XXXXXX` via `mkstemp`. This hardcoded /tmp path causes decompression to fail. **Workaround: manually decompress with `zstd -d` to a non-/tmp path and eval from there.** The quality is identical (zstd is lossless).

- **Formula change (m=sc coupling) is catastrophic** (exp-029): Forcing per-sub-block m to equal sc (formula x=d*sc*q-dmin*sc) eliminates the independent m degree of freedom but catastrophically restricts the effective min range. Sub-block mins need independent range from scales — mins relate to offset while scales relate to width. In FWHT space, the correlation is not strong enough for forced proportionality. PPL went to 150K (vs 22). Do NOT attempt formula simplifications that eliminate degrees of freedom without a recovery mechanism.

- **Coarsening as regularization eventually saturates** (exp-030→031): m 6→5 bits IMPROVED quality (KLD -2.9%) via regularization, but m 5→4 bits DEGRADED quality (KLD +3.9%, PPL +0.75). The regularization benefit only appears at the first coarsening step; subsequent steps hit true precision loss. Per-bit KLD cost: m 6→5 was -0.00163 (improvement), m 5→4 was +0.00211 (degradation). This pattern likely repeats for other parameters — use the first coarsening step to capture the regularization free lunch, then stop.

- **Local byte grouping makes zstd compression WORSE** (exp-026): Grouping correlated values together in consecutive bytes (e.g., all 8 sc nibbles together, all 8 m bytes together) seems intuitive for zstd gradient detection, but actually INCREASES file size. The stock interleaved packing scatters bits across bytes in a way that creates narrow-value-range bytes (e.g., stock bytes 0-3 hold only values 0-15) which zstd matches ACROSS blocks at offset 144-byte stride. Grouped layouts create per-block-unique byte sequences (each block's sc gradient is a slightly different 4-byte pattern) that don't repeat across blocks. zstd's LZ77 engine benefits more from CROSS-BLOCK byte-level repetition (same byte value at a known offset stride) than from within-block adjacency correlation. For future byte-layout experiments: prefer layouts that produce the same byte values at fixed offsets across many blocks, not layouts that produce locally-correlated but globally-unique byte sequences.

- **Min precision below 4 bits is catastrophic even at 140-byte block** (exp-033): Reducing per-sub-block min (m) from 4 to 3 bits within CAQ's 140-byte block degraded KLD by +6.4% (-0.004) despite simultaneously improving scale precision from 4 to 5 bits. Scale and min have asymmetric quality sensitivity: below 4 bits, coarse min centering creates systematic per-sub-block bias that optimal scale cannot recover. Min is NOT a free parameter to trade against scale — the 4-bit floor for mins is real, even with FWHT preprocessing.

- **Snapping refined d/dmin back to fp16 grid catastrophically degrades quality** (exp-034): Even though d/dmin refinement only adjusts by ±1-2%, the resulting off-grid values are essential. Snapping them back to the 4-bit mantissa (0xFFC0) grid causes 4.4x KLD increase. The refinement escape from the coarse grid is a feature, not a bug. Do NOT re-round at write time.

- **Struct size change + FWHT + 4+4 bit scales produces catastrophic output** (exp-036): Reducing the struct from 144 to 140 bytes (scales 12→8, 4+4 bit sc/m) with FWHT preprocessing produced PPL 6.9M (300,000x baseline), KLD 13.04, same_top_p 0.028%. Even with clean self-contained quant/dequant functions and updated CUDA kernel, the combined effect of struct size reduction plus 4+4 bit scales (sharpest reduction in sc/m precision yet) is a complete breakdown. The exact root cause (wrong qs offset access, scale encoding bug, or FWHT interaction) is undetermined due to the scale of the failure. When struct size changes, test with a known-good quant algorithm first (e.g., standard Q4_K quant on the new struct) before introducing algorithm-level changes like CSE. Do not change struct size AND scale encoding AND precision simultaneously — unbundle into separate experiments.

- **Block struct size change shifts field offsets for ALL dispatch paths** (exp-035): Changing scales[] from 12 to 2 bytes moved qs[] from offset 16 to offset 6. This broke not just the primary dequant kernel but also all secondary CUDA dispatch paths that assume block_q4_K layout (MMVQ's `get_mmvq_mmid_max_batch`, `should_use_mmvq`, the MUL_MAT_ID dispatch chain that calls MMVQ directly without going through `should_use_mmvq`). The CPU path also broke because `vec_dot_type = GGML_TYPE_COUNT` was an out-of-bounds array index. Full block compression requires systematically disabling ALL accelerated paths that read the struct: remove clone from mmvq.cu fallthrough cases, from get_mmvq_mmid_max_batch, from mmq.cuh dispatch, from repack, AND fix vec_dot_type to a valid type.

- **Sc coarsening also gives a "first step regularization free lunch"** (exp-037): exp-030 showed m 6→5 improves KLD via regularization. exp-037 replicated this for sc 6→5 (-7.3% KLD, +0.82pp STP from clean FWHT base). The pattern is general: sc and m at 6-bit precision are over-parameterized for FWHT-preprocessed data, and the greedy heuristic overfits. The first coarsening step (6→5) on each parameter is consistently beneficial. Second steps (5→4) cost quality (exp-031 for m, not yet tested for sc). When designing experiments, try 6→5 before trying 5→4 on any parameter.

- **sc + m coarsening benefits DO NOT compound** (exp-038): Combining 5-bit sc + 5-bit m produced KLD 0.060695 (-3.6% vs baseline) which is still better than baseline but significantly WORSE than sc-only at 5-bit (exp-037: KLD 0.058347, -7.3%). The regularization from both parameters overlaps substantially — they both prevent the same greedy heuristic overfitting. Coarsening the FIRST parameter captures most of the benefit; coarsening the second introduces genuine precision loss with almost no additional regularization. The optimal single-variable configuration is exp-037 (sc=5, m=6). Do not pursue combined coarsening of both sc and m.
