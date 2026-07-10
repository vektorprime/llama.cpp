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
