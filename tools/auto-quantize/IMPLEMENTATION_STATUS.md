# Implementation Status — IQ2_XXS Per-Tensor Codebook Research

**Date**: 2026-07-03  
**Repo**: `/home/user/llm/auto_research_llama/llama.cpp`  
**Branch**: `auto_research_llama` (HEAD at `0982b922b exp-001: ...`)  
**Model**: Qwen3.5-2B (BF16 GGUF at `/home/user/llm/models/Qwen3.5-2B/`)

## Overview

Implemented per-tensor K-means codebook learning for IQ2_XXS quantization, with
a companion `.iq2xxs_grids` GGUF file for storage and loading at inference time.
The full pipeline (learn → save → load → apply) works end-to-end, confirmed by
a diagnostic test where forcing all centroids to 5 increased KL from 0.993 to
11.204.

**Result**: K-means learning with centroids constrained to {1,3,5} produces
grids nearly identical to the E8 lattice grid. No meaningful quality improvement
over the baseline IQ2_XXS (KL: 0.993 both with and without learned grids).

## Files Modified

### Core Quantization (`ggml/`)
| File | Changes |
|------|---------|
| `ggml/src/ggml-quants.c` | `iq2xxs_learn_grid()` (~line 3476) — K-means codebook learner (10 iters, 16K samples, warm-start from E8 grid, centroid rounding to {1,3,5}). `iq2xxs_rebuild_map_and_neighbours()` (~line 3344) — rebuilds kmap/neighbor graph from new grid. `iq2xxs_resolve_grid()` (~line 2542) — checks per-tensor grid pointer, falls back to `iq2xxs_grid`. `ggml_iq2xxs_get_per_tensor_grid()` / `ggml_iq2xxs_set_per_tensor_grid()` — global TLS for per-tensor grid during inference. `iq2xxs_get_learned_grid()` (~line 2903) — returns learned grid for companion file writing. |
| `ggml/src/ggml-quants.h` | Declares `iq2xxs_learn_grid()`, `ggml_iq2xxs_get/set_per_tensor_grid()`, `ggml_iq2xxs_set/get_learn_codebook()`, `iq2xxs_get_learned_grid()` |
| `ggml/include/ggml.h` | Added `void * iq2xxs_grid_data` to `ggml_tensor` (replaced `char padding[8]`) |
| `ggml/src/ggml.c` | `ggml_quantize_chunk()` unchanged (still uses global `iq2_data`) |

### CPU Inference (`ggml-cpu/`)
| File | Changes |
|------|---------|
| `ggml/src/ggml-cpu/ggml-cpu.c:1179` | Before `ggml_compute_forward_mul_mat_one_chunk`: sets `ggml_iq2xxs_set_per_tensor_grid(src0->iq2xxs_grid_data)` |
| `ggml/src/ggml-cpu/ops.cpp:491` | Before `ggml_compute_forward_dup_from_q`: same grid setting |
| `ggml/src/ggml-cpu/quants.c:868` | `ggml_vec_dot_iq2_xxs_q8_K_generic`: reads `ggml_iq2xxs_get_per_tensor_grid()`, falls back to `iq2xxs_grid` |
| `ggml/src/ggml-cpu/arch/x86/quants.c:2673` | AVX2 dot product: same fallback pattern. Changed `_mm256_set_epi64x(iq2xxs_grid[...])` → `_mm256_set_epi64x(local_grid[...])` |

### CUDA Inference (`ggml-cuda/`)
| File | Changes |
|------|---------|
| `ggml/src/ggml-cuda/common.cuh` | Added `g_dev_iq2xxs_grid` device pointer + `ggml_cuda_set_iq2xxs_grid()` using `cudaMemcpyToSymbol` |
| `ggml/src/ggml-cuda/ggml-cuda.cu:947` | `ggml_backend_cuda_split_buffer_init_tensor`: uploads per-tensor grid to GPU via `cudaMemcpyAsync` if `tensor->iq2xxs_grid_data` exists. Also frees in `clear_tensor_extras`. |
| `ggml/src/ggml-cuda/mmvq.cu:1192+1242` | Sets `ggml_cuda_set_iq2xxs_grid()` before kernel launches (both `ggml_cuda_mul_mat_vec_q` and `ggml_cuda_op_mul_mat_vec_q`) |
| `ggml/src/ggml-cuda/vecdotq.cuh:994` | `vec_dot_iq2_xxs_q8_1`: `grid = g_dev_iq2xxs_grid ? g_dev_iq2xxs_grid : iq2xxs_grid` |
| `ggml/src/ggml-cuda/mmq.cuh:2775` | `load_quant_iq2_xxs_f16`: same fallback pattern |
| `ggml/src/ggml-cuda/convert.cu:306` | `dequantize_block_iq2_xxs`: same fallback pattern |
| `ggml/src/ggml-cuda/mmq.cu:85` | NOTE: mmq template instances do NOT support per-tensor grids (compiled in separate TUs). Falls back to default grid. Only mmvq.cu path supports per-tensor grids. |

### Model Load/Save (`src/`)
| File | Changes |
|------|---------|
| `src/llama-model.h:678-679` | Added `void load_iq2xxs_grids(const std::string & fname)` |
| `src/llama-model.cpp:1017` | Added `std::vector<std::vector<uint8_t>> iq2xxs_grids_storage` to impl |
| `src/llama-model.cpp:1635` | `load_iq2xxs_grids()` — opens companion GGUF file (`{fname}.iq2xxs_grids`), matches grid tensors to weight tensors by stripping `.iq2xxs_grid` suffix, stores in `tensor->iq2xxs_grid_data` |
| `src/llama.cpp:334` | Calls `model->load_iq2xxs_grids(fname)` after tensor loading |
| `src/llama-quant.cpp:716` | `llama_tensor_quantize_impl()`: calls `iq2xxs_learn_grid()` before quantization if `ggml_iq2xxs_get_learn_codebook()` is true |
| `src/llama-quant.cpp:1266` | After quantizing each IQ2_XXS tensor: captures learned grid via `iq2xxs_get_learned_grid()`, stores in `learned_grids` map |
| `src/llama-quant.cpp:1292` | After all tensors: writes learned grids to companion GGUF file as GGML_TYPE_I64 tensors named `"{weight_name}.iq2xxs_grid"` |

### CLI Flags (`tools/`)
| File | Changes |
|------|---------|
| `tools/quantize/quantize.cpp` | Added `--iq2xxs-learn` flag, `extern ggml_iq2xxs_set_learn_codebook()`, `iq2xxs_learn_codebook` variable |

### Documentation (`tools/auto-quantize/`)
| File | Changes |
|------|---------|
| `program.md` | Full rewrite: target IQ2_XXS, K-means learning protocol, backend scope, evaluation command |
| `PLAN.md` | Full rewrite: per-tensor codebook implementation plan, backend scope |
| `auto_quantize.py` | Updated for IQ2_XXS: `--type IQ2_XXS` default, imatrix auto-detection, tensor-type-file, `--pure` opt-in |

## Results

| Experiment | KL | PPL | Same top P | Size | Notes |
|-----------|-----|-----|------------|------|-------|
| Q2_K baseline | 3.241 | 334.41 | 27.07% | 620 MB | Original baseline |
| Unsloth UD-IQ2_XXS | 0.721 | 26.44 | 60.25% | 733 MB | Reference from HuggingFace |
| Our IQ2_XXS (no learn) | 0.993 | 34.75 | 55.89% | 776 MB | Same type map as Unsloth |
| K-means (3 tensors, shared grid) | 0.981 | 34.26 | 56.31% | 787 MB | Noise — one grid dominated all |
| K-means (all 160 tensors, shared grid) | 1.018 | 34.75 | 55.89% | 787 MB | Per-tensor but no companion file |
| K-means per-tensor + companion file | 0.993 | 34.76 | 55.89% | 776 MB | Full pipeline, no improvement |
| Diagnostic (all centroids = 5) | 11.204 | 988K | 1.68% | 776 MB | Proves pipeline works end-to-end |

## Why No Improvement

The E8-lattice grid (256 centroids of 8D vectors with values in {1,3,5}) is
already near-optimal for the weight distribution. K-means with centroid values
forced to {1,3,5} after each iteration cannot discover meaningfully different
grid entries. The quantization's exhaustive neighbor search across all 256
entries already picks the optimal match.

AutoQuant's -40% KL came from learning **arbitrary 4-value bf16 codebooks per
group** — fundamentally different from IQ2_XXS's restricted 8D int8 grid.

## Known Caveats

1. **Imatrix OOB**: The imatrix has `n_per_row` elements but `learn_grid` 
   indexes with `(idx*8 + k) % n_per_row`. This is correct — the imatrix is
   per-column, same for all rows.
2. **mmq.cu does not support per-tensor grids**: Only the mmvq (vec) CUDA path
   sets the per-tensor grid. The mmq path falls back to the default grid.
3. **9 tensors missing imatrix**: `token_embd.weight` and blk.24 tensors have no
   imatrix data. Tensor type file overrides them to q5_K / q4_K.
4. **OMP library**: LSP reports `omp.h` not found — this is a build system issue,
   not a code problem. The code compiles with OpenMP support.
5. **Binary size warning**: llama-quantize/perplexity are small wrappers (~16-18KB)
   that load `.so` libraries at runtime. Check `nm` on the `.so` files, not the binaries.
6. **Temporary files toast after reboot**: `/tmp/tensor_types.txt`, `/tmp/dryrun*.txt`
   etc. are lost on server restart. Must regenerate.

## Key Commands

### Regenerate tensor type file
```bash
./build/bin/llama-quantize --dry-run \
  --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf /dev/null IQ2_XXS \
  > /tmp/dryrun.txt 2>&1

python3 -c "
import re
maps = {}
with open('/tmp/dryrun.txt') as f:
    for line in f:
        m = re.search(r'\]\s+(\S+)\s+-.*\(\s*(\w+)\s*\)', line)
        if m: maps[m.group(1)] = m.group(2)
fixes = {'token_embd.weight':'q5_K'}
for n in ['attn_k','attn_output','attn_q','attn_v','ffn_down','ffn_gate','ffn_up','nextn.eh_proj']:
    fixes[f'blk.24.{n}.weight'] = 'q4_K'
for name, qt in fixes.items():
    if name in maps: maps[name] = qt
with open('/tmp/tensor_types.txt', 'w') as f:
    for name in sorted(maps):
        f.write(f'{name}={maps[name]}\n')
print(f'{len(maps)} mappings')
"
```

### Quantize with K-means learning
```bash
rm -f /tmp/test-learn.gguf /tmp/test-learn.gguf.iq2xxs_grids
./build/bin/llama-quantize \
  --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
  --tensor-type-file /tmp/tensor_types.txt \
  --iq2xxs-learn \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
  /tmp/test-learn.gguf IQ2_XXS
```

### Evaluate (USE RTX 3080, device 1)
```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-perplexity \
  -m /tmp/test-learn.gguf \
  -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
  -t 8 -c 512 --chunks 200 \
  -fa on --cache-type-k bf16 --cache-type-v bf16 \
  --no-mmap -ngl 999 -np 1 \
  --kl-divergence \
  --kl-divergence-base /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits
```

### Verify grids are loaded (verbose)
```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-perplexity ... -lv 5 2>&1 | grep "load_iq2xxs_grids"
```

### Verify symbol exists
```bash
nm build/bin/libllama.so | grep load_iq2xxs
```

## Model Paths

| Resource | Path |
|---|---|
| BF16 model | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf` |
| BF16 reference logits | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits` |
| Imatrix (Unsloth) | `/home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf` |
| Eval data | `/home/user/llm/wikitext-2-raw/wiki.test.raw` |
| Calibration data | `/tmp/calibration_short.txt` |
| Tensor type file | `/tmp/tensor_types.txt` |
| Unsloth IQ2_XXS | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-UD-IQ2_XXS.gguf` (733MB) |
| Our IQ2_XXS | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-OUR-IQ2_XXS.gguf` (776MB) |

## GPU Info

| Device | Model | VRAM | CC |
|--------|-------|------|-----|
| 0 | RTX 5090 | 32110 MB | 12.0 |
| 1 | RTX 3080 | 20054 MB | 8.6 |
| 2 | RTX 3080 | 20054 MB | 8.6 |
| 3 | RTX 3050 | 5806 MB | 8.6 |

**Use device 1 (RTX 3080)** for eval: `CUDA_VISIBLE_DEVICES=1`

Device 0 (RTX 5090, CC 12.0) lacks kernel images in the CUDA build and
produces "no kernel image is available for execution on the device" errors.
