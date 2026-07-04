# autoresearch — llama.cpp IQ2_XXS quantization

Research driven by an autonomous LLM agent, following the pattern of
karpathy/autoresearch. The agent runs inside opencode and follows this
protocol to explore, implement, and evaluate quantization improvements.

## Objective

Minimize **KL divergence** of IQ2_XXS-quantized models against the BF16 reference,
by improving the quantization algorithm in `ggml/src/ggml-quants.c`.

The IQ2_XXS type uses a 256-entry 8D codebook grid (stored as `uint64_t[256]`).
The primary research direction is **per-tensor learned codebooks** — replacing
the global E8-lattice grid with K-means-optimized grids per tensor, with
companion `.iq2xxs_grids` GGUF file for storage and loading at inference time.

## Baselines

| Experiment | KL | PPL | Same top P | Size | Notes |
|-----------|-----|-----|------------|------|-------|
| Unsloth UD-IQ2_XXS | **0.721** | 26.44 | 60.19% | 733 MB | Reference target — uses Unsloth's quantizer |
| Our IQ2_XXS (Unsloth types) | **0.748** | 27.73 | 58.73% | 755 MB | Same type map as Unsloth, llama.cpp quantizer |
| exp-002 best (error-aware + data-driven init) | 0.971 | 33.95 | 56.50% | 786 MB | Best per-tensor codebook learning result (OLD baseline types) |
| Our IQ2_XXS (auto types, no learn) | 0.993 | 34.75 | 55.89% | 776 MB | Old baseline with auto-assigned types |

**Current quantizer gap: KL Δ = 0.027 (3.8%)** — this is the target for codebook learning improvements.

## Models and Data

| Resource | Path |
|---|---|
| BF16 model | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf` |
| BF16 reference logits (LOCKED) | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits` |
| Imatrix (Unsloth) | `/home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf` |
| Unsloth IQ2_XXS model (reference) | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-UD-IQ2_XXS.gguf` |
| Eval data (LOCKED) | `/home/user/llm/wikitext-2-raw/wiki.test.raw` |
| Calibration data | `/tmp/calibration_short.txt` |
| Tensor type file (Unsloth-matched) | `tools/auto-quantize/tensor_types_unsloth_match.txt` |

## Critical: Exact Type Match from Unsloth Model

**Do NOT use auto-assigned types.** Extract the exact tensor→type mapping from the
Unsloth reference model to get apples-to-apples comparisons. The auto-assigner
produces inferior type choices (different type-to-importance thresholds).

The quantized Unsloth GGUF uses the SAME `GGML_TYPE` enumeration as llama.cpp
(types 16=iq2_xxs, 18=iq3_xxs, 21=iq3_s, 22=iq2_s, 8=q8_0, 10=q2_K, 12=q4_K, 13=q5_K).

### Generate matching tensor type file:

```bash
python3 -c "
import struct
tname = {0:'f32',1:'f16',2:'q4_0',3:'q4_1',6:'q5_0',7:'q5_1',8:'q8_0',9:'q8_1',
         10:'q2_K',11:'q3_K',12:'q4_K',13:'q5_K',14:'q6_K',15:'q8_K',
         16:'iq2_xxs',17:'iq2_xs',18:'iq3_xxs',19:'iq1_s',20:'iq4_nl',
         21:'iq3_s',22:'iq2_s',23:'iq4_xs',29:'iq1_m'}
with open('/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-UD-IQ2_XXS.gguf','rb') as f:
    d = f.read()
pos = d.index(b'GGUF')
ntensors = struct.unpack('<Q', d[pos+8:pos+16])[0]
nmeta = struct.unpack('<Q', d[pos+16:pos+24])[0]
pos += 24
for _ in range(nmeta):
    kl = struct.unpack('<Q', d[pos:pos+8])[0]; pos += 8
    pos += kl
    vt = struct.unpack('<I', d[pos:pos+4])[0]; pos += 4
    if vt in (0,1,7): pos+=1
    elif vt in (2,3): pos+=2
    elif vt in (4,5,6): pos+=4
    elif vt in (10,11,12): pos+=8
    elif vt==8: sl = struct.unpack('<Q', d[pos:pos+8])[0]; pos += 8+sl
    elif vt==9:
        at = struct.unpack('<I', d[pos:pos+4])[0]; al = struct.unpack('<Q', d[pos+4:pos+12])[0]; pos += 12
        for _ in range(al):
            if at==8: sl=struct.unpack('<Q', d[pos:pos+8])[0]; pos+=8+sl
            else: pos += {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(at,4)
mapping = {}
for _ in range(ntensors):
    nl = struct.unpack('<Q', d[pos:pos+8])[0]; name = d[pos+8:pos+8+nl].decode(); pos += 8+nl
    nd = struct.unpack('<I', d[pos:pos+4])[0]; dims = struct.unpack(f'<{nd}Q', d[pos+4:pos+4+nd*8]); pos += 4+nd*8
    tt = struct.unpack('<I', d[pos:pos+4])[0]; pos += 4+8
    mapping[name] = tname.get(tt, f'type_{tt}')

skip_bases = ['attn_norm', 'post_attention_norm', 'attn_k_norm', 'attn_q_norm',
              'nextn.hnorm', 'nextn.enorm', 'nextn.shared_head_norm',
              'output_norm', 'ssm_norm', 'ssm_conv1d', 'ssm_dt']

for name in sorted(mapping):
    utype = mapping[name]
    if not name.endswith('.weight'):
        continue
    parts = name.split('.')
    base = parts[-2] if len(parts) >= 2 else ''
    if base in skip_bases:
        continue
    print(f'{name}={utype}')

# blk.24 fallback (not in Unsloth model, no imatrix data)
for t in ['attn_k','attn_output','attn_q','attn_v','ffn_down','ffn_gate','ffn_up','nextn.eh_proj']:
    print(f'blk.24.{t}.weight=q4_K')
" > tools/auto-quantize/tensor_types_unsloth_match.txt
```

### Resulting type distribution (195 tensors):

| Type | Count | BPW | Notes |
|------|-------|-----|-------|
| iq2_xxs | 95 | 2.06 | Main quantization type |
| iq3_xxs | 24 | 3.06 | `attn_gate.weight` — attention gating (sensitivity-critical) |
| iq3_s | 5 | 3.44 | Some `ffn_down.weight` in early gated-attention layers |
| iq2_s | 5 | 2.06 | Selected `attn_output`/`ffn` tensors |
| q8_0 | 36 | 8.0 | `ssm_alpha.weight` + `ssm_beta.weight` — SSM temporal params (sensitivity-critical) |
| q4_K | 26 | 4.50 | `ssm_out.weight`, `attn_v.weight` (gated layers), blk.24 fallback |
| q2_K | 3 | 2.56 | `ffn_down.weight` in SSM layers 0,1,4,5 |
| q5_K | 1 | 5.50 | `token_embd.weight` |

### CAVEAT: Why SSM params must be q8_0 (not q8_1)

`ssm_alpha/beta` tensors have shape [2048, 16] — only 16 columns per row.
**q8_1 requires ≥32 columns** (QK8_1=32 block size) and WILL CRASH with:
`GGML_ASSERT(result == nrows * row_size) failed`
q8_0 (QK8_0=32) handles skinny matrices correctly — pads internally.
**Never replace q8_0 with q8_1 for ssm_alpha/beta.**

## IQ2_XXS Codebook Learning Protocol

### Step 1: Quantize with K-means learning

**Quantization may take up to 30 minutes with learning enabled.** Always pass
`--timeout`/`-t 120m` when running via sub-agent or automation, and do not
kill the process unless it exceeds 120 minutes.

```bash
rm -f /tmp/learned.gguf /tmp/learned.gguf.iq2xxs_grids
./build/bin/llama-quantize \
  --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
  --tensor-type-file tools/auto-quantize/tensor_types_unsloth_match.txt \
  --iq2xxs-learn \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
  /tmp/learned.gguf IQ2_XXS
```

This calls `iq2xxs_learn_grid()` in `ggml/src/ggml-quants.c` for each IQ2_XXS
tensor BEFORE quantization. The learned grid replaces the global E8 lattice grid
for that tensor. Grids are saved to `{output}.iq2xxs_grids` companion file.

### Step 2 (baseline, no learning):

```bash
rm -f /tmp/quantized-model.gguf
./build/bin/llama-quantize \
  --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
  --tensor-type-file tools/auto-quantize/tensor_types_unsloth_match.txt \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
  /tmp/quantized-model.gguf IQ2_XXS
```

## Evaluation (LOCKED — never change these flags)

```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-perplexity \
  -m MODEL.gguf \
  -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
  -t 8 -c 512 --chunks 200 \
  -fa on --cache-type-k bf16 --cache-type-v bf16 \
  --no-mmap -ngl 999 -np 1 \
  --kl-divergence \
  --kl-divergence-base /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits
```

**Use `CUDA_VISIBLE_DEVICES=1`** (RTX 3080, CC 8.6). Device 0 (RTX 5090, CC 12.0)
lacks kernel images and produces "no kernel image available" errors.

## Infrastructure: Per-Tensor Codebook Pipeline

The full pipeline (learn → save → load → apply) is implemented and verified:

### Files with per-tensor grid plumbing:

**Core learning:**
- `ggml/src/ggml-quants.c:3482` — `iq2xxs_learn_grid(x, weights, nrows, n_per_row, tensor_name)` (5 params!)
- `ggml/src/ggml-quants.c:2895` — `ggml_iq2xxs_set_per_tensor_grid()`
- `ggml/src/ggml-quants.c:2899` — `ggml_iq2xxs_get_per_tensor_grid()`
- `ggml/src/ggml-quants.c:2903` — `iq2xxs_get_learned_grid()`
- `ggml/src/ggml-quants.c:2531` — `iq2xxs_resolve_grid()` — per-tensor or fallback

**Storage:**
- `ggml/include/ggml.h:698` — `void * iq2xxs_grid_data` field on `ggml_tensor`
- `src/llama-quant.cpp:1292` — writes companion `.iq2xxs_grids` GGUF file
- `src/llama-model.cpp:1635` — `load_iq2xxs_grids()` — loads and attaches to tensors

**CPU inference (per-tensor grid: YES):**
- `ggml/src/ggml-cpu/ggml-cpu.c:1180` — sets per-tensor grid before mul_mat
- `ggml/src/ggml-cpu/ops.cpp:492` — sets per-tensor grid before dequant
- `ggml/src/ggml-cpu/quants.c:868` — reads per-tensor grid (scalar path)
- `ggml/src/ggml-cpu/arch/x86/quants.c:2673` — reads per-tensor grid (AVX2 path)

**CUDA inference (per-tensor grid: mmvq/vecdotq/convert only):**
- `ggml/src/ggml-cuda/ggml-cuda.cu:948` — uploads per-tensor grid to GPU
- `ggml/src/ggml-cuda/mmvq.cu:1195+1244` — sets per-tensor grid before kernel
- `ggml/src/ggml-cuda/vecdotq.cuh:994` — grid fallback: `g_dev_iq2xxs_grid ?: iq2xxs_grid`
- `ggml/src/ggml-cuda/mmq.cuh:2777` — grid fallback
- `ggml/src/ggml-cuda/convert.cu:306` — grid fallback
- `ggml/src/ggml-cuda/mmq.cu:87` — **mmq path does NOT support per-tensor grids** (separate TU)

**Plumbing:**
- `src/llama.cpp:334` — calls `load_iq2xxs_grids(fname)` during model load
- `src/llama-quant.cpp:720` — calls `iq2xxs_learn_grid()` before quantization
- `tools/quantize/quantize.cpp:541` — `--iq2xxs-learn` CLI flag

### Diagnostic test (proving pipeline works end-to-end):

Forcing all centroids to value 5 (by modifying the grid write path) produces
KL=11.204 vs baseline KL=0.993 — confirms grids are loaded and applied correctly
at inference.

## Current `iq2xxs_learn_grid()` Implementation

The function at `ggml/src/ggml-quants.c:3482` (exp-005 state, commit `2de18b5fa`):

```c
void iq2xxs_learn_grid(const float * x, const float * weights,
        int64_t nrows, int64_t n_per_row, const char * tensor_name);
```

Parameters:
- `tensor_name` — used to classify tensors into ATTN/MLP/OTHER categories for per-category codebooks
- 3 shared grids maintained across all tensors of the same type (refined cumulatively)
- `weights` — imatrix importance values (per-column, activates `exp-017` activation-weighted updates)

Algorithm:
1. 5 trials — trial 0 = per-category warm-start, trials 1-4 = data-driven random init
2. 40 iterations of float-space K-means (no int8 rounding during training)
3. L1 distance metric for assignment
4. Error-aware int8 snap: floor vs ceil evaluated against assigned samples' weighted error
5. 3 rounds of multi-round refinement: re-assign + ±1 gradient descent per centroid dim
6. Centroid value range: [0, 127] (allow-zero)
7. Activation-weighted centroid updates (`sum(w * sample) / sum(w)`)

### Previous Results (with old auto-assigned tensor types):

| Experiment | KL | Technique |
|-----------|-----|-----------|
| exp-baseline | 0.993 | No learning (E8 lattice grid) |
| exp-001 | 0.993 | Float-space K-means (no difference) |
| exp-002 | **0.971** | Error-aware snap + data-driven init + allow-zero + 40 iters/5 trials |
| exp-003 | 0.971 | L1 distance metric |
| exp-004 | 0.971 | Multi-round refinement (no additional gain) |
| exp-005 | 0.971 | Per-tensor-type codebooks (ATTN/MLP/OTHER categories) |

## Files That Exist

### Editable (modify for experiments):
- `ggml/src/ggml-quants.c` — `iq2xxs_learn_grid()`, quantize kernels, codebook init
- `ggml/src/ggml-quants.h` — declarations
- `ggml/include/ggml.h` — type enums, tensor struct
- `ggml/src/ggml-cpu/quants.c` — CPU scalar dot
- `ggml/src/ggml-cpu/arch/x86/quants.c` — CPU AVX2 dot
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU dispatch
- `ggml/src/ggml-cpu/ops.cpp` — CPU dequant dispatch
- `ggml/src/ggml-cuda/*.cu,*.cuh` — CUDA kernels
- `src/llama-quant.cpp` — quantization orchestrator
- `src/llama-model.cpp` — grid loading
- `tools/quantize/quantize.cpp` — CLI flags
- `tools/auto-quantize/auto_quantize.py` — experiment orchestrator

### Locked (never touch):
- `tools/perplexity/**` — evaluation source and binary
- `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits` — reference logits
- `/home/user/llm/wikitext-2-raw/wiki.test.raw` — evaluation data

### Documentation (read, update freely):
- `tools/auto-quantize/program.md` — this file
- `tools/auto-quantize/PLAN.md` — implementation plan
- `tools/auto-quantize/synthesis.md` — per-session synthesis
- `tools/auto-quantize/idea_ledger.md` — hypothesis log
- `tools/auto-quantize/results.tsv` — experiment results (append only)
- `tools/auto-quantize/IMPLEMENTATION_STATUS.md` — infrastructure docs (may be stale)

## GPU Info

| Device | Model | VRAM | CC | Use |
|--------|-------|------|----|-----|
| 0 | RTX 5090 | 32110 MB | 12.0 | **BROKEN** — "no kernel image available" |
| 1 | RTX 3080 | 20054 MB | 8.6 | USE FOR EVAL |
| 2 | RTX 3080 | 20054 MB | 8.6 | Available |
| 3 | RTX 3050 | 5806 MB | 8.6 | Available |

## Experiment Loop

```
1. READ STATE: results.tsv, idea_ledger.md, synthesis.md, current code

2. PROPOSE HYPOTHESIS: log to idea_ledger.md

3. EDIT CODE: modify ggml-quants.c and related files

4. COMMIT: git commit -m "exp-NNN: description"

5. BUILD: cmake --build build/ -j

6. QUANTIZE (baseline):
   rm -f /tmp/quantized-model.gguf
   ./build/bin/llama-quantize \
     --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
     --tensor-type-file tools/auto-quantize/tensor_types_unsloth_match.txt \
     /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
     /tmp/quantized-model.gguf IQ2_XXS

   Or with codebook learning:
   ./build/bin/llama-quantize \
     --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
     --tensor-type-file tools/auto-quantize/tensor_types_unsloth_match.txt \
     --iq2xxs-learn \
     /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
     /tmp/learned.gguf IQ2_XXS

7. EVALUATE (always device 1):
   CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-perplexity \
     -m /tmp/quantized-model.gguf \
     -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
     -t 8 -c 512 --chunks 200 -fa on \
     --cache-type-k bf16 --cache-type-v bf16 \
     --no-mmap -ngl 999 -np 1 \
     --kl-divergence \
     --kl-divergence-base /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits

8. RECORD: append to results.tsv

9. EVALUATE OUTCOME:
   - KL improved → keep code
   - KL regressed → git revert CODE commit (keep results)
   - Build/quantize fails → fix, retry max 3 attempts

10. SYNTHESIS (every 5 experiments): update synthesis.md

11. REPEAT
```

## Integrity Rules

- NEVER modify `tools/auto-quantize/auto_quantize.py`
- NEVER modify `tools/perplexity/`
- NEVER regenerate the BF16 reference logits
- NEVER modify the Wikitext-2 test data
- NEVER change evaluation parameters (context length, chunks, cache types)
- ALWAYS record every experiment (even failures) in results.tsv
- NEVER change past rows in results.tsv
- NEVER use `git commit --amend` or rewrite history
- If experiment regresses KL → revert the code, keep the results
- If experiment improves KL → keep the code
