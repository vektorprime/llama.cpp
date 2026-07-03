# autoresearch — llama.cpp IQ2_XXS quantization

Research driven by an autonomous LLM agent, following the pattern of
karpathy/autoresearch. The agent runs inside opencode and follows this
protocol to explore, implement, and evaluate quantization improvements.

## Objective

Minimize **KL divergence** of IQ2_XXS-quantized models against the BF16 reference,
by improving the quantization algorithm in `ggml/src/ggml-quants.c`.

The IQ2_XXS type uses a learned codebook (256-entry grid of 8D centroids)
stored as a global lookup table. The table is hardcoded at compile time.
The primary research direction is **per-layer learned codebooks** (Path A
from PLAN.md) — replacing the single hardcoded grid with K-means-optimized
grids per layer, keeping the same block structure and bpw.

Secondary metrics to track (optimize KL, don't let these regress too much):
- **Perplexity (PPL)** — overall language modeling quality
- **Same top P** — % of positions where argmax matches the reference
- **Model size** (MB) — must stay within limits
- **Tokens/sec** — inference speed must stay reasonable

## Baselines

| Experiment | KL | PPL | Same top P | Size | Tok/s | Notes |
|-----------|-----|-----|------------|------|-------|-------|
| Q2_K (exp-001) | 3.241 | 334.41 | 27.07% | 620 MB | — | Pure Q2_K baseline |
| Unsloth UD-IQ2_XXS | 0.721 | 26.44 | 60.25% | 733 MB | — | Mixed types: iq2_xxs+q4_K+q2_K+q5_K |
| Our IQ2_XXS | 0.993 | 34.75 | 55.89% | 776 MB | — | Same type map but layer 24 → q4_K (imatrix gap) |

## Models and Data

| Resource | Path |
|---|---|
| BF16 model | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf` |
| BF16 reference logits | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits` |
| Calibration data | `/tmp/calibration_short.txt` |
| Imatrix (Unsloth) | `/home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf` |
| Eval data | `/home/user/llm/wikitext-2-raw/wiki.test.raw` |
| Tensor type file | `/tmp/tensor_types.txt` |
| Quantized output | `/tmp/quantized-model.gguf` |

## IQ2_XXS Quantization Workflow

### Step 1: Generate or obtain importance matrix

The Unsloth imatrix is pre-downloaded but has limited coverage (186/195 tensors).
Our calibration data is at `/tmp/calibration_short.txt` (200 lines, English + code).

```
./build/bin/llama-imatrix \
  -m /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
  -f /tmp/calibration_short.txt \
  -o /home/user/llm/models/Qwen3.5-2B/imatrix.gguf \
  --output-format gguf -ngl 0 --no-ppl -t 8
```

### Step 2: Quantize with type mapping

IQ2_XXS **requires** an importance matrix. Tensors without imatrix entries must
use a non-IQ type (q4_K, q5_K, etc.) or the quantize will crash with
`GGML_ASSERT(imatrix != NULL)`.

Our tensor type file at `/tmp/tensor_types.txt` uses `name=type` format:

```
# Generate the type map (matches Unsloth's distribution):
./build/bin/llama-quantize --dry-run \
  --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf /dev/null IQ2_XXS \
  2>/tmp/dryrun.txt

# Run actual quantize:
./build/bin/llama-quantize \
  --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
  --tensor-type-file /tmp/tensor_types.txt \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
  /tmp/quantized-model.gguf
```

### CAVEAT: 9 tensors missing from Unsloth imatrix

The following tensors have no imatrix data and **must** use a type that doesn't
require imatrix (q4_K, q5_K, q2_K) — never iq2_xxs or iq2_xs:

```
token_embd.weight           → q5_K
blk.24.attn_k.weight        → q4_K
blk.24.attn_output.weight   → q4_K
blk.24.attn_q.weight        → q4_K
blk.24.attn_v.weight        → q4_K
blk.24.ffn_down.weight      → q4_K
blk.24.ffn_gate.weight      → q4_K
blk.24.ffn_up.weight        → q4_K
blk.24.nextn.eh_proj.weight  → q4_K
```

The tensor type file at `/tmp/tensor_types.txt` already accounts for these.

### Type distribution produced

| Type | Count | Size | Notes |
|------|-------|------|-------|
| iq2_xxs | 160 | 291 MB | Main quantization for FFN + most attention |
| q4_K | 30 | 164 MB | attn_qkv (SSM), attn_v (full attn), blk.24 fallback |
| q2_K | 3 | 12 MB | ffn_down in SSM layers 0-2 |
| q5_K | 1 | 333 MB | token embeddings |
| f32 | 140 | ~1 MB | Norms, biases (left unquantized) |
| **Total** | **335** | **~776 MB** | |

## Evaluation (LOCKED — never change these flags)

**THE EXACT COMMAND.** This is what generated the BF16 reference logits.
All KL divergence evaluations must use the same parameters:

```bash
./build/bin/llama-perplexity \
  -m MODEL.gguf \
  -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
  -t 8 -c 512 --chunks 200 \
  -fa on --cache-type-k bf16 --cache-type-v bf16 \
  --no-mmap -ngl 999 -np 1 \
  --kl-divergence \
  --kl-divergence-base /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits
```

**Never change** the context length (512), chunk count (200), cache types,
or any evaluation parameters. The reference logits were generated with these
exact settings — changing them invalidates all historical comparisons.

## Setup

The research branch is `auto_research_llama`. All changes happen in this repo.

### Files you edit (free to modify):

```
ggml/src/ggml-quants.c         IQ2_XXS quantization kernels + codebook initialization
ggml/src/ggml-quants.h         Quantization function declarations
ggml/src/ggml.c                Block struct layouts, dequant helpers
ggml/include/ggml.h            GGML_TYPE enums, type traits
tools/quantize/quantize.cpp    Quantize driver, layer categories, CLI flags
```

### Files that are LOCKED (never touch):

```
tools/perplexity/**                    Evaluation source and binary — THE JUDGE
tools/auto-quantize/auto_quantize.py   Experiment orchestrator — immutable
/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits   Reference logits — NEVER regenerate
/home/user/llm/wikitext-2-raw/wiki.test.raw                Evaluation data — NEVER modify
```

### CRITICAL — Reference logits already exist

The BF16 full-precision logits for Qwen3.5-2B were pre-computed once and cached.
**Never regenerate them.** All KL divergence evaluations must use this file via
the `--kl-divergence-base` flag in the evaluation command above.

Regenerating the reference logits would invalidate comparisons with all
historical experiments. The auto_quantize.py orchestrator already uses
this flag — you don't need to pass it manually.

## IQ2_XXS Architecture

### Block structure (block_iq2_xxs)

```c
typedef struct {
    ggml_half d;              // 2 bytes — super-block scale
    uint16_t qs[QK_K/8];      // 64 bytes — 32 entries × 16 bits each
                               //   8 bits: grid index (0-255)
                               //   7 bits: sign bits for 7 elements
                               //   (sign of 8th element from parity)
} block_iq2_xxs;              // 66 bytes for 256 elements = 2.0625 bpw
```

### Codebook (iq2xxs_grid)

A **global, hardcoded** 256-entry table of 8D centroids:
- Each entry: 8 × int8 values (stored in `uint64_t`)
- Values are odd integers {1, 3, 5, 7} from the E8 lattice subset
- Initialized at runtime in `iq2xs_init_impl()` from `kgrid_2bit_256[]`
- Table size: 256 × 8 = 2 KB (fits in L1 cache)

### Dequant formula

```c
y = db * grid[position] * sign
// db = d * (0.5 + 4bit_scale) * 0.25  (from super-block scale)
// grid[position] = one int8 from the selected codebook entry
// sign = ±1
```

### Where the codebook lives

- `ggml-common.h:550-615` — `iq2xxs_grid` table (GGML_TABLE_BEGIN/END)
- `ggml-common.h:516-548` — `kgrid_2bit_256[256]` (raw 2-bit patterns)
- `ggml-quants.c:2892-2915` — `iq2xs_init_impl()` (runtime expansion to int8 grid)
- `ggml-quants.c:2555-2578` — `dequantize_row_iq2_xxs()` (uses grid for lookups)
- `ggml-quants.c:3309-3331` — `iq2_find_best_neighbour()` (nearest codebook entry search)

### Path A — Per-layer learned codebooks

The primary research direction: replace the **global hardcoded `iq2xxs_grid`**
with per-layer K-means-optimized grids. The grid entries become `uint64_t`
arrays stored per layer (metadata overhead, not per-block). Since the
dequant reads from a table pointer, **zero dequant/SIMD/GPU code changes**
are needed — the pointer just points to a different location per layer.

## Experiment Loop

```
1. READ STATE:
   - Read tools/auto-quantize/results.tsv for last experiments
   - Read tools/auto-quantize/idea_ledger.md for past hypotheses
   - Read the current code in ggml-quants.c (IQ2_XXS functions)
   - Determine current best KL and what's been tried

2. PROPOSE HYPOTHESIS:
   - Formulate: "I believe technique X will reduce KL because Y"
   - Log to tools/auto-quantize/idea_ledger.md with timestamp

3. EDIT CODE:
   - Modify ggml-quants.c (codebook initialization, quantize functions)
   - If new parameters needed, add CLI flags in quantize.cpp
   - If new functions/declarations, edit ggml-quants.h

4. COMMIT CODE:
   git add ggml/src/ggml-quants.c [other modified files]
   git commit -m "exp-NNN: <description of change>"

5. BUILD:
   cmake --build build/ -j

6. QUANTIZE:
   ./build/bin/llama-quantize \
     --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
     --tensor-type-file /tmp/tensor_types.txt \
     /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
     /tmp/quantized-model.gguf

7. EVALUATE (use the exact LOCKED command — never change flags):
   ./build/bin/llama-perplexity \
     -m /tmp/quantized-model.gguf \
     -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
     -t 8 -c 512 --chunks 200 \
     -fa on --cache-type-k bf16 --cache-type-v bf16 \
     --no-mmap -ngl 999 -np 1 \
     --kl-divergence \
     --kl-divergence-base /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits

8. RECORD:
   - Append results to tools/auto-quantize/results.tsv
   - Record: KL, PPL, Same top P, model size, tokens/sec

9. EVALUATE OUTCOME:
   - If KL improved over global best → keep code
   - If KL regressed → git revert CODE commit (keep results)
   - If build/quantize fails → fix error, retry (max 3 attempts, then move on)

10. SYNTHESIS (every 5 experiments):
    - Write tools/auto-quantize/synthesis.md
    - Summarize: best KL so far, what worked/didn't, top 3 next hypotheses

11. REPEAT from step 1
```

## Lessons Learned from AutoQuant (vektorprime/AutoQuant on Qwen3.5-0.8B)

The AutoQuant agent ran 18 experiments on Q2_K-style quant. Here is what
DIDN'T work (do NOT repeat):

| Failed Technique | KL Regression | Why It Failed |
|-----------------|---------------|---------------|
| **4-bit codebook compression** (16 centroids per layer) | **+115%** (3.36→7.23) | Codebooks need ≥256 levels (Q8) for near-lossless fidelity. 16 levels destroy expressiveness. |
| **Channel-shared codebooks** (K=2 output channels share a codebook) | **+45%** (3.04→4.40) | Per-channel specificity is critical at 2-bit. Even sharing across 2 channels costs too much. |
| **Uniform error diffusion** (no activation-weighting in eps) | **+4.9%** (5.63→5.91) | Error must be inversely weighted by activation magnitude (eps_w = 1/h_g), otherwise low-importance channels get over-amplified. |
| **Outlier-robust scale** (90th percentile, no-clipping) | **+37-58%** | MSE-optimal scale (maxabs or EM-refined) consistently beats percentile-based approaches at 2-bit. |
| **Global scale refinement** (against original weights after loop) | **+0.3%** (barely negative) | Scale refinement applied AFTER error diffusion loop doesn't help; must be integrated INTO the sub-block loop. |
| **bf16 compact format changes** | **+5.7%** | Changing the storage representation (scalar zero_pt + bf16 scales) gave no benefit over existing format. |

### What Worked (in implementation order, for reference)

1. Layer-type error diffusion (attn=0.7, mlp=0.3, lm_head=0.1) — **-6.4%**
2. Non-uniform K-means learned codebooks (4 values per group) — **-40.4%** (biggest breakthrough)
3. Q8 codebook compression + re-optimized codes — **-12.6%**
4. Improved K-means (quantile init, 3×20 iters, act-weighted L2) — **-11.1%**
5. Channel sorting by activation importance — **-6.4%**
6. Per-channel bias correction — **-2.1%**
7. Activation-weighted centroid update — **-9.0%**
8. Code reassignment against Q8-dequant centroids — **-21.6%**

**Final**: KL 6.02 → 2.47 (-58.9% total). The single most important finding: the
hardcoded uniform `{-d, 0, d, 2d}` levels are the bottleneck, not the bit budget.

## Exploration Directions

The primary direction is **Path A: Per-layer learned codebooks**.

### Phase 1: Per-layer codebook learning (K-means on weight data)

1. In `iq2xs_init_impl()`, instead of initializing from the hardcoded
   `kgrid_2bit_256[]`, run K-means clustering on the target weight tensor:
   - Extract 8-dimensional vectors from the weight matrix
   - Initialize centroids from the existing E8-based grid (warm start)
   - Run 20-50 iterations of activation-weighted K-means (L2 distance)
   - Store the learned grid in `iq2_data[1].grid`

2. The grid pointer replaces the hardcoded table. The CPU (scalar + AVX2)
   and CUDA dequant paths must be modified to accept a per-tensor grid
   pointer instead of reading the compile-time constant. This is a
   mechanical change — add a `const uint64_t * grid` parameter — and
   does NOT require algorithm changes.

3. Storage: the grid is ~2 KB per layer (256 × 8 bytes). For 25 layers,
   that's ~50 KB of metadata — negligible vs model size.

4. With per-layer codebooks:
   - Each layer's weights are quantized against its own optimized grid
   - The neighbor graph (`kneighbors_q2xs`) must be regenerated for each
     layer (done once at quantization time, O(4096 × 256 × 8))

### Phase 2: Codebook re-optimization (Exp-009b style)

5. After learning per-layer codebooks, re-assign codes against the
   Q8-dequantized centroids (since grid entries are int8, this is the
   natural format). This fixes the float-optimal vs int8-stored inconsistency
   that gave AutoQuant a -21.6% KL improvement in exp-018.

### Phase 3+: Advanced directions

- **Per-tensor-type codebooks**: different grid for attention vs MLP tensors
- **Per-super-block scale optimization**: joint optimization of grid index
  and super-block scale d (currently scale is optimized separately)
- **Activation-weighted K-means**: weight the K-means objective by activation
  statistics from the imatrix (mimicking AutoQuant's activation-weighted approach)
- **Multi-trial K-means**: run multiple random initializations, keep best

### Novel Directions to Consider

- Importance-aware K-means clustering within blocks
- Cross-block joint optimization of adjacent sub-blocks
- Non-uniform quantization levels (different codebook value distributions)
- Residual quantization (quantize → compute residual → quantize residual)
- Block layout changes (different sub-block sizes, different scale encoding)
- Channel grouping/sorting by activation importance before quantization

Novel ideas are more valuable than re-implementing known techniques.

## Backend Scope

**ONLY CPU (x86/AVX2) and CUDA matter.** All other backends are out of scope:

| Skip | Never touch |
|------|-------------|
| ARM NEON, RISC-V, PowerPC, LoongArch, s390x | `ggml-cpu/arch/arm/`, `arch/riscv/`, `arch/powerpc/`, `arch/loongarch/`, `arch/s390/` |
| Metal | `ggml-metal/`, `*.metal` |
| Vulkan | `ggml-vulkan/`, `vulkan-shaders/` |
| WebGPU | `ggml-webgpu/`, `wgsl-shaders/` |
| SYCL | `ggml-sycl/` |

Files you NEED to modify for per-tensor grids:

| Backend | Files |
|---------|-------|
| CPU scalar dequant | `ggml/src/ggml-quants.c:2542` |
| CPU scalar dot | `ggml/src/ggml-cpu/quants.c:883` |
| CPU AVX2 | `ggml/src/ggml-cpu/arch/x86/quants.c:2691-2692` |
| CUDA vec_dot | `ggml/src/ggml-cuda/vecdotq.cuh:997` |
| CUDA mmq | `ggml/src/ggml-cuda/mmq.cuh:2775` |
| CUDA dequant | `ggml/src/ggml-cuda/convert.cu:306` |
| Quantize plumbing | `ggml/src/ggml-quants.c` (quantize impl, rebuild) |
| Chunk dispatch | `ggml/src/ggml.c:7706` |

## Resource Constraints

- Model size: within +10% of IQ2_XXS baseline (currently ~776 MB, max ~854 MB)
- GPU: any of the 3 available (RTX 3080 x2, RTX 3050 x1)
- CPU quantize: uses all available cores
- KL evaluation: uses GPU (ngl=999)
- Eval timeout: max 30 minutes per experiment

## Integrity Rules

- NEVER modify `tools/auto-quantize/auto_quantize.py` (read-only orchestrator)
- NEVER modify `tools/perplexity/` (evaluation source/binary)
- NEVER regenerate the BF16 reference logits
- NEVER modify the Wikitext-2 test data
- NEVER change evaluation parameters (context length, chunks, cache types, etc.)
- ALWAYS record every experiment (even failures) in results.tsv
- NEVER change past rows in results.tsv
- NEVER use `git commit --amend` or rewrite history
- If experiment regresses KL → revert the code, keep the results
- If experiment improves KL → keep the code, it's now the baseline
