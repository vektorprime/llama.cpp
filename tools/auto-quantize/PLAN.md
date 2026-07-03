# PLAN.md — LLM-Driven Auto Research for llama.cpp IQ2_XXS Quantization

## Overview

Build an **autonomous LLM agent** that improves llama.cpp's IQ2_XXS quantization
by running the Karpathy-style auto-research loop. The primary research direction
is **Path A: Per-layer learned codebooks** — replacing the single global codebook
with K-means-optimized grids per layer, at zero storage/dequant cost.

The agent explores **freely** across the entire ggml quantization codebase.
Safety comes from **metric-based constraints** (KL divergence, model size,
tokens/sec), not from restricting which files the agent can edit. The evaluation
pipeline is locked and immutable — it is the honest referee.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Agent Controller (opencode Task tool)                            │
│                                                                   │
│  Orchestrates the loop. Mediates tool calls. Enforces:            │
│   - LOCKED files (eval, orchestrator, reference logits)           │
│   - METRIC constraints (KL, size, tok/s) — gatekeeper             │
│   - Git discipline (commit code, revert on regression)            │
│                                                                   │
│  ┌────────────────────────┐  ┌─────────────────────────────────┐ │
│  │ Code Editor             │  │ Experiment Runner               │ │
│  │ - Edits ggml-quants.c   │  │ - Build: cmake --build          │ │
│  │ - Edits ggml-common.h   │  │ - Quantize: llama-quantize      │ │
│  │ - Edits quantize.cpp    │  │   with imatrix + tensor types   │ │
│  │ - All other source files│  │ - Eval: llama-perplexity        │ │
│  │ - Not locked files      │  │   (LOCKED flags, no changes)    │ │
│  └──────────┬─────────────┘  └────────────────┬────────────────┘ │
│             │                                  │                  │
└─────────────┼──────────────────────────────────┼──────────────────┘
              │                                   │ subprocess
              ▼                                   ▼
┌──────────────────────────┐   ┌───────────────────────────────────┐
│ Experiment Code           │   │ Immutable Evaluation Pipeline      │
│                           │   │                                    │
│ ggml-quants.c             │   │ auto_quantize.py  — LOCKED         │
│ ggml-common.h             │   │ llama-perplexity  — LOCKED         │
│ quantize.cpp              │   │ Reference logits  — LOCKED         │
│ ggml-quants.h             │   │ Wikitext-2 test   — LOCKED         │
└──────────────────────────┘   └───────────────────────────────────┘
```

## Security Model

**Lock the judge, not the researcher.**

The evaluation pipeline is the immutable referee — if the agent's changes
genuinely improve quantization quality, the locked eval will confirm it. If
the agent tries anything that doesn't actually help (or makes things worse),
the metrics will show it and the code gets reverted.

### Where the Agent Can Edit Freely

```
ggml/src/ggml-quants.c         IQ2_XXS quantization + codebook initialization
ggml/src/ggml-quants.h         Quantization function declarations
ggml/src/ggml-common.h         Codebook tables (iq2xxs_grid, kgrid_2bit_256, etc.)
ggml/src/ggml.c                Block struct layouts, dequant helpers
ggml/include/ggml.h            GGML_TYPE enums, type traits
tools/quantize/quantize.cpp    Quantize driver, layer categories, CLI flags
src/llama-quant.cpp            Tensor-to-type assignment logic
```

### Locked (Untouchable) Files

```
tools/perplexity/**                    The evaluation binary and source
tools/auto-quantize/auto_quantize.py   Experiment orchestrator
/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits   Reference logits (NEVER regenerate)
/home/user/llm/wikitext-2-raw/wiki.test.raw                Evaluation dataset
```

**CRITICAL — Reference logits already exist.** The BF16 full-precision logits
for Qwen3.5-2B were pre-computed once and cached at
`/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits`. All KL divergence
evaluations must use this file via the locked evaluation command. Never
regenerate the reference logits. They are the fixed baseline against which
all improvements are measured.

### Metric Constraints (Enforced per Experiment)

Every experiment measures and records all five metrics. The KL divergence is
the primary optimization target; the others act as guardrails.

| Metric           | Source in llama-perplexity output  | Target     |
|-----------------|------------------------------------|------------|
| **KL divergence** | `Mean KLD: <value>`               | Primary — optimize downward |
| **Perplexity**    | `Mean PPL(Q): <value>`            | Secondary — track alongside KL    |
| **Same top P**    | `Same top p: <value>`             | Secondary — % where argmax matches reference |
| **Model size**    | `du -sm output.gguf`              | Guardrail — < baseline + 10% |
| **Tokens per sec**| `<value> tokens per second`       | Guardrail — > 50% of baseline |
| Build success     | cmake exit code                   | Must be 0                          |
| Eval completion   | Perplexity exit code + timeout    | Must finish within 30 min          |

If any guardrail constraint fails (size, speed, build, eval), the experiment
is logged as `failed_<reason>` and the code change is reverted.

## IQ2_XXS Baselines

| Experiment | KL | PPL | Same top P | Size |
|-----------|-----|-----|------------|------|
| Q2_K baseline | 3.241 | 334.41 | 27.07% | 620 MB |
| Unsloth UD-IQ2_XXS | 0.721 | 26.44 | 60.25% | 733 MB |
| Our IQ2_XXS | 0.993 | 34.75 | 55.89% | 776 MB |

## Agent Research Loop

Each iteration follows this sequence:

```
1. READ STATE:
   - results.tsv (last 10 experiments)
   - Current best KL and params
   - idea_ledger.md (past hypotheses and outcomes)
   - Relevant source code (ggml-quants.c IQ2_XXS functions, ggml-common.h tables)

2. PROPOSE HYPOTHESIS:
   - "I believe technique X will reduce KL because..."
   - Record hypothesis in idea_ledger.md

3. EDIT CODE:
   - Modify ggml-quants.c, ggml-common.h, quantize.cpp, etc.
   - Do NOT touch locked files

4. COMMIT:
   git add <modified files>
   git commit -m "exp-NNN: <one-line description>"

5. BUILD + QUANTIZE + EVAL:
   cmake --build build/ -j
   llama-quantize --imatrix imatrix_unsloth.gguf --tensor-type-file /tmp/tensor_types.txt \
     BF16.gguf /tmp/quantized-model.gguf
   llama-perplexity (LOCKED command — never change flags)

6. EVALUATE OUTCOME:
   - KL improved → keep code
   - KL regressed → git revert code commit (keep results)
   - Build/quantize fails → fix, retry (max 3x)

7. SYNTHESIS (every 5 experiments):
   - Write synthesis.md: best KL, what worked/failed, top 3 next hypotheses

8. REPEAT
```

## IQ2_XXS Quantization Workflow

### The tensor type file

IQ2_XXS **requires** an importance matrix. Tensors without imatrix entries
(must use non-IQ type) are pre-mapped in `/tmp/tensor_types.txt`. The file
uses `name=type` format with 195 entries covering all weight tensors.

### CAVEAT: 9 missing imatrix entries

The Unsloth imatrix covers 186/195 tensors. These 9 must use non-IQ types:

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

### Evaluation command (LOCKED — NEVER CHANGE)

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

The context length (512), chunk count (200), and cache types are locked
because the reference logits were generated with these exact settings.

## Implementation Phases

### Phase 1: Per-Layer Codebook Learning (Path A — Primary)

| # | Deliverable | Files | Notes |
|---|------------|-------|-------|
| 1 | K-means codebook learner | ggml-quants.c | Replace hardcoded kgrid_2bit_256 with learned grid per layer |
| 2 | Per-layer grid storage | ggml-quants.c | Store learned grids (256 × 8 bytes per layer) |
| 3 | Neighbor graph regeneration | ggml-quants.c | Rebuild kneighbors_q2xs for each per-layer grid |
| 4 | Benchmark vs baselines | results.tsv | KL, PPL, Same top P against Unsloth and our baselines |

### Phase 2: Codebook Re-optimization (Exp-009b)

| # | Deliverable | Files | Notes |
|---|------------|-------|-------|
| 5 | Re-assign codes against int8 grid | ggml-quants.c | Fix float-optimal vs int8-stored inconsistency (-21.6% in AutoQuant) |
| 6 | Activation-weighted K-means | ggml-quants.c | Use imatrix data for weighted centroid updates |

### Phase 3: Advanced Techniques

| # | Deliverable |
|---|------------|
| 7 | Per-tensor-type codebooks (different grids for attn vs MLP) |
| 8 | Joint grid-scale optimization |
| 9 | Multi-trial K-means with best-of-N selection |
| 10 | Channel sorting by activation importance before grouping |

## Reference: IQ2_XXS Architecture

### Block structure
```c
typedef struct {
    ggml_half d;              // 2 bytes — super-block scale
    uint16_t qs[QK_K/8];      // 64 bytes — 32 entries: 8b grid-index + 7b signs + 1b scale
} block_iq2_xxs;              // 66 bytes for 256 elements = 2.0625 bpw
```

### Global codebook (current — to be replaced per-layer)
- `iq2xxs_grid[256]`: 256 entries of 8 × int8 values (2 KB total)
- Initialized from hardcoded `kgrid_2bit_256[256]` in `iq2xs_init_impl()`
- Dequant just reads from table pointer: `y = db * grid[pos] * sign`

### Key source locations
- `ggml/common.h:550-615` — `iq2xxs_grid` table
- `ggml/common.h:516-548` — `kgrid_2bit_256` raw patterns
- `ggml/quants.c:2892-2915` — `iq2xs_init_impl()` initialization
- `ggml/quants.c:2527-2546` — `dequantize_row_iq2_xxs()`
- `ggml/quants.c:3432-3511` — `quantize_row_iq2_xxs_impl()`
- `ggml/quants.c:3309-3331` — `iq2_find_best_neighbour()`

## Backend Scope

**ONLY CPU (x86/AVX2) and CUDA.** All other backends (ARM, RISC-V, PowerPC,
LoongArch, Metal, Vulkan, WebGPU, SYCL) are out of scope and must NOT be
touched.

Files requiring changes for per-tensor grid:
- CPU: `ggml-quants.c`, `ggml-cpu/quants.c`, `ggml-cpu/arch/x86/quants.c`
- CUDA: `ggml-cuda/vecdotq.cuh`, `ggml-cuda/mmq.cuh`, `ggml-cuda/convert.cu`
- Plumbing: `ggml.c:7706`, `llama-quant.cpp:716`

## Files Created by This Plan

| File | Purpose |
|------|---------|
| `tools/auto-quantize/program.md` | Research protocol — the instruction manual for the agent |
| `tools/auto-quantize/auto_quantize.py` | Experiment orchestrator (build → quantize → eval → log) |
| `tools/auto-quantize/PLAN.md` | This implementation plan |
| `tools/auto-quantize/results.tsv` | Experiment results log (append-only) |
| `/tmp/tensor_types.txt` | Type mapping for IQ2_XXS quantization |

## Success Criteria

- [ ] Agent self-recovers from build failures and quantization errors
- [ ] KL divergence on Qwen3.5-2B improves over our IQ2_XXS baseline (0.993 → lower)
- [ ] Or: KL divergence approaches Unsloth baseline (0.721)
- [ ] Model size never exceeds baseline + 10% (~854 MB max)
- [ ] Tokens/sec never drops below 50% of baseline
- [ ] Zero unauthorized edits to LOCKED files (full audit trail via git)
