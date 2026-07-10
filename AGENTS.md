# AGENTS.md — Q4_K_M_CLONE Auto-Research Agent

You are an **auto-research agent** for the Q4_K_M_CLONE quantization research project.

## Objective

Reduce the **GGUF file size** of Q4_K_M_CLONE quantization without degrading
**KL divergence** and **same top p** metrics. The baseline is stock Q4_K_M.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Agent Controller (opencode Task tool)                            │
│                                                                   │
│  Orchestrates the loop. Mediates tool calls. Enforces:            │
│   - LOCKED files (eval, reference logits, Q4_K_M ref)             │
│   - METRIC constraints (size, KLD, same top p) — gatekeeper       │
│   - Git discipline (commit code, revert on regression)            │
│                                                                   │
│  ┌────────────────────────┐  ┌─────────────────────────────────┐ │
│  │ Code Editor             │  │ Experiment Runner               │ │
│  │ - Edits ggml-quants.c   │  │ - Build: cmake --build          │ │
│  │ - ggml-quants.h         │  │ - Quantize: llama-quantize      │ │
│  │ - ggml-common.h         │  │ - Eval: llama-perplexity        │ │
│  │ - ggml.c / ggml-cpu.c   │  │   (LOCKED flags, no changes)    │ │
│  │ - ggml-cuda/*.cu        │  │ - Size: ls -l output.gguf       │ │
│  │ - llama-quant.cpp       │  │                                 │ │
│  │ - Not locked files      │  │                                 │ │
│  └──────────┬─────────────┘  └────────────────┬────────────────┘ │
│             │                                  │                  │
└─────────────┼──────────────────────────────────┼──────────────────┘
              │                                   │
              ▼                                   ▼
┌──────────────────────────┐   ┌───────────────────────────────────┐
│ Experiment Code           │   │ Immutable Evaluation Pipeline      │
│                           │   │                                    │
│ ggml-quants.c             │   │ llama-perplexity — LOCKED          │
│ ggml-common.h             │   │ Reference logits — LOCKED          │
│ ggml-quants.h             │   │ Wikitext-2 test  — LOCKED          │
│ ggml-cpu/ggml-cpu.c       │   │ Q4_K_M ref GGUF — LOCKED          │
│ ggml.c                    │   │                                    │
│ ggml-cpu/ops.cpp          │   │                                    │
│ ggml-cuda/                │   │                                    │
└──────────────────────────┘   └───────────────────────────────────┘
```

**Lock the judge, not the researcher.**

## Model Architecture: Qwen3.5-0.8B

| Property | Value |
|----------|-------|
| Type | Causal Language Model with Vision Encoder |
| Parameters | 0.8B |
| Hidden Dim | 1024 |
| Token Embedding | 248320 (padded, tied to LM output) |
| Layers | 24 |
| Context Length | 262,144 native |
| Hidden Layout | 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |
| Gated DeltaNet | 16 V-heads × 128 + 16 QK-heads × 128 |
| Gated Attention | 8 Q-heads × 256 + 2 KV-heads × 256, RoPE dim 64 |
| FFN Intermediate | 3584 |
| MTP | trained with multi-steps |

## Q4_K_M_CLONE Architecture

Q4_K_M_CLONE is an exact structural copy of Q4_K_M, created as a sandbox for
size-reduction research. It uses the same block structure, quantization algorithm,
and dequantization logic as Q4_K_M.

### Block structure (`block_q4_K_M_CLONE` — identical to `block_q4_K`):
- `ggml_half d` (2 bytes) — super-block scale for quantized scales
- `ggml_half dmin` (2 bytes) — super-block scale for quantized mins
- `uint8_t scales[K_SCALE_SIZE]` (12 bytes) — scales and mins, 6-bit quantized
- `uint8_t qs[QK_K/2]` (128 bytes) — 4-bit quantized values (half-byte per element)
- **Total: 144 bytes per 256 elements = 4.5 bpw**

## Research Direction

**ABSOLUTELY FORBIDDEN: Switching between pre-built quantization types.**
- Do NOT change a tensor from Q6_K to Q5_K, or Q4_K to Q8_0, etc.
- Do NOT selectively quantize attention layers differently from FFN layers.
- Do NOT tweak `use_more_bits()` or `llama-quant.cpp` per-tensor conditions.
- Using a stock quant type without modifying its implementation is NOT novel research.
- The ONLY allowed changes are to the Q4_K_M_CLONE quant/dequant code itself.

**What TO do: modify the clone's quantization algorithm in `ggml-quants.c`.**
- Change the quantize function: new encoding schemes, custom bit packing
- Change the dequantize function: match your new encoding
- Shrink the block struct — but pair every removed byte with a recovery technique
- Implement codebook quantization, residual quantization, mixed precision
- Exploit weight distribution properties: sparsity, clustering, outlier handling
- Create a genuinely new quant format, not a remix of existing types

**Use the web search tool** to find the latest arXiv papers and research on
LLM weight compression, novel quantization formats, and GPU-friendly encoding
schemes. Search terms like "llm weight compression 2025", "novel quantization
format gptq", "4-bit quantization improvements", "quip quantization" etc.
Get ideas from cutting-edge research before coding.

## Baselines

| Quant | GGUF Size | PPL | KLD | Same top P | RMS Δp |
|-------|-----------|-----|-----|------------|--------|
| BF16 (reference) | ~1.41 GB | 21.5386 | 0.0 (identity) | 100% | 0.0% |
| **Q4_K_M (stock)** | **~505 MB** | **22.4499** | **0.062947** | **86.387%** | **5.753%** |

Goal: reduce GGUF size below 505 MB while maintaining KLD ≤ 0.062947
and same top p ≥ 86.387%.

## Models & Data Paths

| Resource | Path |
|---|---|
| BF16 model | `/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf` |
| Q4_K_M reference (LOCKED) | `/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf` |
| Reference logits (LOCKED) | `/home/user/llm/models/Qwen3.5-0.8B-BF16.logits` |
| Eval data (LOCKED) | `/home/user/llm/wikitext-2-raw/wiki.test.raw` |
| Quantized output (experiments) | `/tmp/qwen3.5-0.8b-q4km-clone-exp.gguf` (overwrites each experiment) |

## GPU Info

| Device | Model | VRAM | Use |
|--------|-------|------|-----|
| 3 | RTX 3050 | 5806 MB | USE FOR EVAL |

## Locked Files (do not modify)

- `tools/perplexity/**` — eval binary and source
- `/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf` — BF16 input model
- `/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf` — Q4_K_M reference
- `/home/user/llm/models/Qwen3.5-0.8B-BF16.logits` — reference logits
- `/home/user/llm/wikitext-2-raw/wiki.test.raw` — eval data

Everything else in the repo is fair game. Do NOT maintain a whitelist of
editable files; that list will always be stale.

## Experiment Loop

```
0. READ Q4_K_CLONE_INFO.md AND PITFALLS.md — understand every file the clone
   touches and known traps before writing any code.

1. READ STATE: results.tsv, IDEA_LEDGER.md, SYNTHESIS.md, current code
   - IDEA_LEDGER.md has an Experiment Index at the top — read that first for a
     quick scan of what's been tried. Then grep for details.
   - Also commit any uncommitted results.tsv changes from prior experiments:
     git add results.tsv IDEA_LEDGER.md && git commit -m "auto-research: churn"

2. RESEARCH + PROPOSE HYPOTHESIS:
   - Search the web for recent arXiv papers and novel quantization techniques
   - Log your hypothesis to IDEA_LEDGER.md
   - Commit: git add IDEA_LEDGER.md && git commit -m "exp-NNN: hypothesis: ..."

3. EDIT CODE: modify source files (DO NOT COMMIT yet)

4. BUILD (CPU + CUDA) — 20-minute timeout:
   timeout 1200 cmake --build build -j16

5. QUANTIZE — must finish in ≤20 minutes (HARD limit):
   rm -f /tmp/qwen3.5-0.8b-q4km-clone-exp.gguf
   timeout 1200 ./build/bin/llama-quantize \
     /home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
     /tmp/qwen3.5-0.8b-q4km-clone-exp.gguf Q4_K_M_CLONE

6. EVALUATE (always device 3, locked flags):
   CUDA_VISIBLE_DEVICES=3 timeout 1200 build/bin/llama-perplexity \
     -m /tmp/qwen3.5-0.8b-q4km-clone-exp.gguf \
     -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
     -t 8 -c 256 --chunks 250 -fa on \
     --cache-type-k bf16 --cache-type-v bf16 \
     --no-mmap -ngl 999 -np 1 \
     --kl-divergence --kl-divergence-base /home/user/llm/models/Qwen3.5-0.8B-BF16.logits
   Capture ALL lines of output — do NOT grep-filter. You need ALL of:
   ====== Perplexity statistics ======
     Mean PPL(Q), Mean PPL(base), Cor(...), Mean ln(PPL(Q)/PPL(base)),
     Mean PPL(Q)/PPL(base), Mean PPL(Q)-PPL(base)
   ====== KL divergence statistics ======
     Mean, Maximum, 99.9%, 99.0%, 95.0%, 90.0%, Median,
     10.0%, 5.0%, 1.0%, 0.1%, Minimum
   ====== Token probability statistics ======
     Mean, Maximum, 99.9%, 99.0%, 95.0%, 90.0%, 75.0%, Median,
     25.0%, 10.0%, 5.0%, 1.0%, 0.1%, Minimum, RMS Δp, Same top p

7. MEASURE SIZE: check GGUF size:
   ls -l /tmp/qwen3.5-0.8b-q4km-clone-exp.gguf

8. RECORD: append results to results.tsv — ALL 42 columns:
   timestamp, exp_id, code_sha, parent_sha, description, status,
   gguf_size_bytes, base_type, diffusion, refine_iterations, model_size_mb,
   quantize_time_s, eval_time_s, tokens_per_sec,
   ppl, ppl_base, ppl_cor, ppl_ln_ratio, ppl_ratio, ppl_diff,
   kld_mean, kld_max, kld_99_9, kld_99_0, kld_95_0, kld_90_0,
   kld_median, kld_10_0, kld_5_0, kld_1_0, kld_0_1, kld_min,
   dp_mean, dp_max, dp_99_9, dp_99_0, dp_95_0, dp_90_0, dp_75_0,
   dp_median, dp_25_0, dp_10_0, dp_5_0, dp_1_0, dp_0_1, dp_min,
   dp_rms, same_top_p
   Tab-separated, one row per experiment.

9. EVALUATE OUTCOME and COMMIT:
   a) If SIZE REDUCED and KLD/same top p maintained (or improved):
      git add -A && git commit -m "exp-NNN: hypothesis — results: size=..., KLD=..., same_top_p=..."

   b) If REGRESSED or NULL:
      # Revert ALL code. Keep only persistent record-keeping files:
      git add results.tsv IDEA_LEDGER.md SYNTHESIS.md
      git checkout -- .
      git add results.tsv IDEA_LEDGER.md SYNTHESIS.md
      git commit -m "auto-research: record exp-NNN (regression/null)"

   c) If build/quantize FAILS:
      git checkout -- .
      git commit --allow-empty -m "auto-research: record exp-NNN (failed)"

10. SYNTHESIS (every 5 experiments): update SYNTHESIS.md

11. UPDATE PITFALLS.md: if you encountered any bug, misconfiguration, or
    time-sink that could be avoided in future experiments, add 1-3 sentences
    to PITFALLS.md before returning.

12. REPEAT
```

## Integrity Rules

- NEVER regenerate the BF16 reference logits or modify eval data/models
- NEVER change evaluation parameters (context length, chunks, cache types, device)
- ALWAYS record every experiment (even failures) in results.tsv
- NEVER change past rows in results.tsv
- NEVER use `git commit --amend` or rewrite history
- Code commits only for improvements. Regressions/null → discard code, keep results.
- Always reset working tree to clean state (no experimental code) before next experiment.
