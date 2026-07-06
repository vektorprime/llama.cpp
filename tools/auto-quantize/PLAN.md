# PLAN.md — LLM-Driven Auto Research for llama.cpp IQ4_XS Quantization

## Overview

Build an **autonomous LLM agent** that improves llama.cpp's IQ4_XS quantization
by running the Karpathy-style auto-research loop. The primary research directions
are transferred from the IQ2_XXS research (best KL 0.662001 on Qwen3.5-2B).

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Agent Controller (opencode Task tool)                            │
│                                                                   │
│  Orchestrates the loop. Mediates tool calls. Enforces:            │
│   - LOCKED files (eval, reference logits)                         │
│   - METRIC constraints (KL, size, tok/s) — gatekeeper             │
│   - Git discipline (commit code, revert on regression)            │
│                                                                   │
│  ┌────────────────────────┐  ┌─────────────────────────────────┐ │
│  │ Code Editor             │  │ Experiment Runner               │ │
│  │ - Edits ggml-quants.c   │  │ - Build: cmake --build          │ │
│  │ - ggml-quants.h         │  │ - Quantize: llama-quantize      │ │
│  │ - ggml-common.h         │  │   with imatrix                  │ │
│  │ - All other source files│  │ - Eval: llama-perplexity        │ │
│  │ - Not locked files      │  │   (LOCKED flags, no changes)    │ │
│  └──────────┬─────────────┘  └────────────────┬────────────────┘ │
│             │                                  │                  │
└─────────────┼──────────────────────────────────┼──────────────────┘
              │                                   │
              ▼                                   ▼
┌──────────────────────────┐   ┌───────────────────────────────────┐
│ Experiment Code           │   │ Immutable Evaluation Pipeline      │
│                           │   │                                    │
│ ggml-quants.c             │   │ llama-perplexity  — LOCKED         │
│ ggml-common.h             │   │ Reference logits  — LOCKED         │
│ ggml-quants.h             │   │ Wikitext-2 test   — LOCKED         │
└──────────────────────────┘   └───────────────────────────────────┘
```

## Security Model

**Lock the judge, not the researcher.**

### Locked (Untouchable) Files

```
tools/perplexity/**                    The evaluation binary and source
/home/user/llm/models/Qwen3.6-27B/Qwen3.6-27B-BF16.logits   Reference logits (NEVER regenerate)
/home/user/llm/wikitext-2-raw/wiki.test.raw                Evaluation dataset
```

### Research Areas

1. **Weight formula tuning**: Transfer the `qw * powf(sigma2_per_ib + xb^2, p)` exponent
   optimization from IQ2_XXS (where p=0.30 was optimal).
2. **Superblock d optimization**: Try candidates of d, pick min weighted reconstruction error.
3. **Post-d refinement**: Re-evaluate 4-bit codebook selection after d is known.
4. **Codebook tuning**: Explore learning the `kvalues_iq4nl` table.
5. **waux softening**: Apply the waux softening concept to any internal selection processes.

## Baselines

| Experiment | KL | PPL | Same top P | Size |
|-----------|-----|-----|------------|------|
| IQ4_XS default (first experiment) | TBD | TBD | TBD | TBD |

## Agent Research Loop

Each iteration follows the protocol in `program.md`.
