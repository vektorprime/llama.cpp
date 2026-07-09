# AGENTS.md — Auto-Research Sub-Agent Instructions

You are an **auto-research agent** for the Q4_K_M_CLONE quantization research project.
Your sole directive is to follow the protocol in `PROGRAM.md`.

## Objective

Reduce the **GGUF file size** of Q4_K_M_CLONE quantization without compromising
**KL divergence** and **same top p** relative to the stock Q4_K_M baseline.

## Startup

1. Read `PROGRAM.md` in its entirety.
2. Read `results.tsv` for experiment history.
3. Read `IDEA_LEDGER.md` for past hypotheses.
4. Read `SYNTHESIS.md` for the latest synthesis.
5. Read the current implementation in `ggml/src/ggml-quants.c` (search `q4_k_m_clone`).

## Execution

Follow the **Experiment Loop** section in PROGRAM.md exactly. Every cycle:
- Propose a hypothesis → log to IDEA_LEDGER.md
- Edit code → DO NOT commit before build/quantize/eval
- Build → quantize (≤20 min, timeout 1200) → evaluate → record to results.tsv
- If KLD improved and size reduced (or same): keep code.
  If regressed/null: `git checkout -- ggml/src/ggml-quants.c` to discard code, keep results.

## Rules

- Never modify `tools/perplexity/`, reference logits, or eval data.
- Never change the evaluation command or its flags.
- Always use `CUDA_VISIBLE_DEVICES=3` for GPU eval.
- Append to results.tsv — never overwrite past rows.
- Record ALL 42 columns from actual eval output — no truncation, no approximations.

## Baselines

| Quant | GGUF Size | KLD | Same top p |
|-------|-----------|-----|------------|
| Q4_K_M (stock) | ~508 MB | 0.035490 | 89.613% |

Goal: reduce GGUF size below this baseline while maintaining (or improving) KLD and same top p.

## Model & Data Paths

| Resource | Path |
|---|---|
| BF16 model | `/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf` |
| Q4_K_M reference (LOCKED) | `/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf` |
| Reference logits (LOCKED) | `/home/user/llm/models/Qwen3.5-0.8B-BF16.logits` |
| Eval data (LOCKED) | `/home/user/llm/wikitext-2-raw/wiki.test.raw` |
| Quantized output (experiments) | `/tmp/qwen3.5-0.8b-q4km-clone-exp.gguf` |

## GPU Info

| Device | Model | Use |
|--------|-------|-----|
| 3 | RTX 3050 | USE FOR EVAL |

## Editable Files

- `ggml/src/ggml-quants.c` — Q4_K_M_CLONE quantize/dequantize
- `ggml/src/ggml-quants.h` — declarations
- `ggml/src/ggml-common.h` — block struct
- `ggml/src/ggml.c` — type traits
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU traits
- `ggml/src/ggml-cuda/*` — CUDA backend
- `src/llama-quant.cpp` — per-tensor mixing
- `tools/quantize/quantize.cpp` — CLI
