# AGENTS.md — Auto-Research Sub-Agent Instructions

You are an **auto-research agent** for the IQ4_XS quantization project.
Your sole directive is to follow the protocol in `tools/auto-quantize/program.md`.

## Startup

1. Read `tools/auto-quantize/program.md` in its entirety.
2. Read `tools/auto-quantize/results.tsv` for experiment history.
3. Read `tools/auto-quantize/idea_ledger.md` for past hypotheses.
4. Read `tools/auto-quantize/synthesis.md` for the latest synthesis.
5. Read the current implementation in `ggml/src/ggml-quants.c`.

## Execution

Follow the **Experiment Loop** section in program.md exactly. Every cycle:
- Propose a hypothesis → log to idea_ledger.md
- Edit code → DO NOT commit before build/quantize/eval
- Build → quantize (≤20 min, timeout 1200) → evaluate → record to results.tsv
- If KL improved: keep code. If regressed/null: `git checkout -- ggml/src/ggml-quants.c` to discard code, keep results.

## Rules

- Never modify `tools/perplexity/`, reference logits, or eval data.
- Never change the evaluation command or its flags.
- Always use `CUDA_VISIBLE_DEVICES=0` for GPU eval (RTX 5090, works for this model).
- Append to results.tsv — never overwrite past rows.
- Record ALL 41 columns from actual eval output — no truncation, no approximations.
