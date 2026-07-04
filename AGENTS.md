# AGENTS.md — Auto-Research Sub-Agent Instructions

You are an **auto-research agent** for the IQ2_XXS quantization project.
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
- Edit code → commit with `exp-NNN:` prefix
- Build → quantize → evaluate → record to results.tsv
- If KL improved: keep code. If regressed: `git revert` the code commit.

## Rules

- Never modify `tools/perplexity/`, reference logits, or eval data.
- Never change the evaluation command or its flags.
- Always use `CUDA_VISIBLE_DEVICES=1` for GPU eval.
- Always use `--tensor-type-file tools/auto-quantize/tensor_types_unsloth_match.txt` for quantization.
- Append to results.tsv — never overwrite past rows.
- Check `IMPLEMENTATION_STATUS.md` for known caveats and infrastructure details.
