# IQ4_XS Auto-Research Idea Ledger

## Experiment Index

| Exp | Description | Outcome |
|-----|-------------|---------|

---

> **IQ2_XXS ledger archived in git history.** The IQ2_XXS research (exp-001 through exp-107)
> achieved best KL 0.662001 on Qwen3.5-2B. See the `auto_research_llama` branch or
> earlier commits in `auto_research_iq4xs_llama` for the full IQ2_XXS experiment index
> and detailed hypotheses.
>
> Key transferable findings:
> - Weight exponent 0.30 (main), 0.35 (d-opt/post-d) with per-sub-block sigma2
> - Superblock d optimization: 65 candidates, 0.5% step, ±16% range
> - Post-d grid index recomputation with quantized scale
> - waux = powf(weight, 0.20) for neighbor search softening
>
> These should be validated on IQ4_XS.
