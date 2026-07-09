# Q4_K_M_CLONE Auto-Research Idea Ledger

## Experiment Index

| Exp | Description | Outcome |
|-----|-------------|---------|
| EXAMPLE | This is an example entry — format reference only | Example — not a real experiment |
| exp-001 | Remove ATTENTION_QKV Q5_K boost for clone — keep Q4_K for QKV tensors | Pending |

## NOTE

The entry above is an **example** only. It is not a real experiment.
Use it as a formatting reference for future entries.

---

## exp-001: Remove ATTENTION_QKV Q5_K Boost for Clone

**Hypothesis:** The ATTENTION_QKV tensors are currently boosted to Q5_K (5.5 bpw)
for the clone, same as stock Q4_K_M. Removing this boost and keeping QKV at the
default Q4_K (4.5 bpw) for the clone should reduce GGUF file size while
maintaining KLD ≤ 0.062947 and same top p ≥ 86.387%. QKV tensors are intermediate
in size (not as large as FFN, not as small as output), so the size reduction
should be modest but measurable.

**Changes:**
1. `src/llama-quant.cpp` line 646: Remove `ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M_CLONE`
   from the condition that boosts QKV tensors to Q5_K, so only stock Q4_K_M gets
   the boost and the clone stays at default Q4_K.

**Expected outcome:** GGUF size reduces by ~2-5 MB. KLD may increase slightly but
should stay below 0.062947. Same top p should remain near baseline.

**Actual outcome:** PENDING

**Lesson:** PENDING

---
## exp-E: EXAMPLE — Format Reference

**Hypothesis:** Brief description of what you think will happen and why.
Reference prior findings or code analysis.

**Changes:**
1. Step-by-step list of code changes made
2. Include file paths and approximate line numbers
3. Include key code snippets in ```c blocks

**Expected outcome:** What you predict (e.g., "KL improvement to 0.030"),
with rationale. Include expected quantize time impact.

**Actual outcome:** SUCCESS / FAILED / REGRESSION / NULL — with metrics:
- GGUF size: X MB (vs baseline Y MB)
- KLD: X (vs baseline Y)
- Same top p: X% (vs baseline Y%)

**Lesson:** What was learned. Why did it work or fail? What should be tried
next? What does this reveal about the problem?
