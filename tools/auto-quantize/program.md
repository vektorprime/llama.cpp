# autoresearch — llama.cpp Q2_K quantization

Research driven by LLM agents, following the pattern of karpathy/autoresearch.

## Objective

Minimize **KL divergence** of Q2_K-quantized models against the BF16 reference,
by improving the quantization algorithm in `ggml/src/ggml-quants.c`.

## Setup

The research branch is `llama_auto_research`. All changes happen in this repo.

**Files you edit:**
- `ggml/src/ggml-quants.c` — Q2_K quantization kernels (lines 833-1167)
- `tools/quantize/quantize.cpp` — CLI flags for new parameters

**Files that are read-only:**
- `tools/auto-quantize/auto_quantize.py` — experiment orchestrator
- The compiled `llama-perplexity` binary — fixed evaluation
- The BF16 reference logits — never regenerate

## Experiment Loop

```
1. Modify C source (ggml-quants.c and/or quantize.cpp)
2. git add && git commit the changes
3. Run: python tools/auto-quantize/auto_quantize.py run --description "..." [--diffusion X] [--refine Y]
   (this builds, quantizes, evaluates, and logs)
4. Check tools/auto-quantize/results.tsv for KL
5. If KL improved → keep the commit. If regressed → git revert the code commit.
```

## Models and Data

| Resource | Path |
|---|---|
| BF16 model | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf` |
| BF16 logits | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits` |
| Eval data | `/home/user/llm/wikitext-2-raw/wiki.test.raw` |
| Quantized output | `/home/user/llm/models/Qwen3.5-2B/our-quantized-model.gguf` |

## Where to Edit in ggml-quants.c

### Q2_K reference quantizer (no imatrix)
**File:** `ggml/src/ggml-quants.c`
**Function:** `quantize_row_q2_K_ref` — lines 833-901
**Flow:** For each of 16 sub-blocks (16 elements each) within a super-block (256):
1. `make_qkx2_quants(16, 3, ..., -0.5f, 0.1f, 15, true)` — find scale/min for this sub-block
2. Collect `max_scale` and `max_min` across all 16 sub-blocks
3. Quantize the 16 sub-scales into 4-bit values (stored in `scales[]`)
4. Store super-scales `d` and `dmin` as fp16
5. Re-quantize all weights: `l = round((x + dm) / d)`, clamp to [0,3]

### Q2_K imatrix-aware quantizer
**Function:** `quantize_row_q2_K_impl` — lines 1091-1151
**Flow:** Similar but uses importance weights from imatrix:
- `make_qkx3_quants(16, 3, ..., -0.9f, 0.05f, 36, false)` — wider search, MSE metric
- `make_qp_quants(...)` — joint optimization of 16 sub-scales/mins

### Key helper functions
- `make_qkx2_quants` — line 741 (grid search over scale, MAD metric)
- `make_qkx3_quants` — line 935 (wider grid search, MSE metric)
- `make_qp_quants` — line 1018 (grid + coordinate descent for sub-scales)

### Super-block structure (block_q2_K)
```c
typedef struct {
    uint8_t scales[QK_K/16];  // 16 bytes — 4-bit scale + 4-bit min per sub-block
    uint8_t qs[QK_K/4];       // 64 bytes — 2-bit quants packed 4 per byte
    ggml_half d;              // 2 bytes — super-scale
    ggml_half dmin;           // 2 bytes — super-min-scale
} block_q2_K;  // 84 bytes for 256 elements = 2.625 bpw
```

## Verified KL Improvements to Implement

Sources: vektorprime/AutoQuant experiments on Qwen models
Note: llama.cpp already has imatrix support (= activation-weighted MSE)

### Technique 1: Error Diffusion Across Sub-Blocks
- **Proven win**: 8.53 → 8.43 KL (AutoQuant exp-006)
- **Mechanism**: After quantizing sub-block i, compute error = W_orig - W_quant, then add `diffusion * error` to the next sub-block's weights
- **Implementation point**: Inside the super-block loop in `quantize_row_q2_K_ref` (line 845+) and `quantize_row_q2_K_impl` (line 1105+)
- **Parameter**: diffusion strength (0.0–1.0)

### Technique 2: MSE-Optimal Scale Refinement
- **Proven win**: 8.43 → 6.69 KL with error diffusion (AutoQuant exp-008)
- **Mechanism**: After initial maxabs scale, run 2-3 EM iterations:
  `scale = Σ(w * q) / Σ(q * q)` where q = clamp(round(w/scale), 0, maxq)
- **Implementation point**: Inside `make_qkx2_quants` or in the calling function after the initial scale is computed

### Technique 3: Layer-Type-Specific Parameters
- **Proven win**: 6.02 → 5.63 KL (AutoQuant exp-003, 0.8b branch)
- **Mechanism**: Different diffusion/refine strength per tensor category
  - attn (q/k/v/o): diffusion = 0.7
  - mlp (gate/up/down): diffusion = 0.3
  - lm_head: diffusion = 0.1
  - embed: default
- **Implementation point**: The quantize driver (`tools/quantize/quantize.cpp`) already categorizes tensors. Expose parameters per category via CLI flags.

## Resource Constraints

- Model size: within +10% of pure Q2_K baseline (currently ~683 MB max)
- GPU: any of the 3 available (RTX 3080 x2, RTX 3050 x1)
- CPU quantize: uses all available cores
- KL evaluation: uses GPU (ngl=999)

## Integrity Rules

- NEVER modify `tools/auto-quantize/auto_quantize.py` (read-only orchestrator)
- NEVER regenerate the BF16 reference logits
- ALWAYS record every experiment (even failures) in results.tsv
- NEVER change eval parameters (context length, data file, etc.)
- NEVER edit past rows in results.tsv
- NEVER `git commit --amend` or rewrite history
