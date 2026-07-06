# autoresearch — llama.cpp IQ4_XS quantization

Research driven by an autonomous LLM agent, following the pattern of
karpathy/autoresearch. The agent runs inside opencode and follows this
protocol to explore, implement, and evaluate quantization improvements.

## Objective

Minimize **KL divergence** of IQ4_XS-quantized models against the BF16 reference,
by improving the quantization algorithm in `ggml/src/ggml-quants.c`.

The IQ4_XS type uses **4-bit non-linear quantization** with a fixed 16-entry
non-uniform codebook (`kvalues_iq4nl`), sub-blocks of 32 elements, and a
superblock scale `d` shared across 4 sub-blocks with 4-bit quantized levels.

Research directions include:
- **Weight formula tuning** — the `qw * powf(sigma2 + xb^2, p)` pattern
- **Superblock d optimization** — try candidates of `d`, pick min weighted error
- **Post-d refinement** — re-evaluate 4-bit codebook selection after d known
- **Codebook tuning** — the `kvalues_iq4nl` table can potentially be learned

## Baselines (to establish)

| Experiment | KL | PPL | Same top P | Size | Notes |
|-----------|-----|-----|------------|------|-------|
| IQ4_XS default quant (no tuning) | TODO | TODO | TODO | ~ MB | First experiment: establish baseline |

## Models and Data

| Resource | Path |
|---|---|
| BF16 model | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf` |
| BF16 reference logits (LOCKED) | `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits` |
| Imatrix (Unsloth) | `/home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf` |
| Eval data (LOCKED) | `/home/user/llm/wikitext-2-raw/wiki.test.raw` |
| Calibration data | `/tmp/calibration_short.txt` |

## IQ4_XS Architecture

### Block structure (`block_iq4_xs` in `ggml/src/ggml-common.h`):
```c
typedef struct {
    ggml_half d;                    // 2 bytes — superblock scale
    uint16_t scales_h;              // 2 bytes — high bits of 4 sub-block 4-bit scales
    uint8_t  scales_l[QK_K/64];     // 4 bytes — low bits of 4 sub-block 4-bit scales
    uint8_t  qs[QK_K/2];            // 128 bytes — 4-bit quantized values (half byte each)
} block_iq4_xs;                     // 136 bytes for 256 elements = 4.25 bpw
```

### Quantization pipeline:
- 256-element superblock → 4 sub-blocks of 32 elements
- Each sub-block: choose 4-bit codebook entry from `kvalues_iq4nl` (16 non-uniform values)
- Sub-block scale: 4-bit level via `l = nearest_int(0.5*(id*scale-1))`
- Superblock scale `d`: shared, optimized via candidate search (or `max_scale/31`)
- Weight formula: `w[i] = qw[i] * sqrtf(sigma2_per_ib + xb[i]*xb[i])`

### Key function:
- `quantize_row_iq4_nl_impl(QK_K, 32, src, &d, qs, &scales_h, scales_l, scales, weight, L, kvalues_iq4nl, qw, 7)` — the inner quantize kernel for one superblock

### No learned grid:
Unlike IQ2_XXS, IQ4_XS has **no 8D codebook grid, no kmap, no neighbor search**.
The 16-entry non-uniform codebook is fixed (`kvalues_iq4nl`).

## Quantization

```bash
rm -f /tmp/quantized-model.gguf
./build/bin/llama-quantize \
  --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
  /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
  /tmp/quantized-model.gguf IQ4_XS
```

**Quantization limit: 7 minutes HARD.** If a change pushes quantize time over
7 min, it must be abandoned UNLESS it delivers ≥10% KL improvement.

## Evaluation (LOCKED — never change these flags)

```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-perplexity \
  -m MODEL.gguf \
  -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
  -t 8 -c 512 --chunks 200 \
  -fa on --cache-type-k bf16 --cache-type-v bf16 \
  --no-mmap -ngl 999 -np 1 \
  --kl-divergence \
  --kl-divergence-base /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits
```

**Use `CUDA_VISIBLE_DEVICES=1`** (RTX 3080, CC 8.6). Device 0 (RTX 5090, CC 12.0)
lacks kernel images and produces "no kernel image available" errors.

## Editable Files

### Core quantization:
- `ggml/src/ggml-quants.c` — `quantize_row_iq4_nl_impl()`, weight formulas, d optimization
- `ggml/src/ggml-quants.c` — `quantize_iq4_xs()` — IQ4_XS entry point
- `ggml/src/ggml-quants.h` — declarations

### Locked (never touch):
- `tools/perplexity/**` — evaluation source and binary
- `/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits` — reference logits
- `/home/user/llm/wikitext-2-raw/wiki.test.raw` — evaluation data

## GPU Info

| Device | Model | VRAM | CC | Use |
|--------|-------|------|-----|-----|
| 0 | RTX 5090 | 32110 MB | 12.0 | **BROKEN** — "no kernel image available" |
| 1 | RTX 3080 | 20054 MB | 8.6 | USE FOR EVAL |
| 2 | RTX 3080 | 20054 MB | 8.6 | Available |
| 3 | RTX 3050 | 5806 MB | 8.6 | Available |

## Experiment Loop

```
1. READ STATE: results.tsv, idea_ledger.md, synthesis.md, current code
   - idea_ledger.md has an Experiment Index at the top — read that first for a
     quick scan of what's been tried. Then grep for details.
   - Also commit any uncommitted results.tsv changes from prior experiments:
     git add tools/auto-quantize/results.tsv tools/auto-quantize/idea_ledger.md && git commit -m "auto-research: churn"

2. PROPOSE HYPOTHESIS: log to idea_ledger.md (commit so it's saved):
   git add tools/auto-quantize/idea_ledger.md && git commit -m "exp-NNN: hypothesis: ..."

3. EDIT CODE: modify ggml-quants.c (DO NOT COMMIT yet)

4. BUILD: cmake --build build -j16

5. QUANTIZE — must finish in ≤7 minutes (HARD limit, unless ≥10% KL gain):
   rm -f /tmp/quantized-model.gguf
   timeout 420 ./build/bin/llama-quantize \
     --imatrix /home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf \
     /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf \
     /tmp/quantized-model.gguf IQ4_XS

6. EVALUATE (always device 1):
   CUDA_VISIBLE_DEVICES=1 timeout 600 ./build/bin/llama-perplexity \
     -m /tmp/quantized-model.gguf \
     -f /home/user/llm/wikitext-2-raw/wiki.test.raw \
     -t 8 -c 512 --chunks 200 -fa on \
     --cache-type-k bf16 --cache-type-v bf16 \
     --no-mmap -ngl 999 -np 1 \
     --kl-divergence \
     --kl-divergence-base /home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits
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

7. RECORD: append results to results.tsv — ALL 41 columns:
   timestamp, exp_id, code_sha, parent_sha, description, status,
   kl_divergence, base_type, diffusion, refine_iterations, model_size_mb,
   quantize_time_s, eval_time_s, tokens_per_sec,
   ppl, ppl_base, ppl_cor, ppl_ln_ratio, ppl_ratio, ppl_diff,
   kld_mean, kld_max, kld_99_9, kld_99_0, kld_95_0, kld_90_0,
   kld_median, kld_10_0, kld_5_0, kld_1_0, kld_0_1, kld_min,
   dp_mean, dp_max, dp_99_9, dp_99_0, dp_95_0, dp_90_0, dp_75_0,
   dp_median, dp_25_0, dp_10_0, dp_5_0, dp_1_0, dp_0_1, dp_min,
   dp_rms, same_top_p
   Tab-separated, one row per experiment.

8. EVALUATE OUTCOME and COMMIT:
   a) If KL IMPROVED (vs best known):
      git add -A && git commit -m "exp-NNN: hypothesis — results: KL=..."
   b) If KL REGRESSED or NULL:
      git checkout -- ggml/src/ggml-quants.c  (discard code changes)
      git add tools/auto-quantize/results.tsv
      git commit -m "auto-research: record exp-NNN (regression/null)"
   c) If build/quantize FAILS:
      git checkout -- .  (discard all file changes)
      git add tools/auto-quantize/idea_ledger.md
      git commit -m "auto-research: record exp-NNN (failed)"

9. SYNTHESIS (every 5 experiments): update synthesis.md

10. REPEAT
```

## Integrity Rules

- NEVER modify `tools/perplexity/`
- NEVER regenerate the BF16 reference logits
- NEVER modify the Wikitext-2 test data
- NEVER change evaluation parameters (context length, chunks, cache types)
- ALWAYS record every experiment (even failures) in results.tsv
- NEVER change past rows in results.tsv
- NEVER use `git commit --amend` or rewrite history
- Code commits only happen for improvements. Regressions/null → discard code, keep results.
- Always reset working tree to clean state before next experiment.
