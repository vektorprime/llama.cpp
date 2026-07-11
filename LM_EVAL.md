# LM-EVAL with Quantized GGUF Models

How to run standard LLM benchmarks against quantized GGUF models served
via llama-server on `fwht_only_clone` branch (or any branch with echo=True
for completions).

## Prerequisites

```bash
pip install lm-eval transformers tenacity --break-system-packages
```

## Why Custom llama.cpp Is Needed

Standard llama-server returns logprobs only for *generated* tokens, not prompt
tokens. lm-eval's `local-completions` model type requires `echo=True` in the
`/v1/completions` endpoint to compute loglikelihood-based metrics (used by
HellaSwag, MMLU-mc, ARC, PIQA, Winogrande, LAMBADA, etc.).

**Without our patch:** Continuation logprobs are always empty → scores are
random noise (we observed 26.5% on HellaSwag, matching BF16 and quantized,
confirming the evaluation pipeline was broken, not the model).

**With our patch:** The server returns logprobs for ALL tokens (prompt +
generated) when `echo=True` is set. lm-eval's `parse_logprobs` slices at the
context/continuation boundary using HuggingFace tokenizer alignment.

### Two lm-eval Format Issues (Both Patched)

1. **`content` vs `token_logprobs` format** — llama-server returns logprobs as
   `content: [{token, logprob, top_logprobs}, ...]`. lm-eval's
   `openai_completions.py` expects the legacy OpenAI format
   `token_logprobs: [float, ...]`. Patched `parse_logprobs()` to handle both.

2. **String vs token offsets** — The `--model gguf` lm-eval type uses
   `text_offset` (byte) compared against Python `len(string)` (characters),
   breaking for non-ASCII text. Use `--model local-completions` which
   computes context length from tokenized input.

### Files Modified (Server Side)

| File | Change |
|------|--------|
| `server-common.cpp:804` | Forward `echo` to llama_params (was rejected with throw) |
| `server-schema.cpp:34` | Add `echo` field to completion schema → `task_params.echo` |
| `server-task.h:75,357-358` | `echo` flag + `prompt_probs_output` vector in result struct |
| `server-context.cpp:42-54` | `n_outputs_max` raised to `n_batch` (echo needs all prompt positions as outputs) |
| `server-context.cpp:203-204` | `prompt_probs` vector + `echo` flag in slot struct |
| `server-context.cpp:3494-3498` | Request logits for all prompt positions when `slot.echo=true` |
| `server-context.cpp:3676-3710` | After `llama_decode`, capture prompt logprobs via `llama_get_logits_ith()` |
| `server-context.cpp:1855` | Copy `echo` from task params to slot |
| `server-context.cpp:2190-2191` | Copy `prompt_probs` and `echo` from slot to result |
| `server-task.cpp:398-434` | `to_json_oaicompat()`: prepend prompt probs + prompt text when echo=true |

### Files Modified (lm-eval Python Package)

| File | Change |
|------|--------|
| `lm_eval/models/openai_completions.py:parse_logprobs` | Handle `content` format alongside legacy `token_logprobs` |

This patch is installed in site-packages and must be re-applied on new
environments. See patched code below.

## Task Categories and Compatibility

### Generation Tasks — Always Work

These use `generate_until`, which sends prompts and scores generated text
against reference answers. No `echo=True` needed. Compatible with stock
llama.cpp.

| Task | Type | Description | Notes |
|------|------|-------------|-------|
| `gsm8k` | generate_until | Grade-school math | 5-shot default, ~42% for 0.8B |
| `gsm8k_platinum` | generate_until | Curated GSM8K subset | Has reference answers |
| `gsm_plus` | generate_until | Harder GSM variants | Multi-step reasoning |
| `mgsm` | generate_until | Multilingual GSM | Translations of GSM8K |
| `minerva_math` | generate_until | Hendrycks MATH | Competition-level math |
| `hendrycks_math` | generate_until | Same as minerva_math | Alias |
| `aime` | generate_until | AIME 2024 problems | Very hard, 0.8B ≈ 0% |
| `gpqa` | generate_until | Grad-level physics | Chain-of-thought |

### Loglikelihood Tasks — Require Our echo=True Patch

These compute `P(continuation | context)` via token-level logprobs from the
echoed completions response. Broken without our server patch.

| Task | Type | Description | Notes |
|------|------|-------------|-------|
| `hellaswag` | loglikelihood | Commonsense inference | 4-choice via continuations |
| `arc_easy` | loglikelihood | Science reasoning (easy) | Multiple choice |
| `arc_challenge` | loglikelihood | Science reasoning (hard) | Multiple choice |
| `piqa` | loglikelihood | Physical commonsense | 2-choice |
| `winogrande` | loglikelihood | Pronoun resolution | 2-choice |
| `mmlu` | loglikelihood | 57-subject multitask | Per-subject accuracy |
| `mmlu_pro` | loglikelihood | Enhanced MMLU | 10-answer choices |
| `lambada` | loglikelihood | Language modeling | Next-word prediction |
| `boolq` | loglikelihood | Yes/no QA | Boolean continuation |
| `sciq` | loglikelihood | Science QA | Multiple choice |
| `openbookqa` | loglikelihood | Open-book QA | Multiple choice |
| `truthfulqa` | loglikelihood | Fact vs misconception | Multiple choice via MC1/MC2 |
| `mathqa` | loglikelihood | Math multiple choice | Continuation scoring |

## Usage

### 1. Build and Start Server

```bash
cd llama.cpp
cmake --build build -j16 --target llama-server

# Start server (adjust GPU and model path)
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
    -m /path/to/model.gguf \
    --port 8000 --host 0.0.0.0 \
    -ngl 99 --no-mmap --threads 8
```

Verify echo works:
```bash
curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"meta-llama","prompt":"The capital of France is","max_tokens":1,"logprobs":true,"top_logprobs":5,"echo":true}' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['choices'][0]['logprobs']['content']), 'tokens')"
# Should print: 6 tokens (5 prompt + 1 generated)
```

### 2. Verify Tokenizer Alignment

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B')
print('HF:', tok.encode('The capital of France is', add_special_tokens=False))
"
# Compare with server tokenization:
curl -s -X POST http://localhost:8000/tokenize \
    -H "Content-Type: application/json" \
    -d '{"content":"The capital of France is"}' \
    | python3 -c "import json,sys; print('SRV:', json.load(sys.stdin)['tokens'])"
# Both should produce identical token ID lists
```

If they differ, the task's loglikelihood scores will be wrong — the context
length (ctxlen) computed by the HuggingFace tokenizer won't match the
server's tokenization, causing logprobs to be sliced at the wrong boundary.

### 3. Run Evaluation

**Generation tasks (always work):**
```bash
lm_eval --model local-completions \
    --model_args model=meta-llama,base_url=http://localhost:8000/v1/completions,tokenizer=Qwen/Qwen3.5-0.8B \
    --tasks gsm8k \
    --limit 100 \
    --batch_size 1
```

**Loglikelihood tasks (require our echo=True patch):**
```bash
lm_eval --model local-completions \
    --model_args model=meta-llama,base_url=http://localhost:8000/v1/completions,tokenizer=Qwen/Qwen3.5-0.8B \
    --tasks hellaswag \
    --limit 100 \
    --batch_size 16
```

Key flags:
- `--batch_size 1` for generation tasks (avoids server batch errors)
- `--batch_size 16` for loglikelihood (server handles batched echo=True fine)
- `--limit N` to run N samples instead of full dataset
- `--num_fewshot N` for n-shot evaluation (some tasks default to 5-shot)
- `--output_path results/` to save results to disk

### 4. Compare Quantized vs BF16

```bash
# Quantize with imatrix (see FWHT_IMATRIX_GUIDE.md)
build/bin/llama-quantize --tensor-type-file tensor_map.txt \
    --imatrix imatrix.gguf model-f16.gguf model-q4kmc.gguf Q4_K_M_CLONE

# Run same eval on both
for MODEL in model-f16.gguf model-q4kmc.gguf; do
    pkill -f llama-server; sleep 2
    CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server -m $MODEL \
        --port 8000 -ngl 99 --no-mmap &>/tmp/server.log &
    sleep 5
    lm_eval --model local-completions \
        --model_args model=meta-llama,base_url=http://localhost:8000/v1/completions,tokenizer=Qwen/Qwen3.5-0.8B \
        --tasks $TASK --limit 100 --batch_size 1
done
```

## lm-eval Python Patch

The patch to `openai_completions.py` replaces the `parse_logprobs` static
method to handle llama-server's `content` format (token objects) alongside
the legacy `token_logprobs` format (float list).

```python
# File: lm_eval/models/openai_completions.py
# In class LocalCompletionsAPI, replace parse_logprobs with:

@staticmethod
def parse_logprobs(
    outputs: Union[Dict, List[Dict]],
    tokens: List[List[int]] = None,
    ctxlens: List[int] = None,
    **kwargs,
) -> List[Tuple[float, bool]]:
    res = []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for out in outputs:
        for choice, ctxlen in zip(
            sorted(out["choices"], key=itemgetter("index")), ctxlens
        ):
            assert ctxlen > 0
            lp = choice["logprobs"]
            # llama-server returns content (list of {token,logprob,top_logprobs})
            if "token_logprobs" not in lp and "content" in lp:
                token_logprobs = [t["logprob"] for t in lp["content"]]
                top_logprobs = [{
                    e["token"]: e["logprob"]
                    for e in (t.get("top_logprobs", []) or [])
                } for t in lp["content"]]
            else:
                token_logprobs = lp["token_logprobs"]
                top_logprobs = lp["top_logprobs"]
            logprobs = sum(token_logprobs[ctxlen:-1])
            tokens_logprobs = token_logprobs[ctxlen:-1]
            top_logprobs = top_logprobs[ctxlen:-1]
            is_greedy = True
            for tok, top in zip(tokens_logprobs, top_logprobs):
                if tok != max(top.values()):
                    is_greedy = False
                    break
            res.append((logprobs, is_greedy))
    return res
```

File location: `~/.local/lib/python3.14/site-packages/lm_eval/models/openai_completions.py`

## Known Limitations

1. **HellaSwag scores are lower than expected** — Our echo=True implementation
   works correctly (tokenizer alignment verified, prompt logprobs present),
   but Qwen3.5-0.8B scores ~23% on 100 samples vs community reports of ~40%.
   The model card doesn't list HellaSwag as a benchmark. Possible causes:
   - Qwen3.5 uses a different tokenizer pre-processing than standard
   - The `local-completions` endpoint format differs from the training-time
     tokenization in subtle ways (whitespace, BOS handling)
   - True 0-shot Qwen3.5-0.8B hellaswag score may be lower than reported
     by third parties using different eval pipelines

2. **Slow for full datasets** — HellaSwag with 10K samples × 4 endings = 40K
   API calls. Each call processes a full prompt through the model. Full suite
   takes hours. Use `--limit N` for quick checks.

3. **Server parallelism** — The `--batch_size` flag in lm-eval sends batched
   API requests (multiple prompts in one HTTP call). Server supports this.
   For generation tasks, keep `--batch_size 1` (batched generation causes
   500 errors on some endpoints).

4. **`--model gguf` is broken** — This lm-eval model type uses string-length
   offsets instead of tokenized context length, breaking loglikelihood for
   any non-ASCII text. Always use `--model local-completions`.

5. **`--model local-chat-completions`** — The chat completions endpoint
   handles logprobs differently (only for generated tokens, never for prompt
   tokens). Use `local-completions` for loglikelihood tasks and
   `local-chat-completions` only for generation tasks that need chat
   formatting.

## Cleanup

```bash
# Stop server
pkill -f llama-server

# Remove temp logs
rm -f /tmp/llama-server.log /tmp/server.log
```
