# RYS (Repeat Your Self) -- Layer Looping in llama.cpp

## Core Concepts (from dnhkng's RYS papers)

### Three-Phase Hypothesis
Transformers have functional anatomy:
1. **Early layers (~0-15)**: Decode surface-form tokens into internal representation
2. **Middle layers (~16-48)**: Reason in a format-agnostic "semantic hub" space
3. **Late layers (~49-63)**: Re-encode internal representation back to surface-form tokens

### RYS (Repeat Your Self)
Duplicating a block of middle layers at inference time -- with NO weight changes, NO training -- can measurably improve model performance. The duplicated layers reuse the same weights; the model just runs through them multiple times.

### Layer Pattern
```
expand_single_block(num_layers, (i, j)):
    return list(range(0, j)) + list(range(i, num_layers))
```
For (24, 35) on 64 layers: [0..34] + [24..63] = 75 effective layers
Layers 24-34 appear twice, providing extra "thinking" depth.

### Inference-Time Looping Approach (our implementation)
Instead of physically duplicating layers in the model file, we loop back during graph building:
```
for il in 0..23: process normally
for il in 24..35: process (first pass)
for il in 24..35: process again (loop -- reuses same weights)
for il in 36..63: process normally
```
Total effective layers: 64 + 12 = 76

## Target Model: Qwen3.6 27B
- Architecture: `qwen3next` (LLM_ARCH_QWEN3NEXT)
- Layers: 64
- Hidden Dim: 5120
- Attention: Gated DeltaNet (linear attn, recurrent) alternating with Gated Attention
  - Pattern: 3x(Gated DeltaNet -> FFN) -> 1x(Gated Attention -> FFN), repeated 16 times
  - Every 4th layer is full attention, others are linear attention
- Loop target: layers 24 to 35 (inclusive) -- 12 layers repeated once = 12 extra layer computations

## RYS Paper Summary (Qwen3.5-27B Pareto)
| Config | Extra Layers | Overhead | Math delta | EQ delta |
|--------|-------------|----------|-----------|----------|
| (33,34) | +1 | +1.56% | +0.018 | +0.094 |
| (31,34) | +3 | +4.69% | +0.021 | +0.097 |
| (30,35) | +5 | +7.81% | +0.028 | +0.098 |
| (26,34) | +8 | +12.5% | +0.028 | +0.101 |

## Implementation

### CLI Parameters
- `--loop-layer-start N`: First layer of the loop block (default: -1 = disabled)
- `--loop-layer-stop N`: Last layer of the loop block, inclusive (default: -1 = disabled)
- `--loop-count N`: Number of extra iterations (default: 1)
- `--custom-logs`: Enable debug logging in llama-cli and llama-perplexity

### Data Flow
```
common_params (common/common.h)
  -> llama_context_params (include/llama.h)
    -> llama_cparams (src/llama-cparams.h)
      -> llm_graph_params (src/llama-graph.h)
        -> model graph builder (src/models/qwen3next.cpp)
```

### Files Modified
| File | Change |
|------|--------|
| ggml/include/ggml.h | Add ggml_set_custom_logs, ggml_custom_logs_enabled |
| ggml/src/ggml.c | Implement custom_logs global flag |
| include/llama.h | Add RYS fields + llama_set_custom_logs |
| src/llama.cpp | Implement llama_set_custom_logs |
| common/common.h | Add loop_layer_start, loop_layer_stop, loop_count, custom_logs |
| common/arg.cpp | Parse --loop-layer-*, --custom-logs |
| common/common.cpp | Wire common_params -> llama_context_params |
| src/llama-cparams.h | Add RYS fields |
| src/llama-context.cpp | Wire cparams from context_params + defaults |
| src/models/qwen3next.cpp | Layer schedule with loop iterations |
| tools/cli/cli.cpp | Wire llama_set_custom_logs |
| tools/perplexity/perplexity.cpp | Wire llama_set_custom_logs |

### KV Cache Considerations
When layers are looped, the same layer index is processed multiple times.
The KV cache for those layers will be written multiple times per forward pass.
This is the INTENDED behavior -- the second pass through layer 24 should see the
KV cache updates from the first pass through layers 24-35.

### Hardware Scope
Only CUDA, CPU, and tensor split pipelines are targeted.
No implementations for other hardware backends (Vulkan, Metal, SYCL, etc.).

## Progress Log

| Step | Status | Notes |
|------|--------|-------|
| Fork repo, create branch | Done | Clean dup_layers_llama on vektorprime/llama.cpp |
| Understand RYS concept | Done | Three-phase hypothesis, layer duplication |
| custom_logs infra (ggml) | Done | Global flag in ggml.c, wrapper in llama.cpp |
| CLI params (common) | Done | --loop-layer-start/stop/count, --custom-logs |
| Data flow wiring | Done | common -> llama.h -> cparams -> graph builder |
| Layer scheduling (qwen3next) | Done | layer_schedule vector with loop insertions |
| custom_logs in llama-cli | Done | llama_set_custom_logs after backend_init |
| custom_logs in llama-perplexity | Done | llama_set_custom_logs after backend_init |
| Build on 10.0.0.188 | Pending | Remote build with CUDA |
| Validation test | Pending | Test with Qwen3.6 27B, loop 24-35 |
