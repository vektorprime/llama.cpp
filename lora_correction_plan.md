# LoRA Quantization-Error Adapter: Implementation Plan

## Summary

A 1GB sidecar LoRA GGUF adapter that corrects IQ2_XXS quantization errors by learning `B @ A ≈ W_bf16 - dequant(W_iq2_xxs)`, delivering inference as `y = dequant(W_iq2) @ x + scale * B @ (A @ x)`.

---

## 1. What Already Works (No Changes Needed)

llama.cpp already has a complete LoRA adapter pipeline. The core inference pattern `W_q @ x + scale * (B @ (A @ x))` works today via graph-level decomposition:

**Existing infrastructure:**
- **Graph construction** (`src/llama-graph.cpp:1085-1152`): `build_lora_mm()` and `build_lora_mm_id()` decompose LoRA into standard ggml ops (2× `MUL_MAT` + `SCALE` + `ADD`)
- **Adapter loading** (`src/llama-adapter.cpp:149-418`): Loads GGUF files with `.lora_a`/`.lora_b` tensor pairs, validates shapes, allocates on matching backend
- **Adapter management** (`src/llama-context.cpp:1245-1261`): Multi-adapter support with per-adapter scaling, runtime hot-swap
- **GGUF format** (`convert_lora_to_gguf.py`): Existing converter creates proper adapter GGUF files
- **All backends**: Since LoRA is decomposed into standard ops (`GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_SCALE`), every backend (CUDA, CPU, Metal, Vulkan, SYCL, etc.) already supports it with zero backend-specific code

**A rank-16 IQ2_XXS error-correction adapter for a 96-layer model can be loaded today with:**
```
llama-cli -m base-iq2_xxs.gguf --lora error_lora.gguf
```

---

## 2. What Needs To Be Modified

### 2.1 CUDA Backend: Fused IQ2_XXS + LoRA Kernel

**Priority: HIGH** | **Files:** `ggml/src/ggml-cuda/mmvq.cu`, `ggml/src/ggml-cuda/mmq.cu`, `ggml/src/ggml-cuda/common.cuh`, `ggml/src/ggml-cuda/ggml-cuda.cu`

**Problem:** Currently LoRA adds 4-6 separate kernel launches per adapted weight matrix. For a 32-layer model with LoRA on 7 weight types (Q/K/V/O/gate/up/down), that is:
- 32 × 7 = 224 weight matrices
- 224 × 4 = 896 extra kernel launches per token (2× mul_mat + scale + add)
- For a 96-layer model: ~2,688 extra kernel launches per token
- Each CUDA kernel launch has ~5-15µs overhead → **13-40ms additional latency per token** during single-token decode

**Required changes:**

#### a) Extend fusion argument struct (`common.cuh:1504-1515`)
```cpp
struct ggml_cuda_mm_fusion_args_host {
    const ggml_tensor * x_bias = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * gate_bias = nullptr;
    ggml_glu_op glu_op;
    // ADD:
    const ggml_tensor * lora_a = nullptr;    // [rank, in_dim]
    const ggml_tensor * lora_b = nullptr;    // [out_dim, rank]
    float lora_scale = 0.0f;                 // 0 = no LoRA
};
```

#### b) Add LoRA fusion pattern to `ggml_cuda_try_fuse()` (`ggml-cuda.cu:3838`)
Detect the pattern: `GGML_OP_MUL_MAT` → `GGML_OP_ADD` where the add source is `GGML_OP_SCALE` → `GGML_OP_MUL_MAT` → `GGML_OP_MUL_MAT` (the lora_a @ x → lora_b @ intermediate chain).

#### c) Fuse LoRA into `dequant_mul_mat_vec` kernel (`mmvq.cu`)
For single-token decode (vector case), the kernel already dequantizes blocks and computes dot products. After computing the base IQ2_XXS result, add:
```
lora_hidden = dequant(lora_a) @ x   // small [rank x 1] intermediate
output += scale * lora_b @ lora_hidden
```

The lora_a and lora_b weights are tiny (rank × d) and fit in __constant__ memory or can be pre-loaded once.

#### d) Fuse LoRA into `dequant_mul_mat` kernel (`mmq.cu`)
For larger batch sizes (prompt processing), same pattern but with the block-tiled matmul.

#### e) Fusion for cuBLAS path
For the cuBLAS fallback path, the LoRA computation could be done as separate cublasSgemv/cublasSgemm calls, but still within the same kernel dispatch to avoid additional launch overhead.

**Key implementation details:**
- Rank is typically 8-32, so `lora_a @ x` produces a tiny [rank × 1] vector
- The lora_b @ hidden multiplication is [out_dim × rank] @ [rank × 1], which is bandwidth-bound but tiny
- LoRA factors should be in F16 to match typical adapter formats
- For fused QKV tensors, the LoRA correction must be applied to the correct slice of the output

### 2.2 CPU Backend: Optimized LoRA Matmul

**Priority: MEDIUM** | **Files:** `ggml/src/ggml-cpu/ggml-cpu.c`, architecture-specific files in `ggml/src/ggml-cpu/arch/`

**Problem:** The CPU backend has no special LoRA handling, but for a 1GB adapter used with CPU inference, the extra matmuls add noticeable overhead during single-token decode since each is a separate thread-pool dispatch.

**Required changes:**
- Add a dedicated `GGML_OP_LORA` (or fused mul_mat + lora) op to avoid per-op thread pool overhead
- The CPU path is less critical than CUDA since CPU inference is already slow; the primary concern is avoiding extra thread synchronization points

### 2.3 Graph Construction: NVFP4 Guard Extension

**Priority: MEDIUM** | **Files:** `src/llama-graph.cpp:1204-1244, 1282-1303`

**Problem:** The current NVFP4 + LoRA restriction is only checked in `build_ffn()` for gate/up/down projections. Q/K/V/O projections in `build_qkv()` have no check.

**Required changes:**
- Add NVFP4 assertions to `build_lora_mm()` or `build_qkv()` for Q/K/V/O projections
- This is a safety measure; IQ2_XXS is not affected, but prevents future bugs

### 2.4 Memory Tracking

**Priority: MEDIUM** | **Files:** `src/llama-model.cpp:1667-1683`

**Problem:** Adapter VRAM is not included in `llama_model::memory_breakdown()`, so a 1GB adapter silently consumes VRAM.

**Required changes:**
- Add adapter buffer sizes to the memory breakdown output
- Optionally add a warning if total (model + adapters) exceeds a configurable threshold

### 2.5 Thread Safety

**Priority: LOW** | **Files:** `src/llama-context.cpp:1245-1261`, `src/llama-context.h:280-281`

**Problem:** `llama_set_adapters_lora()` replaces the `loras` unique_ptr without synchronization. If called during inference, this causes a data race.

**Required changes:**
- Add a mutex around adapter swaps, or
- Document that adapter changes must happen between inference calls (practical solution for now)

### 2.6 Adapter Buffer Type Handling

**Priority: LOW** | **Files:** `src/llama-adapter.cpp:337-350`

**Problem:** IQ2_XXS tensors may use repacking buffer types. Currently the code falls back to CPU for extra BUFTs, which would cause GPU↔CPU transfers for LoRA computation.

**Required changes/verification:**
- Verify whether IQ2_XXS tensors actually use repacking BUFTs
- If they do, the CUDA fusion approach (Section 2.1) must handle CPU-resident lora_a/lora_b by pre-copying them to GPU once at init time

---

## 3. Gotchas and Edge Cases

### 3.1 Performance Gotchas

| Issue | Impact | Mitigation |
|---|---|---|
| **448+ extra kernel launches per token** | 13-40ms added latency for 32-96 layer models | Fused CUDA kernel (Section 2.1) |
| **Single-token decode overhead** | LoRA adds ~0.26% FLOPs but 10-100x kernel launch overhead | Fusion eliminates launch overhead |
| **LoRA A/B are tiny but on wrong device** | Stalls the GPU pipeline | Pre-copy to GPU at init |
| **Multi-adapter overhead is linear** | N adapters = N× kernel launches per weight | Fuse multiple adapters into single kernel |
| **Prompt processing (large batch)** | Separate cuBLAS calls for base and LoRA matmuls | Batch or fuse into single cuBLAS call |
| **Fused QKV tensor matching** | LoRA for fused QKV needs single adapter for merged tensor, not separate Q/K/V adapters | Document in adapter training guide |

### 3.2 Correctness Gotchas

| Issue | Impact | Mitigation |
|---|---|---|
| **NVFP4 + LoRA silent incorrectness** | Scale applied after LoRA residual gives wrong results | Already guarded with GGML_ASSERT; extend to QKV (Section 2.3) |
| **Tensor name mismatches** | Adapter won't match if tensor naming differs from expected | Verify against target quantization flow |
| **Token embedding LoRA transposition** | Uses different tensor layout (`get_rows` vs `mul_mat`) | Existing code handles this correctly |
| **Shape validation against quant blocks** | No check that LoRA rank is compatible with quant block boundaries | Validation not needed for correct operation; matmul handles this internally |
| **Norm adapter weights silently skipped** | `_norm.weight` tensors are ignored | Norm correction is tiny; not needed for 1GB budget |
| **Multiple adapters targeting same weight** | All applied, order is adapter-map iteration order (non-deterministic) | Single error-correction adapter avoids this issue |

### 3.3 Operational Gotchas

| Issue | Impact | Mitigation |
|---|---|---|
| **Adapter state not saved in checkpoints** | Restart loses adapter config | Restore adapter separately |
| **Model destructor deletes adapters** | Dangling pointers if context outlives model | Manage adapter lifetime explicitly at application level |
| **Server cache clearing on LoRA change** | Prompt re-encoding needed | Error adapter is static; loaded once at startup |
| **No adapter count/size limits** | Can exhaust VRAM with too many/large adapters | Validate at load time against available VRAM |

---

## 4. Implementation Priority Phases

### Phase 1: Verification (No Code Changes)
1. Create a small rank-8 IQ2_XXS error-correction LoRA GGUF file using existing `convert_lora_to_gguf.py`
2. Test with `llama-cli -m base-iq2_xxs.gguf --lora error_lora.gguf` on CPU
3. Profile the existing path to measure actual overhead
4. Verify IQ2_XXS tensor buffer types (repacking or standard)

### Phase 2: CUDA Fused Kernel (Critical Path)
1. Extend `ggml_cuda_mm_fusion_args_host` with LoRA fields
2. Add fusion pattern detection in `ggml_cuda_try_fuse()` 
3. Implement fused dequant+matmul+lora in `mmvq.cu` (vector case, single-token decode)
4. Implement fused dequant+matmul+lora in `mmq.cu` (batch case, prompt processing)
5. Handle cuBLAS path separately (lower priority, larger batches amortize launch overhead)

### Phase 3: CPU Optimization (Secondary)
1. Add a dedicated `GGML_OP_LORA` or fused matmul+add op for CPU
2. Thread-pool-aware implementation to reduce synchronization points

### Phase 4: Polish
1. Memory breakdown reporting
2. NVFP4 guard extension
3. Documentation and adapter generation tools

---

## 5. Adapter Generation Plan (Offline)

The adapter GGUF file is generated offline, not in llama.cpp:

### Method 1: Direct Residual SVD (Simplest)
```python
for each adapted tensor:
    R = W_bf16 - dequant(W_iq2_xxs)
    U, S, Vt = randomized_svd(R, rank=r)
    lora_b = U @ sqrt(S)     # [out_dim, r]
    lora_a = sqrt(S) @ Vt    # [r, in_dim]
    # Save as .lora_a / .lora_b in GGUF
```

### Method 2: Activation-Weighted (Better Quality)
```python
# Using calibration data activations X
for each adapted tensor:
    target = (W_bf16 - dequant(W_iq2_xxs)) @ X
    # Fit lora_b @ lora_a to approximate target via alternating least squares
    # Or gradient descent on ||lora_b @ lora_a @ X - target||²
```

### Method 3: Teacher Distillation (Best Quality)
```python
# Freeze IQ2_XXS model, attach LoRA, train with KL loss against BF16 teacher
for batch in calibration_data:
    logits_bf16 = teacher_model(batch)
    logits_iq2_lora = iq2_model + lora(batch)
    loss = KL_divergence(logits_bf16, logits_iq2_lora)
    # Backprop only through LoRA weights
```

Recommended approach: Start with Method 1 for quick validation, then move to Method 2 or 3 for production quality.

---

## 6. Budget and Scaling

For an illustrative 96-layer Llama-like model (d=12288, ffn=32768), 7 adapted tensor types per layer (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj):

### Unquantized (BF16/F16) LoRA Adapter Sizes

| Rank | Total Adapter Size |
|------|-------------------|
| 8 | ~359 MB |
| 16 | ~717 MB |
| 24 | ~1.08 GB |
| 32 | ~1.43 GB |
| 16 (down,o,v only) | ~256 MB |

### Q8_0 Quantized LoRA Adapter Sizes

Q8_0 stores each block of 32 floats as 34 bytes (2B fp16 scale + 32B int8), giving ~1.75x compression vs F16.

| Rank | Total Adapter Size |
|------|-------------------|
| 8 | ~200 MB |
| 16 | ~400 MB |
| 32 | ~800 MB |
| 64 | ~1.60 GB |
| 32 (down,o,v only) | ~286 MB |

### Q8_0 Block Alignment Constraint

Q8_0 requires the last tensor dimension to be divisible by 32 (`block_size`). For lora_a (rank, in_dim) and lora_b (out_dim, rank):
- lora_a: last dim = in_dim (e.g., 12288). 12288 % 32 = 0. Always ok for standard d_model values.
- lora_b: last dim = rank. rank % 32 = 0. So rank must be a multiple of 32 (e.g., 8 or 16 would ONLY get Q8_0 on lora_a, not lora_b).

**Recommended for 1GB budget: rank-32 Q8_0 full adapter (~800 MB)** or rank-64 selective.

### Per-Tensor Storage (Q8_0, rank=32, d=12288)

| Tensor | Shape | BF16 Bytes | Q8_0 Bytes | Savings |
|--------|-------|-----------|-----------|---------|
| lora_a (attn_q) | (32, 12288) | 786,432 | 417,792 | 47% |
| lora_b (attn_q) | (12288, 32) | 786,432 | 417,792 | 47% |
| lora_a (attn_k) | (32, 12288) | 786,432 | 417,792 | 47% |
| lora_b (attn_k) | (12288, 32) | 786,432 | 417,792 | 47% |
| lora_a (ffn_up) | (32, 32768) | 2,097,152 | 1,114,112 | 47% |
| lora_b (ffn_up) | (32768, 32) | 2,097,152 | 1,114,112 | 47% |
| lora_a (ffn_down) | (32, 32768) | 2,097,152 | 1,114,112 | 47% |
| lora_b (ffn_down) | (32768, 32) | 2,097,152 | 1,114,112 | 47% |

Per tensor pair: ~816 KiB (Q8_0) vs ~1.5 MiB (F16). The Q8_0 LoRA adapter saves ~47% storage and compute bandwidth vs F16 LoRA.

### Generation Tool

```bash
python tools/generate_quant_error_lora.py \
    --bf16-model model-f16.gguf \
    --iq2-model model-iq2_xxs.gguf \
    --outfile error_lora-q8_0-r32.gguf \
    --rank 32
```

---

## 7. Key Files Reference

| File | Purpose |
|------|---------|
| `src/llama-adapter.cpp` | Adapter loading, buffer allocation, tensor pairing |
| `src/llama-adapter.h` | `llama_adapter_lora` and `llama_adapter_lora_weight` structs |
| `src/llama-graph.cpp:1085-1152` | `build_lora_mm()` and `build_lora_mm_id()` graph construction |
| `src/llama-graph.cpp:1832-1876` | `build_inp_embd()` token embedding LoRA |
| `src/llama-context.cpp:1245-1261` | `set_adapters_lora()` adapter management |
| `src/llama-context.cpp:2323-2331` | `graph_max_nodes()` node budget |
| `ggml/src/ggml-cuda/ggml-cuda.cu:3838` | `ggml_cuda_try_fuse()` fusion pattern detection |
| `ggml/src/ggml-cuda/common.cuh:1504` | `ggml_cuda_mm_fusion_args_host` fusion arguments |
| `ggml/src/ggml-cuda/mmvq.cu` | Quantized vector matmul (single-token decode) |
| `ggml/src/ggml-cuda/mmq.cu` | Quantized batch matmul (prompt processing) |
| `ggml/src/ggml-cpu/ggml-cpu.c:1702` | CPU compute forward dispatch |
| `ggml/src/ggml-backend.cpp:878` | Scheduler backend assignment |
| `convert_lora_to_gguf.py` | GGUF adapter file generation (HF PEFT -> GGUF) |
| `tools/generate_quant_error_lora.py` | **NEW**: Generate Q8_0 LoRA adapter from IQ2_XXS residual via SVD |
| `include/llama.h:648-687` | Public LoRA API |

---

## 8. Success Criteria

1. **No regression:** `llama-cli -m base.gguf --lora error_lora.gguf` works identically to current LoRA path for non-IQ2_XXS models
2. **CUDA fusion:** Single-token decode with IQ2_XXS + rank-32 Q8_0 LoRA has <5% overhead vs IQ2_XXS alone (down from ~10-40ms extra)
3. **Quality improvement:** IQ2_XXS + rank-32 LoRA perplexity is measurably closer to BF16 than IQ2_XXS alone
4. **Memory budget:** Q8_0 rank-32 adapter fits in 1GB VRAM (target ~800 MB) including overhead
5. **Stability:** No crashes, no memory leaks with adapter loaded/unloaded cycles
6. **End-to-end generation:** `tools/generate_quant_error_lora.py` successfully produces a valid GGUF adapter from two GGUF model files
