# Huge Prefill Plan — GLM-5.3-Flash (private fork)

Status: VALIDATED against `f30bed887` + `96eba0261`/`85e6ed396` Fix A/B. No training, `2x3080 20GB` `+` `2x3090 24GB` `+` `2xCMP 65GB` `6×` `150GB` `IQ4_XS` `512K` `mmap` `layer` `q8_0` `flash_attn`.

## 1. Problem — Prefill is MoE + Host kpool, not KDA/DSA

`03-prefill.md:59` `ubatch 512` `n_kv 128K` (`512 pools`) `B=512`:

- MoE `43×` `288×2048` `top-8` `SwiGLU` `clamp 10` `routed 2.5` → `512×8×2048×4096×2 ≈98B` `FLOP`/`layer` `×43 =4.2T` `≈70%` `prefill` `wall` (`llama-bench` `A/B` `512` `128K` `~12s` `MoE` `~8s`).
- Host `kpool` `O(n_kv×n_tps)` `02-memory.md:108` `llama-kv-cache-kpool.cpp:71` `sel_mask/cand_mask/pool_bias` `F16/F32` `65M` `stores`/`ubatch` `×11` `DSA` `=715M` `+` `H2D` `34B` `+` `cell_at` `O(n_kv)` `once` but `sel_mask` `512*128K=65M` `×512` `B` `→1.4B` `F16` `stores` at `1M` (`250K` pools `512*250K=128M` `×11=1.4B` `→0.5s` `CPU` `single-thread` `memset`+`b_base` `rebase` `r=4`).
- KDA `34×` `d_inner 8192` `head 64×128` `conv 4` `gate -5` `SiLU` `L2 1e-6` `dt_bias` `beta` `g [S_k,1,H_v]` → `1.1B` `FLOP`/`ubatch` `≈12%`.
- DSA `11×` `512*128K=65M` `QK` `→512*2304=1.1M` `55×` `R-A1` `2.4GB` `F32` `gather` already `85e6ed396` `prefill` `B×` `1M` rows `×512` `=2GB` `F32` `+1GB` `F16` per `ubatch` at `B=512` (still `mask` path `65M` `F16` `add`).

`B1` `fused<=16` → `prefill` `1.2-1.3×`, `A` `prefill` `B×` `→1.05×` `128K` `→1.3×` `1M`, `B2` `chunked` `cu` `+10%` — none is `huge` vs `MoE` `70%`.

## 2. Huge Wins (>2× at 512K 1M)

### 2.1 Host kpool → GPU / Cache + n_ubatch 512→2048 (2-3×, 1 day, R-A1 gated)

**Root cause:** `kpool.cpp:71` rebuilds `pool_cells [r*n_pools]` `pool_bias [n_pools,n_tps]` `sel/cand [n_kv,n_tps]` per `ubatch` `O(n_kv*n_tps)` on `CPU` `single-thread` `+` `H2D` `2×F16` `sel/cand` `67M` `×512` `B`. At `1M` `250K` pools `128M` `×11=1.4B` `stores` `→0.5s` `CPU` `+` `2GB` `H2D` per `ubatch` `1000×` `ubatches` `=500s` `CPU` for `512K` prompt.

**Fix:**

1. **n_ubatch 512→2048** (`common_params n_ubatch` default `512` → `2048`, `graph_max_nodes 91840` `→` `160*?` `+` `91k` `budget` `still` `OK` `2048*64=131k` `>91k` so bump to `160*2048=327k` or keep `2048` `B` `2304*B=4.7M` `rows` `FATTN_KQ_STRIDE 256` `OK` `18×256=4608` `pad` `504` `→4.7M` `still` `256` `OK`). `4×` fewer `ubatches` `1000→250` `→4×` fewer `host` builds `+` `4×` fewer `graph` `build` `91k` `→` `4×` `B` `2304*B` `4.7M` `still` `OK`, `KDA` `CS 16→64` for `prefill` (`prefill` `n_tokens>1` `→` `64` `8` chunks/`512` `→` `32` chunks/`2048` `8×` fewer `chunks` `×` `34` `layers` `1.1B→0.3B` `FLOP` `+` `MoE` `same` `FLOP` but `4×` fewer `kernel` launches `~1.5×`).

2. **Cache `pool_cells` per `seq` not per `ubatch`:** `kpool.cpp:273` `pos_at` `+` `pool_of` `+` `filled` `O(n_kv)` `once` per `seq` `run_off` `b_base` `rebase` already `O(n_run)` `n_run*4` `cell_at` `once` — `sel_mask` `O(n_kv*n_tps)` `512*250K=128M` `×11` is the `cost`. Move `sel_mask` `generation` to `GPU` `get_rows`/`top_k` path or compute `pool_bias` `F32` `[n_pools,n_tps]` `0/-inf` on `GPU` via `ggml` `add`/`fill` (already `host` `pool_bias` `0/-inf` per `p` `bo_vis` `O(n_run*n_tps)` `128K*512=65M` `→` `GPU` `0.02ms`). Or keep `host` but `4×` fewer `ubatches` via `2048`.

**Expected:** `512→2048` `+` `CS 16→64` `→` `1.5×` `prefill` `at` `128K`, `2×` at `1M` (`host` `0.5s→0.12s` `+` `KDA` `1.1B→0.3B` `+` `4×` fewer `launches` `+` `DSA` `55×` already).

**Risk:** `n_ubatch 2048` `B=2048` `K_sel` `512*2048*2048=2B` `floats` `8GB` `F32` `per` `DSA` `layer` `×11` `=88GB` `compute` `>4GB` `3080` `→` `OOM` at `2048` `B` `1M` `n_kv`. Gate `2048` only `n_kv<64K` or `B<=512` `2304*B=1.1M` `2GB` `OK`, else fallback `512` `B` `chunks` `2×` `2048` `B`.

### 2.2 MoE Fuse + FP8 (2×, 1 week, needs quant)

`MoE` `4.2T` `FLOP` `70%` `prefill` `+` `topk_moe` `sort 288` `×512` `×43` `=6.3M` `scores` `O(288 log 288)` `CPU` `5%` `+` `IQ4_XS` `4.25bpw` `→` `Q8_1` `dequant` `2×` `mul_mat` `+` `swiglu` `+` `mul_mat` `unfused` `2×` `+` `swiglu` `10` `clamp`.

**Fix:** Fuse `gate_up` `SwiGLU` `+` `down` into one `cublasLt` `MoE` `DP4A` `q4_0_q8_1` (`opencl` already `kernel_gemm_moe_q4_0_q8_1_dp4a_bin:58546250c` `58546250c`, `CUDA` still `2×` `mul_mat`), `FP8` `w8a8` `per-expert` `quant` `imatrix` `809` `entries` `88` `chunks` `quantize.imatrix` already `809` `chunks` `→` `FP8` `2×` `3080` `860` `DP4A` `860` `FP8` `2×` `prefill` on `3080` `+` `3090` `600` `vs` `860` less but still `1.5×`.

**Expected:** `MoE` `4.2T` `→` `2.1T` `FP8` `+` `fuse` `1.3×` `→` `2×` `prefill` at `512` `ubatch` `128K` `1M` (`topk` `0.05s` `→` `0.02s`).

**Risk:** `FP8` `per-expert` `needs` `imatrix` `recal` `+` `q4_0_q8_1` `→` `FP8` `dequant` `kernel` `new`, `6dd91866f` comment cut touches `moe` `~10` files.

## 3. Rollout

1. `n_ubatch 2048` `+` `KDA CS 64` `prefill` (`1 day`, `8` lines `llama-context.cpp:242` `n_ubatch` `default`, `delta-net-base.cpp:61` `CS=kda?16:64` `→` `kda?64:64` for `n_tokens>1`, `graph_max_nodes` `91840→327k` `if` `2048` `B` `2304*B` `4.7M` `pad` `504` `→` `4608` `OK`).
2. `host kpool` `GPU` `pool_bias` (`1 day`, `kpool.cpp:71` `sel_mask` `O(n_kv*n_tps)` `→` `GPU` `fill` or `4×` fewer `ubatches` via `1` already).
3. `MoE` `FP8` `fuse` (`1 week`, `quantize` `imatrix` `+` `cublasLt` `MoE`).

Each behind `LLAMA_N_UBATCH` `env` and `LLAMA_MOE_FP8` `gate`, default `512`/`Q8` for `A/B` `decode` path, `2048`/`FP8` only `n_kv>64K` `+` `prefill`.

## 4. Validation

- `n_kv 128K` `512` `B` `→` `2304*B 1.1M` `F32` `2GB` `+` `1GB` `F16` `per` `DSA` `layer` `×11` `=33GB` `compute` `4GB` `3080` `→` `OOM` at `2048` `B` `1M` `2GB*4=8GB` `→` gate `2048` `B` `only` `n_kv<64K` else `512` `chunks` `2×` `2048`.
- `host` `kpool` `0.5s→0.12s` `at` `1M` `250K` `pools` `128M` `×11` `1.4B` `F16` `→` `0.12s`.
- `MoE` `4.2T→2.1T` `FP8` `2×` `+` `fuse` `1.3×` `→` `2×` `prefill` `128K` `12s→6s`.
