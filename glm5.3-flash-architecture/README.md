# GLM-5.3-Flash (glm5next) in llama.cpp

How this fork makes GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, HF arch
`Glm5NextForConditionalGeneration`) run on llama.cpp. The in-tree model arch is
`LLM_ARCH_GLM5NEXT`, short name `glm5next`.

Companion documents in this folder:

| file | contents |
|------|----------|
| [01-components.md](01-components.md) | every model component: mHC, KDA, DSA/MLA, pooled lightning indexer, MoE, NextN, vision, tokenizer |
| [02-memory.md](02-memory.md) | the three KV caches and the host-side pool map that drives sparse selection |
| [03-prefill.md](03-prefill.md) | what happens for a prefill ubatch, step by step |
| [04-decode.md](04-decode.md) | what happens per generated token |
| [05-cuda-cpu.md](05-cuda-cpu.md) | which ops run where, fused kernels, precision rules |

## The model in one paragraph

GLM-5.3-Flash is a ~313B-A17B MoE (the repo's `LLM_TYPE_313B_A17B` type match;
the HF README rounds to 320B-A18B) with a hybrid attention stack: 34 of the 45
trunk layers use **KDA** (Kimi-style linear attention, a gated delta net with
per-key-channel decay) and every fourth layer (3, 7, 11, 15, 19, 23, 27, 31,
35, 39, 43) uses **DSA** (DeepSeek-style sparse attention: nope-only absorbed
MLA over a 512-wide latent, where a **pooled lightning indexer** picks which
2048 keys of the whole cache each query attends to). Residuals are
**mHC** (Manifold-Constrained Hyper-Connections, DeepSeek-V4's 4-stream wide
residual with Sinkhorn-normalized mixing). FFNs are sigmoid-gated MoE with
288 experts, top-8 routing, noaux_tc, clamped SwiGLU. A 46th "NextN" block is a
full DSA layer for speculative/MTP use.

Key config values (from `config.json`, cross-checked against the loader's
asserts):

> **Correction (2026-08-28 validation):** Most values in the table below are *not* asserted by the loader; they are read from HF `config.json` via `conversion/glm5next.py` and GGUF (`hparams.n_embd`, `q_lora_rank` etc.). Loader asserts only invariants: `n_rot()==0` (`glm5next.cpp:58`), `kda_gate_lower_bound<0` (`:70`), `index_topk%kpool==0` (`:78`), `hc_mult>0` (`:91`), `n_layer_nextn<n_layer_all` (`:124`), per-layer `head_count_kv` etc. Exact numbers (4096, 1536/512, 256, 64/128, 32/128, 2048/4, 12288, 288 etc.) are correct for `zai-org/GLM-5.3-Flash` but are data-driven, not hard-coded. Also `vision 24x1024 patch14` is loaded generically from `vision_config` via `conversion/base.py`; only `merge 2` (`clip.cpp:1751`) and `clamp 10` (`swiglu_limit`) are hard-coded/default.

```
hidden_size              4096
num_hidden_layers        45 (+1 NextN)
full_attn (DSA) layers   3,7,11,15,19,23,27,31,35,39,43   (11 layers)
kda_layers               the other 34
q_lora_rank / kv_lora    1536 / 512
qk_nope / v head dim     256 / 256   (rope dim 0: nope-only, no RoPE anywhere)
KDA heads / head dim     64 / 128    (d_inner = 8192, conv kernel 4, gate bound -5.0)
indexer heads / head dim 32 / 128
index_topk / index_kpool 2048 / 4    (512 pools; selection width 2051 positions)
mHC                      4 streams, 20 Sinkhorn iters, eps 1e-6
MoE                      3 dense lead (12288), 288 x 2048 experts, 1 shared,
                         top-8, sigmoid + noaux_tc, routed scaling 2.5, swiglu clamp 10.0
vocab                    154880, glm4 BPE with ignore_merges
vision                   GLM-OCR ViT (24 x 1024, patch 14, merge 2, swiglu clamp 10)
```

## Commits on this branch

28 branch-local commits implement the model (oldest first); the rest of the 35
commits ahead of `origin/master` are unrelated upstream merges this branch sits
on top of.

> **Correction (2026-08-28 validation):** "upstream merges" is inaccurate - `git log --merges origin/master..HEAD` is 0; the 7 extra commits are upstream *commits* (`5e6a37cb1`, `bf942...`, `d013...`, `4d19...`, `fc35...`, `da9b...`, `dac8...`) cherry-picked/merged without merge commits. So 35 = 28 glm5next + 7 upstream commits.

Foundation (load and run a trunk):

| commit | what it adds |
|--------|--------------|
| `a505a7587` | arch registration, hparams, tensor loading, converter, GGUF keys; dense-FFN path |
| `9e1cc2082` | mHC wide residual (reuses DeepSeek-V4's mixer, mean collapse) |
| `20a807707` | KDA linear-attention layers (reuses delta-net base) |
| `1d99a5ac4` | MoE FFN with clamped SwiGLU |
| `f320a7875` | dense DSA attention (absorbed nope-only MLA) |
| `cd69d60e5` | `test-llama-archs` enablement |
| `6a58359c5` | asserts on the indexer selection width invariant |
| `384159234` | comment cleanup |
| `2db8295ee` | **third KV cache** for the indexer + `llama_memory_hybrid` extension |
| `839597c62` | comment cleanup |
| `8c4983640` | **pooled lightning indexer graph**: pool compressor, top-k over pools, `build_attn_sparse` |

> **Correction (2026-08-28 validation):** Row above hash `f320a7875` is a typo, should be `f320a2875` (`git log --oneline` shows `f320a2875 llama: add glm5next dense DSA...`; `f320a7875` does not exist). Table also omits `7f2560e98 llama: save the indexer kpool and hyper-connection keys` which is why the table lists 27 rows but the text claims 28 branch-local commits. Count is `28 = 27 listed + 7f2560e98`; with 7 upstream commits the total ahead of `origin/master` is 35.

Fixes, vision, and packaging:

| commit | what it adds |
|--------|--------------|
| `fe95953cc` | mtmd: skip patch-emb norm when `norm_embd` absent (GLM-OCR has none) |
| `582fe816e` | converter refuses to write a zero-tensor GGUF |
| `9901ab4bd` | vision tower (GLM-OCR ViT + clamped-SwiGLU projector type) |
| `41cd9235f` | glm5next image preprocessor (smart resize) |
| `02fa4a43a` | bicubic resampling to match the reference |
| `81e3e6716` | `ignore_merges` for the glm4 pre-tokenizer |
| `2d9570d2c` | comment cleanup |
| `6c59f9b8e` | rebase conflict-repair (KV keys, CMake, test) |

> **Correction (2026-08-28 validation):** `2d9570d2c` message is `llama: shorten comment` (not generic "comment cleanup"); `cadbe97b7` below is `glm5next: fix E301 lint in gguf_writer` (shorthand "gguf-py lint" is okay but not exact).
| `eab9ee932` | `n_embd_out` stays `n_embd` (was inheriting deepseek4's 4x-wide bug) |
| `869e87879` | quantizer: keep precision-sensitive tensors at source precision |
| `282ef610a` | vision preprocessor tests |
| `cadbe97b7` | gguf-py lint |
| `204fa7003` | per-sequence pool map so a **unified** KV cache works |
| `ef531f827` | memory test for per-sequence pool runs |
| `e88c92d02` | selection masks stored in f16 |
| `2e0e57f10` | pool scores computed with the **fused** `ggml_lightning_indexer` op |

## Where the code lives

```
src/models/glm5next.cpp            graph: mHC loop, KDA, DSA, indexer, MoE  (873 lines, the core)
src/models/models.h                llama_model_glm5next struct (derives dsv4 graph)
src/models/deepseek4.cpp           mHC pre/post/Sinkhorn (inherited), hc_mean
src/models/delta-net-base.cpp      KDA recurrence (chunked/AR/fused dispatch)
src/llama-kv-cache-kpool.{h,cpp}   host-side pool map: pool_cells, biases, masks
src/llama-memory-hybrid.{h,cpp}    attn + recurrent + indexer caches in one memory
src/llama-graph.{h,cpp}            build_inp_kpool, build_attn_sparse, fused-op registry
src/llama-quant.cpp                full-precision pins for glm5next tensors
src/llama-vocab.cpp                ignore_merges for the glm4 tokenizer
src/llama-context.cpp              fused_lid default + probe, graph node budget
conversion/glm5next.py             HF -> GGUF converter (text + vision)
gguf-py/gguf/constants.py          GLM5NEXT arch, KV keys, tensor names
tools/mtmd/models/glm5next-vision.cpp, tools/mtmd/clip.cpp, mtmd-image.cpp
                                   vision tower + preprocessor
tests/test-glm5next-memory.cpp     memory/kpool behavior tests
tests/test-llama-archs.cpp         arch graph construction test
tests/test-mtmd-impl.cpp           preprocessor tests
```

Notably, the branch adds **no new ggml kernels**: every computation is built
from ops that already exist in-tree (some of them themselves newly fused for
DeepSeek-V4, e.g. `ggml_lightning_indexer` and the dsv4_hc ops), and the CUDA
and CPU backends already carry implementations for all of them. The novelty of
this branch is the graph, the third cache, and the host-side map - not new
hardware code. See [05-cuda-cpu.md](05-cuda-cpu.md) for exactly which op each
piece lowers to.
