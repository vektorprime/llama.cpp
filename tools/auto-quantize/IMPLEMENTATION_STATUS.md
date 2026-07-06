# Implementation Status — IQ4_XS Quantization Research

**Date**: 2026-07-06
**Branch**: `auto_research_iq4xs_llama`
**Model**: Qwen3.6-27B (BF16 GGUF at `/home/user/llm/models/Qwen3.6-27B/`)

## Overview

New research project targeting IQ4_XS quantization (4.25 bpw, 4-bit non-linear).
Transferred findings from the IQ2_XXS research (best KL 0.662001, 68 experiments).

## Key Source Locations

| What | Location |
|------|----------|
| IQ4_XS block struct | `ggml/src/ggml-common.h:444` — `block_iq4_xs` |
| IQ4_XS quantize entry | `ggml/src/ggml-quants.c:5731` — `quantize_iq4_xs()` |
| Inner quantize kernel | `quantize_row_iq4_nl_impl()` — sub-block 32, 4-bit, kvalues_iq4nl |
| Non-uniform codebook | `kvalues_iq4nl` — 16-entry fixed table |
| Dequantize | `ggml/src/ggml-quants.c:2793` — `dequantize_row_iq4_xs()` |

## Current Status

- No experiments run yet.
- First experiment: establish baseline KL by quantizing with stock IQ4_XS.

## Model Paths

| Resource | Path |
|---|---|
| BF16 model | `/llmdata/Qwen3.6-27B/Qwen_Qwen3.6-27B-bf16-00001-of-00002.gguf` |
| Reference logits | `/llmdata/Qwen3.6-27B/Qwen_Qwen3.6-27B-Q8.logits` |
| Imatrix | `/llmdata/Qwen3.6-27B/imatrix_bartowski_q3.6-27b.gguf` |
| Eval data | `/llmdata/Qwen3.6-27B/wiki.test.raw` |
| Calibration data | `/llmdata/Qwen3.6-27B/bartowski_calibration_data_v5.txt` |
| Quantized output (experiments) | `/llmdata/Qwen3.6-27B/Qwen_Qwen3.6-27B-IQ4_XS-exp.gguf` |

## GPU Info

| Device | Model | VRAM | CC |
|--------|-------|------|-----|
| 0 | RTX 5090 | 32110 MB | 12.0 | **USE FOR EVAL** (works for this model) |
| 1 | RTX 3080 | 20054 MB | 8.6 | Available |
| 2 | RTX 3080 | 20054 MB | 8.6 | Available |
| 3 | RTX 3050 | 5806 MB | 8.6 | Available |

**Use device 0 (RTX 5090)** for eval: `CUDA_VISIBLE_DEVICES=0`

Device 0 (RTX 5090, CC 12.0) lacks kernel images in the CUDA build and
produces "no kernel image is available for execution on the device" errors.
