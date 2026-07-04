#!/usr/bin/env python3
"""
Generate a Q8_0 quantized LoRA adapter that corrects IQ2_XXS quantization errors.

Computes the residual R = W_bf16 - dequant(W_iq2_xxs) for each target tensor,
performs randomized SVD to rank r, and writes a GGUF LoRA adapter with Q8_0
quantized lora_a and lora_b tensors.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

if "NO_LOCAL_GGUF" not in __import__("os").environ:
    sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES, Keys

logger = logging.getLogger("generate-quant-error-lora")


def randomized_svd(
    R: np.ndarray,
    rank: int,
    n_oversamples: int = 5,
    n_iter: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = R.shape
    p = min(rank + n_oversamples, min(m, n))

    Omega = np.random.randn(n, p).astype(R.dtype)
    Y = R @ Omega

    for _ in range(n_iter):
        Y = R @ (R.T @ Y)

    Q, _ = np.linalg.qr(Y)
    B = Q.T @ R

    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)

    U = Q @ Ub
    return U[:, :rank], S[:rank], Vt[:rank, :]


def compute_lora_from_residual(
    R: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("  randomized SVD...")
    U, S, Vt = randomized_svd(R, rank)

    sqrt_S = np.sqrt(S.astype(np.float64)).astype(np.float32)
    lora_b = U * sqrt_S[np.newaxis, :]
    lora_a = sqrt_S[:, np.newaxis] * Vt

    return lora_b, lora_a


def read_tensor_f32(reader: gguf.GGUFReader, tensor: gguf.ReaderTensor) -> np.ndarray:
    raw = reader.data[tensor.data_offset : tensor.data_offset + tensor.n_bytes]

    ttype = tensor.tensor_type
    ggml_shape = tensor.shape.tolist()
    # ggml ne[0] is fastest dim; numpy row-major needs (ne[1], ne[0], ...)
    np_shape = tuple(reversed(ggml_shape))

    if ttype == GGMLQuantizationType.F32:
        arr = raw.view(np.float32).reshape(np_shape).copy()
    elif ttype == GGMLQuantizationType.F16:
        arr = raw.view(np.float16).astype(np.float32).reshape(np_shape).copy()
    elif ttype == GGMLQuantizationType.BF16:
        arr = raw.view(np.uint16).astype(np.uint32)
        arr = (arr << 16).view(np.float32).reshape(np_shape).copy()
    else:
        n_elems = int(np.prod(ggml_shape))
        block_size, type_size = gguf.constants.GGML_QUANT_SIZES[ttype]
        n_blocks = tensor.n_bytes // type_size
        blocks = raw.reshape((n_blocks, type_size))

        dq_cls = gguf.quants._type_traits.get(ttype)
        if dq_cls is None:
            raise ValueError(f"No dequant handler for {ttype.name}")
        dq_cls.init_grid()
        vals = dq_cls.dequantize_blocks(blocks)
        arr = vals.ravel()[:n_elems].reshape(np_shape)

    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Q8_0 LoRA adapter from BF16-to-IQ2_XXS quantization residual"
    )
    p.add_argument("--bf16-model", type=Path, required=True,
                   help="Path to BF16 (or F16) GGUF base model")
    p.add_argument("--iq2-model", type=Path, required=True,
                   help="Path to IQ2_XXS GGUF model (same architecture)")
    p.add_argument("--outfile", type=Path, required=True,
                   help="Output LoRA adapter GGUF file")
    p.add_argument("--rank", type=int, default=32,
                   help="LoRA rank (default: 32)")
    p.add_argument("--alpha", type=float, default=None,
                   help="LoRA alpha (default: rank)")
    p.add_argument("--tensors", type=str, default="all",
                   help="Comma-separated tensor name suffixes to adapt (default: all), "
                        "e.g. 'attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down'")
    p.add_argument("--exclude", type=str, default="",
                   help="Comma-separated tensor name suffixes to exclude")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p.parse_args()


def filter_tensors(
    tensors: Sequence[gguf.ReaderTensor],
    include: set[str],
    exclude: set[str],
) -> list[gguf.ReaderTensor]:
    result = []
    for t in tensors:
        name = t.name
        if any(name.endswith(f".{s}.weight") for s in exclude):
            continue
        if include == {"all"}:
            result.append(t)
        elif any(name.endswith(f".{s}.weight") for s in include):
            result.append(t)
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    rank = args.rank
    alpha = args.alpha if args.alpha is not None else float(rank)

    include = set(s.strip() for s in args.tensors.split(",") if s.strip())
    exclude = set(s.strip() for s in args.exclude.split(",") if s.strip())

    logger.info("Loading BF16 model: %s", args.bf16_model)
    reader_bf16 = gguf.GGUFReader(str(args.bf16_model))

    logger.info("Loading IQ2_XXS model: %s", args.iq2_model)
    reader_iq2 = gguf.GGUFReader(str(args.iq2_model))

    arch_field = reader_bf16.get_field(Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError("BF16 model missing general.architecture key")
    arch = str(arch_field.parts[-1].tobytes(), encoding="utf-8")

    bf16_tensors = {t.name: t for t in reader_bf16.tensors}
    iq2_tensors = {t.name: t for t in reader_iq2.tensors}

    common_names = sorted(set(bf16_tensors.keys()) & set(iq2_tensors.keys()))
    candidates = [bf16_tensors[n] for n in common_names]
    candidates = filter_tensors(candidates, include, exclude)

    logger.info("Will adapt %d tensors:", len(candidates))
    for t in candidates:
        logger.info("  %s  shape=%s  bf16_type=%s  iq2_type=%s",
                    t.name, tuple(t.shape.tolist()),
                    t.tensor_type.name, iq2_tensors[t.name].tensor_type.name)

    writer = gguf.GGUFWriter(str(args.outfile), arch)
    writer.add_architecture()
    writer.add_string(Keys.General.TYPE, "adapter")
    writer.add_string(Keys.Adapter.TYPE, "lora")
    writer.add_float32(Keys.Adapter.LORA_ALPHA, alpha)

    total_start = time.time()

    for bf16_t in candidates:
        name = bf16_t.name

        if name.endswith("token_embd.weight"):
            logger.info("Skipping %s (token embeddings need different LoRA layout)", name)
            continue

        if name.endswith("_norm.weight"):
            logger.info("Skipping %s (1D norm tensor, not suitable for LoRA)", name)
            continue

        iq2_t = iq2_tensors[name]

        logger.info("Processing: %s", name)

        t0 = time.time()
        W_bf16 = read_tensor_f32(reader_bf16, bf16_t)
        W_iq2 = read_tensor_f32(reader_iq2, iq2_t)
        t_load = time.time() - t0
        logger.info("  loaded in %.1fs (bf16 shape=%s, iq2 shape=%s)",
                    t_load, W_bf16.shape, W_iq2.shape)

        if W_bf16.shape != W_iq2.shape:
            logger.warning("  shape mismatch: bf16=%s iq2=%s, skipping", W_bf16.shape, W_iq2.shape)
            continue

        R = W_bf16 - W_iq2

        lora_b, lora_a = compute_lora_from_residual(R, rank)
        logger.info("  lora_b shape=%s  lora_a shape=%s", lora_b.shape, lora_a.shape)

        t0 = time.time()
        lora_a_q8 = gguf.quants.quantize(lora_a, GGMLQuantizationType.Q8_0)
        lora_b_q8 = gguf.quants.quantize(lora_b, GGMLQuantizationType.Q8_0)
        t_q = time.time() - t0
        logger.info("  Q8_0 quantized in %.1fs (a=%d bytes, b=%d bytes)",
                    t_q, lora_a_q8.nbytes, lora_b_q8.nbytes)

        writer.add_tensor(
            name + ".lora_a", lora_a_q8,
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        writer.add_tensor(
            name + ".lora_b", lora_b_q8,
            raw_dtype=GGMLQuantizationType.Q8_0,
        )

    logger.info("Writing adapter to: %s", args.outfile)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    total = time.time() - total_start
    logger.info("Done in %.1fs", total)
    logger.info("Output: %s", args.outfile)


if __name__ == "__main__":
    main()
