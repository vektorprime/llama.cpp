#!/usr/bin/env python3
"""
auto_quantize.py — orchestration script for llama.cpp quantization research.

Runs the experiment loop: build → quantize → evaluate KL divergence → log results.
Follows the autoresearch pattern (karpathy/autoresearch, vektorprime/AutoQuant).

Target type: IQ2_XXS (with per-layer codebook improvements).

Usage:
    # Full experiment with defaults:
    python auto_quantize.py run --description "my experiment"

    # IQ type with imatrix + tensor type file:
    python auto_quantize.py run \\
        --type IQ2_XXS \\
        --description "per-layer codebook" \\
        --extra-quantize-args --my-flag=0.7

    # Pure quant (Q2_K etc., no imatrix needed):
    python auto_quantize.py run \\
        --type Q2_K --pure \\
        --description "Q2_K experiment"

    # Eval only:
    python auto_quantize.py eval --model /tmp/quantized-model.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = REPO_ROOT / "build"
RESULTS_TSV = Path(__file__).resolve().parent / "results.tsv"

DEFAULT_INPUT  = "/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.gguf"
DEFAULT_OUTPUT = "/tmp/quantized-model.gguf"
DEFAULT_EVAL_DATA = "/home/user/llm/wikitext-2-raw/wiki.test.raw"
DEFAULT_LOGITS_BASE = "/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits"
DEFAULT_IMATRIX = "/home/user/llm/models/Qwen3.5-2B/imatrix_unsloth.gguf"
DEFAULT_TENSOR_TYPES = "/tmp/tensor_types.txt"

QUANTIZE_BIN = BUILD_DIR / "bin" / "llama-quantize"
PERPLEXITY_BIN = BUILD_DIR / "bin" / "llama-perplexity"

HEADER = [
    "timestamp", "exp_id", "code_sha", "parent_sha",
    "description", "status", "kl_divergence",
    "base_type", "model_size_mb", "quantize_time_s", "eval_time_s",
    "tokens_per_sec", "ppl", "same_top_p",
]


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def safe_git(*args: str) -> str:
    return subprocess.run(
        ["git"] + list(args), capture_output=True, text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def read_best_kl() -> float:
    """Return the best (lowest) KL divergence recorded so far."""
    if not RESULTS_TSV.exists():
        return float("inf")
    lines = RESULTS_TSV.read_text().strip().split("\n")
    if len(lines) < 2:
        return float("inf")
    best = float("inf")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 7 and parts[5] == "success":
            try:
                kl = float(parts[6])
                if kl < best:
                    best = kl
            except ValueError:
                pass
    return best


def next_exp_id() -> str:
    dt = datetime.now(timezone.utc).strftime("%Y%m%d")
    count = 0
    if RESULTS_TSV.exists():
        count = len(RESULTS_TSV.read_text().strip().split("\n")) - 1
    return f"exp-{dt}-{count:03d}"


# ──────────────────────────────────────────────────────────────────────
# build
# ──────────────────────────────────────────────────────────────────────

def build_llama() -> bool:
    print("=== Building llama.cpp ===")
    result = subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "-j"],
        cwd=REPO_ROOT,
    )
    return result.returncode == 0


# ──────────────────────────────────────────────────────────────────────
# quantize
# ──────────────────────────────────────────────────────────────────────

def _is_iq_type(ftype: str) -> bool:
    """IQ types require an importance matrix."""
    return ftype.upper().startswith("IQ")


def run_quantize(
    input_model: str,
    output_model: str,
    base_type: str,
    extra_args: list[str] | None = None,
    pure: bool = True,
    imatrix: str | None = None,
    tensor_type_file: str | None = None,
    token_embedding_type: str | None = None,
    output_tensor_type: str | None = None,
) -> tuple[bool, float, float]:
    """Run llama-quantize. Returns (success, size_mb, elapsed_s)."""
    cmd = [str(QUANTIZE_BIN)]

    # IQ types default to NOT pure (need mixed types for imatrix coverage)
    if pure:
        cmd.append("--pure")

    # Importance matrix (required for IQ types)
    if imatrix:
        cmd.extend(["--imatrix", imatrix])

    # Tensor type file for mixed-type IQ quantization
    if tensor_type_file:
        cmd.extend(["--tensor-type-file", tensor_type_file])

    # Embedding and output tensor types
    if token_embedding_type:
        cmd.extend(["--token-embedding-type", token_embedding_type])
    if output_tensor_type:
        cmd.extend(["--output-tensor-type", output_tensor_type])

    cmd.extend([input_model, output_model])

    # Only add type if not using tensor-type-file (which specifies types explicitly)
    if not tensor_type_file:
        cmd.append(base_type)

    # Insert extra args after binary name, before the main args
    if extra_args:
        cmd = cmd[:1] + extra_args + cmd[1:]

    print(f"=== Quantizing: {' '.join(cmd)} ===")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                            timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print("QUANTIZE FAILED:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False, 0.0, elapsed

    size_mb = os.path.getsize(output_model) / (1024 * 1024)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    print(f"Model size: {size_mb:.1f} MiB, time: {elapsed:.1f}s")
    return True, size_mb, elapsed


# ──────────────────────────────────────────────────────────────────────
# evaluate
# ──────────────────────────────────────────────────────────────────────

def run_eval(model_path: str) -> tuple[bool, float, float, float, float, float]:
    """Run llama-perplexity --kl-divergence. Returns (success, kl, elapsed_s, tok_s, ppl, same_top_p)."""
    cmd = [
        str(PERPLEXITY_BIN),
        "-m", model_path,
        "-f", DEFAULT_EVAL_DATA,
        "-t", "8",
        "-c", "512",
        "--chunks", "200",
        "-fa", "on",
        "--cache-type-k", "bf16",
        "--cache-type-v", "bf16",
        "--no-mmap",
        "-ngl", "999",
        "-np", "1",
        "--kl-divergence",
        "--kl-divergence-base", DEFAULT_LOGITS_BASE,
    ]
    print(f"=== Evaluating KL divergence ===")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                            timeout=1800)
    elapsed = time.time() - t0

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print("EVAL FAILED:")
        print(stderr[-2000:] if len(stderr) > 2000 else stderr)
        return False, 0.0, elapsed, 0.0, 0.0, 0.0

    combined = stdout + stderr

    kl_match = re.search(r"Mean\s+KLD:\s+([\d.]+)", combined)
    tok_match = re.search(r"(\d+\.?\d*)\s+tokens per second", combined)
    ppl_match = re.search(r"Mean PPL\(Q\)\s*:\s+([\d.]+)", combined)
    top_match = re.search(r"Same top p:\s+([\d.]+)", combined)

    kl = float(kl_match.group(1)) if kl_match else 0.0
    ppl = float(ppl_match.group(1)) if ppl_match else 0.0
    same_top_p = float(top_match.group(1)) if top_match else 0.0
    tok_s = float(tok_match.group(1)) if tok_match else 0.0
    if tok_s == 0.0 and elapsed > 0:
        tok_s = 5000.0 / elapsed  # rough: ~5000 eval tokens in KL mode

    tail = stdout.split("\n")[-20:]
    print("\n".join(tail))
    print(f"\nKL divergence: {kl:.6f},  PPL: {ppl:.4f},  Same top P: {same_top_p:.2f}%,  eval time: {elapsed:.1f}s")

    return True, kl, elapsed, tok_s, ppl, same_top_p


# ──────────────────────────────────────────────────────────────────────
# logging
# ──────────────────────────────────────────────────────────────────────

def append_result(
    exp_id: str,
    code_sha: str,
    parent_sha: str,
    description: str,
    status: str,
    kl_divergence: float,
    base_type: str,
    model_size_mb: float,
    quantize_time_s: float,
    eval_time_s: float,
    tokens_per_sec: float,
    ppl: float = 0.0,
    same_top_p: float = 0.0,
) -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("\t".join(HEADER) + "\n")

    row = [
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        exp_id,
        code_sha,
        parent_sha,
        description.replace("\t", " ").replace("\n", " "),
        status,
        f"{kl_divergence:.6f}",
        base_type,
        f"{model_size_mb:.1f}",
        f"{quantize_time_s:.1f}",
        f"{eval_time_s:.1f}",
        f"{tokens_per_sec:.1f}",
        f"{ppl:.4f}" if ppl > 0 else "",
        f"{same_top_p:.2f}" if same_top_p > 0 else "",
    ]
    with RESULTS_TSV.open("a") as f:
        f.write("\t".join(row) + "\n")


# ──────────────────────────────────────────────────────────────────────
# commands
# ──────────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    best_before = read_best_kl()
    exp_id = next_exp_id()

    # Build
    if not args.skip_build:
        if not build_llama():
            print("BUILD FAILED — aborting")
            sys.exit(1)

    # Git shas
    parent_sha = safe_git("rev-parse", "HEAD")
    code_sha = parent_sha

    # Determine imatrix (auto for IQ types)
    imatrix = args.imatrix
    if not imatrix and _is_iq_type(args.type) and not args.pure:
        if os.path.exists(DEFAULT_IMATRIX):
            imatrix = DEFAULT_IMATRIX

    # Determine tensor type file (auto for IQ types)
    tensor_types = args.tensor_type_file
    if not tensor_types and _is_iq_type(args.type) and not args.pure:
        if os.path.exists(DEFAULT_TENSOR_TYPES):
            tensor_types = DEFAULT_TENSOR_TYPES

    # Quantize
    ok, size_mb, quant_time = run_quantize(
        args.input, args.output, args.type,
        extra_args=args.extra_quantize_args,
        pure=args.pure,
        imatrix=imatrix,
        tensor_type_file=tensor_types,
        token_embedding_type=args.token_embedding_type,
        output_tensor_type=args.output_tensor_type,
    )
    if not ok:
        append_result(exp_id, code_sha, parent_sha, args.description,
                      "failed_quantize", 0.0, args.type,
                      0.0, quant_time, 0.0, 0.0)
        print(f"\nQUANTIZE FAILED — recorded in results.tsv")
        sys.exit(1)

    # Evaluate
    ok, kl, eval_time, tok_s, ppl, same_top_p = run_eval(args.output)
    if not ok:
        append_result(exp_id, code_sha, parent_sha, args.description,
                      "failed_eval", 0.0, args.type,
                      size_mb, quant_time, eval_time, tok_s,
                      ppl, same_top_p)
        print(f"\nEVAL FAILED — recorded in results.tsv")
        sys.exit(1)

    # Log
    append_result(exp_id, code_sha, parent_sha, args.description,
                  "success", kl, args.type,
                  size_mb, quant_time, eval_time, tok_s,
                  ppl, same_top_p)

    # Report
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_id}")
    print(f"KL divergence: {kl:.6f}")
    print(f"PPL:              {ppl:.4f}")
    print(f"Same top P:       {same_top_p:.2f}%")
    print(f"Model size: {size_mb:.1f} MiB")
    print(f"Tokens/sec:       {tok_s:.1f}")
    print(f"Quantize time: {quant_time:.1f}s")
    print(f"Eval time: {eval_time:.1f}s")

    if best_before == float("inf"):
        print(f"Best KL: {kl:.6f} (first run)")
    elif kl < best_before:
        print(f"Best KL: {kl:.6f} (IMPROVED from {best_before:.6f}, delta: {best_before - kl:.6f})")
    else:
        print(f"Best KL: {best_before:.6f} (regressed by {kl - best_before:.6f})")

    print(f"Recorded in: {RESULTS_TSV}")
    print(f"{'='*60}")


def cmd_status(_args: argparse.Namespace) -> None:
    best = read_best_kl()
    print(f"Current best KL: {best:.6f}")
    if RESULTS_TSV.exists():
        lines = RESULTS_TSV.read_text().strip().split("\n")
        print(f"Total experiments: {len(lines) - 1}")
        print(f"\nLast 5 experiments:")
        for line in lines[-5:]:
            parts = line.split("\t")
            if len(parts) >= 7:
                print(f"  {parts[0][:16]} | {parts[7]:6s} | KL={parts[6]:10s} | {parts[4][:50]}")


def cmd_eval(args: argparse.Namespace) -> None:
    ok, kl, elapsed, tok_s, ppl, same_top_p = run_eval(args.model)
    if ok:
        print(f"\nKL divergence: {kl:.6f},  PPL: {ppl:.4f},  Same top P: {same_top_p:.2f}%,  time: {elapsed:.1f}s,  tok/s: {tok_s:.1f}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="auto_quantize.py — llama.cpp quantization research orchestrator",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run full experiment (build → quantize → eval → log)")
    p_run.add_argument("--type", default="IQ2_XXS", help="Base quantization type (default: IQ2_XXS)")
    p_run.add_argument("--pure", action="store_true",
                       help="Enable --pure mode (all tensors to same type, no imatrix needed)")
    p_run.add_argument("--imatrix", default=None,
                       help=f"Importance matrix file (auto-detected for IQ types: {DEFAULT_IMATRIX})")
    p_run.add_argument("--tensor-type-file", default=None,
                       help=f"Tensor type mapping file (auto-detected for IQ types: {DEFAULT_TENSOR_TYPES})")
    p_run.add_argument("--token-embedding-type", default=None,
                       help="Token embedding tensor type override")
    p_run.add_argument("--output-tensor-type", default=None,
                       help="Output tensor type override")
    p_run.add_argument("--input", default=DEFAULT_INPUT, help="Input BF16 GGUF")
    p_run.add_argument("--output", default=DEFAULT_OUTPUT, help="Output quantized GGUF")
    p_run.add_argument("--tag", default="", help="Short experiment tag")
    p_run.add_argument("--description", default="", help="One-line experiment description")
    p_run.add_argument("--skip-build", action="store_true", help="Skip cmake build step")
    p_run.add_argument("--extra-quantize-args", nargs=argparse.REMAINDER, default=[],
                       help="Extra args passed directly to llama-quantize (must be LAST argument)")

    # status
    p_status = sub.add_parser("status", help="Show current best KL and experiment history")

    # eval
    p_eval = sub.add_parser("eval", help="Run KL evaluation only")
    p_eval.add_argument("--model", default=DEFAULT_OUTPUT, help="Quantized model to evaluate")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
