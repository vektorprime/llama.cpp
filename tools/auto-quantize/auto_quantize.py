#!/usr/bin/env python3
"""
auto_quantize.py — orchestration script for llama.cpp quantization research.

Runs the experiment loop: build → quantize → evaluate KL divergence → log results.
Follows the autoresearch pattern (karpathy/autoresearch, vektorprime/AutoQuant).

Usage:
    python auto_quantize.py run \\
        --type Q2_K --diffusion 0.5 --refine 3 \\
        --tag "err-diff-0.5" --description "error diffusion diff=0.5"
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
DEFAULT_OUTPUT = "/home/user/llm/models/Qwen3.5-2B/our-quantized-model.gguf"
DEFAULT_EVAL_DATA = "/home/user/llm/wikitext-2-raw/wiki.test.raw"
DEFAULT_LOGITS_BASE = "/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-BF16.logits"

QUANTIZE_BIN = BUILD_DIR / "bin" / "llama-quantize"
PERPLEXITY_BIN = BUILD_DIR / "bin" / "llama-perplexity"

HEADER = [
    "timestamp", "exp_id", "code_sha", "parent_sha",
    "description", "status", "kl_divergence",
    "base_type", "diffusion", "refine_iterations",
    "model_size_mb", "quantize_time_s", "eval_time_s", "tokens_per_sec",
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

def run_quantize(
    input_model: str,
    output_model: str,
    base_type: str,
    extra_args: list[str] | None = None,
) -> tuple[bool, float, float]:
    """Run llama-quantize. Returns (success, size_mb, elapsed_s)."""
    cmd = [
        str(QUANTIZE_BIN),
        "--pure",
        input_model,
        output_model,
        base_type,
    ]
    if extra_args:
        cmd = cmd[:1] + extra_args + cmd[1:]  # insert extra args after binary name

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

def run_eval(model_path: str) -> tuple[bool, float, float, float]:
    """Run llama-perplexity --kl-divergence. Returns (success, kl, elapsed_s, tok_s)."""
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
        return False, 0.0, elapsed, 0.0

    # Parse KL divergence from output
    kl_match = re.search(r"Mean\s+KLD:\s+([\d.]+)", stdout)
    tok_match = re.search(r"(\d+\.?\d*)\s+tokens per second", stdout)

    if not kl_match:
        # Try combined stdout+stderr
        combined = stdout + stderr
        kl_match = re.search(r"Mean\s+KLD:\s+([\d.]+)", combined)

    kl = float(kl_match.group(1)) if kl_match else 0.0
    tok_s = float(tok_match.group(1)) if tok_match else 0.0

    # Print last 20 lines for visibility
    tail = stdout.split("\n")[-20:]
    print("\n".join(tail))
    print(f"\nKL divergence: {kl:.6f}, eval time: {elapsed:.1f}s")

    return True, kl, elapsed, tok_s


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
    diffusion: float,
    refine_iterations: int,
    model_size_mb: float,
    quantize_time_s: float,
    eval_time_s: float,
    tokens_per_sec: float,
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
        f"{diffusion:.3f}" if diffusion >= 0 else "",
        str(refine_iterations) if refine_iterations >= 0 else "",
        f"{model_size_mb:.1f}",
        f"{quantize_time_s:.1f}",
        f"{eval_time_s:.1f}",
        f"{tokens_per_sec:.1f}",
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
    code_sha = parent_sha  # will be updated after commit if --commit

    # Quantize
    extra_args = []
    if args.diffusion >= 0:
        extra_args.append(f"--q2k-diffusion={args.diffusion}")
    if args.refine_iterations >= 0:
        extra_args.append(f"--q2k-refine={args.refine_iterations}")

    ok, size_mb, quant_time = run_quantize(
        args.input, args.output, args.type, extra_args=extra_args,
    )
    if not ok:
        append_result(exp_id, code_sha, parent_sha, args.description,
                      "failed_quantize", 0.0, args.type,
                      args.diffusion, args.refine_iterations,
                      0.0, quant_time, 0.0, 0.0)
        print(f"\nQUANTIZE FAILED — recorded in results.tsv")
        sys.exit(1)

    # Evaluate
    ok, kl, eval_time, tok_s = run_eval(args.output)
    if not ok:
        append_result(exp_id, code_sha, parent_sha, args.description,
                      "failed_eval", 0.0, args.type,
                      args.diffusion, args.refine_iterations,
                      size_mb, quant_time, eval_time, tok_s)
        print(f"\nEVAL FAILED — recorded in results.tsv")
        sys.exit(1)

    # Log
    append_result(exp_id, code_sha, parent_sha, args.description,
                  "success", kl, args.type,
                  args.diffusion, args.refine_iterations,
                  size_mb, quant_time, eval_time, tok_s)

    # Report
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_id}")
    print(f"KL divergence: {kl:.6f}")
    print(f"Model size: {size_mb:.1f} MiB")
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
    ok, kl, elapsed, tok_s = run_eval(args.model)
    if ok:
        print(f"\nKL divergence: {kl:.6f}, time: {elapsed:.1f}s, tok/s: {tok_s:.1f}")


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
    p_run.add_argument("--type", default="Q2_K", help="Base quantization type")
    p_run.add_argument("--diffusion", type=float, default=-1,
                       help="Error diffusion strength (0.0-1.0, -1 = disabled)")
    p_run.add_argument("--refine-iterations", type=int, default=-1,
                       help="MSE scale refinement iterations (-1 = disabled)")
    p_run.add_argument("--input", default=DEFAULT_INPUT, help="Input BF16 GGUF")
    p_run.add_argument("--output", default=DEFAULT_OUTPUT, help="Output quantized GGUF")
    p_run.add_argument("--tag", default="", help="Short experiment tag")
    p_run.add_argument("--description", default="", help="One-line experiment description")
    p_run.add_argument("--skip-build", action="store_true", help="Skip cmake build step")

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
