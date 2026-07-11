#!/usr/bin/env python3
"""Analyze FWHT-transformed weight statistics to find compression opportunities.

Reads BF16 GGUF model, applies FWHT to selected tensors, and produces
statistical reports on:
  1. Per-position mean/std across all 256 FWHT outputs
  2. DC (pos 0) vs AC (pos 1-255) component separation
  3. Per-channel (input column) importance distribution after FWHT
  4. Quantization error simulation at different bit widths
  5. Sparsity and outlier analysis

Output: stdout report + interactive exploration.
"""

import struct, sys, os, math
import numpy as np
from collections import defaultdict
from pathlib import Path

# --- fwht_256 implementation (matches ggml-quants.c) ---

def fwht_256(x):
    """Unnormalized 256-point Walsh-Hadamard transform (in-place)."""
    # stride = 1
    for step in range(1, 256, 2):
        for i in range(0, 256, 2*step):
            for j in range(i, i + step):
                u = x[j]
                v = x[j + step]
                x[j] = u + v
                x[j + step] = u - v

    # stride = 2
    for step in range(2, 256, 4):
        for i in range(0, 256, 2*step):
            for j in range(i, i + step, 2):
                u = x[j]
                v = x[j + step]
                x[j] = u + v
                x[j + step] = u - v

    # stride = 4
    for step in range(4, 256, 8):
        for i in range(0, 256, 2*step):
            for j in range(i, i + step, 4):
                u = x[j]
                v = x[j + step]
                x[j] = u + v
                x[j + step] = u - v

    # stride = 8, 16, 32, 64, 128
    for h in range(3, 8):
        step = 1 << h
        for i in range(0, 256, 2*step):
            for j in range(i, i + step, 8):
                u = x[j]
                v = x[j + step]
                x[j] = u + v
                x[j + step] = u - v

# --- GGUF reader (minimal, no dependencies) ---

def read_gguf_tensor_data(fp, offset, n_dims, dims, ftype):
    """Read raw tensor data from GGUF file."""
    n_elements = 1
    for d in dims:
        n_elements *= d
    fp.seek(offset)
    if ftype == 1:  # F32
        return np.frombuffer(fp.read(n_elements * 4), dtype=np.float32)
    elif ftype == 3:  # BF16
        raw = np.frombuffer(fp.read(n_elements * 2), dtype=np.uint16)
        f32 = np.zeros(n_elements, dtype=np.float32)
        for i in range(n_elements):
            # BF16: upper 16 bits of float32
            f32_bytes = struct.pack('<I', int(raw[i]) << 16)
            f32[i] = struct.unpack('<f', f32_bytes)[0]
        return f32
    else:
        raise ValueError(f"Unsupported ftype: {ftype}")

def parse_gguf(path, tensor_filter=None):
    """Parse GGUF file, return metadata and tensor info."""
    fp = open(path, 'rb')

    # Check magic
    magic = fp.read(4)
    if magic != b'GGUF':
        raise ValueError("Not a GGUF file")

    # Read version
    version = struct.unpack('<I', fp.read(4))[0]

    # Read tensor count and metadata kv count
    n_tensors, n_kv = struct.unpack('<QQ', fp.read(16))

    # Skip metadata KV pairs (we only need tensor info)
    for _ in range(n_kv):
        key_type = struct.unpack('<I', fp.read(4))[0]
        key_len = struct.unpack('<Q', fp.read(8))[0]
        key = fp.read(key_len).decode('utf-8')
        # Read value based on type
        if key_type == 0:  # UINT8
            fp.read(1)
        elif key_type == 1:  # INT8
            fp.read(1)
        elif key_type == 2:  # UINT16
            fp.read(2)
        elif key_type == 3:  # INT16
            fp.read(2)
        elif key_type == 4:  # UINT32
            fp.read(4)
        elif key_type == 5:  # INT32
            fp.read(4)
        elif key_type == 6:  # FLOAT32
            fp.read(4)
        elif key_type == 7:  # BOOL
            fp.read(1)
        elif key_type == 8:  # STRING
            vlen = struct.unpack('<Q', fp.read(8))[0]
            fp.read(vlen)
        elif key_type == 9:  # ARRAY
            atype = struct.unpack('<I', fp.read(4))[0]
            alen = struct.unpack('<I', fp.read(4))[0]
            fp.read(alen * {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:8}.get(atype, 4))
        elif key_type == 10:  # UINT64
            fp.read(8)
        elif key_type == 11:  # INT64
            fp.read(8)
        elif key_type == 12:  # FLOAT64
            fp.read(8)

    # Read tensor infos
    tensor_names = []
    tensor_info = {}
    current_offset = fp.tell()
    alignment = 32

    for i in range(n_tensors):
        name_len = struct.unpack('<Q', fp.read(8))[0]
        name = fp.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<I', fp.read(4))[0]
        dims = struct.unpack(f'<{n_dims}Q', fp.read(8 * n_dims))
        ftype = struct.unpack('<I', fp.read(4))[0]
        offset_var = struct.unpack('<Q', fp.read(8))[0]

        if not tensor_filter or any(p in name for p in tensor_filter):
            tensor_names.append(name)
            tensor_info[name] = {
                'n_dims': n_dims,
                'dims': dims,
                'ftype': ftype,
                'offset': offset_var,
            }

    return fp, tensor_names, tensor_info

# --- Analysis ---

def analyze_model(model_path, tensor_patterns=None):
    """Main analysis function."""
    if tensor_patterns is None:
        tensor_patterns = ['blk']

    print(f"Loading model: {model_path}")
    fp, tensor_names, tensor_info = parse_gguf(model_path, tensor_patterns)
    print(f"Found {len(tensor_names)} tensors matching pattern")

    # Accumulators
    pos_sum = np.zeros(256, dtype=np.float64)
    pos_sum2 = np.zeros(256, dtype=np.float64)
    pos_count = 0
    pos_min = np.full(256, np.inf, dtype=np.float32)
    pos_max = np.full(256, -np.inf, dtype=np.float32)

    layer_stats = []
    all_fwht_blocks = []

    for tname in tensor_names:
        info = tensor_info[tname]
        dims = info['dims']
        if len(dims) < 2:
            continue
        n_rows = dims[0]
        n_cols = dims[1]
        if n_cols % 256 != 0:
            continue

        data = read_gguf_tensor_data(fp, info['offset'], info['n_dims'], dims, info['ftype'])
        data = data.reshape(n_rows, n_cols)

        # Apply FWHT to each block in each row
        n_blocks_per_row = n_cols // 256
        layer_blocks = []

        for row_idx in range(n_rows):
            for block_idx in range(n_blocks_per_row):
                col_start = block_idx * 256
                block = data[row_idx, col_start:col_start + 256].copy().astype(np.float64)

                # Apply FWHT
                fwht_256(block)

                # Accumulate per-position stats
                pos_sum += block
                pos_sum2 += block * block
                pos_min = np.minimum(pos_min, block.astype(np.float32))
                pos_max = np.maximum(pos_max, block.astype(np.float32))
                pos_count += 1

                layer_blocks.append(block)
                all_fwht_blocks.append(block)

        if layer_blocks:
            layer_blocks = np.array(layer_blocks)
            layer_stats.append({
                'name': tname,
                'n_blocks': len(layer_blocks),
                'mean': float(np.mean(layer_blocks)),
                'std': float(np.std(layer_blocks)),
                'abs_mean': float(np.mean(np.abs(layer_blocks))),
            })

        print(f"  {tname}: {n_rows} rows x {n_cols} cols -> {int(n_rows * n_blocks_per_row)} blocks")

    fp.close()

    all_fwht_blocks = np.array(all_fwht_blocks, dtype=np.float64)
    pos_mean = pos_sum / pos_count
    pos_std = np.sqrt(pos_sum2 / pos_count - pos_mean * pos_mean)

    return {
        'pos_mean': pos_mean,
        'pos_std': pos_std,
        'pos_min': pos_min,
        'pos_max': pos_max,
        'pos_count': pos_count,
        'layer_stats': layer_stats,
        'all_blocks': all_fwht_blocks,
    }


def print_report(results):
    pos_mean = results['pos_mean']
    pos_std = results['pos_std']
    pos_min = results['pos_min']
    pos_max = results['pos_max']
    all_blocks = results['all_blocks']

    print("\n" + "=" * 72)
    print("FWHT WEIGHT ANALYSIS REPORT")
    print("=" * 72)

    # 1. Overview
    print(f"\n--- Overview ---")
    print(f"Total 256-element blocks analyzed: {len(all_blocks):,}")
    global_mean = float(np.mean(all_blocks))
    global_std = float(np.std(all_blocks))
    global_min = float(np.min(all_blocks))
    global_max = float(np.max(all_blocks))
    print(f"Global mean: {global_mean:.6f}")
    print(f"Global std:  {global_std:.6f}")
    print(f"Global min:  {global_min:.6f}")
    print(f"Global max:  {global_max:.6f}")

    # Fraction of near-zeros
    near_zero = float(np.mean(np.abs(all_blocks) < 1e-4))
    print(f"Fraction |x| < 1e-4: {near_zero*100:.2f}%")

    # 2. DC vs AC split
    print(f"\n--- DC (position 0) vs AC (positions 1-255) ---")
    dc = all_blocks[:, 0]
    ac = all_blocks[:, 1:].flatten()
    dc_mean = float(np.mean(dc))
    dc_std = float(np.std(dc))
    ac_mean = float(np.mean(ac))
    ac_std = float(np.std(ac))
    dc_var = float(np.var(dc))
    ac_var = float(np.var(ac))
    total_var = float(np.var(all_blocks.flatten()))

    print(f"DC component:")
    print(f"  mean: {dc_mean:.6f}  std: {dc_std:.6f}")
    print(f"  var fraction of total: {dc_var/total_var*100:.1f}%  (position 0 = {dc_var/total_var/np.size(dc)*256*100:.1f}% of 1/256th)")
    print(f"  fraction |x| < 1e-4: {float(np.mean(np.abs(dc) < 1e-4))*100:.2f}%")

    print(f"AC components (avg across 255 positions):")
    print(f"  mean: {ac_mean:.6f}  std: {ac_std:.6f}")
    print(f"  var fraction of total: {ac_var/total_var*100:.1f}%")
    print(f"  fraction |x| < 1e-4: {float(np.mean(np.abs(ac) < 1e-4))*100:.2f}%")

    # 3. Per-position variance (top/bottom positions)
    print(f"\n--- Per-Position Statistics (256 positions) ---")
    variances = pos_std ** 2

    # Top 10 highest variance positions
    top_indices = np.argsort(-variances)[:10]
    print(f"Top 10 highest-variance positions:")
    for idx in top_indices:
        print(f"  pos {idx:3d}: mean={pos_mean[idx]:10.6f} std={pos_std[idx]:10.6f} "
              f"range=[{pos_min[idx]:.4f}, {pos_max[idx]:.4f}]")

    # Bottom 10 lowest variance positions
    bottom_indices = np.argsort(variances)[:10]
    print(f"Bottom 10 lowest-variance positions:")
    for idx in bottom_indices:
        print(f"  pos {idx:3d}: mean={pos_mean[idx]:10.6f} std={pos_std[idx]:10.6f} "
              f"range=[{pos_min[idx]:.4f}, {pos_max[idx]:.4f}]")

    # Variance spread
    print(f"\nVariance spread: min={variances.min():.6f} max={variances.max():.6f} "
          f"ratio={variances.max()/variances.max():.2f}x  "
          f"(max/min={variances.max()/variances.min():.2f}x)")

    # 4. Scale statistics (per-block mean = DC/16, which approximates scale)
    print(f"\n--- Per-Block Scale Analysis (|block mean|) ---")
    block_means = np.abs(np.mean(all_blocks, axis=1))
    print(f"  mean: {float(np.mean(block_means)):.6f}")
    print(f"  std:  {float(np.std(block_means)):.6f}")
    print(f"  min:  {float(np.min(block_means)):.6f}")
    print(f"  max:  {float(np.max(block_means)):.6f}")

    # 5. Quantization error simulation
    print(f"\n--- Quantization Error Simulation ---")
    print(f"(MSE of round-to-n-bits on FWHT-space values)")

    for n_bits in [2, 3, 4, 5, 6, 8]:
        N = 2 ** n_bits
        scale_factor = N / 2
        # Uniform quantization: quantize to [0, N-1], dequantize to [-1, 1]
        # Clip values to [-10, 10] sigma range for realistic assessment
        clipped = np.clip(all_blocks, global_mean - 4 * global_std, global_mean + 4 * global_std)
        normalized = (clipped - global_mean) / global_std  # ~N(0,1)
        q = np.clip(np.round(normalized * scale_factor / 4), -scale_factor, scale_factor - 1)
        dq = q / scale_factor * 4
        mse = float(np.mean((normalized - dq) ** 2))
        print(f"  {n_bits}-bit: MSE={mse:.6f}  SQNR={-10*math.log10(mse):.1f} dB")

    # 6. Layer-wise comparison
    print(f"\n--- Layer-Wise Statistics (top 5 + bottom 5 by abs mean) ---")
    for ls in sorted(results['layer_stats'], key=lambda x: x['abs_mean'], reverse=True)[:5]:
        print(f"  {ls['name']}: blocks={ls['n_blocks']:6d} |mean|={ls['abs_mean']:.6f} std={ls['std']:.6f}")
    print(f"  ...")
    for ls in sorted(results['layer_stats'], key=lambda x: x['abs_mean'], reverse=False)[:5]:
        print(f"  {ls['name']}: blocks={ls['n_blocks']:6d} |mean|={ls['abs_mean']:.6f} std={ls['std']:.6f}")

    # 7. Potential compression ideas summary
    print(f"\n--- Compression Opportunity Assessment ---")
    print(f"  Intra-block variance ratio: {variances.max()/variances.min():.1f}x")
    if variances.max() / variances.min() > 4:
        print(f"  => SIGNIFICANT: positions differ in variance — uneven bit allocation could help")

    dc_fraction = dc_var / total_var
    if dc_fraction > 0.1:
        print(f"  => DC component carries {dc_fraction*100:.1f}% of total variance — encode DC separately")

    zero_fraction = float(np.mean(np.abs(all_blocks) < 1e-4))
    if zero_fraction > 0.05:
        print(f"  => {zero_fraction*100:.1f}% near-zero values — sparse encoding may help")

    abs_corr = np.abs(np.corrcoef(all_blocks[:, :10].T)) if all_blocks.shape[0] > 10 else np.eye(1)
    off_diag = abs_corr[~np.eye(abs_corr.shape[0], dtype=bool)]
    if np.mean(off_diag) > 0.1:
        print(f"  => Adjacent positions show non-trivial correlation (mean |r|={np.mean(off_diag):.3f}) — residual coding viable")

    print(f"\n(Done.)")


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not model_path:
        model_path = '/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf'

    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    patterns = [pattern] if pattern else ['blk']

    results = analyze_model(model_path, patterns)
    print_report(results)
