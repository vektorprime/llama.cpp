#!/usr/bin/env python3
"""Vectorized scale analysis for Q4_K_M_CLONE. Uses batched NumPy operations."""

import struct, sys, numpy as np

FWHT_SIZE = 256
N_SUB = 8
SUB_SIZE = 32

def fwht_256_batch(blocks):
    """Apply FWHT to rows of 256-element blocks. blocks shape: (N, 256)"""
    step = 1
    while step < 256:
        for i in range(0, 256, 2 * step):
            a = blocks[:, i:i+step].copy()
            b = blocks[:, i+step:i+2*step]
            blocks[:, i:i+step] = a + b
            blocks[:, i+step:i+2*step] = a - b
        step <<= 1

def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8')

def rd_type(f):
    return struct.unpack('<I', f.read(4))[0]

def skip_value(f, vtype, n=1):
    sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:8, 9:0, 10:8, 11:8, 12:8}
    ct, cn = vtype, n
    while True:
        if ct == 8:
            for _ in range(cn):
                slen = struct.unpack('<Q', f.read(8))[0]; f.read(slen)
        elif ct == 9:
            et = struct.unpack('<I', f.read(4))[0]; en = struct.unpack('<Q', f.read(8))[0]
            ct, cn = et, en; continue
        else:
            f.read(sizes.get(ct, 4) * cn)
        break

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf'
    filt = sys.argv[2] if len(sys.argv) > 2 else 'blk'
    max_tensors = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    max_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 500000

    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<q', f.read(8))[0]
    n_kv = struct.unpack('<q', f.read(8))[0]

    for _ in range(n_kv):
        rd_str(f); vtype = rd_type(f); skip_value(f, vtype)

    tensors = []
    for i in range(n_tensors):
        name = rd_str(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack(f'<{n_dims}q', f.read(8 * n_dims))
        ftype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        if filt in name and ftype in (1, 30) and n_dims >= 2:
            tensors.append((i, name, n_dims, dims, ftype, offset))

    print(f"Matching tensors: {len(tensors)} (processing up to {max_tensors})", flush=True)

    # Collectors
    all_ls = []       # all ls values (0..63), per sub-block
    all_lm = []
    ls_ranges = []    # max-min ls within superblock
    lm_ranges = []
    ls_cv = []        # std/mean ls within superblock
    all_d = []
    all_dmin = []

    block_ls_means = []
    block_lm_means = []
    block_d_vals = []
    block_dmin_vals = []

    n_blocks_done = 0

    processed = 0
    for ti, (idx, name, n_dims, dims, ftype, offset) in enumerate(tensors):
        if max_tensors and processed >= max_tensors:
            break
        if n_blocks_done >= max_blocks:
            break

        ne0, ne1 = dims[1], dims[0]
        if ne0 % FWHT_SIZE != 0:
            continue

        n_elems = 1
        for d in dims: n_elems *= d

        f.seek(offset)
        if ftype == 30:
            raw = np.frombuffer(f.read(n_elems * 2), dtype=np.uint16)
            f32 = np.zeros(n_elems, dtype=np.float32)
            f32.view(np.uint32)[:] = raw.astype(np.uint32) << 16
            data = f32.reshape(dims)
        else:
            data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32).reshape(dims)

        n_blocks_per_row = ne0 // FWHT_SIZE
        total_blocks_this = ne1 * n_blocks_per_row

        # Process rows in batches
        batch_size = min(200, ne1)
        for row_start in range(0, ne1, batch_size):
            row_end = min(row_start + batch_size, ne1)
            n_rows = row_end - row_start
            n_batch_blocks = n_rows * n_blocks_per_row

            if n_blocks_done + n_batch_blocks > max_blocks:
                # Trim
                blocks_wanted = max_blocks - n_blocks_done
                n_rows = max(1, blocks_wanted // n_blocks_per_row)
                row_end = row_start + n_rows
                n_batch_blocks = n_rows * n_blocks_per_row

            # Extract all blocks from this batch: shape (n_batch_blocks, 256)
            block_data = data[row_start:row_end, :]
            # Reshape to (n_batch_blocks, 256)
            blocks = block_data.reshape(-1, FWHT_SIZE).astype(np.float64)

            # Apply FWHT to all blocks
            fwht_256_batch(blocks)

            # Reshape blocks to (n_batch_blocks, 8, 32) for sub-block analysis
            blocks_3d = blocks.reshape(-1, N_SUB, SUB_SIZE)

            # Per sub-block: min and max
            sub_mins = np.min(blocks_3d, axis=2)    # (N, 8)
            sub_maxs = np.max(blocks_3d, axis=2)    # (N, 8)
            sub_ranges = sub_maxs - sub_mins          # (N, 8) - proxy for scales
            sub_mins_offset = -sub_mins               # (N, 8) - proxy for mins

            # Secondary quantization: ls = round(63 * range / max_range_per_block)
            max_range_per_block = np.max(sub_ranges, axis=1, keepdims=True)  # (N, 1)
            max_range_per_block[max_range_per_block < 1e-10] = 1e-10
            ls_vals = np.clip(np.round(63.0 * sub_ranges / max_range_per_block).astype(np.int32), 0, 63)

            max_min_per_block = np.max(sub_mins_offset, axis=1, keepdims=True)  # (N, 1)
            max_min_per_block[max_min_per_block < 1e-10] = 1e-10
            lm_vals = np.clip(np.round(63.0 * sub_mins_offset / max_min_per_block).astype(np.int32), 0, 63)

            d_vals = max_range_per_block.flatten() / 63.0
            dmin_vals = max_min_per_block.flatten() / 63.0

            # Per-superblock stats
            ls_range_batch = np.max(ls_vals, axis=1) - np.min(ls_vals, axis=1)
            ls_mean_batch = np.mean(ls_vals.astype(np.float64), axis=1)
            ls_std_batch = np.std(ls_vals.astype(np.float64), axis=1)
            ls_cv_batch = np.divide(ls_std_batch, ls_mean_batch, where=ls_mean_batch > 0,
                                     out=np.zeros_like(ls_mean_batch))

            lm_range_batch = np.max(lm_vals, axis=1) - np.min(lm_vals, axis=1)
            lm_mean_batch = np.mean(lm_vals.astype(np.float64), axis=1)

            all_ls.extend(ls_vals.flatten().tolist())
            all_lm.extend(lm_vals.flatten().tolist())
            ls_ranges.extend(ls_range_batch.tolist())
            lm_ranges.extend(lm_range_batch.tolist())
            ls_cv.extend(ls_cv_batch.tolist())
            all_d.extend(d_vals.tolist())
            all_dmin.extend(dmin_vals.tolist())
            block_ls_means.extend(ls_mean_batch.tolist())
            block_lm_means.extend(lm_mean_batch.tolist())
            block_d_vals.extend(d_vals.tolist())
            block_dmin_vals.extend(dmin_vals.tolist())

            n_blocks_done += n_batch_blocks

            if n_blocks_done >= max_blocks:
                break

            if n_blocks_done % 50000 == 0:
                print(f"  ... {n_blocks_done:,} blocks processed", flush=True)

        processed += 1
        print(f"  [{processed}] {name}: {ne1}x{ne0} -> {total_blocks_this:,} blocks "
              f"(total: {n_blocks_done:,})", flush=True)

    f.close()

    n_blocks = len(block_d_vals)
    print(f"\n{'='*72}")
    print(f"SCALE ANALYSIS — {n_blocks:,} superblocks ({n_blocks*8:,} sub-blocks)")
    print(f"{'='*72}")

    # Convert to numpy arrays
    all_ls_arr = np.array(all_ls, dtype=np.int32)
    all_lm_arr = np.array(all_lm, dtype=np.int32)
    ls_r = np.array(ls_ranges, dtype=np.int32)
    lm_r = np.array(lm_ranges, dtype=np.int32)
    blk_d = np.array(block_d_vals, dtype=np.float64)
    blk_dmin = np.array(block_dmin_vals, dtype=np.float64)

    # === Q1: Scale Uniformity ===
    print(f"\n--- Q1: Sub-block Scale Uniformity After FWHT ---")
    print(f"  ls range (max-min) distribution:")
    for k in [1, 2, 3, 4, 5, 8, 16, 32, 63]:
        n = np.sum(ls_r <= k)
        pct = 100.0 * n / len(ls_r)
        mark = " <<<" if pct >= 99.0 else ""
        print(f"    <= {k:2d}: {n:10,} ({pct:5.1f}%){mark}")
    print(f"  ls range: p50={np.median(ls_r):.1f} p75={np.percentile(ls_r,75):.1f} "
          f"p90={np.percentile(ls_r,90):.1f} p95={np.percentile(ls_r,95):.1f}")
    print(f"  ls CV (std/mean) p50={np.percentile(ls_cv,50):.4f} "
          f"p75={np.percentile(ls_cv,75):.4f} p90={np.percentile(ls_cv,90):.4f}")

    print(f"\n  lm range distribution:")
    for k in [1, 2, 3, 4, 5, 8, 16, 32, 63]:
        n = np.sum(lm_r <= k)
        pct = 100.0 * n / len(lm_r)
        mark = " <<<" if pct >= 99.0 else ""
        print(f"    <= {k:2d}: {n:10,} ({pct:5.1f}%){mark}")

    # === Delta encoding (computed from per-superblock ls sequences) ===
    print(f"\n--- Q3: Delta Encoding ---")
    # ls values are stored flat: ls[sub0], ls[sub1], ... ls[sub7] repeating
    # Reshape to (n_blocks, 8) for delta computation
    ls_mat = all_ls_arr.reshape(-1, N_SUB)  # (n_blocks, 8)
    lm_mat = all_lm_arr.reshape(-1, N_SUB)

    ls_deltas = (ls_mat[:, 1:] - ls_mat[:, :-1]).flatten()
    lm_deltas = (lm_mat[:, 1:] - lm_mat[:, :-1]).flatten()

    print(f"  ls deltas (7 per block, {len(ls_deltas):,} total):")
    print(f"    range: [{np.min(ls_deltas)}, {np.max(ls_deltas)}]")
    print(f"    |delta| mean={np.mean(np.abs(ls_deltas)):.2f} "
          f"median={np.median(np.abs(ls_deltas)):.1f}")
    print(f"    delta == 0: {np.sum(ls_deltas == 0):,} "
          f"({100*np.sum(ls_deltas==0)/len(ls_deltas):.1f}%)")
    for k in [1, 2, 3, 4, 5, 7, 8, 15, 31]:
        n = np.sum(np.abs(ls_deltas) <= k)
        pct = 100.0 * n / len(ls_deltas)
        bits = 1 + int(np.ceil(np.log2(2*k+1)))  # sign bit + value bits
        mark = " <<<" if pct >= 99.0 else ""
        print(f"    |d| <= {k:2d}: {n:10,}/{len(ls_deltas):,} ({pct:5.1f}%)  [{bits}-bit signed]{mark}")

    print(f"\n  lm deltas ({len(lm_deltas):,} total):")
    print(f"    range: [{np.min(lm_deltas)}, {np.max(lm_deltas)}]")
    for k in [1, 2, 3, 4, 7, 8, 15, 31]:
        n = np.sum(np.abs(lm_deltas) <= k)
        pct = 100.0 * n / len(lm_deltas)
        bits = 1 + int(np.ceil(np.log2(2*k+1)))
        mark = " <<<" if pct >= 99.0 else ""
        print(f"    |d| <= {k:2d}: {n:10,}/{len(lm_deltas):,} ({pct:5.1f}%)  [{bits}-bit signed]{mark}")

    # === Q2: Prediction from d ===
    print(f"\n--- Q2: Linear Regression d -> ls_mean ---")
    valid = (blk_d > 1e-15) & (np.array(block_ls_means) > 0)
    log_d = np.log(blk_d[valid])
    log_ls = np.log(np.array(block_ls_means)[valid])
    corr = np.corrcoef(log_d, log_ls)[0, 1]
    A = np.vstack([log_d, np.ones_like(log_d)]).T
    slope, intercept = np.linalg.lstsq(A, log_ls, rcond=None)[0]
    resid = log_ls - (slope * log_d + intercept)
    print(f"  ln(ls_mean) vs ln(d): corr={corr:.4f}, slope={slope:.4f}")
    print(f"  Residual std: {np.std(resid):.4f}, R^2={corr**2:.4f}")

    valid_m = (blk_dmin > 1e-15) & (np.array(block_lm_means) > 0)
    if np.sum(valid_m) > 10:
        log_dmin = np.log(blk_dmin[valid_m])
        log_lm_m = np.log(np.array(block_lm_means)[valid_m])
        corr_m = np.corrcoef(log_dmin, log_lm_m)[0, 1]
        print(f"  ln(lm_mean) vs ln(dmin): corr={corr_m:.4f}, R^2={corr_m**2:.4f}")

    # === Q4: Bit reduction ===
    print(f"\n--- Q4: Bit Reduction Impact ---")
    for bits in [1, 2, 3, 4, 5, 6]:
        max_val = 2**bits - 1
        n_ok_ls = np.sum(all_ls_arr <= max_val)
        n_ok_lm = np.sum(all_lm_arr <= max_val)
        print(f"  {bits}-bit (0..{max_val:2d}): "
              f"ls={n_ok_ls:10,}/{len(all_ls_arr):,} ({100*n_ok_ls/len(all_ls_arr):.2f}%)  "
              f"lm={n_ok_lm:10,}/{len(all_lm_arr):,} ({100*n_ok_lm/len(all_lm_arr):.2f}%)")

    # Value histograms
    print(f"\n  ls histogram (most common values):")
    ls_unique, ls_counts = np.unique(all_ls_arr, return_counts=True)
    top_idx = np.argsort(-ls_counts)[:15]
    for i in top_idx:
        print(f"    ls={ls_unique[i]:2d}: {ls_counts[i]:10,} ({100*ls_counts[i]/len(all_ls_arr):.2f}%)")
    print(f"    (other {len(ls_unique)-15} values): "
          f"{len(all_ls_arr)-sum(ls_counts[top_idx]):,} "
          f"({100*(len(all_ls_arr)-sum(ls_counts[top_idx]))/len(all_ls_arr):.2f}%)")

    # ls == 0 frequency
    print(f"\n  ls == 0: {np.sum(all_ls_arr==0):,}/{len(all_ls_arr):,} "
          f"({100*np.sum(all_ls_arr==0)/len(all_ls_arr):.1f}%)")
    print(f"  lm == 0: {np.sum(all_lm_arr==0):,}/{len(all_lm_arr):,} "
          f"({100*np.sum(all_lm_arr==0)/len(all_lm_arr):.1f}%)")

    # === Q5: dmin elimination ===
    print(f"\n--- Q5: dmin Elimination Potential ---")
    corr_ls_lm = np.corrcoef(all_ls_arr.astype(float), all_lm_arr.astype(float))[0, 1]
    print(f"  corr(ls, lm) across all sub-blocks: {corr_ls_lm:.4f}")

    # Per-block ls/lm correlation
    per_block_corrs = []
    sample_n = min(n_blocks, 20000)
    for b in range(sample_n):
        sls = ls_mat[b].astype(float)
        slm = lm_mat[b].astype(float)
        if np.std(sls) > 0 and np.std(slm) > 0:
            c = np.corrcoef(sls, slm)[0, 1]
            if not np.isnan(c):
                per_block_corrs.append(c)
    if per_block_corrs:
        pbc = np.array(per_block_corrs)
        print(f"  Per-block corr(ls, lm) ({len(pbc):,} blocks):")
        print(f"    p25={np.percentile(pbc, 25):.4f} p50={np.percentile(pbc, 50):.4f} "
              f"p75={np.percentile(pbc, 75):.4f}")
        print(f"    |r| > 0.5: {np.sum(np.abs(pbc)>0.5)/len(pbc)*100:.1f}%")
        print(f"    |r| > 0.8: {np.sum(np.abs(pbc)>0.8)/len(pbc)*100:.1f}%")

    # === Entropy ===
    print(f"\n--- Information Content ---")
    ls_probs = ls_counts / len(all_ls_arr)
    ls_entropy = -np.sum(ls_probs * np.log2(ls_probs))
    lm_unique, lm_counts_np = np.unique(all_lm_arr, return_counts=True)
    lm_probs = lm_counts_np / len(all_lm_arr)
    lm_entropy = -np.sum(lm_probs * np.log2(lm_probs))
    print(f"  ls Shannon entropy: {ls_entropy:.3f} bits (of 6.0 max)")
    print(f"  lm Shannon entropy: {lm_entropy:.3f} bits (of 6.0 max)")

    # Conditional entropy: H(ls_j | ls_mean) = how much uncertainty remains
    # after knowing the mean
    ls_mean_arr = np.array(block_ls_means)
    ls_mat_mean = np.repeat(ls_mean_arr, N_SUB).reshape(-1, N_SUB)
    ls_residual = ls_mat.astype(float) - ls_mat_mean
    ls_resid_std = np.std(ls_residual)
    print(f"  ls residual std (after subtracting block mean): {ls_resid_std:.2f} "
          f"(out of max range 63)")

    # === RECOMMENDATIONS ===
    print(f"\n{'='*72}")
    print(f"ESTIMATED BYTE SAVINGS PER BLOCK")
    print(f"{'='*72}")

    # Delta coverage stats
    ls_d4 = np.sum(np.abs(ls_deltas) <= 7) / len(ls_deltas) * 100
    ls_d3 = np.sum(np.abs(ls_deltas) <= 3) / len(ls_deltas) * 100
    ls_d2 = np.sum(np.abs(ls_deltas) <= 1) / len(ls_deltas) * 100

    lm_d3 = np.sum(np.abs(lm_deltas) <= 3) / len(lm_deltas) * 100
    lm_d2 = np.sum(np.abs(lm_deltas) <= 1) / len(lm_deltas) * 100

    print(f"\n  Delta encoding coverage:")
    print(f"    ls deltas: |d|<=1: {ls_d2:.1f}%, |d|<=3: {ls_d3:.1f}%, "
          f"|d|<=7: {ls_d4:.1f}%")
    print(f"    lm deltas: |d|<=1: {lm_d2:.1f}%, |d|<=3: {lm_d3:.1f}%")

    print(f"\n  Technique 1: Delta encoding (lossless or near-lossless)")
    print(f"    6b base_ls + 7×4b ls_signed_delta + 6b base_lm + 7×4b lm_signed_delta")
    print(f"    = 6+28+6+28 = 68 bits = 8.5 bytes → 9 bytes with alignment")
    print(f"    Coverage: ls={ls_d4:.1f}%, lm={lm_d3:.1f}%")
    print(f"    Saving: 12 → 9 bytes (3 bytes saved per block, 2.08%)")

    # If deltas fit in 3-bit
    if ls_d3 > 99:
        print(f"\n  Technique 1b: Tighter delta encoding")
        print(f"    6b base_ls + 7×3b ls_delta + 6b base_lm + 7×3b lm_delta")
        print(f"    = 6+21+6+21 = 54 bits = 6.75 bytes → 8 bytes")
        print(f"    Saving: 12 → 8 bytes (4 bytes, 2.78%)")

    # If deltas fit in 2-bit (highly uniform)
    if ls_d2 > 95:
        print(f"\n  Technique 1c: Extremely tight delta encoding")
        print(f"    6b base_ls + 7×2b ls_delta + 6b base_lm + 7×2b lm_delta")
        print(f"    = 6+14+6+14 = 40 bits = 5 bytes → 8 bytes")
        print(f"    Saving: 12 → 8 bytes (4 bytes, 2.78%) — same savings, lower complexity")

    ls_ok_5bit = np.sum(all_ls_arr <= 31) / len(all_ls_arr) * 100
    ls_ok_4bit = np.sum(all_ls_arr <= 15) / len(all_ls_arr) * 100

    print(f"\n  Technique 2: 5-bit scales")
    print(f"    sc 6→5 bits: {ls_ok_5bit:.2f}% of sub-blocks fit in 0..31")
    print(f"    {100-ls_ok_5bit:.2f}% get clipped — combined savings: ~2 bytes")
    print(f"    Quality risk: known from exp-023 (+1.5% KLD)")

    print(f"\n  Technique 3: 4-bit scales")
    print(f"    sc 6→4 bits: {ls_ok_4bit:.2f}% of sub-blocks fit in 0..15")
    print(f"    {100-ls_ok_4bit:.2f}% get clipped — significant quality risk")
    print(f"    Known from exp-024: +1.1% KLD on top of exp-023")

    # Check correlation with d
    print(f"\n  Technique 4: Use d to normalize scales (eliminate per-sub-block scale)")
    print(f"    ln(ls_mean) vs ln(d): R^2={corr**2:.4f}")
    print(f"    Only {corr**2*100:.1f}% of variance explained — NOT reliable")

    # === SUMMARY ===
    print(f"\n{'='*72}")
    print(f"CONCRETE RECOMMENDATIONS (ordered by feasibility)")
    print(f"{'='*72}")
    print(f"")
    print(f"  1. DELTA ENCODING (most promising)")
    ls_de = max(ls_d3, ls_d4)
    if ls_d2 > 95:
        print(f"     ls deltas: {ls_d2:.0f}% within |d|<=1 → 2-bit signed sufficient")
        print(f"     lm deltas: {lm_d2:.0f}% within |d|<=1 → 2-bit signed sufficient")
        print(f"     Encoded: 6b base + 7*2b deltas = 20b per group")
        print(f"     Total: 20+20 = 40 bits → 5 bytes → padded to 8 bytes")
        print(f"     Savings: 4 bytes/block = 2.78% per-block, ~1.15% overall")
        print(f"     Quality: Lossless (exact same ls/lm values)")
    else:
        print(f"     Savings: 3-4 bytes/block, quality preserved")
        print(f"     Must also change CUDA dequant to unpack new encoding")

    print(f"")
    print(f"  2. 5-BIT SCALE QUANTIZATION")
    print(f"     Save ~2 bytes/block ({ls_ok_5bit:.1f}% coverage)")
    print(f"     Known quality: +1.5% KLD (exp-023) — very small")
    print(f"     Scales[]: 12 → 10 bytes, block 144 → 142 bytes (padded to 144?")
    print(f"     PITFALL: 142-byte block pads to 144 due to alignment!")

    print(f"")
    print(f"  3. COMBINE Delta + 5-bit: encode 5-bit values with deltas")
    print(f"     5b base + 7×3b deltas (if |d|<=3 works) = 26b per group")
    print(f"     Total 52 bits → 7 bytes → padded to 8 bytes")
    print(f"     Combined savings: 4 bytes + quality from coarsening")

    print(f"")
    print(f"  4. SHARED SCALE APPROACH (for uniform blocks)")
    ls_range_2_frac = np.sum(ls_r <= 2) / len(ls_r) * 100
    print(f"     {ls_range_2_frac:.1f}% of blocks have ls range <= 2")
    print(f"     For these: store 1 shared scale + 4-way per-sub-block flags")
    print(f"     Hybrid: uniform blocks save max, non-uniform fall back to delta")

    print(f"")
    pbc_med = f"{np.median(pbc):.4f}" if per_block_corrs else 'N/A'
    print(f"  dmin ELIMINATION: NOT practical")
    print(f"     Only {np.sum(all_lm_arr==0)/len(all_lm_arr)*100:.1f}% lm==0")
    print(f"     Per-block corr(ls,lm) p50={pbc_med}")
    print(f"     Prior art: eliminating dmin cost 82% KLD (exp-003)")

if __name__ == '__main__':
    main()
