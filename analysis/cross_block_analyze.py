#!/usr/bin/env python3
"""Cross-block and inter-row pattern analysis for FWHT-transformed weights.

Efficient version — subsamples rows/blocks for large tensors.
"""
import struct, sys, numpy as np
from collections import defaultdict

FWHT_SIZE = 256

def fwht_256(x):
    step = 1
    while step < 256:
        for i in range(0, 256, 2 * step):
            for j in range(i, i + step):
                a, b = x[j], x[j + step]
                x[j], x[j + step] = a + b, a - b
        step <<= 1

def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8')

def rd_type(f):
    return struct.unpack('<I', f.read(4))[0]

def skip_value(f, vtype, n=1):
    sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:8, 9:0, 10:8, 11:8, 12:8}
    current_type, current_n = vtype, n
    while True:
        if current_type == 8:
            for _ in range(current_n):
                slen = struct.unpack('<Q', f.read(8))[0]
                f.read(slen)
        elif current_type == 9:
            elem_type = struct.unpack('<I', f.read(4))[0]
            elem_n = struct.unpack('<Q', f.read(8))[0]
            current_type, current_n = elem_type, elem_n
            continue
        else:
            f.read(sizes.get(current_type, 4) * current_n)
        break

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf'
    max_tensors = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_rows_per_tensor = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    max_blks_per_row = int(sys.argv[4]) if len(sys.argv) > 4 else 256

    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<q', f.read(8))[0]
    n_kv = struct.unpack('<q', f.read(8))[0]
    print(f"GGUF v{ver}, tensors={n_tensors}, kv={n_kv}")

    for _ in range(n_kv):
        rd_str(f)
        vtype = rd_type(f)
        skip_value(f, vtype)

    tensors = []
    for i in range(n_tensors):
        name = rd_str(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack(f'<{n_dims}q', f.read(8 * n_dims))
        ftype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        if max_tensors == 0 or len(tensors) < max_tensors:
            tensors.append((i, name, n_dims, dims, ftype, offset))

    print(f"Matching tensors: {len(tensors)}")

    # Accumulators
    adj_block_corr_fwht = []
    same_pos_corr = []
    d_ratio = []
    d_gradient = []
    fwht_pos_corrs = defaultdict(list)
    pred_mse_data = []
    base_mse_data = []
    row_rms_fwht_arr = []

    processed = 0

    for ti, (idx, name, n_dims, dims, ftype, offset) in enumerate(tensors):
        if ftype != 30 and ftype != 1:
            print(f"  SKIP {name}: ftype={ftype} (not BF16/F32)")
            continue
        if n_dims < 2:
            print(f"  SKIP {name}: n_dims={n_dims} < 2")
            continue

        ne0, ne1 = int(dims[1]), int(dims[0])  # rows=n_rows, cols=n_cols
        if ne0 % FWHT_SIZE != 0:
            print(f"  SKIP {name}: ne0={ne0} not multiple of {FWHT_SIZE}")
            continue

        n_elems = 1
        for d in dims:
            n_elems *= d

        f.seek(offset)
        if ftype == 30:
            raw = np.frombuffer(f.read(n_elems * 2), dtype=np.uint16).copy()
            f32 = np.zeros(n_elems, dtype=np.float32)
            f32.view(np.uint32)[:] = raw.astype(np.uint32, copy=False) << 16
            f32 = np.ascontiguousarray(f32)
            data = f32.reshape(dims)
        else:
            data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32).copy().reshape(dims)

        n_blocks_per_row = ne0 // FWHT_SIZE
        max_rows = min(ne1, max_rows_per_tensor)
        max_blks = min(n_blocks_per_row, max_blks_per_row)

        # Process subset of rows
        for row in range(max_rows):
            fwht_row = data[row, :ne0].astype(np.float64).copy()
            for blk in range(n_blocks_per_row):
                fwht_256(fwht_row[blk*FWHT_SIZE:(blk+1)*FWHT_SIZE])
            row_rms_fwht_arr.append(float(np.sqrt(np.mean(fwht_row**2))))

            if row < max_rows - 1 and n_blocks_per_row >= 2:
                # Process adjacent blocks for this row
                for blk in range(min(n_blocks_per_row - 1, max_blks)):
                    b0 = fwht_row[blk*FWHT_SIZE:(blk+1)*FWHT_SIZE]
                    b1 = fwht_row[(blk+1)*FWHT_SIZE:(blk+2)*FWHT_SIZE]
                    c = np.corrcoef(b0, b1)[0, 1]
                    if not np.isnan(c):
                        adj_block_corr_fwht.append(c)

            # For Q4 per-position analysis - sample a few blocks per row
            if row < 16 and n_blocks_per_row >= 2:
                for blk in range(min(n_blocks_per_row - 1, 64)):
                    b0 = fwht_row[blk*FWHT_SIZE:(blk+1)*FWHT_SIZE]
                    b1 = fwht_row[(blk+1)*FWHT_SIZE:(blk+2)*FWHT_SIZE]
                    for pos in range(FWHT_SIZE):
                        if len(fwht_pos_corrs.get(pos, [])) < 30000:
                            if pos not in fwht_pos_corrs:
                                fwht_pos_corrs[pos] = []
                            fwht_pos_corrs[pos].append((float(b0[pos]), float(b1[pos])))

        # Q2: Same-position cross-row
        if max_rows >= 2:
            for blk in range(min(n_blocks_per_row, 100)):
                for row in range(max_rows - 1):
                    b0 = np.zeros(FWHT_SIZE, np.float64)
                    b1 = np.zeros(FWHT_SIZE, np.float64)
                    fwht_row0 = data[row, blk*FWHT_SIZE:(blk+1)*FWHT_SIZE].astype(np.float64).copy()
                    fwht_256(fwht_row0)
                    fwht_row1 = data[row+1, blk*FWHT_SIZE:(blk+1)*FWHT_SIZE].astype(np.float64).copy()
                    fwht_256(fwht_row1)
                    c = np.corrcoef(fwht_row0, fwht_row1)[0, 1]
                    if not np.isnan(c):
                        same_pos_corr.append(c)

        # Q3: d/dmin smoothness (proxy: max abs per block)
        for row in range(max_rows):
            for blk in range(n_blocks_per_row):
                b = data[row, blk*FWHT_SIZE:(blk+1)*FWHT_SIZE].astype(np.float64).copy()
                fwht_256(b)
                block_max = float(np.max(np.abs(b)))
                # Store adjacent pairs later

        # Process block max values for adj comparison
        if n_blocks_per_row >= 2:
            for row in range(max_rows):
                prev_max = None
                for blk in range(min(n_blocks_per_row, max_blks)):
                    b = data[row, blk*FWHT_SIZE:(blk+1)*FWHT_SIZE].astype(np.float64).copy()
                    fwht_256(b)
                    cur_max = float(np.max(np.abs(b)))
                    if prev_max is not None and prev_max > 0 and cur_max > 0:
                        d_ratio.append(min(prev_max/cur_max, cur_max/prev_max))
                        d_gradient.append(abs(cur_max - prev_max) / prev_max)
                    prev_max = cur_max

        processed += 1
        print(f"  [{processed}] {name}: {ne1}x{ne0} -> {ne1 * n_blocks_per_row} blocks (sampled {max_rows} rows x {max_blks} blks)")

    f.close()

    print(f"\n{'='*72}")
    print(f"CROSS-BLOCK AND INTER-ROW ANALYSIS (FWHT Space)")
    print(f"{'='*72}")

    # ====== Q1 ======
    print(f"\n--- Q1: Adjacent Block Correlation (same row, FWHT space) ---")
    if adj_block_corr_fwht:
        corrs = np.array(adj_block_corr_fwht)
        print(f"  N={len(corrs):,}")
        print(f"  mean={corrs.mean():.6f}  median={np.median(corrs):.6f}  std={corrs.std():.6f}")
        print(f"  %ile: P10={np.percentile(corrs,10):.4f} P50={np.median(corrs):.4f} P90={np.percentile(corrs,90):.4f}")
        print(f"  |corr|>0.1: {100*np.mean(np.abs(corrs)>0.1):.1f}%  |corr|>0.2: {100*np.mean(np.abs(corrs)>0.2):.1f}%  |corr|>0.3: {100*np.mean(np.abs(corrs)>0.3):.1f}%")
        se = corrs.std() / np.sqrt(len(corrs))
        print(f"  t-stat: {corrs.mean()/se:.1f}")

    # ====== Q2 ======
    print(f"\n--- Q2: Same-Position Cross-Row Correlation (same column blk, diff rows) ---")
    if same_pos_corr:
        corrs = np.array(same_pos_corr)
        print(f"  N={len(corrs):,}")
        print(f"  mean={corrs.mean():.6f}  median={np.median(corrs):.6f}  std={corrs.std():.6f}")
        print(f"  %ile: P10={np.percentile(corrs,10):.4f} P50={np.median(corrs):.4f} P90={np.percentile(corrs,90):.4f}")
        print(f"  |corr|>0.1: {100*np.mean(np.abs(corrs)>0.1):.1f}%  |corr|>0.2: {100*np.mean(np.abs(corrs)>0.2):.1f}%  |corr|>0.3: {100*np.mean(np.abs(corrs)>0.3):.1f}%")

    # ====== Q3 ======
    print(f"\n--- Q3: Block Magnitude (proxy for d) Smoothness ---")
    if d_gradient:
        grads = np.array(d_gradient)
        ratios = np.array(d_ratio)
        print(f"  N={len(grads):,}")
        print(f"  d ratio (adjacent): mean={ratios.mean():.4f} median={np.median(ratios):.4f}")
        print(f"  |Δd|/d (adjacent):  mean={grads.mean():.4f} median={np.median(grads):.4f}")
        print(f"  %ile: P50={np.percentile(grads,50):.4f} P75={np.percentile(grads,75):.4f} P90={np.percentile(grads,90):.4f}")
        for pct in [5, 10, 20]:
            print(f"  |Δd|/d < {pct}%: {100*np.mean(grads < pct/100):.1f}%")

    # ====== Q4 ======
    print(f"\n--- Q4: Per-Position Cross-Block Value Correlation (FWHT space) ---")
    if fwht_pos_corrs:
        pos_corrs = {}
        for pos in range(FWHT_SIZE):
            if pos in fwht_pos_corrs and len(fwht_pos_corrs[pos]) > 10:
                pairs = np.array(fwht_pos_corrs[pos])
                c = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
                if not np.isnan(c):
                    pos_corrs[pos] = c

        corr_values = np.array(list(pos_corrs.values()))
        corr_keys = np.array(list(pos_corrs.keys()))
        print(f"  Positions with data: {len(pos_corrs)}")
        print(f"  Cross-block per-pos correlation: mean={corr_values.mean():.6f} median={np.median(corr_values):.6f} std={corr_values.std():.6f}")
        print(f"  |corr|>0.1: {100*np.mean(np.abs(corr_values)>0.1):.1f}%  |corr|>0.2: {100*np.mean(np.abs(corr_values)>0.2):.1f}%  |corr|>0.3: {100*np.mean(np.abs(corr_values)>0.3):.1f}%")

        # Per-position linear prediction R²
        for pos in range(FWHT_SIZE):
            if pos in fwht_pos_corrs and len(fwht_pos_corrs[pos]) > 10:
                pairs = np.array(fwht_pos_corrs[pos])
                x, y = pairs[:, 0], pairs[:, 1]
                base_mse_data.append(np.mean((y - y.mean())**2))
                if np.var(x) > 1e-30:
                    slope = np.cov(x, y)[0, 1] / np.var(x)
                    intercept = y.mean() - slope * x.mean()
                    pred = slope * x + intercept
                    pred_mse_data.append(np.mean((y - pred)**2))

        if len(pred_mse_data) > 0:
            pm = np.array(pred_mse_data)
            bm = np.array(base_mse_data)
            r2 = 1 - pm.mean() / bm.mean()
            print(f"  Linear prediction R² (adjacent-block same-position): {r2:.6f}")

    # ====== Q5 ======
    print(f"\n--- Q5: Row-Level Magnitude Patterns (FWHT) ---")
    if row_rms_fwht_arr:
        rms = np.array(row_rms_fwht_arr)
        print(f"  N rows={len(rms):,}")
        print(f"  Row RMS (FWHT): mean={rms.mean():.6f} std={rms.std():.6f}")
        cv = rms.std() / rms.mean()
        print(f"  Coef of variation: {cv:.4f}")
        print(f"  Ratio max/min: {rms.max()/rms.min():.2f}x")
        print(f"  %ile: P10={np.percentile(rms,10):.4f} P50={np.median(rms):.4f} P90={np.percentile(rms,90):.4f}")

    # ====== TECHNIQUE PROPOSALS ======
    print(f"\n{'='*72}")
    print(f"TECHNIQUE PROPOSALS WITH ESTIMATED BYTE SAVINGS")
    print(f"{'='*72}")

    # T1: d/dmin sharing
    print(f"\n## T1: d/dmin sharing between adjacent blocks")
    if d_gradient:
        grads = np.array(d_gradient)
        print(f"  Block max gradient: median={np.median(grads):.4f}, P90={np.percentile(grads,90):.4f}")
        print(f"  d+dmin is 4 bytes per 144-byte block (2.78%)")
        for N in [2, 4, 8]:
            bytes_per_N = N * 144
            saved = 4 * (N - 1)  # save N-1 pairs of d+dmin
            pct = saved / bytes_per_N * 100
            est_mb = saved / 144 * (280e6 / 144) / (1024**2) * 144  # ~280MB of Q4_K_CLONE data
            est_mb2 = (280 * saved / bytes_per_N)
            frac_ok = np.mean(grads < 0.10)
            print(f"  N={N}: save {saved}/{bytes_per_N}B ({pct:.1f}%), ~{est_mb2:.1f} MB on Q4_K_CLONE data, {frac_ok*100:.1f}% blocks within 10% gradient")
    else:
        print(f"  No d/dmin gradient data")

    # T2: Differential qs[]
    print(f"\n## T2: Differential qs[] between adjacent blocks")
    if adj_block_corr_fwht:
        c_mean = np.array(adj_block_corr_fwht).mean()
        var_red = c_mean ** 2
        bit_saved = -0.5 * np.log2(max(0.0001, 1 - var_red))
        print(f"  Adjacent block FWHT correlation: {c_mean:.6f}")
        print(f"  Variance reduction via differential: {var_red*100:.3f}%")
        print(f"  Entropy saving: {bit_saved:.3f} bits per 4-bit element")
        print(f"  On 128B qs[]: saves ~{bit_saved * 256:.1f} bits = {bit_saved * 256 / 8:.1f} bytes per block ({bit_saved*256/8/144*100:.2f}%)")
    else:
        print(f"  No adjacent block correlation data")

    # T3: Per-row scale predictor
    print(f"\n## T3: Per-row scale predictor (row-level d/dmin table)")
    if row_rms_fwht_arr:
        rms = np.array(row_rms_fwht_arr)
        cv = rms.std() / rms.mean()
        print(f"  Row RMS CV: {cv:.4f}")
        if cv > 0.03:
            bits_per_row = 8  # fp8 row scale
            rows = 1024  # typical hidden dim
            blocks_per_row = 3584 // 256  # typical FFN input dim
            total_blocks = rows * blocks_per_row
            current_cost = 4 * total_blocks  # 4 bytes d+dmin per block
            new_cost = bits_per_row/8 * rows + 4 * total_blocks  # per-row scale + block d+dmin
            saved = current_cost - new_cost
            print(f"  Example (1024x3584 weight): current={current_cost}B d+dmin, new={new_cost:.0f}B")
            print(f"  BUT: d+dmin is still needed per block if row variation is CV={cv:.4f}")
            print(f"  Row predictor saves (d,dmin) per block only if block-to-row scaling captures ALL variation")
            print(f"  STATUS: CV={cv:.4f} is {'sufficient' if cv > 0.1 else 'insufficient'} for per-row-only scaling")
        else:
            print(f"  Row-level variation too small (CV={cv:.4f}) — per-row prediction saves nothing")
    else:
        print(f"  No row RMS data")

    # T4: Cross-row block codebook
    print(f"\n## T4: Cross-row block codebook (same column blk, different rows)")
    if same_pos_corr:
        c_mean = np.array(same_pos_corr).mean()
        print(f"  Same-position cross-row correlation: {c_mean:.6f}")
        if abs(c_mean) < 0.05:
            print(f"  STATUS: Essentially uncorrelated — codebook sharing would NOT help")
            print(f"  After FWHT, same column-position blocks are independent across rows")
        elif abs(c_mean) < 0.2:
            print(f"  STATUS: Very weak correlation — codebook savings < {c_mean**2*100:.1f}%")
            print(f"  Not worth the dequant overhead of codebook lookup")
        else:
            print(f"  STATUS: Meaningful correlation — codebook approach viable")
            print(f"  Savings: {c_mean**2*100:.1f}% of qs[] entropy ≈ {c_mean**2*128:.1f} bytes/block")
    else:
        print(f"  No cross-row correlation data")

    # T5: Per-position prediction
    print(f"\n## T5: Per-position differential (position i in block r predicts position i in block r+1)")
    if len(pred_mse_data) > 0:
        pm = np.array(pred_mse_data)
        bm = np.array(base_mse_data)
        r2 = 1 - pm.mean() / bm.mean()
        print(f"  R² of adjacent-block same-position prediction: {r2:.6f}")
        if r2 < 0.01:
            print(f"  STATUS: No predictive power — per-position values are independent across blocks")
            print(f"  This is expected: independent FWHT on each block decorrelates per-position values")
        elif r2 < 0.1:
            print(f"  STATUS: Weak signal ({r2*100:.1f}% var) — differential encoding saves < {r2*0.5:.3f} bpw")
        else:
            print(f"  STATUS: Meaningful prediction — save ~{-0.5*np.log2(1-r2):.2f} bpw via differential")
    else:
        print(f"  No per-position prediction data")

    # Overall
    print(f"\n{'='*72}")
    print(f"OVERALL ASSESSMENT")
    print(f"{'='*72}")
    if adj_block_corr_fwht:
        c = np.array(adj_block_corr_fwht).mean()
        print(f"Correlation between adjacent FWHT blocks: {c:.6f}")
        if abs(c) < 0.05:
            print(f"=> Adjacent blocks are essentially uncorrelated after FWHT")
            print(f"=> Inter-block compression techniques will NOT work after FWHT")
            print(f"=> To exploit inter-block redundancy, operate BEFORE FWHT")
            print(f"=> The only cross-block signal is d/dmin (block energy) which varies slowly")
    if same_pos_corr:
        c = np.array(same_pos_corr).mean()
        print(f"Correlation same-position cross-row: {c:.6f}")
        if abs(c) < 0.05:
            print(f"=> Cross-row blocks are independent — SqueezeLLM-style sharing won't work")


if __name__ == '__main__':
    main()
