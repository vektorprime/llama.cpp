#!/usr/bin/env python3
"""FWHT weight analysis — reads BF16 GGUF, applies FWHT, prints statistics."""
import struct, sys, numpy as np

FWHT_SIZE = 256

def fwht_256(x):
    step = 1
    while step < 256:
        for i in range(0, 256, 2 * step):
            for j in range(i, i + step):
                a = x[j]
                b = x[j + step]
                x[j] = a + b
                x[j + step] = a - b
        step <<= 1

def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8')

def rd_type(f):
    t = struct.unpack('<I', f.read(4))[0]
    return t

def skip_value(f, vtype, n=1):
    sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:8, 9:0, 10:8, 11:8, 12:8}
    # Handle arrays: read element type + count, then skip elements
    current_type = vtype
    current_n = n

    while True:
        if current_type == 8:  # string
            for _ in range(current_n):
                slen = struct.unpack('<Q', f.read(8))[0]
                f.read(slen)
        elif current_type == 9:  # array
            elem_type = struct.unpack('<I', f.read(4))[0]
            elem_n = struct.unpack('<Q', f.read(8))[0]
            current_type = elem_type
            current_n = elem_n
            continue  # loop to handle the inner type
        else:
            f.read(sizes.get(current_type, 4) * current_n)
        break

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/home/user/llm/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf'
    filt = sys.argv[2] if len(sys.argv) > 2 else 'blk'
    max_tensors = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver = struct.unpack('<I', f.read(4))[0]
    print(f"GGUF v{ver}")
    n_tensors = struct.unpack('<q', f.read(8))[0]
    n_kv = struct.unpack('<q', f.read(8))[0]
    print(f"tensors={n_tensors} kv={n_kv}")

    # Skip KV pairs
    for _ in range(n_kv):
        rd_str(f)  # key (string = length + data)
        vtype = rd_type(f)  # type
        skip_value(f, vtype)  # value

    # Read tensor infos
    tensors = []
    for i in range(n_tensors):
        name = rd_str(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack(f'<{n_dims}q', f.read(8 * n_dims))
        ftype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        if filt in name:
            tensors.append((i, name, n_dims, dims, ftype, offset))

    print(f"Matching tensors: {len(tensors)}")

    # Accumulators
    pos_sum = np.zeros(FWHT_SIZE, np.float64)
    pos_sum2 = np.zeros(FWHT_SIZE, np.float64)
    pos_min = np.full(FWHT_SIZE, np.inf, np.float32)
    pos_max = np.full(FWHT_SIZE, -np.inf, np.float32)
    pos_count = 0

    block_magnitudes = []  # mean abs per block
    block_stddevs = []     # std per block

    processed = 0
    for ti, (idx, name, n_dims, dims, ftype, offset) in enumerate(tensors):
        if max_tensors and processed >= max_tensors:
            break
        if ftype != 30 and ftype != 1:  # BF16 or F32
            continue
        if n_dims < 2:
            continue

        ne0, ne1 = dims[1], dims[0]  # rows, cols
        if ne0 % FWHT_SIZE != 0:
            continue

        n_elems = 1
        for d in dims:
            n_elems *= d

        # Read tensor data
        f.seek(offset)
        if ftype == 30:  # BF16
            raw = np.frombuffer(f.read(n_elems * 2), dtype=np.uint16)
            f32 = np.zeros(n_elems, dtype=np.float32)
            f32.view(np.uint32)[:] = raw.astype(np.uint32) << 16
            data = f32.reshape(dims)
        else:
            data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32).reshape(dims)

        n_blocks_per_row = ne0 // FWHT_SIZE

        for row in range(ne1):
            for blk in range(n_blocks_per_row):
                col = blk * FWHT_SIZE
                block = data[row, col:col+FWHT_SIZE].copy().astype(np.float64)
                fwht_256(block)

                pos_sum += block
                pos_sum2 += block * block
                pos_min = np.minimum(pos_min, block.astype(np.float32))
                pos_max = np.maximum(pos_max, block.astype(np.float32))
                pos_count += 1

                block_magnitudes.append(float(np.mean(np.abs(block))))
                block_stddevs.append(float(np.std(block)))

        processed += 1
        print(f"  [{processed}] {name}: {ne1}x{ne0} -> {ne1 * n_blocks_per_row} blocks")

    f.close()

    pos_mean = pos_sum / pos_count
    pos_var = pos_sum2 / pos_count - pos_mean * pos_mean
    pos_std = np.sqrt(np.maximum(0, pos_var))

    print(f"\n{'='*72}")
    print(f"FWHT WEIGHT ANALYSIS")
    print(f"{'='*72}")
    print(f"Blocks: {pos_count:,}")

    # Global stats
    print(f"\n--- Global ---")
    print(f"mean={pos_mean.mean():.6f} std={np.sqrt(pos_var.mean()):.6f}")
    print(f"range=[{pos_min.min():.4f}, {pos_max.max():.4f}]")

    # DC vs AC
    print(f"\n--- DC (pos 0) vs AC (pos 1-255) ---")
    dc = pos_mean[0]
    dc_s = pos_std[0]
    ac_mean = pos_mean[1:].mean()
    ac_var_avg = pos_var[1:].mean()
    print(f"DC: mean={dc:.6f}  std={dc_s:.6f}  var={pos_var[0]:.6f}")
    print(f"AC: mean={ac_mean:.6f}  std avg={np.sqrt(ac_var_avg):.6f}  var avg={ac_var_avg:.6f}")
    print(f"DC var / AC var avg = {pos_var[0]/ac_var_avg:.2f}x")

    dc_var_frac = pos_var[0] / pos_var.sum()
    print(f"DC fraction of total variance: {dc_var_frac*100:.1f}%  (just 1/256 = 0.4% of positions)")

    # Top/bottom variance positions
    print(f"\n--- Per-Position Variance ---")
    var_rank = np.argsort(-pos_var)
    print(f"Top 10 highest-variance positions:")
    for r in range(10):
        i = var_rank[r]
        print(f"  pos {i:3d}: mean={pos_mean[i]:10.6f} std={pos_std[i]:10.6f} var={pos_var[i]:10.6f}")

    print(f"Bottom 10 lowest-variance positions:")
    for r in range(256-10, 256):
        i = var_rank[r]
        print(f"  pos {i:3d}: mean={pos_mean[i]:10.6f} std={pos_std[i]:10.6f} var={pos_var[i]:10.6f}")

    var_ratio = pos_var.max() / pos_var.min()
    print(f"\nVariance ratio (max/min): {var_ratio:.2f}x")

    # Magnitude and std distribution
    mags = np.array(block_magnitudes)
    stds = np.array(block_stddevs)
    print(f"\n--- Per-Block Magnitudes ---")
    print(f"  |mean|: {np.percentile(mags, [10,25,50,75,90])}")
    print(f"  block std: {np.percentile(stds, [10,25,50,75,90])}")

    # Quantization error
    print(f"\n--- Quantization Error Simulation ---")
    gmean = pos_mean.mean()
    gstd = np.sqrt(pos_var.mean())
    ref = np.random.randn(10000) * gstd + gmean
    for b in [2,3,4,5,6,8]:
        n = 2**b
        q = np.clip(np.round(ref * n/2 / (4*gstd)), -n/2, n/2-1)
        dq = q * (4*gstd) / (n/2)
        mse = np.mean((ref - dq)**2)
        print(f"  {b}-bit uniform: MSE={mse:.2e}")

    # Compression ideas
    print(f"\n--- Compression Assessment ---")
    if dc_var_frac > 0.05:
        print(f"  DC carries {dc_var_frac*100:.1f}% var -> separate encoding promising")
    if var_ratio > 3:
        print(f"  Var ratio {var_ratio:.1f}x -> uneven bit allocation viable")
    if var_rank.tolist() == list(range(FWHT_SIZE)):
        print(f"  Variance follows order 0,1,2,... -> FWHT spectral ordering holds")
    else:
        print(f"  Variance order NOT purely spectral -> look at per-position patterns")


if __name__ == '__main__':
    main()
