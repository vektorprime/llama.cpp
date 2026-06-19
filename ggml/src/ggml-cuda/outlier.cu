#include "outlier.cuh"
#include "common.cuh"
#include "convert.cuh"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// Kernel: sparse BF16 outlier block correction for Q8_0_BF16_OUTLIER
//
// For each outlier block, compute:
//   out[row, token] += sum_{j=0..31} values[j, block] * x[col0 + j, token]
//
// Grid dimensions:
//   blockIdx.x: outlier block index (0..n_blocks-1)
//   blockIdx.y: token index (0..n_tokens-1)
//
// Each thread in the block handles one element of the 32-element dot product,
// then the thread block does a warp reduction to get the final sum.

#define OUTLIER_BLOCK_SIZE 32

// Nibble-diff decoding: 4 bits per weight, 16 bytes = 32 weights per block
// Bit layout: bit3=enable, bit2=sign(1=pos), bit1=zero_cnt, bit0=digit(0=1,1=2)
static __device__ inline float nibble_diff_decode(uint8_t nibble) {
    if (!(nibble & 0x08)) return 0.0f;
    float v = (nibble & 0x02) ? 0.001f : 0.01f;
    if (nibble & 0x01) v *= 2.0f;
    if (!(nibble & 0x04)) v = -v;
    return v;
}

// Q2_K nibble-diff decoding: same bit layout, different value table
// Bits 1-0: 00=0.002, 01=0.005, 10=0.02, 11=0.05
static __device__ inline float nibble_diff_decode_q2k(uint8_t nibble) {
    if (!(nibble & 0x08)) return 0.0f;
    float v;
    switch (nibble & 0x03) {
        case 0: v = 0.002f; break;
        case 1: v = 0.005f; break;
        case 2: v = 0.02f;  break;
        case 3: v = 0.05f;  break;
        default: v = 0.0f; break;
    }
    if (!(nibble & 0x04)) v = -v;
    return v;
}

static __global__ void outlier_blocks_kernel_nibble(
        const int32_t * __restrict__ idx,       // [2, n_blocks]
        const uint8_t * __restrict__ values,    // [16, n_blocks] packed nibbles
        const float *   __restrict__ x,         // [n_cols_x, n_tokens] shard-local
        float *         __restrict__ dst,       // [n_rows_out, n_tokens]
        const int64_t n_blocks,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens,
        const int32_t value_type) {             // LLAMA_OUTLIER_VALUE_TYPE_*

    const int64_t block_idx = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (block_idx >= n_blocks || token_idx >= n_tokens) {
        return;
    }

    const int32_t row       = idx[block_idx * 2];
    const int32_t block_col = idx[block_idx * 2 + 1];
    const int64_t col0      = (int64_t) block_col * 32;

    if (col0 + 31 < col_offset || col0 >= col_offset + n_cols_x) {
        return;
    }

    if (col0 + 31 >= n_cols_all) {
        return;
    }

    const int tid = threadIdx.x;
    float sum = 0.0f;

    if (tid < 32) {
        const int64_t j = tid;
        const int64_t col_global = col0 + j;
        if (col_global >= col_offset && col_global < col_offset + n_cols_x) {
            const uint8_t byte = values[block_idx * 16 + (j >> 1)];
            const uint8_t nibble = (j & 1) ? (byte >> 4) : (byte & 0x0F);
            const float w = (value_type == 3) ? nibble_diff_decode_q2k(nibble) : nibble_diff_decode(nibble);
            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token_idx * x_stride];
            sum = w * a;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0 && row >= 0 && row < n_rows_out) {
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

// Single-outlier kernel: 3 bytes per block (2 BF16 delta + 1 u8 position)
static __global__ void outlier_blocks_kernel_single(
        const int32_t * __restrict__ idx,       // [2, n_blocks]
        const uint8_t *  __restrict__ values,   // [3, n_blocks] packed
        const float *    __restrict__ x,        // [n_cols_x, n_tokens] shard-local
        float *          __restrict__ dst,      // [n_rows_out, n_tokens]
        const int64_t n_blocks,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    const int64_t block_idx = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (block_idx >= n_blocks || token_idx >= n_tokens) return;

    const int32_t row       = idx[block_idx * 2];
    const int32_t block_col = idx[block_idx * 2 + 1];

    // Read packed values: 2 bytes BF16 delta + 1 byte position
    const uint8_t * p = values + block_idx * 3;
    nv_bfloat16 bf16;
    bf16.__x = ((uint16_t)p[1] << 8) | p[0];
    float delta = __bfloat162float(bf16);
    int   pos   = (int)p[2];

    const int64_t col_global = (int64_t)block_col * 32 + pos;

    if (col_global >= col_offset && col_global < col_offset + n_cols_x &&
        col_global < n_cols_all &&
        row >= 0 && row < n_rows_out) {
        const int64_t col_local = col_global - col_offset;
        float sum = delta * x[col_local + token_idx * x_stride];
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

static __global__ void outlier_blocks_kernel_bf16(
        const int32_t *     __restrict__ idx,       // [2, n_blocks]
        const nv_bfloat16 * __restrict__ values,    // [32, n_blocks] in BF16
        const float *       __restrict__ x,         // [n_cols_x, n_tokens] shard-local
        float *             __restrict__ dst,       // [n_rows_out, n_tokens]
        const int64_t n_blocks,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    const int64_t block_idx = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (block_idx >= n_blocks || token_idx >= n_tokens) {
        return;
    }

    const int32_t row       = idx[block_idx * 2];
    const int32_t block_col = idx[block_idx * 2 + 1];
    const int64_t col0      = (int64_t) block_col * 32;

    if (col0 + 31 < col_offset || col0 >= col_offset + n_cols_x) {
        return;
    }

    if (col0 + 31 >= n_cols_all) {
        return;
    }

    const int tid = threadIdx.x;
    float sum = 0.0f;

    if (tid < 32) {
        const int64_t j = tid;
        const int64_t col_global = col0 + j;
        if (col_global >= col_offset && col_global < col_offset + n_cols_x) {
            const float w = __bfloat162float(values[block_idx * 32 + j]);
            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token_idx * x_stride];
            sum = w * a;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0 && row >= 0 && row < n_rows_out) {
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

// Q8_0 block layout: ggml_half d (2 bytes) + int8_t qs[32] (32 bytes) = 34 bytes
typedef struct {
    uint16_t d;
    char qs[32];
} block_q8_0_cuda;

static __global__ void outlier_blocks_kernel_q8_0(
        const int32_t *          __restrict__ idx,       // [2, n_blocks]
        const block_q8_0_cuda *  __restrict__ values,    // [n_blocks] Q8_0 blocks
        const float *            __restrict__ x,
        float *                  __restrict__ dst,
        const int64_t n_blocks,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    const int64_t block_idx = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (block_idx >= n_blocks || token_idx >= n_tokens) {
        return;
    }

    const int32_t row       = idx[block_idx * 2];
    const int32_t block_col = idx[block_idx * 2 + 1];
    const int64_t col0      = (int64_t) block_col * 32;

    if (col0 + 31 < col_offset || col0 >= col_offset + n_cols_x) {
        return;
    }

    if (col0 + 31 >= n_cols_all) {
        return;
    }

    const block_q8_0_cuda * q8block = &values[block_idx];
    const float d = __half2float(__ushort_as_half(q8block->d));
    const int tid = threadIdx.x;
    float sum = 0.0f;

    if (tid < 32) {
        const int64_t j = tid;
        const int64_t col_global = col0 + j;
        if (col_global >= col_offset && col_global < col_offset + n_cols_x) {
            const float w = d * (float)q8block->qs[j];
            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token_idx * x_stride];
            sum = w * a;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0 && row >= 0 && row < n_rows_out) {
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

static void outlier_blocks_cuda(
        ggml_backend_cuda_context & ctx,
        const int32_t *     idx_d,
        const void *        values_d,
        const float *       x_d,
        float *             dst_d,
        const int64_t n_blocks,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens,
        enum ggml_type values_type,
        int32_t value_type_i32) {

    cudaStream_t stream = ctx.stream();

    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, n_rows_out * n_tokens * sizeof(float), stream));

    if (n_blocks == 0 || n_tokens == 0 || n_cols_x == 0) {
        return;
    }

    dim3 block_dim(OUTLIER_BLOCK_SIZE, 1, 1);
    dim3 grid_dim((uint32_t) n_blocks, (uint32_t) n_tokens, 1);

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-cuda] kernel launch: grid=(%u,%u,%u) block=(%u,%u,%u) values_type=%s\n",
                (unsigned)grid_dim.x, (unsigned)grid_dim.y, (unsigned)grid_dim.z,
                (unsigned)block_dim.x, (unsigned)block_dim.y, (unsigned)block_dim.z,
                values_type == GGML_TYPE_Q8_0 ? "Q8_0" : values_type == GGML_TYPE_I8 ?
                (value_type_i32 == 4 ? "I8_single" : "I8_nibble") : "BF16");
        fflush(stderr);
    }

    if (values_type == GGML_TYPE_Q8_0) {
        const block_q8_0_cuda * values_q8 = (const block_q8_0_cuda *) values_d;
        outlier_blocks_kernel_q8_0<<<grid_dim, block_dim, 0, stream>>>(
                idx_d, values_q8, x_d, dst_d,
                n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);
    } else if (values_type == GGML_TYPE_I8 && value_type_i32 == 4) {
        const uint8_t * values_single = (const uint8_t *) values_d;
        dim3 single_block_dim(1, 1, 1);
        outlier_blocks_kernel_single<<<grid_dim, single_block_dim, 0, stream>>>(
                idx_d, values_single, x_d, dst_d,
                n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);
    } else if (values_type == GGML_TYPE_I8) {
        const uint8_t * values_nibble = (const uint8_t *) values_d;
        outlier_blocks_kernel_nibble<<<grid_dim, block_dim, 0, stream>>>(
                idx_d, values_nibble, x_d, dst_d,
                n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens,
                value_type_i32);
    } else {
        const nv_bfloat16 * values_bf16 = (const nv_bfloat16 *) values_d;
        outlier_blocks_kernel_bf16<<<grid_dim, block_dim, 0, stream>>>(
                idx_d, values_bf16, x_d, dst_d,
                n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);
    }

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_mul_mat_outlier_blocks(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * idx    = dst->src[0];
    const ggml_tensor * values = dst->src[1];
    const ggml_tensor * x      = dst->src[2];

    GGML_ASSERT(idx->type    == GGML_TYPE_I32);
    GGML_ASSERT(values->type == GGML_TYPE_BF16 || values->type == GGML_TYPE_Q8_0 || values->type == GGML_TYPE_I8);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);

    const int64_t n_rows_out = ggml_get_op_params_i32(dst, 0);
    const int64_t n_cols_all = ggml_get_op_params_i32(dst, 1);
    const int32_t value_type_i32 = ggml_get_op_params_i32(dst, 2);
    const int64_t n_blocks   = idx->ne[1];
    const int64_t n_tokens   = x->ne[1];
    const int64_t n_cols_x   = x->ne[0];
    const int64_t x_stride   = x->nb[1] / (int64_t)sizeof(float);

    int64_t col_offset = 0;
    if (x->view_src && x->view_offs > 0) {
        col_offset = x->view_offs / (int64_t)sizeof(float);
    } else if (n_cols_x < n_cols_all) {
        col_offset = ctx.device * n_cols_x;
    }

    GGML_ASSERT(dst->ne[0] == n_rows_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(n_cols_x    <= n_cols_all);
    GGML_ASSERT(idx->ne[0]  == 2);
    GGML_ASSERT(values->ne[0] == 32 || (values->type == GGML_TYPE_I8 && (values->ne[0] == 16 || values->ne[0] == 3)));

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-cuda] enter: n_blocks=%lld n_rows_out=%lld n_tokens=%lld n_cols_all=%lld n_cols_x=%lld col_offset=%lld values_type=%s\n",
                (long long)n_blocks, (long long)n_rows_out, (long long)n_tokens,
                (long long)n_cols_all, (long long)n_cols_x, (long long)col_offset,
                values->type == GGML_TYPE_Q8_0 ? "Q8_0" : "BF16");
        fflush(stderr);
    }

    const int32_t * idx_d = (const int32_t *) idx->data;
    const void *    values_d = values->data;
    const float *   x_d      = (const float *) x->data;
    float *         dst_d    = (float *) dst->data;

    outlier_blocks_cuda(ctx, idx_d, values_d, x_d, dst_d,
            n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens,
            values->type, value_type_i32);

    if (ggml_custom_logs_enabled() && n_blocks > 0) {
        int sample_count = n_blocks < 5 ? (int)n_blocks : 5;
        std::vector<int32_t> idx_sample(sample_count * 2);
        CUDA_CHECK(cudaMemcpy(idx_sample.data(), idx_d, sample_count * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost));
        int skip_col_shard = 0, skip_col_global = 0, skip_row = 0, valid = 0;
        for (int i = 0; i < sample_count; i++) {
            int32_t row = idx_sample[i * 2];
            int32_t bcol = idx_sample[i * 2 + 1];
            int64_t col0 = (int64_t)bcol * 32;
            if (row < 0 || row >= n_rows_out) { skip_row++; continue; }
            if (col0 + 31 < col_offset || col0 >= col_offset + n_cols_x) { skip_col_shard++; continue; }
            if (col0 + 31 >= n_cols_all) { skip_col_global++; continue; }
            valid++;
        }
        fprintf(stderr, "[delta-cuda] idx_sample(%d/%lld): valid=%d skip_col_shard=%d skip_col_global=%d skip_row=%d | first: ",
                sample_count, (long long)n_blocks, valid, skip_col_shard, skip_col_global, skip_row);
        for (int i = 0; i < sample_count; i++) {
            fprintf(stderr, "(r=%d,bc=%d) ", idx_sample[i*2], idx_sample[i*2+1]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    if (ggml_custom_logs_enabled()) {
        {
            int64_t col_n = n_rows_out;
            if (col_n > 0 && n_tokens > 0) {
                std::vector<float> col(col_n, 0);
                CUDA_CHECK(cudaMemcpy(col.data(), dst_d, col_n * sizeof(float), cudaMemcpyDeviceToHost));
                double l2 = 0.0;
                int nz = 0;
                for (int64_t i = 0; i < col_n; i++) {
                    l2 += col[i] * col[i];
                    if (col[i] != 0.0f) nz++;
                }
                fprintf(stderr, "[delta-cuda] output L2=%.6e non_zero_rows=%d/%lld\n",
                        sqrt(l2), nz, (long long)col_n);
                fflush(stderr);
            }
        }
    }
}

// ============================================================================
// Merged contiguous outlier block kernels (Proposal 4.1)
// Each merged run can process multiple contiguous 32-weight blocks in one
// kernel launch, reducing grid size from n_blocks to n_merged_runs.
// ============================================================================

// Helper: next power of 2 (for thread block sizing)
static __device__ __host__ inline int32_t next_pow2(int32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

static __global__ void outlier_blocks_kernel_nibble_merged(
        const int32_t * __restrict__ merged_idx, // [4, n_merged_runs]: row, start_bc, count, vals_start
        const uint8_t * __restrict__ values,     // [16, n_blocks] packed nibbles
        const float *   __restrict__ x,          // [n_cols_x, n_tokens] shard-local
        float *         __restrict__ dst,        // [n_rows_out, n_tokens]
        const int64_t n_merged_runs,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens,
        const int32_t value_type) {              // LLAMA_OUTLIER_VALUE_TYPE_*

    const int64_t run_idx  = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (run_idx >= n_merged_runs || token_idx >= n_tokens) {
        return;
    }

    const int32_t row            = merged_idx[run_idx * 4];
    const int32_t start_block_col = merged_idx[run_idx * 4 + 1];
    const int32_t count          = merged_idx[run_idx * 4 + 2];
    const int32_t values_start   = merged_idx[run_idx * 4 + 3];

    const int32_t total_elems = count * 32;

    // Bounds check: skip if entire run is outside the activation shard
    const int64_t run_col0 = (int64_t)start_block_col * 32;
    const int64_t run_col_last = run_col0 + (int64_t)total_elems - 1;
    if (run_col_last < col_offset || run_col0 >= col_offset + n_cols_x) {
        return;
    }
    if (run_col_last >= n_cols_all) {
        return;
    }

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float sum = 0.0f;

    // Each thread handles elements_per_thread weight elements
    for (int32_t j = tid; j < total_elems; j += block_size) {
        const int32_t block_idx_in_run = j >> 5;       // j / 32
        const int32_t elem_in_block    = j & 31;        // j % 32
        const int32_t orig_block       = values_start + block_idx_in_run;
        const int64_t col_global       = (int64_t)(start_block_col + block_idx_in_run) * 32 + elem_in_block;

        if (col_global >= col_offset && col_global < col_offset + n_cols_x) {
            const uint8_t byte = values[orig_block * 16 + (elem_in_block >> 1)];
            const uint8_t nibble = (elem_in_block & 1) ? (byte >> 4) : (byte & 0x0F);
            const float w = (value_type == 3) ? nibble_diff_decode_q2k(nibble) : nibble_diff_decode(nibble);
            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token_idx * x_stride];
            sum += w * a;
        }
    }

    // Warp reduce
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if ((tid & 31) == 0 && row >= 0 && row < n_rows_out) {
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

static __global__ void outlier_blocks_kernel_bf16_merged(
        const int32_t *     __restrict__ merged_idx,
        const nv_bfloat16 * __restrict__ values,
        const float *       __restrict__ x,
        float *             __restrict__ dst,
        const int64_t n_merged_runs,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    const int64_t run_idx  = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (run_idx >= n_merged_runs || token_idx >= n_tokens) {
        return;
    }

    const int32_t row            = merged_idx[run_idx * 4];
    const int32_t start_block_col = merged_idx[run_idx * 4 + 1];
    const int32_t count          = merged_idx[run_idx * 4 + 2];
    const int32_t values_start   = merged_idx[run_idx * 4 + 3];

    const int32_t total_elems = count * 32;

    const int64_t run_col0 = (int64_t)start_block_col * 32;
    const int64_t run_col_last = run_col0 + (int64_t)total_elems - 1;
    if (run_col_last < col_offset || run_col0 >= col_offset + n_cols_x) {
        return;
    }
    if (run_col_last >= n_cols_all) {
        return;
    }

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float sum = 0.0f;

    for (int32_t j = tid; j < total_elems; j += block_size) {
        const int32_t block_idx_in_run = j >> 5;
        const int32_t elem_in_block    = j & 31;
        const int32_t orig_block       = values_start + block_idx_in_run;
        const int64_t col_global       = (int64_t)(start_block_col + block_idx_in_run) * 32 + elem_in_block;

        if (col_global >= col_offset && col_global < col_offset + n_cols_x) {
            const float w = __bfloat162float(values[orig_block * 32 + elem_in_block]);
            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token_idx * x_stride];
            sum += w * a;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if ((tid & 31) == 0 && row >= 0 && row < n_rows_out) {
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

static __global__ void outlier_blocks_kernel_q8_0_merged(
        const int32_t *          __restrict__ merged_idx,
        const block_q8_0_cuda *  __restrict__ values,
        const float *            __restrict__ x,
        float *                  __restrict__ dst,
        const int64_t n_merged_runs,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    const int64_t run_idx  = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (run_idx >= n_merged_runs || token_idx >= n_tokens) {
        return;
    }

    const int32_t row            = merged_idx[run_idx * 4];
    const int32_t start_block_col = merged_idx[run_idx * 4 + 1];
    const int32_t count          = merged_idx[run_idx * 4 + 2];
    const int32_t values_start   = merged_idx[run_idx * 4 + 3];

    const int32_t total_elems = count * 32;

    const int64_t run_col0 = (int64_t)start_block_col * 32;
    const int64_t run_col_last = run_col0 + (int64_t)total_elems - 1;
    if (run_col_last < col_offset || run_col0 >= col_offset + n_cols_x) {
        return;
    }
    if (run_col_last >= n_cols_all) {
        return;
    }

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float sum = 0.0f;

    for (int32_t j = tid; j < total_elems; j += block_size) {
        const int32_t block_idx_in_run = j >> 5;
        const int32_t elem_in_block    = j & 31;
        const int32_t orig_block       = values_start + block_idx_in_run;
        const int64_t col_global       = (int64_t)(start_block_col + block_idx_in_run) * 32 + elem_in_block;

        if (col_global >= col_offset && col_global < col_offset + n_cols_x) {
            const block_q8_0_cuda * q8block = &values[orig_block];
            const float d = __half2float(__ushort_as_half(q8block->d));
            const float w = d * (float)q8block->qs[elem_in_block];
            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token_idx * x_stride];
            sum += w * a;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if ((tid & 31) == 0 && row >= 0 && row < n_rows_out) {
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

static void outlier_blocks_merged_cuda(
        ggml_backend_cuda_context & ctx,
        const int32_t *     merged_idx_d,
        const void *        values_d,
        const float *       x_d,
        float *             dst_d,
        const int64_t n_merged_runs,
        const int64_t n_blocks_total,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens,
        enum ggml_type values_type,
        int32_t value_type_i32) {

    cudaStream_t stream = ctx.stream();

    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, n_rows_out * n_tokens * sizeof(float), stream));

    if (n_merged_runs == 0 || n_tokens == 0 || n_cols_x == 0) {
        return;
    }

    GGML_UNUSED(n_blocks_total);

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-cuda-merged] kernel launch: n_merged_runs=%lld n_tokens=%lld values_type=%s\n",
                (long long)n_merged_runs, (long long)n_tokens,
                values_type == GGML_TYPE_Q8_0 ? "Q8_0" : values_type == GGML_TYPE_I8 ? "I8_nibble" : "BF16");
        fflush(stderr);
    }

    // Use 1024 threads per block — handles up to 32 blocks per run without
    // looping. Larger runs get handled by the per-thread loop.
    dim3 block_dim(1024, 1, 1);
    dim3 grid_dim((uint32_t) n_merged_runs, (uint32_t) n_tokens, 1);

    if (values_type == GGML_TYPE_Q8_0) {
        const block_q8_0_cuda * values_q8 = (const block_q8_0_cuda *) values_d;
        outlier_blocks_kernel_q8_0_merged<<<grid_dim, block_dim, 0, stream>>>(
                merged_idx_d, values_q8, x_d, dst_d,
                n_merged_runs, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);
    } else if (values_type == GGML_TYPE_I8) {
        const uint8_t * values_nibble = (const uint8_t *) values_d;
        outlier_blocks_kernel_nibble_merged<<<grid_dim, block_dim, 0, stream>>>(
                merged_idx_d, values_nibble, x_d, dst_d,
                n_merged_runs, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens,
                value_type_i32);
    } else {
        const nv_bfloat16 * values_bf16 = (const nv_bfloat16 *) values_d;
        outlier_blocks_kernel_bf16_merged<<<grid_dim, block_dim, 0, stream>>>(
                merged_idx_d, values_bf16, x_d, dst_d,
                n_merged_runs, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);
    }

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_mul_mat_outlier_blocks_merged(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * merged_idx = dst->src[0];
    const ggml_tensor * values     = dst->src[1];
    const ggml_tensor * x          = dst->src[2];

    GGML_ASSERT(merged_idx->type == GGML_TYPE_I32);
    GGML_ASSERT(values->type == GGML_TYPE_BF16 || values->type == GGML_TYPE_Q8_0 || values->type == GGML_TYPE_I8);
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_rows_out    = ggml_get_op_params_i32(dst, 0);
    const int64_t n_cols_all    = ggml_get_op_params_i32(dst, 1);
    const int32_t value_type_i32 = ggml_get_op_params_i32(dst, 2);
    const int64_t n_merged_runs = merged_idx->ne[1];
    const int64_t n_tokens      = x->ne[1];
    const int64_t n_cols_x      = x->ne[0];
    const int64_t x_stride      = x->nb[1] / (int64_t)sizeof(float);
    const int64_t n_blocks_total = values->ne[1];

    int64_t col_offset = 0;
    if (x->view_src && x->view_offs > 0) {
        col_offset = x->view_offs / (int64_t)sizeof(float);
    } else if (n_cols_x < n_cols_all) {
        col_offset = ctx.device * n_cols_x;
    }

    GGML_ASSERT(dst->ne[0] == n_rows_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(n_cols_x <= n_cols_all);
    GGML_ASSERT(merged_idx->ne[0] == 4);

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-cuda-merged] enter: n_merged_runs=%lld n_blocks_total=%lld n_rows_out=%lld n_tokens=%lld n_cols_all=%lld n_cols_x=%lld col_offset=%lld values_type=%s\n",
                (long long)n_merged_runs, (long long)n_blocks_total, (long long)n_rows_out, (long long)n_tokens,
                (long long)n_cols_all, (long long)n_cols_x, (long long)col_offset,
                values->type == GGML_TYPE_Q8_0 ? "Q8_0" : values->type == GGML_TYPE_I8 ? "I8_nibble" : "BF16");
        fflush(stderr);
    }

    const int32_t * merged_idx_d = (const int32_t *) merged_idx->data;
    const void *    values_d     = values->data;
    const float *   x_d          = (const float *) x->data;
    float *         dst_d        = (float *) dst->data;

    outlier_blocks_merged_cuda(ctx, merged_idx_d, values_d, x_d, dst_d,
            n_merged_runs, n_blocks_total, n_cols_all, n_cols_x, col_offset, x_stride,
            n_rows_out, n_tokens, values->type, value_type_i32);
}

// ============================================================================
// Fused outlier-aware matmul kernel (Proposal 4.2)
//
// Replaces the separate base matmul + sparse correction with a single fused
// kernel that dequantizes Q4_0/Q8_0 weights, checks for outlier delta via CSR
// lookup, adds delta inline, and computes the dot product.
//
// Grid: (n_rows_out, n_tokens) — one thread block per output element
// Block: 256 threads
// Each thread block loops over all blocks in its assigned row.
// ============================================================================

// Q4_0 block layout for direct access in kernel
typedef struct {
    half  d;
    uint8_t qs[16];
} block_q4_0_cuda;

#define FUSED_BLOCK_SIZE 256

static __global__ void fused_outlier_q4_0_kernel(
        const block_q4_0_cuda * __restrict__ w_q4,      // [n_blocks_per_row, n_rows_out]
        const float *           __restrict__ x,          // [n_cols_x, n_tokens]
        const int32_t *         __restrict__ idx,        // [2, n_outlier_blocks]
        const uint8_t *         __restrict__ values,     // [16, n_outlier_blocks] nibble-diff
        float *                 __restrict__ dst,        // [n_rows_out, n_tokens]
        const int64_t n_rows_out,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_tokens,
        const int64_t n_blocks_per_row,
        const int64_t n_outlier_blocks,
        const int32_t *        __restrict__ row_ptr,     // [n_rows_out + 1] CSR prefix sum
        const int32_t *        __restrict__ block_col_csr, // [n_outlier_blocks] sorted block cols
        const int32_t value_type) {                      // LLAMA_OUTLIER_VALUE_TYPE_*

    const int64_t row   = blockIdx.x;
    const int64_t token = blockIdx.y;

    if (row >= n_rows_out || token >= n_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float sum = 0.0f;

    // Iterate over all blocks in this row
    for (int64_t bk = 0; bk < n_blocks_per_row; bk++) {
        const int64_t col0 = bk * 32;
        const int64_t weight_idx = row * n_blocks_per_row + bk;
        const block_q4_0_cuda * q4block = &w_q4[weight_idx];
        const float d = __half2float(q4block->d);

        // CSR lookup: check if this block has a delta
        int32_t delta_idx = -1;
        const int32_t r_start = row_ptr[row];
        const int32_t r_end   = row_ptr[row + 1];
        for (int32_t k = r_start; k < r_end; k++) {
            if (block_col_csr[k] == (int32_t)bk) {
                delta_idx = k;
                break;
            }
        }
        const bool has_delta = (delta_idx >= 0);

        // Each thread handles elements_per_thread in this block
        for (int32_t j = tid; j < 32; j += block_size) {
            const int64_t col_global = col0 + j;

            if (col_global < col_offset || col_global >= col_offset + n_cols_x) {
                continue;
            }
            if (col_global >= n_cols_all) {
                continue;
            }

            // Dequantize Q4_0 weight
            const uint8_t byte = q4block->qs[j >> 1];
            const uint8_t nibble = (j & 1) ? (byte >> 4) : (byte & 0x0F);
            float w = d * ((float)(int)nibble - 8.0f);

            // Add delta if present (nibble-diff format)
            if (has_delta) {
                const uint8_t dbyte = values[delta_idx * 16 + (j >> 1)];
                const uint8_t dnibble = (j & 1) ? (dbyte >> 4) : (dbyte & 0x0F);
                if (dnibble & 0x08) {
                    float dv;
                    if (value_type == 3) {
                        switch (dnibble & 0x03) {
                            case 0: dv = 0.002f; break;
                            case 1: dv = 0.005f; break;
                            case 2: dv = 0.02f;  break;
                            case 3: dv = 0.05f;  break;
                            default: dv = 0.0f; break;
                        }
                    } else {
                        dv = (dnibble & 0x02) ? 0.001f : 0.01f;
                        if (dnibble & 0x01) dv *= 2.0f;
                    }
                    if (!(dnibble & 0x04)) dv = -dv;
                    w += dv;
                }
            }

            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token * x_stride];
            sum += w * a;
        }
    }

    // Warp reduce
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread of each warp writes result
    if ((tid & 31) == 0) {
        dst[row + token * n_rows_out] = sum;
    }
}

// Fallback kernel for generic Q8_0 or other types using ggml dequant API
static __global__ void fused_outlier_generic_kernel(
        const char *            __restrict__ w_data,     // quantized weight data
        const float *           __restrict__ x,
        const int32_t *         __restrict__ idx,
        const uint8_t *         __restrict__ values,     // nibble-diff by default
        float *                 __restrict__ dst,
        const int64_t n_rows_out,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_tokens,
        const int64_t n_blocks_per_row,
        const int64_t n_outlier_blocks,
        const int32_t *        __restrict__ row_ptr,
        const int32_t *        __restrict__ block_col_csr,
        const int32_t weight_type_size,         // bytes per Q block
        const int32_t value_type) {             // LLAMA_OUTLIER_VALUE_TYPE_*

    const int64_t row   = blockIdx.x;
    const int64_t token = blockIdx.y;

    if (row >= n_rows_out || token >= n_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    float sum = 0.0f;

    for (int64_t bk = 0; bk < n_blocks_per_row; bk++) {
        const int64_t col0 = bk * 32;
        const int64_t weight_idx = row * n_blocks_per_row + bk;

        // Dequantize this block on-the-fly (shared memory dequant buf)
        __shared__ float dequant_buf[32];
        if (tid < 32) {
            // Simple dequant: read raw bytes, convert
            const char * block_ptr = w_data + weight_idx * weight_type_size;
            // For Q8_0: 2 bytes d + 32 bytes qs = 34 bytes
            // For simplicity, use half d to float conversion
            if (weight_type_size >= 34) {
                const half * d_ptr = (const half *)block_ptr;
                const char * qs = block_ptr + 2;
                const float d = __half2float(*d_ptr);
                dequant_buf[tid] = d * (float)qs[tid];
            } else {
                // Q4_0 fallback: 2 bytes d + 16 bytes qs
                const half * d_ptr = (const half *)block_ptr;
                const uint8_t * qs = (const uint8_t *)(block_ptr + 2);
                const float d = __half2float(*d_ptr);
                const uint8_t byte = qs[tid >> 1];
                const uint8_t nibble = (tid & 1) ? (byte >> 4) : (byte & 0x0F);
                dequant_buf[tid] = d * ((float)(int)nibble - 8.0f);
            }
        }
        __syncthreads();

        // CSR lookup for delta
        int32_t delta_idx = -1;
        const int32_t r_start = row_ptr[row];
        const int32_t r_end   = row_ptr[row + 1];
        for (int32_t k = r_start; k < r_end; k++) {
            if (block_col_csr[k] == (int32_t)bk) {
                delta_idx = k;
                break;
            }
        }
        const bool has_delta = (delta_idx >= 0);

        for (int32_t j = tid; j < 32; j += block_size) {
            const int64_t col_global = col0 + j;
            if (col_global < col_offset || col_global >= col_offset + n_cols_x) continue;
            if (col_global >= n_cols_all) continue;

            float w = dequant_buf[j];

            if (has_delta) {
                const uint8_t dbyte = values[delta_idx * 16 + (j >> 1)];
                const uint8_t dnibble = (j & 1) ? (dbyte >> 4) : (dbyte & 0x0F);
                if (dnibble & 0x08) {
                    float dv;
                    if (value_type == 3) {
                        switch (dnibble & 0x03) {
                            case 0: dv = 0.002f; break;
                            case 1: dv = 0.005f; break;
                            case 2: dv = 0.02f;  break;
                            case 3: dv = 0.05f;  break;
                            default: dv = 0.0f; break;
                        }
                    } else {
                        dv = (dnibble & 0x02) ? 0.001f : 0.01f;
                        if (dnibble & 0x01) dv *= 2.0f;
                    }
                    if (!(dnibble & 0x04)) dv = -dv;
                    w += dv;
                }
            }

            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token * x_stride];
            sum += w * a;
        }
        __syncthreads();
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if ((tid & 31) == 0) {
        dst[row + token * n_rows_out] = sum;
    }
}

void ggml_cuda_op_mul_mat_outlier_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w      = dst->src[0];
    const ggml_tensor * x      = dst->src[1];
    const ggml_tensor * idx    = dst->src[2];
    const ggml_tensor * values = dst->src[3];

    GGML_ASSERT(ggml_is_quantized(w->type));
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(idx->type == GGML_TYPE_I32);
    GGML_ASSERT(values->type == GGML_TYPE_BF16 || values->type == GGML_TYPE_Q8_0 || values->type == GGML_TYPE_I8);
    GGML_ASSERT(idx->ne[0] == 2);
    GGML_ASSERT(idx->ne[1] == values->ne[1]);

    const int64_t n_rows_out      = ggml_get_op_params_i32(dst, 0);
    const int64_t n_cols_all      = ggml_get_op_params_i32(dst, 1);
    const int32_t value_type_i32   = ggml_get_op_params_i32(dst, 2);
    const int64_t n_outlier_blocks = idx->ne[1];
    const int64_t n_tokens         = x->ne[1];
    const int64_t n_cols_x         = x->ne[0];
    const int64_t x_stride         = x->nb[1] / (int64_t)sizeof(float);
    const int64_t n_blocks_per_row = n_cols_all / 32;

    int64_t col_offset = 0;
    if (x->view_src && x->view_offs > 0) {
        col_offset = x->view_offs / (int64_t)sizeof(float);
    } else if (n_cols_x < n_cols_all) {
        col_offset = ctx.device * n_cols_x;
    }

    GGML_ASSERT(dst->ne[0] == n_rows_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(n_cols_x <= n_cols_all);

    cudaStream_t stream = ctx.stream();

    // Build CSR layout from idx on GPU
    // We need row_ptr and block_col on device for fast lookup
    // Since the idx is already on GPU, we can build CSR on-the-fly using a small temp buffer
    // For now, copy idx to host and build CSR there, then upload
    const int32_t * idx_d = (const int32_t *) idx->data;

    // Build CSR on host
    std::vector<int32_t> row_ptr_h(n_rows_out + 1, 0);
    std::vector<int32_t> block_col_h(n_outlier_blocks);
    std::vector<int32_t> idx_h(n_outlier_blocks * 2);

    CUDA_CHECK(cudaMemcpyAsync(idx_h.data(), idx_d,
            n_outlier_blocks * 2 * sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t k = 0; k < n_outlier_blocks; k++) {
        int32_t row = idx_h[k * 2];
        if (row >= 0 && row < (int32_t)n_rows_out) {
            row_ptr_h[row + 1]++;
        }
    }
    for (int64_t r = 0; r < n_rows_out; r++) {
        row_ptr_h[r + 1] += row_ptr_h[r];
    }

    std::vector<int32_t> row_cursor = row_ptr_h;
    for (int64_t k = 0; k < n_outlier_blocks; k++) {
        int32_t row = idx_h[k * 2];
        int32_t bcol = idx_h[k * 2 + 1];
        if (row >= 0 && row < (int32_t)n_rows_out) {
            int32_t pos = row_cursor[row]++;
            block_col_h[pos] = bcol;
        }
    }

    // Upload CSR to GPU
    int32_t * row_ptr_d = nullptr;
    int32_t * block_col_d = nullptr;
    CUDA_CHECK(cudaMalloc(&row_ptr_d, (n_rows_out + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&block_col_d, n_outlier_blocks * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpyAsync(row_ptr_d, row_ptr_h.data(),
            (n_rows_out + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(block_col_d, block_col_h.data(),
            n_outlier_blocks * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    const bool is_q4_0 = (w->type == GGML_TYPE_Q4_0);
    const bool is_nibble = (values->type == GGML_TYPE_I8);

    dim3 block_dim(FUSED_BLOCK_SIZE, 1, 1);
    dim3 grid_dim((uint32_t) n_rows_out, (uint32_t) n_tokens, 1);

    if (is_q4_0 && is_nibble) {
        const block_q4_0_cuda * w_q4_d = (const block_q4_0_cuda *) w->data;
        const uint8_t * values_d = (const uint8_t *) values->data;
        const float * x_d = (const float *) x->data;
        float * dst_d = (float *) dst->data;

        fused_outlier_q4_0_kernel<<<grid_dim, block_dim, 0, stream>>>(
                w_q4_d, x_d, idx_d, values_d, dst_d,
                n_rows_out, n_cols_all, n_cols_x, col_offset, x_stride,
                n_tokens, n_blocks_per_row, n_outlier_blocks,
                row_ptr_d, block_col_d, value_type_i32);
    } else {
        const char * w_data_d = (const char *) w->data;
        const uint8_t * values_d = (const uint8_t *) values->data;
        const float * x_d = (const float *) x->data;
        float * dst_d = (float *) dst->data;
        const int32_t w_type_size = (int32_t) ggml_type_size(w->type);

        fused_outlier_generic_kernel<<<grid_dim, block_dim, 0, stream>>>(
                w_data_d, x_d, idx_d, values_d, dst_d,
                n_rows_out, n_cols_all, n_cols_x, col_offset, x_stride,
                n_tokens, n_blocks_per_row, n_outlier_blocks,
                row_ptr_d, block_col_d, w_type_size, value_type_i32);
    }

    CUDA_CHECK(cudaGetLastError());

    // Cleanup
    CUDA_CHECK(cudaFree(row_ptr_d));
    CUDA_CHECK(cudaFree(block_col_d));
}
