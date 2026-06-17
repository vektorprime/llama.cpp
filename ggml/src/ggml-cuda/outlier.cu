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

// Q4_0 block layout: ggml_half d (2 bytes) + uint8_t qs[16] (nibbles, 16 bytes) = 18 bytes
typedef struct {
    uint16_t d;
    uint8_t qs[16];
} block_q4_0_cuda;

static __global__ void outlier_blocks_kernel_q4_0(
        const int32_t *          __restrict__ idx,       // [2, n_blocks]
        const block_q4_0_cuda *  __restrict__ values,    // [n_blocks] Q4_0 blocks
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

    const block_q4_0_cuda * q4block = &values[block_idx];
    const float d = __half2float(__ushort_as_half(q4block->d));
    const int tid = threadIdx.x;
    float sum = 0.0f;

    if (tid < 32) {
        const int64_t j = tid;
        const int64_t col_global = col0 + j;
        if (col_global >= col_offset && col_global < col_offset + n_cols_x) {
            // Q4_0 nibble dequant: lower nibble at even j, upper nibble at odd j, biased by 8
            const uint8_t nibble_byte = q4block->qs[j / 2];
            const int8_t q = (int8_t)((j & 1) ? (nibble_byte >> 4) : (nibble_byte & 0x0F));
            const float w = d * ((float)q - 8.0f);
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
        enum ggml_type values_type) {

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
                values_type == GGML_TYPE_Q8_0 ? "Q8_0" :
                values_type == GGML_TYPE_Q4_0 ? "Q4_0" : "BF16");
        fflush(stderr);
    }

    if (values_type == GGML_TYPE_Q8_0) {
        const block_q8_0_cuda * values_q8 = (const block_q8_0_cuda *) values_d;
        outlier_blocks_kernel_q8_0<<<grid_dim, block_dim, 0, stream>>>(
                idx_d, values_q8, x_d, dst_d,
                n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);
    } else if (values_type == GGML_TYPE_Q4_0) {
        const block_q4_0_cuda * values_q4 = (const block_q4_0_cuda *) values_d;
        outlier_blocks_kernel_q4_0<<<grid_dim, block_dim, 0, stream>>>(
                idx_d, values_q4, x_d, dst_d,
                n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);
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
    GGML_ASSERT(values->type == GGML_TYPE_BF16 || values->type == GGML_TYPE_Q8_0 || values->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);

    const int64_t n_rows_out = ggml_get_op_params_i32(dst, 0);
    const int64_t n_cols_all = ggml_get_op_params_i32(dst, 1);
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
    GGML_ASSERT(values->ne[0] == 32);

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-cuda] enter: n_blocks=%lld n_rows_out=%lld n_tokens=%lld n_cols_all=%lld n_cols_x=%lld col_offset=%lld values_type=%s\n",
                (long long)n_blocks, (long long)n_rows_out, (long long)n_tokens,
                (long long)n_cols_all, (long long)n_cols_x, (long long)col_offset,
                values->type == GGML_TYPE_Q8_0 ? "Q8_0" :
                values->type == GGML_TYPE_Q4_0 ? "Q4_0" : "BF16");
        fflush(stderr);
    }

    const int32_t * idx_d = (const int32_t *) idx->data;
    const void *    values_d = values->data;
    const float *   x_d      = (const float *) x->data;
    float *         dst_d    = (float *) dst->data;

    outlier_blocks_cuda(ctx, idx_d, values_d, x_d, dst_d,
            n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens,
            values->type);

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
