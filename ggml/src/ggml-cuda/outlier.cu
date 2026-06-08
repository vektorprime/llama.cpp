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
//   out[row, token] += sum_{j=0..31} bf16_to_float(values[j, block]) * x[col0 + j, token]
//
// Grid dimensions:
//   blockIdx.x: outlier block index (0..n_blocks-1)
//   blockIdx.y: token index (0..n_tokens-1)
//
// Each thread in the block handles one element of the 32-element dot product,
// then the thread block does a warp reduction to get the final sum.

#define OUTLIER_BLOCK_SIZE 32

static __global__ void outlier_blocks_kernel(
        const int32_t *     __restrict__ idx,       // [2, n_blocks]
        const nv_bfloat16 * __restrict__ values,    // [32, n_blocks] in BF16
        const float *       __restrict__ x,         // [n_cols_x, n_tokens] shard-local
        float *             __restrict__ dst,       // [n_rows_out, n_tokens]
        const int64_t n_blocks,
        const int64_t n_cols_all,      // full column count (all shards)
        const int64_t n_cols_x,        // shard-local column count
        const int64_t col_offset,      // starting column of this shard in global space
        const int64_t x_stride,        // stride between tokens in x
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    const int64_t block_idx = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (block_idx >= n_blocks || token_idx >= n_tokens) {
        return;
    }

    // Read block metadata
    const int32_t row       = idx[block_idx * 2];
    const int32_t block_col = idx[block_idx * 2 + 1];
    const int64_t col0      = (int64_t) block_col * 32;

    // Check if this block touches our shard: [col_offset, col_offset + n_cols_x)
    // must overlap [col0, col0 + 32)
    if (col0 + 31 < col_offset || col0 >= col_offset + n_cols_x) {
        return; // block entirely outside this shard
    }

    if (col0 + 31 >= n_cols_all) {
        return; // out of bounds globally
    }

    // Each thread handles one element of the dot product
    const int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread processes element j = tid
    if (tid < 32) {
        const int64_t j = tid;
        const int64_t col_global = col0 + j;
        // Only participate if this column is in our shard
        if (col_global >= col_offset && col_global < col_offset + n_cols_x) {
            // Load BF16 weight and convert to float
            const float w = __bfloat162float(values[block_idx * 32 + j]);

            // Load corresponding activation (use x_stride for correct row stride)
            const int64_t col_local = col_global - col_offset;
            const float a = x[col_local + token_idx * x_stride];

            sum = w * a;
        }
    }

    // Warp reduction: reduce sum within the warp (32 threads)
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Thread 0 writes the result (atomic add since multiple blocks may target the same row)
    if (tid == 0 && row >= 0 && row < n_rows_out) {
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

static void outlier_blocks_cuda(
        ggml_backend_cuda_context & ctx,
        const int32_t *     idx_d,
        const nv_bfloat16 * values_d,
        const float *       x_d,
        float *             dst_d,
        const int64_t n_blocks,
        const int64_t n_cols_all,
        const int64_t n_cols_x,
        const int64_t col_offset,
        const int64_t x_stride,
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    cudaStream_t stream = ctx.stream();

    // Clear dst to zero first (important since we use atomicAdd)
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, n_rows_out * n_tokens * sizeof(float), stream));

    if (n_blocks == 0 || n_tokens == 0 || n_cols_x == 0) {
        return;
    }

    dim3 block_dim(OUTLIER_BLOCK_SIZE, 1, 1);
    dim3 grid_dim((uint32_t) n_blocks, (uint32_t) n_tokens, 1);

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-cuda] kernel launch: grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                (unsigned)grid_dim.x, (unsigned)grid_dim.y, (unsigned)grid_dim.z,
                (unsigned)block_dim.x, (unsigned)block_dim.y, (unsigned)block_dim.z);
        fflush(stderr);
    }

    outlier_blocks_kernel<<<grid_dim, block_dim, 0, stream>>>(
            idx_d, values_d, x_d, dst_d,
            n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);

    CUDA_CHECK(cudaGetLastError());
}

// Main entry point called from ggml-cuda dispatch
void ggml_cuda_op_mul_mat_outlier_blocks(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * idx    = dst->src[0];
    const ggml_tensor * values = dst->src[1];
    const ggml_tensor * x      = dst->src[2];

    GGML_ASSERT(idx->type    == GGML_TYPE_I32);
    GGML_ASSERT(values->type == GGML_TYPE_BF16);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);

    const int64_t n_rows_out = ggml_get_op_params_i32(dst, 0);
    const int64_t n_cols_all = ggml_get_op_params_i32(dst, 1);
    const int64_t n_blocks   = idx->ne[1];
    const int64_t n_tokens   = x->ne[1];
    const int64_t n_cols_x   = x->ne[0];
    const int64_t x_stride   = x->nb[1] / (int64_t)sizeof(float);

    // Compute shard column offset from CUDA device index.
    // When tensor parallelism splits activations, each GPU gets a contiguous
    // shard but the idx data uses global column indices. We need to know
    // which global column range this GPU owns.
    int64_t col_offset = 0;
    if (x->view_src && x->view_offs > 0) {
        col_offset = x->view_offs / (int64_t)sizeof(float);
    } else if (n_cols_x < n_cols_all) {
        // Activation is split but not a view — compute offset from device index.
        // Assumes equal shard sizes and device ordering matching shard order.
        col_offset = ctx.device * n_cols_x;
    }

    GGML_ASSERT(dst->ne[0] == n_rows_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(n_cols_x    <= n_cols_all);
    GGML_ASSERT(idx->ne[0]  == 2);
    GGML_ASSERT(values->ne[0] == 32);

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-cuda] enter: n_blocks=%lld n_rows_out=%lld n_tokens=%lld n_cols_all=%lld n_cols_x=%lld col_offset=%lld\n",
                (long long)n_blocks, (long long)n_rows_out, (long long)n_tokens,
                (long long)n_cols_all, (long long)n_cols_x, (long long)col_offset);
        fflush(stderr);
    }

    const int32_t *     idx_d    = (const int32_t *)     idx->data;
    const nv_bfloat16 * values_d = (const nv_bfloat16 *) values->data;
    const float *       x_d      = (const float *)       x->data;
    float *             dst_d    = (float *)             dst->data;

#if 0 // DEBUG — verify kernel data (no sync — crashes during graph capture)
    {
        if (n_blocks > 0 && n_tokens > 0) {
            std::vector<int32_t> idx_host(2);
            std::vector<uint16_t> val_host(32);
            std::vector<float> x_host(32);
            CUDA_CHECK(cudaMemcpy(idx_host.data(), idx_d, 2*sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(val_host.data(), values_d, 32*sizeof(uint16_t), cudaMemcpyDeviceToHost));
            int32_t row0 = idx_host[0], bcol0 = idx_host[1];
            int64_t col0 = (int64_t)bcol0 * 32;
            if (col0 + 31 >= col_offset && col0 < col_offset + n_cols_x && col0 + 31 < n_cols_all) {
                int64_t col_local = col0 - col_offset;
                CUDA_CHECK(cudaMemcpy(x_host.data(), x_d + col_local, 32*sizeof(float), cudaMemcpyDeviceToHost));
                float expected_dot = 0.0f;
                for (int j = 0; j < 32; j++) {
                    float w = __bfloat162float(*(const nv_bfloat16*)&val_host[j]);
                    expected_dot += w * x_host[j];
                }
                fprintf(stderr, "[CUDA] n_blocks=%lld row=%d block_col=%d col0=%lld expected_dot=%f\n",
                        (long long)n_blocks, row0, bcol0, (long long)col0, expected_dot);
            }
        }
        fflush(stderr);
    }
#endif

    outlier_blocks_cuda(ctx, idx_d, values_d, x_d, dst_d,
            n_blocks, n_cols_all, n_cols_x, col_offset, x_stride, n_rows_out, n_tokens);

    // [delta-cuda] compute output L2 norm by copying first column and measuring on host
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
