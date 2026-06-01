#include "outlier.cuh"
#include "common.cuh"
#include "ggml.h"

#include <cstdint>

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
//
// For small n_blocks, we use a thread-per-block approach (each thread block
// computes one dot product for one token).
//
// For the warp reduction, we use __shfl_down_sync which requires warpSize threads.

#define OUTLIER_BLOCK_SIZE 32

static __global__ void outlier_blocks_kernel(
        const int32_t * __restrict__ idx,       // [2, n_blocks]
        const half *     __restrict__ values,    // [32, n_blocks] in BF16 (stored as half)
        const float *    __restrict__ x,         // [n_cols, n_tokens]
        float *          __restrict__ dst,       // [n_rows_out, n_tokens]
        const int64_t n_blocks,
        const int64_t n_cols,
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    const int64_t block_idx = blockIdx.x;
    const int64_t token_idx = blockIdx.y;

    if (block_idx >= n_blocks || token_idx >= n_tokens) {
        return;
    }

    // Read block metadata
    const int32_t row      = idx[block_idx * 2];
    const int32_t block_col = idx[block_idx * 2 + 1];
    const int64_t col0      = (int64_t) block_col * 32;

    if (col0 + 31 >= n_cols) {
        return; // out of bounds
    }

    // Each thread handles one element of the dot product
    const int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread processes element j = tid
    if (tid < 32) {
        const int64_t j = tid;
        // Load BF16 weight (stored as half in values tensor: [32, n_blocks])
        const half w_half = values[block_idx * 32 + j];
        const float w = __half2float(w_half);

        // Load corresponding activation
        const int64_t x_idx = (col0 + j) + token_idx * n_cols;
        const float a = x[x_idx];

        sum = w * a;
    }

    // Warp reduction: reduce sum within the warp (32 threads)
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Thread 0 writes the result (atomic add since multiple blocks may target the same row)
    if (tid == 0) {
        const int64_t dst_idx = row + token_idx * n_rows_out;
        atomicAdd(dst + dst_idx, sum);
    }
}

static void outlier_blocks_cuda(
        ggml_backend_cuda_context & ctx,
        const int32_t * idx_d,
        const half     * values_d,
        const float    * x_d,
        float          * dst_d,
        const int64_t n_blocks,
        const int64_t n_cols,
        const int64_t n_rows_out,
        const int64_t n_tokens) {

    cudaStream_t stream = ctx.stream();

    // Clear dst to zero first (important since we use atomicAdd)
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, n_rows_out * n_tokens * sizeof(float), stream));

    if (n_blocks == 0 || n_tokens == 0) {
        return;
    }

    dim3 block_dim(OUTLIER_BLOCK_SIZE, 1, 1);
    dim3 grid_dim((uint32_t) n_blocks, (uint32_t) n_tokens, 1);

    outlier_blocks_kernel<<<grid_dim, block_dim, 0, stream>>>(
            idx_d, values_d, x_d, dst_d,
            n_blocks, n_cols, n_rows_out, n_tokens);

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
    const int64_t n_cols     = ggml_get_op_params_i32(dst, 1);
    const int64_t n_blocks   = idx->ne[1];
    const int64_t n_tokens   = x->ne[1];

    GGML_ASSERT(dst->ne[0] == n_rows_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(x->ne[0]    == n_cols);
    GGML_ASSERT(idx->ne[0]  == 2);
    GGML_ASSERT(values->ne[0] == 32);

    const int32_t * idx_d    = (const int32_t *) idx->data;
    const half     * values_d = (const half *)     values->data;
    const float    * x_d      = (const float *)    x->data;
    float          * dst_d    = (float *)          dst->data;

    outlier_blocks_cuda(ctx, idx_d, values_d, x_d, dst_d,
            n_blocks, n_cols, n_rows_out, n_tokens);
}
