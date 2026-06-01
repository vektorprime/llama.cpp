#pragma once

#include "common.cuh"

void ggml_cuda_op_mul_mat_outlier_blocks(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
