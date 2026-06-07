#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <set>

#ifdef GGML_USE_CUDA
#include <ggml-cuda.h>
#endif

static bool g_failed = false;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
            g_failed = true; \
            return false; \
        } \
    } while (0)

#define TEST_PASS() \
    do { \
        printf("  PASS\n"); \
        return true; \
    } while (0)

/*
 * Helper: fill tensor with deterministic data via F32 then convert.
 */
static void fill_tensor_seq(ggml_tensor * t, float scale) {
    std::vector<float> f32(ggml_nelements(t));
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        f32[i] = (float)(i % 1024) * scale;
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f32.data(), 0, f32.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> bf16(ggml_nelements(t));
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            bf16[i] = ggml_fp32_to_bf16(f32[i]);
        }
        ggml_backend_tensor_set(t, bf16.data(), 0, bf16.size() * sizeof(ggml_bf16_t));
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> i32(ggml_nelements(t));
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            i32[i] = (int32_t)(i % 256);
        }
        ggml_backend_tensor_set(t, i32.data(), 0, i32.size() * sizeof(int32_t));
    } else if (t->type == GGML_TYPE_I64) {
        std::vector<int64_t> i64(ggml_nelements(t));
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            i64[i] = (int64_t)(i % 256);
        }
        ggml_backend_tensor_set(t, i64.data(), 0, i64.size() * sizeof(int64_t));
    } else {
        // For quantized types, quantize from f32
        size_t row_size = ggml_row_size(t->type, ggml_nelements(t));
        std::vector<uint8_t> qbuf(row_size);
        ggml_quantize_chunk(t->type, f32.data(), qbuf.data(), 0,
                            (size_t)ggml_nelements(t) / ggml_blck_size(t->type),
                            ggml_blck_size(t->type), nullptr);
        ggml_backend_tensor_set(t, qbuf.data(), 0, row_size);
    }
}

/*
 * Helper: get tensor data as F32 vector.
 */
static std::vector<float> tensor_to_f32(ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> bf16(ggml_nelements(t));
        ggml_backend_tensor_get(t, bf16.data(), 0, ggml_nbytes(t));
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            out[i] = ggml_bf16_to_fp32(bf16[i]);
        }
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> i32(ggml_nelements(t));
        ggml_backend_tensor_get(t, i32.data(), 0, ggml_nbytes(t));
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            out[i] = (float)i32[i];
        }
    } else if (t->type == GGML_TYPE_I64) {
        std::vector<int64_t> i64(ggml_nelements(t));
        ggml_backend_tensor_get(t, i64.data(), 0, ggml_nbytes(t));
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            out[i] = (float)i64[i];
        }
    } else {
        // Quantized: decode via to_float
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
        const ggml_type_traits_t traits = ggml_get_type_traits(t->type);
        int64_t bs = ggml_blck_size(t->type);
        std::vector<float> tmp(bs);
        size_t off = 0;
        for (int64_t i = 0; i < ggml_nelements(t); i += bs) {
            traits->to_float(raw.data() + off, tmp.data(), bs);
            for (int64_t j = 0; j < bs && i + j < ggml_nelements(t); j++) {
                out[i + j] = tmp[j];
            }
            off += ggml_row_size(t->type, bs);
        }
    }
    return out;
}

/*
 * Test 1: CPU-only path.
 *
 * Build a synthetic graph:
 *   base   = MUL_MAT(W_base, X)       -- F32 weight, F32 input
 *   corr   = MUL_MAT_OUTLIER_BLOCKS(idx, values, X, n_rows, n_cols)
 *   result = ADD(base, corr)
 *
 * Run on CPU backend and verify the output is non-trivial
 * (i.e. the correction was actually applied, not silently zeroed).
 */
static bool test_cpu_only_path() {
    printf("Test 1: CPU-only path (-ngl 0)...\n");

    const int64_t n_rows = 256;
    const int64_t n_cols = 128;
    const int64_t n_tokens = 4;
    const int64_t n_outlier_blocks = 32;

    ggml_init_params params = {
        /* .mem_size   = */ (size_t)(10 * 1024 * 1024),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    // W_base: [n_cols, n_rows], BF16 (simulating quantized base weight)
    ggml_tensor * w_base = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_cols, n_rows);
    ggml_set_name(w_base, "w_base");

    // X: [n_cols, n_tokens], F32 input
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    ggml_set_name(x, "x");

    // idx: [n_outlier_blocks], I64 indices into rows
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_outlier_blocks);
    ggml_set_name(idx, "idx");

    // values: [n_cols, n_outlier_blocks], BF16 sparse correction values
    ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_cols, n_outlier_blocks);
    ggml_set_name(values, "values");

    // base = W_base^T @ X  => [n_rows, n_tokens]
    ggml_tensor * base = ggml_mul_mat(ctx, w_base, x);
    ggml_set_name(base, "base_mul_mat");

    // corr = MUL_MAT_OUTLIER_BLOCKS(idx, values, X, n_rows, n_cols) => [n_rows, n_tokens]
    ggml_tensor * corr = ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows, n_cols);
    ggml_set_name(corr, "corr_outlier");

    // result = base + corr
    ggml_tensor * result = ggml_add(ctx, base, corr);
    ggml_set_name(result, "result");

    // Build graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // Allocate on CPU
    ggml_backend_t cpu = ggml_backend_cpu_init();
    TEST_ASSERT(cpu != nullptr, "ggml_backend_cpu_init failed");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    TEST_ASSERT(buf != nullptr, "ggml_backend_alloc_ctx_tensors failed");

    // Fill tensors with known data
    fill_tensor_seq(w_base, 0.01f);
    fill_tensor_seq(x, 0.1f);

    // Fill idx with distinct row indices
    {
        std::vector<int64_t> idx_data(n_outlier_blocks);
        for (int64_t i = 0; i < n_outlier_blocks; i++) {
            idx_data[i] = i * (n_rows / n_outlier_blocks);
        }
        ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));
    }

    fill_tensor_seq(values, 0.05f);

    // Compute
    ggml_status status = ggml_backend_graph_compute(cpu, gf);
    TEST_ASSERT(status == GGML_STATUS_SUCCESS, "ggml_backend_graph_compute failed");

    // Read result
    std::vector<float> result_data = tensor_to_f32(result);
    TEST_ASSERT(result_data.size() == (size_t)(n_rows * n_tokens), "result shape mismatch");

    // Verify result is non-trivial: at least some elements should be non-zero
    // (with our fill patterns, both base and corr contribute)
    float sum = 0.0f;
    for (float v : result_data) {
        sum += v * v;
    }
    TEST_ASSERT(sum > 1.0f, "result is trivially small - correction may have been dropped");

    // Now run a second computation with zeroed correction values;
    // the result should differ from the first run, proving the correction path is active.
    fill_tensor_seq(values, 0.0f);
    status = ggml_backend_graph_compute(cpu, gf);
    TEST_ASSERT(status == GGML_STATUS_SUCCESS, "second compute failed");

    std::vector<float> result_zero_corr = tensor_to_f32(result);
    float diff = 0.0f;
    for (size_t i = 0; i < result_data.size(); i++) {
        float d = result_data[i] - result_zero_corr[i];
        diff += d * d;
    }
    TEST_ASSERT(diff > 1.0f, "zeroing correction values did not change output - correction path inactive");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    TEST_PASS();
}

/*
 * Test 2: Backend support check.
 *
 * Verify that ggml_backend_supports_op() returns true for
 * GGML_OP_MUL_MAT_OUTLIER_BLOCKS on CPU backend, and on CUDA
 * backend if available.
 */
static bool test_backend_support_check() {
    printf("Test 2: Backend support check...\n");

    const int64_t n_rows = 256;
    const int64_t n_cols = 128;
    const int64_t n_tokens = 4;
    const int64_t n_outlier_blocks = 32;

    ggml_init_params params = {
        /* .mem_size   = */ (size_t)(10 * 1024 * 1024),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_outlier_blocks);
    ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_cols, n_outlier_blocks);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);

    ggml_tensor * corr = ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows, n_cols);
    ggml_set_name(corr, "corr_outlier");

    // CPU backend must support the op
    ggml_backend_t cpu = ggml_backend_cpu_init();
    TEST_ASSERT(cpu != nullptr, "ggml_backend_cpu_init failed");

    bool cpu_supports = ggml_backend_supports_op(cpu, corr);
    printf("  CPU supports MUL_MAT_OUTLIER_BLOCKS: %s\n", cpu_supports ? "yes" : "no");
    TEST_ASSERT(cpu_supports, "CPU backend does not support MUL_MAT_OUTLIER_BLOCKS");

    ggml_backend_free(cpu);

    // CUDA backend (if available)
#ifdef GGML_USE_CUDA
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (cuda) {
        bool cuda_supports = ggml_backend_supports_op(cuda, corr);
        printf("  CUDA supports MUL_MAT_OUTLIER_BLOCKS: %s\n", cuda_supports ? "yes" : "no");
        TEST_ASSERT(cuda_supports, "CUDA backend does not support MUL_MAT_OUTLIER_BLOCKS");
        ggml_backend_free(cuda);
    } else {
        printf("  CUDA backend not available, skipping CUDA support check\n");
    }
#else
    printf("  CUDA not compiled in, skipping CUDA support check\n");
#endif

    ggml_free(ctx);

    TEST_PASS();
}

/*
 * Test 3: No silent drop.
 *
 * Build a graph with base MUL_MAT + MUL_MAT_OUTLIER_BLOCKS + ADD.
 * After graph construction and computation, verify that all three
 * ops are present in the graph node list (no op was silently removed).
 */
static bool test_no_silent_drop() {
    printf("Test 3: No silent drop...\n");

    const int64_t n_rows = 256;
    const int64_t n_cols = 128;
    const int64_t n_tokens = 4;
    const int64_t n_outlier_blocks = 32;

    ggml_init_params params = {
        /* .mem_size   = */ (size_t)(10 * 1024 * 1024),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_tensor * w_base = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_cols, n_rows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_outlier_blocks);
    ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_cols, n_outlier_blocks);

    ggml_tensor * base = ggml_mul_mat(ctx, w_base, x);
    ggml_tensor * corr = ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows, n_cols);
    ggml_tensor * result = ggml_add(ctx, base, corr);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // Collect op types from the graph nodes
    std::set<enum ggml_op> ops_in_graph;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        ops_in_graph.insert(node->op);
    }

    bool has_mul_mat = ops_in_graph.count(GGML_OP_MUL_MAT) > 0;
    bool has_outlier = ops_in_graph.count(GGML_OP_MUL_MAT_OUTLIER_BLOCKS) > 0;
    bool has_add     = ops_in_graph.count(GGML_OP_ADD) > 0;

    printf("  Graph contains MUL_MAT: %s\n", has_mul_mat ? "yes" : "no");
    printf("  Graph contains MUL_MAT_OUTLIER_BLOCKS: %s\n", has_outlier ? "yes" : "no");
    printf("  Graph contains ADD: %s\n", has_add ? "yes" : "no");

    TEST_ASSERT(has_mul_mat, "MUL_MAT was silently dropped from the graph");
    TEST_ASSERT(has_outlier, "MUL_MAT_OUTLIER_BLOCKS was silently dropped from the graph");
    TEST_ASSERT(has_add, "ADD was silently dropped from the graph");

    // Also verify that the graph computes without error
    ggml_backend_t cpu = ggml_backend_cpu_init();
    TEST_ASSERT(cpu != nullptr, "ggml_backend_cpu_init failed");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    TEST_ASSERT(buf != nullptr, "allocation failed");

    fill_tensor_seq(w_base, 0.01f);
    fill_tensor_seq(x, 0.1f);
    fill_tensor_seq(values, 0.05f);
    {
        std::vector<int64_t> idx_data(n_outlier_blocks);
        for (int64_t i = 0; i < n_outlier_blocks; i++) {
            idx_data[i] = i * (n_rows / n_outlier_blocks);
        }
        ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));
    }

    ggml_status status = ggml_backend_graph_compute(cpu, gf);
    TEST_ASSERT(status == GGML_STATUS_SUCCESS, "graph compute failed");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    TEST_PASS();
}

/*
 * Test 4: Scheduler assigns correction to supporting backend.
 *
 * When both CPU and CUDA backends are available and CUDA supports
 * MUL_MAT_OUTLIER_BLOCKS, the scheduler should assign the
 * correction op to CUDA (not silently move it to CPU).
 *
 * Uses ggml_backend_sched_split_graph + ggml_backend_sched_get_tensor_backend
 * to inspect the assignment.
 */
static bool test_scheduler_correction_assignment() {
    printf("Test 4: Scheduler assigns correction to supporting backend...\n");

    const int64_t n_rows = 256;
    const int64_t n_cols = 128;
    const int64_t n_tokens = 4;
    const int64_t n_outlier_blocks = 32;

    ggml_init_params params = {
        /* .mem_size   = */ (size_t)(10 * 1024 * 1024),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_tensor * w_base = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_cols, n_rows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_outlier_blocks);
    ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_cols, n_outlier_blocks);

    ggml_tensor * base = ggml_mul_mat(ctx, w_base, x);
    ggml_set_name(base, "base_mul_mat");

    ggml_tensor * corr = ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows, n_cols);
    ggml_set_name(corr, "corr_outlier");

    ggml_tensor * result = ggml_add(ctx, base, corr);
    ggml_set_name(result, "result");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    TEST_ASSERT(cpu != nullptr, "ggml_backend_cpu_init failed");

#ifdef GGML_USE_CUDA
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) {
        printf("  CUDA backend not available, skipping scheduler assignment test\n");
        ggml_backend_free(cpu);
        ggml_free(ctx);
        TEST_PASS();
    }

    // Check CUDA supports the outlier op
    bool cuda_supports_outlier = ggml_backend_supports_op(cuda, corr);
    if (!cuda_supports_outlier) {
        printf("  CUDA does not support MUL_MAT_OUTLIER_BLOCKS, skipping scheduler assignment test\n");
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        TEST_PASS();
    }

    // Build scheduler with CUDA first (higher priority), then CPU
    ggml_backend_t backends[2] = {cuda, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2,
                                                         GGML_DEFAULT_GRAPH_SIZE, false, true);
    TEST_ASSERT(sched != nullptr, "ggml_backend_sched_new failed");

    // Split the graph to determine assignments (without allocating)
    ggml_backend_sched_split_graph(sched, gf);

    // Check the backend assignment for the correction op
    ggml_backend_t corr_backend = ggml_backend_sched_get_tensor_backend(sched, corr);
    TEST_ASSERT(corr_backend != nullptr, "scheduler did not assign backend for correction op");

    const char * corr_backend_name = ggml_backend_name(corr_backend);
    printf("  Correction op assigned to: %s\n", corr_backend_name);

    // The correction op should be assigned to CUDA since CUDA supports it
    // and CUDA is listed first (higher priority) in the backend array.
    bool assigned_to_cuda = (corr_backend == cuda);
    TEST_ASSERT(assigned_to_cuda,
                "Correction op was not assigned to CUDA backend even though CUDA supports it");

    // Also verify the base MUL_MAT is assigned to a backend
    ggml_backend_t base_backend = ggml_backend_sched_get_tensor_backend(sched, base);
    TEST_ASSERT(base_backend != nullptr, "scheduler did not assign backend for base op");
    printf("  Base MUL_MAT assigned to: %s\n", ggml_backend_name(base_backend));

    ggml_backend_sched_free(sched);
    ggml_backend_free(cuda);
#else
    printf("  CUDA not compiled in, verifying CPU-only scheduler behavior\n");

    // With only CPU, the scheduler should still assign the op
    ggml_backend_t backends[1] = {cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1,
                                                         GGML_DEFAULT_GRAPH_SIZE, false, true);
    TEST_ASSERT(sched != nullptr, "ggml_backend_sched_new failed");

    ggml_backend_sched_split_graph(sched, gf);

    ggml_backend_t corr_backend = ggml_backend_sched_get_tensor_backend(sched, corr);
    TEST_ASSERT(corr_backend != nullptr, "scheduler did not assign backend for correction op");

    printf("  Correction op assigned to: %s\n", ggml_backend_name(corr_backend));
    TEST_ASSERT(corr_backend == cpu, "Correction op not assigned to CPU backend");

    ggml_backend_sched_free(sched);
#endif

    ggml_backend_free(cpu);
    ggml_free(ctx);

    TEST_PASS();
}

int main(int argc, char **argv) {
    printf("=== Q8_0_BF16_OUTLIER Offload Safety Tests ===\n\n");

    bool all_pass = true;

    all_pass &= test_cpu_only_path();
    printf("\n");

    all_pass &= test_backend_support_check();
    printf("\n");

    all_pass &= test_no_silent_drop();
    printf("\n");

    all_pass &= test_scheduler_correction_assignment();
    printf("\n");

    printf("=== Summary: %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return all_pass ? 0 : 1;
}
