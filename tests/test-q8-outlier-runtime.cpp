// CPU runtime tests for Q8_0_BF16_OUTLIER sparse outlier block correction
// Tests CSR construction, sparse matmul kernel, hybrid output, and validation

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "src/llama-q8-outlier.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <cassert>

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            fprintf(stderr, "ASSERT_EQ failed: %s != %s at %s:%d\n", #a, #b, __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "ASSERT_TRUE failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        if (fabsf((a) - (b)) > (eps)) { \
            fprintf(stderr, "ASSERT_NEAR failed: %s (%f) != %s (%f) eps=%f at %s:%d\n", \
                    #a, (float)(a), #b, (float)(b), (eps), __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

static constexpr int Q8_BLOCK_SIZE = 32;
static constexpr size_t Q8_BLOCK_BYTES = 2 + Q8_BLOCK_SIZE;

/* ========================================================================
 * Helper: build graph, alloc, set data, compute, read output
 * ======================================================================== */

// Build tensors, then set data, then compute on CPU.
// The callback sets tensor data AFTER buffer allocation.
static bool run_cpu_graph(
        size_t ctx_size,
        std::function<ggml_tensor*(ggml_context*)> build_fn,
        std::function<void(ggml_context*)> set_data_fn,
        std::function<bool(const ggml_tensor*)> check_fn) {

    ggml_init_params params = {ctx_size, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    // 1. Build tensor graph
    ggml_tensor * out = build_fn(ctx);
    if (!out) {
        ggml_free(ctx);
        return false;
    }

    // 2. Init CPU backend and allocate buffers
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "Failed to init CPU backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) {
        fprintf(stderr, "Failed to allocate tensors\n");
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return false;
    }

    // 3. Set tensor data (NOW that buffers are allocated)
    set_data_fn(ctx);

    // 4. Build and compute
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, out);

    ggml_status status = ggml_backend_graph_compute(cpu, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_graph_compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return false;
    }

    // 5. Check output
    bool ok = check_fn(out);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    return ok;
}

/* ========================================================================
 * Test 1: CSR construction correctness
 * ======================================================================== */

static bool test_csr_construction() {
    fprintf(stderr, "  test_csr_construction...\n");

    // --- Sub-test A: blocks spread across multiple rows ---
    {
        std::vector<int32_t> idx_data = {0, 5, 2, 1, 0, 10};
        int64_t n_blocks = 3;
        int64_t n_rows_out = 4;

        std::vector<int32_t> row_ptr(n_rows_out + 1, 0);
        for (int64_t k = 0; k < n_blocks; k++) {
            int32_t row = idx_data[k * 2];
            row_ptr[row + 1]++;
        }
        for (int64_t r = 0; r < n_rows_out; r++) {
            row_ptr[r + 1] += row_ptr[r];
        }

        std::vector<int32_t> block_col(n_blocks, 0);
        std::vector<int32_t> row_cursor = row_ptr;
        for (int64_t k = 0; k < n_blocks; k++) {
            int32_t row = idx_data[k * 2];
            int32_t bcol = idx_data[k * 2 + 1];
            int32_t pos = row_cursor[row]++;
            block_col[pos] = bcol;
        }

        ASSERT_EQ(row_ptr.size(), (size_t)5);
        ASSERT_EQ(row_ptr[0], 0);
        ASSERT_EQ(row_ptr[1], 2);
        ASSERT_EQ(row_ptr[2], 2);
        ASSERT_EQ(row_ptr[3], 3);
        ASSERT_EQ(row_ptr[4], 3);

        ASSERT_EQ(block_col[0], 5);
        ASSERT_EQ(block_col[1], 10);
        ASSERT_EQ(block_col[2], 1);
    }

    // --- Sub-test B: multiple blocks in the same row ---
    {
        std::vector<int32_t> idx_data = {1, 0, 1, 3, 1, 7, 1, 1};
        int64_t n_blocks = 4;
        int64_t n_rows_out = 3;

        std::vector<int32_t> row_ptr(n_rows_out + 1, 0);
        for (int64_t k = 0; k < n_blocks; k++) {
            row_ptr[idx_data[k * 2] + 1]++;
        }
        for (int64_t r = 0; r < n_rows_out; r++) {
            row_ptr[r + 1] += row_ptr[r];
        }

        std::vector<int32_t> block_col(n_blocks, 0);
        std::vector<int32_t> row_cursor = row_ptr;
        for (int64_t k = 0; k < n_blocks; k++) {
            int32_t row = idx_data[k * 2];
            int32_t bcol = idx_data[k * 2 + 1];
            block_col[row_cursor[row]++] = bcol;
        }

        ASSERT_EQ(row_ptr[0], 0);
        ASSERT_EQ(row_ptr[1], 0);
        ASSERT_EQ(row_ptr[2], 4);
        ASSERT_EQ(row_ptr[3], 4);

        ASSERT_EQ(block_col[0], 0);
        ASSERT_EQ(block_col[1], 3);
        ASSERT_EQ(block_col[2], 7);
        ASSERT_EQ(block_col[3], 1);
    }

    // --- Sub-test C: empty (no blocks) ---
    {
        int64_t n_blocks = 0;
        int64_t n_rows_out = 5;

        std::vector<int32_t> row_ptr(n_rows_out + 1, 0);
        std::vector<int32_t> block_col(n_blocks);

        ASSERT_EQ(row_ptr.size(), (size_t)6);
        for (int64_t r = 0; r <= n_rows_out; r++) {
            ASSERT_EQ(row_ptr[r], 0);
        }
        ASSERT_EQ(block_col.size(), (size_t)0);
    }

    return true;
}

/* ========================================================================
 * Test 2: CPU sparse correction matches dense reference
 * ======================================================================== */

static bool test_sparse_correction_matches_dense(int64_t n_tokens) {
    const int64_t n_rows_out = 64;
    const int64_t n_cols = 128;
    const int64_t n_blocks = 5;

    int32_t idx_vals[2 * 5] = {
        0, 0,
        15, 1,
        32, 3,
        50, 2,
        63, 0,
    };

    std::vector<ggml_bf16_t> values_data(32 * n_blocks);
    for (int64_t ib = 0; ib < n_blocks; ib++) {
        for (int64_t j = 0; j < 32; j++) {
            values_data[ib * 32 + j] = ggml_fp32_to_bf16(0.1f * (ib + 1) + 0.01f * (float)j);
        }
    }

    std::vector<float> x_data(n_cols * n_tokens);
    for (int64_t i = 0; i < n_cols * n_tokens; i++) {
        x_data[i] = 0.01f * (float)(i % 64);
    }

    // Dense reference
    std::vector<float> expected(n_rows_out * n_tokens, 0.0f);
    for (int64_t ib = 0; ib < n_blocks; ib++) {
        int32_t row = idx_vals[ib * 2];
        int32_t block_col = idx_vals[ib * 2 + 1];
        int64_t col0 = (int64_t)block_col * 32;

        for (int64_t it = 0; it < n_tokens; it++) {
            float sum = 0.0f;
            for (int64_t j = 0; j < 32; j++) {
                float w = ggml_bf16_to_fp32(values_data[ib * 32 + j]);
                float a = x_data[(col0 + j) + it * n_cols];
                sum += w * a;
            }
            expected[row + it * n_rows_out] += sum;
        }
    }

    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();

    auto build_fn = [n_blocks, n_cols, n_tokens, n_rows_out](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_blocks);
        ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, n_blocks);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
        return ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows_out, n_cols);
    };

    auto set_data_fn = [&, idx_vals, values_data = std::move(values_data), x_data = std::move(x_data)](ggml_context * ctx) {
        ggml_tensor * t = ggml_get_first_tensor(ctx);
        ggml_backend_tensor_set(t, idx_vals, 0, 2 * n_blocks * sizeof(int32_t));
        t = ggml_get_next_tensor(ctx, t);
        ggml_backend_tensor_set(t, values_data.data(), 0, 32 * n_blocks * sizeof(ggml_bf16_t));
        t = ggml_get_next_tensor(ctx, t);
        ggml_backend_tensor_set(t, x_data.data(), 0, n_cols * n_tokens * sizeof(float));
    };

    auto check_fn = [n_rows_out, n_tokens, expected = std::move(expected)](const ggml_tensor * out) -> bool {
        std::vector<float> out_data(n_rows_out * n_tokens);
        ggml_backend_tensor_get(out, out_data.data(), 0, n_rows_out * n_tokens * sizeof(float));

        const float eps = 1e-4f;
        for (int64_t i = 0; i < n_rows_out * n_tokens; i++) {
            if (fabsf(out_data[i] - expected[i]) > eps) {
                fprintf(stderr, "  MISMATCH at idx %lld: got %f, expected %f (diff=%f)\n",
                        (long long)i, out_data[i], expected[i], fabsf(out_data[i] - expected[i]));
                return false;
            }
        }
        return true;
    };

    return run_cpu_graph(ctx_size, std::move(build_fn), std::move(set_data_fn), std::move(check_fn));
}

static bool test_sparse_correction() {
    fprintf(stderr, "  test_sparse_correction_matches_dense...\n");

    bool ok = true;
    std::vector<int64_t> n_tokens_list = {1, 8, 128};
    for (int64_t nt : n_tokens_list) {
        fprintf(stderr, "    n_tokens=%lld...", nt);
        if (!test_sparse_correction_matches_dense(nt)) {
            fprintf(stderr, " FAILED\n");
            ok = false;
        } else {
            fprintf(stderr, " ok\n");
        }
    }
    return ok;
}

/* ========================================================================
 * Test 3: Hybrid output correctness
 * ======================================================================== */

static bool test_hybrid_output(int64_t n_rows, int64_t n_cols, int64_t n_tokens) {
    ASSERT_TRUE(n_cols % Q8_BLOCK_SIZE == 0);

    int64_t n_blocks_per_row = n_cols / Q8_BLOCK_SIZE;

    std::vector<float> w_orig(n_rows * n_cols);
    for (int64_t i = 0; i < n_rows * n_cols; i++) {
        w_orig[i] = 0.1f * sinf(0.1f * (float)i) + 0.05f * cosf(0.05f * (float)i);
    }

    std::vector<float> x_data(n_cols * n_tokens);
    for (int64_t i = 0; i < n_cols * n_tokens; i++) {
        x_data[i] = 0.1f * sinf(0.2f * (float)i);
    }

    size_t q8_total_size = ggml_row_size(GGML_TYPE_Q8_0, n_rows * n_cols);
    std::vector<uint8_t> q8_data(q8_total_size);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w_orig.data(), q8_data.data(), 0, n_rows * n_cols, n_cols, nullptr);

    std::vector<std::pair<int32_t, int32_t>> outlier_blocks;
    if (n_rows > 1 && n_blocks_per_row > 1) {
        outlier_blocks.push_back({0, 0});
        outlier_blocks.push_back({(int32_t)(n_rows / 2), (int32_t)(n_blocks_per_row / 2)});
    } else {
        outlier_blocks.push_back({0, 0});
    }
    int64_t n_outlier = outlier_blocks.size();

    std::vector<ggml_bf16_t> values_data(32 * n_outlier);
    for (int64_t ib = 0; ib < n_outlier; ib++) {
        int32_t row = outlier_blocks[ib].first;
        int32_t bcol = outlier_blocks[ib].second;
        int64_t col0 = (int64_t)bcol * 32;
        for (int64_t j = 0; j < 32; j++) {
            values_data[ib * 32 + j] = ggml_fp32_to_bf16(w_orig[row * n_cols + col0 + j]);
        }
    }

    std::vector<int32_t> idx_data(2 * n_outlier);
    for (int64_t ib = 0; ib < n_outlier; ib++) {
        idx_data[ib * 2] = outlier_blocks[ib].first;
        idx_data[ib * 2 + 1] = outlier_blocks[ib].second;
    }

    // Zero outlier blocks
    size_t row_stride = ggml_row_size(GGML_TYPE_Q8_0, n_cols);
    for (const auto & ob : outlier_blocks) {
        int32_t row = ob.first;
        int32_t bcol = ob.second;
        size_t offset = (size_t)row * row_stride + (size_t)bcol * Q8_BLOCK_BYTES;
        memset(q8_data.data() + offset, 0, Q8_BLOCK_BYTES);
    }

    // --- Compute hybrid output ---
    {
        size_t ctx_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead();

        auto build_fn = [n_cols, n_rows, n_tokens, n_outlier](ggml_context * ctx) -> ggml_tensor * {
            ggml_tensor * w_q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_cols, n_rows);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
            ggml_tensor * idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_outlier);
            ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, n_outlier);

            ggml_tensor * base_out = ggml_mul_mat(ctx, w_q8, x);
            ggml_tensor * corr = ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows, n_cols);
            return ggml_add(ctx, base_out, corr);
        };

        auto set_data_fn = [&, q8_data = std::move(q8_data), x_data = std::move(x_data),
                            idx_data = std::move(idx_data), values_data = std::move(values_data)](ggml_context * ctx) {
            ggml_tensor * t = ggml_get_first_tensor(ctx);
            ggml_backend_tensor_set(t, q8_data.data(), 0, q8_data.size());
            t = ggml_get_next_tensor(ctx, t);
            ggml_backend_tensor_set(t, x_data.data(), 0, x_data.size() * sizeof(float));
            t = ggml_get_next_tensor(ctx, t);
            ggml_backend_tensor_set(t, idx_data.data(), 0, idx_data.size() * sizeof(int32_t));
            t = ggml_get_next_tensor(ctx, t);
            ggml_backend_tensor_set(t, values_data.data(), 0, values_data.size() * sizeof(ggml_bf16_t));
        };

        std::vector<float> hybrid_data;

        auto check_fn = [&](const ggml_tensor * out) -> bool {
            hybrid_data.resize(n_rows * n_tokens);
            ggml_backend_tensor_get(out, hybrid_data.data(), 0, n_rows * n_tokens * sizeof(float));
            return true;
        };

        if (!run_cpu_graph(ctx_size, build_fn, set_data_fn, check_fn)) {
            return false;
        }

        // --- Compute reference output ---
        {
            size_t ctx_size2 = ggml_tensor_overhead() * 64 + ggml_graph_overhead();

            auto build_fn2 = [n_cols, n_rows, n_tokens](ggml_context * ctx) -> ggml_tensor * {
                ggml_tensor * w_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_rows);
                ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
                return ggml_mul_mat(ctx, w_f32, x);
            };

            auto set_data_fn2 = [&, w_orig = std::move(w_orig), x_data = std::move(x_data)](ggml_context * ctx) {
                ggml_tensor * t = ggml_get_first_tensor(ctx);
                ggml_backend_tensor_set(t, w_orig.data(), 0, w_orig.size() * sizeof(float));
                t = ggml_get_next_tensor(ctx, t);
                ggml_backend_tensor_set(t, x_data.data(), 0, x_data.size() * sizeof(float));
            };

            std::vector<float> ref_data;

            auto check_fn2 = [&](const ggml_tensor * out) -> bool {
                ref_data.resize(n_rows * n_tokens);
                ggml_backend_tensor_get(out, ref_data.data(), 0, n_rows * n_tokens * sizeof(float));
                return true;
            };

            if (!run_cpu_graph(ctx_size2, build_fn2, set_data_fn2, check_fn2)) {
                return false;
            }

            // Compare
            const float eps = 5e-3f;
            for (int64_t i = 0; i < n_rows * n_tokens; i++) {
                if (fabsf(hybrid_data[i] - ref_data[i]) > eps) {
                    fprintf(stderr, "  MISMATCH at idx %lld: hybrid=%f ref=%f (diff=%f)\n",
                            (long long)i, hybrid_data[i], ref_data[i], fabsf(hybrid_data[i] - ref_data[i]));
                    return false;
                }
            }
        }
    }

    return true;
}

static bool test_hybrid() {
    fprintf(stderr, "  test_hybrid_output_correctness...\n");

    bool ok = true;

    {
        fprintf(stderr, "    attention (4096x4096)...");
        if (!test_hybrid_output(4096, 4096, 8)) {
            fprintf(stderr, " FAILED\n");
            ok = false;
        } else {
            fprintf(stderr, " ok\n");
        }
    }

    {
        fprintf(stderr, "    FFN up (14336x4096)...");
        if (!test_hybrid_output(14336, 4096, 4)) {
            fprintf(stderr, " FAILED\n");
            ok = false;
        } else {
            fprintf(stderr, " ok\n");
        }
    }

    {
        fprintf(stderr, "    FFN down (4096x14336)...");
        if (!test_hybrid_output(4096, 14336, 4)) {
            fprintf(stderr, " FAILED\n");
            ok = false;
        } else {
            fprintf(stderr, " ok\n");
        }
    }

    return ok;
}

/* ========================================================================
 * Test 4: Loader / tensor validation
 * ======================================================================== */

static bool test_validation_missing_sidecar() {
    fprintf(stderr, "    missing sidecar (n_blocks=0)...\n");

    const int64_t n_rows_out = 16;
    const int64_t n_cols = 64;
    const int64_t n_tokens = 2;
    const int64_t n_blocks = 0;

    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();

    auto build_fn = [n_blocks, n_cols, n_tokens, n_rows_out](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_blocks);
        ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, n_blocks);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
        return ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows_out, n_cols);
    };

    auto set_data_fn = [n_cols, n_tokens](ggml_context * ctx) {
        ggml_tensor * t = ggml_get_first_tensor(ctx);
        t = ggml_get_next_tensor(ctx, t); // skip idx
        t = ggml_get_next_tensor(ctx, t); // skip values
        std::vector<float> x_data(n_cols * n_tokens, 1.0f);
        ggml_backend_tensor_set(t, x_data.data(), 0, x_data.size() * sizeof(float));
    };

    auto check_fn = [n_rows_out, n_tokens](const ggml_tensor * out) -> bool {
        std::vector<float> out_data(n_rows_out * n_tokens);
        ggml_backend_tensor_get(out, out_data.data(), 0, n_rows_out * n_tokens * sizeof(float));
        for (int64_t i = 0; i < n_rows_out * n_tokens; i++) {
            if (fabsf(out_data[i]) > 1e-7f) {
                fprintf(stderr, "ASSERT_NEAR failed: out_data[%lld] (%f) != 0.0f eps=1e-7f at %s:%d\n",
                        (long long)i, out_data[i], __FILE__, __LINE__);
                return false;
            }
        }
        return true;
    };

    return run_cpu_graph(ctx_size, build_fn, set_data_fn, check_fn);
}

static bool test_validation_wrong_type() {
    fprintf(stderr, "    correct types produce expected output...\n");

    const int64_t n_rows_out = 8;
    const int64_t n_cols = 64;
    const int64_t n_tokens = 1;
    const int64_t n_blocks = 1;

    int32_t idx_vals[2] = {0, 0};
    std::vector<ggml_bf16_t> values_data(32);
    for (int64_t j = 0; j < 32; j++) {
        values_data[j] = ggml_fp32_to_bf16(1.0f);
    }
    std::vector<float> x_data(n_cols, 1.0f);

    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();

    auto build_fn = [n_blocks, n_cols, n_tokens, n_rows_out](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_blocks);
        ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, n_blocks);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
        return ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows_out, n_cols);
    };

    auto set_data_fn = [&, idx_vals, values_data = std::move(values_data), x_data = std::move(x_data)](ggml_context * ctx) {
        ggml_tensor * t = ggml_get_first_tensor(ctx);
        ggml_backend_tensor_set(t, idx_vals, 0, 2 * sizeof(int32_t));
        t = ggml_get_next_tensor(ctx, t);
        ggml_backend_tensor_set(t, values_data.data(), 0, 32 * sizeof(ggml_bf16_t));
        t = ggml_get_next_tensor(ctx, t);
        ggml_backend_tensor_set(t, x_data.data(), 0, x_data.size() * sizeof(float));
    };

    auto check_fn = [n_rows_out, n_tokens](const ggml_tensor * out) -> bool {
        std::vector<float> out_data(n_rows_out * n_tokens);
        ggml_backend_tensor_get(out, out_data.data(), 0, n_rows_out * n_tokens * sizeof(float));
        if (fabsf(out_data[0] - 32.0f) > 1e-2f) {
            fprintf(stderr, "check failed: out_data[0]=%f expected 32.0f\n", out_data[0]);
            return false;
        }
        for (int64_t i = 1; i < n_rows_out; i++) {
            if (fabsf(out_data[i]) > 1e-7f) {
                fprintf(stderr, "check failed: out_data[%lld]=%f expected 0.0f\n", (long long)i, out_data[i]);
                return false;
            }
        }
        return true;
    };

    return run_cpu_graph(ctx_size, build_fn, set_data_fn, check_fn);
}

static bool test_validation_shape_mismatch() {
    fprintf(stderr, "    out-of-range blocks skipped gracefully...\n");

    const int64_t n_rows_out = 8;
    const int64_t n_cols = 64;
    const int64_t n_tokens = 1;
    const int64_t n_blocks = 3;

    int32_t idx_vals[6] = {0, 0, 100, 0, 3, 1};
    std::vector<ggml_bf16_t> values_data(32 * n_blocks);
    for (int64_t ib = 0; ib < n_blocks; ib++) {
        for (int64_t j = 0; j < 32; j++) {
            values_data[ib * 32 + j] = ggml_fp32_to_bf16((float)(ib + 1));
        }
    }
    std::vector<float> x_data(n_cols, 1.0f);

    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();

    auto build_fn = [n_blocks, n_cols, n_tokens, n_rows_out](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_blocks);
        ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, n_blocks);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
        return ggml_mul_mat_outlier_blocks(ctx, idx, values, x, n_rows_out, n_cols);
    };

    auto set_data_fn = [&, idx_vals, values_data = std::move(values_data), x_data = std::move(x_data)](ggml_context * ctx) {
        ggml_tensor * t = ggml_get_first_tensor(ctx);
        ggml_backend_tensor_set(t, idx_vals, 0, 6 * sizeof(int32_t));
        t = ggml_get_next_tensor(ctx, t);
        ggml_backend_tensor_set(t, values_data.data(), 0, values_data.size() * sizeof(ggml_bf16_t));
        t = ggml_get_next_tensor(ctx, t);
        ggml_backend_tensor_set(t, x_data.data(), 0, x_data.size() * sizeof(float));
    };

    auto check_fn = [n_rows_out, n_tokens](const ggml_tensor * out) -> bool {
        std::vector<float> out_data(n_rows_out * n_tokens);
        ggml_backend_tensor_get(out, out_data.data(), 0, n_rows_out * n_tokens * sizeof(float));
        if (fabsf(out_data[0] - 32.0f) > 1e-2f) {
            fprintf(stderr, "check failed: out_data[0]=%f expected 32.0f\n", out_data[0]);
            return false;
        }
        if (fabsf(out_data[3] - 64.0f) > 1e-2f) {
            fprintf(stderr, "check failed: out_data[3]=%f expected 64.0f\n", out_data[3]);
            return false;
        }
        for (int64_t i = 0; i < n_rows_out; i++) {
            if (i == 0 || i == 3) continue;
            if (fabsf(out_data[i]) > 1e-7f) {
                fprintf(stderr, "check failed: out_data[%lld]=%f expected 0.0f\n", (long long)i, out_data[i]);
                return false;
            }
        }
        return true;
    };

    return run_cpu_graph(ctx_size, build_fn, set_data_fn, check_fn);
}

static bool test_validation() {
    fprintf(stderr, "  test_validation...\n");

    bool ok = true;
    if (!test_validation_missing_sidecar()) { ok = false; }
    if (!test_validation_wrong_type()) { ok = false; }
    if (!test_validation_shape_mismatch()) { ok = false; }
    return ok;
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char * argv[]) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    ggml_backend_load_all();

    fprintf(stderr, "=== Q8_0_BF16_OUTLIER CPU runtime tests ===\n");

    int num_failed = 0;

    fprintf(stderr, "[1/4] CSR construction correctness:\n");
    if (!test_csr_construction()) {
        fprintf(stderr, "  FAILED\n");
        num_failed++;
    } else {
        fprintf(stderr, "  PASSED\n");
    }

    fprintf(stderr, "[2/4] CPU sparse correction matches dense reference:\n");
    if (!test_sparse_correction()) {
        fprintf(stderr, "  FAILED\n");
        num_failed++;
    } else {
        fprintf(stderr, "  PASSED\n");
    }

    fprintf(stderr, "[3/4] Hybrid output correctness:\n");
    if (!test_hybrid()) {
        fprintf(stderr, "  FAILED\n");
        num_failed++;
    } else {
        fprintf(stderr, "  PASSED\n");
    }

    fprintf(stderr, "[4/4] Loader / tensor validation:\n");
    if (!test_validation()) {
        fprintf(stderr, "  FAILED\n");
        num_failed++;
    } else {
        fprintf(stderr, "  PASSED\n");
    }

    fprintf(stderr, "=== %d/%d tests failed ===\n", num_failed, 4);
    return num_failed > 0 ? 1 : 0;
}
