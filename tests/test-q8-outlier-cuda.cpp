#ifdef GGML_USE_CUDA

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const double kNmseTolerance = 1e-3;

struct test_params {
    int64_t  n_rows_out;
    int64_t  n_cols;
    int64_t  n_tokens;
    double   outlier_fraction;
    std::string shape_name;

    std::string desc() const {
        char buf[256];
        snprintf(buf, sizeof(buf), "shape=%s n_tokens=%lld fraction=%.4f",
                 shape_name.c_str(), (long long)n_tokens, outlier_fraction);
        return buf;
    }
};

static void fill_idx(ggml_tensor * t, int64_t n_rows_out, int64_t n_cols, std::mt19937 & gen) {
    int64_t n_blocks = t->ne[1];
    int64_t n_block_cols = n_cols / 32;
    GGML_ASSERT(n_block_cols > 0);

    std::vector<int32_t> data(2 * n_blocks);
    std::uniform_int_distribution<int32_t> row_dist(0, static_cast<int32_t>(n_rows_out - 1));
    std::uniform_int_distribution<int32_t> col_dist(0, static_cast<int32_t>(n_block_cols - 1));

    for (int64_t i = 0; i < n_blocks; i++) {
        data[i * 2 + 0] = row_dist(gen);
        data[i * 2 + 1] = col_dist(gen);
    }

    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int32_t));
}

static void fill_values(ggml_tensor * t, std::mt19937 & gen) {
    int64_t nels = ggml_nelements(t);
    std::vector<float> src(nels);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int64_t i = 0; i < nels; i++) {
        src[i] = dist(gen);
    }

    std::vector<ggml_bf16_t> dst(nels);
    ggml_fp32_to_bf16_row(src.data(), dst.data(), nels);

    ggml_backend_tensor_set(t, dst.data(), 0, dst.size() * sizeof(ggml_bf16_t));
}

static void fill_x(ggml_tensor * t, std::mt19937 & gen) {
    int64_t nels = ggml_nelements(t);
    std::vector<float> data(nels);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int64_t i = 0; i < nels; i++) {
        data[i] = dist(gen);
    }

    ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
}

static std::vector<float> tensor_to_f32(ggml_tensor * t) {
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));

    std::vector<float> out(ggml_nelements(t));
    const ggml_type_traits_t traits = ggml_get_type_traits(t->type);
    int64_t bs = ggml_blck_size(t->type);
    std::vector<float> tmp(bs);

    int64_t idx = 0;
    for (int64_t i = 0; i < ggml_nelements(t); i += bs) {
        if (t->type == GGML_TYPE_F32) {
            for (int64_t j = 0; j < bs; j++) {
                out[idx + j] = ((const float *)raw.data())[i / bs + j];
            }
        } else {
            traits->to_float(raw.data() + t->nb[0] * (i / bs), tmp.data(), bs);
            for (int64_t j = 0; j < bs; j++) {
                out[idx + j] = tmp[j];
            }
        }
        idx += bs;
    }
    return out;
}

static double nmse(const float * a, const float * b, size_t n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (size_t i = 0; i < n; i++) {
        mse_ab += (a[i] - b[i]) * (a[i] - b[i]);
        mse_a0 += a[i] * a[i];
    }
    return mse_a0 > 0.0 ? mse_ab / mse_a0 : mse_ab;
}

static bool run_test(const test_params & p, ggml_backend_t cpu, ggml_backend_t cuda) {
    int64_t n_block_cols = p.n_cols / 32;
    int64_t total_blocks = p.n_rows_out * n_block_cols;
    int64_t n_blocks = std::max<int64_t>(1ll, static_cast<int64_t>(std::ceil(total_blocks * p.outlier_fraction)));

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "  [FAIL] failed to init context\n");
        return false;
    }

    // idx: [2, n_blocks], I32
    ggml_tensor * idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_blocks);
    ggml_set_name(idx, "idx");

    // values: [32, n_blocks], BF16
    ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, n_blocks);
    ggml_set_name(values, "values");

    // x: [n_cols, n_tokens], F32
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.n_cols, p.n_tokens);
    ggml_set_name(x, "x");

    // out: [n_rows_out, n_tokens], F32
    ggml_tensor * out = ggml_mul_mat_outlier_blocks(ctx, idx, values, x, p.n_rows_out, p.n_cols);
    ggml_set_name(out, "out");

    // Check backend support
    bool supported = true;
    for (ggml_backend_t be : {cpu, cuda}) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (!ggml_backend_supports_op(be, t)) {
                supported = false;
                break;
            }
        }
        if (!supported) break;
    }

    if (!supported) {
        ggml_free(ctx);
        printf("  [SKIP] not supported\n");
        return true;
    }

    // Allocate on CPU backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) {
        fprintf(stderr, "  [FAIL] failed to allocate tensors on CPU\n");
        ggml_free(ctx);
        return false;
    }

    // Build graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Initialize tensors with deterministic seed
    std::mt19937 gen(42);
    fill_idx(idx, p.n_rows_out, p.n_cols, gen);
    fill_values(values, gen);
    fill_x(x, gen);

    // Compare CPU vs CUDA
    bool ok = true;

    auto callback = [](int /*index*/, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
        if (t1->op == GGML_OP_NONE) {
            return true;
        }

        std::vector<float> * results = static_cast<std::vector<float> **>(user_data);
        std::vector<float> f1 = tensor_to_f32(t1);
        std::vector<float> f2 = tensor_to_f32(t2);

        for (size_t i = 0; i < f1.size(); i++) {
            if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                fprintf(stderr, "  NaN at index %zu\n", i);
                ok = false;
                return true;
            }
        }

        double e = nmse(f1.data(), f2.data(), f1.size());
        results->push_back(e);
        if (e > kNmseTolerance) {
            fprintf(stderr, "  NMSE = %.9f > %.9f\n", e, kNmseTolerance);
            ok = false;
        }
        return true;
    };

    std::vector<float> errors;
    ggml_tensor * test_nodes[] = {out};
    bool cmp_ok = ggml_backend_compare_graph_backend(cpu, cuda, gf, callback, &errors, test_nodes, 1);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (!cmp_ok) {
        fprintf(stderr, "  [FAIL] compare failed\n");
        return false;
    }

    if (!ok) {
        fprintf(stderr, "  [FAIL] numerical mismatch\n");
        return false;
    }

    return true;
}

static std::vector<test_params> make_test_cases() {
    std::vector<test_params> cases;

    std::array<int64_t, 3> batch_sizes = {1, 8, 128};
    std::array<double, 5>  fractions   = {0.001, 0.005, 0.01, 0.02, 0.05};

    struct shape_def {
        int64_t  n_rows_out;
        int64_t  n_cols;
        std::string name;
    };

    std::vector<shape_def> shapes = {
        {4096,  4096,  "attention"},
        {14336, 4096,  "ffn_up"},
        {4096,  14336, "ffn_down"},
    };

    for (const auto & shape : shapes) {
        for (int64_t nt : batch_sizes) {
            for (double frac : fractions) {
                cases.push_back({
                    shape.n_rows_out,
                    shape.n_cols,
                    nt,
                    frac,
                    shape.name
                });
            }
        }
    }

    return cases;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "Failed to init CPU backend\n");
        return 1;
    }

    // Find first CUDA backend
    ggml_backend_t cuda = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            strcmp(ggml_backend_dev_name(dev), "CUDA") == 0) {
            cuda = ggml_backend_dev_init(dev, nullptr);
            if (cuda) break;
        }
    }

    if (!cuda) {
        fprintf(stderr, "No CUDA backend available\n");
        ggml_backend_free(cpu);
        return 0;
    }

    printf("Testing Q8_0_BF16_OUTLIER: CPU vs CUDA\n");
    printf("CUDA device: %s\n\n", ggml_backend_dev_description(ggml_backend_get_device(cuda)));

    std::vector<test_params> cases = make_test_cases();
    size_t passed = 0;
    size_t failed = 0;

    for (size_t i = 0; i < cases.size(); i++) {
        const test_params & p = cases[i];
        printf("[%zu/%zu] %s\n", i + 1, cases.size(), p.desc().c_str());

        if (run_test(p, cpu, cuda)) {
            printf("  OK\n");
            passed++;
        } else {
            printf("  FAIL\n");
            failed++;
        }
    }

    printf("\n---\n");
    printf("%zu/%zu tests passed\n", passed, passed + failed);

    ggml_backend_free(cuda);
    ggml_backend_free(cpu);

    return failed > 0 ? 1 : 0;
}

#else // GGML_USE_CUDA

int main() {
    printf("CUDA not available, skipping test\n");
    return 0;
}

#endif
