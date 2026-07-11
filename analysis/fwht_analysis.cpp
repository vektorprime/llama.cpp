// FWHT Analysis Tool: prints per-position statistics of FWHT-transformed weights
// Build: cmake --build build --target fwht_analysis  (requires CMakeLists addition)
// Or: build manually with: g++ -O3 -o fwht_analysis fwht_analysis.cpp -I../include -I../ggml/include \
//     -L../build/bin -lllama -lggml -lggml-cpu -lggml-base -lm

#include "ggml.h"
#include "ggml-quants.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define FWHT_SIZE 256

extern "C" void fwht_256(float * x);

struct FWHTStats {
    double sum[FWHT_SIZE]  = {};
    double sum2[FWHT_SIZE] = {};
    float  min_val[FWHT_SIZE];
    float  max_val[FWHT_SIZE];
    int64_t count = 0;

    FWHTStats() {
        for (int i = 0; i < FWHT_SIZE; i++) {
            min_val[i] =  INFINITY;
            max_val[i] = -INFINITY;
        }
    }

    void add_block(const float * x) {
        for (int i = 0; i < FWHT_SIZE; i++) {
            float v = x[i];
            sum[i]   += v;
            sum2[i]  += (double)v * v;
            if (v < min_val[i]) min_val[i] = v;
            if (v > max_val[i]) max_val[i] = v;
        }
        count++;
    }

    void print() {
        printf("Position-level FWHT statistics (%lld blocks):\n", (long long)count);
        printf("%4s %12s %12s %12s %10s\n", "pos", "mean", "std", "var", "range");
        for (int i = 0; i < FWHT_SIZE; i++) {
            double mean = sum[i] / count;
            double var  = sum2[i] / count - mean * mean;
            double std  = std::sqrt(std::max(0.0, var));
            if (i < 10 || i > FWHT_SIZE - 10 || std > 0.05) {
                printf("%4d %12.6f %12.6f %12.6f [%8.4f, %8.4f]\n",
                       i, mean, std, var, min_val[i], max_val[i]);
            }
        }
    }
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [tensor_filter]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * filter = argc > 2 ? argv[2] : "blk";

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };

    struct gguf_context * ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "Failed to open GGUF: %s\n", model_path);
        return 1;
    }

    int n_tensors = gguf_get_n_tensors(ctx);
    printf("Model has %d tensors\n", n_tensors);

    // Prepare GGML context for dequantization
    struct ggml_init_params gparams = {
        /*.mem_size   = */ 16 * 1024,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };

    struct ggml_context * gctx = ggml_init(gparams);

    FWHTStats stats;
    FWHTStats dc_stats;
    FWHTStats ac_stats;

    int total_blocks = 0;
    int total_tensors = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        if (!name || !strstr(name, filter)) continue;

        struct ggml_tensor * t = ggml_get_tensor(gctx, name);
        if (t == nullptr) {
            t = ggml_new_tensor(gctx, gguf_get_tensor_type(ctx, i), 0, nullptr);
        }

        size_t offset = gguf_get_tensor_offset(ctx, i);
        int n_dims = gguf_get_tensor_n_dims(ctx, i);
        if (n_dims < 2) continue;

        int64_t ne0 = t->ne[0];
        int64_t ne1 = t->ne[1];
        if (ne0 % FWHT_SIZE != 0) continue;

        int n_blocks_per_row = ne0 / FWHT_SIZE;
        int64_t n_rows = ne0 > 0 ? gguf_get_tensor_size(ctx, i) / gguf_get_tensor_type_size(ctx, i) / ne0 : 0;

        // This is getting complicated with tensor sizing
        printf("  %s\n", name);
        total_tensors++;
    }

    printf("Total tensors matching pattern: %d\n", total_tensors);
    printf("NOTE: This tool needs linking against gguf/ggml for tensor data access.\n");
    printf("Run Python analysis instead: python3 analysis/fwht_analysis.py\n");

    ggml_free(gctx);
    gguf_free(ctx);
    return 0;
}
