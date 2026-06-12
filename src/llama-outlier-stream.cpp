#include "llama-outlier-stream.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <regex>

llama_outlier_stream_cache::~llama_outlier_stream_cache() {
    for (auto & [name, entry] : entries) {
        free_entry_gpu(entry);
    }
}

void llama_outlier_stream_cache::add_entry(
        const std::string & name,
        const int32_t     * idx_data,
        const uint8_t     * values_data,
        ggml_type           values_type,
        int64_t             n_blocks,
        int64_t             n_rows_out,
        int64_t             n_cols) {

    llama_outlier_cache_entry entry;
    entry.name        = name;
    entry.values_type = values_type;
    entry.n_blocks    = n_blocks;
    entry.n_rows_out  = n_rows_out;
    entry.n_cols      = n_cols;

    entry.idx_data.assign(idx_data, idx_data + n_blocks * 2);

    size_t elem_bytes = (values_type == GGML_TYPE_Q8_0)
        ? (size_t)34 : (size_t)(32 * sizeof(ggml_bf16_t));
    size_t total = (size_t)n_blocks * elem_bytes;
    entry.values_data.assign(values_data, values_data + total);

    entries[name] = std::move(entry);
}

void llama_outlier_stream_cache::finalize() {
    sorted_names.clear();
    sorted_names.reserve(entries.size());
    for (const auto & [n, e] : entries) {
        sorted_names.push_back(n);
    }

    // Sort by (layer_number, tensor_type) for predictive prefetch.
    // This matches the order that tensors are encountered during
    // graph building (blk.0.attn_q → attn_k → ... → ffn_down →
    // blk.1.attn_q → ...).
    //
    // Token_embd and output come first/last respectively.
    std::sort(sorted_names.begin(), sorted_names.end(),
        [](const std::string & a, const std::string & b) {
            // Extract layer number (or use -1 for non-layer tensors)
            auto extract = [](const std::string & s) -> std::pair<int, std::string> {
                std::regex re(R"(blk\.(\d+)\.)");
                std::smatch m;
                if (std::regex_search(s, m, re)) {
                    return {std::stoi(m[1]), s};
                }
                // Non-layer tensors: token_embd → -2, output → -1
                if (s.find("token_embd") != std::string::npos) return {-2, s};
                if (s.find("output") != std::string::npos)     return {-1, s};
                return {0, s};
            };
            auto pa = extract(a);
            auto pb = extract(b);
            if (pa.first != pb.first) return pa.first < pb.first;
            return pa.second < pb.second;
        });

    LLAMA_LOG_INFO("[outlier-stream] finalized %zu entries, sorted for predictive prefetch\n",
            sorted_names.size());
}

bool llama_outlier_stream_cache::has_entry(const std::string & name) const {
    return entries.find(name) != entries.end();
}

bool llama_outlier_stream_cache::ensure_gpu(ggml_backend_t backend, const std::string & name) {
    auto it = entries.find(name);
    if (it == entries.end()) return false;

    auto & entry = it->second;
    if (entry.loaded) {
        touch(name);
        return true;
    }

    while (loaded_count >= window_size) {
        evict_lru(backend);
    }

    if (!upload_entry(backend, entry)) return false;

    touch(name);
    loaded_count++;

    // Prefetch the next tensors in sorted order
    prefetch_ahead(backend, name);

    return true;
}

ggml_tensor * llama_outlier_stream_cache::get_gpu_idx(const std::string & name) const {
    auto it = entries.find(name);
    return (it != entries.end() && it->second.loaded) ? it->second.idx_gpu : nullptr;
}

ggml_tensor * llama_outlier_stream_cache::get_gpu_values(const std::string & name) const {
    auto it = entries.find(name);
    return (it != entries.end() && it->second.loaded) ? it->second.values_gpu : nullptr;
}

void llama_outlier_stream_cache::touch(const std::string & name) {
    auto it = std::find(lru.begin(), lru.end(), name);
    if (it != lru.end()) lru.erase(it);
    lru.push_front(name);
}

size_t llama_outlier_stream_cache::total_cpu_bytes() const {
    size_t total = 0;
    for (const auto & [n, e] : entries) {
        total += e.idx_data.size() * sizeof(int32_t);
        total += e.values_data.size();
    }
    return total;
}

size_t llama_outlier_stream_cache::total_gpu_bytes() const {
    size_t total = 0;
    for (const auto & [n, e] : entries) {
        if (e.loaded) total += e.gpu_bytes();
    }
    return total;
}

void llama_outlier_stream_cache::evict_lru(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
        auto eit = entries.find(*it);
        if (eit != entries.end() && eit->second.loaded) {
            LLAMA_LOG_DEBUG("[outlier-stream] evicting '%s' from GPU\n", it->c_str());
            free_entry_gpu(eit->second);
            eit->second.loaded = false;
            loaded_count--;
            return;
        }
    }
}

void llama_outlier_stream_cache::prefetch_ahead(
        ggml_backend_t backend, const std::string & current) {

    // Find current position in sorted order
    auto it_cur = std::find(sorted_names.begin(), sorted_names.end(), current);
    if (it_cur == sorted_names.end()) return;

    // Load the next N unloaded entries (up to window_size)
    size_t idx = (size_t)(it_cur - sorted_names.begin());
    for (size_t i = idx + 1;
         i < sorted_names.size() && loaded_count < window_size;
         i++) {

        const auto & n = sorted_names[i];
        auto eit = entries.find(n);
        if (eit == entries.end() || eit->second.loaded) continue;

        if (!upload_entry(backend, eit->second)) break;

        touch(n);
        loaded_count++;

        LLAMA_LOG_DEBUG("[outlier-stream] prefetched '%s' (ahead of '%s', %d/%d loaded)\n",
                n.c_str(), current.c_str(), loaded_count, window_size);
    }
}

bool llama_outlier_stream_cache::upload_entry(
        ggml_backend_t backend, llama_outlier_cache_entry & entry) {

    const int64_t n_blocks = entry.n_blocks;

    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2 + 512,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ctx_params);

    ggml_tensor * idx_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_blocks);
    ggml_tensor * val_t = ggml_new_tensor_2d(ctx, entry.values_type, 32, n_blocks);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        LLAMA_LOG_WARN("[outlier-stream] failed to alloc GPU buffer for '%s'\n",
                entry.name.c_str());
        return false;
    }

    ggml_backend_tensor_set(idx_t, entry.idx_data.data(), 0,
            n_blocks * 2 * sizeof(int32_t));
    ggml_backend_tensor_set(val_t, entry.values_data.data(), 0,
            entry.values_data.size());

    entry.idx_gpu    = idx_t;
    entry.values_gpu = val_t;
    entry.loaded     = true;

    return true;
}

void llama_outlier_stream_cache::free_entry_gpu(
        llama_outlier_cache_entry & entry) {

    if (!entry.loaded) return;

    if (entry.idx_gpu && entry.idx_gpu->buffer) {
        ggml_backend_buffer_free(entry.idx_gpu->buffer);
    }

    entry.idx_gpu    = nullptr;
    entry.values_gpu = nullptr;
    entry.loaded     = false;
}