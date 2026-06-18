#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

// Predictive streaming cache for outlier sidecar data.
// Sidecar tensors (idx + values) stay on CPU and are uploaded to GPU
// on-demand with a sliding window. The cache keeps N most-recently-used
// sidecars in VRAM and starts prefetching the next ones before they are
// needed, so the upload overlaps with compute.
//
// Usage:
//   1. At model load: add_entry() for each outlier-protected tensor
//   2. At graph build time: ensure_gpu() before using sidecars
//   3. After each layer: prefetch() the next few tensor names
//
// The window size trades VRAM for latency (default 6 = current + 5 ahead).

struct llama_outlier_cache_entry {
    std::string name;

    // CPU-resident data (copied from GGUF at load time)
    std::vector<int32_t>  idx_data;       // [n_blocks * 2]
    std::vector<uint8_t>  values_data;    // [n_blocks * element_bytes]
    ggml_type             values_type = GGML_TYPE_BF16;
    int64_t               n_blocks    = 0;
    int64_t               n_rows_out  = 0;
    int64_t               n_cols      = 0;

    // GPU-resident tensors — valid only when loaded=true
    bool             loaded    = false;
    ggml_tensor    * idx_gpu    = nullptr;
    ggml_tensor    * values_gpu = nullptr;

    // Latched: entry was used in the current graph build and must NOT be
    // evicted until after compute completes. Call release_all_latches()
    // between forward passes.
    bool             latched   = false;

    // Total byte size on GPU (idx + values)
    size_t gpu_bytes() const {
        size_t elem;
        if (values_type == GGML_TYPE_Q8_0) {
            elem = 34;                                 // sizeof(block_q8_0) = 34 bytes
        } else if (values_type == GGML_TYPE_I8) {
            elem = 16;                                 // 16 bytes of packed nibbles
        } else {
            elem = 32 * sizeof(ggml_bf16_t);           // 64 bytes per block
        }
        return (size_t)(n_blocks * 2 * sizeof(int32_t))  // idx [2, n_blocks]
             + (size_t)(n_blocks * elem);                  // values per block
    }
};

struct llama_outlier_stream_cache {
    // Maximum number of tensors kept in VRAM simultaneously.
    // Must cover one full transformer layer + lookahead for next layer.
    // A typical dense layer has up to 8 weight tensors (attn_q, attn_k, attn_v,
    // attn_output, ffn_gate, ffn_up, ffn_down, plus optional extra like nextn.eh_proj).
    // Window=12 gives room for one full layer + prefetch headroom.
    static constexpr int DEFAULT_WINDOW_SIZE = 12;

    // All known entries (CPU data always resident)
    std::unordered_map<std::string, llama_outlier_cache_entry> entries;

    // LRU order: front() = most recently used, back() = least recently used
    std::deque<std::string> lru;

    // Tracks which entries are currently loaded on GPU
    int window_size     = DEFAULT_WINDOW_SIZE;
    int loaded_count    = 0;

    // Sorted tensor names (by layer, then by type) for predictive prefetch.
    // Populated by finalize() after all entries are added.
    std::vector<std::string> sorted_names;

    ~llama_outlier_stream_cache();

    // --- Lifecycle ---

    // Register a new entry with CPU-side data (called at model load).
    void add_entry(
        const std::string & name,
        const int32_t     * idx_data,
        const uint8_t     * values_data,
        ggml_type           values_type,
        int64_t             n_blocks,
        int64_t             n_rows_out,
        int64_t             n_cols);

    // Must be called after all add_entry() calls. Sorts entries by
    // layer number + tensor type for deterministic prefetch order.
    void finalize();

    // Check if an entry exists
    bool has_entry(const std::string & name) const;

    // --- GPU operations ---

    // Ensure the entry for `name` is loaded on GPU. Also prefetches
    // the next N tensors (up to window_size) in sorted order.
    // Latches the entry so it cannot be evicted until release_all_latches().
    bool ensure_gpu(ggml_backend_t backend, const std::string & name);

    // Release all latched entries — call between forward passes so
    // entries from the previous pass can be evicted.
    void release_all_latches();

    // Get GPU tensor pointers. Call ensure_gpu() first.
    ggml_tensor * get_gpu_idx(const std::string & name) const;
    ggml_tensor * get_gpu_values(const std::string & name) const;

    // Mark a tensor as recently used (moves to front of LRU).
    void touch(const std::string & name);

    // --- Statistics ---

    int  gpu_loaded_count() const { return loaded_count; }
    int  total_entries()    const { return (int)entries.size(); }
    size_t total_cpu_bytes() const;
    size_t total_gpu_bytes() const;

private:
    void evict_lru(ggml_backend_t backend);
    bool upload_entry(ggml_backend_t backend, llama_outlier_cache_entry & entry);
    void free_entry_gpu(llama_outlier_cache_entry & entry);

    // Look ahead in sorted order and load unloaded entries
    void prefetch_ahead(ggml_backend_t backend, const std::string & current);
};