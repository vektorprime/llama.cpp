#ifndef LLAMA_Q8_OUTLIER_H
#define LLAMA_Q8_OUTLIER_H

#include <cstdint>

// Q8_0_BF16_OUTLIER format constants
static constexpr int32_t LLAMA_Q8_OUTLIER_VERSION = 1;
static constexpr int32_t LLAMA_Q8_OUTLIER_BLOCK_SIZE = 32;

// GGUF metadata key strings
static constexpr const char * LLAMA_Q8_OUTLIER_VERSION_KEY = "llama.q8_outlier.version";
static constexpr const char * LLAMA_Q8_OUTLIER_BLOCK_SIZE_KEY = "llama.q8_outlier.block_size";
static constexpr const char * LLAMA_Q8_OUTLIER_BASE_TYPE_KEY = "llama.q8_outlier.base_type";
static constexpr const char * LLAMA_Q8_OUTLIER_VALUE_TYPE_KEY = "llama.q8_outlier.value_type";
static constexpr const char * LLAMA_Q8_OUTLIER_INDEX_ENCODING_KEY = "llama.q8_outlier.index_encoding";
static constexpr const char * LLAMA_Q8_OUTLIER_STORE_KEY = "llama.q8_outlier.store";
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_COUNT_KEY = "llama.q8_outlier.tensor_count";

// Per-tensor key suffixes
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_NAME_SUFFIX = ".name";
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_INDEX_SUFFIX = ".index";
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_VALUES_SUFFIX = ".values";
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_N_BLOCKS_SUFFIX = ".n_blocks";

// Sidecar tensor name suffixes
static constexpr const char * LLAMA_Q8_OUTLIER_IDX_SUFFIX = ".outlier_idx";
static constexpr const char * LLAMA_Q8_OUTLIER_VALUES_SUFFIX = ".outlier_bf16";

// Q4_0_BF16_OUTLIER format constants (shares block size with Q8)
static constexpr int32_t LLAMA_Q4_OUTLIER_BLOCK_SIZE = 32;

// Q4 GGUF metadata key strings (separate namespace from Q8)
static constexpr const char * LLAMA_Q4_OUTLIER_VERSION_KEY = "llama.q4_outlier.version";
static constexpr const char * LLAMA_Q4_OUTLIER_BLOCK_SIZE_KEY = "llama.q4_outlier.block_size";
static constexpr const char * LLAMA_Q4_OUTLIER_BASE_TYPE_KEY = "llama.q4_outlier.base_type";
static constexpr const char * LLAMA_Q4_OUTLIER_VALUE_TYPE_KEY = "llama.q4_outlier.value_type";
static constexpr const char * LLAMA_Q4_OUTLIER_INDEX_ENCODING_KEY = "llama.q4_outlier.index_encoding";
static constexpr const char * LLAMA_Q4_OUTLIER_STORE_KEY = "llama.q4_outlier.store";
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_COUNT_KEY = "llama.q4_outlier.tensor_count";

// Q4 per-tensor key suffixes
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_NAME_SUFFIX = ".name";
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_INDEX_SUFFIX = ".index";
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_VALUES_SUFFIX = ".values";
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_N_BLOCKS_SUFFIX = ".n_blocks";

// Q4 sidecar tensor name suffixes (same as Q8)
static constexpr const char * LLAMA_Q4_OUTLIER_IDX_SUFFIX = ".outlier_idx";
static constexpr const char * LLAMA_Q4_OUTLIER_VALUES_SUFFIX = ".outlier_bf16";

#endif
