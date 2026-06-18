#ifndef LLAMA_Q8_OUTLIER_H
#define LLAMA_Q8_OUTLIER_H

#include <cstdint>
#include <cmath>

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
static constexpr const char * LLAMA_Q8_OUTLIER_COLUMN_PERM_KEY = "llama.q8_outlier.column_perm";

// Per-tensor key suffixes
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_NAME_SUFFIX = ".name";
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_INDEX_SUFFIX = ".index";
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_VALUES_SUFFIX = ".values";
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_N_BLOCKS_SUFFIX = ".n_blocks";
static constexpr const char * LLAMA_Q8_OUTLIER_TENSOR_COLUMN_PERM_SUFFIX = ".column_perm";

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
static constexpr const char * LLAMA_Q4_OUTLIER_COLUMN_PERM_KEY = "llama.q4_outlier.column_perm";

// Q4 per-tensor key suffixes
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_NAME_SUFFIX = ".name";
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_INDEX_SUFFIX = ".index";
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_VALUES_SUFFIX = ".values";
static constexpr const char * LLAMA_Q4_OUTLIER_TENSOR_N_BLOCKS_SUFFIX = ".n_blocks";

// Q4 sidecar tensor name suffixes (same as Q8)
static constexpr const char * LLAMA_Q4_OUTLIER_IDX_SUFFIX = ".outlier_idx";
static constexpr const char * LLAMA_Q4_OUTLIER_VALUES_SUFFIX = ".outlier_bf16";

// Nibble-diff encoding: 4 bits per weight, 32 weights = 16 bytes per block
static constexpr int32_t LLAMA_OUTLIER_NIBBLE_BLOCK_BYTES = 16;

// Nibble-diff encoding: convert float diff to 4-bit nibble
// Bit layout (MSB to LSB):
//   bit 3: enable (1 = apply diff, 0 = skip)
//   bit 2: sign     (0 = negative, 1 = positive)
//   bit 1: zero cnt (0 = 0.0X, 1 = 0.00X)
//   bit 0: digit    (0 = 1, 1 = 2)
//
// Possible values: -0.02, -0.01, -0.002, -0.001, 0 (disabled), +0.001, +0.002, +0.01, +0.02
static inline uint8_t llama_outlier_nibble_diff_encode(float diff) {
    if (fabsf(diff) < 0.0005f) {
        return 0x00; // disabled, diff too small
    }

    uint8_t nibble = 0x08; // enable bit

    if (diff >= 0.0f) {
        nibble |= 0x04; // sign: positive
    }

    const float ad = fabsf(diff);
    if (ad >= 0.015f) {
        nibble |= 0x01; // digit = 2, zero_count = 0 → 0.02
    } else if (ad >= 0.005f) {
        // digit = 1, zero_count = 0 → 0.01 (bits 1,0 are already 00)
    } else if (ad >= 0.0015f) {
        nibble |= 0x02; // zero_count = 1
        nibble |= 0x01; // digit = 2 → 0.002
    } else {
        nibble |= 0x02; // zero_count = 1 → 0.001 (digit=1, bit 0 already 0)
    }

    return nibble;
}

// Decode a 4-bit nibble back to float diff
static inline float llama_outlier_nibble_diff_decode(uint8_t nibble) {
    if (!(nibble & 0x08)) {
        return 0.0f; // disabled
    }

    float base = 0.001f;
    if (!(nibble & 0x02)) {
        base = 0.01f; // zero_count = 0 → 0.0X
    }
    if (nibble & 0x01) {
        base *= 2.0f; // digit = 2
    }
    if (!(nibble & 0x04)) {
        base = -base; // negative sign
    }

    return base;
}

// Nibble-diff encoding for Q2_K: same 4-bit packing, scaled decode values
// Decode table (bits 1-0): 00=±0.002, 01=±0.005, 10=±0.02, 11=±0.05
static inline uint8_t llama_outlier_nibble_diff_encode_q2k(float diff) {
    if (fabsf(diff) < 0.001f) {
        return 0x00; // disabled, diff too small
    }

    uint8_t nibble = 0x08; // enable bit

    if (diff >= 0.0f) {
        nibble |= 0x04; // sign: positive
    }

    const float ad = fabsf(diff);
    if (ad >= 0.035f) {
        nibble |= 0x03; // bits 1-0 = 11 → ±0.05
    } else if (ad >= 0.01f) {
        nibble |= 0x02; // bits 1-0 = 10 → ±0.02
    } else if (ad >= 0.0035f) {
        nibble |= 0x01; // bits 1-0 = 01 → ±0.005
    }
    // else: bits 1-0 = 00 → ±0.002

    return nibble;
}

// Decode a 4-bit nibble back to float diff (Q2_K table)
static inline float llama_outlier_nibble_diff_decode_q2k(uint8_t nibble) {
    if (!(nibble & 0x08)) {
        return 0.0f; // disabled
    }

    float base;
    switch (nibble & 0x03) {
        case 0: base = 0.002f; break;  // 00
        case 1: base = 0.005f; break;  // 01
        case 2: base = 0.02f;  break;  // 10
        case 3: base = 0.05f;  break;  // 11
        default: base = 0.0f; break;
    }

    if (!(nibble & 0x04)) {
        base = -base; // negative sign
    }

    return base;
}

// Pack 32 4-bit nibbles into 16 bytes
// Lower nibble of byte j/2 = weight j, upper nibble = weight j+1
static inline void llama_outlier_nibble_diff_pack_block(const uint8_t nibbles[32], uint8_t packed[16]) {
    for (int j = 0; j < 16; ++j) {
        packed[j] = (nibbles[j * 2] & 0x0F) | ((nibbles[j * 2 + 1] & 0x0F) << 4);
    }
}

// Unpack 16 bytes to 32 4-bit nibbles
// Lower nibble of byte j/2 = weight j, upper nibble = weight j+1
static inline void llama_outlier_nibble_diff_unpack_block(const uint8_t packed[16], uint8_t nibbles[32]) {
    for (int j = 0; j < 16; ++j) {
        nibbles[j * 2]       = packed[j] & 0x0F;
        nibbles[j * 2 + 1]   = packed[j] >> 4;
    }
}

// Q2_K_OUTLIER format constants (shares block size with Q8/Q4)
static constexpr int32_t LLAMA_Q2K_OUTLIER_BLOCK_SIZE = 32;

// Q2_K GGUF metadata key strings (separate namespace from Q8 and Q4)
static constexpr const char * LLAMA_Q2K_OUTLIER_VERSION_KEY = "llama.q2k_outlier.version";
static constexpr const char * LLAMA_Q2K_OUTLIER_BLOCK_SIZE_KEY = "llama.q2k_outlier.block_size";
static constexpr const char * LLAMA_Q2K_OUTLIER_BASE_TYPE_KEY = "llama.q2k_outlier.base_type";
static constexpr const char * LLAMA_Q2K_OUTLIER_VALUE_TYPE_KEY = "llama.q2k_outlier.value_type";
static constexpr const char * LLAMA_Q2K_OUTLIER_INDEX_ENCODING_KEY = "llama.q2k_outlier.index_encoding";
static constexpr const char * LLAMA_Q2K_OUTLIER_STORE_KEY = "llama.q2k_outlier.store";
static constexpr const char * LLAMA_Q2K_OUTLIER_TENSOR_COUNT_KEY = "llama.q2k_outlier.tensor_count";

// Q2_K per-tensor key suffixes
static constexpr const char * LLAMA_Q2K_OUTLIER_TENSOR_NAME_SUFFIX = ".name";
static constexpr const char * LLAMA_Q2K_OUTLIER_TENSOR_INDEX_SUFFIX = ".index";
static constexpr const char * LLAMA_Q2K_OUTLIER_TENSOR_VALUES_SUFFIX = ".values";
static constexpr const char * LLAMA_Q2K_OUTLIER_TENSOR_N_BLOCKS_SUFFIX = ".n_blocks";

// Q2_K sidecar tensor name suffixes (same as Q8/Q4)
static constexpr const char * LLAMA_Q2K_OUTLIER_IDX_SUFFIX = ".outlier_idx";
static constexpr const char * LLAMA_Q2K_OUTLIER_VALUES_SUFFIX = ".outlier_bf16";

#endif
