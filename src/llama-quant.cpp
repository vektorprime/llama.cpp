#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <mutex>
#include <regex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

// result of parsing --tensor-type option
// (changes to this struct must be reflected in tools/quantize/quantize.cpp)
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
};

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

#include "llama-q8-outlier.h"

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix(const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

//
// helper functions for tensor name matching
//

static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}

//
// tensor categorization for quantization
//
// (this is different from LLM_TN - we want broad categories, not specific tensor names per arch)
//

static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(tensor_name.c_str())) {
        return tensor_category::OUTPUT;
    }
    if (tensor_name_match_token_embd(tensor_name.c_str())) {
        return tensor_category::TOKEN_EMBD;
    }
    if (tensor_name.find("attn_qkv.weight") != std::string::npos) {
        return tensor_category::ATTENTION_QKV;
    }
    if (tensor_name.find("attn_kv_b.weight") != std::string::npos) {
        return tensor_category::ATTENTION_KV_B;
    }
    if (tensor_name.find("attn_v.weight") != std::string::npos) {
        return tensor_category::ATTENTION_V;
    }
    if (tensor_name.find("attn_k.weight") != std::string::npos) {
        return tensor_category::ATTENTION_K;
    }
    if (tensor_name.find("attn_q.weight") != std::string::npos) {
        return tensor_category::ATTENTION_Q;
    }
    if (tensor_name.find("attn_output.weight") != std::string::npos) {
        return tensor_category::ATTENTION_OUTPUT;
    }
    if (tensor_name.find("ffn_up") != std::string::npos) {
        return tensor_category::FFN_UP;
    }
    if (tensor_name.find("ffn_gate") != std::string::npos) {
        return tensor_category::FFN_GATE;
    }
    if (tensor_name.find("ffn_down") != std::string::npos) {
        return tensor_category::FFN_DOWN;
    }
    return tensor_category::OTHER;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V     ||
           cat == tensor_category::ATTENTION_QKV   ||
           cat == tensor_category::ATTENTION_KV_B;
}

//
// quantization state
//

struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight

    // tensor type override patterns (compiled once, used twice)
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params):
        model(model), params(params)
    {
        // compile regex patterns once - they are expensive
        if (params->tt_overrides) {
            for (const auto * p = params->tt_overrides; p->pattern != nullptr; p++) {
                tensor_type_patterns.emplace_back(std::regex(p->pattern), p->type);
            }
        }
    }
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    tensor_category category;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
};

//
// dequantization
//

static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

//
// do we allow this tensor to be quantized?
//

static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    // trivial checks first -- no string ops needed
    if (params->only_copy)       return false;

    // quantize only 2D and 3D tensors (experts)
    if (ggml_n_dims(tensor) < 2) return false;

    const std::string name = ggml_get_name(tensor);

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

    // do not quantize norm tensors
    quantize &= name.find("_norm.weight") == std::string::npos;

    quantize &= params->quantize_output_tensor || name != "output.weight";

    // do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // these are very small (e.g. 4x4)
    quantize &= name.find("altup")  == std::string::npos;
    quantize &= name.find("laurel") == std::string::npos;

    // these are not too big so keep them as it is
    quantize &= name.find("per_layer_model_proj") == std::string::npos;

    // do not quantize positional embeddings and token types (BERT)
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight");
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

    // do not quantize Mamba/Kimi's small conv1d weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ssm_conv1d") == std::string::npos;
    quantize &= name.find("shortconv.conv.weight") == std::string::npos;

    // do not quantize RWKV's small yet 2D weights
    quantize &= name.find("time_mix_first.weight") == std::string::npos;
    quantize &= name.find("time_mix_w0.weight") == std::string::npos;
    quantize &= name.find("time_mix_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_v0.weight") == std::string::npos;
    quantize &= name.find("time_mix_v1.weight") == std::string::npos;
    quantize &= name.find("time_mix_v2.weight") == std::string::npos;
    quantize &= name.find("time_mix_a0.weight") == std::string::npos;
    quantize &= name.find("time_mix_a1.weight") == std::string::npos;
    quantize &= name.find("time_mix_a2.weight") == std::string::npos;
    quantize &= name.find("time_mix_g1.weight") == std::string::npos;
    quantize &= name.find("time_mix_g2.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // do not quantize relative position bias (T5)
    quantize &= name.find("attn_rel_b.weight") == std::string::npos;

    // do not quantize specific multimodal tensors
    quantize &= name.find(".position_embd") == std::string::npos;
    quantize &= name.find("sam.pos_embd")   == std::string::npos;
    quantize &= name.find("sam.neck.")      == std::string::npos;
    quantize &= name.find("sam.net_")       == std::string::npos;
    quantize &= name.find(".rel_pos")       == std::string::npos;
    quantize &= name.find(".patch_embd")    == std::string::npos;
    quantize &= name.find(".patch_merger")  == std::string::npos;

    return quantize;
}

//
// tensor type selection
//

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    ggml_type return_type = target_type;

    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);

    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
                        t->name, ncols, qk_k, ggml_type_name(target_type));
        ++qs.n_fallback;

        switch (target_type) {
            // types on the left: block size 256
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:   // types on the right: block size 32
            case GGML_TYPE_IQ4_XS:  return_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
            default:
                throw std::runtime_error(format("no tensor type fallback is defined for type %s",
                                                ggml_type_name(target_type)));
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            //
            // the fallback return type is still not compatible for this tensor!
            //
            // most likely, this tensor's first dimension is not divisible by 32.
            // this is very rare. we can either abort the quantization, or
            // fallback to F16 / F32.
            //
            LLAMA_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
            return_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(return_type));
    }
    return return_type;
}

// internal standard logic for selecting the target tensor type based on tensor category, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype, tensor_category category) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (category == tensor_category::OUTPUT || (qs.has_tied_embeddings && category == tensor_category::TOKEN_EMBD)) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0 && ftype != LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (category == tensor_category::TOKEN_EMBD) {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (category_is_attn_v(category)) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && category == tensor_category::ATTENTION_K) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (category == tensor_category::FFN_DOWN) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (category == tensor_category::ATTENTION_OUTPUT) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (category == tensor_category::ATTENTION_K) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::ATTENTION_Q) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::FFN_DOWN) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (category == tensor_category::ATTENTION_OUTPUT) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (category == tensor_category::ATTENTION_QKV) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (category == tensor_category::FFN_GATE) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (category == tensor_category::FFN_UP) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    return new_type;
}

// outer wrapper: determine the ggml_type that this tensor should be quantized to
static ggml_type llama_tensor_get_type(quantize_state_impl & qs, const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type default_type, const tensor_metadata & tm) {
    if (!tensor_allows_quantization(params, qs.model.arch, tensor)) {
        return tensor->type;
    }
    if (params->token_embedding_type < GGML_TYPE_COUNT && tm.category == tensor_category::TOKEN_EMBD) {
        return params->token_embedding_type;
    }
    if (params->output_tensor_type < GGML_TYPE_COUNT && tm.category == tensor_category::OUTPUT) {
        return params->output_tensor_type;
    }

    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (!params->pure && ggml_is_quantized(default_type)) {
        // if the user provided tensor types - use those
        bool manual = false;
        if (!qs.tensor_type_patterns.empty()) {
            const std::string tensor_name(tensor->name);
            for (const auto & [pattern, qtype] : qs.tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        LLAMA_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\n",
                                       __func__, tensor_name.c_str(), ggml_type_name(new_type), ggml_type_name(qtype));
                        new_type = qtype;
                    }
                    manual = true;
                    break;
                }
            }
        }

        // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
        if (!manual) {
            new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
        }

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);
    }

    return new_type;
}

//
// quantization implementation
//

static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

//
// Q8_0 + sparse BF16 outlier sidecars
//

struct q8_outlier_candidate {
    int64_t row;
    int64_t block_col;
    float score;
};

struct q8_outlier_tensor_data {
    std::string name;
    std::string idx_name;
    std::string values_name;

    int64_t n_blocks_total = 0;
    int64_t n_blocks_per_row = 0;

    std::vector<int32_t> idx;
    std::vector<ggml_bf16_t> values;
    std::unordered_set<int64_t> protected_blocks;

    float estimated_bpw = 0.0f;

    int64_t n_blocks() const {
        return (int64_t) idx.size() / 2;
    }
};

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static bool q8_outlier_name_matches(const char * pattern, const std::string & name) {
    if (pattern == nullptr || pattern[0] == '\0') {
        return false;
    }

    return std::regex_search(name, std::regex(pattern));
}

static bool q8_outlier_tensor_enabled(const llama_model_quantize_params * params, const std::string & name) {
    if (!params->q8_outlier_enable) {
        return false;
    }
    if (params->q8_outlier_include_weights && !q8_outlier_name_matches(params->q8_outlier_include_weights, name)) {
        return false;
    }
    if (params->q8_outlier_exclude_weights && q8_outlier_name_matches(params->q8_outlier_exclude_weights, name)) {
        return false;
    }
    return true;
}

static bool q8_outlier_block_is_candidate(
        const float * block,
        const float * imatrix,
        float ratio_threshold,
        float rel_rmse_threshold,
        float & score) {
    constexpr float eps = 1.0e-12f;

    int j_max = 0;
    float max_abs = 0.0f;
    float second_abs = 0.0f;

    for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
        const float a = std::fabs(block[j]);
        if (a > max_abs) {
            second_abs = max_abs;
            max_abs = a;
            j_max = j;
        } else if (a > second_abs) {
            second_abs = a;
        }
    }

    if (max_abs <= eps) {
        return false;
    }

    const float ratio = max_abs / std::max(second_abs, eps);
    if (ratio < ratio_threshold) {
        return false;
    }

    const float d = max_abs / 127.0f;
    double sqerr = 0.0;
    double energy = 0.0;
    double wsum = 0.0;

    for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
        if (j == j_max) {
            continue;
        }

        const float qf = std::round(block[j] / d);
        const float q = std::max(-127.0f, std::min(127.0f, qf));
        const float x_hat = q * d;
        const float err = block[j] - x_hat;
        const float w = imatrix ? std::max(imatrix[j], 0.0f) : 1.0f;

        sqerr += double(w) * double(err) * double(err);
        energy += double(w) * double(block[j]) * double(block[j]);
        wsum += double(w);
    }

    if (wsum <= 0.0) {
        return false;
    }

    const float rel_rmse = std::sqrt(float(sqerr / std::max(energy, double(eps))));
    if (rel_rmse < rel_rmse_threshold) {
        return false;
    }

    score = ratio * rel_rmse;

    // DEBUG: log first few candidates
    {
        static std::mutex log_mutex;
        static int log_count = 0;
        std::lock_guard<std::mutex> lock(log_mutex);
        if (log_count < 5) {
            fprintf(stderr, "[q8_outlier] CANDIDATE #%d: max_abs=%f second_abs=%f ratio=%f rel_rmse=%f score=%f\n",
                    log_count, max_abs, second_abs, ratio, rel_rmse, score);
            fprintf(stderr, "[q8_outlier]   block[0..3]: %f %f %f %f\n",
                    block[0], block[1], block[2], block[3]);
            fflush(stderr);
            log_count++;
        }
    }

    return true;
}

static q8_outlier_tensor_data q8_outlier_analyze_tensor(
        const std::string & name,
        const float * f32_data,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix,
        const llama_model_quantize_params * params) {
    q8_outlier_tensor_data result;
    result.name = name;
    result.idx_name = name + ".outlier_idx";
    result.values_name = name + ".outlier_bf16";

    if (n_per_row % LLAMA_Q8_OUTLIER_BLOCK_SIZE != 0) {
        LLAMA_LOG_INFO("%s: %s — n_per_row=%lld not divisible by %d, skipping\n",
                __func__, name.c_str(), (long long)n_per_row, LLAMA_Q8_OUTLIER_BLOCK_SIZE);
        return result;
    }

    result.n_blocks_per_row = n_per_row / LLAMA_Q8_OUTLIER_BLOCK_SIZE;
    result.n_blocks_total = nrows * result.n_blocks_per_row;

    std::vector<q8_outlier_candidate> candidates;
    candidates.reserve((size_t) std::min<int64_t>(result.n_blocks_total, 1024));

    int64_t n_ratio_pass = 0;
    int64_t n_rmse_pass = 0;
    float max_ratio_seen = 0.0f;
    float max_rmse_seen = 0.0f;

    for (int64_t row = 0; row < nrows; ++row) {
        const float * row_data = f32_data + row * n_per_row;
        for (int64_t block_col = 0; block_col < result.n_blocks_per_row; ++block_col) {
            const int64_t col = block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
            const float * block = row_data + col;
            const float * block_imatrix = imatrix ? imatrix + col : nullptr;

            float score = 0.0f;
            if (q8_outlier_block_is_candidate(
                    block,
                    block_imatrix,
                    params->q8_outlier_ratio,
                    params->q8_outlier_nonmax_rel_rmse,
                    score)) {
                candidates.push_back({ row, block_col, score });
            }
        }
    }

    const int64_t max_blocks = (int64_t) std::floor(double(result.n_blocks_total) * double(params->q8_outlier_max_frac));
    if ((int64_t) candidates.size() > max_blocks) {
        if (max_blocks <= 0) {
            candidates.clear();
        } else {
            std::partial_sort(candidates.begin(), candidates.begin() + max_blocks, candidates.end(),
                    [](const q8_outlier_candidate & a, const q8_outlier_candidate & b) {
                        return a.score > b.score;
                    });
            candidates.resize(max_blocks);
        }
    }

    LLAMA_LOG_INFO("%s: %s — total_blocks=%lld candidates=%lld max=%lld protected=%lld (%.3f%%)\n",
            __func__, name.c_str(),
            (long long)result.n_blocks_total, (long long)candidates.size(),
            (long long)max_blocks, (long long)candidates.size(),
            result.n_blocks_total > 0 ? 100.0 * candidates.size() / result.n_blocks_total : 0.0);

    std::sort(candidates.begin(), candidates.end(),
            [](const q8_outlier_candidate & a, const q8_outlier_candidate & b) {
                return a.row != b.row ? a.row < b.row : a.block_col < b.block_col;
            });

    result.idx.reserve(candidates.size() * 2);
    result.values.reserve(candidates.size() * LLAMA_Q8_OUTLIER_BLOCK_SIZE);
    result.protected_blocks.reserve(candidates.size() * 2);

    for (const q8_outlier_candidate & candidate : candidates) {
        const int64_t key = candidate.row * result.n_blocks_per_row + candidate.block_col;
        result.protected_blocks.insert(key);
        result.idx.push_back((int32_t) candidate.row);
        result.idx.push_back((int32_t) candidate.block_col);

        const float * block = f32_data + candidate.row * n_per_row + candidate.block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
            result.values.push_back(ggml_fp32_to_bf16(block[j]));
        }
    }

    result.estimated_bpw = 8.5f + (result.n_blocks_total > 0 ? float(result.n_blocks()) / float(result.n_blocks_total) * 18.0f : 0.0f);
    return result;
}


// Compute deltas = original_F32 - Q8_dequantized for each outlier block,
// replacing the stored full BF16 values with BF16 deltas.
// This allows the Q8 base tensor to remain intact (no zeroing), so operations
// that don't support correction (e.g., ggml_get_rows) still get Q8 approximation.
static void q8_outlier_compute_deltas(
        const float * f32_data,
        const void * q8_data,
        int64_t nrows,
        int64_t n_per_row,
        q8_outlier_tensor_data & outliers) {

    const size_t block_size = ggml_type_size(GGML_TYPE_Q8_0);
    const uint8_t * data = (const uint8_t *) q8_data;

    // Iterate using idx vector which has (row, block_col) pairs in order
    for (size_t k = 0; k < outliers.idx.size() / 2; k++) {
        const int64_t row = outliers.idx[k * 2];
        const int64_t block_col = outliers.idx[k * 2 + 1];
        const int64_t col = block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;

        // Original F32 values
        const float * orig = f32_data + row * n_per_row + col;

        // Dequantize Q8 block: layout is ggml_fp16_t d (2 bytes) + int8_t qs[32]
        const uint8_t * block_ptr = data + row * (n_per_row / 32) * block_size + block_col * block_size;
        const ggml_fp16_t * d_ptr = (const ggml_fp16_t *) block_ptr;
        float q8_d = ggml_fp16_to_fp32(*d_ptr);
        const int8_t * q8_data_block = (const int8_t *)(block_ptr + sizeof(ggml_fp16_t));

        // Compute delta and store as BF16
        for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; j++) {
            float q8_val = q8_data_block[j] * q8_d;
            float delta = orig[j] - q8_val;
            outliers.values[k * LLAMA_Q8_OUTLIER_BLOCK_SIZE + j] = ggml_fp32_to_bf16(delta);
        }
    }

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-quant] %s: computed %zu deltas (original - Q8_dequant)\n",
                outliers.name.c_str(), outliers.idx.size() / 2);

        // Detailed logging for first 3 outlier blocks
        const size_t n_blocks = outliers.idx.size() / 2;
        const size_t log_blocks = n_blocks < 3 ? n_blocks : 3;
        for (size_t k = 0; k < log_blocks; k++) {
            double l2 = 0.0;
            double mean_abs = 0.0;
            double max_abs = 0.0;
            for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; j++) {
                float d = ggml_bf16_to_fp32(outliers.values[k * LLAMA_Q8_OUTLIER_BLOCK_SIZE + j]);
                l2 += (double)d * d;
                double ad = fabs((double)d);
                mean_abs += ad;
                if (ad > max_abs) max_abs = ad;
            }
            mean_abs /= LLAMA_Q8_OUTLIER_BLOCK_SIZE;
            fprintf(stderr, "[delta-quant]   block[%zu] row=%d bcol=%d L2=%.6e mean_abs=%.6e max_abs=%.6e\n",
                    k, (int)outliers.idx[k * 2], (int)outliers.idx[k * 2 + 1],
                    sqrt(l2), mean_abs, max_abs);
        }

        // Print all 32 delta values for the first block
        if (n_blocks > 0) {
            fprintf(stderr, "[delta-quant]   block[0] all 32 deltas:");
            for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; j++) {
                float d = ggml_bf16_to_fp32(outliers.values[j]);
                fprintf(stderr, " %.6e", d);
            }
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }
}

static void q8_outlier_reconstruct_tensor(
        const ggml_tensor * tensor,
        const std::vector<int32_t> & idx,
        const std::vector<ggml_bf16_t> & values,
        float * f32_buf,
        int64_t rows,
        int64_t cols) {
    const int64_t n_blocks = (int64_t) idx.size() / 2;
    for (int64_t k = 0; k < n_blocks; ++k) {
        const int64_t row = idx[k * 2];
        const int64_t block_col = idx[k * 2 + 1];
        const ggml_bf16_t * block_vals = values.data() + k * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        float * dst = f32_buf + row * cols + block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
            dst[j] += ggml_bf16_to_fp32(block_vals[j]);
        }
    }
}

static void q8_outlier_add_tensor_meta(
        struct gguf_context * ctx,
        const std::string & name,
        ggml_type type,
        int64_t ne0,
        int64_t ne1) {
    ggml_tensor tensor;
    std::memset(&tensor, 0, sizeof(tensor));
    tensor.type = type;
    tensor.ne[0] = ne0;
    tensor.ne[1] = std::max<int64_t>(ne1, 1);
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = ggml_type_size(type);
    tensor.nb[1] = tensor.nb[0] * (tensor.ne[0] / ggml_blck_size(type));
    tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
    tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
    ggml_set_name(&tensor, name.c_str());

    gguf_add_tensor(ctx, &tensor);
}

static void q8_outlier_write_metadata(
        struct gguf_context * ctx,
        const std::vector<q8_outlier_tensor_data> & tensors) {
    gguf_set_val_u32(ctx, LLAMA_Q8_OUTLIER_VERSION_KEY, 1);
    gguf_set_val_u32(ctx, LLAMA_Q8_OUTLIER_BLOCK_SIZE_KEY, LLAMA_Q8_OUTLIER_BLOCK_SIZE);
    gguf_set_val_str(ctx, LLAMA_Q8_OUTLIER_BASE_TYPE_KEY, "q8_0");
    gguf_set_val_str(ctx, LLAMA_Q8_OUTLIER_VALUE_TYPE_KEY, "bf16");
    gguf_set_val_str(ctx, LLAMA_Q8_OUTLIER_INDEX_ENCODING_KEY, "row_block_col");
    gguf_set_val_str(ctx, LLAMA_Q8_OUTLIER_STORE_KEY, "delta");
    gguf_set_val_u32(ctx, LLAMA_Q8_OUTLIER_TENSOR_COUNT_KEY, (uint32_t) tensors.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & t = tensors[i];
        const std::string prefix = "llama.q8_outlier.tensor." + std::to_string(i);
        gguf_set_val_str(ctx, (prefix + ".name").c_str(), t.name.c_str());
        gguf_set_val_str(ctx, (prefix + ".index").c_str(), t.idx_name.c_str());
        gguf_set_val_str(ctx, (prefix + ".values").c_str(), t.values_name.c_str());
        gguf_set_val_u32(ctx, (prefix + ".n_blocks").c_str(), (uint32_t) t.n_blocks());
    }
}

static void q8_outlier_write_report(
        const std::vector<q8_outlier_tensor_data> & tensors,
        const llama_model_quantize_params * params) {
    if (params->q8_outlier_report_path == nullptr || params->q8_outlier_report_path[0] == '\0') {
        return;
    }

    std::ofstream fout(params->q8_outlier_report_path);
    fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    fout << "{\n";
    fout << "  \"format\": \"Q8_0_BF16_OUTLIER\",\n";
    fout << "  \"block_size\": " << LLAMA_Q8_OUTLIER_BLOCK_SIZE << ",\n";
    fout << "  \"cuda_focused\": true,\n";
    fout << "  \"tensors\": [\n";
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & t = tensors[i];
        const double frac = t.n_blocks_total > 0 ? double(t.n_blocks()) / double(t.n_blocks_total) : 0.0;
        fout << "    {\n";
        fout << "      \"name\": \"" << json_escape(t.name) << "\",\n";
        fout << "      \"blocks_total\": " << t.n_blocks_total << ",\n";
        fout << "      \"blocks_protected\": " << t.n_blocks() << ",\n";
        fout << "      \"protected_fraction\": " << std::setprecision(8) << frac << ",\n";
        fout << "      \"estimated_bpw\": " << std::setprecision(6) << t.estimated_bpw << ",\n";
        fout << "      \"thresholds\": {\n";
        fout << "        \"ratio\": " << params->q8_outlier_ratio << ",\n";
        fout << "        \"nonmax_rel_rmse\": " << params->q8_outlier_nonmax_rel_rmse << ",\n";
        fout << "        \"max_frac\": " << params->q8_outlier_max_frac << "\n";
        fout << "      }\n";
        fout << "    }" << (i + 1 == tensors.size() ? "\n" : ",\n");
    }
    fout << "  ]\n";
    fout << "}\n";
}

static size_t q8_outlier_sidecar_size(const q8_outlier_tensor_data & t) {
    return t.idx.size() * sizeof(int32_t) + t.values.size() * sizeof(ggml_bf16_t);
}

// reconstruct a full F32 tensor from a Q8_0 base (already dequantized to f32_buf)
// by adding BF16 delta corrections for outlier blocks
static void q8_outlier_reconstruct_tensor(
    const ggml_tensor * base_tensor,
    const ggml_tensor * idx_tensor,
    const ggml_tensor * values_tensor,
    float * f32_buf,
    int64_t n_rows,
    int64_t n_cols) {
    const int32_t * idx = (const int32_t *) idx_tensor->data;
    const ggml_bf16_t * values = (const ggml_bf16_t *) values_tensor->data;
    const int64_t n_blocks = ggml_nelements(idx_tensor) / 2;

    for (int64_t b = 0; b < n_blocks; ++b) {
        const int64_t row = idx[b * 2];
        const int64_t block_col = idx[b * 2 + 1];
        const int64_t off = row * n_cols + block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        const ggml_bf16_t * v = values + b * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
            f32_buf[off + j] += ggml_bf16_to_fp32(v[j]);
        }
    }
}

// Q4_0_BF16_OUTLIER functions — mirror q8_outlier_* with Q4-specific dequantization

static bool q4_outlier_name_matches(const char * pattern, const std::string & name) {
    if (pattern == nullptr || pattern[0] == '\0') {
        return false;
    }
    return std::regex_search(name, std::regex(pattern));
}

static bool q4_outlier_tensor_enabled(const llama_model_quantize_params * params, const std::string & name) {
    if (!params->q4_outlier_enable) {
        return false;
    }
    if (params->q4_outlier_include_weights && !q4_outlier_name_matches(params->q4_outlier_include_weights, name)) {
        return false;
    }
    if (params->q4_outlier_exclude_weights && q4_outlier_name_matches(params->q4_outlier_exclude_weights, name)) {
        return false;
    }
    return true;
}

static bool q4_outlier_block_is_candidate(
        const float * block,
        const float * imatrix,
        float ratio_threshold,
        float rel_rmse_threshold,
        float & score) {
    constexpr float eps = 1.0e-12f;

    int j_max = 0;
    float max_abs = 0.0f;
    float second_abs = 0.0f;
    float max_val = 0.0f; // signed value with largest absolute magnitude

    for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
        const float a = std::fabs(block[j]);
        if (a > max_abs) {
            second_abs = max_abs;
            max_abs = a;
            max_val = block[j];
            j_max = j;
        } else if (a > second_abs) {
            second_abs = a;
        }
    }

    if (max_abs <= eps) {
        return false;
    }

    const float ratio = max_abs / std::max(second_abs, eps);
    if (ratio < ratio_threshold) {
        return false;
    }

    // Q4_0: d = max / -8 (matches reference quantize_row_q4_0_ref)
    const float d = max_val / -8.0f;
    double sqerr = 0.0;
    double energy = 0.0;
    double wsum = 0.0;

    for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
        if (j == j_max) {
            continue;
        }

        const float qf = std::round(block[j] / d);
        // Q4_0: clamp to -8..7
        const float q = std::max(-8.0f, std::min(7.0f, qf));
        const float x_hat = q * d;
        const float err = block[j] - x_hat;
        const float w = imatrix ? std::max(imatrix[j], 0.0f) : 1.0f;

        sqerr += double(w) * double(err) * double(err);
        energy += double(w) * double(block[j]) * double(block[j]);
        wsum += double(w);
    }

    if (wsum <= 0.0) {
        return false;
    }

    const float rel_rmse = std::sqrt(float(sqerr / std::max(energy, double(eps))));
    if (rel_rmse < rel_rmse_threshold) {
        return false;
    }

    score = ratio * rel_rmse;
    return true;
}

static q8_outlier_tensor_data q4_outlier_analyze_tensor(
        const std::string & name,
        const float * f32_data,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix,
        const llama_model_quantize_params * params) {
    q8_outlier_tensor_data result;
    result.name = name;
    result.idx_name = name + ".outlier_idx";
    result.values_name = name + ".outlier_bf16";

    if (n_per_row % LLAMA_Q8_OUTLIER_BLOCK_SIZE != 0) {
        LLAMA_LOG_INFO("%s: %s — n_per_row=%lld not divisible by %d, skipping\n",
                __func__, name.c_str(), (long long)n_per_row, LLAMA_Q8_OUTLIER_BLOCK_SIZE);
        return result;
    }

    result.n_blocks_per_row = n_per_row / LLAMA_Q8_OUTLIER_BLOCK_SIZE;
    result.n_blocks_total = nrows * result.n_blocks_per_row;

    std::vector<q8_outlier_candidate> candidates;
    candidates.reserve((size_t) std::min<int64_t>(result.n_blocks_total, 1024));

    for (int64_t row = 0; row < nrows; ++row) {
        const float * row_data = f32_data + row * n_per_row;
        for (int64_t block_col = 0; block_col < result.n_blocks_per_row; ++block_col) {
            const int64_t col = block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
            const float * block = row_data + col;
            const float * block_imatrix = imatrix ? imatrix + col : nullptr;

            float score = 0.0f;
            if (q4_outlier_block_is_candidate(
                    block,
                    block_imatrix,
                    params->q4_outlier_ratio,
                    params->q4_outlier_nonmax_rel_rmse,
                    score)) {
                candidates.push_back({ row, block_col, score });
            }
        }
    }

    const int64_t max_blocks = (int64_t) std::floor(double(result.n_blocks_total) * double(params->q4_outlier_max_frac));
    if ((int64_t) candidates.size() > max_blocks) {
        if (max_blocks <= 0) {
            candidates.clear();
        } else {
            std::partial_sort(candidates.begin(), candidates.begin() + max_blocks, candidates.end(),
                    [](const q8_outlier_candidate & a, const q8_outlier_candidate & b) {
                        return a.score > b.score;
                    });
            candidates.resize(max_blocks);
        }
    }

    LLAMA_LOG_INFO("%s: %s — total_blocks=%lld candidates=%lld max=%lld protected=%lld (%.3f%%)\n",
            __func__, name.c_str(),
            (long long)result.n_blocks_total, (long long)candidates.size(),
            (long long)max_blocks, (long long)candidates.size(),
            result.n_blocks_total > 0 ? 100.0 * candidates.size() / result.n_blocks_total : 0.0);

    std::sort(candidates.begin(), candidates.end(),
            [](const q8_outlier_candidate & a, const q8_outlier_candidate & b) {
                return a.row != b.row ? a.row < b.row : a.block_col < b.block_col;
            });

    result.idx.reserve(candidates.size() * 2);
    result.values.reserve(candidates.size() * LLAMA_Q8_OUTLIER_BLOCK_SIZE);
    result.protected_blocks.reserve(candidates.size() * 2);

    for (const q8_outlier_candidate & candidate : candidates) {
        const int64_t key = candidate.row * result.n_blocks_per_row + candidate.block_col;
        result.protected_blocks.insert(key);
        result.idx.push_back((int32_t) candidate.row);
        result.idx.push_back((int32_t) candidate.block_col);

        const float * block = f32_data + candidate.row * n_per_row + candidate.block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
            result.values.push_back(ggml_fp32_to_bf16(block[j]));
        }
    }

    // Q4_0 base: 4 bits data + 2 bits scale = 4.5 bpw
    result.estimated_bpw = 4.5f + (result.n_blocks_total > 0 ? float(result.n_blocks()) / float(result.n_blocks_total) * 18.0f : 0.0f);
    return result;
}

// Compute deltas = original_F32 - Q4_dequantized for each outlier block
static void q4_outlier_compute_deltas(
        const float * f32_data,
        const void * q4_data,
        int64_t nrows,
        int64_t n_per_row,
        q8_outlier_tensor_data & outliers) {

    const size_t block_size = ggml_type_size(GGML_TYPE_Q4_0);  // 18 bytes
    const uint8_t * data = (const uint8_t *) q4_data;

    for (size_t k = 0; k < outliers.idx.size() / 2; k++) {
        const int64_t row = outliers.idx[k * 2];
        const int64_t block_col = outliers.idx[k * 2 + 1];
        const int64_t col = block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;

        const float * orig = f32_data + row * n_per_row + col;

        // Q4_0 block: ggml_fp16_t d (2 bytes) + uint8_t qs[16] (16 bytes) = 18 bytes
        const uint8_t * block_ptr = data + row * (n_per_row / 32) * block_size + block_col * block_size;
        const ggml_fp16_t * d_ptr = (const ggml_fp16_t *) block_ptr;
        float q4_d = ggml_fp16_to_fp32(*d_ptr);
        const uint8_t * q4_data_block = (const uint8_t *)(block_ptr + sizeof(ggml_fp16_t));

        // Dequantize Q4: each byte holds two 4-bit nibbles, biased by -8
        for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; j++) {
            uint8_t byte = q4_data_block[j / 2];
            int8_t q;
            if (j % 2 == 0) {
                q = (int8_t)(byte & 0x0F) - 8;  // lower nibble
            } else {
                q = (int8_t)(byte >> 4) - 8;    // upper nibble
            }
            float q4_val = q * q4_d;
            float delta = orig[j] - q4_val;
            outliers.values[k * LLAMA_Q8_OUTLIER_BLOCK_SIZE + j] = ggml_fp32_to_bf16(delta);
        }
    }

    if (ggml_custom_logs_enabled()) {
        fprintf(stderr, "[delta-quant] %s: computed %zu deltas (original - Q4_dequant)\n",
                outliers.name.c_str(), outliers.idx.size() / 2);
        const size_t n_blocks = outliers.idx.size() / 2;
        const size_t log_blocks = n_blocks < 3 ? n_blocks : 3;
        for (size_t k = 0; k < log_blocks; k++) {
            double l2 = 0.0;
            double mean_abs = 0.0;
            double max_abs = 0.0;
            for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; j++) {
                float d = ggml_bf16_to_fp32(outliers.values[k * LLAMA_Q8_OUTLIER_BLOCK_SIZE + j]);
                l2 += (double)d * d;
                double ad = fabs((double)d);
                mean_abs += ad;
                if (ad > max_abs) max_abs = ad;
            }
            mean_abs /= LLAMA_Q8_OUTLIER_BLOCK_SIZE;
            fprintf(stderr, "[delta-quant]   block[%zu] row=%d bcol=%d L2=%.6e mean_abs=%.6e max_abs=%.6e\n",
                    k, (int)outliers.idx[k * 2], (int)outliers.idx[k * 2 + 1],
                    sqrt(l2), mean_abs, max_abs);
        }
        fflush(stderr);
    }
}

// reconstruct from vector data (same as q8 version — reconstruction is format-agnostic)
static void q4_outlier_reconstruct_tensor(
        const ggml_tensor * tensor,
        const std::vector<int32_t> & idx,
        const std::vector<ggml_bf16_t> & values,
        float * f32_buf,
        int64_t rows,
        int64_t cols) {
    const int64_t n_blocks = (int64_t) idx.size() / 2;
    for (int64_t k = 0; k < n_blocks; ++k) {
        const int64_t row = idx[k * 2];
        const int64_t block_col = idx[k * 2 + 1];
        const ggml_bf16_t * block_vals = values.data() + k * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        float * dst = f32_buf + row * cols + block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
            dst[j] += ggml_bf16_to_fp32(block_vals[j]);
        }
    }
}

// reconstruct from pointer tensors (same as q8 version — format-agnostic)
static void q4_outlier_reconstruct_tensor(
    const ggml_tensor * base_tensor,
    const ggml_tensor * idx_tensor,
    const ggml_tensor * values_tensor,
    float * f32_buf,
    int64_t n_rows,
    int64_t n_cols) {
    const int32_t * idx = (const int32_t *) idx_tensor->data;
    const ggml_bf16_t * values = (const ggml_bf16_t *) values_tensor->data;
    const int64_t n_blocks = ggml_nelements(idx_tensor) / 2;

    for (int64_t b = 0; b < n_blocks; ++b) {
        const int64_t row = idx[b * 2];
        const int64_t block_col = idx[b * 2 + 1];
        const int64_t off = row * n_cols + block_col * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        const ggml_bf16_t * v = values + b * LLAMA_Q8_OUTLIER_BLOCK_SIZE;
        for (int j = 0; j < LLAMA_Q8_OUTLIER_BLOCK_SIZE; ++j) {
            f32_buf[off + j] += ggml_bf16_to_fp32(v[j]);
        }
    }
}

static void q4_outlier_add_tensor_meta(
        struct gguf_context * ctx,
        const std::string & name,
        ggml_type type,
        int64_t ne0,
        int64_t ne1) {
    ggml_tensor tensor;
    std::memset(&tensor, 0, sizeof(tensor));
    tensor.type = type;
    tensor.ne[0] = ne0;
    tensor.ne[1] = std::max<int64_t>(ne1, 1);
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = ggml_type_size(type);
    tensor.nb[1] = tensor.nb[0] * (tensor.ne[0] / ggml_blck_size(type));
    tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
    tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
    ggml_set_name(&tensor, name.c_str());

    gguf_add_tensor(ctx, &tensor);
}

static void q4_outlier_write_metadata(
        struct gguf_context * ctx,
        const std::vector<q8_outlier_tensor_data> & tensors) {
    gguf_set_val_u32(ctx, LLAMA_Q4_OUTLIER_VERSION_KEY, 1);
    gguf_set_val_u32(ctx, LLAMA_Q4_OUTLIER_BLOCK_SIZE_KEY, LLAMA_Q4_OUTLIER_BLOCK_SIZE);
    gguf_set_val_str(ctx, LLAMA_Q4_OUTLIER_BASE_TYPE_KEY, "q4_0");
    gguf_set_val_str(ctx, LLAMA_Q4_OUTLIER_VALUE_TYPE_KEY, "bf16");
    gguf_set_val_str(ctx, LLAMA_Q4_OUTLIER_INDEX_ENCODING_KEY, "row_block_col");
    gguf_set_val_str(ctx, LLAMA_Q4_OUTLIER_STORE_KEY, "delta");
    gguf_set_val_u32(ctx, LLAMA_Q4_OUTLIER_TENSOR_COUNT_KEY, (uint32_t) tensors.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & t = tensors[i];
        const std::string prefix = "llama.q4_outlier.tensor." + std::to_string(i);
        gguf_set_val_str(ctx, (prefix + ".name").c_str(), t.name.c_str());
        gguf_set_val_str(ctx, (prefix + ".index").c_str(), t.idx_name.c_str());
        gguf_set_val_str(ctx, (prefix + ".values").c_str(), t.values_name.c_str());
        gguf_set_val_u32(ctx, (prefix + ".n_blocks").c_str(), (uint32_t) t.n_blocks());
    }
}

static void q4_outlier_write_report(
        const std::vector<q8_outlier_tensor_data> & tensors,
        const llama_model_quantize_params * params) {
    if (params->q4_outlier_report_path == nullptr || params->q4_outlier_report_path[0] == '\0') {
        return;
    }

    std::ofstream fout(params->q4_outlier_report_path);
    fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    fout << "{\n";
    fout << "  \"format\": \"Q4_0_BF16_OUTLIER\",\n";
    fout << "  \"block_size\": " << LLAMA_Q4_OUTLIER_BLOCK_SIZE << ",\n";
    fout << "  \"cuda_focused\": true,\n";
    fout << "  \"tensors\": [\n";
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & t = tensors[i];
        const double frac = t.n_blocks_total > 0 ? double(t.n_blocks()) / double(t.n_blocks_total) : 0.0;
        fout << "    {\n";
        fout << "      \"name\": \"" << json_escape(t.name) << "\",\n";
        fout << "      \"blocks_total\": " << t.n_blocks_total << ",\n";
        fout << "      \"blocks_protected\": " << t.n_blocks() << ",\n";
        fout << "      \"protected_fraction\": " << std::setprecision(8) << frac << ",\n";
        fout << "      \"estimated_bpw\": " << std::setprecision(6) << t.estimated_bpw << ",\n";
        fout << "      \"thresholds\": {\n";
        fout << "        \"ratio\": " << params->q4_outlier_ratio << ",\n";
        fout << "        \"nonmax_rel_rmse\": " << params->q4_outlier_nonmax_rel_rmse << ",\n";
        fout << "        \"max_frac\": " << params->q4_outlier_max_frac << "\n";
        fout << "      }\n";
        fout << "    }" << (i + 1 == tensors.size() ? "\n" : ",\n");
    }
    fout << "  ]\n";
    fout << "}\n";
}

static size_t q4_outlier_sidecar_size(const q8_outlier_tensor_data & t) {
    return t.idx.size() * sizeof(int32_t) + t.values.size() * sizeof(ggml_bf16_t);
}

//
// imatrix requirement check
//

static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}

//
// given a file type, get the default tensor type
//

ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_Q8_0_BF16_OUTLIER: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;
        case LLAMA_FTYPE_MOSTLY_Q1_0: return GGML_TYPE_Q1_0;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return GGML_TYPE_MXFP4;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  return GGML_TYPE_Q4_K;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  return GGML_TYPE_Q5_K;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   return GGML_TYPE_TQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return GGML_TYPE_IQ2_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;

        default: return GGML_TYPE_COUNT;
    }
}


static void init_quantize_state_counters(quantize_state_impl & qs, std::vector<tensor_metadata> & metadata) {
    for (auto & tm : metadata) {
        tensor_category cat = tensor_get_category(tm.name);
        tm.category = cat;

        if (category_is_attn_v(cat)) {
            ++qs.n_attention_wv;
        }

        if (cat == tensor_category::OUTPUT) {
            qs.has_tied_embeddings = false;
        }
    }
    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer;
}

//
// main quantization driver
//

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    llama_ftype ftype = params->ftype;
    llama_model_quantize_params params_local;
    if (ftype == LLAMA_FTYPE_MOSTLY_Q8_0_BF16_OUTLIER && !params->q8_outlier_enable) {
        params_local = *params;
        params_local.q8_outlier_enable = true;
        params = &params_local;
    }
    if (ftype == LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER && !params->q4_outlier_enable) {
        if (&params_local != params) {
            params_local = *params;
        }
        params_local.q4_outlier_enable = true;
        params_local.q4_outlier_ratio = 16.0f;
        params_local.q4_outlier_nonmax_rel_rmse = 0.01f;
        params_local.q4_outlier_max_frac = 0.02f;
        params = &params_local;
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) {
        throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }
    const bool q8_outlier_enabled = params->q8_outlier_enable || ftype == LLAMA_FTYPE_MOSTLY_Q8_0_BF16_OUTLIER;
    const bool q4_outlier_enabled = params->q4_outlier_enable || ftype == LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER;
    if (q8_outlier_enabled) {
        if (params->q8_outlier_store != LLAMA_Q8_OUTLIER_STORE_FULL) {
            throw std::runtime_error("Q8_0_BF16_OUTLIER currently supports only full outlier block storage");
        }
        if (params->keep_split) {
            throw std::runtime_error("Q8_0_BF16_OUTLIER does not support --keep-split in this CUDA-focused prototype");
        }
        if (params->q8_outlier_ratio <= 0.0f || params->q8_outlier_nonmax_rel_rmse < 0.0f ||
                params->q8_outlier_max_frac < 0.0f || params->q8_outlier_max_frac > 1.0f) {
            throw std::runtime_error("invalid Q8_0_BF16_OUTLIER thresholds");
        }
    }
    if (q4_outlier_enabled) {
        if (params->keep_split) {
            throw std::runtime_error("Q4_0_BF16_OUTLIER does not support --keep-split in this CUDA-focused prototype");
        }
        if (params->q4_outlier_ratio <= 0.0f || params->q4_outlier_nonmax_rel_rmse < 0.0f ||
                params->q4_outlier_max_frac < 0.0f || params->q4_outlier_max_frac > 1.0f) {
            throw std::runtime_error("invalid Q4_0_BF16_OUTLIER thresholds");
        }
    }

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    const llama_model_kv_override * kv_overrides = params->kv_overrides;
    std::vector<std::string> splits = {};
    llama_model_loader ml(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
        fname_inp, splits, /*file*/ nullptr, use_mmap, /*use_direct_io*/ false, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    auto mparams = llama_model_default_params();
    std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, mparams));

    auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
    if (model == nullptr) {
        GGML_ABORT("fatal error: model does not implement llama_model_base");
    }

    model->load_hparams(ml);
    model->load_stats  (ml);

    quantize_state_impl qs(*model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    std::unordered_map<std::string, std::vector<float>> i_data;
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        for (const llama_model_imatrix_data * p = params->imatrix; p->name != nullptr; p++) {
            i_data.emplace(p->name, std::vector<float>(p->data, p->data + p->size));
        }
        imatrix_data = & i_data;
        if (imatrix_data) {
            LLAMA_LOG_INFO("\n%s: have importance matrix data with %d entries\n",
                           __func__, (int)imatrix_data->size());
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        for (const int32_t * p = params->prune_layers; * p != -1; p++) {
            prune_list.push_back(* p);
        }
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.metadata);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION); // TODO: use LLM_KV
    gguf_set_val_u32(ctx_out.get(), "general.file_type",
            q8_outlier_enabled ? LLAMA_FTYPE_MOSTLY_Q8_0_BF16_OUTLIER : ftype); // TODO: use LLM_KV

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    // Remove stale Q8_0_BF16_OUTLIER metadata when target is plain Q8_0
    if (!q8_outlier_enabled) {
        for (int k = 0; k < gguf_get_n_kv(ctx_out.get()); ++k) {
            const char * key = gguf_get_key(ctx_out.get(), k);
            if (key && std::string(key).rfind("llama.q8_outlier.", 0) == 0) {
                gguf_remove_key(ctx_out.get(), key);
                --k;
            }
        }
    }

    if (params->kv_overrides) {
        for (const llama_model_kv_override * o = params->kv_overrides; o->key[0] != 0; ++o) {
            if (o->tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o->key, o->val_f64);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), o->key, (uint32_t)std::abs(o->val_i64));
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o->key, o->val_bool);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o->key, o->val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o->key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        tensors.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    // compute tensor metadata once and cache it
    std::vector<tensor_metadata> metadata(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        metadata[i].name = ggml_get_name(tensors[i]->tensor);
    }

    // initialize quantization state counters and metadata categories
    init_quantize_state_counters(qs, metadata);

    int idx = 0;
    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // flag for --dry-run
    bool will_require_imatrix = false;

    //
    // preliminary iteration over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * it = tensors[i];
        const struct ggml_tensor * tensor = it->tensor;

        // Skip sidecar tensors when outputting plain Q8_0
        if (!q8_outlier_enabled) {
            const std::string & tname = ggml_get_name(tensor);
            if ((tname.size() > 12 && tname.rfind(".outlier_idx") == tname.size() - 12) ||
                (tname.size() > 13 && tname.rfind(".outlier_bf16") == tname.size() - 13)) {
                continue;
            }
        }

        // Skip sidecar tensors when outputting plain Q4_0
        if (!q4_outlier_enabled) {
            const std::string & tname = ggml_get_name(tensor);
            if ((tname.size() > 12 && tname.rfind(".outlier_idx") == tname.size() - 12) ||
                (tname.size() > 13 && tname.rfind(".outlier_bf16") == tname.size() - 13)) {
                continue;
            }
        }

        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        metadata[i].allows_quantization = tensor_allows_quantization(params, model->arch, tensor);

        if (metadata[i].allows_quantization) {
            metadata[i].target_type = llama_tensor_get_type(qs, params, tensor, default_type, metadata[i]);
        } else {
            metadata[i].target_type = tensor->type;
        }

        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);

        if (params->imatrix) {
            metadata[i].remapped_imatrix_name = remap_imatrix(tensor->name, mapped);
        } else if (metadata[i].allows_quantization && metadata[i].requires_imatrix) {
            if (params->dry_run) {
                will_require_imatrix = true;
            } else {
                LLAMA_LOG_ERROR("\n============================================================================\n"
                                " ERROR: this quantization requires an importance matrix!\n"
                                "        - offending tensor: %s\n"
                                "        - target type: %s\n"
                                "============================================================================\n\n",
                                metadata[i].name.c_str(), ggml_type_name(metadata[i].target_type));
                throw std::runtime_error("this quantization requires an imatrix!");
            }
        }
    }

    std::vector<q8_outlier_tensor_data> q8_outlier_tensors;
    std::unordered_map<std::string, size_t> q8_outlier_by_name;
    if (q8_outlier_enabled) {
        std::vector<no_init<uint8_t>> scan_read_data;
        std::vector<no_init<float>> scan_f32_conv_buf;
        std::vector<std::thread> scan_workers;
        scan_workers.reserve(nthread);

        // Load source outlier sidecar data for reconstruction during re-analysis
        struct q8_outlier_source_info {
            std::vector<int32_t> idx;
            std::vector<ggml_bf16_t> values;
        };
        std::unordered_map<std::string, q8_outlier_source_info> source_outlier_info;
        if (ml.has_q8_outlier_metadata()) {
            auto src_meta = ml.read_q8_outlier_metadata();
            for (const auto & meta : src_meta) {
                auto idx_it = ml.weights_map.find(meta.idx_name);
                auto vals_it = ml.weights_map.find(meta.values_name);
                if (idx_it == ml.weights_map.end() || vals_it == ml.weights_map.end()) {
                    continue;
                }
                q8_outlier_source_info info;
                const ggml_tensor * idx_tensor = idx_it->second.tensor;
                const ggml_tensor * vals_tensor = vals_it->second.tensor;
                info.idx.resize(ggml_nelements(idx_tensor));
                info.values.resize(ggml_nelements(vals_tensor));
                ml.load_data_for(const_cast<ggml_tensor *>(idx_tensor));
                ml.load_data_for(const_cast<ggml_tensor *>(vals_tensor));
                std::memcpy(info.idx.data(), idx_tensor->data, ggml_nbytes(idx_tensor));
                std::memcpy(info.values.data(), vals_tensor->data, ggml_nbytes(vals_tensor));
                source_outlier_info.emplace(meta.base_name, std::move(info));
            }
        }

        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto & tm = metadata[i];
            ggml_tensor * tensor = tensors[i]->tensor;

            if (!tm.allows_quantization || tm.target_type != GGML_TYPE_Q8_0) {
                continue;
            }
            if (ggml_n_dims(tensor) != 2) {
                LLAMA_LOG_WARN("%s: skipping Q8 outlier sidecars for %s because only 2D tensors are supported in this prototype\n",
                        __func__, tm.name.c_str());
                continue;
            }
            if (!q8_outlier_tensor_enabled(params, tm.name)) {
                continue;
            }

            const size_t tensor_size = ggml_nbytes(tensor);
            if (!ml.use_mmap) {
                if (scan_read_data.size() < tensor_size) {
                    scan_read_data.resize(tensor_size);
                }
                tensor->data = scan_read_data.data();
            }
            ml.load_data_for(tensor);

            const int64_t nelements = ggml_nelements(tensor);
            float * f32_data = nullptr;
            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_tensor_dequantize_impl(tensor, scan_f32_conv_buf, scan_workers, nelements, nthread);
                f32_data = (float *) scan_f32_conv_buf.data();
            }

            // If source has outlier sidecars, reconstruct full tensor for accurate re-analysis
            if (tensor->type == GGML_TYPE_Q8_0) {
                auto src_outlier_it = source_outlier_info.find(tm.name);
                if (src_outlier_it != source_outlier_info.end()) {
                    q8_outlier_reconstruct_tensor(
                            tensor,
                            src_outlier_it->second.idx,
                            src_outlier_it->second.values,
                            f32_data,
                            tensor->ne[1],
                            tensor->ne[0]);
                }
            }

            const float * imatrix = nullptr;
            if (imatrix_data) {
                auto it_imatrix = imatrix_data->find(tm.remapped_imatrix_name);
                if (it_imatrix != imatrix_data->end()) {
                    if (it_imatrix->second.size() == (size_t) tensor->ne[0]) {
                        imatrix = it_imatrix->second.data();
                    } else if (!tensor_name_match_token_embd(tensor->name)) {
                        throw std::runtime_error(format("imatrix size %d is different from tensor row size %d for %s",
                                int(it_imatrix->second.size()), int(tensor->ne[0]), tensor->name));
                    }
                }
            }

            q8_outlier_tensor_data outliers = q8_outlier_analyze_tensor(
                    tm.name,
                    f32_data,
                    tensor->ne[1],
                    tensor->ne[0],
                    imatrix,
                    params);

            if (outliers.n_blocks() == 0) {
                continue;
            }

            const double frac = double(outliers.n_blocks()) / double(outliers.n_blocks_total);
            LLAMA_LOG_INFO("%s: %-36s - q8 outlier blocks = %" PRId64 "/%" PRId64 " (%.3f%%), est %.3f bpw\n",
                    __func__, tm.name.c_str(), outliers.n_blocks(), outliers.n_blocks_total, 100.0 * frac, outliers.estimated_bpw);

            q8_outlier_by_name.emplace(outliers.name, q8_outlier_tensors.size());
            q8_outlier_tensors.push_back(std::move(outliers));
        }

        for (const q8_outlier_tensor_data & outliers : q8_outlier_tensors) {
            q8_outlier_add_tensor_meta(ctx_outs[0].get(), outliers.idx_name, GGML_TYPE_I32, 2, outliers.n_blocks());
            q8_outlier_add_tensor_meta(ctx_outs[0].get(), outliers.values_name, GGML_TYPE_BF16, LLAMA_Q8_OUTLIER_BLOCK_SIZE, outliers.n_blocks());
        }
        // Only write/overwrite outlier metadata if we actually analyzed tensors.
        // In copy mode, source metadata was already preserved by gguf_set_kv above.
        if (!params->only_copy) {
            q8_outlier_write_metadata(ctx_outs[0].get(), q8_outlier_tensors);
        }
    }

    // Q4_0_BF16_OUTLIER: analyze tensors for outlier blocks
    std::vector<q8_outlier_tensor_data> q4_outlier_tensors;
    std::unordered_map<std::string, size_t> q4_outlier_by_name;
    if (q4_outlier_enabled) {
        std::vector<no_init<uint8_t>> scan_read_data;
        std::vector<no_init<float>> scan_f32_conv_buf;
        std::vector<std::thread> scan_workers;
        scan_workers.reserve(nthread);

        // Load source outlier sidecar data for reconstruction during re-analysis
        struct q4_outlier_source_info {
            std::vector<int32_t> idx;
            std::vector<ggml_bf16_t> values;
        };
        std::unordered_map<std::string, q4_outlier_source_info> source_outlier_info;
        if (ml.has_q4_outlier_metadata()) {
            auto src_meta = ml.read_q4_outlier_metadata();
            for (const auto & meta : src_meta) {
                auto idx_it = ml.weights_map.find(meta.idx_name);
                auto vals_it = ml.weights_map.find(meta.values_name);
                if (idx_it == ml.weights_map.end() || vals_it == ml.weights_map.end()) {
                    continue;
                }
                q4_outlier_source_info info;
                const ggml_tensor * idx_tensor = idx_it->second.tensor;
                const ggml_tensor * vals_tensor = vals_it->second.tensor;
                info.idx.resize(ggml_nelements(idx_tensor));
                info.values.resize(ggml_nelements(vals_tensor));
                ml.load_data_for(const_cast<ggml_tensor *>(idx_tensor));
                ml.load_data_for(const_cast<ggml_tensor *>(vals_tensor));
                std::memcpy(info.idx.data(), idx_tensor->data, ggml_nbytes(idx_tensor));
                std::memcpy(info.values.data(), vals_tensor->data, ggml_nbytes(vals_tensor));
                source_outlier_info.emplace(meta.base_name, std::move(info));
            }
        }

        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto & tm = metadata[i];
            ggml_tensor * tensor = tensors[i]->tensor;

            if (!tm.allows_quantization || tm.target_type != GGML_TYPE_Q4_0) {
                continue;
            }
            if (ggml_n_dims(tensor) != 2) {
                LLAMA_LOG_WARN("%s: skipping Q4 outlier sidecars for %s because only 2D tensors are supported in this prototype\n",
                        __func__, tm.name.c_str());
                continue;
            }
            if (!q4_outlier_tensor_enabled(params, tm.name)) {
                continue;
            }

            const size_t tensor_size = ggml_nbytes(tensor);
            if (!ml.use_mmap) {
                if (scan_read_data.size() < tensor_size) {
                    scan_read_data.resize(tensor_size);
                }
                tensor->data = scan_read_data.data();
            }
            ml.load_data_for(tensor);

            const int64_t nelements = ggml_nelements(tensor);
            float * f32_data = nullptr;
            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_tensor_dequantize_impl(tensor, scan_f32_conv_buf, scan_workers, nelements, nthread);
                f32_data = (float *) scan_f32_conv_buf.data();
            }

            // If source has outlier sidecars, reconstruct full tensor for accurate re-analysis
            if (tensor->type == GGML_TYPE_Q4_0) {
                auto src_outlier_it = source_outlier_info.find(tm.name);
                if (src_outlier_it != source_outlier_info.end()) {
                    q4_outlier_reconstruct_tensor(
                            tensor,
                            src_outlier_it->second.idx,
                            src_outlier_it->second.values,
                            f32_data,
                            tensor->ne[1],
                            tensor->ne[0]);
                }
            }

            const float * imatrix = nullptr;
            if (imatrix_data) {
                auto it_imatrix = imatrix_data->find(tm.remapped_imatrix_name);
                if (it_imatrix != imatrix_data->end()) {
                    if (it_imatrix->second.size() == (size_t) tensor->ne[0]) {
                        imatrix = it_imatrix->second.data();
                    } else if (!tensor_name_match_token_embd(tensor->name)) {
                        throw std::runtime_error(format("imatrix size %d is different from tensor row size %d for %s",
                                int(it_imatrix->second.size()), int(tensor->ne[0]), tensor->name));
                    }
                }
            }

            q8_outlier_tensor_data outliers = q4_outlier_analyze_tensor(
                    tm.name,
                    f32_data,
                    tensor->ne[1],
                    tensor->ne[0],
                    imatrix,
                    params);

            if (outliers.n_blocks() == 0) {
                continue;
            }

            const double frac = double(outliers.n_blocks()) / double(outliers.n_blocks_total);
            LLAMA_LOG_INFO("%s: %-36s - q4 outlier blocks = %" PRId64 "/%" PRId64 " (%.3f%%), est %.3f bpw\n",
                    __func__, tm.name.c_str(), outliers.n_blocks(), outliers.n_blocks_total, 100.0 * frac, outliers.estimated_bpw);

            q4_outlier_by_name.emplace(outliers.name, q4_outlier_tensors.size());
            q4_outlier_tensors.push_back(std::move(outliers));
        }

        for (const q8_outlier_tensor_data & outliers : q4_outlier_tensors) {
            q4_outlier_add_tensor_meta(ctx_outs[0].get(), outliers.idx_name, GGML_TYPE_I32, 2, outliers.n_blocks());
            q4_outlier_add_tensor_meta(ctx_outs[0].get(), outliers.values_name, GGML_TYPE_BF16, LLAMA_Q4_OUTLIER_BLOCK_SIZE, outliers.n_blocks());
        }
        if (!params->only_copy) {
            q4_outlier_write_metadata(ctx_outs[0].get(), q4_outlier_tensors);
            q4_outlier_write_report(q4_outlier_tensors, params);
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)tensors.size());
        }
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    // no output file for --dry-run
    if (!params->dry_run) {
        new_ofstream(0);
    }

    //
    // main loop: iterate over all weights
    //

    // Pre-load sidecar data for reconstruction when converting Q8_0_BF16_OUTLIER -> plain Q8_0
    std::unordered_map<std::string, std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> q8_outlier_sidecars;
    if (!q8_outlier_enabled) {
        for (const auto & it : ml.weights_map) {
            const std::string & name = it.first;
            const size_t idx_len = std::string(".outlier_idx").size();
            const size_t bf16_len = std::string(".outlier_bf16").size();
            if (name.size() > idx_len && name.rfind(".outlier_idx") == name.size() - idx_len) {
                const std::string base_name = name.substr(0, name.size() - idx_len);
                const size_t sz = ggml_nbytes(it.second.tensor);
                q8_outlier_sidecars[base_name].first.resize(sz);
                ml.load_data_for(it.second.tensor);
                std::memcpy(q8_outlier_sidecars[base_name].first.data(), it.second.tensor->data, sz);
            } else if (name.size() > bf16_len && name.rfind(".outlier_bf16") == name.size() - bf16_len) {
                const std::string base_name = name.substr(0, name.size() - bf16_len);
                const size_t sz = ggml_nbytes(it.second.tensor);
                q8_outlier_sidecars[base_name].second.resize(sz);
                ml.load_data_for(it.second.tensor);
                std::memcpy(q8_outlier_sidecars[base_name].second.data(), it.second.tensor->data, sz);
            }
        }
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & weight = *tensors[i];
        const auto & tm = metadata[i];
        ggml_tensor * tensor = weight.tensor;

        // Skip sidecar tensors when outputting plain Q8_0
        if (!q8_outlier_enabled) {
            const std::string & tname = tm.name;
            if ((tname.size() > 12 && tname.rfind(".outlier_idx") == tname.size() - 12) ||
                (tname.size() > 13 && tname.rfind(".outlier_bf16") == tname.size() - 13)) {
                continue;
            }
        }

        if (!params->dry_run && (weight.idx != cur_split && params->keep_split)) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const size_t tensor_size = ggml_nbytes(tensor);

        if (!params->dry_run) {
            if (!ml.use_mmap) {
                if (read_data.size() < tensor_size) {
                    read_data.resize(tensor_size);
                }
                tensor->data = read_data.data();
            }
            ml.load_data_for(tensor);
        }

        LLAMA_LOG_INFO("[%4d/%4d] %-36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        const ggml_type cur_type = tensor->type;
        const ggml_type new_type = tm.target_type;

        // If we've decided to quantize to the same type the tensor is already
        // in then there's nothing to do.
        bool quantize = cur_type != new_type;

        void * new_data;
        size_t new_size;

        if (params->dry_run) {
            // the --dry-run option calculates the final quantization size without quantizing
            if (quantize) {
                new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB (%s)\n",
                               tensor_size/1024.0/1024.0,
                               new_size/1024.0/1024.0,
                               ggml_type_name(new_type));
                if (!will_require_imatrix && tm.requires_imatrix) {
                    will_require_imatrix = true;
                }
            } else {
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
            continue;
        } else {
            // no --dry-run, perform quantization
            if (!quantize) {
                new_data = tensor->data;
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", tensor_size/1024.0/1024.0);
            } else {
                const int64_t nelements = ggml_nelements(tensor);

                const float * imatrix = nullptr;
                if (imatrix_data) {
                    auto it = imatrix_data->find(tm.remapped_imatrix_name);
                    if (it == imatrix_data->end()) {
                        LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                    } else {
                        if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                            imatrix = it->second.data();
                        } else {
                            LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                            // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                            // this is a significant error and it may be good idea to abort the process if this happens,
                            // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                            // tok_embd should be ignored in this case, since it always causes this warning
                            if (!tensor_name_match_token_embd(tensor->name)) {
                                throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                        int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                            }
                        }
                    }
                }
                if (!imatrix && tm.requires_imatrix) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }

                float * f32_data;

                if (tensor->type == GGML_TYPE_F32) {
                    f32_data = (float *) tensor->data;
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();

                    // Reconstruct full tensor from Q8_0 base + BF16 outlier sidecar
                    if (tensor->type == GGML_TYPE_Q8_0 && ggml_n_dims(tensor) == 2) {
                        auto sit = q8_outlier_sidecars.find(tm.name);
                        if (sit != q8_outlier_sidecars.end() &&
                            !sit->second.first.empty() && !sit->second.second.empty()) {
                            ggml_tensor idx_tensor;
                            ggml_tensor values_tensor;
                            std::memset(&idx_tensor, 0, sizeof(idx_tensor));
                            std::memset(&values_tensor, 0, sizeof(values_tensor));
                            idx_tensor.type = GGML_TYPE_I32;
                            idx_tensor.data = sit->second.first.data();
                            values_tensor.type = GGML_TYPE_BF16;
                            values_tensor.data = sit->second.second.data();
                            q8_outlier_reconstruct_tensor(tensor, &idx_tensor, &values_tensor, f32_data, tensor->ne[1], tensor->ne[0]);
                        }
                    }
                }

                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
                fflush(stdout);

                if (work.size() < (size_t)nelements * 4) {
                    work.resize(nelements * 4); // upper bound on size
                }
                new_data = work.data();

                const int64_t n_per_row = tensor->ne[0];
                const int64_t nrows = tensor->ne[1];

                static const int64_t min_chunk_size = 32 * 512;
                const int64_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

                const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
                const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
                const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

                // quantize each expert separately since they have different importance matrices
                new_size = 0;
                for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
                    const float * f32_data_03 = f32_data + i03 * nelements_matrix;
                    void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
                    const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

                    new_size += llama_tensor_quantize_impl(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use);
                }

                // Compute BF16 deltas for outlier blocks: delta = original_F32 - Q8_dequantized
                // Store deltas instead of full values, so Q8 base remains intact for
                // operations that don't support correction (e.g., ggml_get_rows for embedding lookup).
                // Where correction IS applied (ggml_mul_mat via build_lora_mm): Q8 + delta ≈ original
                // Where correction is NOT applied (ggml_get_rows): Q8 approximation (same as plain Q8_0)
                if (q8_outlier_enabled && new_type == GGML_TYPE_Q8_0) {
                    auto it = q8_outlier_by_name.find(tm.name);
                    if (it != q8_outlier_by_name.end()) {
                        q8_outlier_compute_deltas(f32_data, new_data, nrows, n_per_row, q8_outlier_tensors[it->second]);
                    }
                }
                if (q4_outlier_enabled && new_type == GGML_TYPE_Q4_0) {
                    auto it = q4_outlier_by_name.find(tm.name);
                    if (it != q4_outlier_by_name.end()) {
                        q4_outlier_compute_deltas(f32_data, new_data, nrows, n_per_row, q4_outlier_tensors[it->second]);
                    }
                }

                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;

            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), metadata[i].name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_data);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
        } // no --dry-run
    } // main loop

    // Write Q8_0_BF16_OUTLIER sidecar tensor data
    if (q8_outlier_enabled && !params->dry_run) {
        for (const q8_outlier_tensor_data & outliers : q8_outlier_tensors) {
            // Write outlier_idx tensor data
            {
                const size_t idx_size = outliers.idx.size() * sizeof(int32_t);
                gguf_set_tensor_data(ctx_outs[cur_split].get(), outliers.idx_name.c_str(), outliers.idx.data());
                fout.write((const char *) outliers.idx.data(), idx_size);
                zeros(fout, GGML_PAD(idx_size, align) - idx_size);
            }
            // Write outlier_bf16 tensor data
            {
                const size_t values_size = outliers.values.size() * sizeof(ggml_bf16_t);
                gguf_set_tensor_data(ctx_outs[cur_split].get(), outliers.values_name.c_str(), outliers.values.data());
                fout.write((const char *) outliers.values.data(), values_size);
                zeros(fout, GGML_PAD(values_size, align) - values_size);
            }
        }
    }

    // Write Q4_0_BF16_OUTLIER sidecar tensor data
    if (q4_outlier_enabled && !params->dry_run) {
        for (const q8_outlier_tensor_data & outliers : q4_outlier_tensors) {
            {
                const size_t idx_size = outliers.idx.size() * sizeof(int32_t);
                gguf_set_tensor_data(ctx_outs[cur_split].get(), outliers.idx_name.c_str(), outliers.idx.data());
                fout.write((const char *) outliers.idx.data(), idx_size);
                zeros(fout, GGML_PAD(idx_size, align) - idx_size);
            }
            {
                const size_t values_size = outliers.values.size() * sizeof(ggml_bf16_t);
                gguf_set_tensor_data(ctx_outs[cur_split].get(), outliers.values_name.c_str(), outliers.values.data());
                fout.write((const char *) outliers.values.data(), values_size);
                zeros(fout, GGML_PAD(values_size, align) - values_size);
            }
        }
    }

    if (!params->dry_run) {
        close_ofstream();
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n",
                       __func__
        );
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, ml.n_tensors);
    }
}

//
// interface implementation
//

llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q8_0,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_type                 =*/ nullptr,
        /*.prune_layers                =*/ nullptr,
        /*.q8_outlier_enable           =*/ false,
        /*.q8_outlier_ratio            =*/ 16.0f,
        /*.q8_outlier_nonmax_rel_rmse  =*/ 0.01f,
        /*.q8_outlier_max_frac         =*/ 0.02f,
        /*.q8_outlier_store            =*/ LLAMA_Q8_OUTLIER_STORE_FULL,
        /*.q8_outlier_report_path      =*/ nullptr,
        /*.q8_outlier_include_weights  =*/ nullptr,
        /*.q8_outlier_exclude_weights  =*/ nullptr,
        /*.q4_outlier_enable           =*/ false,
        /*.q4_outlier_ratio            =*/ 16.0f,
        /*.q4_outlier_nonmax_rel_rmse  =*/ 0.01f,
        /*.q4_outlier_max_frac         =*/ 0.02f,
        /*.q4_outlier_report_path      =*/ nullptr,
        /*.q4_outlier_include_weights  =*/ nullptr,
        /*.q4_outlier_exclude_weights  =*/ nullptr
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}

//
// Helper functions for external tools exposed in llama-ext.h
//

quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params) {
    return new quantize_state_impl(*model, params);
}

void llama_quant_free(quantize_state_impl * qs) {
    delete qs;
}

llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc) {
    struct llama_model_params mparams = llama_model_default_params();
    auto arch = llm_arch_from_string(desc->architecture);
    auto * model = llama_model_create(arch, mparams);
    model->arch = arch;

    // infer llm_type: only LLM_TYPE_70B matters for quantization logic
    if (model->arch == LLM_ARCH_LLAMA && desc->n_layer == 80 && desc->n_head != desc->n_head_kv) {
        model->type = LLM_TYPE_70B;
    }

    model->hparams.n_embd             = desc->n_embd;
    model->hparams.n_embd_head_k_full = desc->n_embd_head_k;
    model->hparams.n_embd_head_v_full = desc->n_embd_head_v;
    model->hparams.n_layer            = desc->n_layer;
    model->hparams.n_expert           = desc->n_expert;

    for (uint32_t i = 0; i < desc->n_layer; i++) {
        model->hparams.n_head_arr[i]    = desc->n_head;
        model->hparams.n_head_kv_arr[i] = desc->n_head_kv;
        model->hparams.n_ff_arr[i]      = desc->n_ff;
    }

    return model;
}

bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor) {
    return tensor_allows_quantization(qs->params, qs->model.arch, tensor);
}

void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors) {
    // reset per-computation state
    qs->n_attention_wv      = 0;
    qs->n_ffn_down          = 0;
    qs->n_ffn_gate          = 0;
    qs->n_ffn_up            = 0;
    qs->i_attention_wv      = 0;
    qs->i_ffn_down          = 0;
    qs->i_ffn_gate          = 0;
    qs->i_ffn_up            = 0;
    qs->n_fallback          = 0;
    qs->has_imatrix         = false;
    qs->has_tied_embeddings = true;

    // build metadata from tensor names
    std::vector<tensor_metadata> metadata(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        metadata[i].name = ggml_get_name(tensors[i]);
    }

    // initialize counters and categories
    init_quantize_state_counters(*qs, metadata);

    // use a local copy of params with the requested ftype
    llama_model_quantize_params local_params = *qs->params;
    local_params.ftype = ftype;

    ggml_type default_type = llama_ftype_get_default_type(ftype);

    // compute types
    for (size_t i = 0; i < n_tensors; i++) {
        result_types[i] = llama_tensor_get_type(*qs, &local_params, tensors[i], default_type, metadata[i]);
    }
}
