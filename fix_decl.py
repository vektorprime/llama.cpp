import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# Insert struct definition + vector after p_diff_values line
old = '    std::vector<float> p_diff_values(size_t(n_ctx - 1 - n_ctx/2)*n_chunk);\n    std::vector<float> logits;'

new = '''    std::vector<float> p_diff_values(size_t(n_ctx - 1 - n_ctx/2)*n_chunk);

    // Per-token analysis tracking: record token IDs and positions alongside p_diff/kld
    struct token_analysis {
        int32_t chunk;
        int32_t seq;
        int32_t pos;        // position within context window
        int32_t token_id;
        float   p_diff;
        float   kld;
    };
    std::vector<token_analysis> token_analyses;
    token_analyses.reserve(p_diff_values.size());

    std::vector<float> logits;'''

if old not in content:
    print("ERROR: old string not found")
    sys.exit(1)

content = content.replace(old, new)
with open(sys.argv[1], 'w') as f:
    f.write(content)
print("done")
