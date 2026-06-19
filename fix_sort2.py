import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

old_block = '''    // Also write a top-N worst-tokens report with context
    if (!params.analysis_csv.empty() && !token_analyses.empty()) {
        // Sort token_analyses by absolute p_diff
        std::vector<token_analysis> sorted = token_analyses;
        std::sort(sorted.begin(), sorted.end(),
            [](const token_analysis & a, const token_analysis & b) {
                float fa = a.p_diff < 0 ? -a.p_diff : a.p_diff;
                float fb = b.p_diff < 0 ? -b.p_diff : b.p_diff;
                return fa > fb;
            });

        std::string worst_path = params.analysis_csv + ".worst_tokens.txt";'''

new_block = '''    // Also write a top-N worst-tokens report with context
    if (!params.analysis_csv.empty() && !token_analyses.empty()) {
        // Sort token_analyses by absolute p_diff (manual insertion sort, avoids lambda issues)
        std::vector<token_analysis> sorted = token_analyses;
        for (size_t i = 1; i < sorted.size(); i++) {
            token_analysis key = sorted[i];
            float key_abs = key.p_diff < 0 ? -key.p_diff : key.p_diff;
            int j = (int)i - 1;
            while (j >= 0) {
                float cur_abs = sorted[j].p_diff < 0 ? -sorted[j].p_diff : sorted[j].p_diff;
                if (cur_abs >= key_abs) break;
                sorted[j + 1] = sorted[j];
                j--;
            }
            sorted[j + 1] = key;
        }

        std::string worst_path = params.analysis_csv + ".worst_tokens.txt";'''

if old_block not in content:
    print("ERROR: old block not found!")
    # Try to show context
    idx = content.find('worst-tokens report')
    if idx >= 0:
        print(content[idx:idx+800])
    sys.exit(1)

content = content.replace(old_block, new_block)
with open(sys.argv[1], 'w') as f:
    f.write(content)
print("done")
