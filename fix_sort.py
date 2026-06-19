import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

old = '''        std::sort(sorted.begin(), sorted.end(),
            [](const token_analysis & a, const token_analysis & b) {
                return std::fabs(a.p_diff) > std::fabs(b.p_diff);
            });'''

new = '''        std::sort(sorted.begin(), sorted.end(),
            [](const token_analysis & a, const token_analysis & b) {
                float fa = a.p_diff < 0 ? -a.p_diff : a.p_diff;
                float fb = b.p_diff < 0 ? -b.p_diff : b.p_diff;
                return fa > fb;
            });'''

if old not in content:
    print("ERROR: old string not found")
    sys.exit(1)

content = content.replace(old, new)
with open(sys.argv[1], 'w') as f:
    f.write(content)
print("done")
