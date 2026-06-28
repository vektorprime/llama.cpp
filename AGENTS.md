Building may take up to 15 minutes
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel


You may test with this model:
/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-Q4T.gguf

llama-server \
-m/home/user/llm/models/Qwen3.5-2B/Qwen3.5-2B-Q4T.gguf \
 --port 8128 --host 0.0.0.0 -a Qwen3.6-2B  \
 --no-mmap --threads 12 --jinja   -c 30000 \
 --cache-type-k bf16 --cache-type-v bf16 --flash-attn on -kvu  -ngl 99 -np 1 \
 --temp 1.0  --top-p 1.0 --top-k 20 --min-p 0.0 --presence-penalty 2.0 --repeat-penalty 1.0 \

Track the implementation, issues and fixes in an IMPLEMENTATION.md file
