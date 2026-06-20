# RYS (Repeat Your Self) Implementation Instructions

## Overview
Branch of llama.cpp that does RYS (inference-time looping) based on parameters:
- --loop-layer-start X
- --loop-layer-stop X

Fork: https://github.com/vektorprime/llama.cpp
Branch: dup_layers_llama

## Core Concepts
- RYS papers: https://dnhkng.github.io/posts/sapir-whorf/ https://dnhkng.github.io/posts/rys/ https://dnhkng.github.io/posts/rys-ii/
- Python reference: https://github.com/dnhkng/RYS
- Three-phase hypothesis: early layers decode, middle layers reason, late layers re-encode
- Layer duplication at inference time with NO weight changes, NO training

## Target Model
- Qwen 3.6 27B: https://huggingface.co/Qwen/Qwen3.6-27B
- Architecture: qwen3next (LLM_ARCH_QWEN3NEXT), 64 layers
- Loop target: layers 24 to 35 (inclusive)
- Model file: /home/user/llm/models/Qwen3.6-27B/Qwen3.6-27B-UD-Q6_K_XL.gguf

## Development Machine
- Only validate syntax/code on this machine
- Do NOT compile on this machine

## Build/Test Server
- Ubuntu 26.04, IP: 10.0.0.188
- user/pass: user/Ubuntu1337!
- Folder: /home/user/llm/dup_layers_llama/llama.cpp/
- Model: /home/user/llm/models/Qwen3.6-27B/

## Build commands on 10.0.0.188
```
cd /home/user/llm/dup_layers_llama/llama.cpp
git pull
rm -rf build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel
```

## Validation test on 10.0.0.188
```
/home/user/llm/dup_layers_llama/llama.cpp/build/bin/llama-cli \
-m /home/user/llm/models/Qwen3.6-27B/Qwen3.6-27B-UD-Q6_K_XL.gguf \
-t 8 -c 32768 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --no-mmap -ngl 999 -np 1 \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 --presence-penalty 0.0 --repeat-penalty 1.0 \
--loop-layer-start 24 --loop-layer-stop 35 --custom-logs \
-p "hello"
```

## Debugging
- Use --custom-logs for debug logging (works in llama-cli and llama-perplexity)
- For crashes with massive logs, redirect to file
- Check VRAM with nvidia-smi

## Hardware Scope
- Only CUDA, CPU, and tensor split pipelines
- No implementations for other backends

## Constraints
- Test with Qwen3.6 27B dense model
- Don't implement useless changes; test every change
- Follow C++ core guidelines
- Prefer not asking questions; work with constraints
- Do not compile on this machine (only on 10.0.0.188)

## Tracking
Progress tracked in dup-layers-architecture-implementation.md
