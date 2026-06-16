We only care about CUDA and CPU pipelines

Unless stated otherwise, we are testing with Qwen3.5 and Qwen 3.6 dense models, usually Qwen 3.5 2B.

If working on the q4_0 outlier code, always make sure you've read Q4_0_BF16_OUTLIER_IMPLEMENTATION.md
If working on the q8_0 outlier clode, always make sure you've read Q8_0_BF16_OUTLIER_IMPLEMENTATION.md

Do not build this project, only analyze and edit the code.

When complete, validate syntax with MSVC/s cl.exe
example with cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" \Zs llama-io.cpp
example with powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" \Zs llama-io.cpp

When fixing compilation errors, remember to also run git push after the commit.

## Gotchas

### OUTPUT tensor category in llama_tensor_get_type_impl
`src/llama-quant.cpp`: In the OUTPUT category block (~line 447), there's a catch-all `else if`
at the end that forces unknown ftypes to `GGML_TYPE_Q6_K`. When adding a new ftype, you must
either add an explicit case in that block or exclude the new ftype from the catch-all condition
(like `LLAMA_FTYPE_MOSTLY_Q4_0_BF16_OUTLIER` and `LLAMA_FTYPE_MOSTLY_DF11` are excluded).
Otherwise output tensors (output_norm.weight, etc.) won't use the new type.

