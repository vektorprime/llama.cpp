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

Then, push your changes to git

