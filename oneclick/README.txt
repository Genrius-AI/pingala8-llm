Pingala-8 one-click runtime
===========================
Patched llama.cpp (commit 7ac59a6) that runs the Pingala-8 2-bit models. v2.5 adds the loader change for the hybrid
Qwen3.5 models (patches/q35_loader_count_delta.patch). Needs an NVIDIA GPU (RTX 30, 40 or 50 series; the build
targets these three architectures) with a recent driver.

Five models (v2.5). Pick one launcher; the first run downloads that model, then a chat page opens at
http://127.0.0.1:8080. Close the black window (Windows) or press Ctrl+C (Linux) to stop.

  Model           Download   Windows                 Linux
  Qwen3-4B        1.24 GB    run_qwen3_4b.bat        ./run.sh qwen
  Llama-3.1-8B    2.39 GB    run_llama31_8b.bat      ./run.sh llama   (base model: it continues text; Built with Llama)
  Qwen3.5-4B      1.51 GB    run_qwen35_4b.bat       ./run.sh qwen35-4b    (or ./run_qwen35_4b.sh)
  Qwen3.5-2B      0.78 GB    run_qwen35_2b.bat       ./run.sh qwen35-2b    (or ./run_qwen35_2b.sh)
  Qwen3.5-0.8B    0.35 GB    run_qwen35_0_8b.bat     ./run.sh qwen35-0.8b  (or ./run_qwen35_0_8b.sh)

Each part is downloaded to a .part file and checked (size and sha256) before it is used.

Same file size as llama.cpp IQ2_XXS. WikiText-2 perplexity on Qwen3-4B: 16.64 (IQ2_XXS 28.61). It loses to IQ2_KT on
Qwen3-14B. Paper: https://doi.org/10.5281/zenodo.23481757
Code: https://github.com/Genrius-AI/pingala8-llm (Apache-2.0). llama.cpp is MIT (llama.cpp-LICENSE).
