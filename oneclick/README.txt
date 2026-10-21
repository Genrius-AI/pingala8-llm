Pingala-8 one-click runtime
===========================
Patched llama.cpp (commit 7ac59a6) that runs the Pingala-8 B4 2-bit models. Needs an NVIDIA GPU (RTX 20-series or newer)
with a recent driver.

Windows: double-click run_qwen3_4b.bat (or run_llama31_8b.bat). The first run downloads the model, then a chat page
opens at http://127.0.0.1:8080. Close the black window to stop.
Linux:   ./run.sh qwen   (or ./run.sh llama), then open http://127.0.0.1:8080.

Same file size as llama.cpp IQ2_XXS. WikiText-2 perplexity on Qwen3-4B: 16.64 (IQ2_XXS 28.61). It loses to IQ2_KT on
Qwen3-14B. Paper: https://doi.org/10.5281/zenodo.23481757
Code: https://github.com/Genrius-AI/pingala8-llm (Apache-2.0). llama.cpp is MIT (llama.cpp-LICENSE).
