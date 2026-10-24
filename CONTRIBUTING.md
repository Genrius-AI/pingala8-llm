# Contributing to Pingala-8

This repository holds the llama.cpp patches, scripts and logs behind the Pingala-8 preprint. Issues labelled
[good first issue](https://github.com/Genrius-AI/pingala8-llm/labels/good%20first%20issue) and
[help wanted](https://github.com/Genrius-AI/pingala8-llm/labels/help%20wanted) are the best places to start.
Questions go to [Discussions](https://github.com/Genrius-AI/pingala8-llm/discussions).

## What is where

- `patches/`: two alternative patches against llama.cpp `7ac59a6`: `llama_cuda_vs_7ac59a6.patch` (CUDA) and
  `llama_cpu_vs_7ac59a6.patch` (AVX2 CPU backend plus the same CUDA path). Apply one, not both.
- `research/`: encoder, builders, evaluation and benchmark scripts. Research code; read the caveats in `REPRODUCE.md`.
- `logs/`, `logs_bak_1014/`: the logs behind the paper's numbers. Do not edit them; they are the record.
- `oneclick/`: the launchers packed into the release downloads (`.github/workflows/build-windows.yml`).
- `B4.md` (the released files), `REPRODUCE.md` (runtime build, B3 reference build, file:line map).

## Build

CUDA, as in the README Quick start:

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 7ac59a6e3ad851cd41af00f678effab0598ba9a8
git apply --check ../pingala8-llm/patches/llama_cuda_vs_7ac59a6.patch && git apply ../pingala8-llm/patches/llama_cuda_vs_7ac59a6.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
cmake --build build -j8 --target llama-perplexity llama-bench llama-cli
```

Set `CMAKE_CUDA_ARCHITECTURES` for your GPU (86 RTX 30, 89 RTX 40, 120 RTX 50). On WSL2, `source bin/cuda_env.sh` first.
CPU: apply `llama_cpu_vs_7ac59a6.patch` instead, build without `-DGGML_CUDA=ON`, and run with `-ngl 0 --no-mmap`
(it does not read the Qwen3-4B file yet, see [#5](https://github.com/Genrius-AI/pingala8-llm/issues/5)).
Every run of a Pingala-8 file needs:

```
export GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2
```

Run stock GGUF files (IQ2_XXS, Q4_K_M) on the same binary with `GGML_F44=0` and without the other variables.

## Checks before a pull request

There is no unit-test suite. Run what applies to your change:

1. Any change (CPU only, seconds): `bash derived/recompute.sh | diff - derived/recomputed.txt` prints nothing.
2. Patch or kernel changes: `git apply --check` on a clean `7ac59a6` tree, a build, and the Qwen3-4B B4 perplexity with the
   README command (`-c 2049 --ppl-stride 4095 -ngl 99`). Expected `[1]16.0556,[2]19.6539,[3]16.9107,[4]14.7582` and
   `[73]16.6413` (`logs/r36_hfj_csq_native.log:15`). Explain any change in these values in the pull request.
3. Speed changes: llama-bench with the protocol of `research/detail/excl_bench_1017.sh`
   (`-ngl 99 -fa 1 -ctk f16 -ctv f16 -r 5`, pp512 and tg128), with a stock IQ2_XXS or Q4_K_M file on the same binary in the
   same session. Report ratios and the raw `-o json` output, not tok/s alone.

## Reporting results

Use the "Benchmark result" issue form, or comment on [#4](https://github.com/Genrius-AI/pingala8-llm/issues/4). Give the GPU, driver, CUDA version, OS, how you built, the exact
commands and the raw output. One GPU per report. Numbers that cannot be traced to a command and its output are not
added to the documentation.

## Code style

- Patches: follow llama.cpp's style. Keep new code in the files the patch adds (`f44.cu`, `vr49.cuh`, `vr49_dec.h`,
  `vr-native-cpu*`, `ggml-vr-native.h`, `llama-vr3.h`) and keep edits to upstream files small. A rebase or a new backend
  goes into a new file, `patches/llama_<backend>_vs_<short sha>.patch`; the `7ac59a6` patches stay as they are because
  the paper's numbers come from them.
- Scripts: run from the repository root, relative paths only, no personal paths or host names.
- Plain ASCII in code, docs and logs. One topic per pull request, with a short plain commit message.
- Do not commit model weights, GGUF files or datasets.

## Licence

- By contributing you agree that your contribution is licensed under the Apache License 2.0 (`LICENSE`, Section 5).
  Keep the `NOTICE` file.
- The patches modify llama.cpp, which is MIT-licensed (`third_party/llama.cpp-LICENSE`).
- Models follow their base-model licences: the Qwen3-4B file is Apache-2.0; the Llama-3.1-8B file, and any file you build
  from a Llama 3.1 model, falls under the Llama 3.1 Community License and its Acceptable Use Policy ("Built with Llama").
- "Pingala-8" and "Genrius AI" are names of Genrius AI; see `TRADEMARKS.md` before naming a derived work.
