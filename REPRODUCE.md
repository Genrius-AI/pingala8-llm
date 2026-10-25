# Reproduction guide (supplementary code and logs)

This is the README of the paper's supplementary code and logs archive. It was written for the B3 reference build and maps the
numbers of that build to file:line. The two released B4 files (the build the paper's body reports) are covered in `B4.md`.
"This archive" below means this repository.

This archive holds the runtime patch, the encoder and evaluation scripts, and the logs behind the numbers in the paper.
Every repository path printed in the paper (for example `research/detail/r48k.py`, `logs/r48k3.tensors`) is a path
relative to the root of this archive. No model weights, GGUF files or datasets are included; the only binaries are the small
codebook tables the encoder and one benchmark arm read (`.pt`, `.bin`, 12 KB or less each).

## Folders

| path | contents |
|---|---|
| `patches/llama_cuda_vs_7ac59a6.patch` | CUDA runtime: unified diff (13 files) against llama.cpp commit `7ac59a6e3ad851cd41af00f678effab0598ba9a8`. This is the GPU path behind every reported number. |
| `patches/llama_cpu_vs_7ac59a6.patch` | Standalone alternative: the x86 AVX2 CPU backend plus the CUDA path (19 files) against the same commit. Apply one patch or the other, not both. |
| `research/detail/` | Encoder and builder (`r48_a1gpu.py`, `r48k*.py`, `r52a_k.py`, `r52a_build.py`, `r52a_full.sh`, `r57_flat_drive.sh`, `r51_transcode.py`, `r51_delta.py`, `r51_native.py`, `r59_b3.sh`, `b3p_native.py`, `r59_b3p.sh`, `r35_hcal.py`), evaluation (`r36_hfj.sh`, `r59_paired.py`, `r59_acorr.py`, `r59_kldc4*.{sh,py}`, `r56_kteq.sh`, `r54kt_gpu.sh`, `r59_ktm*.{sh,py}`), speed (`excl_bench_1017.sh`, `b3_bench.sh`), zero-shot (`pe52_zs.sh`, `pe52_gpu*.sh`, `pe52_mmlu_bin.py`, `r36_zs*.sh`, `r59_zs*.sh`), Gaussian-source study (`r59_sourcecoding.py`), appendix calculators (`ab_scaling_calc.py`, `ab_gguf_check.py`). |
| `research/kernels/` | `llama_f44/r51_nat.sh` and `r51_real_sum.py` (the quiet-run speed and energy benchmark), `joint_table_i8.bin` (516-byte int8 table read by a timing-only arm of the CUDA patch). |
| `research/work/` | Codebook tables read by the encoder (`r47/`, `r48p/`, `r48k3/tables_vrc.*`), the per-tensor byte budget (`r36_e1.vedq.stats.json`), the B2 trim plan (`r57/flattrim_cls.json`), and the B3 pre-registrations (`r59/prereg_b3.txt`, `r59/prereg_b3_heldout.txt`, written before the runs). |
| `logs/` | The logs the paper cites, at their original relative paths. |
| `logs_bak_1014/` | One log whose live copy was overwritten by a failed rerun; this is the backup taken on the day (Q2_K perplexity). |
| `derived/` | `recompute.sh` recomputes every paired per-window statistic and the 72-chunk KL divergence from the shipped logs (CPU, seconds); `recomputed.txt` is its output on this archive. |
| `tools/` | `make_hfjoin.py` (rebuilds the perplexity text with a sha256 check), `assemble_b3.py` (assembles the B3 blob set from the two flat builds; written for this archive and checked to reproduce the original B3 inputs exactly). |
| `bin/cuda_env.sh` | WSL2 workaround: preloads the PTX JIT library that matches the Windows driver. Not needed on native Linux. |

## Edits relative to the original files

The archive was first prepared for double-blind review; this public release keeps those edits so that the shipped files match the
reviewed archive. Edits made for this release: the B4 scripts added to `research/detail/` (`cs_*`, `r51g_native.py`, `l8_*`,
`r48l_*`, `r59_*70.py`) had absolute paths made relative to the repository root (Python literals such as `llama.cpp/gguf-py`;
shell scripts `cd` to the root themselves; one WSL2 disk path became the variable `VHDX`); `research/detail/r59_kldc4_report.py`
writes its report to `eval_r59_kldc4.txt` by default; in the added B4 logs 33 non-ASCII characters were mapped as below.
Edits of the review copy:

- `/home/<user>/` became `~/`, the repository directory became `repo` (so `~/repo/...`), and user, host and
  organisation names were replaced by `user`, `host` and `anon-org`. Python scripts therefore contain literals such as
  `"~/repo/llama.cpp/gguf-py"`; Python does not expand `~`, so run `sed -i "s#~/repo#$HOME/repo#g"` on the scripts or edit the constants.
- Non-ASCII characters were removed from 44 log files (4,518 characters): the plus-minus sign (U+00B1) became `+/-`, Greek capital delta (U+0394) became `D`, progress-bar
  block glyphs became `#`, and 218 tokenizer glyphs (U+0120, U+010A) in model-loading dumps were dropped. No newline was added or
  removed, so every line number below is the line number of the original log. The scripts and patches were already ASCII.
- Otherwise every log is byte-identical to the original. No log was over 2 MB, so none was trimmed.
- In the CUDA patch one default path string (`$HOME/repo/research/kernels/joint_table_i8.bin`, overridable with `GGML_MB_CW`) was anonymized.
  Applied to a clean `7ac59a6` tree, the patch reproduces the source tree used for the runs except for that string (checked file by file).
- The scripts call `bin/job` / `bin/gjob2`, local memory and GPU guards that only wrap a command; they are not shipped. Drop the prefix.

## Build the runtime

The appendix gives only the commit and the GPU. The CMake flags below are the non-default entries of the CMake cache of the build
directory that produced the numbers (the configure command line itself was not logged): CUDA 13.2.51, GCC 15.2, sm_120 (RTX 5070 Ti Laptop GPU, 12 GB).

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 7ac59a6e3ad851cd41af00f678effab0598ba9a8
git apply --check ../pingala8-llm/patches/llama_cuda_vs_7ac59a6.patch && git apply ../pingala8-llm/patches/llama_cuda_vs_7ac59a6.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF
cmake --build build -j8 --target llama-perplexity llama-bench
```

Resulting cache: shared libraries, `GGML_CUDA_GRAPHS`, `GGML_CUDA_FA`, `GGML_NATIVE` and `GGML_OPENMP` on (defaults).
Runtime environment for the native files (Model B, B3), as set in `research/detail/r59_b3.sh`; offload all layers (`-ngl 99`):

```
export GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2
```

The loader in the patch also reads a later file layout (`vr.layout = 3`, `b3p_native.py`); that branch is taken only for files that carry
this key. The Model B and B3 files do not, so they load through the same code path as in the runs.

## Rerun one headline number: B3 WikiText-2 perplexity 18.0660

Inputs not shipped: Qwen3-4B from the Hugging Face hub (`Qwen/Qwen3-4B`), the C4 train shard 00000, the WikiText-2 raw test split.
All commands run from the repository root (`~/repo`), with `~/repo/llama.cpp` the patched tree from above (its `gguf-py` is imported by the scripts).

1. Reference files with stock llama.cpp tools: the fp16 GGUF (`convert_hf_to_gguf.py`, `models/qwen3-4B-f16.gguf`), the importance matrix
   (`llama-imatrix` on C4 calibration text, context 512, 200 chunks, `models/qwen3-4B.imatrix`) and the IQ2_XXS mix
   (stock `llama-quantize --imatrix ... IQ2_XXS`, read by the scripts as `work/r45b/4B-IQ2_XXS-r45b.gguf`, 1,246,621,312 bytes), which supplies the KV metadata and the non-linear tensors.
2. Hessian calibration text: `python research/detail/r35_hcal.py` (C4 shard 00000 from document 200000, 6.0M characters, 10-gram overlap check).
3. The two flat encoder runs (GPU; each writes per-tensor blobs, an 8-bit carrier and `stats.json`):
   `research/detail/r52a_full.sh f1 roff:8:research/work/r48k3:rateoff` (one class per tensor; log `logs/r52a_f1.tensors`) and
   `research/detail/r52a_full.sh f4 flat:8:research/work/r48k3:flattrim` (as in `r57_flat_drive.sh`; log `logs/r52a_f4.tensors`).
   `r52a_full.sh` line 11 holds every encoder setting (NSEQ 512, NEV 32, DAMP 0.1, ALPHA 1.05, NSAL 4, shifts). It copies a Qwen3-4B 8-bit
   carrier (`work/r50ev/4B-r48k-vrc8_all.gguf`) as a template whose tensors are all overwritten.
4. B3 blob set: `python3 tools/assemble_b3.py work/r52a/f1 work/r52a/f4 work/r59b3` (250 tensors from f1, `blk.4/5.ffn_gate` from f4, per the pre-registration amendment).
5. Native file: `bash research/detail/r59_b3.sh` runs `r51_delta.py` (clamp delta) and `r51_native.py plan | dryrun | build | verify`
   (expected `NATIVE TOTAL 1245915872 B`, `logs/r59_b3.log:13`, two shards), and then steps 7 and 8 itself (set `FB` in the script to your `build/bin`).
6. Perplexity text: `python tools/make_hfjoin.py datasets/wikitext2_test_hfjoin.raw` (expect 1,296,370 bytes, sha256 `696cca6b...`).
7. Perplexity (the environment above):
   `HFJ_BIN=llama.cpp/build/bin bash research/detail/r36_hfj.sh r59_b3_native work/r59b3/native/4B-b3-vrnative-00001-of-00002.gguf`
   runs `llama-perplexity -c 2049 --ppl-stride 4095` (73 windows of 4096) and writes `logs/r36_hfj_r59_b3_native.log`; the last value is `[73]18.0660` (line 15).
8. Paired test: `python3 research/detail/r59_paired.py logs/r36_hfj_r59_b3_native.log logs/r36_hfj_iq2xxs.log` gives `dNLL -0.4597 +/- 0.0197, 73/73`.

Without a GPU, `bash derived/recompute.sh` reproduces all paired statistics in the paper from the shipped logs.

## Where each number is

Ratios to fp16 are CALC: perplexity / 12.4567. Line numbers refer to the files in this archive.

### Abstract

| number | file:line |
|---|---|
| IQ2_XXS 28.61, 2.297x fp16 (fp16 12.4567) | `logs/r36_hfj_iq2xxs.log:10` (`[73]28.6083`); `logs/r36_hfj_fp16.log:10` |
| B3 1,245,915,872 B; IQ2_XXS 1,246,621,312 B | `logs/r59_b3.log:13` |
| B3 perplexity 18.07 (1.450x) | `logs/r36_hfj_r59_b3_native.log:15` (`[73]18.0660`) |
| IQ2_KT at B3's bytes 18.65 (1,245,914,752 B) | `logs/r36_hfj_r59_ktm.log:212`; bytes `logs/r59_ktm.log:2` |
| paired -0.032 +/- 0.011 | `logs/r59_ktm.log:6`; `derived/recomputed.txt:4` |
| +/-0.018 after autocorrelation correction | `derived/recomputed.txt:10` (0.0175) |
| held-out C4: B3 26.75, IQ2_KT (B3 bytes) 28.00, paired -0.046 +/- 0.010 (66/73) | `logs/r59/c4_b2.log:15` (b2 label = B3); `logs/r59/c4_ktm.log:212`; `logs/r59_ktm.log:8` |
| C4: IQ2_XXS 37.28; Q8_0 19.49 | `logs/r59_kldc4.log:106` (summary line only, see Gaps); `logs/r59/c4_q8.log:10` |
| first flat build B2 tied IQ2_KT | `logs/r36_hfj_r52a_flat.log:10` (18.5738); `derived/recomputed.txt:6` (-0.0023 +/- 0.0109) |
| Model B KLD 0.640; IQ2_XXS 1.220; IQ2_KT (Model B bytes) 0.785 | `logs/pe52/gpuA.log:1`, `:2`; `logs/r56_kteq_kld.log:239`; same 12 chunks also in `logs/r59/kld_mB_p1.log:36`, `kld_xxs_p1.log:31`, `kld_kt_p1.log:239` |
| token generation 0.97-1.09x, prompts 0.70-0.75x | Table 2 rows below |
| LLVQ checkpoint better (17.07) | `logs/bl52_hfppl_llvq.log:11` |
| 1.2-1.3 dB above IQ2_XXS; about 1.8 dB from the bound | `logs/r59_sourcecoding.log:105` (IQ2_XXS 9.280 dB), `:75-77` (S=4/8/16: 10.506/10.536/10.548 dB, gap 1.911/1.882/1.869), `:8` (bound 12.417 dB) |
| per-tile 18.23 vs one class per tensor 17.79 | `logs/r36_hfj_r52a_mirror.log:10`; `logs/r36_hfj_r52a_roff.log:10`; paired `derived/recomputed.txt:7` |
| 16 states beat 8 by 1.7% | `logs/r36_hfj_r50ev_vrc_all.log:10` (18.7848) vs `logs/r36_hfj_r48k3_vrc8_all_b.log:10` (19.1036), CALC |

### Table 1 (main result)

| row | bytes | PPL | dNLL, B3 - row | KLD 12 ch. | KLD 72 ch. | C4 |
|---|---|---|---|---|---|---|
| fp16 | `logs/build_ggufs.log:5356` | `logs/r36_hfj_fp16.log:10` | | | | |
| IQ2_XXS | `logs/r59_b3.log:13` | `logs/r36_hfj_iq2xxs.log:10` | `logs/r59_b3.log:17` | `logs/pe52/gpuA.log:2` | `derived/recomputed.txt:23` | `logs/r59_kldc4.log:106` |
| IQ2_KT, Model B's bytes | `logs/r56_kteq_sha256.txt:1`, `logs/r56_kteq_drive.log:4` | `logs/r56_kteq_ppl.log:212` | `logs/r59_b3.log:18` | `logs/r56_kteq_kld.log:239` | `derived/recomputed.txt:21` | `logs/r59/c4_kt.log:212` |
| IQ2_KT, B3's bytes | `logs/r59_ktm.log:2` | `logs/r36_hfj_r59_ktm.log:212` | `logs/r59_ktm.log:6` | | | `logs/r59/c4_ktm.log:212` |
| Model B | `logs/r51_native.log:14` | `logs/r36_hfj_r51_native_PPC2.log:15` | `logs/r59_b3.log:19` | `logs/pe52/gpuA.log:1` | `derived/recomputed.txt:22` | `logs/r59/c4_mB.log:15` |
| B3 | `logs/r59_b3.log:13` | `logs/r36_hfj_r59_b3_native.log:15` | 0 | `logs/r59/kld_b2_p1.log:36` | `derived/recomputed.txt:20` | `logs/r59/c4_b2.log:15` |
| IQ2_KT, default recipe | `logs/r54kt_sha256.txt:1` | `logs/r54kt_ppl_IQ2_KT.log:212` | | `logs/r54kt_kld_IQ2_KT.log:239` | | |
| IQ2_XS | `logs/pe52/quant_drive.log:2` | `logs/r36_hfj_pe52_IQ2_XS.log:10` | | `logs/pe52/gpuA.log:6` | | |
| Q2_K | `logs/build_ggufs.log:5353` | `logs_bak_1014/r36_hfj_pe52_Q2_K.log:10`; `logs/pe52/gpuB.log:3` | | `logs/pe52/gpuA.log:5` | | |

Section 5 text: C4 paired B3 vs IQ2_KT at Model B's bytes -0.036 +/- 0.010 (62/73) and vs Model B -0.019 (53/73): `derived/recomputed.txt:12`, `:14`;
the corrected intervals 0.0165 / 0.0175 and bootstrap intervals: `derived/recomputed.txt:9-10`; ik_llama.cpp on the IQ2_XXS file 28.6037: `logs/r59_ktm.log:4`.

### Table 2 (speed)

| run | tg128 | pp512 |
|---|---|---|
| Quiet, IQ2_XXS 203.1 / 7528.8 tok/s | `logs/r51_ppc2_sum.txt:3` | `logs/r51_ppc2_sum.txt:1` |
| Quiet, Model B 1.003 (201.7) / 0.702 (5777.8) | `logs/r51_ppc2_sum.txt:7` | `logs/r51_ppc2_sum.txt:5` |
| Exclusive, Model B 0.967 / 0.748 | `logs/excl_bench.log:21` | `logs/excl_bench.log:19` |
| Exclusive, IQ2_KT 1.001 / 0.948 | `logs/excl_bench.log:22` | `logs/excl_bench.log:20` |
| Shared, B3 1.164 / 0.653 and Model B 1.087 / 0.680 | `logs/b3_bench.log:27` | `logs/b3_bench.log:26` |

The per-round data of the quiet run are the `PWR` lines of `logs/r51_ppc2.log`; the shared-run summary in `logs/b3_bench.log` was recomputed from
`logs/excl_bench/*.json`, as its own header line says.

### Table 3 (zero-shot)

| row | HellaSwag | WinoGrande | ARC-Easy | ARC-Challenge | MMLU (14,042) |
|---|---|---|---|---|---|
| Q8_0 | `logs/pe52/gpuB.log:31` | `gpuB.log:32` | `gpuB.log:33` | `gpuB.log:34` | `gpuB.log:36` |
| Model B | `logs/pe52/zs_modelB_native_hs.log:1015` | `zs_modelB_native_wg.log:1281` | `zs_modelB_native_arc-easy.log:587` | `logs/pe52/gpuA.log:11` | `gpuA.log:13` |
| B3 | `logs/pe52/zs_b3_hs.log:1013` (998 items) | `zs_b3_wg.log:1281` | `zs_b3_arc-easy.log:587` | `zs_b3_arc-challenge.log:316` | `zs_b3_mmluF.log:14059` |
| IQ2_KT, B3's bytes | `logs/pe52/zs_ktm_hs.log:1212` | `zs_ktm_wg.log:1485` | `zs_ktm_arc-easy.log:791` | `zs_ktm_arc-challenge.log:520` | fail: `zs_ktm_mmluF.log:215` |
| IQ2_XXS | `logs/r36_zs_iq2xxs_hs.log:1010` | `r36_zs_iq2xxs_wg.log:1276` | `r36_zs_iq2xxs_arc-easy.log:582` | `r36_zs_iq2xxs_arc-challenge.log:311` | `logs/pe52/gpuB.log:28` |
| IQ2_XS | `logs/pe52/gpuB.log:39` | `zs_IQ2_XS_wg.log:1276` | `zs_IQ2_XS_arc-easy.log:582` | `zs_IQ2_XS_arc-challenge.log:311` | `zs_IQ2_XS_mmluF.log:14054` |

The s.e. row (Model B) is the `+/-` on the same lines; for HellaSwag it is half the printed 95% interval.

### Paths printed in the appendices

`research/detail/ab_scaling_calc.py` with `logs/ab_scaling_calc.log`; `research/detail/ab_gguf_check.py` with `logs/ab_gguf_check.log`;
`logs/r48k3.tensors` (build times, class histograms); `research/detail/r51_native.py` with `logs/r51_native.log`; `research/detail/r48k.py`;
`f44.cu` = `ggml/src/ggml-cuda/f44.cu` in the CUDA patch; `logs/r51_ppc2.log`, `logs/r51_d1_q8ceiling.log`; `logs/l8_build.log` and the other
`logs/l8_*` files (a Llama-3.1-8B build that was in progress; no number in the paper comes from it); `logs/r36_hfj_g3_native_PPC2.log`.

## Gaps (stated, not filled)

- `logs/r59/c4_xxs.log` was overwritten by an interrupted rerun and stops at window 33. The 73-window C4 perplexity of IQ2_XXS (37.28) survives only as the
  summary line `logs/r59_kldc4.log:106`, and the paired C4 difference B3 vs IQ2_XXS (-0.332) cannot be recomputed from this archive.
- The per-run 12-chunk KL divergence logs of Model B, IQ2_XXS, IQ2_XS and Q2_K were truncated by a failed rerun; the values come from the summary lines of
  the run driver (`logs/pe52/gpuA.log`). For Model B, IQ2_XXS and IQ2_KT the same 12 chunks were run again as part 1 of the 72-chunk set and give the same values.
- Per-run logs for Q8_0 zero-shot, IQ2_XS HellaSwag, Model B ARC-Challenge and MMLU, and IQ2_XXS MMLU did not survive (the files present are failed retries);
  the values are the driver summary lines in `logs/pe52/gpuA.log` and `gpuB.log`.
- B3 HellaSwag covers 998 of 1000 items; IQ2_KT could not read the MMLU task file. Both are stated in the paper.
