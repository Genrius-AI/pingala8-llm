#!/bin/bash
# pe52 zero-shot: identical command lines to research/detail/r50ev_zs_drive.sh (r36_zs.sh / r36_zs_arc.sh), one bin/job per task,
# plus FULL MMLU (cais/mmlu test, 57 subjects, 14042 tasks, datasets/zs/mmlu-test-full.bin) with the MMLU command line.
# usage: pe52_zs.sh LABEL GGUF [TASKS]   env NATIVE=1 -> native build + native env (common rules)
cd ~/repo; source bin/cuda_env.sh; RUN=${RUN:-bin/job}; Z=datasets/zs; L=$1; M=$2; T=${3:-"hs wg arc-easy arc-challenge mmlu mmluF"}; O="-ngl 99 -c 2048 -b 2048"
if [ -n "${NATIVE:-}" ]; then B=research/kernels/llama_f44/build/bin
  E="env GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
else B=llama.cpp/build-cuda/bin; E=env; fi
o() { echo logs/pe52/zs_${L}_$1.log; }
echo "START $L $(date +%T) bin $B env '$E' -m $M"
for t in $T; do case $t in
 hs) $RUN $E $B/llama-perplexity -m $M $O --hellaswag -f $Z/hellaswag_val_full.txt --hellaswag-tasks 1000 > $(o hs) 2>&1
     echo "RES $L HS $(date +%T) $(grep -a -P '^\d+\t[0-9.]+%' $(o hs) | tail -1 | tr '\t' ' ')" ;;
 wg) $RUN $E $B/llama-perplexity -m $M $O --winogrande -f $Z/winogrande-debiased-eval.csv > $(o wg) 2>&1
     echo "RES $L WG $(date +%T) $(grep -a -o 'Final Winogrande score.*' $(o wg))" ;;
 arc-*) $RUN $E $B/llama-perplexity -m $M -ngl 99 -c 2048 -b 512 -np 5 --multiple-choice -f $Z/$t-validation.bin > $(o $t) 2>&1
     echo "RES $L $t $(date +%T) $(grep -a -o 'Final result: .*' $(o $t))" ;;
 mmlu) $RUN $E $B/llama-perplexity -m $M $O --multiple-choice -f $Z/mmlu-validation.bin > $(o mmlu) 2>&1
     echo "RES $L mmlu $(date +%T) $(grep -a -o 'Final result: .*' $(o mmlu))" ;;
 mmluF) $RUN $E $B/llama-perplexity -m $M $O --multiple-choice -f $Z/mmlu-test-full.bin > $(o mmluF) 2>&1
     echo "RES $L mmluF $(date +%T) $(grep -a -o 'Final result: .*' $(o mmluF))" ;;
esac; done
echo "DONE $L $(date +%T)"
