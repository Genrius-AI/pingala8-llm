#!/bin/bash
# r36 zero-shot with llama-perplexity built-ins (standard llama.cpp data files in datasets/zs). usage: r36_zs.sh LABEL GGUF [HS_TASKS]
cd ~/repo; B=llama.cpp/build-cuda/bin; Z=datasets/zs; L=$1; M=$2; N=${3:-2000}; O="-ngl 99 -c 2048 -b 2048"
$B/llama-perplexity -m $M $O --hellaswag -f $Z/hellaswag_val_full.txt --hellaswag-tasks $N > logs/r36_zs_${L}_hs.log 2>&1
$B/llama-perplexity -m $M $O --winogrande -f $Z/winogrande-debiased-eval.csv > logs/r36_zs_${L}_wg.log 2>&1
for t in arc-challenge arc-easy mmlu; do $B/llama-perplexity -m $M $O --multiple-choice -f $Z/$t-validation.bin > logs/r36_zs_${L}_$t.log 2>&1; done
echo "ZS $L HS $(grep -a -P '^\d+\t[0-9.]+%' logs/r36_zs_${L}_hs.log | tail -1 | tr '\t' ' ') | WG $(grep -a -o 'Final Winogrande score.*' logs/r36_zs_${L}_wg.log) | ARC-C $(grep -a -o 'Final result: .*' logs/r36_zs_${L}_arc-challenge.log) | ARC-E $(grep -a -o 'Final result: .*' logs/r36_zs_${L}_arc-easy.log) | MMLU $(grep -a -o 'Final result: .*' logs/r36_zs_${L}_mmlu.log)"
