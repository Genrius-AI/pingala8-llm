#!/bin/bash
# r36 ARC rerun: tasks with 5 options need --parallel >= 5 (first run aborted at task 210). usage: r36_zs_arc.sh LABEL GGUF
cd ~/repo; B=llama.cpp/build-cuda/bin; Z=datasets/zs; L=$1; M=$2
for t in arc-challenge arc-easy; do $B/llama-perplexity -m $M -ngl 99 -c 2048 -b 512 -np 5 --multiple-choice -f $Z/$t-validation.bin > logs/r36_zs_${L}_$t.log 2>&1; done
echo "ZSARC $L ARC-C $(grep -a -o 'Final result: .*' logs/r36_zs_${L}_arc-challenge.log) | ARC-E $(grep -a -o 'Final result: .*' logs/r36_zs_${L}_arc-easy.log)"
