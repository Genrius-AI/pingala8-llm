#!/bin/bash
# r36: LLVQ-protocol all-token ppl on the HF-joined test text. usage: r36_hfj.sh LABEL GGUF
source ~/repo/bin/cuda_env.sh
cd ~/repo; B=${HFJ_BIN:-llama.cpp/build-cuda/bin}; H=datasets/wikitext2_test_hfjoin.raw
$B/llama-perplexity -f $H -c 2049 --ppl-stride 4095 -ngl 99 -m $2 > logs/r36_hfj_$1.log 2>&1
echo "HFJ $1 rc=$? last $(grep -a -o '\[[0-9]*\][0-9.]*' logs/r36_hfj_$1.log | tail -1) $(grep -a -o 'computing over [0-9]* chunks, n_ctx=[0-9]*' logs/r36_hfj_$1.log) $(grep -a -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' logs/r36_hfj_$1.log)"
