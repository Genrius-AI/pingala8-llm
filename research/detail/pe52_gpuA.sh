#!/bin/bash
# pe52 driver A (bin/job slot), relaunched 18:5x after the CUDA ptxjit fix. Remaining: native zs, KLD all, bench.
cd ~/repo; source bin/cuda_env.sh; D=research/detail; W=work/pe52; K=$W/q8_kld_c512x12.bin; SB=llama.cpp/build-cuda/bin; NB=research/kernels/llama_f44/build/bin
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
N=work/r51t/native/4B-pingala8-vrnative-00001-of-00002.gguf; X=work/r45b/4B-IQ2_XXS-r45b.gguf
kld() { local b=$SB e=env; [ -n "${3:-}" ] && { b=$NB; e="env $NAT"; }
  bin/job $e $b/llama-perplexity -m $2 --kl-divergence-base $K --kl-divergence -c 512 --chunks 12 -ngl 99 > logs/pe52/kld_$1.log 2>&1
  echo "KLD $1 rc=$? $(date +%T) $(grep -a -E 'Mean +KLD|99.0% +KLD|Same top p' logs/pe52/kld_$1.log | tr -s ' ' | tr '\n' '|')"; }
kld modelB_native $N native
kld iq2xxs $X
for q in IQ2_M IQ3_XXS Q2_K; do kld $q models/qwen3-4B-$q.gguf; done
for q in IQ2_XS IQ1_M IQ2_S; do kld $q $W/qwen3-4B-$q.gguf; done
rm -f $K; echo "KLDBASE deleted $(date +%T)"
NATIVE=1 bash $D/pe52_zs.sh modelB_native $N "arc-challenge mmlu mmluF"
bin/job bash $D/pe52_bench.sh > logs/pe52/bench.log 2>&1; echo "BENCH rc=$? $(date +%T)"
echo "GPUA_END $(date +%T)"
