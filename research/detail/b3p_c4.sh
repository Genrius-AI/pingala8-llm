#!/bin/bash
# b3p_c4.sh (2026-10-18): held-out C4 (c4_val_part, 73 windows, same command line as logs/r59/c4_b2.log) for B3+ (VR3 file), only if its
# GPU chain (r59_b3p.sh gpu) finished without a rollback; paired vs B3 (c4_b2.log), IQ2_KT at B3's bytes (c4_ktm.log), IQ2_XXS. Plain ASCII.
cd ~/repo || exit 1; source bin/cuda_env.sh; L=logs/r59_b3p.log; D=research/detail; FB=research/kernels/llama_f44/build/bin
say() { echo "$(date '+%F %T') $*" | tee -a $L; }
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
grep -aq 'gpu done' $L && ! grep -aq 'WRAP rollback' $L || { say "C4 skipped: B3+ gpu chain not done or rolled back"; exit 0; }
M=$(ls work/b3p/native/4B-b3p-vr3-00001-of-*.gguf); CL=logs/r59/c4_b3p.log
grep -aqE '\[73\][0-9]' $CL 2>/dev/null || bin/job env $NAT $FB/llama-perplexity -f datasets/c4_val_part.raw -c 2049 --ppl-stride 4095 -ngl 99 -m $M > $CL 2>&1
say "C4 B3+ $(grep -a -o '\[73\][0-9.]*' $CL | tail -1) (B3 26.7548, IQ2_KT-b3bytes 28.0002)"
for c in "logs/r59/c4_b2.log B3" "logs/r59/c4_ktm.log IQ2_KT-b3bytes" "logs/r59/c4_xxs.log IQ2_XXS"; do set -- $c
  say "PAIRED C4 B3+ vs $2: $(python3 $D/r59_paired.py $CL $1 2>&1 | tail -1) | $(python3 $D/r59_acorr.py $CL $1 2>&1 | tail -1)"; done
