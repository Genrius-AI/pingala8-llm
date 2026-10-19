#!/bin/bash
# b3p_gpu_wrap.sh (2026-10-18 02:10): run the B3+ GPU chain (r59_b3p.sh gpu) with a rollback. The chain relinks the fork's libllama.so
# (patched loader, VR3 support). If the make or the layout/back-compat check fails, restore the shipped libllama.so and the loader sources
# so every later fork run (excl_bench, l8 tables) uses the binary the paper's numbers came from. Plain ASCII.
cd ~/repo || exit 1; B=research/kernels/llama_f44/build/bin; S=research/kernels/llama_f44/src/src; W=work/b3p
say() { echo "$(date '+%F %T') $*" | tee -a logs/r59_b3p.log; }
cp -n $B/libllama.so.0.5.0 $W/libllama.so.0.5.0.pre_b3p; n0=$(grep -ac 'BLOCKER' logs/r59_b3p.log)
bin/job bash research/detail/r59_b3p.sh gpu >> logs/r59_b3p.out 2>&1; rc=$?
if [ $(grep -ac 'BLOCKER' logs/r59_b3p.log) -gt $n0 ] && tail -n +1 logs/r59_b3p.log | tail -5 | grep -aqE 'BLOCKER (make|layout)'; then
  cp $W/libllama.so.0.5.0.pre_b3p $B/libllama.so.0.5.0
  cp $W/llama-model-loader.cpp.pre_b3p $S/llama-model-loader.cpp; cp $W/llama-model-loader.h.pre_b3p $S/llama-model-loader.h
  say "WRAP rollback: shipped libllama.so and loader sources restored (rc=$rc)"
else say "WRAP gpu chain rc=$rc (no rollback)"; fi
