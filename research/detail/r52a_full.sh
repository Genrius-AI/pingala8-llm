#!/bin/bash
# R52a full-model ablation build: TAG then arm specs name:NS:TABDIR:mode (r52a_build.py). One Q8_0 carrier per arm (copy of the
# Pingala-8 snapshot as template, every linear and non-linear overwritten), LLVQ ppl (r36_hfj.sh), carrier deleted after ppl.
cd ~/repo; export HF_HUB_OFFLINE=1; source bin/cuda_env.sh
if [ -n "${GJ2:-}" ]; then J="env VRAM_MB=5000 RAM_MB=5500 bin/gjob2 bash -c"; PRE="source ~/repo/.venv/bin/activate; export HF_HOME=~/repo/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;"; else J="bin/job bash -c"; PRE=""; fi
 TAG=$1; shift; W=work/r52a/$TAG; mkdir -p $W; TPL=work/r50ev/4B-r48k-vrc8_all.gguf
need=$(( 12 + 6 + 4 * $# ))
until [ $(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9) -ge $need ]; do echo "$(date +%T) waiting for C: >= $need GB"; sleep 120; done
A=""; for s in "$@"; do n=${s%%:*}; cp $TPL $W/4B-$n.gguf; A="$A,$s:$W/4B-$n.gguf"; done; A=${A#,}
date +"F0 %T $TAG $A"
$J "$PRE GPUCHOL_MAXN=10000 GPUFRAC=0.40 R48K_SHIFTS=0,0,0,0,2,3,2,4/0,0,0,0,0,0,0,2 ARMS=$A NSEQ=512 NEV=32 DAMP=0.1 ALPHA=1.05 NSAL=4 AVB=2.0625 python research/detail/r52a_build.py work/r45b/4B-IQ2_XXS-r45b.gguf $W > logs/r52a_$TAG.tensors 2>&1"
date +"F1 %T build rc=$?"; grep -E '^SUMMARY|ABORT|Error' logs/r52a_$TAG.tensors | tail -6
grep -q '^GATES' logs/r52a_$TAG.tensors || exit 5
for s in "$@"; do n=${s%%:*}; C=$W/4B-$n.gguf
  echo "PPL_MODEL -m $C sha256 $(sha256sum $C | cut -c1-64) variant $n" | tee -a logs/r52a_ppl.log
  if [ -n "$GJ2" ]; then VRAM_MB=4500 RAM_MB=2500 bin/gjob2 bash research/detail/r36_hfj.sh r52a_$n $C; else bin/job bash research/detail/r36_hfj.sh r52a_$n $C; fi | tee -a logs/r52a_ppl.log
  grep -q "rc=0" <(tail -n 1 logs/r52a_ppl.log) && rm -f $C; date +"F2 %T $n ppl done"
done
