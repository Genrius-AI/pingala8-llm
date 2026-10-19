#!/bin/bash
# B3 paired speed bench (2026-10-17 17:10, copy of excl_bench_1017.sh with a b3 arm; GPU shared with the Llama build in slot 2, so read only the ratios).
# Same protocol as r57_post bench_stage / r56_kteq: llama-bench -ngl 99 -fa 1 -ctk f16 -ctv f16, pp512 and tg128, rotated arm order over 3 rounds, -r 5.
cd ~/repo; source bin/cuda_env.sh; O=logs/b3_bench; mkdir -p $O
FB=research/kernels/llama_f44/build/bin; IK=ext/ik_llama.cpp/build-cuda/bin
B3=work/r59b3/native/4B-b3-vrnative-00001-of-00002.gguf; MB=work/r51t/native/4B-pingala8-vrnative-00001-of-00002.gguf; MX=work/r45b/4B-IQ2_XXS-r45b.gguf; KT=work/r59/qwen3-4B-IQ2_KT-eq-ik.gguf
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
AR=(b3 mB mnXX ikKTeq); [ -f $KT ] || AR=(b3 mB mnXX)
run() { local a=$1 t=$2 p=$3 bin m e args; case $a in b3) bin=$FB; m=$B3; e="env $NAT";; mB) bin=$FB; m=$MB; e="env $NAT";; mnXX) bin=$FB; m=$MX; e="env GGML_F44=0";; ikKTeq) bin=$IK; m=$KT; e="env";; esac
  [ $t = pp512 ] && args="-p 512 -n 0" || args="-p 0 -n 128"
  echo "PRE $a $t r$p $(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader,nounits | head -1)" >> $O/pre.log
  $e $bin/llama-bench -m $m -ngl 99 -fa 1 -ctk f16 -ctv f16 $args -r 5 -o json > $O/${a}_${t}_r$p.json 2> $O/${a}_${t}_r$p.err; echo "BENCH $a $t r$p rc=$?"; }
for p in 1 2 3; do n=${#AR[@]}; for ((k = 0; k < n; k++)); do a=${AR[$(( (k + p - 1) % n ))]}; for t in pp512 tg128; do run $a $t $p; done; done; done
python3 - <<'PY'
import json, glob, statistics as st
R = {}
for f in glob.glob("logs/b3_bench/*_r?.json"):
    a, t, r = f.split("/")[-1][:-5].rsplit("_", 2); 
    try: R[(a, t, r)] = json.load(open(f))[0]["avg_ts"]
    except Exception as e: print("BAD", f, e)
for t in ("pp512", "tg128"):
    for a in ("b3", "mB", "ikKTeq"):
        q = [R[(a, t, r)] / R[("mnXX", t, r)] for r in ("r1", "r2", "r3") if (a, t, r) in R and ("mnXX", t, r) in R]
        if q: print(f"EXCL {t} {a}/IQ2_XXS median {st.median(q):.3f} rounds {[round(x, 3) for x in q]} tok/s {a} {[round(R[(a, t, r)], 1) for r in ('r1','r2','r3') if (a, t, r) in R]} xxs {[round(R[('mnXX', t, r)], 1) for r in ('r1','r2','r3') if ('mnXX', t, r) in R]}")
PY
