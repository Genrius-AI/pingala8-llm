#!/bin/bash
# R51 real-format bench (coordinator 12:3x-12:4x): Pingala-8 B loader with the format's side steps, paired vs IQ2_XXS.
#   iq2    stock IQ2_XXS
#   B      weight-stationary NJ=2, class 4-7 int8-rounded (R51 7-round arm)
#   BR     B + real: x' = FWHT256(sgn * x / c)/16 per matmul (batch: unfused pass; GEMV: fused into the q8_128 quantize) + prm un-permute in the epilogue
#   BHL    B with exact hi/lo int8 planes for class 4-7 (IQ1_S slot), no real steps
#   BHLR   BHL + real
#   BHLRD  BHLR + clamp delta (sparse y += Delta x; GEMV fused into k_vr, batch one launch per matmul)
# Checks (graphs off): R51CHECK for BHL, BR, BHLR. Bench: 3 rotated rounds, pp512 and tg128 at d 0 and 4096 as separate llama-bench runs,
# nvidia-smi every 100 ms (power, sm/mem clock, temp, memory) -> logs/r51_power/<arm>_<test>_d<d>_r<k>.csv; J/token = mean in-run W / tok/s.
set -u
cd ~/repo
B=research/kernels/llama_f44/build/bin; M=work/r45b/4B-IQ2_XXS-r45b.gguf; L=${LOGD:-logs/r51_power}; mkdir -p $L
export LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nvlt.inf_amd64_138f16e1d7ecfcb8:${LD_LIBRARY_PATH:-}
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1"; N=work/r51t/native/4B-pingala8-vrnative-00001-of-00002.gguf
envof() { case $1 in iq2) echo "GGML_F44=0" ;; NHL) echo "$NAT" ;; NHLP) echo "$NAT GGML_VR_HLP=1" ;; NPPC) echo "$NAT GGML_VR_PPC=1" ;; NPPC2) echo "$NAT GGML_VR_PPC=2" ;; esac; }
echo "START $(date +%T) binary $(stat -c %y $B/llama-bench)"
for a in ${RCHECK-BHL BR BHLR BHLRD}; do
  env $(envof $a) GGML_F44_CHECK=1 GGML_CUDA_DISABLE_GRAPHS=1 $B/llama-bench -m $N -ngl 99 -fa 1 -p 512 -n 4 -r 1 -o jsonl > $L/check_$a.jsonl 2> $L/check_$a.err
  rc=$?; echo "CHECK_RC $a $rc"; [ $rc -ne 0 ] && tail -3 $L/check_$a.err
  grep -E "batch path|VR delta" $L/check_$a.err | sort -u | sed "s/^/$a /"
  grep R51CHECK $L/check_$a.err | awk -v a=$a '{for(i=1;i<=NF;i++){if($i=="not_bit_equal")ne=$(i+1); if($i=="rel_rms")rr=$(i+1); if($i=="nonfinite")nf=$(i+1); if($i=="wq_rel_rms")wq=$(i+1); if($i=="maxabsdiff"&&!m)m=$(i+1)}; print "R51CHECK_SUM", a, $2, "neq_vs_R49b", ne, "maxabsdiff", m, "cpu_rel_rms", rr, "wq_rel_rms", wq, "nonfinite", nf; m=""}'
done
[ -n "${CHECKONLY:-}" ] && exit 0
modelof() { case $1 in iq2) echo $M ;; *) echo $N ;; esac; }
run() {  # arm test depth round
  local a=$1 t=$2 d=$3 p=$4 args
  [ $t = pp512 ] && args="-p 512 -n 0" || args="-p 0 -n 128"
  local f=$L/${a}_${t}_d${d}_r$p.csv
  ( while true; do echo "$(date +%s.%N),$(nvidia-smi --query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu,memory.used --format=csv,noheader,nounits | head -1)"; sleep 0.1; done ) > $f 2>/dev/null & local smp=$!
  env $(envof $a) $B/llama-bench -m $(modelof $a) -ngl 99 -fa 1 -ctk f16 -ctv f16 $args -d $d -r 5 -o jsonl > $L/${a}_${t}_d${d}_r$p.jsonl 2> $L/${a}_${t}_d${d}_r$p.err
  local rc=$?; kill $smp 2>/dev/null; wait $smp 2>/dev/null
  python3 - $f $L/${a}_${t}_d${d}_r$p.jsonl $a $t $d $p $rc <<'PY'
import sys, json, statistics
f, j, a, t, d, p, rc = sys.argv[1:8]
S = []
for ln in open(f):
    x = ln.strip().split(",")
    if len(x) == 6:
        try: S.append([float(v) for v in x])
        except ValueError: pass
act = [s for s in S if s[2] > 1000]
tps = statistics.median(json.loads(open(j).readline())["samples_ts"]) if rc == "0" else 0
W = statistics.mean(s[1] for s in act) if act else 0
med = lambda k: statistics.median(s[k] for s in act) if act else 0
print("PWR arm=%s test=%s d=%s round=%s rc=%s tps=%.2f active_samples=%d mean_W=%.2f J_per_tok=%.5f sm_MHz_med=%.0f mem_MHz_med=%.0f temp_med=%.0f peak_mem_MiB=%.0f" % (
    a, t, d, p, rc, tps, len(act), W, W / tps if tps else 0, med(2), med(3), med(4), max(s[5] for s in S) if S else 0))
PY
}
ARMS=${RARMS:-"iq2 B BR BHL BHLR BHLRD"}; read -a AR <<< "$ARMS"; n=${#AR[@]}
for p in 1 2 3; do for ((k = 0; k < n; k++)); do idx=$(( (k + p - 1) % n )); for t in pp512 tg128; do for d in 0 4096; do run "${AR[$idx]}" $t $d $p; done; done; done; done
echo "END $(date +%T)"
