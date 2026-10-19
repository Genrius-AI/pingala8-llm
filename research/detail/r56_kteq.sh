#!/bin/bash
# r56_kteq: ik_llama.cpp IQ2_KT at EQUAL BYTES to Model B (1,241,608,288 B), Qwen3-4B, our harness. NO wait/gate: start it when the GPU slot is free.
# Everything (quantize, ppl, KLD, paired bench, report, cleanup) runs inside ONE bin/job (main GPU lock) hold. Safe to call either way:
#   bash research/detail/r56_kteq.sh          (re-execs itself under bin/job)      or      bin/job bash research/detail/r56_kteq.sh
# Idempotent: finished steps are skipped; the report is written only if the guard (r56_kteq_report.py --complete) says every log is complete.
# Recipe, predicted bytes and importance method: see r56_kteq_cpu.sh header and the "Equal-bytes arm (auto)" section of research/agents/baseline_r54_iq2kt.md.
cd ~/repo || exit 1
if ! grep -q 'proj.slice' /proc/self/cgroup; then exec ~/repo/bin/job bash "$HOME/repo/research/detail/r56_kteq.sh" "$@"; fi
source bin/cuda_env.sh
D=research/detail; L=logs; W=work/r54kt; IK=ext/ik_llama.cpp/build-cuda/bin; SB=llama.cpp/build-cuda/bin; FB=research/kernels/llama_f44/build/bin
H=datasets/wikitext2_test_hfjoin.raw; K=$W/q8_kld_c512x12.bin
KT=$W/qwen3-4B-IQ2_KT-eq-ik.gguf; Q8=$W/qwen3-4B-Q8_0.gguf
MX=work/r45b/4B-IQ2_XXS-r45b.gguf; MB=work/r51t/native/4B-pingala8-vrnative-00001-of-00002.gguf
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
LG=$L/r56_kteq_drive.log
say() { echo "$(date '+%F %T') $*" | tee -a $LG; }
free_gb() { df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9; }
vused() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
vgate() { for i in $(seq 1 90); do [ "$(vused)" -le 3000 ] && return 0; sleep 20; done; say "VGATE timeout, VRAM used $(vused) MiB"; return 1; }
pplok() { grep -aq '\[73\][0-9]' "$1" 2>/dev/null; }
say "r56_kteq start pid $$ C: free $(free_gb) GB"
[ "$(free_gb)" -ge 12 ] || { say "STOP: C: below 12 GB"; exit 3; }
if python3 $D/r56_kteq_report.py --done >/dev/null 2>&1; then say "report already written, nothing to do"; exit 0; fi

# ---- CPU stage (quantize) inside the GPU hold; take the CPU lock if free within 5 min (E=99 on timeout -> run unlocked, RAM is capped by the slice anyway)
flock -w 300 -E 99 ~/repo/.cpu.lock bash $D/r56_kteq_cpu.sh 2>&1 | tee -a $LG; rc=${PIPESTATUS[0]}
if [ $rc -eq 99 ]; then say "cpu lock busy 5 min, running quantize unlocked"; bash $D/r56_kteq_cpu.sh 2>&1 | tee -a $LG; rc=${PIPESTATUS[0]}; fi
say "CPU stage rc=$rc"
[ $rc -eq 0 ] || { say "BLOCKER quantize stage rc=$rc (see $L/r56_kteq_quant.log); files kept"; exit $rc; }

( while :; do echo "$(date +%T) $(vused)" >> $L/r56_kteq_vram.log; sleep 5; done ) & VS=$!; trap 'kill $VS 2>/dev/null' EXIT

gpu_stage() {
  # a. ppl, r36_hfj protocol (-c 2049 --ppl-stride 4095 => n_ctx 4096, 73 chunks), ik build
  if pplok $L/r56_kteq_ppl.log; then say "PPL already complete"; else
    vgate || return 9
    $IK/llama-perplexity -f $H -c 2049 --ppl-stride 4095 -ngl 99 -m $KT > $L/r56_kteq_ppl.log 2>&1
    say "PPL rc=$? last $(grep -a -o '\[[0-9]*\][0-9.]*' $L/r56_kteq_ppl.log | tail -1)"
  fi
  # b. KLD: base logits from the mainline Q8_0 (ctx 512 x 12 chunks, pe52 protocol), then the ik KTeq arm
  if ! grep -aq 'Mean *KLD' $L/r56_kteq_kld.log 2>/dev/null; then
    grep -aq 'Final estimate' $L/r56_kteq_kld_base_q8.log 2>/dev/null || rm -f $K
    if [ ! -f $K ]; then
      vgate || return 9
      $SB/llama-perplexity -m $Q8 -f $H -c 512 --chunks 12 -ngl 99 --kl-divergence-base $K > $L/r56_kteq_kld_base_q8.log 2>&1
      say "KLDBASE rc=$? bytes=$(stat -c %s $K 2>/dev/null) $(grep -a -o 'Final estimate: PPL = [0-9.]*' $L/r56_kteq_kld_base_q8.log)"
    fi
    s=$(stat -c %s $K 2>/dev/null || echo 0); [ "$s" -lt 1000000000 ] || { say "KLD file >= 1 GB, deleting"; rm -f $K; }
    if [ -f $K ]; then
      vgate || return 9
      $IK/llama-perplexity -m $KT --kl-divergence-base $K --kl-divergence -c 512 --chunks 12 -ngl 99 > $L/r56_kteq_kld.log 2>&1
      say "KLD rc=$? $(grep -a -E 'Mean +KLD|99.0% +KLD|Same top p' $L/r56_kteq_kld.log | tr -s ' ' | tr '\n' '|')"
    fi
  else say "KLD already complete"; fi
  rm -f $K
  # c. paired speed: arms ikKTeq (ik build), mnXX (mainline IQ2_XXS, F44 binary with GGML_F44=0), mB (Model B native, F44 binary); same run,
  #    3 rounds with rotated arm order, llama-bench -fa 1 -ctk f16 -ctv f16 -r 15, pp512 then tg128 per arm (identical to r54kt_gpu.sh).
  BL=$L/r56_kteq_bench; mkdir -p $BL
  idle() { for i in $(seq 1 20); do ok=1; for j in 1 2 3; do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); [ "${u:-100}" -lt 8 ] || ok=0; sleep 2; done; [ $ok = 1 ] && return 0; sleep 24; done; return 1; }
  AR=(ikKTeq mnXX mB); n=${#AR[@]}
  run() {  # arm test round
    local a=$1 t=$2 p=$3 bin m e args f
    case $a in ikKTeq) bin=$IK; m=$KT; e="env";; mnXX) bin=$FB; m=$MX; e="env GGML_F44=0";; mB) bin=$FB; m=$MB; e="env $NAT";; esac
    [ -f $m ] || { say "BENCH $a missing model $m"; return; }
    [ $t = pp512 ] && args="-p 512 -n 0" || args="-p 0 -n 128"
    f=$BL/${a}_${t}_r$p
    grep -aq samples_ts $f.json 2>/dev/null && { say "BENCH arm=$a test=$t round=$p already done"; return; }
    echo "PRE arm=$a test=$t round=$p $(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader,nounits | head -1)" >> $BL/pre.log
    $e $bin/llama-bench -m $m -ngl 99 -fa 1 -ctk f16 -ctv f16 $args -r 15 -o json > $f.json 2> $f.err
    say "BENCH arm=$a test=$t round=$p rc=$?"
  }
  for p in 1 2 3; do
    vgate || return 9; idle && say "ROUND $p idle" || say "ROUND $p CONTENDED (GPU util >= 8% for 10 min)"
    for ((k = 0; k < n; k++)); do idx=$(( (k + p - 1) % n )); for t in pp512 tg128; do run "${AR[$idx]}" $t $p; done; done
  done
  return 0
}

for att in 1 2; do
  gpu_stage; grc=$?
  say "GPU stage attempt $att rc=$grc"
  python3 $D/r56_kteq_report.py --complete >> $LG 2>&1 && break
  say "INCOMPLETE after attempt $att: $(python3 $D/r56_kteq_report.py --complete 2>&1 | tail -1 | cut -c1-300)"
  [ $att -lt 2 ] && sleep 120
done

# ---- report (guard refuses partial logs: no file is touched unless COMPLETE) then cleanup
if python3 $D/r56_kteq_report.py --complete >> $LG 2>&1; then
  python3 $D/r56_kteq_report.py >> $LG 2>&1; rrc=$?
  say "REPORT rc=$rrc"
  if [ $rrc -eq 0 ] && [ -s $L/r56_kteq_sha256.txt ]; then
    say "CLEANUP sha256 logged: $(cut -c1-16 $L/r56_kteq_sha256.txt)"
    rm -f $KT $Q8 $W/qwen3-4B.imatrix.dat $K
  fi
else
  say "BLOCKER report refused (partial logs); quant + Q8_0 kept in $W for a rerun; logits deleted"
  rm -f $K
fi
say "r56_kteq end; C: free $(free_gb) GB; leftover in $W: $(ls $W | tr '\n' ' ')"
