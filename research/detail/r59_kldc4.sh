#!/bin/bash
# r59_kldc4: unattended large-sample KLD + C4 perplexity driver (review_icml_lead.md items 2 and 4). Plain ASCII.  Launch:
#   nohup setsid bash research/detail/r59_kldc4.sh > logs/r59_kldc4.out 2>&1 &
# Stages (all idempotent: finished logs are skipped on a rerun; never waits on pgrep -f, only on log markers and a recorded pid):
#   1. WAIT   until logs/r57_post.log (section after its last "r57_post start pid N" line) shows the Model B2 chain finished: "REPORT rc=0" / "report already written",
#             "R57 POST skipped", "r57_post end", or BLOCKER with the recorded pid gone; also finished if that pid has been dead 5 polls without any marker. Poll 60 s, give up after 5 h.
#   2. PREP   on CPU (CPU=1 bin/job): WikiText-2 split into 6 parts x 12 chunks of 512 (stored base logits per part 929,897,392 B < 1 GB, CALC from perplexity.cpp:
#             2*(2*((V+1)/2)+4) B/token x 255 tokens/chunk x 12), C4 validation shard 0 -> datasets/c4_val_part.raw (73 windows), Q8_0 reference, ik IQ2_KT equal-bytes requant
#             (exact r56_kteq_cpu.sh recipe, bytes must be 1,241,589,376, sha256 logged).
#   3. GPU    every llama.cpp run in its own bin/job hold, one at a time; inside the hold it first waits until used VRAM <= 5500 MiB (4900 for Q8_0 runs), so the
#             second slot (gjob2, <= 4.8 GB) plus this job stays < 10.5 GB; samples total VRAM every 2 s and logs the peak.
#             per part: Q8_0 base logits -> KLD for Model B native, Model B2 native (only if research/agents/model_b2_flat.md exists), mainline IQ2_XXS, ik IQ2_KT-eq -> delete logits.
#             then C4 perplexity (r36_hfj.sh flags: -c 2049 --ppl-stride 4095 => ctx 4096) for Q8_0, Model B, Model B2, IQ2_XXS, IQ2_KT-eq.
#   4. REPORT research/detail/r59_kldc4_report.py (guard refuses partial logs) -> research/agents/eval_r59_kldc4.md, notes copy, MODEL_B_PROGRESS.md row.
#   5. CLEAN  delete Q8_0, the KT quant (sha256 logged first), imatrix .dat, all logits.
# Test hooks: R59_POSTLOG R59_B2REPORT R59_POLL R59_GIVEUP R59_DEADN R59_PIDS R59_LOG R59_LOCK R59_R R59_W R59_WAIT_ONLY=1 (stop after stage 1) R59_DRY=1 (print every job instead of running it).
cd ~/repo || exit 1
export HF_HOME=$HOME/repo/hf_cache
SELF=$HOME/repo/research/detail/r59_kldc4.sh
D=research/detail; L=logs; R=${R59_R:-$L/r59}; W=${R59_W:-work/r59}
LOG=${R59_LOG:-$L/r59_kldc4.log}; PL=${R59_POSTLOG:-$L/r57_post.log}; B2REP=${R59_B2REPORT:-research/agents/model_b2_flat.md}
POLL=${R59_POLL:-60}; GIVEUP=${R59_GIVEUP:-18000}; DEADN=${R59_DEADN:-5}; PIDS=${R59_PIDS:-$L/run_1017/pids}; LOCK=${R59_LOCK:-$L/r59_kldc4.lock}
SB=llama.cpp/build-cuda/bin; FB=research/kernels/llama_f44/build/bin; IKR=ext/ik_llama.cpp; IK=$IKR/build-cuda/bin
F16=models/qwen3-4B-f16.gguf; H=datasets/wikitext2_test_hfjoin.raw; C4=datasets/c4_val_part.raw; PP=datasets/r59_wt2_p; CPP=12
B2G1=${R59_B2GLOB:-work/r57/native/4B-flat-vrnative-00001-of-*.gguf}; B2GA=${B2G1/00001/0000*}
MX=work/r45b/4B-IQ2_XXS-r45b.gguf; MB=work/r51t/native/4B-pingala8-vrnative-00001-of-00002.gguf
Q8=$W/qwen3-4B-Q8_0.gguf; KT=$W/qwen3-4B-IQ2_KT-eq-ik.gguf; IM=$W/qwen3-4B.imatrix.dat
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"   # research/detail/pe52_gpuA.sh
KT_BYTES=1241589376; R56SHA=d5fca9c5c2379106968a8859e47422de4919db4799252f74387208ea4bed55a1
CQ='blk\.(7|8|9|1[0-9]|2[0-9]|3[0-3])\.attn_output\.weight=iq2_kt,blk\.(7|8|9|1[1-9]|2[01])\.attn_v\.weight=iq3_k,blk\.(10|11)\.ffn_down\.weight=iq2_kt'   # r56_kteq_cpu.sh
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
free_gb() { df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9; }
vused() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); echo "${u:-99999}"; }
need_c() { local n=$1 i; for i in $(seq 1 30); do [ "$(free_gb)" -ge "$n" ] && return 0; sleep 120; done; say "BLOCKER C: free $(free_gb) GB < $n GB for 60 min"; return 1; }
kld_ok() { grep -aq 'Mean *KLD' "$1" 2>/dev/null && grep -aqE "^ +$CPP +[0-9]" "$1" 2>/dev/null; }
c4_ok() { grep -aq 'Final estimate' "$1" 2>/dev/null && grep -aqE '\[73\][0-9]' "$1" 2>/dev/null; }
fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }
sumsize() { local t=0 f; for f in "$@"; do [ -f "$f" ] && t=$((t + $(fsize "$f"))); done; echo $t; }
arm_def() { case $1 in   # sets BIN EV M
  mB) BIN=$FB; EV="$NAT"; M=$MB;; b2) BIN=$FB; EV="$NAT"; M=$M2;; xxs) BIN=$SB; EV=""; M=$MX;; kt) BIN=$IK; EV=""; M=$KT;; q8) BIN=$SB; EV=""; M=$Q8;; esac; }

# ------------------------------------------------------------------ bodies run INSIDE bin/job (re-exec of this file)
gated() {   # args: LIMIT_MiB LABEL LOGFILE cmd... ; GPU lock is held; wait for VRAM headroom (second slot), run, sample total VRAM
  local lim=$1 lab=$2 lf=$3 i; shift 3
  source bin/cuda_env.sh
  for i in $(seq 1 ${R59_GATEN:-60}); do [ "$(vused)" -le "$lim" ] && break; sleep ${R59_GATESLEEP:-20}; done
  if [ "$(vused)" -gt "$lim" ]; then echo "VGATE timeout: used $(vused) MiB > $lim ($lab)"; return 76; fi
  local pk=$R/.peak_$lab; vused > $pk
  ( while :; do vused >> $pk; sleep 2; done ) & local vs=$!
  "$@" > $lf 2>&1; local rc=$?
  kill $vs 2>/dev/null; wait $vs 2>/dev/null; vused >> $pk
  local peak; peak=$(sort -n $pk | tail -1); rm -f $pk
  echo "RUN $lab rc=$rc total_vram_peak=${peak:-?} MiB (sampled 2 s, gate $lim)$([ "${peak:-0}" -gt 10500 ] && echo ' OVER 10.5 GB')"
  return $rc
}
cpu_q8() {
  source bin/cuda_env.sh; local s; s=$(date +%s)
  $SB/llama-quantize $F16 $Q8 Q8_0 12 > $R/quant_q8.log 2>&1; local rc=$?
  echo "QUANT Q8_0 rc=$rc secs=$(( $(date +%s) - s )) bytes=$(fsize $Q8)"; [ $rc -eq 0 ] && [ "$(fsize $Q8)" -gt 4200000000 ]
}
cpu_kt() {   # exactly research/detail/r56_kteq_cpu.sh (imatrix conversion + custom-q requant)
  source bin/cuda_env.sh
  [ -s $IM ] || python -I $IKR/convert_imatrix_gguf_to_dat.py --outfile $IM models/qwen3-4B.imatrix > $R/imatrix_convert.log 2>&1
  [ -s $IM ] || { echo "imatrix conversion failed"; return 4; }
  local s; s=$(date +%s); rm -f $KT
  echo "$IK/llama-quantize --imatrix $IM --custom-q $CQ $F16 $KT IQ2_KT 12" > $R/quant_kt.cmd
  $IK/llama-quantize --imatrix $IM --custom-q "$CQ" $F16 $KT IQ2_KT 12 > $R/quant_kt.log 2>&1; local rc=$?
  echo "QUANT KTeq rc=$rc secs=$(( $(date +%s) - s )) bytes=$(fsize $KT) expected=$KT_BYTES"
  [ $rc -eq 0 ] && [ "$(fsize $KT)" = "$KT_BYTES" ]
}
case "${1:-}" in __gated) shift; gated "$@"; exit $?;; __q8) cpu_q8; exit $?;; __kt) cpu_kt; exit $?;; esac

final_clean() {   # delete every temp: logits, Q8_0, KT quant (its sha256 is logged first), imatrix .dat
  [ -s $R/kt_sha256.txt ] || { [ -s $KT ] && sha256sum $KT | awk -v s=$(fsize $KT) '{print $1"  "$2"  "s}' > $R/kt_sha256.txt; }
  rm -f $W/q8_base_p*.bin $Q8 $KT $IM; rmdir $W 2>/dev/null
}
# ------------------------------------------------------------------ job wrappers (main shell)
cpu_job() { if [ -n "${R59_DRY:-}" ]; then say "DRY cpu_job: $*"; return 0; fi; CPU=1 ~/repo/bin/job "$@"; }
gpu_job() {   # LABEL LIMIT LOGFILE cmd... ; one bin/job hold per call; VRAM-gate timeouts (76) and C: refusals (75) are retried after 2 min
  local lab=$1 lim=$2 lf=$3 t out rc ln; shift 3
  if [ -n "${R59_DRY:-}" ]; then say "DRY gpu_job $lab gate=$lim log=$lf: $*"; return 0; fi
  for t in 1 2 3 4 5 6; do
    out=$(~/repo/bin/job bash $SELF __gated $lim $lab $lf "$@" 2>&1); rc=$?
    while IFS= read -r ln; do [ -n "$ln" ] && say "GPU[$lab] $ln"; done <<< "$out"
    { [ $rc -eq 76 ] || [ $rc -eq 75 ]; } || return $rc
    sleep 120
  done
  return $rc
}

# ------------------------------------------------------------------ stage 1: wait for the Model B2 chain
b2_state() {   # prints WAIT | REPORT | SKIPPED | END | BLOCKER | DEAD  (markers only count if they come after the LAST "r57_post start pid" line)
  local sect pid alive=0
  sect=$(tac "$PL" 2>/dev/null | awk '/r57_post start pid/{exit} {print}')
  pid=$(grep -a 'r57_post start pid' "$PL" 2>/dev/null | tail -1 | sed -E 's/.*r57_post start pid ([0-9]+).*/\1/')
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
  if grep -aq 'R57 POST skipped' <<< "$sect"; then echo SKIPPED
  elif grep -aqE 'REPORT rc=0|report already written' <<< "$sect"; then echo REPORT
  elif grep -aq 'r57_post end' <<< "$sect"; then echo END
  elif grep -aq 'BLOCKER' <<< "$sect"; then [ $alive = 0 ] && echo BLOCKER || echo WAIT
  elif [ -n "$pid" ] && [ $alive = 0 ]; then echo DEAD
  else echo WAIT; fi
}

mkdir -p $L $R $W $(dirname $PIDS)
exec 8>$LOCK; flock -n 8 || { echo "another r59_kldc4 holds $LOCK"; exit 0; }
SID=$(ps -o sid= -p $$ | tr -d ' '); grep -qx "$SID" $PIDS 2>/dev/null || echo "$SID" >> $PIDS
say "r59_kldc4 start pid $$ sid $SID poll ${POLL}s giveup ${GIVEUP}s deadn $DEADN C: free $(free_gb) GB post-log $PL"
if [ -z "${R59_WAIT_ONLY:-}" ] && python3 $D/r59_kldc4_report.py --done >/dev/null 2>&1; then say "report already written and complete, nothing to do"; exit 0; fi

t0=$(date +%s); dead=0
say "waiting for the Model B2 chain: markers in $PL after its last 'r57_post start' line (REPORT rc=0 | R57 POST skipped | r57_post end | BLOCKER with pid gone)"
while :; do
  st=$(b2_state)
  case $st in
    REPORT|SKIPPED|END|BLOCKER) say "Model B2 chain finished: $st"; break;;
    DEAD) dead=$((dead + 1)); if [ $dead -ge $DEADN ]; then st=DEADFINAL; say "r57_post pid gone without a final marker for $DEADN polls: treating the chain as finished"; break; fi;;
    *) dead=0;;
  esac
  if [ $(( $(date +%s) - t0 )) -ge "$GIVEUP" ]; then say "BLOCKER R59: gave up after ${GIVEUP}s, Model B2 chain state $st"; exit 2; fi
  sleep $POLL
done
if [ -n "${R59_WAIT_ONLY:-}" ]; then echo "WAIT_RESULT=$st"; exit 0; fi

# arms
KARMS="mB"; CARMS="q8 mB"; M2=""
if { [ -f $B2REP ] || grep -aqE '\[73\][0-9]' ${R59_B2PPL:-logs/r36_hfj_r57_flat_native.log} 2>/dev/null; } && ls $B2G1 >/dev/null 2>&1; then
  M2=$(ls $B2G1 | head -1); KARMS="$KARMS b2"; CARMS="$CARMS b2"; say "Model B2 present: $M2"
else say "Model B2 skipped ($B2REP missing or no native shards)"; fi
KARMS="$KARMS xxs kt"; CARMS="$CARMS xxs kt"
trap final_clean EXIT
for x in $SB/llama-perplexity $SB/llama-quantize $FB/llama-perplexity $IK/llama-perplexity $IK/llama-quantize $IKR/convert_imatrix_gguf_to_dat.py llama.cpp/build/bin/llama-tokenize \
         $F16 $H $MX $MB models/qwen3-4B.imatrix $D/r59_kldc4_prep.py $D/r59_kldc4_report.py; do
  [ -e "$x" ] || { say "BLOCKER missing $x"; exit 4; }; done
need_c 12 || exit 3

# ------------------------------------------------------------------ stage 2: CPU prep
if ! { grep -aq 'PARTS n=' $R/split.log 2>/dev/null && [ -s ${PP}1.raw ]; }; then
  cpu_job python -I $D/r59_kldc4_prep.py split $H $PP $F16 --ctx 512 --nparts 6 --cpp $CPP > $R/split.log 2>&1; say "SPLIT rc=$? $(grep -a 'PARTS n=' $R/split.log)"
fi
if [ -z "${R59_DRY:-}" ]; then
  grep -aq 'PARTS n=' $R/split.log || { say "BLOCKER split failed (see $R/split.log)"; exit 5; }
  EST=$(sed -n 's/.*est_file_bytes= \([0-9]*\).*/\1/p' $R/split.log | tail -1); [ "${EST:-9999999999}" -lt 1000000000 ] || { say "BLOCKER stored base logits per part $EST >= 1 GB"; exit 5; }
fi
NP=$(sed -n 's/.*PARTS n= \([0-9]*\) .*/\1/p' $R/split.log | tail -1); NP=${NP:-6}
if ! { grep -aq 'strided chunks (ctx 4096, stride 4095) 73 ' $R/c4_prep.log 2>/dev/null && [ -s $C4 ]; }; then
  for att in 1 2; do
    cpu_job env HF_HOME=$HF_HOME python -I $D/r59_kldc4_prep.py c4 $C4 $F16 > $R/c4_prep.log 2>&1; say "C4 prep attempt $att rc=$? $(grep -a 'C4 docs' $R/c4_prep.log | cut -c1-200)"
    grep -aq 'strided chunks (ctx 4096, stride 4095) 73 ' $R/c4_prep.log && break; [ -n "${R59_DRY:-}" ] && break; sleep 60
  done
  [ -n "${R59_DRY:-}" ] || grep -aq 'strided chunks (ctx 4096, stride 4095) 73 ' $R/c4_prep.log || { say "BLOCKER C4 prep failed (see $R/c4_prep.log)"; exit 5; }
fi
need_q8=0; need_kt=0
for a in $KARMS; do for p in $(seq 1 $NP); do kld_ok $R/kld_${a}_p$p.log || { need_q8=1; [ $a = kt ] && need_kt=1; }; done; done
for a in $CARMS; do c4_ok $R/c4_$a.log || { [ $a = q8 ] && need_q8=1; [ $a = kt ] && need_kt=1; }; done
say "prep needs: Q8_0 $need_q8, KT requant $need_kt"
if [ $need_q8 = 1 ] && [ "$(fsize $Q8)" -le 4200000000 ]; then
  need_c 12 || exit 3; cpu_job bash $SELF __q8 2>&1 | while read -r ln; do say "CPU[q8] $ln"; done
  [ -n "${R59_DRY:-}" ] || [ "$(fsize $Q8)" -gt 4200000000 ] || { say "BLOCKER Q8_0 quantize failed (see $R/quant_q8.log)"; exit 5; }
fi
if [ $need_kt = 1 ] && [ "$(fsize $KT)" != "$KT_BYTES" ]; then
  need_c 12 || exit 3
  for att in 1 2; do
    cpu_job bash $SELF __kt 2>&1 | while read -r ln; do say "CPU[kt] $ln"; done
    [ "$(fsize $KT)" = "$KT_BYTES" ] && break; [ -n "${R59_DRY:-}" ] && break; sleep 30
  done
  [ -n "${R59_DRY:-}" ] || [ "$(fsize $KT)" = "$KT_BYTES" ] || { say "BLOCKER KT requant bytes $(fsize $KT) != $KT_BYTES (see $R/quant_kt.log)"; rm -f $KT; exit 5; }
fi
if [ -s $KT ]; then
  sha256sum $KT | awk -v s=$(fsize $KT) '{print $1"  "$2"  "s}' > $R/kt_sha256.txt
  if [ "$(cut -d' ' -f1 $R/kt_sha256.txt)" = "$R56SHA" ]; then say "KT requant sha256 IDENTICAL to r56 ($R56SHA)"; else say "NOTE KT requant sha256 $(cut -d' ' -f1 $R/kt_sha256.txt) differs from r56 $R56SHA"; fi
  [ "$(sed 's#work/r54kt/#X/#g; s#"##g' logs/r56_kteq_quant.cmd | sed -E 's#^[^ ]+llama-quantize #Q #')" = "$(sed 's#work/r59/#X/#g; s#"##g' $R/quant_kt.cmd | sed -E 's#^[^ ]+llama-quantize #Q #')" ] \
    && say "KT recipe/command line identical to logs/r56_kteq_quant.cmd" || say "NOTE KT command differs from logs/r56_kteq_quant.cmd (compare $R/quant_kt.cmd)"
fi
# arms table for the report: key, label, bytes
{ printf 'mB\tModel B\t%s\n' "$(sumsize work/r51t/native/4B-pingala8-vrnative-0000*-of-00002.gguf)"
  [ -n "$M2" ] && printf 'b2\tModel B2\t%s\n' "$(sumsize $B2GA)"
  printf 'xxs\tIQ2_XXS\t%s\n' "$(fsize $MX)"
  printf 'kt\tIQ2_KT-eq\t%s\n' "$( [ -s $KT ] && fsize $KT || sed -n 's/.*  \([0-9]*\)$/\1/p' $R/kt_sha256.txt | tail -1 )"; } > $R/arms.txt
say "arms: $(tr '\t\n' ' |' < $R/arms.txt)"

# ------------------------------------------------------------------ stage 3: GPU
gpu_phase() {
  local p a todo K s rc lim
  for p in $(seq 1 $NP); do
    todo=0; for a in $KARMS; do kld_ok $R/kld_${a}_p$p.log || todo=1; done
    [ $todo = 0 ] && continue
    need_c 12 || return 3
    K=$W/q8_base_p$p.bin
    if ! { [ -f $K ] && grep -aq 'Final estimate' $R/base_p$p.log 2>/dev/null; }; then
      rm -f $K
      gpu_job base_p$p 7000 $R/base_p$p.log $SB/llama-perplexity -m $Q8 -f ${PP}$p.raw -c 512 --chunks $CPP -ngl 99 --kl-divergence-base $K; rc=$?
      say "BASE p$p rc=$rc bytes=$(fsize $K) $(grep -a -o 'Final estimate: PPL = [0-9.]*' $R/base_p$p.log 2>/dev/null)"
    fi
    if [ -z "${R59_DRY:-}" ]; then
      s=$(fsize $K)
      if [ "$s" -le 0 ] || [ "$s" -ge 1000000000 ]; then say "PART SKIPPED p$p: base logits bytes $s (need 0 < size < 1e9); deleted"; rm -f $K; continue; fi
    fi
    for a in $KARMS; do
      kld_ok $R/kld_${a}_p$p.log && continue
      arm_def $a
      gpu_job kld_${a}_p$p 9000 $R/kld_${a}_p$p.log env $EV $BIN/llama-perplexity -m $M --kl-divergence-base $K --kl-divergence -c 512 --chunks $CPP -ngl 99; rc=$?
      say "KLD $a p$p rc=$rc $(grep -a -E 'Mean +KLD|Same top p' $R/kld_${a}_p$p.log 2>/dev/null | tr -s ' ' | tr '\n' '|')"
    done
    rm -f $K
  done
  for a in $CARMS; do
    c4_ok $R/c4_$a.log && continue
    arm_def $a; lim=9000; [ $a = q8 ] && lim=7000
    gpu_job c4_$a $lim $R/c4_$a.log env $EV $BIN/llama-perplexity -f $C4 -c 2049 --ppl-stride 4095 -ngl 99 -m $M; rc=$?
    say "C4 $a rc=$rc last $(grep -a -o '\[[0-9]*\][0-9.]*' $R/c4_$a.log 2>/dev/null | tail -1) $(grep -a -o 'computing over [0-9]* chunks, n_ctx=[0-9]*' $R/c4_$a.log 2>/dev/null) $(grep -a -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' $R/c4_$a.log 2>/dev/null)"
  done
  return 0
}
( while :; do echo "$(date +%T) $(vused)" >> $R/vram.log; sleep 10; done ) & VS=$!
trap 'kill $VS 2>/dev/null; final_clean' EXIT
for att in 1 2; do
  gpu_phase; grc=$?
  say "GPU phase attempt $att rc=$grc"
  python3 $D/r59_kldc4_report.py --complete >/dev/null 2>&1 && break
  say "INCOMPLETE after attempt $att: $(python3 $D/r59_kldc4_report.py --complete 2>&1 | tail -1 | cut -c1-300)"
  [ $att -lt 2 ] && [ -z "${R59_DRY:-}" ] && sleep 120
done

# ------------------------------------------------------------------ stage 4: report (guard refuses partial logs)
if python3 $D/r59_kldc4_report.py --complete >> $LOG 2>&1; then
  python3 $D/r59_kldc4_report.py >> $LOG 2>&1; rrc=$?
  say "REPORT rc=$rrc ($(wc -w < research/agents/eval_r59_kldc4.md 2>/dev/null) words)"
else
  say "BLOCKER report refused (partial logs); see $R, rerun this script to finish the missing runs"
fi

# ------------------------------------------------------------------ stage 5: cleanup (sha256 of the KT quant was logged in $R/kt_sha256.txt before any GPU run)
say "CLEANUP sha256 $(cut -c1-16 $R/kt_sha256.txt 2>/dev/null); deleting Q8_0, KT quant, imatrix, logits"
final_clean
say "r59_kldc4 end; C: free $(free_gb) GB; leftover in $W: $(ls $W 2>/dev/null | tr '\n' ' ')"
