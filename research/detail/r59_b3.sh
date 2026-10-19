#!/bin/bash
# R59 B3 (prereg research/work/r59/prereg_b3.txt): native file from f1 roff blobs + f4 blk.4.ffn_gate (class 2), then native ppl and paired tests.
# Same CPU steps as r57_post.sh cpu_stage (r51_delta.py, r51_native.py plan/dryrun/build/verify); ppl via r36_hfj.sh with the NPPC2 native env (2-bit, safe beside a torch build).
cd ~/repo; export HF_HUB_OFFLINE=1; L=logs/r59b3; mkdir -p $L; LOG=logs/r59_b3.log; W=work/r59b3; D=research/detail
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
BD=$W/blobs_b3; STJ=$W/stats_b3.json; NPZ=$W/delta/vr_delta_b3.npz; STEM=$W/native/4B-b3-vrnative; mkdir -p $W/native $W/delta
NENV="R51N_BD=$BD R51N_STATS=$STJ R51N_VAR=b3 R51N_NPZ=$NPZ"; FB=research/kernels/llama_f44/build/bin
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
say "start pid $$ sid $(ps -o sid= -p $$ | tr -d ' ')"
grep -aq '^DELTA ' $L/delta.log 2>/dev/null && [ -s $NPZ ] || { CPU=1 bin/job env R51D_STATS=$STJ R51D_VAR=b3 R51D_OUT=vr_delta_b3.npz python $D/r51_delta.py $BD $W/delta > $L/delta.log 2>&1; say "DELTA rc=$? $(grep -a '^DELTA ' $L/delta.log | cut -c1-160)"; }
[ -s $NPZ ] || { say "BLOCKER delta"; exit 5; }
grep -aq '^PLAN TOTAL' $L/plan.log 2>/dev/null || { CPU=1 bin/job env $NENV python $D/r51_native.py plan $W/plan_tmp > $L/plan.log 2>&1; rm -f $W/plan_tmp*.gguf; say "PLAN $(grep -a '^PLAN TOTAL' $L/plan.log | cut -c1-200)"; }
grep -aq '^PLAN TOTAL.*(OK)' $L/plan.log || { say "BLOCKER plan over budget (see $L/plan.log)"; exit 5; }
grep -aq 'DRYRUN readback PASS' $L/dry.log 2>/dev/null || { mkdir -p $W/dry; CPU=1 bin/job env $NENV python $D/r51_native.py dryrun $W/dry/4B-dry > $L/dry.log 2>&1; rm -rf $W/dry; say "DRYRUN $(grep -a 'DRYRUN readback' $L/dry.log | cut -c1-120)"; }
grep -aq 'DRYRUN readback PASS' $L/dry.log || { say "BLOCKER dryrun"; exit 5; }
{ grep -aq '^NATIVE TOTAL' $L/build.log 2>/dev/null && ls $STEM-00001-of-*.gguf >/dev/null 2>&1; } || { rm -f $STEM-0000*.gguf; CPU=1 bin/job env $NENV python $D/r51_native.py build $STEM > $L/build.log 2>&1; say "BUILD $(grep -a '^NATIVE TOTAL' $L/build.log | cut -c1-200)"; }
grep -aq '^VERIFY .* PASS' $L/verify.log 2>/dev/null || { CPU=1 bin/job env $NENV python $D/r51_native.py verify $STEM > $L/verify.log 2>&1; say "VERIFY $(grep -a '^VERIFY' $L/verify.log | cut -c1-200)"; }
grep -aq '^VERIFY .* PASS' $L/verify.log || { say "BLOCKER verify"; exit 5; }
M1=$(ls $STEM-00001-of-*.gguf | head -1); say "native $(ls -l $STEM-0000*.gguf | awk '{s+=$5} END {print s}') B (IQ2_XXS 1246621312) $M1"
grep -aqE '\[73\][0-9]' logs/r36_hfj_r59_b3_native.log 2>/dev/null || { bin/job env $NAT HFJ_BIN=$FB bash $D/r36_hfj.sh r59_b3_native $M1 > $L/ppl.out 2>&1; }
say "PPL $(grep -a -o '\[73\][0-9.]*' logs/r36_hfj_r59_b3_native.log | tail -1)"
for c in "logs/r36_hfj_iq2xxs.log IQ2_XXS" "logs/r56_kteq_ppl.log IQ2_KT-eq" "logs/r36_hfj_r51_native_PPC2.log ModelB" "logs/r36_hfj_r52a_roff.log flat-control"; do set -- $c
  [ -f $1 ] && say "PAIRED B3 vs $2: $(python3 $D/r59_paired.py logs/r36_hfj_r59_b3_native.log $1 2>&1 | tail -1)"; done
say "r59_b3 end"
