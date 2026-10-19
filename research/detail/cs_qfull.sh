#!/bin/bash
# cs_qfull.sh (2026-10-18): full Qwen3-4B rebuild with column scale CMODE=opt (prereg research/work/r60/prereg_cscale_qfull.txt).
# 1) GPU blob build (cs_build_q.py, roff arm, all layers, NSEQ 256); 2) CPU native stages of r59_b3p.sh with cs_qnative.py on the new blobs;
# 3) 73-window r36_hfj + paired tests. Log logs/cs_qfull.log. Resumable by markers. Plain ASCII.
cd "$(dirname "$0")/../.." || exit 1; export HF_HUB_OFFLINE=1; LOG=logs/cs_qfull.log; L=logs/csq; mkdir -p $L; D=research/detail
W=work/csq; BO=$W/f1o; STEM=$W/native/4B-csq-vr3; NPZ=$W/delta/vr_delta_csq.npz; mkdir -p $W/native $W/delta
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
waitc() { while [ -e logs/commit_low ]; do sleep 60; done; }
FB=research/kernels/llama_f44/build/bin
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
say "start pid $$"
if ! grep -aq '^SUMMARY' $L/build.tensors 2>/dev/null; then
  waitc; FREE=$(df -BM --output=avail /mnt/c | tail -1 | tr -dc 0-9); [ "$FREE" -ge 15000 ] || { say "BLOCKER C: free ${FREE} MB"; exit 3; }
  rm -rf $BO; mkdir -p $BO
  bin/job bash -c "source bin/cuda_env.sh; source .venv/bin/activate; export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
   CMODE=opt GPUCHOL_MAXN=10000 GPUFRAC=0.75 R48K_SHIFTS=0,0,0,0,2,3,2,4/0,0,0,0,0,0,0,2 ARMS=roff:8:research/work/r48k3:rateoff NSEQ=256 NEV=32 \
   DAMP=0.1 ALPHA=1.05 NSAL=4 AVB=2.0625 exec python $D/cs_build_q.py work/r45b/4B-IQ2_XXS-r45b.gguf $BO > $L/build.tensors 2>&1"
  rc=$?; rm -rf $BO/xs; say "BUILD rc=$rc $(grep -a '^SUMMARY' $L/build.tensors | cut -c1-200)"
  grep -aq '^SUMMARY' $L/build.tensors || { say "BLOCKER build"; exit 5; }
fi
[ $(ls $BO/blobs_roff | wc -l) -eq 253 ] || { say "BLOCKER blob count $(ls $BO/blobs_roff | wc -l)"; exit 5; }
PY="env NT=6 CSQ_BD=$BO/blobs_roff B3P_OUT=$W/state python3 $D/cs_qnative.py"
stage() { local n=$1 m=$2; shift 2; grep -aq -- "$m" $L/$n.log 2>/dev/null && return 0
  for try in 1 2 3 4 5; do waitc; CPU=1 bin/job "$@" > $L/$n.log 2>&1; say "$n try $try rc=$? $(grep -a -- "$m" $L/$n.log | tail -1 | cut -c1-200)"
    grep -aq -- "$m" $L/$n.log && return 0; [ -e logs/commit_low ] || grep -aqE "Failed to start transient scope|signal 9|Killed" $L/$n.log || break; sleep 60; done
  say "BLOCKER $n (see $L/$n.log)"; exit 5; }
stage scan   '^SCAN b3p blobs'            $PY scan
stage delta  '^DELTA b3p '                $PY delta $W/delta
mv -f $W/delta/vr_delta_b3p.npz $NPZ 2>/dev/null
stage plan   '^PLAN layout saving'        $PY plan
grep -aq "size-matched" $L/plan.log || { say "BLOCKER plan: over IQ2_XXS"; exit 5; }
stage check  'CHECK SUMMARY 252/252 PASS' $PY check all
stage build  '^NATIVE3 TOTAL'             env SET=b3p B3P_NPZ=$NPZ $PY build $STEM
stage verify '^VERIFY3 .* PASS'           env SET=b3p B3P_NPZ=$NPZ $PY verify $STEM
say "NATIVE csq $(ls -l $STEM-0000*.gguf | awk '{s+=$5} END {print s}') B (B3+ 1,244,919,264; IQ2_XXS 1,246,622,976?)"
waitc; grep -aqE '\[73\][0-9]' logs/r36_hfj_csq_native.log 2>/dev/null || bin/job env $NAT HFJ_BIN=$FB bash $D/r36_hfj.sh csq_native $(ls $STEM-00001-of-*.gguf) > $L/ppl.out 2>&1
say "HFJ csq $(grep -a -o '\[73\][0-9.]*' logs/r36_hfj_csq_native.log | tail -1) (prereg 17.45 [17.00, 17.85]; B3+ 17.78)"
for c in "logs/r36_hfj_r59_b3p_native.log B3+" "logs/r56_kteq_ppl.log IQ2_KT-eq" "logs/r36_hfj_iq2xxs.log IQ2_XXS"; do set -- $c
  [ -f $1 ] && say "PAIRED csq vs $2: $(python3 $D/r59_paired.py logs/r36_hfj_csq_native.log $1 2>&1 | tail -1)"; done
say "end"
