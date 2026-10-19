#!/bin/bash
# R59 B3+ (prereg research/agents/solver_b3plus_r59.md): all 252 linears from f1 roff in the VR3 native layout (no vr_prm, no M; vr.layout 3).
# usage: bash research/detail/r59_b3p.sh cpu   CPU stages, each its own CPU=1 bin/job (resumable: a stage is skipped once its log has its marker)
#        bin/job bash research/detail/r59_b3p.sh gpu   GPU stages (under the GPU lock; needs cpu done): old-binary determinism run (8 windows),
#                                              incremental libllama build (5 C++ TUs, no CUDA TU, -j2), layout checks (8 windows), B3+ 73-window ppl, paired tests
# Every stage waits while logs/commit_low exists. Caps: <= 2.5 GB RSS per CPU stage, NT 6 threads.
cd ~/repo; export HF_HUB_OFFLINE=1; L=logs/b3p; mkdir -p $L; LOG=logs/r59_b3p.log; W=work/b3p; D=research/detail; mkdir -p $W/native $W/native_b3v3 $W/delta
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
waitc() { while [ -e logs/commit_low ]; do sleep 60; done; }
PY="env NT=6 B3P_OUT=$W/state python3 $D/b3p_native.py"; NPZ=$W/delta/vr_delta_b3p.npz
STEM=$W/native/4B-b3p-vr3; STEM3=$W/native_b3v3/4B-b3-vr3; B3STEM=work/r59b3/native/4B-b3-vrnative
FB=research/kernels/llama_f44/build/bin; HP="-f datasets/wikitext2_test_hfjoin.raw -c 2049 --ppl-stride 4095 -ngl 99"
NAT="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
stage() {   # stage NAME MARKER cmd...   (skip if the log already has MARKER)
  local n=$1 m=$2; shift 2
  grep -aq -- "$m" $L/$n.log 2>/dev/null && return 0
  for try in $(seq 1 30); do   # a guard stop of proj.slice (commit_low) kills or refuses the job: wait for the flag to clear and retry (stages resume)
    waitc; CPU=1 /usr/bin/time -f "maxrss_kB %M secs %e" bin/job "$@" > $L/$n.log 2>&1; say "$n try $try rc=$? $(grep -a -- "$m" $L/$n.log | tail -1 | cut -c1-220) $(grep -a maxrss_kB $L/$n.log | tail -1)"
    grep -aq -- "$m" $L/$n.log && return 0
    if [ -e logs/commit_low ] || grep -aqE "Failed to start transient scope|signal 15|signal 9|Killed" $L/$n.log; then cp $L/$n.log $L/$n.try$try.log; sleep 60; continue; fi
    break
  done
  say "BLOCKER $n (see $L/$n.log)"; exit 5
}
if [ "$1" = cpu ]; then
  say "cpu start pid $$ sid $(ps -o sid= -p $$ | tr -d ' ')"
  stage scan   '^SCAN b3p blobs'            $PY scan
  stage delta  '^DELTA b3p '                $PY delta $W/delta
  stage plan   '^PLAN layout saving'        $PY plan
  grep -aq "size-matched" $L/plan.log || { say "BLOCKER plan: B3+ VR3 over IQ2_XXS"; exit 5; }
  stage check  'CHECK SUMMARY 252/252 PASS' $PY check all
  stage build  '^NATIVE3 TOTAL'             env SET=b3p B3P_NPZ=$NPZ $PY build $STEM
  stage verify '^VERIFY3 .* PASS'           env SET=b3p B3P_NPZ=$NPZ $PY verify $STEM
  stage build_b3v3  '^NATIVE3 TOTAL'        env SET=b3 $PY build $STEM3
  stage verify_b3v3 '^VERIFY3 .* PASS'      env SET=b3 $PY verify $STEM3
  say "cpu done: B3+ $(ls -l $STEM-0000*.gguf | awk '{s+=$5} END {print s}') B; B3-in-VR3 $(ls -l $STEM3-0000*.gguf | awk '{s+=$5} END {print s}') B"
elif [ "$1" = gpu ]; then
  grep -aq '^VERIFY3 .* PASS' $L/verify.log && grep -aq '^VERIFY3 .* PASS' $L/verify_b3v3.log || { say "BLOCKER gpu: cpu stages not done"; exit 5; }
  waitc
  source bin/cuda_env.sh
  run8() { [ -s $L/$1.log ] && grep -aq '\[8\]' $L/$1.log || env $NAT $FB/llama-perplexity $HP --chunks 8 -m $2 > $L/$1.log 2>&1; echo "$(grep -a -o '\[[0-9]*\][0-9.]*' $L/$1.log | head -8 | tr '\n' ' ')"; }
  REF=$(grep -a -o '\[[0-9]*\][0-9.]*' logs/r36_hfj_r59_b3_native.log | head -8 | tr '\n' ' ')
  # 0) determinism: the OLD binary on the B3 v2 file, 8 windows, must reproduce the logged B3 run
  waitc; Z=$(run8 l8_b3v2_old $(ls $B3STEM-00001-of-*.gguf)); say "DETERMINISM v2 file, old binary: $Z | ref $REF | $([ "$Z" = "$REF" ] && echo SAME || echo DIFFER)"
  # 1) incremental build: only libllama C++ TUs (llama-model-loader/-model/-quant/-vocab, llama.cpp; no CUDA TU); run while no llama_f44 binary is in use
  grep -aq 'Built target llama-perplexity' $L/make.log 2>/dev/null || { make -C research/kernels/llama_f44/build -j2 llama-perplexity > $L/make.log 2>&1; say "make rc=$? $(tail -1 $L/make.log)"; }
  grep -aq 'Built target llama-perplexity' $L/make.log || { say "BLOCKER make"; exit 5; }
  waitc; A=$(run8 l8_b3v2 $(ls $B3STEM-00001-of-*.gguf)); say "LAYOUT v2 file, new binary (backward compat): $A | ref $REF | $([ "$A" = "$REF" ] && echo SAME || echo DIFFER)"
  waitc; B=$(run8 l8_b3v3 $(ls $STEM3-00001-of-*.gguf));  say "LAYOUT B3 in VR3 (same weights):          $B | ref $REF | $([ "$B" = "$REF" ] && echo SAME || echo DIFFER)"
  { [ "$A" = "$Z" ] && [ "$B" = "$Z" ]; } || { say "BLOCKER layout check differs from the old-binary run"; exit 5; }
  waitc; grep -aqE '\[73\][0-9]' logs/r36_hfj_r59_b3p_native.log 2>/dev/null || env $NAT HFJ_BIN=$FB bash $D/r36_hfj.sh r59_b3p_native $(ls $STEM-00001-of-*.gguf) > $L/ppl.out 2>&1
  say "PPL B3+ $(grep -a -o '\[73\][0-9.]*' logs/r36_hfj_r59_b3p_native.log | tail -1) (prereg 17.78 [17.73, 17.86]; B3 18.0660; f1 carrier 17.7897)"
  for c in "logs/r36_hfj_r59_b3_native.log B3" "logs/r56_kteq_ppl.log IQ2_KT-eq" "logs/r36_hfj_iq2xxs.log IQ2_XXS" "logs/r36_hfj_r51_native_PPC2.log ModelB" "logs/r36_hfj_r52a_roff.log f1-carrier"; do set -- $c
    [ -f $1 ] && say "PAIRED B3+ vs $2: $(python3 $D/r59_paired.py logs/r36_hfj_r59_b3p_native.log $1 2>&1 | tail -1)"; done
  say "gpu done"
else echo "usage: $0 cpu|gpu"; exit 2; fi
