#!/bin/bash
# R48k driver: kernel-exact VRi + VRii-C(ung16) build on all 252 (GPU, one pass, both carriers) -> containers (all, iq212 x 2 formats)
# -> LLVQ ppl vrc all, vri all -> swap40 -> ppl vrc iq212, vri iq212 -> vrcB boost swap (layers 0-3) -> ppl vrcB iq212 -> verify -> delete carriers.
cd ~/repo; export HF_HUB_OFFLINE=1; W=work/r48k2; MIX=work/r45b/4B-IQ2_XXS-r45b.gguf; CI=$W/4B-r48k-vri.gguf; CC=$W/4B-r48k-vrc.gguf
chk() { a=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9); if [ $a -lt 12 ]; then echo "DISK_RULE_FAIL free ${a}G"; return 3; fi; }
date +"D0 %T"; mkdir -p $W; chk || exit 3
until grep -q TRAIN_DONE logs/r48k2_train.log 2>/dev/null; do sleep 15; done; date +"D0b %T tables ready"
for C in $CI $CC; do [ -f $C ] || NOLOCK=1 bin/job cp work/r45b/4B-r45b-ung8.gguf $C; done; chk || exit 3
if [ -z "$SKIPBUILD" ]; then
  bin/job bash -c "TABDIR=research/work/r48k2 R48K_SHIFTS=0,0,0,0,2,3,2,4/0,0,0,0,0,0,0,2 NSEQ=512 NEV=32 DAMP=0.1 ALPHA=1.05 BOOST=5.9e6 NSAL=4 python research/detail/r48k_build.py $MIX $CI $CC $W > logs/r48k2.tensors 2>&1"; echo "D1 $(date +%T) build rc=$?"
fi
grep -E '^SUMMARY|^BOOST|^GATES|ABORT|Error' logs/r48k2.tensors | tail -8; grep -q '^GATES' logs/r48k2.tensors || exit 5
for F in vri vrc vrcB; do for V in all iq212; do [ $F = vrcB ] && [ $V = all ] && continue
  if [ $F = vrcB ]; then rm -rf $W/blobs_vrcBm; mkdir -p $W/blobs_vrcBm; cp $W/blobs_vrc/*.blob $W/blobs_vrcBm/; cp $W/blobs_vrcB/*.blob $W/blobs_vrcBm/; BD=$W/blobs_vrcBm; else BD=$W/blobs_$F; fi
  NOLOCK=1 bin/job bash -c "OMP_NUM_THREADS=4 python research/detail/r48k_container.py encode $BD $MIX $W/r48k2_${F}_${V}.vedq $V > logs/r48k2_realfile_${F}_$V.log 2>&1"; grep REAL_BYTES logs/r48k2_realfile_${F}_$V.log; done; done
date +"D2 %T containers"
ppl() { echo "PPL_MODEL -m $2 sha256 $(sha256sum $2 | cut -c1-64) variant $1" | tee -a logs/r48k2_ppl.log
  bin/job bash research/detail/r36_hfj.sh r48k2_$1 $2 | tee -a logs/r48k2_ppl.log; date +"P %T $1" | tee -a logs/r48k2_ppl.log; }
ver() { NOLOCK=1 bin/job bash -c "NT=4 OMP_NUM_THREADS=4 python research/detail/r48k_container.py verify $W/r48k2_$1_$2.vedq $3 $4 $MIX $2 >> logs/r48k2_realfile_$1_$2.log 2>&1"; tail -n 1 logs/r48k2_realfile_$1_$2.log; }
ppl vrc_all $CC; ppl vri_all $CI
ver vrc all $W/blobs_vrc $CC &
ver vri all $W/blobs_vri $CI; wait
for C in $CC $CI; do NOLOCK=1 bin/job python research/detail/r48_swap40.py $MIX $C > logs/r48k2_swap40.log 2>&1; tail -n 1 logs/r48k2_swap40.log; done
ppl vrc_iq212 $CC; ppl vri_iq212 $CI
rm -f $CI; echo "vri carrier deleted"
NOLOCK=1 bin/job python research/detail/r48k_swapB.py $W/blobs_vrcB $CC > logs/r48k2_swapB.log 2>&1; tail -n 1 logs/r48k2_swapB.log
ppl vrcB_iq212 $CC
ver vrcB iq212 $W/blobs_vrcBm $CC
rm -f $CC; date +"D5 %T done, carriers deleted"
