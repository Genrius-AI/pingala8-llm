#!/bin/bash
# R48k2 post-build (time-ordered for the 10:00 stop): ppl first (GPU), containers + verify in parallel on CPU.
cd ~/repo; export HF_HUB_OFFLINE=1; W=work/r48k2; MIX=work/r45b/4B-IQ2_XXS-r45b.gguf; CI=$W/4B-r48k-vri.gguf; CC=$W/4B-r48k-vrc.gguf
until grep -qE '^GATES|ABORT|Traceback' logs/r48k2.tensors 2>/dev/null; do sleep 20; done; date +"Q0 %T build done"
grep -E '^SUMMARY|^BOOST|^GATES|ABORT|Error' logs/r48k2.tensors | tail -8; grep -q '^GATES' logs/r48k2.tensors || exit 5
cont() { F=$1; V=$2
  if [ $F = vrcB ]; then rm -rf $W/blobs_vrcBm; mkdir -p $W/blobs_vrcBm; cp $W/blobs_vrc/*.blob $W/blobs_vrcBm/; cp $W/blobs_vrcB/*.blob $W/blobs_vrcBm/; BD=$W/blobs_vrcBm; else BD=$W/blobs_$F; fi
  NOLOCK=1 bin/job bash -c "OMP_NUM_THREADS=4 python research/detail/r48k_container.py encode $BD $MIX $W/r48k2_${F}_${V}.vedq $V > logs/r48k2_realfile_${F}_$V.log 2>&1"; grep REAL_BYTES logs/r48k2_realfile_${F}_$V.log; }
ppl() { echo "PPL_MODEL -m $2 sha256 $(sha256sum $2 | cut -c1-64) variant $1" | tee -a logs/r48k2_ppl.log
  bin/job bash research/detail/r36_hfj.sh r48k2_$1 $2 | tee -a logs/r48k2_ppl.log; date +"P %T $1" | tee -a logs/r48k2_ppl.log; }
ver() { NOLOCK=1 bin/job bash -c "NT=4 OMP_NUM_THREADS=4 python research/detail/r48k_container.py verify $W/r48k2_$1_$2.vedq $3 $4 $MIX $2 >> logs/r48k2_realfile_$1_$2.log 2>&1"; tail -n 1 logs/r48k2_realfile_$1_$2.log; }
( cont vrc all; cont vri all; cont vrc iq212; cont vri iq212; cont vrcB iq212; ver vri all $W/blobs_vri $CI ) > logs/r48k2_cont.log 2>&1 &
CP=$!
ppl vrc_all $CC; ppl vri_all $CI
NOLOCK=1 bin/job python research/detail/r48_swap40.py $MIX $CC > logs/r48k2_swap40.log 2>&1; tail -n 1 logs/r48k2_swap40.log
ppl vrc_iq212 $CC
NOLOCK=1 bin/job python research/detail/r48k_swapB.py $W/blobs_vrcB $CC > logs/r48k2_swapB.log 2>&1; tail -n 1 logs/r48k2_swapB.log
ppl vrcB_iq212 $CC
rm -f $CC; echo "vrc carrier deleted"
wait $CP; date +"Q1 %T containers + vri verify done"
NOLOCK=1 bin/job python research/detail/r48_swap40.py $MIX $CI > logs/r48k2_swap40i.log 2>&1
ppl vri_iq212 $CI
rm -f $CI $CC; date +"Q5 %T done, carriers deleted"
