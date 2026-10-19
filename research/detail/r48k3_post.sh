#!/bin/bash
# R48k3 post: wait build -> relayout blobs to kernel order -> container -> LLVQ ppl -> verify -> delete carrier.
cd ~/repo; export HF_HUB_OFFLINE=1; W=work/r48k3; MIX=work/r45b/4B-IQ2_XXS-r45b.gguf; C=$W/4B-r48k-vrc8.gguf
until grep -qE '^GATES|ABORT|Traceback' logs/r48k3.tensors 2>/dev/null; do sleep 20; done; date +"F0 %T build done"
grep -E '^SUMMARY|^GATES|ABORT|Error' logs/r48k3.tensors | tail -4; grep -q '^GATES' logs/r48k3.tensors || exit 5
NOLOCK=1 bin/job python research/detail/r48k_relayout.py $W/blobs_vrc
echo "PPL_MODEL -m $C sha256 $(sha256sum $C | cut -c1-64) variant vrc8_all" | tee -a logs/r48k3_ppl.log
( NOLOCK=1 bin/job bash -c "OMP_NUM_THREADS=4 python research/detail/r48k_container.py encode $W/blobs_vrc $MIX $W/r48k3_vrc8_all.vedq all > logs/r48k3_realfile.log 2>&1"; grep REAL_BYTES logs/r48k3_realfile.log ) &
bin/job bash research/detail/r36_hfj.sh r48k3_vrc8_all $C | tee -a logs/r48k3_ppl.log; date +"F1 %T ppl" | tee -a logs/r48k3_ppl.log
wait; NOLOCK=1 bin/job bash -c "NT=4 OMP_NUM_THREADS=4 python research/detail/r48k_container.py verify $W/r48k3_vrc8_all.vedq $W/blobs_vrc $C $MIX all >> logs/r48k3_realfile.log 2>&1"; tail -n 2 logs/r48k3_realfile.log
rm -f $C; date +"F2 %T done, carrier deleted"
