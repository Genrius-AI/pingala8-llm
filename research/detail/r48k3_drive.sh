#!/bin/bash
# R48k3: VRii-C 8-state (Pingala gana states, Ungerboeck ff labels z1 = u^s0^s2, z0 = s1; r49_vr.cu NS=108), layout all, same tiles/alloc/fixes
# as r48k2 vrc (shift fix, salient floor, no boost on the all layout). Build -> container -> LLVQ ppl -> verify -> delete carrier.
cd ~/repo; export HF_HUB_OFFLINE=1; W=work/r48k3; MIX=work/r45b/4B-IQ2_XXS-r45b.gguf; C=$W/4B-r48k-vrc8.gguf
date +"E0 %T"
bin/job bash -c "GPUCHOL_MAXN=10000 GPUFRAC=0.84 R48K_NS=8 ONLY=vrc TABDIR=research/work/r48k3 R48K_SHIFTS=0,0,0,0,2,3,2,4/0,0,0,0,0,0,0,2 NSEQ=512 NEV=32 DAMP=0.1 ALPHA=1.05 NSAL=4 python research/detail/r48k_build.py $MIX $C $C $W > logs/r48k3.tensors 2>&1"; echo "E1 $(date +%T) build rc=$?"
grep -E '^SUMMARY|^GATES|ABORT|Error' logs/r48k3.tensors | tail -4; grep -q '^GATES' logs/r48k3.tensors || exit 5
echo "PPL_MODEL -m $C sha256 $(sha256sum $C | cut -c1-64) variant vrc8_all" | tee -a logs/r48k3_ppl.log
( NOLOCK=1 bin/job bash -c "OMP_NUM_THREADS=4 python research/detail/r48k_container.py encode $W/blobs_vrc $MIX $W/r48k3_vrc8_all.vedq all > logs/r48k3_realfile.log 2>&1"; grep REAL_BYTES logs/r48k3_realfile.log ) &
bin/job bash research/detail/r36_hfj.sh r48k3_vrc8_all $C | tee -a logs/r48k3_ppl.log; date +"E2 %T ppl" | tee -a logs/r48k3_ppl.log
wait; NOLOCK=1 bin/job bash -c "NT=4 OMP_NUM_THREADS=4 python research/detail/r48k_container.py verify $W/r48k3_vrc8_all.vedq $W/blobs_vrc $C $MIX all >> logs/r48k3_realfile.log 2>&1"; tail -n 1 logs/r48k3_realfile.log
rm -f $C; date +"E3 %T done, carrier deleted"
