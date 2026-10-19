#!/bin/bash
# cs_lfull.sh (2026-10-18): full 32-layer Llama-3.1-8B rebuild with the column scale that won on Qwen3-4B (CMODE=opt, research/work/r60/prereg_cscale_qfull.txt)
# plus MIXBOS, otherwise the a6 settings of cs_t1b.sh; then cs_lassemble.sh lfull (native, smoke gate, 70-window r36_hfj, paired vs IQ2_XXS, IQ2_KT, a6).
# Prereg research/work/r60/prereg_cscale_lfull.txt. Disk: hidden states (NSEQ+NEV) x 2048 x 4096 fp16 + ffn_down Hessian spills + blobs must leave
# C: >= 12 GB. Log logs/cs_lfull.log, per tensor logs/cs_lfull.tensors. usage: NSEQ=.. NEV=.. cs_lfull.sh. Plain ASCII.
cd "$(dirname "$0")/../.." || exit 1; L=logs/cs_lfull.log; say() { echo "$(date '+%F %T') $*" | tee -a $L; }
OUT=work/cs/lfull; LOG=logs/cs_lfull.tensors; NSEQ=${NSEQ:-256}; NEV=${NEV:-32}
say "start pid $$ NSEQ $NSEQ NEV $NEV"
while [ -e logs/commit_low ]; do sleep 60; done
NEED=$(( (NSEQ + NEV) * 2048 * 4096 * 2 / 1048576 + 1700 + 1900 + 12000 ))
# effective free = C: free + slack inside the (sparse, untrimmed) WSL vhdx, which ext4 rewrites without growing the file on C:
VHDX=${VHDX:?set VHDX to the WSL2 ext4.vhdx path, or remove this WSL2 disk check on native Linux}
cfree() { df -BM --output=avail /mnt/c | tail -1 | tr -dc 0-9; }
FREE=$(cfree); SLACK=$(( $(stat -c %s $VHDX) / 1048576 - $(df -BM --output=used / | tail -1 | tr -dc 0-9) ))
say "disk C: free $FREE MB, vhdx slack $SLACK MB, need $NEED MB (incl. 12000 floor)"
[ $(( FREE + SLACK )) -ge "$NEED" ] && [ "$FREE" -ge 12500 ] || { say "BLOCKER C: free ${FREE} + slack ${SLACK} MB < need ${NEED} MB"; exit 3; }
( while sleep 30; do f=$(cfree); [ "$f" -lt 12000 ] && { say "BLOCKER C: free $f MB < 12000 during build: stopping it"; [ -n "$BP" ] && kill $BP; break; }
  [ -z "$BP" ] && BP=$(pgrep -u $(id -u) -xf "python research/detail/cs_build_l.py work/llama8b/l31-8b-IQ2_XXS.gguf $OUT" | head -1); done ) & WG=$!   # BP = this build's own PID
rm -rf $OUT; mkdir -p $OUT
bin/job bash -c "source bin/cuda_env.sh; source .venv/bin/activate; export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VEDIC_GPU_FRAC=0.75; \
 CMODE=opt DMODE=mean HFIT=all MIXBOS=1 IMATRIX=models/llama31-8b.imatrix Q8REF=work/llama8b/l31-8b-Q8_0.gguf GPUFRAC=0.75 R48K_SHIFTS=0,0,0,0,2,3,2,4/0,0,0,0,0,0,0,2 \
 FLATTRIM_FILE=research/work/llama8b/flat_trim.json ARMS=vrc:8:research/work/r48k3:flat NSEQ=$NSEQ NEV=$NEV DAMP=0.1 ALPHA=1.05 NSAL=4 LB=2 \
 exec python research/detail/cs_build_l.py work/llama8b/l31-8b-IQ2_XXS.gguf $OUT > $LOG 2>&1"
rc=$?; kill $WG 2>/dev/null; rm -rf $OUT/xs $OUT/hspill* 2>/dev/null
say "BUILD rc=$rc $(grep -a '^SUMMARY' $LOG | cut -c1-200)"
[ $rc -eq 0 ] && [ "$(ls $OUT/blobs_vrc/*.blob 2>/dev/null | wc -l)" -eq 225 ] || { say "BLOCKER build rc=$rc or blob count"; exit 4; }
bash research/detail/cs_lassemble.sh lfull
say "end"
