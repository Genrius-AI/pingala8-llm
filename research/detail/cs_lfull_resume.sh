#!/bin/bash
# cs_lfull_resume.sh (2026-10-18 14:20): resume of cs_lfull.sh after slice_guard stopped vedic.slice at 14:12 (Windows free commit 1 GB).
# Same settings as cs_lfull.sh plus RESUME=1 (cs_build_l.py keeps stats.json/blobs of finished layers, replays them, encodes the rest);
# no rm -rf of OUT. Then the same blob check and cs_lassemble.sh lfull. Log logs/cs_lfull.log, per tensor logs/cs_lfull.tensors (appended). Plain ASCII.
cd "$(dirname "$0")/../.." || exit 1; L=logs/cs_lfull.log; say() { echo "$(date '+%F %T') $*" | tee -a $L; }
OUT=work/cs/lfull; LOG=logs/cs_lfull.tensors; NSEQ=${NSEQ:-256}; NEV=${NEV:-32}
say "resume pid $$ NSEQ $NSEQ NEV $NEV stats $(python3 -c "import json;print(len(json.load(open('$OUT/stats.json'))))")"
while [ -e logs/commit_low ]; do sleep 60; done
cfree() { df -BM --output=avail /mnt/c | tail -1 | tr -dc 0-9; }
[ "$(cfree)" -ge 12500 ] || { say "BLOCKER C: free $(cfree) MB < 12500"; exit 3; }
( while sleep 30; do f=$(cfree); [ "$f" -lt 12000 ] && { say "BLOCKER C: free $f MB < 12000 during build: stopping it"; [ -n "$BP" ] && kill $BP; break; }
  [ -z "$BP" ] && BP=$(pgrep -u $(id -u) -xf "python research/detail/cs_build_l.py work/llama8b/l31-8b-IQ2_XXS.gguf $OUT" | head -1); done ) & WG=$!   # BP = this build's own PID
echo "=== RESUME $(date '+%F %T')" >> $LOG
bin/job bash -c "source bin/cuda_env.sh; source .venv/bin/activate; export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VEDIC_GPU_FRAC=0.75; \
 RESUME=1 CMODE=opt DMODE=mean HFIT=all MIXBOS=1 IMATRIX=models/llama31-8b.imatrix Q8REF=work/llama8b/l31-8b-Q8_0.gguf GPUFRAC=0.52 R48K_SHIFTS=0,0,0,0,2,3,2,4/0,0,0,0,0,0,0,2 \
 FLATTRIM_FILE=research/work/llama8b/flat_trim.json ARMS=vrc:8:research/work/r48k3:flat NSEQ=$NSEQ NEV=$NEV DAMP=0.1 ALPHA=1.05 NSAL=4 LB=2 \
 exec python research/detail/cs_build_l.py work/llama8b/l31-8b-IQ2_XXS.gguf $OUT >> $LOG 2>&1"
rc=$?; kill $WG 2>/dev/null; rm -rf $OUT/xs $OUT/hspill* 2>/dev/null
say "BUILD rc=$rc $(grep -a '^SUMMARY' $LOG | tail -1 | cut -c1-200)"
[ $rc -eq 0 ] && [ "$(ls $OUT/blobs_vrc/*.blob 2>/dev/null | wc -l)" -eq 225 ] || { say "BLOCKER build rc=$rc or blob count"; exit 4; }
bash research/detail/cs_lassemble.sh lfull
say "end"
