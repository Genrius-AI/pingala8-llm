#!/bin/bash
# l8_build.sh: Llama-3.1-8B Model B build chain, runs in the SECOND GPU slot (bin/gjob2) + CPU lock, beside the main-lock queue. Plain ASCII.
#   0 wait download marker  1 convert HF bf16 -> Q8_0 (no f16 gguf)  2 imatrix (Q8_0, calib_c4 -c 512 --chunks 200; partial offload, ~4 GB VRAM)
#   3 IQ2_XXS from the Q8_0 + imatrix  4 propagation gate (closed-stack loss) + smoke (layers 0/15/31)  5 Model B build (r48l_build.py, flat mode, r48k3 tables)
#   6 transcode test, clamp delta, native plan/dryrun/build/verify (CPU lock)   -> logs/l8_build.done  (the eval driver research/detail/l8_drive.sh waits on it)
# Idempotent (finished stages are skipped). Never waits on pgrep -f; waits on log markers / files only. Log: logs/l8_build.log
cd "$(dirname "$0")/../.." || exit 1; export HF_HUB_OFFLINE=1; source bin/cuda_env.sh
W=work/llama8b; Q8=$W/l31-8b-Q8_0.gguf; MIX=$W/l31-8b-IQ2_XXS.gguf; IMX=models/llama31-8b.imatrix; C=$W/l31-8b-vrc.gguf; BLD=$W/build; BD=$BLD/blobs_vrc
NATD=$W/native; STEM=$NATD/l31-8b-pingala8-flat-vrnative; DEL=research/work/llama8b; SB=llama.cpp/build-cuda/bin; LOG=logs/l8_build.log
PRE="source $PWD/.venv/bin/activate; export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VEDIC_GPU_FRAC=0.40;"
SHIFTS=0,0,0,0,2,3,2,4/0,0,0,0,0,0,0,2
mkdir -p $W $NATD $DEL logs/run_1017
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
freeg() { df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9; }
need_c() { for i in $(seq 1 60); do [ "$(freeg)" -ge "$1" ] && return 0; sleep 120; done; say "BLOCKER C: free $(freeg) GB < $1 GB for 2 h"; exit 3; }
dropc() { python3 research/detail/l8_dropcache.py "$@" > /dev/null 2>&1; }
die() { say "BLOCKER $*"; exit 5; }
SLICE=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/vedic.slice
anon_mb() { awk '/^anon /{print int($2/1048576)}' $SLICE/memory.stat 2>/dev/null || echo 0; }
# gjob2 gates on MemoryCurrent (includes reclaimable cache, ~11-12 GB here), so it is bypassed (RAMCAP_MB=99999) and the REAL anonymous memory of the slice is gated here
agate() { for i in $(seq 1 480); do [ $(( $(anon_mb) + $1 )) -le 10800 ] && return 0; sleep 30; done; say "WARN anon gate timeout (anon $(anon_mb) MB + $1)"; return 1; }
GJ() { local v=$1 r=$2; shift 2; dropc; agate $r; env VRAM_MB=$v RAM_MB=0 RAMCAP_MB=99999 bin/gjob2 "$@"; }
say "l8_build start sid $(ps -o sid= -p $$ | tr -d ' ') C: free $(freeg) GB"

# 0. download
until grep -aq '^L8_DL_DONE' logs/l8_download.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d hf_cache/hub/models--unsloth--Meta-Llama-3.1-8B/snapshots/*); say "download done: $(du -shL $SNAP | cut -f1) in $SNAP"

# 1. convert to Q8_0 directly (CPU lock)
if [ ! -s $Q8 ] || ! grep -aq 'Writing:.*100%' logs/l8_convert.log 2>/dev/null; then
  need_c 30; rm -f $Q8
  CPU=1 bin/job python llama.cpp/convert_hf_to_gguf.py $SNAP --outtype q8_0 --outfile $Q8 > logs/l8_convert.log 2>&1; say "CONVERT rc=$? bytes $(stat -c %s $Q8 2>/dev/null)"
  dropc $Q8 hf_cache/hub/models--unsloth--Meta-Llama-3.1-8B/blobs
fi
[ "$(stat -c %s $Q8 2>/dev/null || echo 0)" -gt 8000000000 ] || die "Q8_0 missing or too small (see logs/l8_convert.log)"
[ -s $Q8.sha256 ] || sha256sum $Q8 | cut -c1-64 > $Q8.sha256; say "Q8_0 sha256 $(cat $Q8.sha256) bytes $(stat -c %s $Q8)"; dropc $Q8

# 2. propagation gate (needs no imatrix), then imatrix; both in the second slot, one after the other
for i in $(seq 1 20); do grep -aqE '^GATE llama|Traceback' logs/l8_gate.log 2>/dev/null && break; [ -s logs/l8_gate.log ] && [ $(( $(date +%s) - $(stat -c %Y logs/l8_gate.log) )) -lt 60 ] && sleep 15 || break; done
if ! grep -aq '^GATE llama ppl.*PASS' logs/l8_gate.log 2>/dev/null; then
  GJ 3500 2600 bash -c "$PRE python research/detail/r48l_gate.py 4 > logs/l8_gate.log 2>&1"; say "GATE $(grep -a '^GATE' logs/l8_gate.log | tail -1)"
fi
grep -aq '^GATE llama ppl.*PASS' logs/l8_gate.log || die "propagation gate failed (logs/l8_gate.log): closed-stack loss not sane, builder front end (rope/mask/BOS) is wrong"
if [ ! -s $IMX ]; then
  GJ 4300 1500 $SB/llama-imatrix -m $Q8 -f models/calib_c4.txt -o $IMX -ngl 10 -c 512 --chunks 200 > logs/l8_imatrix.log 2>&1; say "IMATRIX rc=$? $(stat -c %s $IMX 2>/dev/null) B; $(grep -a -o 'Final estimate: PPL = [0-9.]*' logs/l8_imatrix.log | tail -1)"
  dropc $Q8
fi
[ -s $IMX ] || die "imatrix missing (logs/l8_imatrix.log)"

# 3. IQ2_XXS (stock recipe, from the Q8_0 reference, allow-requantize)
if [ ! -s $MIX ]; then
  CPU=1 bin/job $SB/llama-quantize --allow-requantize --imatrix $IMX $Q8 $MIX IQ2_XXS 8 > logs/l8_mix.log 2>&1; say "MIX rc=$? bytes $(stat -c %s $MIX 2>/dev/null)"
  dropc $Q8 $MIX
fi
[ -s $MIX ] || die "IQ2_XXS missing (logs/l8_mix.log)"
[ -s $MIX.sha256 ] || sha256sum $MIX | cut -c1-64 > $MIX.sha256; say "IQ2_XXS sha256 $(cat $MIX.sha256) bytes $(stat -c %s $MIX)"
# 3b. a-priori trim list (class 3 -> 2 on 3 tensors, rule in l8_trim.py) so that the flat model sits at the 0.993 x IQ2_XXS tensor budget
TRIMF=$DEL/flat_trim.json; export FLATTRIM_FILE=$TRIMF
[ -s $TRIMF ] || { CPU=1 bin/job python research/detail/l8_trim.py $MIX $IMX $TRIMF > logs/l8_trim.log 2>&1; say "TRIM $(tail -1 logs/l8_trim.log | cut -c1-250)"; }
[ -s $TRIMF ] || die "trim list missing (logs/l8_trim.log)"

# 4. smoke: layers 0/15/31, attn_q (row permutation vs Q8_0), attn_v (Q4_K slot, class 6), ffn_down (n = 14336, Q2_K slot at L0, in-place LAPACK chain)
if ! grep -aq '^GATES' logs/l8_smoke.tensors 2>/dev/null; then
  rm -rf $W/smoke $W/smoke_xs
  GJ 5200 5500 bash -c "$PRE IMATRIX=$IMX Q8REF=$Q8 GPUFRAC=0.40 R48K_SHIFTS=$SHIFTS ARMS=vrc:8:research/work/r48k3:flat FLATTRIM_FILE=$TRIMF ENC_LAYERS=0,4,15,31 KINDS=attn_q,attn_v,ffn_down NSEQ=8 NEV=1 DAMP=0.1 ALPHA=1.05 NSAL=4 XDIR=$W/smoke_xs python research/detail/r48l_build.py $MIX $W/smoke > logs/l8_smoke.tensors 2>&1"
  say "SMOKE rc=$? tensors $(grep -a -c '^vrc ' logs/l8_smoke.tensors)"
fi
grep -aq '^GATES' logs/l8_smoke.tensors && [ "$(grep -a -c '^vrc ' logs/l8_smoke.tensors)" -eq 12 ] || die "smoke failed (logs/l8_smoke.tensors): $(grep -a -E 'ABORT|Error|assert' logs/l8_smoke.tensors | tail -2)"
say "SMOKE ok: $(grep -a '^SUMMARY' logs/l8_smoke.tensors | cut -c1-200)"; rm -rf $W/smoke $W/smoke_xs

# 5. the build (flat mode), carrier = copy of the Q8_0 reference, every linear and non-linear tensor overwritten.
#    Up to 3 attempts; a killed attempt (memguard / CUDA error) is resumed (RESUME=1: finished layers are only re-propagated, ~40 s each) instead of restarted.
build_ok() { grep -aq '^GATES' logs/l8_build.tensors 2>/dev/null; }
BP=logs/l8_build.pid
if ! build_ok && [ -s $BP ] && kill -0 "$(cat $BP)" 2>/dev/null; then       # a builder started by an earlier version of this script: wait on its recorded pid
  say "waiting for running builder pid $(cat $BP)"; while kill -0 "$(cat $BP)" 2>/dev/null; do sleep 30; done
fi
for att in 1 2 3; do
  build_ok && break
  if [ -s $BLD/stats.json ] && [ -f $C ]; then RES=1; else RES=""; need_c 34; rm -rf $BLD; mkdir -p $BLD; cp $Q8 $C; rm -f logs/l8_build.tensors; fi
  dropc $C $Q8; say "BUILD attempt $att resume=${RES:-0} carrier $C"
  GJ 5200 5500 bash -c "echo \$\$ > $BP; $PRE RESUME=$RES IMATRIX=$IMX Q8REF=$Q8 GPUFRAC=0.40 R48K_SHIFTS=$SHIFTS FLATTRIM_FILE=$TRIMF ARMS=vrc:8:research/work/r48k3:flat:$C NSEQ=512 NEV=32 DAMP=0.1 ALPHA=1.05 NSAL=4 LB=2 exec python research/detail/r48l_build.py $MIX $BLD >> logs/l8_build.tensors 2>&1"
  say "BUILD attempt $att rc=$? $(grep -a -E '^SUMMARY|^GATES|ABORT|Error|RESUME' logs/l8_build.tensors | tail -3 | cut -c1-220 | tr '\n' '|')"
  grep -aq 'ABORT' <(tail -5 logs/l8_build.tensors) && break
  sleep 20
done
grep -aq '^GATES' logs/l8_build.tensors || die "build did not finish (logs/l8_build.tensors)"
[ "$(ls $BD/*.blob | wc -l)" -eq 225 ] || die "blob count $(ls $BD/*.blob | wc -l) != 224 + tables"
say "carrier bytes $(stat -c %s $C)"; dropc $C

# 6. CPU stage: transcode bit-check (blk 0/17/31), clamp delta, native file plan/dryrun/build/verify
export G_BD=$BD G_MIX=$MIX G_NPZ=$DEL/vr_delta_l8.npz
if ! grep -aq 'PASS' logs/l8_transcode.log 2>/dev/null; then
  CPU=1 bin/job env R48K_NS=8 python research/detail/r51_transcode.py test $BD blk.0.attn_q.weight blk.0.attn_v.weight blk.17.ffn_down.weight blk.17.attn_output.weight blk.31.ffn_gate.weight blk.31.attn_k.weight > logs/l8_transcode.log 2>&1
  say "TRANSCODE PASS lines $(grep -a -c PASS logs/l8_transcode.log)"
fi
[ "$(grep -a -c PASS logs/l8_transcode.log)" -ge 6 ] || say "WARN transcode PASS lines < 6 (logs/l8_transcode.log); continuing, native verify decides"
if [ ! -s $G_NPZ ]; then
  CPU=1 bin/job bash -c "NT=4 R48K_NS=8 R51D_OUT=vr_delta_l8.npz python research/detail/r51_delta.py $BD $DEL > logs/l8_delta.log 2>&1"; say "DELTA $(tail -1 logs/l8_delta.log | cut -c1-200)"
fi
[ -s $G_NPZ ] || die "delta npz missing (logs/l8_delta.log)"
CPU=1 bin/job bash -c "R48K_NS=8 python research/detail/r51g_native.py plan $STEM > logs/l8_native_plan.log 2>&1"; say "$(grep -a '^PLAN TOTAL' logs/l8_native_plan.log | cut -c1-260)"
rm -f $STEM-0000*.gguf
CPU=1 bin/job bash -c "R48K_NS=8 python research/detail/r51g_native.py dryrun $NATD/l8-dry > logs/l8_native_dry.log 2>&1"; say "$(grep -a 'DRYRUN readback P\|DRYRUN readback F' logs/l8_native_dry.log)"; rm -f $NATD/l8-dry*
need_c 14
CPU=1 bin/job bash -c "R48K_NS=8 python research/detail/r51g_native.py build $STEM > logs/l8_native_build.log 2>&1 && R48K_NS=8 python research/detail/r51g_native.py verify $STEM > logs/l8_native_verify.log 2>&1"
say "NATIVE $(grep -a '^NATIVE TOTAL' logs/l8_native_build.log | cut -c1-200) | $(grep -a '^VERIFY' logs/l8_native_verify.log | cut -c1-240)"
grep -aq '^VERIFY.*PASS' logs/l8_native_verify.log || die "native verify not PASS (logs/l8_native_verify.log)"
grep -a '^SHARD ' logs/l8_native_build.log | awk '{print $6, $2}' > logs/l8_native_sha256.txt
echo "L8 BUILD DONE $(date '+%F %T') native $(grep -a '^NATIVE TOTAL' logs/l8_native_build.log | cut -c1-120)" > logs/l8_build.done; cat logs/l8_build.done | tee -a $LOG
dropc $W
