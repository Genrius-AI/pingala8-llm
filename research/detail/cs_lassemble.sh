# cs_lassemble.sh (2026-10-18): cs_assemble.sh with ALL 32 layers swapped in (hard links, no extra disk), for cs_lfull.sh; log logs/cs_lfull.log. Was: cs_assemble.sh (T1 follow-up
# cs_assemble.sh (2026-10-18, T1 follow-up of research/work/r60/prereg_cscale_T1.txt): copy of l8fix_assemble.sh for one T1 arm. Blob dir =
# hard links to the pre-registered Llama build's 225 blobs with the 14 layer-0/1 blobs of work/cs/ARM swapped in, merged stats, transcode
# bit-check, clamp delta, r51g_native.py plan/dryrun/build/verify (unchanged), then the pre-registered gate: 4-window strided smoke
# (r36 command line, --chunks 4), pass = window [2] < 20; only on pass the full 70-window r36_hfj run and paired tests vs IQ2_XXS and IQ2_KT.
# usage: cs_assemble.sh ARM   log logs/cs_t1.log. Plain ASCII.
cd "$(dirname "$0")/../.." || exit 1; source .venv/bin/activate; A=$1; AD=work/cs/$A
SRC=work/llama8b/build; REP=work/cs/rep_$A; BD=$REP/blobs_vrc; MIX=work/llama8b/l31-8b-IQ2_XXS.gguf; DEL=$REP/delta; NATD=work/cs/native_$A
STEM=$NATD/l31-8b-pingala8-flat-vrnative-cs$A; L=logs/cs_lfull.log; T="cs_$A"
NAMES=$(for l in $(seq 0 31); do for k in attn_q attn_k attn_v attn_output ffn_gate ffn_up ffn_down; do printf 'blk.%s.%s.weight ' $l $k; done; done)
say() { echo "$(date '+%F %T') $*" | tee -a $L; }; die() { say "BLOCKER $A $*"; exit 5; }; freeg() { df -BM --output=avail /mnt/c | tail -1 | tr -dc 0-9; }
while [ -e logs/commit_low ]; do sleep 60; done
rm -rf $REP; mkdir -p $BD; cp -al $SRC/blobs_vrc/. $BD/ || die "hard-link copy failed"
for nm in $NAMES; do
  cmp -s $AD/blobs_vrc/__vrk_tables__.blob $SRC/blobs_vrc/__vrk_tables__.blob || die "tables blob differs"
  [ -s $AD/blobs_vrc/$nm.blob ] || die "missing $nm"; rm -f $BD/$nm.blob; ln $AD/blobs_vrc/$nm.blob $BD/$nm.blob
  [ "$(stat -c %s $BD/$nm.blob)" -le "$(stat -c %s $SRC/blobs_vrc/$nm.blob)" ] || say "NOTE $nm larger than shipped ($(stat -c %s $BD/$nm.blob) vs $(stat -c %s $SRC/blobs_vrc/$nm.blob))"
done
[ "$(ls $BD/*.blob | wc -l)" -eq 225 ] || die "blob count"
python3 - <<PY || die "stats merge failed"
import json, os
names = set("$NAMES".split()); st = json.load(open("$SRC/stats_vrk.json")); new = {x["name"]: x for x in json.load(open("$AD/stats.json")) if x["v"] == "vrc"}; out = []
for s in st:
    if s["name"] in names and s["v"] == "vrc": n = dict(new[s["name"]]); n["posthoc"] = "cs T1 arm $A"; out.append(n)
    else: out.append(s)
assert len(out) == len(st); json.dump(out, open("$REP/stats_vrk.json", "w")); print("merged", len(out))
PY
CPU=1 bin/job env R48K_NS=8 python research/detail/r51_transcode.py test $BD $NAMES > logs/${T}_transcode.log 2>&1
grep -aq "^RESULT ALL PASS (224 tensors)" logs/${T}_transcode.log || die "transcode not ALL PASS"
mkdir -p $DEL; CPU=1 bin/job bash -c "NT=4 R48K_NS=8 R51D_OUT=vr_delta.npz python research/detail/r51_delta.py $BD $DEL > logs/${T}_delta.log 2>&1"
[ -s $DEL/vr_delta.npz ] || die "delta npz missing"; say "DELTA $A $(tail -1 logs/${T}_delta.log | cut -c1-160)"
export G_BD=$BD G_MIX=$MIX G_NPZ=$DEL/vr_delta.npz; mkdir -p $NATD
CPU=1 bin/job bash -c "R48K_NS=8 python research/detail/r51g_native.py plan $STEM > logs/${T}_plan.log 2>&1"; say "$(grep -a '^PLAN TOTAL' logs/${T}_plan.log | cut -c1-200)"
CPU=1 bin/job bash -c "R48K_NS=8 python research/detail/r51g_native.py dryrun $NATD/dry > logs/${T}_dry.log 2>&1"; rm -f $NATD/dry*
grep -aq 'DRYRUN readback PASS' logs/${T}_dry.log || die "dryrun not PASS"
[ "$(freeg)" -ge 14500 ] || die "C: free $(freeg) MB < 14500"
CPU=1 bin/job bash -c "R48K_NS=8 python research/detail/r51g_native.py build $STEM > logs/${T}_build.log 2>&1 && R48K_NS=8 python research/detail/r51g_native.py verify $STEM > logs/${T}_verify.log 2>&1"
say "NATIVE $A $(grep -a '^NATIVE TOTAL' logs/${T}_build.log | cut -c1-160) | $(grep -a '^VERIFY' logs/${T}_verify.log | cut -c1-160)"
grep -aq '^VERIFY.*PASS' logs/${T}_verify.log || die "verify not PASS"
NF=$STEM-00001-of-00003.gguf; NE="GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2"
bin/job bash -c "source bin/cuda_env.sh; env $NE research/kernels/llama_f44/build/bin/llama-perplexity -f datasets/wikitext2_test_hfjoin.raw -c 2049 --ppl-stride 4095 -ngl 99 --chunks 4 -m $NF > logs/${T}_smoke4.log 2>&1"
W=$(grep -a -o '\[[0-9]*\][0-9.]*' logs/${T}_smoke4.log | tr '\n' ' '); say "SMOKE4 $A $W (shipped: [1]11.19 [2]6490; pass [2] < 20)"
python3 -c "import re,sys; v=dict(re.findall(r'\[(\d+)\]([0-9.]+)', '$W')); sys.exit(0 if '2' in v and float(v['2']) < 20 else 1)" || { say "GATE FAIL $A"; exit 0; }
say "GATE PASS $A -> 70-window r36_hfj"
bin/job env $NE HFJ_BIN=research/kernels/llama_f44/build/bin bash research/detail/r36_hfj.sh $T $NF >> logs/${T}_hfj.out 2>&1
F=logs/r36_hfj_$T.log; say "HFJ $A $(grep -a -o '\[70\][0-9.]*' $F | tail -1) (IQ2_XXS 12.3500, IQ2_KT 10.0375)"
for B in l8_iq2 ktllama_LK1 cs_a6; do say "PAIRED $A vs $B: $(python3 research/detail/r59_paired70.py $F logs/r36_hfj_$B.log 2>&1 | tail -1) | $(python3 research/detail/r59_acorr70.py $F logs/r36_hfj_$B.log 2>&1 | tail -1)"; done
