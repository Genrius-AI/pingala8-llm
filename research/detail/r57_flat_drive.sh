#!/bin/bash
# R57 size-matched flat Model B ("flattrim", work/r57/flat_trim_plan.txt, prereg work/r57/prereg_flat.txt). NOT EXECUTED YET. Runs on the main GPU lock (bin/job, GJ2 unset)
# ALONE: do not start while another GPU job holds it. Build ~4,600 s (CALC r48k full 4,623 s) + shared prop/Hessians ~2,500 s + 8 min ppl => ~2-2.5 h; C: need 22 GB; VRAM <= 5 GB.
cd ~/repo; export HF_HUB_OFFLINE=1; unset GJ2
[ -f .gpu_pause ] && { echo "gpu_pause set"; exit 3; }
python3 -c "import json;d=json.load(open('research/work/r57/flattrim_cls.json'));assert len(d['cls'])==252 and len(d['moved'])==4 and len(d['sal'])==5" || exit 4
# 1. build + ppl (r52a_full.sh: F0/F1 lines, GATES check, PPL_MODEL line, bin/job r36_hfj r52a_flat, deletes carrier after rc=0)
research/detail/r52a_full.sh f4 flat:8:research/work/r48k3:flattrim
# 2. byte + gate verdict (must satisfy prereg checks)
python3 - <<'PY' | tee -a logs/r57_flat_verdict.log
import json
S=[s for s in json.load(open('work/r52a/f4/stats.json')) if s['v']=='flat']; b=sum(s['bytes'] for s in S)
print('R57 flat n',len(S),'linear blob bytes',b,'shipped 966271936 ok',b<=966271936,'roff 972043712 diff',b-972043712)
print('gate',sum(s['maxratio']>=0.9 for s in S),'/',len(S),'min',min(s['maxratio'] for s in S))
for n in json.load(open('research/work/r57/flattrim_cls.json'))['moved']:
    s=[x for x in S if x['name']==n][0]; print('moved',n,s['hist'],'Dev',round(s['Dev'],6))
print('sum Dev',sum(s['Dev'] for s in S),'(roff 6.2337, mirror 5.4999)')
PY
grep -a "HFJ r52a_flat" logs/r36_hfj_r52a_flat.log logs/r52a_ppl.log 2>/dev/null | tail -2
# FOLLOW-UPS (manual, after ppl vs prereg; each needs the main GPU lock, one at a time):
#  a. native file: parameterise detail/r51_native.py (BD=work/r52a/f4/blobs_flat, NPZ from detail/r51_delta.py run on blobs_flat, stats path, BUDGET=1246621312;
#     lines 16-17,27), then `r51_native.py plan/dryrun/build work/r57/flat/4B-flat` (VRK2 relayout is already in the build; class 2 is a kernel-supported class).
#  b. KLD vs Q8_0 base: same harness as Model B 0.640 (grep kld in research/detail/*.sh; Q8_0 base logits), report flat vs shipped vs IQ2_KT-eq 0.785 vs IQ2_XXS 1.220.
#  c. zero-shot: reuse logs/r50_zs harness (r36_zs_r50_vrc8_hs.log pattern), 4 tasks with CIs, flat vs shipped.
#  d. paired bench: llama-bench pp512/tg128 flat vs IQ2_KT-eq vs IQ2_XXS (kernel unchanged => expect tg 1.00x, pp 0.70x of IQ2_XXS).
#  e. update research/MODEL_B_PROGRESS.md scoreboard + paper numbers (main.tex 41,72,616,639-640,690,718 per reviewer_r56) ; if flat <= 17.99 ship as Model B'.
