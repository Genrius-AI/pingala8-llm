#!/bin/bash
# Recomputes, from the shipped logs only (CPU, a few seconds), the paired per-window statistics and the 72-chunk KL divergence
# quoted in the paper. Run from the archive root:  bash derived/recompute.sh > derived/recomputed.txt
# The output shipped as derived/recomputed.txt was produced by this script on the anonymized logs in this archive.
set -u
P="python3 research/detail/r59_paired.py"; A="python3 research/detail/r59_acorr.py"
W=logs/r36_hfj_r59_b3_native.log
echo "== WikiText-2 test, 73 windows of 4096 (first file minus second; negative = first better)"
echo "B3 vs IQ2_XXS           : $($P $W logs/r36_hfj_iq2xxs.log)"
echo "B3 vs IQ2_KT (B bytes)  : $($P $W logs/r56_kteq_ppl.log)"
echo "B3 vs IQ2_KT (B3 bytes) : $($P $W logs/r36_hfj_r59_ktm.log)"
echo "B3 vs Model B           : $($P $W logs/r36_hfj_r51_native_PPC2.log)"
echo "B2 vs IQ2_KT (B bytes)  : $($P logs/r36_hfj_r52a_flat.log logs/r56_kteq_ppl.log)"
echo "flat vs per-tile mirror : $($P logs/r36_hfj_r52a_roff.log logs/r36_hfj_r52a_mirror.log)"
echo "ik IQ2_XXS vs mainline  : $($P logs/r36_hfj_r59_ikxxs.log logs/r36_hfj_iq2xxs.log)"
echo "B3 vs IQ2_KT (B bytes)  autocorr: $($A $W logs/r56_kteq_ppl.log)"
echo "B3 vs IQ2_KT (B3 bytes) autocorr: $($A $W logs/r36_hfj_r59_ktm.log)"
echo "== C4 validation text, 73 windows of 4096 (logs/r59/c4_*.log; b2 = B3)"
C=logs/r59/c4_b2.log
echo "B3 vs IQ2_KT (B bytes)  : $($P $C logs/r59/c4_kt.log)"
echo "B3 vs IQ2_KT (B3 bytes) : $($P $C logs/r59/c4_ktm.log)"
echo "B3 vs Model B           : $($P $C logs/r59/c4_mB.log)"
echo "B3 vs IQ2_XXS           : not recomputable: logs/r59/c4_xxs.log was overwritten by an interrupted rerun (stops at window 33);"
echo "                          its 73-window result survives only as the summary line 'C4 xxs ... [73]37.2810' in logs/r59_kldc4.log"
echo "B3 vs IQ2_KT (B bytes)  autocorr: $($A $C logs/r59/c4_kt.log)"
echo "B3 vs IQ2_KT (B3 bytes) autocorr: $($A $C logs/r59/c4_ktm.log)"
echo "== KL divergence to Q8_0, 72 WikiText-2 chunks = mean of 6 parts x 12 chunks; s.e. = sqrt(sum se^2)/6 (logs/r59/kld_<arm>_p<k>.log)"
python3 - <<'EOF'
import re, math
for arm, name in (("b2", "B3"), ("kt", "IQ2_KT (Model B bytes)"), ("mB", "Model B"), ("xxs", "IQ2_XXS")):
    m, s = [], []
    for k in range(1, 7):
        t = open(f"logs/r59/kld_{arm}_p{k}.log", errors="ignore").read()
        x = re.search(r"Mean +KLD: +([0-9.]+) +\+/- +([0-9.]+)", t); m.append(float(x.group(1))); s.append(float(x.group(2)))
    print(f"{name:24s}: KLD72 {sum(m) / 6:.4f} +/- {math.sqrt(sum(v * v for v in s)) / 6:.4f}  parts {m}")
EOF
