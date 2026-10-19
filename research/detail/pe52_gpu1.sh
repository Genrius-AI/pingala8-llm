#!/bin/bash
# pe52 GPU queue part 1 (files that already exist). Each step is its own bin/job.
cd ~/repo; D=research/detail; N=work/r51t/native/4B-pingala8-vrnative-00001-of-00002.gguf; X=work/r45b/4B-IQ2_XXS-r45b.gguf
NATIVE=1 bash $D/pe52_zs.sh modelB_native $N
bash $D/pe52_zs.sh iq2xxs $X mmluF
for q in IQ2_M IQ3_XXS Q2_K; do bin/job bash $D/r36_hfj.sh pe52_$q models/qwen3-4B-$q.gguf; done
echo "GPU1_END $(date +%T)"
