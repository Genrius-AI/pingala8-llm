#!/bin/bash
# R48k2 remaining evals (after Pingala-8): vri_all ppl -> vrc iq212 ppl -> vrcB (layers 0-3 boost) iq212 ppl -> delete carriers.
cd ~/repo; W=work/r48k2; MIX=work/r45b/4B-IQ2_XXS-r45b.gguf; CI=$W/4B-r48k-vri.gguf; CC=$W/4B-r48k-vrc.gguf
ppl() { echo "PPL_MODEL -m $2 sha256 $(sha256sum $2 | cut -c1-64) variant $1" | tee -a logs/r48k2_ppl.log
  bin/job bash research/detail/r36_hfj.sh r48k2_$1 $2 | tee -a logs/r48k2_ppl.log; date +"P %T $1" | tee -a logs/r48k2_ppl.log; }
ppl vri_all $CI; rm -f $CI
NOLOCK=1 bin/job python research/detail/r48_swap40.py $MIX $CC > logs/r48k2_swap40.log 2>&1; tail -n 1 logs/r48k2_swap40.log
ppl vrc_iq212 $CC
NOLOCK=1 bin/job python research/detail/r48k_swapB.py $W/blobs_vrcB $CC > logs/r48k2_swapB.log 2>&1; tail -n 1 logs/r48k2_swapB.log
ppl vrcB_iq212 $CC; rm -f $CC; date +"Z %T done, carriers deleted"
