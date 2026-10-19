#!/bin/bash
# r59 final zero-shot + Llama order (18:45; RAM_MB 4500: the slice count includes ~3 GB reclaimable file cache): B3 full MMLU on the free second slot (gjob2, RAM-gated) while C4 holds bin/job,
# then relaunch the Llama-8B build (resume; it takes gjob2), then IQ2_KT at B3's bytes zero-shot on bin/job beside it. 2026-10-17.
cd ~/repo; L=logs/r59_zs_g2.log; say(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a $L; }
say "final chain start"
RUN="env VRAM_MB=3500 RAM_MB=4500 bin/gjob2" NATIVE=1 bash research/detail/r59_zs.sh b3 work/r59b3/native/4B-b3-vrnative-00001-of-00002.gguf "mmluF" >> logs/r59_zs_b3.log 2>&1
say "B3 mmluF rc=$? $(grep -a 'Final result' logs/pe52/zs_b3_mmluF.log | tail -1)"
setsid nohup bash research/detail/l8_build.sh >> logs/l8_build.out 2>&1 < /dev/null & say "l8_build relaunched pid $!"
ZSBIN=ext/ik_llama.cpp/build-cuda/bin bash research/detail/r59_zs.sh ktm work/r59kt/ktm_keep.gguf "wg arc-easy arc-challenge mmluF hs" > logs/r59_zs_ktm.log 2>&1; say "KTM zs rc=$?"
say "end"
