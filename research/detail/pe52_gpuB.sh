#!/bin/bash
# pe52 driver B (gjob2 second slot, VRAM_MB 5000): ppl of stock quants, IQ2_XXS full MMLU, Q8_0 + stock zero-shot.
cd ~/repo; source bin/cuda_env.sh; D=research/detail; W=work/pe52; G="env VRAM_MB=5000 bin/gjob2"
for f in models/qwen3-4B-IQ2_M models/qwen3-4B-IQ3_XXS models/qwen3-4B-Q2_K $W/qwen3-4B-IQ2_XS $W/qwen3-4B-IQ1_M $W/qwen3-4B-IQ2_S; do
  q=${f##*4B-}; $G bash $D/r36_hfj.sh pe52_$q $f.gguf; done
RUN="$G" bash $D/pe52_zs.sh iq2xxs work/r45b/4B-IQ2_XXS-r45b.gguf mmluF
RUN="$G" bash $D/pe52_zs.sh q8_0 $W/qwen3-4B-Q8_0.gguf
for q in IQ2_XS IQ1_M IQ2_S; do RUN="$G" bash $D/pe52_zs.sh $q $W/qwen3-4B-$q.gguf; done
for q in IQ2_M Q2_K IQ3_XXS; do RUN="$G" bash $D/pe52_zs.sh $q models/qwen3-4B-$q.gguf; done
echo "GPUB_END $(date +%T)"
