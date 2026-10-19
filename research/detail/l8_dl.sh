#!/bin/bash
# Llama-3.1-8B download with retry (HF_HOME in hf_cache). Marker: L8_DL_DONE in logs/l8_download.log
export HF_HOME=$PWD/hf_cache HF_HUB_DISABLE_XET=1
for i in 1 2 3 4 5 6 7 8; do
  hf download unsloth/Meta-Llama-3.1-8B --max-workers 2 >> $PWD/logs/l8_download.log 2>&1 && { echo "L8_DL_DONE $(date +%T)" >> $PWD/logs/l8_download.log; exit 0; }
  echo "L8_DL_RETRY $i $(date +%T)" >> $PWD/logs/l8_download.log; sleep 10
done
