#!/bin/bash
# Pingala-8 one-click (Linux): ./run.sh qwen  or  ./run.sh llama
# Downloads the B4 model once, then serves a chat page at http://127.0.0.1:8080
set -e
cd "$(dirname "$0")"
case "${1:-qwen}" in
  qwen)  R=https://huggingface.co/Genrius-AI/Qwen3-4B-Pingala8-2bit/resolve/main
         F="4B-csq-vr3-00001-of-00002.gguf 4B-csq-vr3-00002-of-00002.gguf" ;;
  llama) R=https://huggingface.co/Genrius-AI/Llama-3.1-8B-Pingala8-2bit/resolve/main   # Built with Llama
         F="l31-8b-pingala8-flat-vrnative-cslfull-00001-of-00003.gguf l31-8b-pingala8-flat-vrnative-cslfull-00002-of-00003.gguf l31-8b-pingala8-flat-vrnative-cslfull-00003-of-00003.gguf" ;;
  *) echo "usage: ./run.sh qwen|llama"; exit 1 ;;
esac
mkdir -p models
for f in $F; do [ -s "models/$f" ] || { echo "Downloading $f ..."; curl -L --fail -o "models/$f" "$R/$f"; }; done
export GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
echo "Open http://127.0.0.1:8080 in your browser."
exec ./bin/llama-server -m "models/${F%% *}" -ngl 99 -c 4096 --port 8080
