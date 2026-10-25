#!/bin/bash
# Pingala-8 one-click (Linux): ./run.sh qwen  or  ./run.sh llama
# Downloads the B4 model once, then serves a chat page at http://127.0.0.1:8080
set -e
cd "$(dirname "$0")"
# F lists each model part as: name bytes sha256 (values from the Hugging Face repo)
case "${1:-qwen}" in
  qwen)  R=https://huggingface.co/Genrius-AI/Qwen3-4B-Pingala8-2bit/resolve/main
         F="4B-csq-vr3-00001-of-00002.gguf 995833472 4611812769418b2d0ffb8659f05892d1e3f9c4d20d80b121e93607494ab44243
            4B-csq-vr3-00002-of-00002.gguf 247320320 ded47243d30315e49836bf51d652c56fe3b21a288c04f8842d93ff3a70300ff8" ;;
  llama) R=https://huggingface.co/Genrius-AI/Llama-3.1-8B-Pingala8-2bit/resolve/main   # Built with Llama
         F="l31-8b-pingala8-flat-vrnative-cslfull-00001-of-00003.gguf 994298432 76b39997009a52e35cb31b32a42ad59ec7859c88ccb997d0ae76ed1da4e44686
            l31-8b-pingala8-flat-vrnative-cslfull-00002-of-00003.gguf 991815584 fbf44b6f9eac230ff654580592e8881da3dffb78ed98e78c42e1704e15fd2cf0
            l31-8b-pingala8-flat-vrnative-cslfull-00003-of-00003.gguf 402096800 8c0146744bc4a91ff067e2cc2fbb2f90fbd0ebb8ee7d279863906a22bc8e67bd" ;;
  *) echo "usage: ./run.sh qwen|llama"; exit 1 ;;
esac
mkdir -p models
# fetch NAME BYTES SHA256: keep a complete file; otherwise download to NAME.part, check it, then rename it.
fetch() {
  local f=$1 want=$2 sum=$3 have try
  if [ -f "models/$f" ]; then
    have=$(( $(wc -c < "models/$f") ))
    [ "$have" = "$want" ] && return 0
    echo "models/$f is $have bytes, expected $want (incomplete download). Downloading it again."
    rm -f "models/$f"
  fi
  for try in 1 2 3; do
    rm -f "models/$f.part"
    echo "Downloading $f ..."
    if curl -L --fail -o "models/$f.part" "$R/$f"; then
      have=$(( $(wc -c < "models/$f.part") ))
      if [ "$have" != "$want" ]; then
        echo "Size check failed for $f: got $have bytes, expected $want."
      elif [ "$(sha256sum "models/$f.part" | cut -d' ' -f1)" != "$sum" ]; then
        echo "Checksum (sha256) check failed for $f."
      else
        mv -f "models/$f.part" "models/$f"
        return 0
      fi
    else
      echo "Download of $f did not finish."
    fi
    rm -f "models/$f.part"
    if [ "$try" -lt 3 ]; then echo "Trying again ($((try + 1)) of 3) ..."; fi
  done
  echo "Download failed. Check your internet connection and run this script again."
  exit 1
}
set -- $F
while [ $# -ge 3 ]; do fetch "$1" "$2" "$3"; shift 3; done
export GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
echo "Open http://127.0.0.1:8080 in your browser."
exec ./bin/llama-server -m "models/${F%% *}" -ngl 99 -c 4096 --port 8080
