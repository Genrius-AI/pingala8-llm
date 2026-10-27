#!/bin/bash
# Pingala-8 one-line install for Linux (NVIDIA RTX 30/40/50 GPU, recent driver):
#   curl -fsSL https://genrius-ai.github.io/pingala8-llm/install.sh | bash
# Downloads the latest Linux one-click build from GitHub Releases, checks its size (and sha256 when
# GitHub lists one), and unpacks it to ~/pingala8 (set PINGALA8_DIR to change). Models you already
# downloaded into ~/pingala8/models are kept. Source: https://github.com/Genrius-AI/pingala8-llm
set -euo pipefail

main() {
  local repo=Genrius-AI/pingala8-llm asset=pingala8-linux-cuda13.tar.gz
  local dest="${PINGALA8_DIR:-$HOME/pingala8}"
  local api json tag size digest url tmp have want_sum got_sum

  for c in curl tar gzip awk; do
    command -v "$c" >/dev/null 2>&1 || { echo "Please install $c and run this again." >&2; exit 1; }
  done
  [ "$(uname -s)" = Linux ] || { echo "This installer is for Linux. Windows: download pingala8-windows-cuda12.zip from https://github.com/$repo/releases/latest" >&2; exit 1; }
  [ "$(uname -m)" = x86_64 ] || { echo "The Linux build is for x86_64; this machine is $(uname -m)." >&2; exit 1; }
  command -v nvidia-smi >/dev/null 2>&1 || echo "Note: nvidia-smi not found. Pingala-8 needs an NVIDIA GPU (RTX 30, 40 or 50 series) and a recent driver."

  api="https://api.github.com/repos/$repo/releases/latest"
  echo "Looking up the latest release ..."
  json=$(curl -fsSL -H "Accept: application/vnd.github+json" "$api") || {
    echo "Could not reach $api (GitHub allows 60 lookups an hour per address). Try again later, or download $asset from https://github.com/$repo/releases/latest" >&2; exit 1; }
  tag=$(awk -F'"' '/"tag_name":/ {print $4; exit}' <<< "$json")
  # Read size, digest and download URL from the asset block whose "name" is the Linux tarball.
  read -r size digest url < <(awk -F'"' -v a="$asset" '
    $2 == "name" && $4 == a { on = 1; next }
    on && $2 == "size" { s = $3; gsub(/[^0-9]/, "", s) }
    on && $2 == "digest" { d = $4 }
    on && $2 == "browser_download_url" { print s, (d == "" ? "-" : d), $4; exit }' <<< "$json") || true
  [ -n "${url:-}" ] && [ -n "${size:-}" ] || { echo "The latest release ($tag) has no $asset." >&2; exit 1; }
  echo "Release $tag: $asset, $size bytes."

  mkdir -p "$dest"
  tmp="$dest/.$asset.part"
  rm -f "$tmp"
  echo "Downloading to $dest ..."
  # Resume after a dropped or stalled connection (no progress for 60 s), up to 10 tries.
  local try ok=0
  for try in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fL -C - --speed-limit 1024 --speed-time 60 -o "$tmp" "$url"; then ok=1; break; fi
    [ "$try" -lt 10 ] && echo "Download interrupted. Resuming ($((try + 1)) of 10) ..."
  done
  [ "$ok" = 1 ] || { rm -f "$tmp"; echo "Download did not finish. Check your connection and run this again." >&2; exit 1; }
  have=$(( $(wc -c < "$tmp") ))
  if [ "$have" != "$size" ]; then
    rm -f "$tmp"; echo "Size check failed: got $have bytes, expected $size. Run this again." >&2; exit 1
  fi
  if [ "$digest" != "-" ] && [ "${digest%%:*}" = sha256 ] && command -v sha256sum >/dev/null 2>&1; then
    want_sum=${digest#sha256:}
    got_sum=$(sha256sum "$tmp" | cut -d' ' -f1)
    if [ "$got_sum" != "$want_sum" ]; then
      rm -f "$tmp"; echo "Checksum (sha256) check failed. Run this again." >&2; exit 1
    fi
    echo "Size and sha256 match."
  else
    echo "Size matches."
  fi
  tar -xzf "$tmp" -C "$dest" --strip-components=1
  rm -f "$tmp"

  cat <<EOF

Pingala-8 $tag is in $dest. Run one model (it downloads once from Hugging Face, then serves a chat page):
  cd $dest && ./run.sh qwen          # Qwen3-4B, 1.24 GB
  cd $dest && ./run.sh llama         # Llama-3.1-8B base model, 2.39 GB (Built with Llama)
  cd $dest && ./run.sh qwen35-4b     # Qwen3.5-4B, 1.51 GB
  cd $dest && ./run.sh qwen35-2b     # Qwen3.5-2B, 0.78 GB
  cd $dest && ./run.sh qwen35-0.8b   # Qwen3.5-0.8B, 0.35 GB
Then open http://127.0.0.1:8080 in your browser. Press Ctrl+C to stop.
EOF
}

main "$@"
