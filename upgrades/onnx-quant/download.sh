#!/usr/bin/env bash
# Upgrade 03 - official NCM checkpoints as ONNX (mrfakename/ncm-webgpu-weights),
# found via yrangana/ncm-lab. Target: local browser demo + int4 for jev-mid.
set -euo pipefail
DEST="${1:-$HOME/ncm-upgrades/onnx-weights}"
mkdir -p "$DEST"
if command -v hf >/dev/null 2>&1; then
  hf download mrfakename/ncm-webgpu-weights --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download mrfakename/ncm-webgpu-weights --local-dir "$DEST"
else
  echo "install hf: pip install -U huggingface_hub" >&2; exit 1
fi
echo "-> $DEST"
