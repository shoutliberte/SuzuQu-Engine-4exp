#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST=${1:-"$ROOT/models/IQ3E-Q8D-MTP"}
mkdir -p "$DEST"
echo "Downloading IQ3E-Q8D-MTP (~86 GB) into $DEST"
hf download pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF \
    --include 'Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf' \
    --local-dir "$DEST"
echo "done: $DEST/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"
