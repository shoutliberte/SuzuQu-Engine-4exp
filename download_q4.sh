#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GGUF=Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
DEST=${1:-}
if [ -z "$DEST" ]; then
    # Default install location is ./models inside the repo; if the model
    # already lives in a sibling models/ next to the repo, keep using it.
    DEST=$ROOT/models/IQ3E-Q8D-MTP
    if [ ! -e "$DEST/$GGUF" ] && [ -e "$ROOT/../models/IQ3E-Q8D-MTP/$GGUF" ]; then
        DEST=$ROOT/../models/IQ3E-Q8D-MTP
    fi
fi
mkdir -p "$DEST"
echo "Downloading IQ3E-Q8D-MTP (~86 GB) into $DEST"
hf download pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF \
    --include "$GGUF" \
    --local-dir "$DEST"
echo "done: $DEST/$GGUF"
