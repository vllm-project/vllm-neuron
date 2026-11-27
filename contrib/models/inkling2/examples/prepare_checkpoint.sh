#!/usr/bin/env bash
# A view of the Inkling-2-Mini checkpoint that vLLM Neuron serves: symlinks to every file, except that
# hf_quant_config.json is left out and config.json is rewritten without "quantization_config". The checkpoint
# declares ModelOpt NVFP4, which vLLM would otherwise try to apply; this model dequantizes the experts to bf16
# itself when it loads them (src/weights.py). The original directory is not modified.
#
# usage: prepare_checkpoint.sh <checkpoint dir> <view dir>
set -euo pipefail
SRC=$(cd "$1" && pwd); DST=$2
mkdir -p "$DST"
for f in "$SRC"/*; do
    b=$(basename "$f")
    case "$b" in hf_quant_config.json|config.json) continue ;; esac
    ln -sfn "$f" "$DST/$b"
done
python3 - "$SRC/config.json" "$DST/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.pop("quantization_config", None)
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
echo "serve $DST"
