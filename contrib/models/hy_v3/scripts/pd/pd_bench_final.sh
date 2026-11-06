#!/bin/bash
# pd_bench_final.sh <out_dir>: the GPU-comparison set on one running 1P1D pair:
#   1. P-only (pd_bench_split.sh, P_ONLY=1) at C = 1 4 8 12
#   2. end-to-end customer case (pd_bench.sh) at C = 8, 1024 output tokens
#   3. the customer's GPU "D 单测" (pd_bench_dcust.sh): replay at ~100% hit, 1200 output, C = 8
# Env as pd_bench_split.sh / pd_bench.sh (CODE, MODEL, PROXY, P_URL / D_URL ...).
set -uo pipefail
OUT=${1:?usage: pd_bench_final.sh <out_dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
P_ONLY=1 bash "$HERE/pd_bench_split.sh" "$OUT/split" "${P_CONCS:-1 4 8 12}"
bash "$HERE/pd_bench.sh" "$OUT/e2e" "${E2E_CONCS:-8}" "${OUT_LEN:-1024}"
bash "$HERE/pd_bench_dcust.sh" "$OUT/dcust" "${D_CONCS:-8}" "${D_OUT_LEN:-1200}"
