#!/bin/bash
# Compile-cache isolation of the FP8 KV scales on a reduced checkpoint (m1 mode).
#
# Three FP8 runs of validate_fp8_kv.py against one NEFF cache:
#   A  profile P            (compiles or reuses whatever the cache has)
#   B  profile P again      -> expect 0 new NEFFs and logits identical to A
#   C  profile P' (other scales, e.g. V dequant x2) -> expect new decode NEFFs
#      (decode takes the scales as compile-time values; prefill reads them from
#      a runtime tensor, so prefill NEFFs may be reused), logits different
#      from A, and C must still pass compare_kv_runs.py against the BF16-KV dump:
#      a stale NEFF built for P would read the cache written with P' at the
#      wrong scale and fail that check.
#
# Usage: kv_cache_isolation.sh <fp8_ckpt> <bf16_m1.pt> <profile P> <profile P'> <out_dir>
# Env:   PY (serving venv python), NUM_BLOCKS (default 1024),
#        NEFF_CACHE (default $VLLM_CACHE_ROOT/neuron/native_neff_cache),
#        EVAL_ONLY=1 re-evaluates existing runs in <out_dir> (summary.txt kept)
set -uo pipefail
CKPT=$1 BF16=$2 P=$3 P2=$4 OUT=$5
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
NEFF_CACHE=${NEFF_CACHE:-${VLLM_CACHE_ROOT:?}/neuron/native_neff_cache}
export NUM_BLOCKS=${NUM_BLOCKS:-1024} KV_DTYPE=fp8
mkdir -p "$OUT"
count() { find "$NEFF_CACHE" -name '*.neff' 2>/dev/null | wc -l; }
rc=0
[ "${EVAL_ONLY:-0}" = 1 ] && RUNS="" || RUNS="A:$P B:$P C:$P2"
for run in $RUNS; do
  name=${run%%:*} prof=${run#*:}
  before=$(count); t0=$(date +%s)
  HY3_KV_SCALE_PROFILE=$prof $PY "$HERE/validate_fp8_kv.py" "$CKPT" "$OUT/$name.pt" m1 > "$OUT/$name.log" 2>&1 || rc=1
  echo "$name profile sha256 $(sha256sum "$prof" | cut -c1-16) new_neffs $(( $(count) - before )) wall $(( $(date +%s) - t0 ))s" | tee -a "$OUT/summary.txt"
done
$PY - "$OUT" <<'EOF' 2>&1 | grep -v -i warn | tee -a "$OUT/summary.txt"
import sys, torch
out = sys.argv[1]
r = {n: torch.load(f"{out}/{n}.pt")["phases"] for n in "ABC"}
def maxdiff(x, y):
    d = 0.0
    for ph in ("cold", "reuse"):
        for a, b in zip(x[ph], y[ph]):
            la, lb = a["logits"], b["logits"]
            n = min(len(la), len(lb))
            la, lb = la[:n], lb[:n]
            if not torch.equal(torch.isfinite(la), torch.isfinite(lb)) and d == 0.0:
                d = 1e-30  # different -inf pattern: not identical
            m = torch.isfinite(la) & torch.isfinite(lb)
            d = max(d, (la[m] - lb[m]).abs().max().item())
    return d
same = maxdiff(r["A"], r["B"]) == 0.0 and all(
    a["gen_ids"] == b["gen_ids"] for ph in r["A"] for a, b in zip(r["A"][ph], r["B"][ph]))
diff = maxdiff(r["A"], r["C"])
print(f"check B==A identical {same}; C vs A max |dlogit| over finite logits {diff:.4f}")
sys.exit(0 if same and diff > 0 else 1)
EOF
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
B_NEW=$(sed -n 's/^B profile .* new_neffs \([0-9]*\) .*/\1/p' "$OUT/summary.txt" | tail -1)
[ "$B_NEW" = 0 ] || { echo "FAIL: same profile compiled $B_NEW new NEFFs" | tee -a "$OUT/summary.txt"; rc=1; }
$PY "$HERE/compare_kv_runs.py" "$BF16" "$OUT/C.pt" > "$OUT/C.compare.txt" 2>&1 || rc=1
echo "C vs BF16: $(grep '===' "$OUT/C.compare.txt")" | tee -a "$OUT/summary.txt"
echo "=== cache isolation $([ $rc = 0 ] && echo PASS || echo FAIL)" | tee -a "$OUT/summary.txt"
exit $rc
