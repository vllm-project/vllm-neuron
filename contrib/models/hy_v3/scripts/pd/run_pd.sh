#!/bin/bash
# run_pd.sh <label> <split|prefill-node|decode-node|decode-bench> [env KEY=VAL ...]
# One Hy3 1P1D run: start the server(s) for this node, the proxy (on the prefill node / in split
# mode), wait for health, run pd_bench.sh, stop everything this script started.
#
#   split         one trn2.48xlarge: prefill on devices 0-7, decode on 8-15 (TP=32 each), proxy,
#                 bench. Needs EFA for NIXL's LIBFABRIC backend on device memory.
#   prefill-node  2-node mode, on the prefill host: prefill server + proxy + bench
#                 (start the decode-node side first; D_HOST must point at it)
#   decode-node   2-node mode, on the decode host: decode server only; stays up until killed
#   decode-bench  one node, no prefill server: NeuronDecodeBenchConnector decode-only step
#                 times at concurrency 1 / 4 / 8 (no MTP: the KV is uninitialized)
#
# CONNECTOR=NeuronMooncakeConnector switches servers and proxy to Mooncake (same on both nodes).
#
# Results: $WORK/results/pd/<label>/, logs $WORK/logs/pd_<label>_*.log (WORK default /mnt/nvme)
# Env: everything pd_serve.sh takes (CODE, PKG = a dir holding a hy_v3_neuron link to $CODE's model, MODEL, KV_DTYPE, MTP_K, HY3_KV_SCALE_PROFILE, ...),
#      P_HOST / D_HOST (2-node), CONCS (default "1 4 8"), UP_TIMEOUT (s, default 43200),
#      P_INSTANCES (prefill-node, Mooncake: 1, or 2 half-node TP=32 prefill instances behind the proxy),
#      OUT_LEN (output tokens, default 256), BENCH (pd_bench = end-to-end 85k/92% | pd_acc = needle + sparse_decode_eval), VENV (default $WORK/venv/sdk233).
set -u
LABEL=${1:?usage: run_pd.sh <label> <split|prefill-node|decode-node|decode-bench> [KEY=VAL ...]}
MODE=${2:?mode}; shift 2
for kv in "$@"; do export "$kv"; done
HERE=$(cd "$(dirname "$0")" && pwd)
N=${WORK:-/mnt/nvme}; L=$N/logs/pd_$LABEL; R=$N/results/pd/$LABEL; mkdir -p $R
export CODE=${CODE:-$(cd "$HERE/../../../../.." && pwd)} MODEL=${MODEL:-$N/models/Hy3-FP8}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$N/cache/pd}
source ${VENV:-$N/venv/sdk233}/bin/activate
pids=()
stop_all() {
  for g in "${pids[@]}"; do kill -TERM -- -"$g" 2>/dev/null; done; sleep 30
  for g in "${pids[@]}"; do kill -KILL -- -"$g" 2>/dev/null; done
  echo "$(date +%T) $LABEL: stopped"
}
trap stop_all EXIT
start() {  # start <name> <cmd...>: in its own process group, log to ${L}_<name>.log
  local name=$1; shift
  setsid "$@" > "${L}_$name.log" 2>&1 < /dev/null &
  pids+=($!)
  echo "$(date +%T) $LABEL: started $name (pgid $!)"
}
wait_up() {  # wait_up <url> <name>
  local t0=$SECONDS
  until curl -sf "$1" >/dev/null; do
    sleep 30
    for g in "${pids[@]}"; do kill -0 -- -"$g" 2>/dev/null || { echo "$(date +%T) $LABEL: a process died; tail of $2:"; tail -5 "${L}_$2.log" | cut -c1-300; exit 1; }; done
    [ $((SECONDS - t0)) -gt ${UP_TIMEOUT:-43200} ] && { echo "$(date +%T) $LABEL: $2 not up"; exit 1; }
  done
  echo "$(date +%T) $LABEL: $2 up after $((SECONDS - t0)) s; $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "${L}_$2.log" | tail -1)"
}
wait_listen() {  # wait_listen <url> <name>: any HTTP answer (the Mooncake proxy has no health route)
  local t0=$SECONDS
  until [ "$(curl -s -o /dev/null -w '%{http_code}' "$1")" != 000 ]; do
    sleep 5; [ $((SECONDS - t0)) -gt 300 ] && { echo "$(date +%T) $LABEL: $2 not listening"; tail -5 "${L}_$2.log"; exit 1; }
  done
  echo "$(date +%T) $LABEL: $2 listening"
}
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null; }

case "$MODE" in
  split)
    drop_caches
    SPLIT=1 start decode bash "$HERE/pd_serve.sh" decode
    SPLIT=1 start prefill bash "$HERE/pd_serve.sh" prefill
    wait_up http://127.0.0.1:8200/health decode
    wait_up http://127.0.0.1:8100/health prefill
    start proxy bash "$HERE/pd_proxy.sh"
    wait_listen http://127.0.0.1:8000/ proxy
    bash "$HERE/${BENCH:-pd_bench}.sh" "$R" "${CONCS:-1 4 8}" ${OUT_LEN:-} | tee "${L}_bench.log" ;;
  prefill-node)
    : "${D_HOST:?set D_HOST to the decode node}"
    drop_caches
    if [ "${P_INSTANCES:-1}" = 2 ]; then
      # Two half-node (TP=32) prefill instances; the second starts once the first has compiled.
      P_HALF=0 start prefill bash "$HERE/pd_serve.sh" prefill
      wait_up http://127.0.0.1:8100/health prefill
      P_HALF=1 start prefill1 bash "$HERE/pd_serve.sh" prefill
      wait_up http://127.0.0.1:8101/health prefill1
    else
      start prefill bash "$HERE/pd_serve.sh" prefill
      wait_up http://127.0.0.1:8100/health prefill
    fi
    until curl -sf "http://$D_HOST:8200/health" >/dev/null; do sleep 30; done   # the decode node's server
    echo "$(date +%T) $LABEL: decode on $D_HOST up"
    D_HOST=$D_HOST start proxy bash "$HERE/pd_proxy.sh"
    wait_listen http://127.0.0.1:8000/ proxy
    D_URL=http://$D_HOST:8200 bash "$HERE/${BENCH:-pd_bench}.sh" "$R" "${CONCS:-1 4 8}" ${OUT_LEN:-} | tee "${L}_bench.log" ;;
  decode-node)
    drop_caches
    start decode bash "$HERE/pd_serve.sh" decode
    wait_up http://127.0.0.1:8200/health decode
    echo "$(date +%T) $LABEL: decode serving; Ctrl-C / kill this script to stop"
    while sleep 60; do :; done ;;
  decode-bench)
    drop_caches
    start decode bash "$HERE/pd_serve.sh" decode-bench
    wait_up http://127.0.0.1:8200/health decode
    # Decode only: prompts of the 85k case, whose prefill the connector skips.
    for C in ${CONCS:-1 4 8}; do
      src=$CODE/contrib/models/hy_v3/bench_data/prefix/in85000_p78336_c${C}.jsonl.gz
      python3 -c "import gzip,json,sys; r=[json.loads(l) for l in gzip.open(sys.argv[1],'rt')]; [print(json.dumps({'prompt': r[0]['prefix']+x['suffix']})) for x in r[2:]]" "$src" > /tmp/pd_db_$C.jsonl
      vllm bench serve --backend openai --base-url http://127.0.0.1:8200 --model hy3 --tokenizer "$MODEL" \
        --dataset-name custom --dataset-path /tmp/pd_db_$C.jsonl --skip-chat-template --disable-shuffle \
        --custom-output-len 256 --ignore-eos --num-prompts $((10 * C)) --max-concurrency $C --ready-check-timeout-sec 0 \
        --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "$R" --result-filename "decode_only_c$C.json" \
        2>&1 | grep -E "Successful|TTFT|TPOT|ITL|throughput" | grep -E "Median|P99|throughput|Successful" | tee -a "${L}_bench.log"
    done ;;
  *) echo "mode: split | prefill-node | decode-node | decode-bench" >&2; exit 1 ;;
esac
