#!/bin/bash
# pd_proxy.sh: a CPU-only proxy on PORT (default 8000) in front of one prefill (P_HOST:8100) and
# one decode (D_HOST:8200) server. Which one depends on CONNECTOR, and must match pd_serve.sh's:
#
#   NeuronNixlConnector (default)  vllm-neuron's toy_proxy_server.py, read mode: the request
#       goes to prefill with max_tokens=1, then to decode with the returned kv_transfer_params;
#       decode pulls the KV over NIXL. Has /healthcheck.
#   NeuronMooncakeConnector        vLLM's mooncake_connector_proxy.py (vendored here): the request
#       goes to prefill (max_tokens=1) and decode at the same time with a shared transfer_id;
#       decode allocates its blocks and the prefill side writes the KV into them when its
#       prefill is done. The proxy reads the prefill engine id from the bootstrap server
#       (P_HOST:MC_BOOTSTRAP_PORT, default 8998) and answers 503 until it has it. No health route.
#
# Env: CODE (vllm-neuron checkout, default: the one holding this script), P_HOST (default: this host's IP), D_HOST (127.0.0.1),
#      P_PORT / D_PORT (8100 / 8200), PORT (8000), CONNECTOR, MC_BOOTSTRAP_PORT,
#      P_INSTANCES (default 1; 2: the two half-node prefill instances of pd_serve.sh P_HALF=0 / 1 on
#      P_HOST, ports 8100 / 8101, bootstrap 8998 / 8999; requests round-robin over them).
set -euo pipefail
CODE=${CODE:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
HERE=$(cd "$(dirname "$0")" && pwd)
# The toy proxy rewrites kv_transfer_params.remote_host to the prefill host it is given, and decode
# handshakes with that address: 127.0.0.1 here sends a remote decode to itself. Default to this
# host's private IP (the proxy runs on the prefill node).
P=${P_HOST:-$(hostname -I | awk '{print $1}')}:${P_PORT:-8100}
D=${D_HOST:-127.0.0.1}:${D_PORT:-8200}
if [ "${CONNECTOR:-NeuronNixlConnector}" = NeuronMooncakeConnector ]; then
  if [ "${P_INSTANCES:-1}" = 2 ]; then
    PH=${P%:*}
    exec python3 "$HERE/mooncake_connector_proxy.py" --host 0.0.0.0 --port "${PORT:-8000}" \
      --prefill "http://$PH:8100" 8998 --prefill "http://$PH:8101" 8999 --decode "http://$D"
  fi
  exec python3 "$HERE/mooncake_connector_proxy.py" --host 0.0.0.0 --port "${PORT:-8000}" \
    --prefill "http://$P" "${MC_BOOTSTRAP_PORT:-8998}" --decode "http://$D"
fi
[ "${P_INSTANCES:-1}" = 1 ] || { echo "P_INSTANCES > 1 needs CONNECTOR=NeuronMooncakeConnector" >&2; exit 1; }
exec python3 "$CODE/examples/vllm_neuron/vllm/disaggregated_inference/toy_proxy_server.py" \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --prefiller-hosts "${P%:*}" --prefiller-ports "${P##*:}" \
  --decoder-hosts "${D%:*}" --decoder-ports "${D##*:}"
