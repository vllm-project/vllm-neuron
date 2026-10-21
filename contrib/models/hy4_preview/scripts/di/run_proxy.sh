#!/bin/bash
# The CPU-only proxy in front of one prefill and one decode server (vllm-neuron's toy proxy).
#     PREFILL_HOST=<ip> DECODE_HOST=<ip> VLLM_NEURON_REPO=<vllm-neuron checkout> bash run_proxy.sh
set -x
# Real addresses: the proxy hands PREFILL_HOST to the decode server as the NIXL
# side-channel host, so 127.0.0.1 makes decode handshake with its own loopback.
PREFILL_HOST="${PREFILL_HOST:?set PREFILL_HOST to the prefill node private IP}"
DECODE_HOST="${DECODE_HOST:?set DECODE_HOST}"
REPO="${VLLM_NEURON_REPO:?set VLLM_NEURON_REPO}"
python3 "$REPO/examples/vllm_neuron/vllm/disaggregated_inference/toy_proxy_server.py" \
    --port 8000 --host 0.0.0.0 \
    --prefiller-host "$PREFILL_HOST" --prefiller-port 8100 \
    --decoder-host "$DECODE_HOST" --decoder-port 8200
