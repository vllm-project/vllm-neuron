#!/bin/bash
set -x
prefill_node_ip=$(host -t A $1 | awk '/has address/ { print $4 }'  )
decode_node_ip=$(host -t A $2 | awk '/has address/ { print $4 }'  )
echo "Prefill node: " $prefill_node_ip " Decode node: " $decode_node_ip

# Model configs
MAX_SEQ_LEN=512
VLLM_BATCH=1
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
TP_DEGREE=8

# Log level settings
# ==========================================================================================
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_NEURON_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1

# Enable the following if you want to see the full debug log from NiXL library and libfabric.
# export NIXL_LOG_LEVEL="trace"
# export NIXL_DEBUG_LOGGING="yes"
# export FI_LOG_LEVEL="info"
# ==========================================================================================

export VLLM_NIXL_SIDE_CHANNEL_HOST="0.0.0.0"

export CUDA_VISIBLE_DEVICES=""


timestamp=$(date '+%Y%m%d_%H%M%S')
mode=$([ "$PREFILL" -eq 1 ] && echo "PREFILL" || echo "DECODE")

if [ "$PROXY" = "1" ]; then
    PREFILL_IP=$prefill_node_ip
    DECODE_IP=$decode_node_ip
    PREFILL_PORT=8100
    DECODE_PORT=8200

    PROXY_PORT=8000
    # 1P1D setup
    PROXY_CMD="python3 toy_proxy_server.py --port $PROXY_PORT"
    PROXY_CMD+=" --prefiller-host ${PREFILL_IP} --prefiller-ports ${PREFILL_PORT}"
    PROXY_CMD+=" --decoder-hosts ${DECODE_IP} --decoder-ports ${DECODE_PORT}"

    # Start the proxy server
    echo "Starting proxy server with command: $PROXY_CMD"
    $PROXY_CMD
else
    KV_IP=$prefill_node_ip
    if [ "$PREFILL" = "1" ]; then
        PORT=8100
        TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_buffer_device":"cuda","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":2e11,"kv_ip":"'"$KV_IP"'","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
    else
        PORT=8200
        TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_buffer_device":"cuda","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":2e11,"kv_ip":"'"$KV_IP"'","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
    fi
    echo "running di server"
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --max-num-seqs $VLLM_BATCH \
        --max-model-len $MAX_SEQ_LEN \
        --tensor-parallel-size $TP_DEGREE \
        --kv-transfer-config $TRANSFER_CONFIG \
        --port ${PORT} 2>&1 | tee logs/di_${timestamp}_${mode}.log
fi
