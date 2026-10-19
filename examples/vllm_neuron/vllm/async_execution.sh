export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_NEURON_LOG_LEVEL=DEBUG
export NEURON_RT_INSPECT_ENABLE=1
rm -rf ./async
export NEURON_RT_INSPECT_OUTPUT_DIR=./async
export CUDA_VISIBLE_DEVICES=""

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

timestamp=$(date '+%Y%m%d_%H%M%S')

python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --max-num-seqs 1 \
        --max-model-len 512 \
        --tensor-parallel-size 1 \
        --async-scheduling \
        --additional_config '{"neuron_config": {"on_device_sampling_config": {"temperature": "0"}}}' \
        --port 8000 2>&1 | tee logs/async_${timestamp}.log
