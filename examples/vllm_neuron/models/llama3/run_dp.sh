#!/bin/bash

# Terminal 1: Start server
# ./run_dp.sh server

# Terminal 2: Send request
# ./run_dp.sh client

if [ "$1" == "server" ]; then
    # If downloading checkpoint from HF (default behavior), consider setting the download_dir param below
    # to download the checkpoint to SSD (this may speed up the download). For example, set 
    # download_dir="/path/to/local/ssd"
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-num-seqs 1 \
        --max-model-len 256 \
        --tensor-parallel-size 8 \
        --data-parallel-size 2 \
        --port 8000

elif [ "$1" == "client" ]; then
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "prompt": "I am gonna keep counting forever, 1 2 3 4 5 ",
            "max_tokens": 10,
            "temperature": 0.0
        }'
else
    echo "Usage: ./run_dp.sh [server|client]"
fi
