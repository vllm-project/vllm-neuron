#!/bin/bash
# Benchmark MiMo-V2.5 on this vllm-neuron port with `vllm bench serve`.
#
# Deliberately mirrors the recipe behind PR148's published Trn2 numbers --
# 900-token input / 90-token output, random dataset, range ratio 0.03, TP=64
# with EP=64 -- so output throughput / TTFT / TPOT line up column for column
# against both the NxDI table and the 8xH100 SGLang/vLLM baselines.
#
# One difference is itself a result: PR148 could not measure BS=1 on the FP8
# path at all, because NxDI's TKG rejects expert parallelism below
# BS = n_routed_experts/top_k = 256/8 = 32. This port drives the NKI moe_cte /
# moe_tkg kernels directly rather than ExpertMLPsV2.forward_selective_loading,
# so BS=1 works and single-stream latency is measurable here.
#
# Assumes a server is ALREADY running and serving --served-model-name MiMo-V2.5
# (see serve_mimo.sh). It is kept separate because the NEFF shape is baked per
# (max_num_seqs, max_model_len): changing BS means restarting + recompiling the
# server, so the bench driver must not own the server lifecycle.
#
# Usage:
#   bash bench.sh                      # c=1,16,32 against the running server
#   CONCURRENCIES="1 8" bash bench.sh
set -u

PORT="${PORT:-8000}"
MODEL="${MODEL:-MiMo-V2.5}"
# --model is the *served alias* that goes into the request payload, and the
# driver also feeds it to AutoTokenizer to synthesize the random dataset -- where
# an alias is not resolvable ("MiMo-V2.5 is not a local folder and is not a valid
# model identifier"). So name the tokenizer source explicitly: a hub repo ID (the
# default, cached in HF_HOME) or the local checkpoint the server is reading.
TOKENIZER="${TOKENIZER:-XiaomiMiMo/MiMo-V2.5}"
INPUT_LEN="${INPUT_LEN:-900}"
OUTPUT_LEN="${OUTPUT_LEN:-90}"
RANGE_RATIO="${RANGE_RATIO:-0.03}"
CONCURRENCIES="${CONCURRENCIES:-1 16 32}"
TAG="${TAG:-bs32}"
RESULTS_DIR="${RESULTS_DIR:-./mimo_bench_results}"
VENV="${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0}"

mkdir -p "$RESULTS_DIR"

if ! curl -s -m 5 "http://localhost:$PORT/health" > /dev/null; then
    echo "ERROR: no healthy server on port $PORT. Start serve_mimo.sh first." >&2
    exit 1
fi

for c in $CONCURRENCIES; do
    # 16 prompts at c=1 keeps the single-stream pass short; 128 at higher
    # concurrency gives every slot enough requests to reach steady state
    # (same prompt counts PR148 used, so the comparison stays apples-to-apples).
    if [ "$c" -eq 1 ]; then n=16; else n=128; fi
    out="$RESULTS_DIR/${TAG}_c${c}.log"
    echo "=== concurrency=$c prompts=$n -> $out"
    PATH="$VENV/bin:$PATH" \
    "$VENV/bin/vllm" bench serve \
        --backend openai \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --trust-remote-code \
        --base-url "http://localhost:$PORT" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio "$RANGE_RATIO" \
        --num-prompts "$n" \
        --max-concurrency "$c" \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$RESULTS_DIR" \
        --result-filename "${TAG}_c${c}.json" \
        2>&1 | tee "$out" | grep -E \
        'Successful|Benchmark duration|Output token throughput|Total Token|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Median ITL'
    echo
done

echo "Results in $RESULTS_DIR"
