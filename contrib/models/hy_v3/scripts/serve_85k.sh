#!/bin/bash
# 85k-context config without DCP, for MTP on the bench_data/prefix prompts
# (MTP does not run with DCP, so scripts/serve_256k.sh cannot take MTP_K).
# Same as serve_256k.sh except: no DCP, max_model_len 86016 (85000-token
# prompts + 256 output tokens + chat template), decode context buckets
# 8192 / 32768 / 65536 (+ the implicit 86016) and max_num_seqs 4.
# The scheduler admits requests by worst case (max_model_len each), so 4
# requests run at once and the rest wait: KV_CAP 0.65 holds 376k tokens
# (4.37x). With MTP the draft layer's KV and buffers need more HBM: 0.65 runs
# out at 4 concurrent requests (nrt_tensor_allocate), 0.60 holds only 3.99
# requests, so the MTP default is 0.61 (349k tokens, 4.05x). KV_CAP 0.85
# leaves too little HBM for the decode NEFFs to load.
# KV_DTYPE=fp8 (FP8 KV, needs HY3_KV_SCALE_PROFILE; with MTP a profile with
# the MTP layer, kv_profiles/Hy3-FP8_mtp_neuron.safetensors) halves the KV
# bytes per token, so 8 requests run at once: max_num_seqs 8, buckets
# [1, 4, 8], KV_CAP 0.60 (0.61 with MTP), about 8.1x, and only the 32768
# decode context bucket below 86016: with all three, the batch-8 decode NEFFs
# do not fit next to the KV (Allocation Failure at batch 8, 65536).
# Env: MODEL, PORT, MTP_K, KV_DTYPE, KV_CAP as for serve.sh, HY3_SPARSE_DECODE; with MTP_K,
#      HY3_DECODE_HITS_MAX_T defaults to 4 x (MTP_K + 1) (4-request verify steps;
#      8-request steps keep moe_tkg, which is faster there than hits at 32).
# Benchmark: bench_pd_split.sh "1 4 8".
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEFAULT_KV_CAP=0.65 SEQS=4 SEQ_BUCKETS='[1, 4]' CTX_BUCKETS='[8192, 32768, 65536]'
if [ "${KV_DTYPE:-auto}" = fp8 ]; then
    DEFAULT_KV_CAP=0.60 SEQS=8 SEQ_BUCKETS='[1, 4, 8]' CTX_BUCKETS='[32768]'
fi
if [ -n "${MTP_K:-}" ]; then
    export HY3_DECODE_HITS_MAX_T=${HY3_DECODE_HITS_MAX_T:-$((4 * (MTP_K + 1)))}
    DEFAULT_KV_CAP=0.61
fi
# HY3_SPARSE_DECODE: the K summaries (1/16 of the KV bytes) do not fit next to
# the 32768-bucket decode NEFFs (nrt_tensor_allocate at runtime); the sparse
# kernel reads only the kept blocks, so that bucket gains little.
if [ -n "${HY3_SPARSE_DECODE:-}" ]; then
    CTX_BUCKETS=  # only the max_model_len decode bucket
fi
KV_CAP=${KV_CAP:-$DEFAULT_KV_CAP} PREFILL_BUCKETS='[512, 1024, 2048, 4096]' \
    exec "$HERE/serve.sh" 86016 $SEQS "$SEQ_BUCKETS" ${CTX_BUCKETS:+"$CTX_BUCKETS"}
