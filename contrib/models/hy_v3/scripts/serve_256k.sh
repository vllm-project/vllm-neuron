#!/bin/bash
# Long-context config: max_model_len 256000, DCP=8 (KV for 8 concurrent
# max-length requests; without DCP the KV holds only one), prefix caching on.
# Prefill buckets stop at 4096: with DCP the 8192-token prefill does not compile
# on neuronx-cc 2.28 (attention_kv_parallel_segmented_cte needs 324640 bytes of
# SBUF per partition, NCC_IGCA037); 4096 compiles.
# Env: MODEL (default tencent/Hy3-FP8), KV_CAP, PORT, as for serve.sh.
# Tested with needle.py run ../bench_data/needle.jsonl.gz (16k-85k prompts) and
# bench_fixed.sh "1024 4096 16000 32768 65536 85000" "1 4 8".
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DCP=8 PREFILL_BUCKETS='[512, 1024, 2048, 4096]' \
    exec "$HERE/serve.sh" 256000 8 '[1, 4, 8]' '[8192, 32768, 65536, 98304]'
