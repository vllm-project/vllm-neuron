#!/bin/bash
# usage: launch.sh <cell> <isl> <bs> <ms> <ctx> [extra args]; env passes through (DGE etc.)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
cell=$1; shift; tag=""; for a in "$@"; do [ "$prev" = "--tag" ] && tag=$a; prev=$a; done
mkdir -p /mnt/nvme/ds41_profiling/profiles/$cell$tag
cd /home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/test
DEEPSEEK_V41_SCHEME=tp64 nohup python /mnt/nvme/ds41_profiling/scripts/profile_cell.py --cell $cell \
  --isl $1 --bs $2 --ms $3 --ctx $4 "${@:5}" > /mnt/nvme/ds41_profiling/profiles/$cell$tag/driver.log 2>&1 &
echo "pid $!"
