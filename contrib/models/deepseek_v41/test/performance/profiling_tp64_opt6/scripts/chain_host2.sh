#!/bin/bash
W=/mnt/nvme/ds41_profiling_opt6
while ! grep -q "HOST CHAIN DONE" $W/hostprof/chain.log; do sleep 15; done
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
export DEEPSEEK_V41_SCHEME=tp64
cd /home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/test
for spec in "in1024_bs1 1024 1 1 1280 3" "in1024_bs10 1024 10 10 1280 2"; do
  set -- $spec; mv $W/hostprof/$1 $W/hostprof/$1.nospy 2>/dev/null; mkdir -p $W/hostprof/$1
  echo "[$(date +%T)] START $1" >> $W/hostprof/chain.log
  python $W/scripts/host_prof.py --cell $1 --isl $2 --bs $3 --ms $4 --ctx $5 --rounds $6 > $W/hostprof/$1/driver.log 2>&1
  echo "[$(date +%T)] END $1 rc=$? $(tail -1 $W/hostprof/$1/driver.log)" >> $W/hostprof/chain.log
  sleep 20
done
echo "[$(date +%T)] HOST CHAIN2 DONE" >> $W/hostprof/chain.log
