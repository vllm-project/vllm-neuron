#!/bin/bash
# runs after chain.sh: re-capture of the in8192 bs10 decode window (first profile session of a fresh server)
W=/mnt/nvme/ds41_profiling_opt6
while ! grep -q "CHAIN DONE" $W/chain.log; do sleep 30; done
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
export DEEPSEEK_V41_SCHEME=tp64 NEURON_RT_PROFILE_BUF_INSTRUCTION_MB=512 NEURON_RT_PROFILE_BUF_EVENT_MB=256
cd /home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/test
name=in8192_bs10_d2; mkdir -p $W/profiles/$name
echo "[$(date +%T)] START $name" >> $W/chain.log
python $W/scripts/profile_cell.py --cell in8192_bs10 --tag _d2 --isl 8192 --bs 10 --ms 10 --ctx 8448 --decode-window 1.5 \
  --phases decode --port 8351 > $W/profiles/$name/driver.log 2>&1
echo "[$(date +%T)] END $name rc=$? $(tail -1 $W/profiles/$name/driver.log)" >> $W/chain.log
echo "[$(date +%T)] CHAIN2 DONE" >> $W/chain.log
