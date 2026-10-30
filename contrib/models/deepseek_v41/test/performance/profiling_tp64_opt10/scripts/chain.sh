#!/bin/bash
# opt10 capture chain (same runs as opt6's chain.sh + chain2.sh); milestones -> /mnt/nvme/ds41_profiling_opt10/chain.log
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
W=/mnt/nvme/ds41_profiling_opt10
export DEEPSEEK_V41_SCHEME=tp64 NEURON_RT_PROFILE_BUF_INSTRUCTION_MB=512 NEURON_RT_PROFILE_BUF_EVENT_MB=256
cd /home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/test
run() {  # name, then profile_cell args
  local name=$1; shift
  mkdir -p $W/profiles/$name
  echo "[$(date +%T)] START $name $*" >> $W/chain.log
  python $W/scripts/profile_cell.py "$@" > $W/profiles/$name/driver.log 2>&1
  echo "[$(date +%T)] END $name rc=$? $(tail -1 $W/profiles/$name/driver.log)" >> $W/chain.log
  sleep 20
}
MR=0,4,9,18,27,36,45,63
run in1024_bs1_r0   --cell in1024_bs1  --tag _r0 --isl 1024 --bs 1  --ms 1  --ctx 1280 --decode-window 0.6 --port 8361
run in1024_bs10_r0  --cell in1024_bs10 --tag _r0 --isl 1024 --bs 10 --ms 10 --ctx 1280 --decode-window 1.2 --port 8362
run in8192_bs1_r0   --cell in8192_bs1  --tag _r0 --isl 8192 --bs 1  --ms 1  --ctx 8448 --decode-window 0.6 --port 8363
run in8192_bs10_r0  --cell in8192_bs10 --tag _r0 --isl 8192 --bs 10 --ms 10 --ctx 8448 --decode-window 1.5 --port 8364
run in8192_bs1_c2   --cell in8192_bs1  --tag _c2 --isl 8192 --bs 1  --ms 1  --ctx 8448 --decode-tokens 16 --phases prefill --delay-iters 4 --max-iters 1 --port 8365
run in8192_bs10_c2  --cell in8192_bs10 --tag _c2 --isl 8192 --bs 10 --ms 10 --ctx 8448 --decode-tokens 16 --phases prefill --delay-iters 4 --max-iters 1 --port 8366
run in1024_bs1_mr   --cell in1024_bs1  --tag _mr --isl 1024 --bs 1  --ms 1  --ctx 1280 --decode-window 0.4 --ranks $MR --port 8367
run in8192_bs1_mr   --cell in8192_bs1  --tag _mr --isl 8192 --bs 1  --ms 1  --ctx 8448 --decode-window 0.4 --ranks $MR --port 8368
run in1024_bs10_mr  --cell in1024_bs10 --tag _mr --isl 1024 --bs 10 --ms 10 --ctx 1280 --decode-window 0.8 --phases decode --ranks $MR --port 8369
run in8192_bs10_mr  --cell in8192_bs10 --tag _mr --isl 8192 --bs 10 --ms 10 --ctx 8448 --decode-window 1.0 --phases decode --ranks $MR --port 8370
# the in8192 bs10 decode window again on a fresh server (its second profile session can fail to allocate buffers)
run in8192_bs10_d2  --cell in8192_bs10 --tag _d2 --isl 8192 --bs 10 --ms 10 --ctx 8448 --decode-window 1.5 --phases decode --port 8371
echo "[$(date +%T)] CHAIN DONE" >> $W/chain.log
