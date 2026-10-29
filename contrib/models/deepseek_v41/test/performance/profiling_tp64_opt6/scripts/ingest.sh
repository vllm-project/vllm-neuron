#!/bin/bash
# usage: ingest.sh <profile-dir-name> <worker-pid-dir-glob-or-""> <session-index 0|1|..> <run_name> [--hier]
# stages one worker's session (NTFF + system trace + NEFFs) and ingests it (nice'd, 24 cores)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
P=/mnt/nvme/ds41_profiling_opt6/profiles/$1; O=$P/output
W=$(ls -d $O/i-*${2}* | head -1); S=$(ls -d $W/*/ | sort | sed -n "$(( $3 + 1 ))p"); run=$4
EXTRA="--ignore-instruction-hierarchy"; [ "$5" = "--hier" ] && EXTRA=""
ST=$P/stage_$run; rm -rf $ST; mkdir -p $ST
for f in $S/*; do ln -s $f $ST/; done
for f in $(find $O/neffs -name "*.neff"); do ln -sf $f $ST/; done
NE=/mnt/nvme/ds41_profiling_opt6/data/$run; rm -rf $NE; mkdir -p $NE
cd $P && /usr/bin/time -f "ingest %e s, maxrss %M KB" nice -n 19 taskset -c 0-23 neuron-explorer view -d $ST --data-path $NE --display-name $run \
  --ingest-only --ignore-nc-buf-usage $EXTRA > $P/ingest_$run.log 2>&1
echo "exit $? $(tail -1 $P/ingest_$run.log) session=$S"
