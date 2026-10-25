#!/bin/bash
# usage: ingest.sh <cell> <session-subdir-name> <run_name>
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
P=/mnt/nvme/ds41_profiling/profiles/$1; O=$P/output; S=$O/$(ls $O | grep '^i-' | head -1)/$2; run=$3
ST=$P/stage_$run; rm -rf $ST; mkdir -p $ST
for f in $S/*; do ln -s $f $ST/; done
for f in $(find $O/neffs -name "*.neff"); do ln -sf $f $ST/; done
NE=/mnt/nvme/ds41_profiling/data/$run; rm -rf $NE; mkdir -p $NE
cd $P && /usr/bin/time -f "ingest %e s, maxrss %M KB" neuron-explorer view -d $ST --data-path $NE --display-name $run \
  --ingest-only --ignore-nc-buf-usage --ignore-instruction-hierarchy > $P/ingest_$run.log 2>&1
echo "exit $? $(tail -1 $P/ingest_$run.log)"
