# Tutorial: Set up a Trainium rollout server for RL training (Qwen3.5)

<!-- meta: description: Set up a Trn2 instance as the rollout sampler for SkyRL LoRA training with ART's weight refresh, serving Qwen3.6/3.8-27B with full-vocabulary NKI sampling. -->
<!-- meta: keywords: RL, rollout, SkyRL, ART, LoRA, weight refresh, Qwen3.5, Qwen3.8-27B, NKI sampling, Trn2, Neuron 2.33 -->
<!-- meta: date_updated: 2026-11-25 -->
<!-- Content type: procedural-tutorial -->

This tutorial sets up one Trn2 instance as the rollout server of an RL
training run. The server samples with Qwen3.6-27B or Qwen3.8-27B. It returns
the logprob of each sampled token, and it loads each new LoRA checkpoint
from the trainer without a restart.

## Overview

The training run has two sides:

- **Trainer**, on GPUs: SkyRL trains a LoRA adapter with FSDP. After each step,
  it uploads the adapter to S3 and sends its sampling requests to the rollout
  server. The trainer's setup belongs to the RL framework. See
  [ART's SkyRL serving guide](https://github.com/awslabs/agentcore-rl-toolkit)
  (`src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/README.md`).
- **Rollout server**, on Trainium (this tutorial): `vllm serve` with ART's
  sampling plugins. A request that names a new checkpoint makes the server
  download the adapter, merge it into the base model on the CPU, and load the
  merged weights on every worker.

The server uses full-vocabulary NKI sampling
(`"stochastic_sampling_backend": "nki"`). It samples in the model graph with
async scheduling and returns each sampled token's logprob. See
[Full-vocabulary NKI sampling](../guides/reference-configuration.md#full-vocabulary-nki-sampling).

The layout in this tutorial is TP8 x DP2: two data-parallel engines of eight
logical NeuronCores each, 64K context, eight sequences per engine.

## Prerequisites

- A `trn2.48xlarge` instance launched from the AWS Deep Learning AMI
  **Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.1**.
  - The AMI provides the Neuron driver, runtime, collectives and tools.
  - It also provides the Python environment
    `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`, with vLLM
    0.29.0, PyTorch 2.13, the Neuron compiler and NKI.
- An instance role with `s3:GetObject` on the trainer's checkpoint prefix. Add
  `s3:ListBucket` to get a 404 instead of a 403 for a missing checkpoint.
- Network access from the trainer to the server port (8000 below), and from
  the instance to S3 and Hugging Face. The ART guide lists the network
  rules and the shared API key under "Network and permissions".
- A checkout of this repository, and a checkout of
  [agentcore-rl-toolkit](https://github.com/awslabs/agentcore-rl-toolkit)
  that contains
  `src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron`
  (added in awslabs/agentcore-rl-toolkit#246).

## Step 1: Mount local NVMe storage

The checkpoint (about 56 GB) and the merged checkpoints go on local NVMe:

- the root volume has about 50 GB free, which does not fit the checkpoint;
- reading from local disk keeps the weight refresh short.

The AMI leaves the instance-store disks unformatted. Pick an empty 1.7 TB
disk from `lsblk`; the root volume is usually `nvme0n1`.

```bash
lsblk
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mkdir -p /mnt/nvme && sudo mount /dev/nvme1n1 /mnt/nvme && sudo chown "$USER" /mnt/nvme
```

The instance store is erased when the instance stops. Download the
checkpoint again after a stop.

## Step 2: Install the plugin and the merge dependencies

Activate the AMI's vLLM environment. Install this repository over its
vllm-neuron. `--no-deps` keeps the Neuron packages of the AMI.

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
cd /path/to/vllm-neuron
pip install --no-deps -e .
pip install peft==0.18.1 accelerate==1.15.0   # ART's CPU merge of the LoRA adapter
python -m pip show vllm vllm-neuron torch neuronx-cc | grep -E "^(Name|Version|Location)"
```

The environment is owned by the default `ubuntu` user, so the installs do not
need `sudo`.

On a new instance, the first `import vllm_neuron` (and the first server start)
can take several minutes before anything is logged. EBS loads the AMI's blocks
on their first read. Later imports take seconds.

## Step 3: Download the base checkpoint

```bash
hf download Qwen/Qwen3.8-27B --local-dir /mnt/nvme/models/Qwen3.8-27B
```

`Qwen/Qwen3.6-27B` has the same architecture. The server reads this
directory as the immutable base. Each checkpoint from the trainer is merged
into a copy under the refresh root, and the base itself is never modified.

## Step 4: Start the rollout server

```bash
export ART_SERVING=/path/to/agentcore-rl-toolkit/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron
export PYTHONPATH=$ART_SERVING TOKENIZERS_PARALLELISM=false
export NEURON_LOGICAL_NC_CONFIG=2 NEURON_VISIBLE_DEVICES=0-15   # TP8 x DP2: 16 logical cores
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NEURON_SKIP_EFA_AFFINITY=1            # the instance has no EFA interface
export VLLM_ENGINE_READY_TIMEOUT_S=14400     # the first start compiles every graph
export ART_CHECKPOINT_DIR=/mnt/nvme/models/Qwen3.8-27B
export ART_WEIGHT_REFRESH_BASE=$ART_CHECKPOINT_DIR
export ART_WEIGHT_REFRESH_ROOT=/mnt/nvme/merged            # merged checkpoints
export ART_WEIGHT_REFRESH_CHECKPOINTS=s3://BUCKET/PREFIX   # the trainer's SKYRL_CHECKPOINTS_BASE
export ART_WEIGHT_REFRESH_S3_REGION=REGION
export VLLM_API_KEY=SHARED_KEY               # the trainer's SKYRL_EXTERNAL_INFERENCE_API_KEY
mkdir -p "$ART_WEIGHT_REFRESH_ROOT"

vllm serve "$ART_CHECKPOINT_DIR" --served-model-name initial --host PRIVATE_IP --port 8000 \
    --middleware server.ModelResolver --worker-extension-cls server.WeightRefreshWorker \
    --api-server-count 1 --trust-remote-code \
    --tensor-parallel-size 8 --data-parallel-size 2 --data-parallel-size-local 2 \
    --max-model-len 65536 --max-num-seqs 8 --max-num-batched-tokens 2048 \
    --enable-prefix-caching --async-scheduling \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --additional-config '{"neuron_config": {"num_seqs_buckets": [1, 4, 8],
        "decode_context_length_buckets": [4096, 8192, 16384, 32768],
        "on_device_sampling_config": {}, "stochastic_sampling_backend": "nki"}}'
```

- **Variables.** Replace `PRIVATE_IP`, `BUCKET/PREFIX`, `REGION` and
  `SHARED_KEY`. Set `NEURON_SKIP_EFA_AFFINITY=1` only when the instance was
  launched without an EFA interface; without it, such a server stops at
  startup.
- **ART's plugins.**
  - `server.ModelResolver` maps a model name of the form
    `<model id>_<checkpoint>` to a weight refresh. It also rewrites a
    completion's `logprobs` to 0, which is the value the NKI sampler computes.
  - `server.WeightRefreshWorker` loads the merged weights on every worker.
  - ART's `serve.sh` starts the server from its own serving profiles. The
    command above is the one validated with this plugin.
- **Startup.** The first start compiles every prefill and decode graph: about
  27 minutes, then the server answers `GET /health`. Later starts reuse the
  compiled graphs in `~/.cache/neuron_libtorch`.
- **Multi-turn agent rollouts.** Add `VLLM_NEURON_DP_PREFIX_AFFINITY=1`. The
  next turn of a session then goes to the engine that holds its prefix. See
  [Data-parallel routing](../guides/reference-configuration.md#data-parallel-routing).

## Step 5: Check the server

Send a request shaped like SkyRL's: a seed, temperature 1, the full
vocabulary (`top_k: -1`, `top_p: 1.0`), and `logprobs: 0`.

```bash
curl -s http://PRIVATE_IP:8000/v1/completions \
    -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
    -d '{"model": "initial", "prompt": "Question: What is 17 + 25?\nAnswer:",
         "max_tokens": 16, "temperature": 1.0, "top_k": -1, "top_p": 1.0,
         "seed": 0, "logprobs": 0}' | python -m json.tool
```

The response has one value in `choices[0].logprobs.token_logprobs` per
generated token. Sending the request again with the same seed returns the
same tokens. In a batch of a different composition, an occasional sequence
can still differ: the bf16 rounding depends on the batch.

The NKI sampler rejects `top_k`/`top_p` other than -1/1.0 with HTTP 400,
along with the other options listed in the reference. The engine keeps
serving after the rejection.

## Step 6: Connect the trainer and check a weight refresh

Point the trainer's external inference at `http://PRIVATE_IP:8000`, with the
same API key and the same checkpoints base. The first sampling request that
names a new checkpoint, for example `model_71d90030_ss1_seq4`, waits while
the server prepares it:

1. download the adapter archive from `ART_WEIGHT_REFRESH_CHECKPOINTS`;
2. merge it into the base on the CPU;
3. pause generation, load the weights on all 16 workers, and resume.

The server log then has a `weight_refresh` JSON line:

- `applied` should list 16 workers.
- `changed_parameters` counts the parameters whose values changed on each
  worker. It includes the 192 DeltaNet state buffers of Qwen3.8-27B, which
  every refresh resets. A value above 192 means the adapter changed weights.

A request that names a replaced version gets HTTP 409.

SkyRL's sampling client times out after 300 s. Keep the base checkpoint and
`ART_WEIGHT_REFRESH_ROOT` on local NVMe so that a refresh stays below that.
On remote storage the CPU merge can take several minutes, and the client then
has to retry the first request after each checkpoint.

## Validated configuration

The setup above was run on a `trn2.48xlarge` instance launched from the
SDK 2.33.1 AMI without EFA. It served Qwen3.8-27B, and the results matched
the same branch run on an internal Neuron environment:

| Check | Result |
| ---- | ---- |
| CPU unit tests (`pytest test/vllm_neuron --ignore=test/vllm_neuron/functional`) | 403 passed, 86 skipped |
| First start (compile and warmup, TP8 x DP2, 64K) | 26.5 min |
| SkyRL-shaped requests (seeds, `logprobs: 0`, `n` > 1, mixed greedy and sampled batches) | 200 with one finite logprob per token. Unsupported options get 400 with a message, and the engine keeps serving |
| Needle retrieval at 4K, 32K and 60K tokens | correct at every length |
| Decode at batch 1, 4K / 32K / 60K context | 18.4 / 19.2 / 20.1 ms per token |
| Weight refresh with a trained SkyRL LoRA adapter | 125 s (77 s download and merge, 47 s load). 16/16 workers, 672 changed parameters each. 409 for the replaced version |
| Chat-format GSM8K, thinking off, greedy, 1,319 problems (zero-config TP8 server) | 0.865. Every answer identical to the internal run |

On the internal environment, the same server ran end-to-end GRPO training:
10 steps of GSM8K, with a SkyRL FSDP LoRA trainer on GPUs. The rollout and
trainer logprobs of the sampled tokens differed by 0.0037 on average.

## Troubleshooting

| Symptom | Cause and fix |
| ---- | ---- |
| `RuntimeError: No EFA device found at /sys/bus/pci/devices/.../infiniband` at startup | The instance has no EFA interface. Set `NEURON_SKIP_EFA_AFFINITY=1`. |
| Nothing is logged for minutes after the first import or start | EBS is loading the AMI's blocks on their first read. Wait; later starts are fast. |
| `TimeoutError: Timed out waiting for engine core processes to start` about 10 minutes into the first start | With more than one data-parallel engine, the API server waits 600 s for the compile by default. Set `VLLM_ENGINE_READY_TIMEOUT_S=14400`. |
| HTTP 400 "does not support top_k/top_p" | Send `top_k: -1` and `top_p: 1.0`, or serve with `--generation-config vllm` so the checkpoint's `generation_config.json` does not set them. |
| HTTP 503 after a failed refresh | ART stops serving when a refresh fails after it blocked admission. Fix the cause in the log and restart the server. |
| The trainer's first request after each checkpoint times out | The refresh takes longer than the client's 300 s timeout. Keep the base and the refresh root on local NVMe, and retry the request. |
