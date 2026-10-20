# GPT-OSS 120B 8K/1K disaggregated serving on Trn3 (6P1D)

Runnable scripts for the high-performance GPT-OSS 120B configuration at 8,192
input tokens and 1,024 output tokens. Six TP4 prefill replicas run on one Trn3
instance. One TP8 DP16 EP128 decode server runs on a second Trn3 instance, with
all decoder layers of a decode step fused into one megakernel. The proxy and the benchmark
client complete the deployment.

The directory is self-contained. It carries the DMA-order schedule of the
megakernel and the helper that prepares it.

For the walkthrough, the prerequisites, the flag reference and the
troubleshooting, see the tutorial:
[`docs/tutorials/tutorial-gpt-oss-120b-8k1k-trn3.md`](../../../../../../../docs/tutorials/tutorial-gpt-oss-120b-8k1k-trn3.md).
The scripts set the tested configuration. The tutorial explains it.

For the smaller 1P1D deployment of the gpt-oss tutorial, use `../multinode_DI/`.

## Topology

| Node | Role | Servers | Parallelism | Logical NeuronCores |
| --- | --- | --- | --- | --- |
| A | Prefill | 6 replicas, ports 8100 to 8105 | TP4 DP1 EP2 for each replica, `NEURON_LOGICAL_NC_CONFIG=2` | 24 of 64 |
| A | Proxy and benchmark client | Proxy on port 8000 | | |
| B | Decode | 1 server, port 8200 | TP8 DP16 EP128, `NEURON_LOGICAL_NC_CONFIG=1` | 128 of 128 |

## Files

| File | Where to run | Purpose |
| --- | --- | --- |
| `env_common.sh` | Sourced by every script | The addresses, the shape, the ports, and the environment that both roles need |
| `run_decode.sh` | Node B | The megakernel decode server (`kv_consumer`) with its tuning. Log: `$SERVER_LOG_DIR/decode.log` |
| `run_prefill.sh <index>` | Node A | One prefill replica (`kv_producer`), index 0 to 5 |
| `run_prefill_fleet.sh` | Node A | All six replicas, one after the other, with a health wait for each one |
| `run_proxy.sh` | Node A | The example proxy on port 8000 |
| `run_client.sh` | Node A | A health check and one completion through the proxy |
| `run_bench.sh` | Node A | The two benchmark legs: a warmup leg and the measured leg |
| `prepare_megakernel_schedule.py` | Called by `run_decode.sh` | Writes the schedule for the kernel-call scope of the active compile route |
| `gpt_oss_120b_megakernel_dma_order.json` | Read by the helper | The DMA-order schedule of the decode megakernel for Neuron SDK 2.33 |

## Order of operations

1. On **both** nodes, export the private IPs:

   ```bash
   export PREFILL_HOST=10.0.1.10   # node A
   export DECODE_HOST=10.0.1.20    # node B
   ```

   Set `MODEL_ID` if the checkpoint is a local path. Every other value has a
   default in `env_common.sh`.

2. On node B, start the decode server: `./run_decode.sh`.
3. On node A, start the six prefill replicas: `./run_prefill_fleet.sh`.
4. On node A, after both roles report `Application startup complete`, start
   the proxy: `./run_proxy.sh`. Then validate with `./run_client.sh`.
5. On node A, run the benchmark: `./run_bench.sh`. Read the measured leg in
   `$BENCH_RESULT_DIR/bench_clean.json`.

Stop the servers with `pkill -INT -f "vllm serve"` on both nodes and
`pkill -f toy_proxy_server` on node A.

## What the scripts need

- vLLM Neuron from Neuron SDK 2.33 on both nodes, with `nixl` and its
  LIBFABRIC runtime libraries.
- EFA connectivity between the two nodes. See the tutorial for the ports.
- The checkpoint `openai/gpt-oss-120b` on both nodes, or a local copy in
  `MODEL_ID`.
- The proxy `toy_proxy_server.py` from `examples/vllm_neuron/vllm/disaggregated_inference/`.
  `run_proxy.sh` finds it by its relative path in the examples tree. Set
  `PROXY_SCRIPT` if you copied this directory somewhere else.
