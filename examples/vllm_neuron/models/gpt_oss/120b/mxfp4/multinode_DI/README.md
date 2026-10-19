# gpt-oss 120B disaggregated-inference target config (1P1D)

Runnable scripts for the **disaggregated-inference (DI)** target configuration
from the gpt-oss tutorial — the deployment that delivers the best throughput and
latency for gpt-oss on Neuron. Prefill and decode run on **separate instances**
fronted by a proxy router; KV cache moves from prefill to decode over NixL on the
`LIBFABRIC` backend (EFA on AWS).

For the full walkthrough, prerequisites, flag explanations, and troubleshooting,
see the tutorial: [`docs/tutorials/tutorial-gpt-oss.md`](../../../../../../../docs/tutorials/tutorial-gpt-oss.md).
These scripts pin the tested topology; the tutorial is the source of truth.

For a simpler single-instance (non-DI) server, use `../non_DI/run_tp8.sh`
instead.

## Target topology

| Role | Parallelism | `max_num_seqs` | Global decode batch |
| --- | --- | --- | --- |
| Prefill (20B and 120B) | attention TP4 (DP1), MoE EP2 | 1 | — |
| Decode, gpt-oss 20B | TP8 DP4 EP32 (1 expert/rank) | 4 | 16 (4 × DP4) |
| Decode, gpt-oss 120B | TP8 DP8 EP64 (2 experts/rank) | 4 | 32 (4 × DP8) |

Scripts default to **gpt-oss 120B**. Override the model with the `MODEL_ID`
environment variable; for 20B, also apply the decode-side changes noted in
`run_decode.sh` (drop DP to 4, omit `--optimization-level 2`).

## Prerequisites

- **Two instances** with all 64 NeuronCores each — one prefill, one decode.
- **EFA connectivity** between them (DI transfers KV cache over NixL / LIBFABRIC).
- **The checkpoint staged on both instances.**

See the tutorial's "DI prerequisites" section for details.

## Files

| Script | Where to run | Purpose |
| --- | --- | --- |
| `run_decode.sh` | Decode instance | Decode server (`kv_consumer`), TP8 DP8 EP64, port 8200. Sets `--max-logprobs 0` to skip the decode-side logit gather. |
| `run_prefill.sh` | Prefill instance | Prefill server (`kv_producer`), TP4 DP1 EP2, port 8100. |
| `run_proxy.sh` | Prefill instance (or any host reaching both) | Proxy router on port 8000; routes prefill → decode. |
| `run_client.sh` | Any host reaching the proxy | Health check + sample completion against the proxy. |

## Order of operations

1. **Set instance addresses.** `run_proxy.sh` and `run_client.sh` read
   `PREFILL_HOST` / `DECODE_HOST` / `PROXY_HOST` from the environment (default
   `127.0.0.1`). Export the real instance addresses before launching.
2. **Launch decode and prefill** (`run_decode.sh` on the decode instance,
   `run_prefill.sh` on the prefill instance). They have no startup dependency on
   each other and can start at the same time. Wait for both to report
   `Application startup complete`.
3. **Launch the proxy** (`run_proxy.sh`) once both servers are up.
4. **Validate** with `run_client.sh` — send requests to the proxy (port 8000),
   never to the prefill or decode servers directly.

## Scaling prefill (16P1D)

A single TP4 prefill server uses only 4 of the 64 cores. For best performance,
tile up to 16 TP4 prefill servers to match one TP8 DP8 EP64 decode server (a
**16P1D** topology). Launch `run_prefill.sh` once per 4-core slice, giving each a
distinct `NEURON_VISIBLE_DEVICES`, `--port`, and `VLLM_NIXL_SIDE_CHANNEL_PORT`,
then pass every prefill port to the proxy's `--prefiller-port`. See the tutorial's
"Scale prefill to fully utilize the instance" section.

Run these as separate prefill servers, **not** one server with a data-parallel
degree of 16 — raising DP with expert parallelism enabled changes the MoE expert
layout.
