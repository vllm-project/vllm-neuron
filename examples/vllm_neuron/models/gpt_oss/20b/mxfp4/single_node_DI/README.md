# gpt-oss 20B single-node disaggregated inference (1P1D)

Runnable scripts for **single-node** disaggregated inference (DI) of
`openai/gpt-oss-20b` (MXFP4) on one 64-core Trn3 instance. Prefill and decode
run as separate servers on the **same host** using disjoint core ranges, and KV
cache transfers over NixL on the `LIBFABRIC` backend via loopback (`127.0.0.1`)
— no second instance or EFA fabric required.

For the flag walkthrough, prerequisites, and troubleshooting, see the tutorial:
[`docs/tutorials/tutorial-gpt-oss.md`](../../../../../../../docs/tutorials/tutorial-gpt-oss.md).
For a simpler single-instance (non-DI) server, use
[`../non_DI/run_tp8.sh`](../non_DI/run_tp8.sh) instead.

## Configuration

| Role | Cores | Parallelism | `max_num_seqs` | HTTP port | NixL side-channel |
| --- | --- | --- | --- | --- | --- |
| Prefill | 0-3 | attention TP4 (DP1), MoE EP2 | 1 | 8100 | 5559 |
| Decode | 32-63 | TP8 DP4 EP32 (1 expert/rank) | 4 | 8200 | 5659 |
| Proxy | — | — | — | 8000 | — |

- **Decode experts/rank:** 32 experts / 32 ranks (TP8 × DP4) → 1 expert per rank.
- **Global decode batch:** 16 (`max_num_seqs 4` × DP4).
- **KV transfer:** `NeuronNixlConnector` with the `LIBFABRIC` backend over
  loopback.
- **Buckets:** `kv_segment_size_buckets [8192]` = `num_batched_tokens_buckets`
  for both roles (one predictable compiled graph per role).

## Prerequisites

- **One 64-core Trn3 instance** with cores 0-3 and 32-63 free.
- **The `openai/gpt-oss-20b` checkpoint** available locally or downloadable.

## Order of operations

Open separate terminals on the same host.

1. **Launch prefill and decode** (`run_prefill.sh` binds cores 0-3,
   `run_decode.sh` binds cores 32-63). They have no startup dependency on each
   other and can start at the same time. Wait for both to report
   `Application startup complete`.
2. **Launch the proxy** (`run_proxy.sh`) once both servers are up. It routes to
   `127.0.0.1:8100` (prefill) and `127.0.0.1:8200` (decode).
3. **Validate** with `run_client.sh` — send requests to the proxy (port 8000),
   never to the prefill or decode servers directly.

## Files

| Script | Purpose |
| --- | --- |
| `run_prefill.sh` | Prefill server (`kv_producer`), cores 0-3, TP4 DP1 EP2, port 8100. |
| `run_decode.sh` | Decode server (`kv_consumer`), cores 32-63, TP8 DP4 EP32, port 8200. Sets `--max-logprobs 0` to skip the decode-side logit gather. |
| `run_proxy.sh` | Proxy router on port 8000; routes prefill → decode over loopback. |
| `run_client.sh` | Health check + sample completion against the proxy. |

## Relation to the 120B recipe

This mirrors the 120B DI setup in
[`../../../120b/mxfp4/multinode_DI/`](../../../120b/mxfp4/multinode_DI/README.md),
with two differences: 20B decode uses TP8 **DP4 EP32** (vs DP8 EP64), and this
recipe co-locates prefill and decode on a single node over loopback rather than
across two EFA-connected instances.
