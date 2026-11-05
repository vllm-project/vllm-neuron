# Hy3 1P1D (prefill / decode disaggregation) scripts

These scripts run one prefill node and one decode node, both trn2.48xlarge at TP=64. KV moves over EFA with Mooncake (`NeuronMooncakeConnector`, vllm-project/vllm-neuron#90) or NIXL (`NeuronNixlConnector`). The test method and results are in `../../README.md`, section "PD test method and results".

| script | what |
|---|---|
| `run_pd.sh <label> <mode> [KEY=VAL ...]` | one run: starts this node's server (and the proxy plus the bench on the prefill node), waits for health, stops everything on exit. Modes: `decode-node` (start first; stays up), `prefill-node` (needs `D_HOST`), `split` (both roles on one node, TP=32 each), `decode-bench` |
| `pd_serve.sh <prefill\|decode\|decode-bench>` | one role on top of `../serve.sh`: KV transfer config, per-role buckets / KV_CAP, sparse attention, DCP, attention DP |
| `pd_proxy.sh` | the proxy matching `CONNECTOR`. Mooncake: `mooncake_connector_proxy.py`, vendored unmodified from vLLM v0.29.0, which sends to prefill and decode concurrently. NIXL: vllm-neuron's toy proxy |
| `pd_bench.sh` | smoke and needle checks, then the 85k / 92% shared-prefix case at the given concurrencies. Reports the prefill and decode hit rates and the MTP acceptance |
| `pd_acc.sh` | `../needle.py` (12 prompts, 16k-85k) and `../sparse_decode_eval.py` (19 greedy continuations) through the proxy |
| `needle_check.py` | the single needle request of `pd_bench.sh` |

Results go to `$WORK/results/pd/<label>/` and logs to `$WORK/logs/pd_<label>_*.log` (`WORK` defaults to `/mnt/nvme`). `VENV` is the vLLM / Neuron venv, default `$WORK/venv/sdk233`.

Notes:
- Give the toy (NIXL) proxy a routable prefill host. It rewrites `remote_host`, so `127.0.0.1` makes a remote decode handshake with itself.
- A decode server with model env flags such as `HY3_DECODE_MOE_FUSED` or `HY3_DECODE_HITS` needs its own `VLLM_CACHE_ROOT` per setting. The precompile cache restores graphs keyed on code, not on these flags.
