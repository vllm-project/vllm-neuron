# Hy3 1P1D (prefill / decode disaggregation) scripts

These scripts run one prefill node and one decode node, both trn2.48xlarge at TP=64. KV moves over EFA with Mooncake (`NeuronMooncakeConnector`, vllm-project/vllm-neuron#90) or NIXL (`NeuronNixlConnector`). The test method and results are in `../../README.md`, section "PD test method and results".

| script | what |
|---|---|
| `run_pd.sh <label> <mode> [KEY=VAL ...]` | one run: starts this node's server (and the proxy plus the bench on the prefill node), waits for health, stops everything on exit. Modes: `decode-node` (start first; stays up), `prefill-node` (needs `D_HOST`; `P_INSTANCES=2`: two TP=32 prefill instances), `split` (both roles on one node, TP=32 each), `decode-bench` |
| `pd_serve.sh <prefill\|decode\|decode-bench>` | one role on top of `../serve.sh`: KV transfer config, per-role buckets / KV_CAP, sparse attention, DCP, attention DP, `P_HALF=0/1` (a TP=32 prefill instance on half the node) |
| `pd_proxy.sh` | the proxy matching `CONNECTOR`. Mooncake: `mooncake_connector_proxy.py`, vendored from vLLM v0.29.0 (sends to prefill and decode concurrently), with a 2 s keep-alive expiry, one retry of the prefill POST on a connection error, `text/event-stream` for streamed responses, and session affinity: requests with the same `X-Session-ID` go to the same prefill instance. `P_INSTANCES=2` puts two half-node prefill instances behind it. NIXL: vllm-neuron's toy proxy |
| `pd_bench.sh` | smoke and needle checks, then the 85k / 92% shared-prefix case at the given concurrencies. Reports the prefill and decode hit rates and the MTP acceptance |
| `agentx_trn2.sh` | BENCH plugin: InferenceX AgentX through the proxy, with the `gpu/agentx.sh` settings; sessions carry `X-Session-ID`; scrapes both prefill instances and the decode `/metrics` (see the main README) |
| `pd_acc.sh` | `../needle.py` (12 prompts, 16k-85k) and `../sparse_decode_eval.py` (19 greedy continuations) through the proxy |
| `pd_bench_split.sh` | Qingzhong's `../bench_pd_split.sh` flags through the proxy: P-only (output 1, 10×C requests) and D replays; `P_ONLY=1` runs only P |
| `pd_bench_final.sh` | the Trn2 vs H100 set: `pd_bench_split.sh` P at C = 1 / 4 / 8 / 12, then `pd_bench.sh` at C = 8 with 1024 output tokens, then `pd_bench_dcust.sh` at C = 8 |
| `pd_bench_dcust.sh` | the customer GPU "D-only" table: warm all 80 prompts of the c8 file with output 1, then replay them with the same `cache_salt` (~100% hit on both nodes, nothing to transfer), 1200 output tokens, mean / P99; `PROMPTS_DIR` overrides the prompt directory |
| `gpu/` | the GPU references (vLLM 0.29.0): `Dockerfile`, `gpu_serve.sh` (H100 1P1D), `gpu_single_serve.sh` (single node) and `agentx.sh` (InferenceX AgentX, see the main README) |
| `needle_check.py` | the single needle request of `pd_bench.sh` |

Results go to `$WORK/results/pd/<label>/` and logs to `$WORK/logs/pd_<label>_*.log` (`WORK` defaults to `/mnt/nvme`). `VENV` is the vLLM / Neuron venv, default `$WORK/venv/sdk233`.

Notes:
- Give the toy (NIXL) proxy a routable prefill host. It rewrites `remote_host`, so `127.0.0.1` makes a remote decode handshake with itself.
- A decode server with model env flags such as `HY3_DECODE_MOE_FUSED` or `HY3_DECODE_HITS` needs its own `VLLM_CACHE_ROOT` per setting. The precompile cache restores graphs keyed on code, not on these flags.
