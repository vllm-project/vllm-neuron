# DeepSeek-V4.1-Flash — architecture and feature status

Modeling code for `DeepseekV41ForCausalLM`. Deployment instructions, the
compatibility matrix and the test commands are in the
[model README](../README.md); this file covers what the code is and what it
supports.

## Layout

| File | Purpose |
| --- | --- |
| `__init__.py` | Package setup, and the registration that makes the model usable: `register_hf_config()` (teaches `AutoConfig` the `deepseek_v41` model type), `register_architecture()`, `declare_storage_quantization()`, the platform config hooks (`require_segmented_prefill`, `require_kv_cache_ceiling`, `require_multi_rank_budget`, `require_dspark_speculation`) and `register_draft_proposer()`. All run at import time. |
| `config.py` | `DeepseekV41Config` plus the deployment constants — `TENSOR_PARALLEL_SIZE = 64`, `ATTENTION_TENSOR_PARALLEL_SIZE = 8`, `MAX_NUM_SEQS`. |
| `factory.py` | The `nn.Module` registered with vLLM. Validates the config and rejects what the port cannot serve, with a reason. |
| `model.py` | The implementation: attention, MoE, hyper-connections, engram, the DSpark stages, `forward`, `get_kv_spec`, `bind_kv_cache`, `load_weights`. |
| `weight_loader.py` | Checkpoint-to-parameter mapping and sharding. Module names are the checkpoint's own paths, so "is this the weight the reference uses here" is a string comparison against `model.safetensors.index.json`. |
| `drafter.py` | `DSparkProposer`, the in-model speculative proposer registered with `vllm_neuron.vllm.spec_decode`. |
| `kernels/` | The NKI kernels. `kernels/__init__.py` maps each kernel entry point to its call site in `model.py`. |
| `chat_template.jinja` | The checkpoint's chat template, applied to cases with `role=user`. |
| `verifier/` | The accuracy harness the integration test drives. Lives here rather than under `test/` because its modules import the model with package-relative imports (`from ..config`), so it has to sit inside the same package. |

## Architecture

A decoder-only MoE reasoning model. The pieces with no analogue in the dense
reference ports (`vllm_neuron/model/llama3/`) are the ones worth knowing about:

- **Multi-head latent attention (MLA)** with low-rank Q/O and latent KV
  projections, all fp8 block-quantised.
- **A sparse attention indexer.** Per query row it scores `context/ratio`
  candidate slots and attends to a gathered subset rather than the full context.
- **KV compression.** A pooling step writes a compressed KV representation, so
  the resident cache is far smaller than dense attention would need.
- **Hyper-connections (MHC).** Each sublayer's input is expanded to 4 streams,
  mixed through a learned 20-iteration Sinkhorn normalisation, and collapsed
  back — so residual routing is learned rather than fixed.
- **Engram memory.** Large per-layer row-partitioned tables (~384M rows each)
  read through a gate, contributing ~5.7 GiB a rank.
- **MoE.** 384 routed experts, top-6 a layer, plus a shared expert. Routed
  expert weights are fp4; the shared expert and the router are fp8 and bf16.
- **DSpark MTP stages.** `mtp.0-2` are speculative draft stages built from the
  same `Attention` / `MoE` / `BlockQuantLinear` components over their own
  128-expert pool, with Markov and confidence heads. They draft **inside** the
  target forward — they are layers of the target, not a second model — which is
  why they need the plugin's in-model proposer support.

Weights per rank at TP=64, against a 24 GiB bank:

```
dense     ~3.9 GiB    heads/groups 8 ways, vocab 64 ways
experts    4.3 GiB    by expert id, 6 of 384 a layer
engram     5.7 GiB    by row, ~6.0M of 384M a table
          ---------
         ~13.9 GiB
```

## Quantization

The checkpoint declares `quant_method: fp8`, `activation_scheme: dynamic`,
`weight_block_size: [32, 32]`, `scale_fmt: ue8m0`, `expert_dtype: fp4`.
Attention and dense projections are fp8 with `ue8m0` scales over 32x32 blocks;
routed experts are fp4. `kernels/blockquant.py` implements the fp4, fp8 and bf16
GEMMs, including a two-core column split — for fp8 the scale row for a split
starting at weight row `lo` has to be read from `lo // 32`, which
`_fp8_2core_span` handles.

Pass `quantization="fp8"` or leave it unset. Anything else is rejected by
`factory.py`.

## Feature status

| Feature | Status | Notes |
| --- | --- | --- |
| Text generation | **Working** | Greedy and sampled. |
| Segmented prefill | **Working**, required | `require_segmented_prefill` pins it; the port does not support single-shot prefill with prefix caching on. |
| Prefix caching | **Working** | On by default in the integration test. |
| Speculative decoding (DSpark MTP) | **Working** | 5 draft tokens a step. Needs the plugin's proposer registry; without it the model logs that speculation is unavailable and decodes one token at a time. |
| fp8 / fp4 quantized weights | **Working** | The only supported scheme; see above. |
| `tensor_parallel_size = 64` | **Working** | The tested configuration. |
| Concurrency (`max_num_seqs > 1`) | **Not supported** | Kept at 1. Uncapped runs stalled mid-decode in the measured pass; the span and `num_seqs` graph axes are bucketed, but the ragged-draft axis is not yet bounded. |
| Vision tower | **Not supported** | The checkpoint carries one; it is off the tokens-to-logits path and not placed by the floorplan. `factory.py` rejects `mm_encoder_only`. |
| `--runner generate` | **Not supported** | Rejected by the factory. |
| Trn3 | **Working** | SDK 2.33 on `trn3.48xlarge`, the integration test passing; see the model README's Trn3 section for what changed. |

## A note on the comments

The comments are dense and they cite their evidence — most measured numbers say
which host and which configuration they came from, which matters because this
port moved from 4 ranks on one device to 64 ranks on sixteen, and several
premises inverted on the way.

Some of those citations point at development artifacts that are **not part of
this contribution**: `.autohelix/notes/iter-N.md` (the optimization loop's
per-iteration notes), `integrate.yaml` (its constraint definitions) and
`partition_artifact/` (a private reference implementation and its golden
traces). The pointers are left in place because they record where a number came
from, and the conclusion is always stated inline next to the pointer — so no
comment depends on a file you cannot see in order to be understood.

## Self-containment

The model depends only on shared plugin packages and public vLLM APIs:
`vllm_neuron.functional`, `vllm_neuron.nn.sampler`,
`vllm_neuron.nn.rejection_sampler`, `vllm_neuron.model.neuron_config`,
`vllm_neuron.model.kv_cache`, `vllm_neuron.parallel.neuron_parallel_state`,
`vllm_neuron.utils.bucket_utils`, `vllm_neuron.envs`,
`vllm_neuron.vllm.spec_decode` and `vllm_neuron.vllm.worker`. It imports nothing
from another model's folder and writes nothing outside this one.

Two of those imports are guarded, so the model degrades rather than failing if
the plugin does not provide them:

- `vllm_neuron.vllm.spec_decode.{register_proposer, attach_speculative_config}` —
  absent means no speculation, logged explicitly, everything else unaffected.
- `vllm_neuron.parallel.neuron_parallel_state.get_neuron_lane_group` — only
  reached below the default `ATTENTION_TENSOR_PARALLEL_SIZE = 8`, or under the
  `DEEPSEEK_V41_PROBE_AR_WIDTH` latency probe, and it falls back to the world
  group.
