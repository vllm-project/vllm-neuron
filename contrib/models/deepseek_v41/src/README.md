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
| `kernels/` | The NKI kernels. `kernels/__init__.py` maps each kernel entry point to its call site. `layer_kernel.py` (one `@nki.jit` per layer archetype and phase) and `draft_kernel.py` (the drafter's prefill and decode kernels) are the megakernels; `mk_gemm.py` (Trn2 GEMMs), `mk_moe.py` (token-grouped fp4 experts), `mk_common.py` / `mk_compress.py` are their helpers. |
| `mk_model.py` | The bridge from the modules to the megakernels: re-lays the weights once at load, builds the per-step index columns, and runs the kernels. |
| `mk_host.py` | The host-side index arithmetic (window and compressed slot tables, compression groups) over the scheduler's block tables. |
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

## Megakernels

Every backbone block runs as **one** NKI kernel a step and the whole DSpark
draft path as another (`DEEPSEEK_V41_MEGAKERNEL=1`, the default). It is a port of
the Trainium3 megakernels (branch `deepseek_v41_trn3`) to Trainium2, with two
splits that branch does not make:

- **One `@nki.jit` per layer archetype.** The 40 blocks are nine kinds --
  `window` (0), `window_engram` (1), `c2_source` (2, 8), `c2_source_engram` (14),
  `c2` (the other ratio-2 layers), `c1_source` (20), `c1_index` (24, 28, 32, 36),
  `c1` (the other ratio-1 layers) and `c1_tap` (37-39) -- and each kernel takes
  only the tensors its kind reads (`kernels/layer_kernel.py`, `ARCH`).
- **One kernel per phase.** `*_decode` serves a step of at most 128 rows (B
  sequences x `1 + num_speculative_tokens`) and splits every GEMM across the two
  physical cores by output columns, because a decode step is a weight read;
  `*_prefill` serves a prefill chunk, splits the GEMMs by row tiles (each weight
  tile reused over up to four of them) and scores and selects the index top-k in
  512-token chunks so its scratch does not grow with the prompt. The drafter has
  `draft_prefill` (the first step: the stages only seed their window caches, then
  the head) and `draft_decode` (the three full stages).

Each kernel runs on both physical cores of the logical core and does its own
all-reduces (over the plugin's 8-rank attention mesh and the 64-rank world), so
a layer is one graph execution and the stream, the caches and the index state
never leave HBM between sublayers. `mk_model.py` is the bridge: at load it
re-lays every fp8 linear the kernels read as its dequantized bf16 transpose
(the per-op GEMMs compute the same bf16 values on every call; the kernels read
them as plain DMAs) and stacks the resident fp4 experts on an expert axis; per
step it turns the scheduler's block tables into the kernels' index columns
(`mk_host.py`).

What NeuronCore-v3 does not have, and what replaces it: the trn3 kernels'
block-scaled MX matmul (`nc_matmul_mx` / `quantize_mx`) becomes bf16 matmuls on
the k-major weights with the activations round-tripped through fp8 as they are
staged (`kernels/mk_gemm.py`, peak 224 as in `blockquant`); the routed experts
decode their fp4 weights in SBUF with `kernels/moe_expert.py`'s pipeline inside
a token-grouped, runtime-bounded loop per expert (`kernels/mk_moe.py`); the trn3
tensor-indirection key selection becomes the per-token gather-and-transpose
attention body (`mk_common.attn_token_body`). The values written to the window
KV cache keep the reference's own fp8 round trip (peak 448, rounded
arithmetically: `mk_compress.fp8_qdq_exact_tile`).

The kernels' arithmetic is the reference's where the trn3 port found the per-op
path drifting from it (fp32 router and hyper-connection projections, fp32
compressor projections and pooling caches, fp8 rounding of the attention output
before `wo_a`, fp32 `wo_b` partials into the all-reduce). The unit tests
(`test/unit/test_layer.py`, `test_draft.py`, `test_moe.py`, `test_gemm.py`, on
eight ranks) check every kernel component by component against a CPU reference
fed the kernel's own inputs. One thing is deliberately simpler than the per-op
path, as on trn3: the candidate-block mask of layer 20 is assumed a no-op, which
holds while the context reads at most `candidate_topk_blocks x
candidate_block_size` compressed groups (16384, ~16k tokens at the ratio-1
layers). Past that the mask changes which compressed positions layers 24-36
select, so it is a correctness limit, not a performance one: `mk_model.run_layer`
raises `NotImplementedError` instead of computing a different top-k, and the TODO
there spells out what implementing it in the `c1_source` / `c1_index` kernels takes.
`DEEPSEEK_V41_MEGAKERNEL=0` restores the per-op path.

## Quantization

The checkpoint declares `quant_method: fp8`, `activation_scheme: dynamic`,
`weight_block_size: [32, 32]`, `scale_fmt: ue8m0`, `expert_dtype: fp4`.
Attention and dense projections are fp8 with `ue8m0` scales over 32x32 blocks;
routed experts are fp4. `kernels/blockquant.py` implements the fp4, fp8 and bf16
GEMMs, including a two-core column split — for fp8 the scale row for a split
starting at weight row `lo` has to be read from `lo // 32`, which
`_fp8_2core_span` handles.

The megakernels read the same fp8 weights decoded once, at load, to bf16 in the
tensor engine's moving-operand layout, and the fp4 experts as the per-op expert
kernel does; see [Megakernels](#megakernels).

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
| Megakernels | **Working**, the default | Every block and the drafter as NKI kernels, one per layer archetype and phase; see [Megakernels](#megakernels). `DEEPSEEK_V41_MEGAKERNEL=0` restores the per-op path (choose before the engine starts). |
| Trn3 | **Not tested** | Kernels target Trainium2 (the trn3 megakernels live on branch `deepseek_v41_trn3`). |

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
