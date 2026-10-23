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
| `kernels/` | The NKI kernels. `kernels/__init__.py` maps each kernel entry point to its call site in `model.py`. `layer_kernel.py` and `draft_kernel.py` are the two megakernels (see below); `mk_mx.py` (the block-scaled fp8/fp4 GEMMs on the tensor engine's MX matmul), `mk_moe.py` (the token-grouped routed experts), `mk_common.py` / `mk_compress.py` are their shared helpers. |
| `mk_model.py` | The bridge from the modules to the megakernels: gathers each block's weights into the kernel's argument order once, builds the per-step index columns, and runs the kernels. |
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

Every backbone block runs as **one** NKI kernel
(`kernels/layer_kernel.py:layer_forward`) and the whole DSpark draft path as
another (`kernels/draft_kernel.py:draft_forward`). Each is a single `@nki.jit`
launched on both physical NeuronCores of a logical core (`nl.program_id(0)`
splits rows or output columns between them) that does its own all-reduces
(`nki.collectives`, over the 8-rank attention group and the 64-rank world), so
a layer is one graph execution and the stream, caches and index state never
leave HBM between sublayers. `mk_model.py` is the bridge: it gathers a block's
weights into the kernel's argument order once (`fn.T` and the router weight as
fp32, the routed experts as one stacked buffer that `MoE.allocate_cache_slots`
now allocates and the slots view into), turns the scheduler's block tables into
the kernels' index columns per step (`mk_host.py`), and hands the top-k, the
compressed/index caches and the DSpark taps down the stack.

The kernels' arithmetic is the reference's (fp8 activation quantization before
the quantized GEMMs, fp32 interiors, bf16 roundings where the model has them)
and is checked component by component against a CPU reference by the unit
tests (`test/unit/test_layer.py`, `test/unit/test_draft.py`). One thing is
deliberately simpler than the per-op path: the candidate-block mask of layer 20
is assumed a no-op, which holds while the context reads at most
`candidate_topk_blocks x candidate_block_size` compressed groups (16384 tokens)
and is asserted.

Three parts of the kernels are written for the Trainium3 hardware directly:

* **Native block-scaled matmuls** (`kernels/mk_mx.py`). The checkpoint's fp8 /
  fp4 weights are microscaled already (one e8m0 scale per 32 contraction
  elements), which is the OCP MX format the NeuronCore-v4 tensor engine
  multiplies natively with `nisa.nc_matmul_mx`: one instruction contracts 512
  elements (128 partitions x 4 packed), dequantizes inside the array and runs at
  four times the bf16 rate (the quad-row successor of Trn2's double-row fp8
  mode; a plain `double_row` matmul has no block scales and cannot take these
  weights). `mk_model.mx_fp8` / `mx_fp4` re-lay each weight once on the host as
  `[K/4, N]` words (`k = 4p + j` in byte / nibble `j`, so a hardware scaling
  group is exactly one checkpoint block) with `[K/32, N]` scale bytes, and the
  kernels stream them into SBUF with plain DMAs; activations are quantized on
  chip as they are staged (four strided PE transposes per 512-chunk into the
  packed layout, then `nisa.quantize_mx`). The resident expert pool is converted
  in place and the per-op layout released (`mk_model._experts_mx`).
* **Keys selected on chip** (`kernels/mk_common.py:attn_token_body_ti`). The
  window and compressed slot rows of a step are gathered from the paged caches
  by indirect DMA into a slot table, DMA-transposed once into a resident
  `[hd, rows]` SBUF table, and each token's QK matmul reads its `window + top-k`
  columns of that table through tensor indirection (`.indirect(index)` on the
  moving operand, the slot indices in the 16-partition snake the engine reads,
  replicated per quadrant) -- no per-token transposes. The values are gathered
  per token by indirect DMA (`.ap(vector_offset=...)`) for the PV matmul. The
  per-token transpose path remains for slot tables too large to keep resident
  (96 KiB a partition, ~12K rows).
* **Token-grouped routed experts** (`kernels/mk_moe.py`). The row ids that
  picked each local expert are compacted on the GpSimd engine
  (`nonzero_with_count` over the mask rows), and every expert runs a runtime-
  bounded `fori_loop` over `ceil(rows / 128)` tiles: gather the rows (indirect
  DMA), stage, w1 / w3, SwiGLU x routing weight, w2, and scatter-add the fp32
  partial into the accumulator with `dma_compute`. An expert nobody picked
  costs nothing, not even its weight reads; no `(row, expert)` pair with a zero
  weight is computed. Both cores run every expert with the same trip count (an
  LNC=2 dynamic loop must) and split each tile along the intermediate axis.
One detail of the in-kernel reduction: the shared expert's weights are
replicated on every rank and the model adds its output once, after the expert
all-reduce; the kernels add it on every rank scaled by `1/world` (an exact
power-of-two scale of the fp32 partial), so the world sum carries it once.
The n-gram row ids reach the engram kernel local to the rank's shard
(`mk_model.StepIndices.hash_cols`): the kernel masks the shard range in fp32, and
a global id of the 384M-row table is not exact above 2^24.
`DEEPSEEK_V41_MEGAKERNEL=0` restores the per-op path. The equivalence evidence
(component R-ratios, a per-layer comparison of the two paths on the real model,
the verifier) is in `test/unit/EQUIVALENCE_REPORT.md`.

## Quantization

The checkpoint declares `quant_method: fp8`, `activation_scheme: dynamic`,
`weight_block_size: [32, 32]`, `scale_fmt: ue8m0`, `expert_dtype: fp4`.
Attention and dense projections are fp8 with `ue8m0` scales over 32x32 blocks;
routed experts are fp4. `kernels/blockquant.py` implements the fp4, fp8 and bf16
GEMMs of the per-op path (weights decoded to bf16), including a two-core column
split — for fp8 the scale row for a split starting at weight row `lo` has to be
read from `lo // 32`, which `_fp8_2core_span` handles. The megakernels multiply
the same weights natively on the tensor engine (`kernels/mk_mx.py`, above); their
activation quantization is the hardware's (`quantize_mx`: scale
`2^(floor(log2 amax) - 7)`, e4m3fn rounding), which keeps one more bit than the
per-op path's `224`-peak round trip in the blocks whose amax lies in
`[1.75, 2) x 2^e` and is otherwise the same grid. The fp8 MX matmul rounds its
four-element partial products internally (about `2.4e-4` relative per GEMM
output, a seventh of the bf16 output rounding; fp4 weights `5e-5`), which the
fp8 re-quantizations downstream amplify, so the two paths agree to about twice
the per-op path's own bf16 noise on a layer's stream.

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
