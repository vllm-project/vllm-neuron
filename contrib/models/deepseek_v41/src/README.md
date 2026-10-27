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
| `kernels/` | The NKI kernels. `kernels/__init__.py` maps each kernel entry point to its call site in `model.py`. The per-op kernels sit at the top level; the megakernels live in **one package per parallelization scheme**, `kernels/tp64/` and `kernels/tp8pp8/`, each with its own copies of `layer_kernel.py` and `draft_kernel.py` (the two megakernels, see below), `mk_mx.py` (the block-scaled fp8/fp4 GEMMs on the tensor engine's MX matmul), `mk_moe.py` (the token-grouped routed experts) and `mk_common.py` / `mk_compress.py` (their helpers); `tp8pp8/` also holds `pp_kernels.py`, the stage boundary. `mk_kernels.py` builds the entries from the scheme's package, and `mk_model.kernel_version` hashes only that package, so work on one scheme cannot touch the other's kernels or their compiled artifacts. |
| `mk_model.py` | The bridge from the modules to the megakernels: the deployment scheme (`PipePlan`), gathers each block's weights into the kernel's argument order once, builds the per-step index columns, selects the specialized kernel entry (phase, archetype, scheme) and runs the block loop, with the stage boundaries under `tp8pp8`. |
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
(`kernels/<scheme>/layer_kernel.py:layer_forward`) and the whole DSpark draft
path as another (`kernels/<scheme>/draft_kernel.py:draft_forward`). The flat
scheme's own drafter runs once per sequence; the pipeline scheme's copy -- which
the flat scheme also launches for a step of several sequences (below) -- drafts
**every sequence of the step in one launch** (the B
sequences' draft blocks are B x 5 rows of one tile, each with its own window in
the kernel's slot table, and the token loop drafts the i-th token of all of them
at once, so a step's drafting on the last stage costs one kernel's collectives
however many sequences it carries -- drafting ten sequences one by one took
~15 ms each). Each is a single `@nki.jit`
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

**One entry point per phase, archetype and scheme.** The two bodies
(`layer_kernel.layer_body`, `draft_kernel.draft_body`) take their configuration
as baked constants, and `kernels/mk_kernels.py` traces a distinct `@nki.jit`
function for every combination the deployment uses, named for it:
`layer_prefill_r2_src_idx_tp64`, `layer_decode_r1_idx_tap_tp64`,
`draft_prefill_tp64`, `engram_tp8pp8`, `pp_boundary_even_sender`, ... The phase
is decided per step (a decode step's spans have `1 + dspark_block_size` rows):
prefill kernels keep the slot table transposed in SBUF and select keys on chip,
decode kernels use the per-token transposes (cheaper for a handful of tokens).
The archetype is the block's `(compress_ratio, kv source, index source, engram,
DSpark tap)`; the eight that occur are traced on first use. The scheme is
`DEEPSEEK_V41_SCHEME`:

* `tp64` (default): the flat world. The attention is sharded 8 ways and
  replicated across the 8 groups, the 384 routed experts are 6 a rank, every
  expert reduce spans the 64 ranks.
* `tp8pp8`: **pipeline parallelism**, vLLM's own (`tensor_parallel_size=8,
  pipeline_parallel_size=8`, which `config.SCHEME` sets from
  `DEEPSEEK_V41_SCHEME=tp8pp8`). The 64 workers form 8 stages of 8 consecutive
  ranks; a stage builds, loads and runs **only its own 5 blocks** -- the
  attention sharded 8 ways within the stage, 48 of the 384 routed experts a
  rank, every reduce over the stage -- while the embedding lives on the first
  stage (and the last, whose DSpark stages embed the drafted ids) and the norm,
  the vocab-sharded head, the sampler and the drafter on the last. Each stage's
  weights are a separate shard of the checkpoint (`weight_loader`), the engram
  table of a stage's engram block is sharded over that stage's 8 ranks, and
  the KV caches a stage declares (`get_kv_spec`) are its own blocks'. The
  scheduler keeps `pipeline_parallel_size` batches in flight, so the stages
  overlap on different batches; a non-last stage runs its blocks and returns
  nothing to sample (`neuron_model_runner._pp_not_last_stage`), the last stage
  samples and drafts as the flat scheme does. The stages schedule synchronously
  (`async_scheduling=False`: the async scheduler's -1 draft placeholders can only
  be substituted on the sampling stage, and the first stage embeds real ids), and
  the drafter's tokens travel **inside each step's `ModelRunnerOutput`**
  (`neuron_model_runner._attach_pp_drafts` sets `neuron_pp_drafts`, the result
  queue pickles the object; `NeuronScheduler.update_from_output` attaches them
  right after the sampled tokens of the same step, only for requests that
  committed a token in it) and `vllm/patches/pp_draft_patch.py` makes the
  engine's `post_step` skip its `take_draft_token_ids` RPC under PP. That RPC
  is answered by the last stage only after the `execute_model` queued ahead of
  it, i.e. after the step has drained the whole pipeline, so with it the engine
  never scheduled the next batch before the previous one had finished (no two
  batches overlapped), and its drafts reached the scheduler several outputs
  ahead of the step they belong to -- at 8192-token inputs (2048-token prefill
  chunks) a non-final chunk's drafts made the request schedulable with five
  spec tokens and no bonus token, and the last stage died tracing a 1-row
  logits tensor against 6-row sampling metadata. The same patch module sizes
  the executor's RPC ring (`MessageQueue`, 10 chunks by default, reused only
  once every worker has read them, two messages a step) to `4 x
  pipeline_parallel_size` chunks so a full pipeline of steps fits in flight.
  On the stages before the last, the runner's cached request state also gets
  the drafts the last stage accepted (`_update_states`: the scheduler ships
  only the newly scheduled bonus token, and a request re-added to the batch
  after another request's batch ran -- every step under PP with several
  requests -- was rebuilt that many tokens short, so its input slice read past
  its known tokens: garbage text at `max_num_seqs=4`, exact at 1).
  The hand-off between stages is
  in-kernel (`kernels/tp8pp8/pp_kernels.py:boundary_body`): a 2-rank collective per
  tensor-parallel offset carries the hyper-connection stream, `pre_mix`, the
  current index top-k and the rows this step wrote into the current kv
  source's compressed-KV / index-key caches -- one compiled program on both
  sides, the role a runtime 0/1 weight on the payload, and both sides traced from
  identical shapes (the runtime hashes a program's collectives per replica group
  and refuses a pair whose two programs differ, which a placeholder top-k of the
  wrong width once caused; `mk_model.topk_width`); the receiving stage keeps a
  **replica** of that source (`model.src_attn`: the source layer's attention
  module, loaded from the source's own checkpoint keys, with caches the
  boundary scatters into) which its consumers read, and forwards the rows at
  its own exit. The collectives of a stage name their replica groups over the
  full 64-rank mesh and run on the stage's ranks alone while the other stages
  run their own kernels; `test/unit/probe_pp_subset.py` checks that the runtime
  allows exactly that (16 ranks, two stages with different kernels and trip
  counts, then the pair hand-off), and `test/unit/test_pp.py` checks the
  boundary kernel bit for bit. The hand-off is a 2-rank all-reduce (the receiver
  adds zeros); `nki.collectives.collective_permute` is the point-to-point
  primitive that would carry the payload once instead of summing it, and is
  the next step for this seam.

**Optimized block kernels (flat scheme).** Every (phase, archetype) of the flat
scheme runs a hand-optimized kernel instead of the generic body's specialization,
one package each under `kernels/tp64/`, named `<archetype>_<phase>`, with its own
helper copies (18 in all). The plain r2 and r1 blocks' four (r2: 3-7, 9-13, 15-19;
r1: 21-23, 25-27, 29-31, 33-35 -- `r2_prefill`, `r2_decode`, `r1_prefill`,
`r1_decode`) were each produced in a standalone workspace (the live server's
recorded kernel inputs and golden outputs on all 64 ranks, an AutoHelix loop). The
other archetypes' kernels (`r0`, `r0_e`, `r2_src_idx`, `r2_src_idx_e`,
`r1_src_idx`, `r1_idx`, `r1_tap`; blocks 0, 1, 2 and 8, 14, 20, 24/28/32/36,
37-39) are the nearest of those four with the generic body's sections that
archetype has, at the same points of the dataflow -- the engram read-out, the
DSpark tap, a kv source's compressor, an index source's scores / all-reduce /
top-k, window-only attention at ratio 0: r1_prefill (which carries r2_prefill's
optimizations) for every prefill kernel, r2_decode for the r0 / r2 archetypes'
decode, r1_decode for the r1 ones' (its compact top-k slot table matters most at
ratio 1). All compute what the generic body computes: same arithmetic, only
scheduling and data movement changed. The one exception is a summation order: at
bs1 the decode kernels send only the T real rows through their all-reduces instead
of 128 (`CC2_T_ROWS_MAX`, `CC_SMALL_BYTES`), and the collective's order depends on
the message size, so a few fp32 sums can differ by an ulp -- enough to change
greedy text after some tokens (the verifier's short cases), not its quality.
`mk_kernels.layer` routes them (`_KIND`), `mk_model.kernel_version` hashes each
package on its own, `DEEPSEEK_V41_OPT_KERNELS` (`all` by default, `none`, or a
comma list of packages) selects which run, and `test/unit/test_opt_kernels_diff.py`
checks every one against the generic body on 8 ranks (shapes the recorded cases do
not cover, every archetype).

**A decode step is a handful of launches (flat scheme).** A decode block runs in well
under a millisecond on the device, while every NEFF execution with collectives costs
~1.5 ms of host submission at the least -- more the more IO tensors it has (~5 ms for an
8-block launch's ~500, with the runtime's IO ring rebuilt every time,
`NEURON_RT_IO_RING_CACHE_SIZE`; a larger ring cache does not help the engine, whose
per-step tensors are new every step). One launch per block made the step host-bound, so
the step is built from as few launches as keep the device fed:

* `DEEPSEEK_V41_FUSE_DECODE=8` (default; 0 or 1 is one launch a block): each run of 8
  consecutive blocks is one `@nki.jit` launch (`mk_kernels.fused`: the members' own
  kernels called in sequence, their named scratch tensors prefixed per member and
  their unnamed shared-HBM outputs named -- left to the compiler, two index sources'
  same-shape top-k were not kept apart -- and their cache copies off). Five launches
  a step instead of 40; larger groups cost more submission than they save, since the
  step's first block waits for the first launch's submission (20-block groups were
  slower). Prefill keeps one launch a block (device-bound).
* The first launch opens with the token embedding and the hyper-connection expand
  (`tp64/head_kernel.embed_expand_body`, `DEEPSEEK_V41_FUSE_EMBED`), the last closes
  with the greedy head (`head_greedy_body`, `DEEPSEEK_V41_FUSE_HEAD`): collapse,
  final norm, the vocab-parallel head and the argmax over the full vocabulary without
  gathering the logits (each rank's (max, index), all-gathered; ties go to the lowest
  index, as `argmax`'s do). An all-greedy speculative step's tokens come out of the
  blocks' last launch; the separate tail's two `[T, vocab]` fp32 all-gathers and the
  plugin sampler's collectives are gone. Any other step -- a sampled row, a prefill, no
  drafts -- keeps the separate tail, which no longer gathers logits a speculative step
  never reads.
* A fused launch's arguments (~500) are planned once per (blocks, phase, T, B)
  (`mk_model.FusedPlan`); a step refills only its own tensors -- the index columns, the
  n-gram ids, the incoming stream / top-k. `DEEPSEEK_V41_FUSED_PLAN_CHECK=1` rebuilds
  the plan every step and checks the cached one against it.
* The drafter drafts every sequence of a step of several in one launch
  (`mk_kernels.draft_batched`: the pipeline scheme's batched body specialized for the
  world; `DEEPSEEK_V41_DRAFT_BATCHED=0` for one launch per sequence), one sequence
  with the per-sequence kernel. Its int32 ids reach the scheduler in one read-back:
  `DSparkProposer.propose` slices them on the host (stacking device slices was two
  eager device ops a step, no int64 cast graph).
* One RPC a step (`VLLM_NEURON_ONE_RPC_STEP`, set by `require_dspark_speculation`;
  `DEEPSEEK_V41_ONE_RPC_STEP=0` keeps vLLM's three): the runner samples inside
  `execute_model` when no scheduled request needs a grammar, so the engine skips its
  `sample_tokens` RPC, and the drafts ride in the step's own output as they do under
  pipeline parallelism (`neuron_pp_drafts`; `pp_draft_patch` skips
  `take_draft_token_ids`).

`DEEPSEEK_V41_HOST_PROF=1` logs each forward's host time by part and the runner's RPCs
with the waits between them (`DEEPSEEK_V41_HOST_PROF_SYNC=1` waits for the device at
every mark, so a part is its host and device time; `DEEPSEEK_V41_CPROFILE=<start>:<count>`
puts rank 0's worker under cProfile). `test/unit/test_fused_decode.py` checks a fused
launch bit for bit against the same kernels launched one by one (`--embed`, `--head`,
`--time` for the host cost per launch), `test/unit/test_head_kernel.py` the head
against a CPU reference (vocabulary padding, ties across ranks).

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

**The forward's tail is one dynamo frame, on every rank.** With `run_blocks` a
single `torch._dynamo.disable` call, everything after the blocks (the collapse,
the final norm, the vocab-sharded head, the logits all-gather, the sampler, the
rejection sampler, the draft seeds) is traced as one frame, and dynamo guards on
every Python value the frame reads. That state has to be identical on all 64
ranks: a lazily filled per-rank cache -- the padded last vocab shard's `-inf`
bias used to be one -- made a single rank fail a guard on its next
prefill-shaped step, re-trace the frame and recompile its all-gather graph while
the other 63 ranks already sat in the collective, and the engine hung without an
error (the verifier's second prompt, 2026-10-23). The bias is a buffer now
(`vocab_pad_bias`, materialized in `allocate_staging_buffers`). To localize a
hang: `DEEPSEEK_V41_MK_TRACE=1` logs every kernel launch on each stage's first
rank, `DEEPSEEK_V41_MK_IN_FLIGHT=0` syncs after each launch (default: 12
outstanding launches per process, counted across forwards and including the drafter's per-sequence graphs -- under pipeline
parallelism a stage's host launches batch after batch without waiting on the
device, and the runtime's queue holds 32 executions: eight prefills in flight on
the first stage overflowed it, "Execution Queue Full", 2026-10-24),
`NEURON_RT_EXEC_TIMEOUT=120` turns a device wait into an error, and
`TORCH_LOGS=recompiles` names the frame and the failed guard.

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
