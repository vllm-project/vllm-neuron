# DeepSeek-V4.1-Flash — architecture and feature status

Modeling code for `DeepseekV41ForCausalLM`. Deployment instructions, the
compatibility matrix and the test commands are in the
[model README](../README.md); this file covers what the code is and what it
supports.

## Layout

| File | Purpose |
| --- | --- |
| `__init__.py` | Package setup, and the registration that makes the model usable: `register_hf_config()` (teaches `AutoConfig` the `deepseek_v41` model type), `register_architecture()`, `declare_storage_quantization()`, the platform config hooks (`require_segmented_prefill`, `require_kv_cache_ceiling`, `require_multi_rank_budget`, `require_dspark_speculation`) and `register_draft_proposer()`. All run at import time. |
| `config.py` | `DeepseekV41Config` plus the deployment constants — `TENSOR_PARALLEL_SIZE = 32`, `ATTENTION_TENSOR_PARALLEL_SIZE = 8`, `MAX_NUM_SEQS`; the scheme is `tp32` only. |
| `factory.py` | The `nn.Module` registered with vLLM. Validates the config and rejects what the port cannot serve, with a reason. |
| `model.py` | The implementation: attention, MoE, hyper-connections, engram, the DSpark stages, `forward`, `get_kv_spec`, `bind_kv_cache`, `load_weights`. |
| `weight_loader.py` | Checkpoint-to-parameter mapping and sharding. Module names are the checkpoint's own paths, so "is this the weight the reference uses here" is a string comparison against `model.safetensors.index.json`. |
| `drafter.py` | `DSparkProposer`, the in-model speculative proposer registered with `vllm_neuron.vllm.spec_decode`. |
| `kernels/` | The NKI kernels. `kernels/__init__.py` maps each kernel entry point to its call site in `model.py`. The per-op kernels sit at the top level; the megakernels live in **one package per parallelization scheme**, `kernels/tp32/` and `kernels/tp8pp8/`, each with its own copies of `layer_kernel.py` and `draft_kernel.py` (the two megakernels, see below), `mk_mx.py` (the block-scaled fp8/fp4 GEMMs on the tensor engine's MX matmul), `mk_moe.py` (the token-grouped routed experts) and `mk_common.py` / `mk_compress.py` (their helpers); `tp8pp8/` also holds `pp_kernels.py`, the stage boundary. `mk_kernels.py` builds the entries from the scheme's package, and `mk_model.kernel_version` hashes only that package, so work on one scheme cannot touch the other's kernels or their compiled artifacts. |
| `mk_model.py` | The bridge from the modules to the megakernels: the deployment scheme (`PipePlan`), gathers each block's weights into the kernel's argument order once, builds the per-step index columns, selects the specialized kernel entry (phase, archetype, scheme) and runs the block loop, with the stage boundaries under `tp8pp8`. |
| `mk_host.py` | The host-side index arithmetic (window and compressed slot tables, compression groups) over the scheduler's block tables. |
| `chat_template.jinja` | The checkpoint's chat template, applied to cases with `role=user`. |
| `verifier/` | The accuracy harness the integration test drives. Lives here rather than under `test/` because its modules import the model with package-relative imports (`from ..config`), so it has to sit inside the same package. |

## The 8-device deployment

This package is the 8-device copy of `contrib/models/deepseek_v41_trn3` (see the
[model README](../README.md) for the table of differences). Everything below this section
was written for the 16-device, 64-rank port and is kept as it was, with the scheme
renamed: read `tp32` for the flat scheme (the 64-rank text's `tp64`), and the rank counts,
shard sizes and timings quoted below as the 64-rank deployment's. What changed for 32 ranks:

- `config.py`: 32 ranks, scheme `tp32` only (the kernels/`tp8pp8` package stays: the flat
  scheme's batched drafter reuses its body).
- `kernels/tp32/moe_tp_layout.py`: the decode TP x EP layout at world 32 is TP8 x EP4 (4 groups
  of 96 experts, 288 hidden columns a rank, `group_of`); the shared expert's 72 32-column blocks
  are dealt two a core (`shared_blocks`), and every `moe_tp.py` / `b1moe.py` copy reads the
  per-core width from the tensors (`sh_cols`).
- `kernels/tp32/fused/*/**/b1moe.py` (the bs1 decode MoE): `PACK_DOWN` (a core's slots' 288 hidden
  values side by side along the down GEMM's contraction, ceil(SLOTS * 72 / 128) tiles; the scale
  rows by one SWDGE gather a tile, `_pack_consts`), `SHARE_PACK` (the two cores' shared blocks in
  one 64-partition tile, combined after the exchange), `SLOTS = 5` (and `moe_tp.SLOTS = 5`).
- Engram tables in fp8 (`weight_loader.ENGRAM_FP8`): `[rows, 264]` uint8 = 256 e4m3fn codes and 8
  e8m0 scales a row; `mk_common.engram_gather_fp8` / `engram_dequant` and `b1eng.readout`
  dequantize after the gather (code x 2^(e - 127), exact in bf16); `model._engram_dequant` for the
  per-op path.
- `mk_model._FREE_PEROP`: an fp8 linear's per-op weight / scale is released once its MX copy
  exists; `_EXPERT_CACHE_BYTES` / `_ENGRAM_TABLE_BYTES` default to 12 GiB.
- `DEEPSEEK_V41_HBM_TRACE=1` also logs rank 0's HBM before each block's TP x EP MoE copy
  (`mk_model.moe_tp_map`).
- `kernels/tp32/draft_decode/layout.py`: `SH_TP` (the batched drafter's shared expert column-sharded
  over the world) is off by default (`DEEPSEEK_V41_DRAFT_SH_TP=1` turns it on): at 32 ranks the batched
  entry with both `SH_TP` and `MAIN_PROJ_TP` drafted deterministically worse (in512_bs10 acceptance 39 %
  against 47 % with either alone; root cause not found). `DEEPSEEK_V41_DRAFT_MP_TP` switches `MAIN_PROJ_TP`.
- `mk_model.moe_tp_max_t()`: a decode step of up to 64 rows (bs10) runs the TP x EP layout at a
  context of at most 4096 tokens (16 rows above that, the 16-device port's limit: at an 8448-token
  context with ten sequences the 60-row launches' NEFFs, loaded after the KV cache is sized, did not
  fit); `DEEPSEEK_V41_MOE_TP_MAX_T` overrides it.

bs1 decode step (2026-11-03, after the first 8-device version). A step is four NEFF executions where
it was six -- the first fused launch with the step prologue in it, the L08-L23 and L24-L39 + head
launches, and the drafter with the step's verify in it:

- **The verify in the drafter launch** (`kernels/tp32/draft_decode/draft_seq.draft_tail_body`,
  `mk_kernels.draft_tail`, `mk_model.drafts_tail` / `run_draft_tail`, `DEEPSEEK_V41_DRAFT_TAIL`): for a
  one-sequence decode step (asynchronous or synchronous, 1 .. K drafts) the drafter compares the step's
  input rows 1 .. n with the fused head's tokens of the rows before them, writes the plugin rejection
  sampler's `[1, n + 1]` output (-1 past the first mismatch, the bonus when all match) and makes its own
  meta (the last accepted token, its row, the prologue's start); it reads the three taps as the blocks
  returned them. That replaces `model._decode_tail_graph` before it (one NEFF execution a step less).
  The kernel is launched straight on its executor (`DEEPSEEK_V41_DRAFT_DIRECT`; through the compiled
  entry that was ~2.2 ms of host time a step), its three read outputs fresh every launch, the sampling
  positions a resident identity column (vLLM samples every row of a one-sequence step: the runner's are
  `arange(n + 1)`, int64, and the executor's direct launch segfaults on an int32 view of them).
- **The step prologue as the first fused launch's first member** (`DEEPSEEK_V41_PROLOGUE_FUSED`,
  `kernels/tp32/fused/layers_decode_pro_embed_L00_L07_tp32/`, `mk_kernels.step_prologue_member`): both
  cores compute it (`step_prologue_body(both_cores=True)`) and write the same values, so every core's
  reads in the members after it follow its own writes, with no core barriers. One NEFF execution a step
  less. `=0` runs the prologue on its own before `layers_decode_embed_L00_L07_tp32`, as before.
- **The drafter kernel** (`draft_decode/draft_seq.py`, `mk_common.py`; exact: the same drafts,
  confidences and window rows on the recorded live case, workspace p10 1.741 -> 1.633 ms):
  `mk_common.HC_MIX_STACK_ROWS` (hc_in's mix GEMM takes x^T from PE transposes of 128-row stacks of the
  rows' K-tiles, 7 a call for 5 rows instead of 160, the same matmuls in the same order; core 1 ran
  ~240 us of transposes and fp32 matmuls a stage), `HEAD_SPLIT` (the markov token loop on each core's
  half of the rank's vocabulary: the logits + pad bias on chip, two argmax candidates a rank in the
  all-gather), `KV_ROWS_CORE1` (the next stage's main-stream KV write's norm / RoPE / scatter on core 1:
  on core 0 it ran past the world all-reduce into the next stage's hc_out and norm), `BIAS_CAST_ACT`
  (the markov bias's bf16 rounding on the scalar engine, pipelined with the vector engine's adds).
  Two non-exact options stay off: `HC_MIX_SPREAD_ROWS` (the mix in a transpose-free order with the
  block-diagonal GEMM) and `mk_common.NORM_SS_SPREAD` (hc_in's rms on chip): ~85 us more, but the
  drafter's MoE routing amplifies the reordered fp32 sums into other drafts, and on the bench that
  moved acceptance (in512 51.3 -> 48.9 %) and, through the verify rows' MoE slot order, some texts.
- **The greedy head** in the L24-L39 + head launch (`fused/layers_decode_L24_L39_head_tp32/l1/head/
  head_kernel.py`; exact): `HEAD_ARGMAX_SPLIT` (the pad mask and the (max, index) on each core's half of
  the rank's vocabulary, one all-gather of both cores' (value, index) pairs -- `2 * world` candidates a
  row in vocabulary order -- instead of core 0 over the whole row and two all-gathers) and `HEAD_SPREAD`
  (the collapse and the norm's elementwise parts on the rows' column blocks over 120 partitions, the
  sum of squares on the rows as before).
- `DEEPSEEK_V41_DRAFT_FUSED=1` (off; `run_fused(draft=True)`, `step.draft_in` / `draft_out`): the
  drafter as the last member of the L24-L39 + head launch. Tried and slower: that launch took longer
  than its two parts and the drafter gap together (5.92 ms against 4.23 + 1.52) and its 755-tensor IO
  ring, rebuilt every step for the fresh outputs, cost ~5 ms of host time a step. Its optimized launch
  is not in the tree (the generated one runs).

Decode launches optimized on the live recorded launch (2026-11-04). `layers_decode_L24_L39_head_tp32` went
through four AutoHelix iterations on the live server's recorded launch (all 32 ranks, in1024 bs1 and bs10;
bit-exact, the outputs and the written cache rows): p10 4.270 -> 3.967 ms at bs1, 15.596 -> 15.536 ms at bs10.

- **The head GEMM in the PE's column-tiling mode** (`l1/head/mk_compress._gemm_ct`): x^T `[128, 32]` is the
  stationary in 4 column tiles, each taking its own 512-column weight tile (2 x 64 for up to 64 rows), on the
  head rows K-major (`HEAD_W_KMAJOR` in the launch package: `mk_model.head_kmajor` passes a `[dim / 128, 128,
  part]` copy, +42 MB a rank). The head's PE time went from 69 to 9.4 us.
- **The L31 -> L32 hand-off** across the merged launch's two parts (`l0.NEXT_FIRST`, `l1.INHERIT_ALL`), as
  every other pair of blocks hands over: L31 prefetches L32's first weights and leaves its stream in SBUF.
- **Scheduling in the members** (`b1.py`, `b1attn.py`, `b1moe.py`): `b1.REST_AFTER_Z` (hc_in's Sinkhorn tail
  behind the attention's z staging: it delayed the z exchange both cores wait for), `b1attn.PREFETCH_ENG` (the
  "ar1done" prefetch on the sync queue, behind the AR-8 result load, instead of the ACT queue, where it ran
  during the AR-8), `b1moe.WS_BUFS = 3` and `UP1_AFTER_S0` (slot 0's scale gather ahead of slot 1's weights on
  the quadrant DMA rings), `b1.HCF_AT = "ar1done"` (hc_in(ffn)'s constants right behind the AR-8 result load)
  and `b1.BDM_LOCAL` (hc_mix's diagonal mask made at each use instead of a launch constant the compiler
  spilled and reloaded late).

The member changes are in the L08-L23 launch and the prologue + L00-L07 launch too; their r0 / r2 members,
which prefetch at "down", keep the ACT queue (with the sync queue L08-L23 was ~45 us longer). Live, rank 0's
device time of a bs1 step's launches (system trace, median of 25 steps) went from 2286 / 4232 / 4226 / 1522 us
(prologue + L00-L07, L08-L23, L24-L39 + head, drafter) to 2252 / 4104 / 3966 / 1522 us; two traces of the same
NEFF differ by up to ~45 us.

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
`layer_prefill_r2_src_idx_tp32`, `layer_decode_r1_idx_tap_tp32`,
`draft_prefill_tp32`, `engram_tp8pp8`, `pp_boundary_even_sender`, ... The phase
is decided per step (a decode step's spans have `1 + dspark_block_size` rows):
prefill kernels keep the slot table transposed in SBUF and select keys on chip,
decode kernels use the per-token transposes (cheaper for a handful of tokens).
The archetype is the block's `(compress_ratio, kv source, index source, engram,
DSpark tap)`; the eight that occur are traced on first use. The scheme is
`DEEPSEEK_V41_SCHEME`:

* `tp32` (default): the flat world. The attention is sharded 8 ways and
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
one package each under `kernels/tp32/`, named `<archetype>_<phase>`, with its own
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

The index sources' scores and top-k in the prefill kernels (`r1_idx`, `r1_src_idx`,
`r2_src_idx`, `r2_src_idx_e`) and in the `r1_src_idx` decode kernel (block 20) carry
the fused decode launches' indexer (`mk_compress.py` in those packages):
`index_scores_span` scores `128 // nh` tokens a pass (32 at the 4 index heads a rank;
their heads side by side on the stationary, the head-weighted sum one matmul against a
block-diagonal weight) instead of one token a dynamic iteration on 4 of the PE's 128
columns, and `index_topk` builds its 16-partition snake tiles from contiguous row loads
and PE transposes and stores the picks as 64-byte runs instead of 2-byte / 4-byte
element-strided DMAs (~14 us a token at an 8k context). Same values: every score is the
same dot products and per-token sum (plus exact zeros), the top-k sees the same bf16
tiles.

**Optimized decode drafter (flat scheme).** A decode step's drafter -- `draft_decode_tp32`
for a step of one sequence, `draft_decode_batched_tp32` for a step of several -- runs the
package `kernels/tp32/draft_decode/` (`draft_seq.py`, `draft_batched.py`, its own helper
copies), produced in the standalone workspace ~/ds41_dspark_draft (the live server's
recorded drafter inputs and golden outputs on all 64 ranks, an AutoHelix loop): bit-exact
on the recorded cases -- drafted tokens, confidences, window-cache rows -- and ~2.7x
faster there. Besides scheduling (the three stages' hyper-connection and norm work split
over both cores and moved into the collectives' shadows, dense per-sequence attention
instead of a per-token loop, live-row embedding all-reduce, the markov head in one
all-gather per token) it lays two weights out differently (`draft_decode/layout.py`,
applied once by `mk_model.draft_weights_decode`): `main_proj` is column-sharded over the
64 ranks (each rank projects its 80 columns, then an all-gather; 1.2 MB a rank instead of
79 MB of replicated weight), and the shared expert's gate / up projections likewise (36 of
2304 hidden columns a rank, its SwiGLU slice all-gathered for the down projection, which
stays replicated so every sum is the original's). The prefill drafter keeps the original
kernels and weights. `mk_kernels.draft` / `draft_batched` route the decode phase there
(`draft_optimized`), `mk_model.kernel_version("draft_decode")` hashes the package, and
`DEEPSEEK_V41_OPT_KERNELS` without `draft_decode` (or `none`) brings the original drafter
back.

Since 2026-11-01 the drafter is no longer bit-exact: its Sinkhorns take 6 vector-engine
instructions an iteration on eps-padded matrices (`mk_common._sinkhorn_fused`, `SINKHORN_FUSED`)
and its attention / world all-reduces carry the live rows instead of the padded 128-row tile
(`AR_LIVE_ROWS`; their fp32 summation order changes with the payload): the recorded cases' drafted
tokens and window-cache rows are unchanged, the confidences (unused by the proposer) move by up to
11 bf16 ulps, and the per-sequence launch is ~7 % faster.

**Decode MoE in a TP x EP layout; the few-row chains (flat scheme).** A bs1 decode step's blocks run
T = 6 rows (one sequence: the token and its 5 drafts). Two things bounded them: the routed experts'
imbalance -- in the deployment layout (6 experts a rank) a step's 36 picks hit a few ranks, and the
world all-reduce after the MoE waited for the busiest -- and chains of tiny dependent instructions
(the hyper-connections' 20-iteration Sinkhorns above all, ~50 us each on few rows). The optimized
fused decode launches now, for a decode step of at most `mk_model.moe_tp_max_t()` rows:

* run the MoE in a **TP x EP layout** (`fused/<launch>/moe_tp.py`; host layout
  `kernels/tp32/moe_tp_layout.py`, built once a block by `mk_model.moe_tp_map` from the checkpoint,
  checked against the resident copy; `DEEPSEEK_V41_MOE_TP=0` keeps the deployment's): 8 groups of 8
  ranks, a group's 48 experts split 8 ways by hidden columns (288 a rank), the shared expert split
  over the 64 ranks (one or two 32-column blocks a rank), the router weight in bf16 (exact). Each
  core runs `SLOTS` statically unrolled slot pairs (a dynamic loop costs ~10 us an iteration on this
  hardware: every engine drains and handshakes) and a rarely-entered loop for the rest; a slot's
  weights take 4 DMAs (the scale rows by SWDGE gathers; padding slots name an out-of-bounds id and
  their DMAs are skipped); the two cores swap the slots' staged activations (`nisa.sendrecv`, the T
  live columns) and each accumulates every slot's down projection for its half of the output
  columns in four PSUM banks (`MOE_CS`), so there is neither an SBUF accumulator nor a cross-core
  combine;
* run the **Sinkhorn** as 6 vector-engine instructions an iteration (eps-padded 5 x 5 matrices: a
  reduce over a row is `sum + eps`) instead of ~8 across DVE and ACT, and place each where core 1
  has slack (hc_in(attn)'s after the attention, schedule-pinned instead of gated by a DMA through
  the queues);
* compute hc_in's **mix GEMM** for few rows block-diagonally (J K-tiles a matmul) from rows read
  straight from HBM and transposed on the tensor engine; the hc mix weights `fn.T` arrive tiled
  for one contiguous DMA (`mk_model.hc_fn_map`); the RMS norm in the spread layout;
* split the **engram** layers' fp8 `wkv` projection (`[6144, 25600]`, ~157 MB, which every rank
  streamed whole) over the attention group and all-gather the parts (`mk_model.engram_tp_map`);
  read the compressor's fp32 projection weights as their exact bf16 values
  (`mk_model.compress_bf16_map`); all-reduce the index scores over the T live rows only; give the
  attention / world all-reduces DMA priority 0.

Only the order of the fp32 sums changes (the down projections' partials across ranks and cores,
the all-reduces' payloads); every quantized value is the deployment's. The fused launches'
outputs move by a few percent relative L2 against the bit-exact ones -- what any summation-order
change makes after fp8 re-quantization -- and the verifier and the integration test pass.

A padding slot (past the group's active experts) must not issue an out-of-bounds dynamic DMA on a
hardware DGE queue: `oob_mode.skip` works in every single-launch workspace, but in the model -- five
such launches a step -- it faulted the sequencer (`BAD_OPCODE` on core 1, "missing collectives
status ... hang after collectives op 1") and hung the first decode step. Pair 0 keeps its hardware-DGE
loads (nearly always active); the other pairs load their rows by software-DGE gathers whose
out-of-bounds rows are skipped (`SLOT_SW_FROM`), which is safe. `nki.experimental.if_else` (to skip
the overflow loop's ~8 us entry / exit when no group overflows) does not compile here.

The single-sequence **decode drafter** runs its three stages' MoE the same way
(`draft_decode/moe_tp.py`, the launches' copy with `SLOTS` 3: a stage routes 5 rows x top-3 of 128
experts; `layout.moe_tp`, its routed slices read from the checkpoint by
`mk_model.draft_weights_decode(tp=True)` and checked against the resident copy;
`DEEPSEEK_V41_DRAFT_MOE_TP=0` keeps the EP64 layout, as the batched entry always does), its router
in bf16. The MX matmuls need an even stationary width, so an odd row count runs the expert GEMMs on
T + 1 rows (the next row is zero, its routing weights zero) and stores T. A bs1 decode step is
~0.34 ms shorter; the drafted tokens follow the new summation order (one of five recorded draft
tokens flips at a near-tie).

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
* Optimized fused launches (`kernels/tp32/fused/<launch>/`, `mk_kernels.optimized_fused`):
  a decode step's launch can be replaced by one optimized as a whole -- written out from
  the model's plan and run through an AutoHelix loop on the live server's recorded
  inputs and golden outputs of all 64 ranks (bit-exact: same arithmetic and accumulation
  order, only scheduling and data movement changed; since 2026-11-01 the decode launches also
  change the order of fp32 sums -- "Decode MoE in a TP x EP layout" above). Each package holds the launch
  (`<launch>.py`, the generated launch's parameters and outputs), its own copies of the
  members' kernels (so a loop can change them for that launch: share the slot / index
  tables and index keys across the blocks reading the same top-k, keep the previous
  block's output tiles in SBUF, prefetch the next block's projection weights during this
  block's all-reduce, split the few-row hc / MoE / indexer work over both cores) and
  `plan.json`, the model plan it replaces (each member's kernel and arguments, the
  parameter count, the returns). `_plan_fused` runs it only for a plan that is
  `plan.json` exactly (else the generated launch, with a warning naming the
  difference); its entry carries a hash of the package's files (the NKI compile caches
  key a kernel on its own source). `DEEPSEEK_V41_FUSED_OPT` (`all` by default, `none`,
  or a comma list of launch names) selects which run.
* The first launch opens with the token embedding and the hyper-connection expand
  (`tp32/head_kernel.embed_expand_body`, `DEEPSEEK_V41_FUSE_EMBED`), the last closes
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

**On-chip decode blocks and merged launches (flat scheme).** In the optimized decode
launches a 6-row span's whole block runs on chip (`DEEPSEEK_V41_B1`, default 1;
`kernels/<scheme>/fused/<launch>/r?_decode/b1*.py`). The residual stream stays in SBUF between
consecutive members in a 128-partition layout, the hyper-connection maps run as tensor-engine
matmuls (the hc mix weights as a hi | lo bf16 pair, `fn_bd`), and the index sources'
queries, scores and top-k (`b1idx.py`), the compressors (`b1src.py`), the engram gate
(`b1eng.py`) and the MoE (`b1moe.py`, a core's experts in static slots plus a dynamic overflow
loop) stay on chip too. A block's DMAs are issued after the previous block's all-reduce
result (`b1.GATE_DMAS`): hoisted into the all-reduce window, they slowed the collective. Blocks
8-23 and 24-39 are one launch each (`DEEPSEEK_V41_FUSE_GROUPS`, default `8,16,16`;
`kernels/<scheme>/fused/merge_launch.py` merges two optimized launches into one), which saves two
launch entries a step, and the merged 8-23 launch hands the stream from block to block across
its two halves on chip. The norms take the hardware rsqrt as is (no Newton step; it is within
1 ulp). `DEEPSEEK_V41_B1=0` falls back to the earlier members.

**A decode step's inputs are made on the device, and the engine schedules ahead (flat
scheme).** With the launches fed, the device's idle time between two decode steps was
the host's: read the drafts back, return the output, the engine's scheduling and RPC, the
runner's metadata, the step's index columns and n-gram hash, the first launch's
submission -- 4.4-5.1 ms a step at in512 bs1 on the live system trace, plus 1.5-8 ms the
first fused launch waited at its entry barrier for the slowest of the 64 hosts. Two
changes remove it:

* `kernels/tp32/step_kernel.step_prologue` (`mk_model.StepPrologue`,
  `DEEPSEEK_V41_STEP_PROLOGUE`, default on): one small launch before the blocks writes,
  in place into the step's persistent buffers, every input of the fused launches that
  depends on the step -- the three `(ratio, window)` sets of `mk_host.AttnIndices`
  columns (row positions, write / window / compressed slots, the compressors' groups),
  both engram layers' n-gram row ids and the embedding's token column -- from one
  upload of the spans' starts and block tables and the device's own token ids. The
  n-gram hash is the reference's 64-bit arithmetic in exact int32 GpSimd ops (products
  from a per-token table, XOR on the 32-bit words, Horner's rule over 7-bit chunks for
  the prime moduli, the quotient from an fp32 estimate corrected by one); the per-slot
  token history it reads is on the device (a prefill chunk, hashed on the host, puts its
  tokens there, `history_put`). `test/unit/test_step_kernel.py` checks every column
  and both layers' ids against `mk_host.build_attn_indices` and
  `NgramHashState.hash_spans` on random steps, bit for bit.
* Async scheduling (`DEEPSEEK_V41_ASYNC`, default on; `config.async_decode`): vLLM
  schedules step N+1 while step N runs, with positions that assume every draft of step N
  was accepted. The runner passes the step its previous step's rejection output
  (`prev_sampled_token_ids`, `_async_spec_decoded`); the prologue subtracts each
  sequence's rejected drafts from its start on the device. Step N+1's tokens are step
  N's drafter output (`[seed, drafts...]` a sequence, which `DSparkProposer.propose`
  returns as a device future), the drafts it verifies are those input rows
  (`_verify_drafts`), the drafter reads the prologue's positions and slots, and nothing
  of a step is read back by the host before the next is submitted. Host-written buffers
  a queued launch still reads are ringed (the prologue's meta) or kept referenced
  (`keep_alive`), and before a step's launches the host waits for the step two back
  (`steps_ahead_wait`, `DEEPSEEK_V41_STEPS_AHEAD`), so the runtime's execution queue
  (`NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS`, at most 63) cannot overflow.
  The runner side is two changes in this branch's plugin (`RUNNER_CAPABILITIES` in
  `neuron_model_runner.py`): the step's host positions, slot mapping and block table for
  a model that plans on the host (`wants_host_step_inputs`, so `_step_rows` reads
  nothing back), and the draft replay across batch-composition changes for a
  model-registered proposer whose drafts are a device future (`device_draft_futures`:
  without it the step after a prefill would verify the scheduler's -1 placeholders).
  Without the second the port keeps synchronous scheduling.
* The step's tail is one graph (`model._decode_tail_graph`, `DEEPSEEK_V41_FUSE_TAIL`, default
  on): the rejection sampler, the drafts' seeds and taps, the drafter's meta and the taps'
  concatenation, which were five executions of ~20 us with ~40-100 us of device idle
  around each; and the drafts-only block the runner wants back is a constant (an async step
  never reads it). A decode step is eight executions: the prologue, the five fused launches,
  the tail, the drafter.
* `VLLM_NEURON_ASYNC_MAX_TOKENS_HOLDBACK` (set by `require_dspark_speculation` under async
  scheduling): the plugin's `NeuronAsyncScheduler` does not schedule a lone request's next
  step while its step in flight may already reach `max_tokens` -- vLLM schedules it unless
  that is certain, and when the request does finish the step runs for nothing while the next
  request's prefill waits behind it (+17-22 ms of TTFT for half the requests of a batch-1
  bench). The runner then knows the next step's positions are exact (an empty step ran in
  between) and passes no correction.

Measured on the live system trace (three ranks, `nc_exec_running`): the device idles
0.68-0.75 ms a decode step at in512 bs1 and at in1024 bs10 -- the gaps between its eight
back-to-back executions -- where the synchronous engine idled 4.4-5.1 ms plus the entry
barrier's wait. `DEEPSEEK_V41_ASYNC=0` serves synchronously (one RPC a step, as before).

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
`DEEPSEEK_V41_MEGAKERNEL=0` restores the per-op path. The equivalence evidence is the
component tests in `test/unit/` (each megakernel piece against its CPU reference) and the
verifier, which the integration test drives.

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
device, and the runtime's queue held 32 executions: eight prefills in flight on
the first stage overflowed it, "Execution Queue Full", 2026-10-24; the port now asks for
128, which the runtime caps at 63, for async scheduling's steps in flight),
`NEURON_RT_EXEC_TIMEOUT=120` turns a device wait into an error, and
`TORCH_LOGS=recompiles` names the frame and the failed guard.

The same guards made the live sequence count a recompile trigger: the forward's head
computed its spans' lengths, the live row count and `len(live_spans)` in traced
code, so every count a bucket had not run with yet -- a request finishing while
others decode -- recompiled four of `forward`'s resume frames, ~1.1-2 s that every
live request waited out (2026-10-30, `max_num_seqs=10`: 18 recompiles in a 6 s
measurement; with faster prefills the requests finish at spread-out times, so it hit
mid-run instead of at the end). `DeepseekV41ForCausalLM._step_rows` computes all of
it outside dynamo; what `forward` reads from it depends on the bucket only.

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
| Structured outputs (`enable_structured_outputs`) | **Not supported** | Rejected by the factory: a grammar truncates the DSpark drafts and the fused decode launches run full 6-row spans only. The grammar mask itself is laid out on the head's padded vocabulary shards (`_vocab_layout_mask`) and applied to greedy rows too. |
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
