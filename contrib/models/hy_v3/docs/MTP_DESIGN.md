# MTP speculative decoding for Hy3 on Neuron: design

Status: implemented on branch `hy3-fp8-opt` (framework part on `mtp-spec-decode`); see Results. Goal: lower single-request TPOT,
which the 0.29 profile puts at 27.8 ms per step (`PROFILE_0.29.md`), by using
the MTP (next-n prediction) layer that tencent/Hy3-FP8 ships.

## What the checkpoint and vLLM provide

- `num_nextn_predict_layers = 1`: `model.layers.80.*` holds `enorm`,
  `hnorm`, `eh_proj` ([H, 2H]), a full decoder layer (attention + 192-expert
  MoE + shared expert, FP8) and `final_layernorm`. There is no embedding or
  lm_head of its own: both are shared with the target.
- MTP step (vLLM `hy_v3_mtp.py`, HF `HYV3MTPDecoderLayer`):
  `h = eh_proj(cat(enorm(embed(t_{i+1})), hnorm(h_i)))`, then the decoder
  layer (with its residual), then `final_layernorm`, then the target lm_head.
  `h_i` is the target's final (post-norm) hidden state at position i; for
  further draft steps the MTP output itself is fed back. (The target hands
  over the pre-norm state and the draft applies the target's final norm; see
  Results.)
- vLLM 0.29: `--speculative-config '{"method": "mtp", "num_speculative_tokens": k}'`
  turns the target `hy_v3` config into a draft config with
  `model_type=hy_v3_mtp`, `architectures=["HYV3MTPModel"]`, same checkpoint.

## What vllm-neuron 0.29 lacks

1. `NeuronModelRunner` accepts only `eagle3` and `dflash`
   (`neuron_model_runner.py`, "Unsupported speculative decoding method").
2. Draft classes are looked up in `vllm_neuron.model.registry.get_models()`
   only (`eagle.py`), so a contrib / plugin model cannot provide one.
3. `EagleProposer` hard-codes `method == "eagle3"`, a `3 * hidden_size`
   target-hidden width (synthetic warmup inputs) and draft layer count =
   draft `num_hidden_layers` (for MTP that is the target's 80).

The proposer / draft-model contract itself (fused draft NEFF with the KV
update plus an unrolled recurrent loop, on-device accepted-token extraction,
`(tokens [bs, 1+k], drafts [bs, k], logits)` output, async correction
kwargs) fits MTP unchanged.

## Plan

### A. Framework PR (vllm_neuron, model-agnostic)

1. `model/registry.py`: `register_neuron_model(arch, cls)`; registered
   classes are returned by `get_models()` (and so registered with vLLM in each
   worker). Plugins call it from their `vllm.general_plugins` hook.
2. `vllm/spec_decode/mtp.py`: `MTPProposer(EagleProposer)`:
   - accepts `method == "mtp"`;
   - draft layer names `layers.{L}..layers.{L+n_predict-1}` from
     `n_predict` / `num_nextn_predict_layers`;
   - `get_target_aux_hidden_state_layers()` returns `(L,)`, meaning "the
     target's final post-norm hidden state" (index == `num_hidden_layers`);
   - `alias_target_weights(target_model)` delegates to the draft model, which
     points its embedding / lm_head at the target's (no extra HBM);
   - target hidden width = `hidden_size` (EagleProposer gets a
     `target_hidden_dim` hook instead of the hard-coded `3 * hidden_size`).
3. `NeuronModelRunner`: `method == "mtp"` -> `MTPProposer`, `is_mtp_spec`
   included in `is_spec_decode`; the aux-layer selection and
   `alias_target_weights` calls go through the drafter when it provides them
   (today only DFlash does). Unit tests for the proposer plumbing in CPU mode.

### B. Hy3 contrib (contrib/models/hy_v3)

1. Target (`model.py`): `SupportsEagle3`-style `set_aux_hidden_state_layers`;
   when it contains `num_hidden_layers`, `forward` also returns the final
   normed hidden states for every token (gathered over SP in prefill).
   `@async_speculative_decoding` + rejection sampler when
   `spec_decode_metadata` is given (as in `llama3/model.py`). Decode already
   accepts `S_decode = 1 + k` tokens per request (attention megakernel,
   `moe_tkg`, shared expert switches to CTE above 96 tokens). DCP stays
   restricted to `S_decode == 1`.
2. Draft (`model_mtp.py`, registered as `HYV3MTPModel`): reuses
   `HYV3DecoderLayer(layer_idx=80)` and the FP8 loaders; `enorm`, `hnorm`,
   `eh_proj` (BF16 or FP8 per checkpoint), `final_layernorm`; forward
   mirrors `Eagle3LlamaForCausalLM.forward` (KV update pass, then k-1
   recurrent decode passes, greedy on-device sampling with the shared lm_head).
3. Tests: 4-layer checkpoint with the MTP layer kept (`make_mini.py` option)
   to compare the draft's step logits with HF `HYV3MTP`; acceptance rate and
   TPOT on the full model (MATH-500 / chat prompts, k = 1 and 2).

## Expected gain (estimate, to be measured)

- Draft cost per draft token: one decoder layer + lm_head + sampler,
  ~0.5 ms vs a 27.8 ms target step.
- Verify step at batch 1 with 1 + k tokens: weight-bound, so close to the
  single-token step (more experts get hit; the 12.6 ms routed-expert part grows
  the most).
- With k = 1 and acceptance ~0.7-0.8, ~1.7-1.8 tokens per target step:
  TPOT from 27.8 ms to roughly 16-18 ms. k = 2 reuses the same MTP layer for
  the second draft; its acceptance is typically much lower, so it is measured,
  not assumed.

## Risks

- Acceptance on Neuron numerics (FP8 MLP / attention) is unknown until
  measured.
- MoE in the verify step: 1 + k tokens select up to 8(1 + k) experts, which
  spreads the decode MoE load and may raise the verify-step cost more than
  for a dense model.
- Compile time: the draft adds one fused NEFF per decode bucket.

## Results (2026-10-22)

### Correctness

- 4-layer checkpoint with the MTP layer (`make_mini.py --keep-mtp`,
  `scripts/check_mtp.py`): greedy MTP output equals plain decoding token for
  token, k=1 and k=2, batch 4, 16 tokens.
- Bug found on the way: returning the target's post-norm hidden states as the
  auxiliary output, while the same tensor also feeds the lm_head, made the
  verify step's lm_head rows wrong for every request but the first at batch > 1
  (the norm output itself was correct in a tensor capture; computing the
  lm_head from it on the host gave the right tokens). Handing over the pre-norm
  last-layer output, as the Neuron capture convention does for EAGLE3, fixed it.
- Full model, 20 MATH-500 problems, greedy, 512 tokens: the MTP and plain
  outputs agree on 2/20 and split early on the others at equally plausible
  wording ("coordinates" vs "point", `\[` vs `$$`). Graded with math-verify:
  plain 11/20 correct (7 truncated), MTP 13/20 (5 truncated). Most likely the
  verify step (1 + k tokens per request) is numerically slightly different from
  a single-token decode step and flips near-ties; not yet confirmed with
  per-token logprobs, and a full MATH-500 run with MTP is still to do.

### Performance (full model, latency config, concurrency 1)

`scripts/serve_latency.sh` (`MTP_K=k` for MTP), 20 MATH-500 problems as chat
prompts (no_think, greedy, up to 512 tokens, `scripts/bench_real_prompts.py`),
acceptance from the server's spec-decode counters. "Step" is the median gap
between streamed chunks (one engine step each) for two 256-token completions.

| | plain | MTP k=1 | k=2 | k=3 | k=4 |
| --- | --- | --- | --- | --- | --- |
| **TPOT, real prompts (median)** | 26.3 ms | 30.9 ms | 23.4 ms | 21.0 ms | **20.6 ms (-22%)** |
| tokens per step (1 + accepted / drafts) | 1 | 1.87 | 2.48 | 2.78 | 2.93 |
| acceptance, draft position 1 | - | 87.0% | 86.5% | 85.4% | 85.4% |
| acceptance, position 2 | - | - | 61.1% | 59.4% | 58.9% |
| acceptance, position 3 | - | - | - | 33.4% | 32.3% |
| acceptance, position 4 | - | - | - | - | 16.0% |
| step time (streamed chunk gap, median) | 26.4 ms | 55-59 ms | 60 ms | 60-61 ms | 61-62 ms |
| 256-token completion, total (2 requests) | 6717 / 6719 ms | - | 6221 / 7407 ms | 5389 / 6068 ms | 5579 / 5790 ms |
| TPOT, random prompts 1k / 16k | 26.3 / 27.7 ms | 39.2 / 43.3 ms | 35.3 / 36.4 ms | 32.6 / 39.1 ms | 33.9 / 32.3 ms |
| TTFT, real prompts (median) | 0.16 s | 0.20 s | 0.20 s | 0.20 s | 0.20 s |

Acceptance per position counts the steps whose draft at that position was
accepted (so it is cumulative: position 2 can only be accepted after position 1).

- Each step costs ~55-62 ms for every k (vs 26.4 ms plain), so the gain comes
  only from tokens per step. An extra draft position costs ~1-2 ms per step, so
  k=3 / k=4 still pay off with the single MTP layer reused recurrently
  (positions 3-4 accept 32% / 16%).
- On real chat prompts k=4 lowers TPOT by 22% and k=3 by 20%; k=1 is slower than
  plain. Random-token prompts are hardly ever accepted, so every k is slower
  than plain there. TTFT grows by ~40 ms (the draft's prefill pass).

#### Concurrency 4

Same server and prompts, 40 MATH-500 problems sent 4 at a time
(`scripts/bench_real_prompts.py --concurrency 4`):

| | plain | MTP k=4 | change |
| --- | --- | --- | --- |
| TPOT per request (median) | 53.2 ms | **16.9 ms** | -68% |
| output throughput | 72.1 tok/s | **219.8 tok/s** | 3.05x |
| end-to-end per request (mean) | 16.7 s | 5.5 s | -67% |
| TTFT (median) | 0.27 s | 0.26 s | |
| acceptance, positions 1-4 | - | 85.4 / 59.4 / 33.4 / 16.3% | |
| tokens per step | 1 | 2.94 | |

At batch 4 a plain decode step costs ~53 ms (twice the batch-1 step), while a
k=4 speculative step costs ~50 ms (16.9 ms x 2.94), so verifying 20 tokens
costs about as much as decoding 4 and the ~1.9 extra tokens per step are
almost free. The fixed per-step overhead that dominates at concurrency 1 no
longer shows; MTP pays off much more at moderate concurrency than for a
single request.

Device profile of a spec step (core 0): the draft NEFF takes 0.6-1.0 ms and
the target verify NEFF 28-30 ms of compute, i.e. the device work per step is
about one plain decode step. The rest of the ~55 ms is waiting: an idle gap
before the verify NEFF or a `BARRIER` at its start of up to 32 ms (this core is
ready early and waits for other ranks). The step is bounded by host-side work
on the slowest rank, which the async pipeline hides in plain decode but not in
speculative decoding.

The same holds for the officially supported Llama-3.1-8B + EAGLE3 (TP=8, k=1,
tutorial config): ~25 ms per spec step against ~7-15 ms per plain step, 68%
acceptance, total time -8% / +35% on two requests. So the overhead is in the
framework's speculative-decoding step, not in the Hy3 draft (reported to
vllm-neuron as an issue).

#### With routed-expert decode (`moe_tkg_hits`, 2026-10-23)

`moe_tkg_hits` / `moe_tkg_hits_batch` (README, "Decode: only the routed
experts") compute only the experts routed to each rank in decode steps of up
to `HY3_DECODE_HITS_MAX_T` tokens (default 4 per 3 local experts: 4 at EP=64).
A verify step has (k + 1) tokens per request, so at concurrency 1 with k=3 it
takes the hits path by default, and at concurrency 4 (16 / 20 tokens) only
with a raised limit. `scripts/serve_latency.sh` sets the limit to
4 x (k + 1) when `MTP_K` is given (16 for k=3 is equivalent to the measured
20: no step of that config has more than 16 tokens). Same prompts and client
as above:

| config | TPOT c=1 | out tok/s c=1 | TTFT c=1 | TPOT c=4 | out tok/s c=4 | TTFT c=4 |
| --- | --- | --- | --- | --- | --- | --- |
| plain, `moe_tkg` | 26.3 ms | 38 | 0.16 s | 53.2 ms | 72 | 0.27 s |
| plain, hits | 21.7 ms | 45 | 0.16 s | 34.0 ms | 111 | 0.22 s |
| MTP k=4, `moe_tkg` | 20.6 ms | 49 | 0.20 s | 16.9 ms | 220 | 0.26 s |
| MTP k=3, hits, limit 4 | 13.6 ms | 70 | 0.18 s | 19.1 ms | 198 | 0.26 s |
| MTP k=3, hits, limit 4, `VLLM_NEURON_PRECOMPILE_GUARD_FILTER=portable` | 13.4 ms | 71 | 0.18 s | 18.8 ms | 201 | 0.26 s |
| **MTP k=3, hits, limit 20** | **13.8 ms** | **70** | 0.19 s | **17.1 ms** | **216** | 0.24 s |
| MTP k=4, hits, limit 20 | 13.8 ms | 70 | 0.19 s | 17.0 ms | 218 | 0.25 s |

c=1 tok/s for the `moe_tkg` rows is 1000 / TPOT. Acceptance per draft
position with hits: 85 / 60 / 35% at k=3 (2.80 tokens per step), 85 / 59 /
33 / 16% at k=4 (2.92), the same as with `moe_tkg`.

- Concurrency 1: hits and MTP compound. The plain step gets 4.6 ms shorter
  with hits, but the spec step gets ~20 ms shorter (TPOT x tokens per step:
  ~59 -> ~38 ms). Much of the per-step wait for other ranks is MoE rank
  skew: with all-expert `moe_tkg` the verify step's ranks finish at different
  times, and hits shortens the slowest one. TPOT is 48% below plain.
- Concurrency 4: MTP gives the 3x on its own. With the default limit the
  16-token verify step falls back to `moe_tkg` (198 tok/s, below k=4 without
  hits); with the limit raised it takes the hits path (216-218 tok/s), on par
  with k=4 + `moe_tkg`. So the hits kernel stays faster than `moe_tkg` up to
  16-20 tokens here, past the 4-token default, which was set for plain decode.
- k=4 gives nothing over k=3 with hits; k=3 is the recommended setting.
- The precompile guard filter changes nothing measurable.

#### Long context and FP8 KV (2026-10-27)

85k-token prompts with a 92% shared prefix (`bench_data/prefix`,
`scripts/bench_pd_split.sh`, temperature 0.9): MTP does not run with DCP
(platform gate; upstream vLLM has the same gap, vllm#50391), so MTP runs
without DCP (`scripts/serve_85k.sh`). With BF16 KV that config holds only 4
requests at once; with an FP8 KV cache 8, and decode-only TPOT at c=1 / 4 / 8
is 23.4 / 39.2 / 51.0 ms against 31.9 / 48.2 / 60.0 ms for DCP=8 without MTP
(full table in the README, "The same split with an FP8 KV cache"). MTP gets
2.0 tokens per step there (60 / 29 / 14% per position), against 2.8 on
MATH-500 greedy.

FP8 KV for the draft: the MTP layer is `model.layers.80`, which the 80-layer
calibrated profile lacks. `model_mtp.py` reads entry 80 of the same profile
(and refuses one without it); `calibrate_kv_scales_neuron.py calibrate --mtp`
measures it with MTP on and copies layers 0-79 from the base profile, so the
target's scales (decode compile-time constants) do not change. On the 4-layer
MTP model, FP8 KV + MTP differs from plain decoding only at near-ties, as BF16
KV + MTP does (first differing token, HF FP32 log-probs of the two choices
0.10-0.21 nats apart; identical with prefix caching on).

### Outlook

- With the per-step overhead removed (step ~= 29 ms device time), k=1 at 87%
  acceptance would give a TPOT of ~16-18 ms (~35-40% lower than plain).
- Until then, raising k amortizes the overhead. Without hits k=4 is best
  (TPOT 20.6 ms, -22%, at concurrency 1; 16.9 ms and 3.05x throughput at
  concurrency 4). With hits, k=3 is best: 13.8 ms (-48%) at concurrency 1 and
  216 tok/s (3x) at concurrency 4.
- The hits batch limit is tuned for plain decode. A per-request limit for
  verify steps, or a limit measured at larger batches with MTP, is left open;
  `scripts/serve.sh` (config B, up to 32 requests) keeps the default.
- Long context: MTP together with DCP (per-token ownership and a causal
  per-token mask in the DCP decode path; the NKI CP decode kernel already takes
  up to 8 tokens per request) would combine DCP's cheaper step with MTP's
  tokens per step.
