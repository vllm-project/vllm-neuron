# SPDX-License-Identifier: Apache-2.0
"""The DSpark proposer: this model's own MTP stages as a speculative drafter.

The three `mtp.*` stages -- checkpoint layers 40-42, fed by the attention inputs tapped from
layers 37/38/39 -- run on every forward and produce a draft. `model.py`'s forward runs them
*after* the sampler, seeded from the token the step committed, and leaves the result on
`self.last_draft`. This file is the other half: the object `NeuronModelRunner` builds for
`speculative_config.method == "deepseek_mtp"`, which hands those drafts to the scheduler.

**The whole path, since "a drafter exists" is not "inference is speculative".** Four pieces,
all of them live:

1. `__init__.register_draft_proposer` puts this class in `vllm_neuron.vllm.spec_decode.PROPOSERS`
   under `METHOD`, at import time, and that registration arms the patch that lets vLLM build a
   `SpeculativeConfig` for a drafter with no draft *model*.
2. `__init__.require_dspark_speculation`, on the platform's `check_and_update_config`, calls
   `attach_speculative_config` so the engine has a speculative config even when the caller
   passed none -- which the verifier does not, so the gate's own command exercises this.
3. `NeuronModelRunner.__init__` resolves the method through `resolve_proposer` (both the
   `deepseek_mtp` spelling and the `mtp` that vLLM rewrites it to) and builds this class with
   `target_model_provider` set; `execute_model` then calls `propose` through
   `_propose_draft_token_ids`, and `take_draft_token_ids` carries the result to the scheduler.
4. The next step arrives with `spec_decode_metadata`, the target verifies the `1 + k` tokens in
   one forward, and `model.py`'s `_rejection_sample` keeps the accepted prefix.

**Exactness, as measured rather than as hoped.** The *acceptance rule* is exact: the target
verifies every draft token and the rejection sampler keeps only the prefix that matches what
the target would have sampled anyway, so no draft token can reach the output without the
target agreeing with it, and a *poor* draft costs acceptance rate rather than correctness.
(Through iteration 6 the seed was `input_ids[begin + length - 1]`, the token fed *in* at the
last position rather than the one sampled from it, which anchored the whole block one position
early; `model._draft` now seeds from the sampler's own output -- under speculation from the
last token the rejection sampler accepted -- and anchors the block there.)

What that argument does **not** cover is the arithmetic: a verify step computes `1 + k` rows
per sequence where a plain decode step computes 1, and an op whose per-row reduction the
compiler tiles differently at a different row count gives a different last bit, which an
argmax over 129,280 logits can turn into a different token. Measured on the verifier's own
four prompts, same tree, same cache, speculation the only difference
(`.autohelix/notes/iter-8.md` §3c): **2 of 4 byte-identical, 2 diverging once each** — one
digit of a sector number 20 tokens into the needle, and token ~5 of the France trivia. It was
1 of 4 before `RMSNorm` became a row-count-invariant NKI kernel (`kernels/rmsnorm.py`), which
is the first of the reductions behind this to be fixed and the largest. The remaining ones are
listed in iter-8 §3d, as is the change that would close the gap for all of them at once
(padding the decode token axis to a fixed bucket so the shapes stop depending on the batch).

So: exactness-preserving in its acceptance rule, and **not yet** token-identical on every
prompt. That is a bug to finish fixing, not a trade to accept.

**What the runner calls.** Duck-typed; `spec_decode/eagle.py` is the reference implementation,
and the consumption side -- `_build_spec_decode_metadata`, the scheduler's
`scheduled_spec_decode_tokens`, the rejection sampler, the accepted/rejected bookkeeping -- is
generic.

    __init__(vllm_config, device, on_device_sampling)
    target_model_provider      set by the runner: a callable returning the loaded target model
    num_speculative_tokens     `config.dspark_block_size` = 5, the draft block's length
    model                      the module the runner may bind caches on; the target itself
    load_model(target_hidden_size)   no weights to load -- the stages came with the checkpoint
    warmup(...)                no separate NEFF to warm; the stages run inside the target's
                               own forward, so its warmup is theirs
    graph_extract(...)         likewise nothing of its own to compile
    propose(...) -> (draft_token_ids, drafts_only)
    validate_same_kv_cache_group(kv_cache_config)

**Two things that are already done**, so they are not part of this work: the MTP stages own
paged KV (`model.get_kv_spec` returns `mtp.{stage}.attn`), and their weights load from
checkpoint layers 40-42 (`add_block(f"mtp.{stage}", ...)` in the shard plan).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

#: `speculative_config.method` this proposer answers to.
METHOD = "deepseek_mtp"

#: How many proposals to print in full. Bounded because each one is a device
#: round trip, and because the first few are all a reader needs to see that the
#: drafts are tokens rather than padding.
_LOG_FIRST_DRAFTS = 4


def _on_host(tensor) -> bool:  # noqa: ANN001
    """Whether `tensor` can be read and arithmetic'd without touching the device.

    Everything in this file runs *outside* the model's compiled region, where the
    Neuron backend's eager fallback implements almost nothing — so a device tensor
    is to be left alone rather than inspected.
    """
    device = getattr(tensor, "device", None)
    return device is not None and device.type == "cpu"


class _InModelDraftModel(torch.nn.Module):
    """What `drafter.model` is when the draft stages belong to the target.

    Three places reach for `drafter.model`, and each of them is written for a
    *separate* draft model:

    * `neuron_worker._get_byte_used_from_model` adds its parameters and buffers
      to the bytes already in use before sizing the KV cache;
    * `NeuronModelRunner.initialize_kv_cache` binds the caches on it, unless it
      `is` the target;
    * `NeuronModelRunner.get_kv_cache_spec` asks it to declare its KV layers,
      unless it `is` the target.

    The last two opt out by identity, so returning the target itself looks like
    the right answer — and it is what makes the first one **count the target's
    weights twice**. Measured on this host, the first thing that happened when
    speculation was switched on: 10.35 GiB of resident weights counted as 20.7
    of a 24 GiB bank, leaving a 0.90 GiB KV budget, and

        RuntimeError: Computed KV cache budget is below minimum threshold.
        effective=0.90 GiB, minimum=1.00 GiB

    before a single cache existed. `neuron_worker.py` is outside this port's
    editable scope, so the answer is given here instead — and it is the *true*
    one in all three cases rather than a way around the check. This drafter has
    no weights of its own (the `mtp.*` stages are checkpoint layers 40-42 of the
    target, already counted in its parameters), no caches of its own to bind,
    and no KV layers of its own to declare: the target's `get_kv_spec` already
    names `mtp.{stage}.attn` beside its 40 backbone layers.
    """

    def bind_kv_cache(self, kv_caches) -> None:  # noqa: ANN001
        """Nothing to bind: `model.bind_kv_cache` already bound the stages."""
        return None

    def get_kv_spec(self):
        """No layers of this drafter's own; the target declares the stages."""
        from vllm_neuron.model.kv_cache import KVSpec

        return KVSpec(layers=[])


class DSparkProposer:
    """Hands `model.last_draft` to the scheduler as a speculative proposal.

    There is no draft *model* here and nothing to load or compile: the stages are part of the
    target's forward, so this class is bookkeeping around a tensor the forward already
    produced. Everything expensive -- the three hyper-connected 128-expert stages, the Markov
    bias, the confidence head -- has already happened on the device by the time `propose` is
    called.
    """

    def __init__(self, vllm_config, device, on_device_sampling) -> None:  # noqa: ANN001
        self.vllm_config = vllm_config
        self.device = device
        self.on_device_sampling = on_device_sampling
        #: Set by `NeuronModelRunner.__init__` right after construction, because a proposer is
        #: built before `load_model` has put a model anywhere.
        self.target_model_provider = None
        self._target = None
        #: What `.model` reports. See `_InModelDraftModel`: not the target, because
        #: the worker's memory accounting would then count the target's weights twice.
        self._draft_model = _InModelDraftModel()

        from .config import DeepseekV41Config

        block = DeepseekV41Config.dspark_block_size
        spec = getattr(vllm_config, "speculative_config", None)
        asked = getattr(spec, "num_speculative_tokens", None) if spec else None
        if asked is not None and int(asked) != block:
            raise ValueError(
                f"num_speculative_tokens={asked} but DSpark drafts a fixed block of "
                f"{block} tokens (`config.dspark_block_size`), and the block length is "
                "baked into the Markov head's sequential bias. Set "
                f"--num-speculative-tokens {block} or leave it unset."
            )
        self.num_speculative_tokens = block

    # ── What the runner reads ─────────────────────────────────────────────

    @property
    def model(self):
        """What the runner and the worker treat as this drafter's own model.

        A weightless, cacheless stand-in rather than the target — `_InModelDraftModel`
        has the measurement that makes that necessary.
        """
        return self._draft_model

    @property
    def target(self):
        """The loaded target model, whose `mtp.*` stages produce the draft."""
        if self._target is None and self.target_model_provider is not None:
            self._target = self.target_model_provider()
        return self._target

    def load_model(self, target_hidden_size: int | None = None, **_: object) -> None:
        """Pick up the target. No weights of its own: layers 40-42 loaded with the rest."""
        if self.target_model_provider is None:
            raise RuntimeError(
                "DSparkProposer needs target_model_provider, which NeuronModelRunner sets "
                "when it builds the proposer. Without it there is no model to draft from."
            )
        self._target = self.target_model_provider()
        logger.info(
            "deepseek_v41_timeshare: DSpark proposer bound to the target model; "
            "%d draft tokens per step from the mtp.* stages",
            self.num_speculative_tokens,
        )

    def warmup(self, *_: object, **__: object) -> None:
        """Nothing to warm: the draft stages execute inside the target's own forward."""
        return None

    def graph_extract(self, *_: object, **__: object) -> None:
        """Nothing to compile separately, for the same reason as `warmup`."""
        return None

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:  # noqa: ANN001
        """The draft stages must share the backbone's cache group, and they do.

        Checked rather than asserted in prose: a group split would mean the `mtp` stages got a
        different `block_table`, and they address the backbone's slots (`forward_spec` passes
        the same `PagedSpan`).
        """
        groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
        for group in groups:
            names = list(getattr(group, "layer_names", []))
            if any(name.startswith("mtp.") for name in names):
                if not any(name.startswith("layers.") for name in names):
                    raise ValueError(
                        "the mtp.* draft stages are in a different KV cache group from the "
                        f"backbone layers ({names[:4]}...). They share the backbone's "
                        "block_table, so the groups must be one."
                    )
                return

    # ── The proposal ──────────────────────────────────────────────────────

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor | None,
        last_token_indices: torch.Tensor,
        attn_metadata: dict,
        raw_sampled_token_ids: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(draft_token_ids [bs, 1 + k], drafts_only [bs, k])`, the Eagle contract.

        The draft is read off the target, not recomputed: `forward` ran the three stages for
        every scheduled sequence and left `(output_ids, logits, confidence)` per sequence on
        `last_draft`. `output_ids` is `[1, k + 1]` — the token the block was seeded from
        followed by its `k` drafts — which is exactly the `[bs, 1 + num_spec]` shape the
        runner's async path wants, with `drafts_only` its tail.

        `raw_sampled_token_ids` is what the target just committed: the sampled token on a
        plain step, the rejection sampler's row on a verify step. Column 0 is replaced with
        it where available. That is a no-op now that `model._draft` seeds from the same
        token — the two agree by construction — and it is kept because it is the runner's
        own contract for column 0, and the one place the two could diverge (a host-side
        sampler the model never sees) is the one where the runner is right.
        """
        target = self.target
        if target is None:
            raise RuntimeError("DSparkProposer.propose before load_model")
        drafts = getattr(target, "last_draft", None)
        if not drafts:
            raise RuntimeError(
                "the DSpark stages produced no draft on the forward that just ran, so there "
                "is nothing to propose. `model.forward` builds `last_draft` on every step; a "
                "None here means the forward returned before the mtp.* stages."
            )

        # **Two stacks, and no slice of the stacked tensor.** `torch.stack` of
        # one-dimensional, step-1 slices builds a fresh contiguous tensor; a
        # `[:, 1:]` view of the `[bs, 1 + k]` one does not, and resolving it died
        # on the device at the first batch wider than one row:
        #
        #     RuntimeError: Expected self.is_contiguous() to be true, but got
        #     false   (drafter.py, `draft_token_ids[:, 1:].contiguous()`)
        #
        # — this method runs outside the compiled region, where the backend's
        # eager fallback cannot materialize a strided copy. At `bs == 1` the view
        # *is* contiguous (a size-1 leading dimension has no stride that matters)
        # which is why the long-needle phase passed and the three-prompt phase
        # did not. The runner's own `_propose_draft_token_ids` carries the same
        # warning about `.contiguous()` on Neuron.
        k = self.num_speculative_tokens
        rows, tails = [], []
        for output_ids, _logits, _confidence in drafts:
            flat = output_ids.reshape(-1)
            rows.append(flat[: k + 1])
            tails.append(flat[1 : k + 1])
        draft_token_ids = torch.stack(rows, dim=0)
        drafts_only = torch.stack(tails, dim=0)

        self._report_acceptance(raw_sampled_token_ids)

        if type(self)._logged_rows < _LOG_FIRST_DRAFTS:
            type(self)._logged_rows += 1
            # One `.cpu()` on a `[bs, 1 + k]` int32 tensor, and only for the first
            # few steps: in the synchronous path the runner does the same copy on
            # the next line, so there is no sync here that the step was not about
            # to pay. Worth the lines because "a proposer was called" and "the
            # draft is a plausible continuation" are different claims.
            proposed = draft_token_ids.cpu().tolist()
            logger.info(
                "deepseek_v41_timeshare: DSpark proposal %d — [seed, drafts...] "
                "= %s",
                type(self)._logged_rows, proposed,
            )
            self._check_seed(proposed, raw_sampled_token_ids)
        return draft_token_ids, drafts_only

    # ── Observability ─────────────────────────────────────────────────────

    #: Running totals over the life of this proposer: draft slots offered, and
    #: how many the target's rejection sampler kept.
    _offered = 0
    _kept = 0
    _verified_steps = 0
    _logged_rows = 0

    def _check_seed(self, proposed, raw_sampled_token_ids) -> None:  # noqa: ANN001
        """Warn if column 0 of the proposal is not the token the step committed.

        The Eagle contract is that `draft_token_ids[:, 0]` is the bonus token —
        where the next step starts — and `model._draft` seeds the MTP block with
        it *inside* the compiled region, read from the sampler or (on a verify
        step) from the last valid column of the rejection sampler's row. This
        checks the two agree instead of asserting it in a docstring. Host-side
        and only while the first few proposals are being logged: a device-side
        strided read here is exactly what the eager fallback refuses.
        """
        if raw_sampled_token_ids is None or not _on_host(raw_sampled_token_ids):
            return
        row = raw_sampled_token_ids.reshape(raw_sampled_token_ids.shape[0], -1)
        committed = []
        for entry in row.tolist():
            kept = [t for t in entry if t >= 0]
            committed.append(kept[-1] if kept else None)
        seeds = [r[0] for r in proposed]
        mismatched = [
            (i, s, c)
            for i, (s, c) in enumerate(zip(seeds, committed))
            if c is not None and s != c
        ]
        if mismatched:
            logger.warning(
                "deepseek_v41_timeshare: DSpark seed disagrees with the token the "
                "step committed, as (row, seed, committed): %s. The drafts are "
                "still verified against the target, so this costs acceptance "
                "rate and not output — see model._draft.",
                mismatched,
            )

    def _report_acceptance(self, raw_sampled_token_ids) -> None:  # noqa: ANN001
        """Log how the *previous* step's drafts fared, which is the one number
        that says whether speculation is working.

        Read off the tensor the runner hands over, which on a verify step is the
        rejection sampler's output: a row per sequence, `1 + k` wide, padded with
        `-1`. The count of valid entries is `1 + accepted drafts` — the extra one
        is the token the target would have produced unaided, so a row of exactly
        one valid entry means every draft was rejected and the step still made
        the progress a non-speculative step would have.

        Free: in the synchronous path the runner has already moved this tensor to
        the host (`_execute_model_forward` ends with `.to("cpu")`), so there is no
        device sync here. Skipped rather than forced when it is still on the
        device, because a sync in the middle of a step costs more than the log is
        worth.
        """
        if raw_sampled_token_ids is None or not _on_host(raw_sampled_token_ids):
            return
        row = raw_sampled_token_ids.reshape(raw_sampled_token_ids.shape[0], -1)
        if row.shape[1] <= 1:
            # A prefill or a non-speculative decode step: nothing was verified.
            return
        kept = int((row >= 0).sum().item()) - row.shape[0]
        offered = row.shape[0] * self.num_speculative_tokens
        type(self)._offered += offered
        type(self)._kept += kept
        type(self)._verified_steps += 1
        logger.info(
            "deepseek_v41_timeshare: DSpark verify step %d — %d/%d draft tokens "
            "accepted this step, %d/%d (%.0f%%) over the run",
            type(self)._verified_steps, kept, offered,
            type(self)._kept, type(self)._offered,
            100.0 * type(self)._kept / max(1, type(self)._offered),
        )
