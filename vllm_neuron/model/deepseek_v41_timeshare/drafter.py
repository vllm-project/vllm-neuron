# SPDX-License-Identifier: Apache-2.0
"""The DSpark proposer: this model's own MTP stages as a speculative drafter.

The three `mtp.*` stages -- checkpoint layers 40-42, fed by the attention inputs tapped from
layers 37/38/39 -- already run on every forward and already produce a draft. `model.py`'s
forward computes `drafts = [self.forward_spec(...) for ...]` unconditionally, stores them on
`self.last_draft`, and returns `drafts[0]` when `spec_decode_metadata is not None`. This file
is the other half: the object `NeuronModelRunner` builds for
`speculative_config.method == "deepseek_mtp"`, which hands those drafts to the scheduler.

**Read this before assuming it is wired up: in this build the runner never calls `propose`.**
`vllm_neuron/vllm/spec_decode` lets a model register a proposer and `NeuronModelRunner.__init__`
does build it (`neuron_model_runner.py:749-764`, with `target_model_provider` set for us), but
the consumption side is still Eagle-only, in two places:

    5915:  if self.is_eagle3_spec and aux_hidden_states is not None:   # the only call path
    8325:      assert isinstance(self.drafter, EagleProposer)           # inside it

and `is_eagle3_spec` is set only for `method == "eagle3"`. `vllm_neuron/vllm/` is outside the
integrate loop's editable scope, so no iteration can relax either line from here; two
one-line changes there (gate on `self.drafter is not None` instead of `is_eagle3_spec`, and
drop the assert) are what this file is waiting for. Supplying a `speculative_config` without
them builds a proposer nobody asks, adds the spec-decode shapes to every step, and yields no
drafts -- which is why `check_and_update_config` does **not** supply one yet.

**Exactness.** Speculative decoding here is exactness-preserving the way it always is: the
target verifies every draft token and `RejectionSampler` keeps only the prefix that matches
what the target would have sampled anyway. So a *poor* draft costs acceptance rate and never
output. That matters for one known weakness of reading `last_draft` (below) which would
otherwise have to be fixed before the path could be trusted:

    `forward_spec` seeds the draft block with `output_ids[:, 0] = input_ids[begin+length-1]`
    -- the token that was *fed in* at the last position, not the one this step sampled from
    that position's logits. The reference conditions the first MTP stage on the accepted
    token, which during steady-state decode is the same token one step later. Getting it
    exactly right means running the stages *after* the sampler, i.e. a second pass rather
    than the one `forward` already does inline; until then the first draft token is
    conditioned one position stale, acceptance suffers, and correctness does not.

**What the runner calls.** Duck-typed; `spec_decode/eagle.py` is the reference implementation,
and the consumption side -- `_build_spec_decode_metadata`, the scheduler's
`scheduled_spec_decode_tokens`, `RejectionSampler`, the accepted/rejected bookkeeping -- is
generic once the two lines above are generic.

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
        """The module the runner binds KV caches on — the target itself.

        `get_kv_spec` already declares `mtp.{stage}.attn` beside the 40 backbone layers and
        `bind_kv_cache` binds them, so there is no second `bind_kv_cache` to make here; the
        runner's `self.drafter.model.bind_kv_cache(...)` would be the same call twice.
        """
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

        `raw_sampled_token_ids` is the bonus token the target just sampled. It replaces column
        0 when it is available, so the tensor the runner feeds forward as the next step's
        `input_ids` starts from what was actually accepted rather than from the seed
        (see the module docstring on the one-position-stale seed).
        """
        target = self.model
        if target is None:
            raise RuntimeError("DSparkProposer.propose before load_model")
        drafts = getattr(target, "last_draft", None)
        if not drafts:
            raise RuntimeError(
                "the DSpark stages produced no draft on the forward that just ran, so there "
                "is nothing to propose. `model.forward` builds `last_draft` on every step; a "
                "None here means the forward returned before the mtp.* stages."
            )

        rows = []
        for output_ids, _logits, _confidence in drafts:
            rows.append(output_ids.reshape(-1)[: self.num_speculative_tokens + 1])
        draft_token_ids = torch.stack(rows, dim=0)

        if raw_sampled_token_ids is not None:
            bonus = raw_sampled_token_ids.reshape(raw_sampled_token_ids.shape[0], -1)
            draft_token_ids = draft_token_ids.clone()
            draft_token_ids[:, 0] = bonus[: draft_token_ids.shape[0], 0].to(
                draft_token_ids.dtype
            )

        drafts_only = draft_token_ids[:, 1:].contiguous()
        return draft_token_ids, drafts_only
