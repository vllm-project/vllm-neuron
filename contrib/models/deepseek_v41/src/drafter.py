# SPDX-License-Identifier: Apache-2.0
"""The DSpark proposer: this model's own MTP stages as a speculative drafter.

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
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

#: Verify steps between two acceptance-rate log lines. The figure it reports is
#: cumulative, so a sample of it is as informative as all of them.
_LOG_EVERY = max(1, int(os.environ.get("DEEPSEEK_V41_DRAFT_LOG_EVERY", 16)))

#: `speculative_config.method` this proposer answers to.
METHOD = "deepseek_mtp"

#: How many proposals to print in full. Bounded because each one is a device
#: round trip, and because the first few are all a reader needs to see that the
#: drafts are tokens rather than padding.
_LOG_FIRST_DRAFTS = 4


def _on_host(tensor) -> bool:  # noqa: ANN001
    """Whether `tensor` can be read and arithmetic'd without touching the device."""
    device = getattr(tensor, "device", None)
    return device is not None and device.type == "cpu"


class _InModelDraftModel(torch.nn.Module):
    """What `drafter.model` is when the draft stages belong to the target.

        RuntimeError: Computed KV cache budget is below minimum threshold.
        effective=0.90 GiB, minimum=1.00 GiB
    """

    def bind_kv_cache(self, kv_caches) -> None:  # noqa: ANN001
        """Nothing to bind: `model.bind_kv_cache` already bound the stages."""
        return None

    def get_kv_spec(self):
        """No layers of this drafter's own; the target declares the stages."""
        from vllm_neuron.model.kv_cache import KVSpec

        return KVSpec(layers=[])


class DSparkProposer:
    """Hands `model.last_draft` to the scheduler as a speculative proposal."""

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


    @property
    def model(self):
        """What the runner and the worker treat as this drafter's own model."""
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
            "deepseek_v41: DSpark proposer bound to the target model; "
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
        """The draft stages must share the backbone's cache group, and they do."""
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
        """`(draft_token_ids [bs, 1 + k], drafts_only [bs, k])`, the Eagle contract."""
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

        k = self.num_speculative_tokens
        # One read-back of the drafted ids (a batched drafter's `[B, k+1]`, else one row
        # per sequence), sliced on the host: stacking device slices was two eager device
        # ops a step. Host tensors are what the synchronous runner wants anyway (it reads
        # `drafts_only` back with `.cpu()`); the port runs the scheduler synchronously
        # (`require_dspark_speculation`).
        batched = getattr(target, "last_draft_ids", None)
        if batched is not None and int(batched.shape[0]) == len(drafts):
            ids = batched.cpu()
        else:
            ids = torch.cat([output_ids.reshape(1, -1)[:, : k + 1].cpu() for output_ids, _l, _c in drafts])
        ids = ids.to(torch.int64)
        draft_token_ids = ids[:, : k + 1].contiguous()
        drafts_only = ids[:, 1 : k + 1].contiguous()

        self._report_acceptance(raw_sampled_token_ids)

        if type(self)._logged_rows < _LOG_FIRST_DRAFTS:
            type(self)._logged_rows += 1
            proposed = draft_token_ids.cpu().tolist()
            logger.info(
                "deepseek_v41: DSpark proposal %d — [seed, drafts...] "
                "= %s",
                type(self)._logged_rows, proposed,
            )
            self._check_seed(proposed, raw_sampled_token_ids)
        return draft_token_ids, drafts_only


    #: Running totals over the life of this proposer: draft slots offered, and
    #: how many the target's rejection sampler kept.
    _offered = 0
    _kept = 0
    _verified_steps = 0
    _logged_rows = 0

    def _check_seed(self, proposed, raw_sampled_token_ids) -> None:  # noqa: ANN001
        """Warn if column 0 of the proposal is not the token the step committed."""
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
                "deepseek_v41: DSpark seed disagrees with the token the "
                "step committed, as (row, seed, committed): %s. The drafts are "
                "still verified against the target, so this costs acceptance "
                "rate and not output — see model._draft.",
                mismatched,
            )

    def _report_acceptance(self, raw_sampled_token_ids) -> None:  # noqa: ANN001
        """Log how the *previous* step's drafts fared, which is the one number
        that says whether speculation is working.
        """
        if raw_sampled_token_ids is None or not _on_host(raw_sampled_token_ids):
            return
        row = raw_sampled_token_ids.reshape(raw_sampled_token_ids.shape[0], -1)
        if row.shape[1] <= 1:
            return
        kept = int((row >= 0).sum().item()) - row.shape[0]
        offered = row.shape[0] * self.num_speculative_tokens
        type(self)._offered += offered
        type(self)._kept += kept
        type(self)._verified_steps += 1
        from .model import _LOG_RANK

        if _LOG_RANK[0] and type(self)._verified_steps % _LOG_EVERY == 0:
            logger.info(
                "deepseek_v41: DSpark verify step %d — %d/%d draft tokens "
                "accepted this step, %d/%d (%.0f%%) over the run",
                type(self)._verified_steps, kept, offered,
                type(self)._kept, type(self)._offered,
                100.0 * type(self)._kept / max(1, type(self)._offered),
            )
