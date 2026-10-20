# SPDX-License-Identifier: Apache-2.0
"""``@async_speculative_decoding`` class decorator for target models.

The decorator wraps a model's ``forward()`` to bridge it with
``NeuronModelRunner``'s async spec decode pipeline. Without it, every
target model would re-implement two identical boilerplate steps:

1. **Prologue:** apply on-device ``num_computed_tokens`` correction so
   positions/slot_mapping account for the previous step's rejected drafts.
2. **Epilogue:** extract the last-accepted token from the rejection
   sampler output and return it alongside the standard outputs.

Model author contract:

* Add ``@async_speculative_decoding`` above the class.
* Accept ``positions``, ``attn_metadata`` and ``spec_decode_metadata``
  as named kwargs in ``forward()``; absorb everything else with
  ``**kwargs``.
* Return the original (non-async) shapes from ``forward()`` -- the
  decorator's epilogue appends ``last_accepted_token``.

The runner injects ``prev_sampled_token_ids`` / ``prev_num_draft_tokens``
/ ``req_indices_per_token`` only when async spec correction is needed
(decode steps in async + spec mode). On other paths (prefill, sync,
non-spec) those kwargs default to ``None`` and the decorator's prologue
no-ops. Each path is precompiled by warmup, so ``torch.compile`` finds
a cache hit at runtime.
"""

from __future__ import annotations

import functools

import torch

from vllm_neuron import functional as NF


def _extract_last_accepted_token(
    rejection_sampled_tokens: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Gather the last non-placeholder token per request.

    ``rejection_sampled_tokens`` is ``[bs, max_spec_len+1]`` int32 with
    ``-1`` at rejected positions and the bonus token at column
    ``num_drafts`` per row. The last valid column index per row is
    therefore ``valid_count - 1``.

    Returns a fresh ``[bs]`` int32 tensor with stride ``(1,)``, safe to
    feed as the next decode step's ``input_ids`` directly.
    """
    valid_mask = (rejection_sampled_tokens != -1) & (
        rejection_sampled_tokens < vocab_size
    )
    valid_count = valid_mask.sum(dim=1)
    last_valid_idx = torch.clamp(valid_count - 1, min=0)
    return (
        rejection_sampled_tokens.gather(1, last_valid_idx.unsqueeze(1).long())
        .squeeze(1)
        .to(torch.int32)
    )


def async_speculative_decoding(cls: type) -> type:
    """Bridge ``forward()`` with the async spec decode pipeline.

    See module docstring for the contract. The decorator branches on
    ``prev_sampled_token_ids is None`` to decide whether to call the
    on-device position correction. Each branch is its own
    ``torch.compile`` graph; warmup precompiles both so runtime hits
    a cached graph.
    """
    original_forward = cls.forward

    # Record the owner now. The code below replaces cls.forward, which hides
    # whether this class owned forward or inherited it from a base class.
    cls_owns_forward = cls.__dict__.get("forward") is original_forward

    @functools.wraps(original_forward)
    def forward(
        self,
        *args,
        positions=None,
        attn_metadata=None,
        spec_decode_metadata=None,
        prev_sampled_token_ids=None,
        prev_num_draft_tokens=None,
        req_indices_per_token=None,
        **kwargs,
    ):
        # Prologue: on-device num_computed_tokens correction. Only runs
        # when the runner injected the prev-step kwargs (async decode
        # path). Each branch traces a separate graph by shape; warmup
        # precompiles both.
        if (
            prev_sampled_token_ids is not None
            and prev_num_draft_tokens is not None
            and req_indices_per_token is not None
        ):
            positions, attn_metadata = (
                NF.correct_spec_decode_positions_and_slot_mapping(
                    positions,
                    attn_metadata,
                    prev_sampled_token_ids,
                    prev_num_draft_tokens,
                    req_indices_per_token,
                    self.config.vocab_size,
                )
            )

        result = original_forward(
            self,
            *args,
            positions=positions,
            attn_metadata=attn_metadata,
            spec_decode_metadata=spec_decode_metadata,
            **kwargs,
        )

        # Epilogue: append last-accepted token only on spec-decode steps.
        if spec_decode_metadata is None:
            return result

        rejection_sampled = result[0] if isinstance(result, tuple) else result
        last_accepted = _extract_last_accepted_token(
            rejection_sampled, self.config.vocab_size
        )
        return (
            (*result, last_accepted)
            if isinstance(result, tuple)
            else (result, last_accepted)
        )

    # functools.wraps copies __module__ from the model. A precompile package
    # then restores the graph aliases into that module, not this one. The warm
    # start fails with a NameError. Point __module__ back at this module.
    forward.__module__ = __name__

    cls.forward = forward
    cls._async_spec_decoded = True

    # cls.forward now points at the wrapper, so the guard pickler can't reach the
    # closed-over original_forward by qualname (it saves a sentinel, breaking warm
    # restore). Expose it under its own name; skip the rename for an inherited
    # forward, which a base shares across subclasses.
    cls._original_forward = original_forward
    # Applied only to leaf model classes that own forward; no decorated class
    # subclasses another, so this renames the owner's original exactly once.
    if cls_owns_forward:
        original_forward.__qualname__ = f"{cls.__qualname__}._original_forward"

    return cls
