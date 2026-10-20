# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128


def unconditional_to_conditional_rates(rates: list[float]) -> list[float]:
    """Per-position unconditional acceptance rates -> conditional rates c_i = p_i / p_{i-1}.

    Mirrors ``vllm.v1.spec_decode.utils.unconditional_to_conditional_rates``.

    Example:
        >>> unconditional_to_conditional_rates([1.0, 0.58, 0.0])
        [1.0, 0.58, 0.0]
    """
    prev = [1.0, *rates[:-1]]
    return [p / q if q > 0.0 else 0.0 for p, q in zip(rates, prev)]


def resolve_synthetic_conditional_rates() -> list[float] | None:
    """Per-position conditional acceptance rates for the current vLLM config.

    Reads ``speculative_config`` from the ambient vLLM config, so synthetic
    rejection sampling needs no plumbing through ``NeuronConfig``. vLLM has
    already validated and populated the unconditional rates by model-init time
    (``SpeculativeConfig._resolve_synthetic_acceptance_rates``).

    Returns ``None`` when synthetic sampling is not requested (standard greedy
    rejection), and raises if it is requested without usable rates.

    Example:
        >>> # under a vLLM config with rejection_sample_method="synthetic"
        >>> resolve_synthetic_conditional_rates()
        [1.0, 0.58, 0.0]
    """
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    spec_config = getattr(vllm_config, "speculative_config", None)
    if spec_config is None:
        return None
    if getattr(spec_config, "rejection_sample_method", "standard") != "synthetic":
        return None
    rates = getattr(spec_config, "synthetic_acceptance_rates", None)
    if not rates:
        raise ValueError(
            "rejection_sample_method='synthetic' requires "
            "synthetic_acceptance_rates (or synthetic_acceptance_length, which "
            f"vLLM resolves into them); got {rates!r}."
        )
    return unconditional_to_conditional_rates(list(rates))


@torch.no_grad()
def rejection_sampler(
    spec_decode_metadata: SpecDecodeMetadata,
    target_token_ids: torch.Tensor,  # [num_target_tokens] - sampled token IDs for all positions including bonus
    synthetic_conditional_rates: list[float] | None = None,
) -> torch.Tensor:
    """
    Perform greedy rejection sampling by comparing draft and target token IDs.

    On-device rejection sampler for speculative decoding. Performs deterministic
    rejection sampling by comparing draft tokens with target tokens. Accepts tokens
    sequentially until the first mismatch.

    The sampling step is performed externally by the Sampler module.
    This function only handles the rejection logic.

    When ``synthetic_conditional_rates`` is provided, accept/reject is drawn
    synthetically (``u < rate[pos]``) instead of by draft/target agreement,
    pinning acceptance length to a calibrated target for throughput benchmarking.
    Mirrors upstream vLLM's ``rejection_sample_method="synthetic"``.

    Args:
        spec_decode_metadata: Object containing:
            - draft_token_ids: Flattened draft tokens [num_draft_tokens]
            - num_draft_tokens: Actual number of draft tokens per request (list of ints)
            - max_spec_len: Maximum number of speculative draft tokens per request
            - cu_num_draft_tokens: Cumulative number of draft tokens [batch_size]
        target_token_ids: Sampled token IDs from all positions [num_target_tokens]
            where num_target_tokens = sum(num_draft_tokens) + batch_size
            Positions are interleaved: [drafts_0, bonus_0, drafts_1, bonus_1, ...]
        synthetic_conditional_rates: Optional ``max_spec_len`` per-position
            *conditional* rates ``c_i = P(accept i | accept i-1)``; enables
            synthetic mode. ``None`` (default) => standard greedy.

    Returns:
        ``[batch_size, max_spec_len+1]`` int32 with ``-1`` for rejected
        positions. The last accepted token per request (used by the
        async-spec-decode pipeline) is extracted by the
        ``@async_speculative_decoding`` decorator's epilogue, not here,
        to keep this kernel free of pipeline-specific concerns.

    Example:
        >>> # spec_decode_metadata contains draft_token_ids, cu_num_draft_tokens, etc.
        >>> # target_token_ids: [num_target_tokens] sampled tokens including bonus positions
        >>> result = rejection_sampler(spec_decode_metadata, target_token_ids)
        >>> # result[0] might be [5, 10, 15, 99] if all drafts accepted + bonus
        >>> # or [5, 11, -1, -1] if second position rejected
    """

    assert spec_decode_metadata.max_spec_len <= MAX_SPEC_LEN

    device = target_token_ids.device
    max_spec_len = spec_decode_metadata.max_spec_len
    draft_ids = spec_decode_metadata.draft_token_ids
    cu = spec_decode_metadata.cu_num_draft_tokens
    batch_size = cu.shape[0]

    # Extract bonus tokens from interleaved structure: [drafts_0, bonus_0, drafts_1, bonus_1, ...]
    batch_idx = torch.arange(batch_size, device=device)
    bonus_tokens = target_token_ids[cu + batch_idx]  # [batch_size]

    # Setup 2D indexing: [batch_size, max_spec_len]
    pos = torch.arange(max_spec_len, device=device).unsqueeze(0)  # [1, max_spec_len]
    num_drafts = torch.diff(
        cu, prepend=torch.zeros(1, device=device, dtype=cu.dtype)
    )  # [batch_size]
    valid = pos < num_drafts.unsqueeze(1)  # [batch_size, max_spec_len]

    # Convert flattened draft tokens to 2D
    cu_start = torch.cat(
        [torch.zeros(1, device=device, dtype=cu.dtype), cu[:-1]]
    )  # [batch_size]
    draft_idx = (cu_start.unsqueeze(1) + pos).clamp(max=draft_ids.shape[0] - 1)
    drafts_2d = torch.where(valid, draft_ids[draft_idx], PLACEHOLDER_TOKEN_ID)

    # Convert flattened target tokens to 2D
    target_idx = (cu_start.unsqueeze(1) + batch_idx.unsqueeze(1) + pos).clamp(
        max=target_token_ids.shape[0] - 1
    )
    targets_2d = torch.where(valid, target_token_ids[target_idx], PLACEHOLDER_TOKEN_ID)

    # Per-position acceptance mask. Both modes mask invalid positions to False so
    # the sequential early-termination below treats padding as a rejection.
    if synthetic_conditional_rates is not None:
        # Synthetic: accept iff u < rate[pos]. One coin/request broadcast across
        # positions (the step-function schedule needs only one); a [batch,
        # spec_len] draw degenerates in the full graph. Rates via per-scalar
        # torch.full + cat, not torch.tensor(list): the full-graph tracer rejects
        # a lifted list constant but traces scalar-fill factory ops cleanly.
        uniform_probs = torch.rand((batch_size, 1), dtype=torch.float32, device=device)
        rates_row = torch.cat(
            [
                torch.full((1,), r, dtype=torch.float32, device=device)
                for r in synthetic_conditional_rates
            ]
        ).unsqueeze(0)  # [1, max_spec_len]
        matches = (uniform_probs < rates_row) & valid
    else:
        # Standard greedy mode: accept[pos] iff draft matches the target argmax.
        matches = (drafts_2d == targets_2d) & valid

    # Accept until the first non-accepting position: argmax on the inverted mask
    # finds it (sentinel column => all-accepting rows resolve to max_spec_len),
    # then pos-masking enforces early termination. Same for both modes; only
    # `matches` differs.
    mismatches = ~matches  # [batch_size, max_spec_len]
    mismatches_with_sentinel = torch.cat(
        [mismatches, torch.ones(cu.shape[0], 1, device=device, dtype=torch.bool)], dim=1
    )  # [batch_size, max_spec_len + 1]
    first_mismatch_pos = mismatches_with_sentinel.float().argmax(dim=1)  # [batch_size]
    pos_indices = torch.arange(max_spec_len, device=device).unsqueeze(
        0
    )  # [1, max_spec_len]
    # Emit draft on accepted positions, target at the reject boundary, -1 beyond
    # (matches upstream). Greedy: draft == target on accepts, so equivalent to
    # emitting target; synthetic: yields the pretended-accepted draft tokens.
    strictly_accepted = (pos_indices < first_mismatch_pos.unsqueeze(1)) & valid
    boundary = (pos_indices == first_mismatch_pos.unsqueeze(1)) & valid
    output = torch.where(
        strictly_accepted,
        drafts_2d,
        torch.where(boundary, targets_2d, PLACEHOLDER_TOKEN_ID),
    )

    # Add bonus token if all drafts accepted
    all_accepted = (matches & valid).all(dim=1)
    result = torch.cat(
        [
            output,
            torch.full(
                (batch_size, 1), PLACEHOLDER_TOKEN_ID, device=device, dtype=torch.int32
            ),
        ],
        dim=1,
    )
    result[batch_idx, num_drafts.long()] = torch.where(
        all_accepted, bonus_tokens, PLACEHOLDER_TOKEN_ID
    )

    return result
