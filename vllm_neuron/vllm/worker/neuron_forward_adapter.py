# SPDX-License-Identifier: Apache-2.0
"""Adapt standalone Neuron runner inputs to the compiled model forward pass.

Use one runner-owned host-to-device interface to build ``input_ids`` in the
graph, select per-request sampling parameters, call the unchanged model by
keyword, and commit device-authoritative state in the graph. Return sampled
tokens with the aliased state bundle and ordinary next-token carry.

Specialize prefill and decode as separate compiled graphs according to whether
``linear_idx`` is present. Allow index values to change between steps while
keeping their shapes fixed. Compile the adapter on a Neuron device and execute
it eagerly during CPU simulation.

Build one explicit attention-metadata dictionary per KV-cache group and alias
that dictionary across all layer names in the group.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_neuron.functional.combine_sampled_tokens import (
    build_decode_input_ids_from_store,
    scatter_sampled_into_slot_store,
)

# In-vocab pad id for masked-out input rows/tokens. 0 is a valid embedding index (never out of range);
# padded rows are excluded from sampling host-side and from attention by the block-table -1 sentinel, so
# the pad embedding never pollutes a real output. NEVER use a negative id (embedding is not defined there).
PAD_TOKEN = 0


# Forked from the V1 model runner's _reinterpret_uint32_as_int32 helper.
def _reinterpret_uint32_as_int32(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Reinterpret uint32 argmax bits as int32 without a device cast.

    Return other dtype combinations unchanged. Token IDs fit in int32, so
    viewing the same bits is exact and avoids materializing a value cast.

    Args:
        tensor: Tensor containing sampled token IDs.
        target_dtype: Integer dtype required by the token store.

    Returns:
        The bit-reinterpreted tensor, or the original tensor when no
        reinterpretation is required.
    """
    if tensor.dtype == torch.uint32 and target_dtype == torch.int32:
        return tensor.view(torch.int32)
    return tensor


def build_attention_metadata(
    kv_cache_groups,
    block_tables: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    max_query_len: int,
    decode_token_threshold: int = 1,
    cached_seq_len: torch.Tensor | None = None,
    kv_segment_size: int = 0,
    slot_mappings: "Sequence[torch.Tensor] | None" = None,
    swa_kv_pos_offsets: "Sequence[torch.Tensor | None] | None" = None,
) -> dict[str, dict]:
    """Build per-layer attention metadata for dense model execution.

    Create one metadata dictionary per KV-cache group and share it by
    reference across every layer in that group. Accept all values explicitly
    so the same builder serves real execution and synthetic warmup. Consume
    block rows that already contain the ``-1`` kernel sentinel.

    Use ``max_query_len`` and ``decode_token_threshold`` to distinguish
    prefill from decode. Include ``cached_seq_len`` and ``kv_segment_size``
    only for segmented prefill.

    Args:
        kv_cache_groups (Sequence[KVCacheGroupSpec]): Cache groups with layer
            names and block sizes.
        block_tables: Per-group ``[rows, width]`` int32 kernel block tables.
        slot_mapping: Padded ``[tokens]`` int64 slot mapping, used for every
            group when ``slot_mappings`` is not given.
        max_query_len: Padded prefill token count or one for decode.
        decode_token_threshold: Query length above which execution is prefill.
        cached_seq_len: Optional ``[[length]]`` int32 cached-prefix tensor.
        kv_segment_size: Positive segment size for segmented prefill, or zero.
        slot_mappings: Optional per-group slot mappings. Required whenever
            groups own distinct block tables (e.g. a sliding-window group next
            to a full-attention group): KV writes must resolve slots through
            their own group's table.
        swa_kv_pos_offsets: Optional per-group ``[rows]`` int32 KV position
            offsets. A non-None entry marks a sliding-window group whose block
            table is window-trimmed; the model subtracts the offset from
            position IDs so the causal mask is computed in the trimmed frame.

    Returns:
        A mapping from layer name to its group's shared metadata dictionary.
    """
    metadata: dict[str, dict] = {}
    for group_idx, group in enumerate(kv_cache_groups):
        block_table = block_tables[group_idx]
        group_slot_mapping = (
            slot_mappings[group_idx] if slot_mappings is not None else slot_mapping
        )
        per_layer = {
            "block_table_tensor": block_table,
            "slot_mapping": group_slot_mapping,
            "max_query_len": max_query_len,
            "block_size": group.kv_cache_spec.block_size,
            "max_blocks_per_seq": block_table.shape[1],
            "decode_token_threshold": decode_token_threshold,
        }
        if swa_kv_pos_offsets is not None:
            offset = swa_kv_pos_offsets[group_idx]
            if offset is not None:
                per_layer["swa_kv_pos_offset"] = offset
        if kv_segment_size:
            per_layer["cached_seq_len"] = cached_seq_len
            per_layer["kv_segment_size"] = kv_segment_size
        for layer_name in group.layer_names:
            metadata[layer_name] = (
                per_layer  # one dict aliased across the group's layers
            )
    return metadata


def _compute_next_device_state(
    token_ids: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    total_len: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    prefill_len: torch.Tensor,
    slot_active_row: torch.Tensor,
    query_len_by_slot: torch.Tensor,
    sampled_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the next device-authoritative state bundle.

    Advance counts for every active slot. Commit sampled tokens to history only
    for decode steps and final prefill chunks whose token store has capacity.
    The scheduler normally stops full sequences; the capacity guard prevents a
    late dispatch from overwriting the final stored token. Update token history
    in place with one indexed cell per stable slot; produce the smaller state
    vectors out of place.

    Args:
        token_ids: ``[max_num_reqs, max_model_len]`` token history.
        num_computed_tokens: ``[max_num_reqs]`` computed-token counts.
        total_len: ``[max_num_reqs]`` stored token lengths.
        last_sampled_tokens: ``[max_num_reqs, 1]`` sampled-token store.
        prefill_len: ``[max_num_reqs]`` prefill completion lengths.
        slot_active_row: ``[max_num_reqs]`` output row per active slot.
        query_len_by_slot: ``[max_num_reqs]`` scheduled token count by slot.
        sampled_tokens: Sampled tokens in output-row order.

    Returns:
        Updated token history, computed counts, total lengths, and sampled
        token store.
    """
    max_model_len = token_ids.shape[1]

    active = slot_active_row >= 0
    delta = torch.where(
        active,
        query_len_by_slot.to(num_computed_tokens.dtype),
        torch.zeros_like(num_computed_tokens),
    )
    new_num_computed = num_computed_tokens + delta
    already_decode = num_computed_tokens >= prefill_len
    in_bounds = total_len < max_model_len
    commit = active & (already_decode | (new_num_computed == prefill_len)) & in_bounds

    inactive_row = torch.full_like(slot_active_row, -1)
    commit_slot_row = torch.where(commit, slot_active_row, inactive_row)
    new_last_sampled = scatter_sampled_into_slot_store(
        last_sampled_tokens, commit_slot_row, sampled_tokens
    )

    row_idx = torch.clamp(slot_active_row.to(torch.int64), min=0)
    token_by_slot = sampled_tokens.reshape(-1).to(token_ids.dtype)[row_idx]

    state_rows = torch.arange(
        token_ids.shape[0], dtype=torch.int64, device=token_ids.device
    )
    append_col = torch.clamp(total_len.to(torch.int64), max=max_model_len - 1)
    old_token_at_append = token_ids[state_rows, append_col]
    token_to_store = torch.where(commit, token_by_slot, old_token_at_append)
    token_ids.index_put_((state_rows, append_col), token_to_store)
    new_token_ids = token_ids
    new_total_len = total_len + commit.to(total_len.dtype)

    return new_token_ids, new_num_computed, new_total_len, new_last_sampled


class NeuronForwardAdapter(torch.nn.Module):
    """Wrap an unchanged model with the runner's compiled forward contract.

    Use an ``nn.Module`` so ``torch.compile`` traces the explicit tensor ABI in
    :meth:`forward`. Retain the wrapped model as an unmodified submodule.
    """

    def __init__(self, model, emit_gathered_logits: bool = False) -> None:
        super().__init__()
        self.model = model
        self.emit_gathered_logits = emit_gathered_logits

    def forward(
        self,
        positions: torch.Tensor,
        attn_metadata,
        sampling_positions: torch.Tensor,
        row_to_slot: torch.Tensor,
        input_valid_mask: torch.Tensor,
        slot_active_row: torch.Tensor,
        query_len_by_slot: torch.Tensor,
        prefill_len: torch.Tensor,
        token_store: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        total_len: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        sampling_params: torch.Tensor,
        rank: torch.Tensor,
        next_token_by_slot: torch.Tensor,
        linear_idx: torch.Tensor | None = None,
    ):
        """Execute one model step and commit state inside the graph.

        * Builds prefill inputs from `linear_idx`.
        * Builds initial decode inputs from token history.
        * Builds subsequent decode inputs from the previous non-aliased output.
        * Selects sampling parameters by stable slot.
        * Runs the model.
        * Updates token history in place.

        Return sampled tokens, four updated state tensors, and a non-aliased
        ``[max_num_reqs, 1]`` clone of the sampled-token store. Every
        tensor-parallel rank passes that clone to the next decode step to keep
        collective ordering stable. In debug mode, append gathered logits
        without moving the existing output positions.

        Args:
            positions: Padded token positions for prefill or decode.
            attn_metadata (dict[str, dict]): Per-layer attention metadata.
            sampling_positions: Hidden-state rows to sample.
            row_to_slot: Mapping from sampled output rows to stable slots.
                ``-1`` marks a padded row.
            input_valid_mask: Validity mask for input tokens or rows.
            slot_active_row: Mapping from stable slots to output rows.
            query_len_by_slot: Scheduled token count for each stable slot.
            prefill_len: Prefill completion length for each stable slot.
            token_store: Persistent token-history tensor.
            num_computed_tokens: Device-authoritative computed-token counts.
            total_len: Device-authoritative token-history lengths.
            last_sampled_tokens: Persistent sampled-token store.
            sampling_params: Per-slot sampling parameters.
            rank: Tensor-parallel rank tensor passed to the model.
            next_token_by_slot: Non-aliased sampled-token output from the
                previous step.
            linear_idx: Optional flat prefill gather index. Omit for decode.

        Returns:
            Sampled tokens, updated state tensors, and the non-aliased
            next-token output. Append gathered logits in debug mode.

        Raises:
            AssertionError: If the decode carry dtype does not match the token
                history dtype.
            RuntimeError: If debug mode does not receive tensor logits.
        """
        # --- prologue: build input_ids on device (never read on host) ---
        if linear_idx is not None:
            # PREFILL: gather each scored token from its request's stable-slot row in the flattened store.
            gathered = torch.index_select(token_store.reshape(-1), 0, linear_idx)
            input_ids = torch.where(
                input_valid_mask, gathered, torch.full_like(gathered, PAD_TOKEN)
            )
        else:
            # DECODE: a DI bootstrap slot must consume its first uncached
            # history token. Other slots consume the non-aliased output from
            # the prior step. Select the history token by its computed count.
            assert next_token_by_slot.dtype == token_store.dtype, (
                "decode carry must match token_store dtype before bootstrap "
                f"selection; got {next_token_by_slot.dtype} and {token_store.dtype}"
            )
            bootstrap_slot = (slot_active_row >= 0) & (
                num_computed_tokens < prefill_len
            )
            bootstrap_col = torch.clamp(
                num_computed_tokens.to(torch.int64),
                max=token_store.shape[1] - 1,
            )
            bootstrap_token_by_slot = torch.gather(
                token_store,
                dim=1,
                index=bootstrap_col.unsqueeze(1),
            )
            decode_token_by_slot = torch.where(
                bootstrap_slot.unsqueeze(1),
                bootstrap_token_by_slot,
                next_token_by_slot,
            )
            placeholder = torch.full_like(
                row_to_slot, PAD_TOKEN, dtype=token_store.dtype
            )
            input_ids = build_decode_input_ids_from_store(
                placeholder, row_to_slot, decode_token_by_slot
            )
            input_ids = torch.where(
                input_valid_mask, input_ids, torch.full_like(input_ids, PAD_TOKEN)
            )

        # Padded rows use -1 for the decode-input preserve sentinel. Clamp only
        # for this gather; their selected parameters are masked and ignored.
        sampling_row_to_slot = torch.clamp(row_to_slot, min=0)
        selected_sampling_params = torch.index_select(
            sampling_params, 0, sampling_row_to_slot
        )

        # --- call the UNCHANGED model by keyword (args[0..1] are inputs_embeds/is_token_ids) ---
        result = self.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=sampling_positions,
            sampling_params=selected_sampling_params,
            logit_mask=None,
            rank=rank,
        )
        sampled_raw = result[0]
        sampled = _reinterpret_uint32_as_int32(sampled_raw, token_store.dtype)

        # --- epilogue: commit this step's device-authoritative state in-graph (aliased outputs) ---
        bundle = _compute_next_device_state(
            token_ids=token_store,
            num_computed_tokens=num_computed_tokens,
            total_len=total_len,
            last_sampled_tokens=last_sampled_tokens,
            prefill_len=prefill_len,
            slot_active_row=slot_active_row,
            query_len_by_slot=query_len_by_slot,
            sampled_tokens=sampled,
        )
        # Return a clone so the next step depends on a non-aliased graph output.
        # Every TP rank consumes it before the next decode, which preserves
        # sampling collective order.
        next_token_out = bundle[3].clone()
        if self.emit_gathered_logits:
            gathered_logits = result[1]
            if not torch.is_tensor(gathered_logits):
                raise RuntimeError(
                    "V2 gathered logits require the model to return a tensor"
                )
            return sampled, bundle, next_token_out, gathered_logits
        return sampled, bundle, next_token_out
