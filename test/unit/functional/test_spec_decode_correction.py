# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``correct_spec_decode_positions_and_slot_mapping``.

The EAGLE3 and PEAGLE drafts run this correction on every pass. On a prompt
pass they get no-op inputs (no rejections), so the correction must leave the
runner's ``slot_mapping`` as it is. The runner sends bucket-padding rows to the
null block, but those rows carry the last real position. A slot recomputed
from that position lands on the KV slot of the last real token, and the padding
rows then overwrite that token's draft KV entry.

The tests need no Neuron hardware. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest test/unit/functional/test_spec_decode_correction.py -v
"""

import torch

from vllm_neuron.functional.spec_decode_correction import (
    correct_spec_decode_positions_and_slot_mapping,
)

BLOCK_SIZE = 16
LAYER = "layers.0.self_attn"
VOCAB_SIZE = 1000
NUM_SPEC = 3
NULL_SLOT = 0  # Slot 0 lies in the reserved null block (block id 0).
PAD_SLOT_ID = -1


def _attn_metadata(block_table, slot_mapping=None):
    table = torch.tensor(block_table, dtype=torch.int32)
    meta = {
        "block_table_tensor": table,
        "full_block_table_tensor": table,
        "block_size": BLOCK_SIZE,
    }
    if slot_mapping is not None:
        meta["slot_mapping"] = torch.tensor(slot_mapping, dtype=torch.int64)
    return {LAYER: meta}


def _slot(block_table, req, pos):
    return block_table[req][pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE


def _noop_prev(num_reqs):
    """All-valid previous samples, as the drafter builds them for a prompt pass."""
    prev_sampled = torch.zeros(num_reqs, NUM_SPEC + 1, dtype=torch.int32)
    prev_num_draft = torch.full((num_reqs,), NUM_SPEC, dtype=torch.int32)
    return prev_sampled, prev_num_draft


def test_noop_correction_keeps_prompt_padding_in_null_block():
    """One prompt of 20 tokens padded to a 32-token bucket."""
    num_real, bucket = 20, 32
    block_table = [[3, 7]]
    real_slots = [_slot(block_table, 0, p) for p in range(num_real)]
    # Padding rows repeat the last real position and write to the null block.
    positions = torch.tensor(
        list(range(num_real)) + [num_real - 1] * (bucket - num_real),
        dtype=torch.int32,
    )
    runner_slots = real_slots + [NULL_SLOT] * (bucket - num_real)
    meta = _attn_metadata(block_table, runner_slots)
    prev_sampled, prev_num_draft = _noop_prev(1)
    req_indices = torch.zeros(bucket, dtype=torch.int64)

    new_positions, new_meta = correct_spec_decode_positions_and_slot_mapping(
        positions, meta, prev_sampled, prev_num_draft, req_indices, VOCAB_SIZE
    )

    assert new_positions.tolist() == positions.tolist()
    slots = new_meta[LAYER]["slot_mapping"].tolist()
    assert slots == runner_slots
    # Only the real row writes the slot of the last real position.
    assert slots.count(_slot(block_table, 0, num_real - 1)) == 1


def test_noop_correction_keeps_pad_slot_id():
    """Rows the runner marked with PAD_SLOT_ID stay unwritten."""
    block_table = [[3]]
    positions = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    runner_slots = [_slot(block_table, 0, p) for p in range(3)] + [PAD_SLOT_ID]
    meta = _attn_metadata(block_table, runner_slots)
    prev_sampled, prev_num_draft = _noop_prev(1)
    req_indices = torch.zeros(4, dtype=torch.int64)

    _, new_meta = correct_spec_decode_positions_and_slot_mapping(
        positions, meta, prev_sampled, prev_num_draft, req_indices, VOCAB_SIZE
    )

    assert new_meta[LAYER]["slot_mapping"].tolist() == runner_slots


def test_noop_correction_ignores_request_index_split():
    """Two prompts of different lengths with evenly split request indices.

    The drafter's no-op inputs split ``req_indices_per_token`` evenly across
    requests. With unequal prompt lengths some rows then carry the wrong
    request index. A no-op correction must not move those rows into the other
    request's blocks.
    """
    block_table = [[2, 6], [4, 9]]
    # Request 0: 4 real rows padded to 8. Request 1: 24 real rows.
    positions = torch.tensor(
        list(range(4)) + [3] * 4 + list(range(24)), dtype=torch.int32
    )
    runner_slots = (
        [_slot(block_table, 0, p) for p in range(4)]
        + [NULL_SLOT] * 4
        + [_slot(block_table, 1, p) for p in range(24)]
    )
    meta = _attn_metadata(block_table, runner_slots)
    prev_sampled, prev_num_draft = _noop_prev(2)
    # The no-op builder's even split: 16 rows per request, so rows 8-15
    # (request 1) carry request index 0.
    req_indices = torch.tensor([0, 1], dtype=torch.int64).repeat_interleave(16)

    _, new_meta = correct_spec_decode_positions_and_slot_mapping(
        positions, meta, prev_sampled, prev_num_draft, req_indices, VOCAB_SIZE
    )

    slots = new_meta[LAYER]["slot_mapping"].tolist()
    assert slots[8:16] == runner_slots[8:16]
    assert slots == runner_slots


def test_rejected_rows_still_move():
    """Verify batch: request 0 had two rejections, request 1 none."""
    block_table = [[2, 5], [4, 6]]
    tokens_per_req = 1 + NUM_SPEC
    optimistic = [20, 21, 22, 23, 8, 9, 10, 11]
    positions = torch.tensor(optimistic, dtype=torch.int32)
    req_indices = torch.tensor([0, 1], dtype=torch.int64).repeat_interleave(
        tokens_per_req
    )
    runner_slots = [
        _slot(block_table, int(r), p) for r, p in zip(req_indices, optimistic)
    ]
    meta = _attn_metadata(block_table, runner_slots)
    prev_sampled = torch.tensor([[7, 8, -1, -1], [7, 8, 9, 10]], dtype=torch.int32)
    prev_num_draft = torch.full((2,), NUM_SPEC, dtype=torch.int32)

    new_positions, new_meta = correct_spec_decode_positions_and_slot_mapping(
        positions, meta, prev_sampled, prev_num_draft, req_indices, VOCAB_SIZE
    )

    expected_positions = [18, 19, 20, 21, 8, 9, 10, 11]
    assert new_positions.tolist() == expected_positions
    assert new_meta[LAYER]["slot_mapping"].tolist() == [
        _slot(block_table, int(r), p) for r, p in zip(req_indices, expected_positions)
    ]


def test_without_runner_slot_mapping_slots_are_computed():
    """Metadata without ``slot_mapping`` still gets slots from positions."""
    block_table = [[0, 1, 2, 3]]
    positions = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    meta = _attn_metadata(block_table)
    prev_sampled = torch.tensor([[100, 200, -1, -1]], dtype=torch.int32)
    prev_num_draft = torch.tensor([3], dtype=torch.int32)
    req_indices = torch.zeros(4, dtype=torch.int64)

    new_positions, new_meta = correct_spec_decode_positions_and_slot_mapping(
        positions, meta, prev_sampled, prev_num_draft, req_indices, VOCAB_SIZE
    )

    assert new_positions.tolist() == [8, 9, 10, 11]
    assert new_meta[LAYER]["slot_mapping"].tolist() == [8, 9, 10, 11]
