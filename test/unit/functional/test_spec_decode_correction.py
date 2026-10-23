# SPDX-License-Identifier: Apache-2.0
"""Async spec-decode position / slot correction (CPU): rejected drafts, padded rows, state_in."""

import importlib.util
import pathlib

import torch

_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "vllm_neuron"
    / "functional"
    / "spec_decode_correction.py"
)
_spec = importlib.util.spec_from_file_location("spec_decode_correction", _PATH)
sdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdc)
correct = sdc.correct_spec_decode_positions_and_slot_mapping


def _meta(block_table, block_size=16, state=False):
    meta = {
        "block_table_tensor": torch.tensor(block_table, dtype=torch.int32),
        "block_size": block_size,
    }
    if state:
        meta["state_in"] = torch.zeros(len(block_table), dtype=torch.int32)
    return {"layers.0": meta}


def test_rejected_drafts_shift_positions_and_slots():
    # One request, 3 drafts, previous step accepted 1 draft + bonus (2 valid of 4).
    positions = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    prev = torch.tensor([[100, 200, -1, -1]], dtype=torch.int32)
    out_pos, meta = correct(
        positions,
        _meta([[0, 1, 2, 3]]),
        prev,
        torch.tensor([3], dtype=torch.int32),
        torch.zeros(4, dtype=torch.int64),
        vocab_size=1000,
    )
    assert out_pos.tolist() == [8, 9, 10, 11]
    assert meta["layers.0"]["slot_mapping"].tolist() == [
        8,
        9,
        10,
        11,
    ]  # block 0, offsets 8..11


def test_noop_dummy_one_valid_token_zero_drafts():
    positions = torch.tensor([5, 6, 7], dtype=torch.int32)
    prev = torch.tensor([[7, -1, -1]], dtype=torch.int32)
    out_pos, _ = correct(
        positions,
        _meta([[2]]),
        prev,
        torch.tensor([0], dtype=torch.int32),
        torch.zeros(3, dtype=torch.int64),
        vocab_size=1000,
    )
    assert out_pos.tolist() == [5, 6, 7]


def test_padded_rows_get_pad_slot():
    # Row 1 is a padded batch row with a stale previous row: it would go negative.
    positions = torch.tensor([20, 21, 0, 1], dtype=torch.int32)
    prev = torch.tensor([[1, 2, 3], [-1, -1, -1]], dtype=torch.int32)
    num_draft = torch.tensor([2, 2], dtype=torch.int32)
    req = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    out_pos, meta = correct(
        positions, _meta([[1, 2], [3, 4]]), prev, num_draft, req, vocab_size=1000
    )
    slots = meta["layers.0"]["slot_mapping"].tolist()
    assert out_pos.tolist()[2:] == [0, 0]  # clamped for the position math
    assert slots[2:] == [-1, -1]  # but never written: pad slot
    assert slots[:2] == [2 * 16 + 4, 2 * 16 + 5]  # real row: table entry 1 = block 2


def test_state_in_resumes_from_last_accepted():
    positions = torch.tensor([10, 11, 12], dtype=torch.int32)
    prev = torch.tensor([[9, 8, -1]], dtype=torch.int32)  # 1 draft accepted + bonus
    _, meta = correct(
        positions,
        _meta([[0]], state=True),
        prev,
        torch.tensor([2], dtype=torch.int32),
        torch.zeros(3, dtype=torch.int64),
        vocab_size=1000,
    )
    assert meta["layers.0"]["state_in"].tolist() == [1]
