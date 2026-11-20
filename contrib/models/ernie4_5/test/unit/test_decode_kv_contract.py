# SPDX-License-Identifier: Apache-2.0
"""Decode-kernel KV contract test for ERNIE 4.5 dense.

At TP 1 each rank holds both KV heads, so the decode megakernel gets the
per-head block table and the per-head write slots. The kernel reads the write
slots as uint32. vLLM's slot mapping is int64, and an int64 slot tensor makes
the device refuse to load the decode NEFF ("invalid offset in reshape"). This
test captures the arguments of ``NF.attention_decode`` and checks the dtype,
the shape and the values of both tables. It does not run a kernel.

Run with:
    pytest contrib/models/ernie4_5/test/unit/test_decode_kv_contract.py -v
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` helper by file location.

    See ``test/_local_src.py`` for why the helper is not imported by name.
    """
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    name = "contrib_ernie4_5_local_src"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the local-src helper at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


_local_src = _load_local_src()
_src_model = _local_src.load("model")
_src_config = _local_src.load("config")

pytestmark = [pytest.mark.fast]

BLOCK_SIZE = 32
NUM_BLOCKS = 4
MAX_BLOCKS_PER_SEQ = 4


def test_ernie4_5_decode_passes_uint32_per_head_slots():
    """TP 1 decode gives the kernel uint32 per-head slots and an int32 table."""
    config = _src_config.Ernie4_5Config()  # ERNIE-4.5-0.3B geometry
    tp_stub = SimpleNamespace(world_size=1, rank_in_group=0)
    with mock.patch.object(_src_model, "get_tp_group", return_value=tp_stub):
        attn = _src_model.Ernie4_5Attention(config, layer_idx=0)
    nkh = attn.num_key_value_heads_per_rank
    assert nkh == 2

    cache_shape = (NUM_BLOCKS, nkh, BLOCK_SIZE, config.head_dim)
    attn.k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    attn.v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)

    # 2 requests, 1 token each: base block 0 position 5, base block 1 position 5.
    slot_mapping = torch.tensor([5, BLOCK_SIZE + 5], dtype=torch.int64)
    block_table = torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]], dtype=torch.int32)
    metadata = {
        "layers.0.self_attn": {
            "slot_mapping": slot_mapping,
            "block_size": BLOCK_SIZE,
            "max_blocks_per_seq": MAX_BLOCKS_PER_SEQ,
            "block_table_tensor": block_table,
        }
    }
    batch = block_table.shape[0]
    hidden = torch.zeros(batch, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.tensor([5, 5], dtype=torch.int32)
    cos = torch.ones(batch, config.head_dim, dtype=torch.bfloat16)
    sin = torch.zeros(batch, config.head_dim, dtype=torch.bfloat16)

    captured = {}

    def fake_attention_decode(**kwargs):
        captured.update(kwargs)
        return torch.zeros(batch, 1, config.hidden_size, dtype=torch.bfloat16)

    with (
        mock.patch.object(
            _src_model.NF, "attention_decode", side_effect=fake_attention_decode
        ),
        mock.patch.object(
            _src_model.NF,
            "gen_attention_decode_mask",
            return_value=torch.zeros(1),
        ),
    ):
        attn.forward_decode(hidden, positions, (cos, sin), metadata)

    assert captured["update_cache"] is True

    idx = captured["kv_cache_update_idx"]
    assert idx.dtype == torch.uint32, (
        f"The decode kernel reads uint32 write slots; got {idx.dtype}. An int64 "
        "slot tensor makes the runtime refuse the decode NEFF."
    )
    assert tuple(idx.shape) == (batch, nkh, 1)
    # Global slot = (base_block * nkh + head) * block_size + position.
    expected_idx = torch.tensor([[[5], [37]], [[69], [101]]], dtype=torch.int64)
    assert torch.equal(idx.to(torch.int64), expected_idx)

    table = captured["active_blocks_table"]
    assert table.dtype == torch.int32
    expected_table = torch.tensor(
        [[[0, -1, -1, -1], [1, -1, -1, -1]], [[2, -1, -1, -1], [3, -1, -1, -1]]],
        dtype=torch.int32,
    )
    assert torch.equal(table, expected_table)
