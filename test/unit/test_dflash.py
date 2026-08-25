# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.functional.attention.attention_decode_mask import (
    gen_attention_decode_mask,
)
from vllm_neuron.model.dflash.config import DFlashConfig
from vllm_neuron.model.dflash.model import (
    _feature_projection_loader,
    resolve_verified_block,
)
from vllm_neuron.model.registry import get_models
from vllm_neuron.vllm.attention.attn import NeuronAttentionBackend
from vllm_neuron.vllm.spec_decode.dflash import (
    dflash_target_capture_layer_ids,
)


def test_gpt_oss_dflash_config_translation():
    config = DFlashConfig.from_configs(
        {
            "hidden_size": 2880,
            "block_size": 8,
            "rope_theta": 150000.0,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
            },
            "dflash_config": {
                "mask_token_id": 200000,
                "target_layer_ids": [1, 6, 11, 16, 21],
            },
        }
    )

    assert config.mask_token_id == 200000
    assert config.target_layer_ids == [1, 6, 11, 16, 21]
    assert config.block_size == 8
    assert config.rope_parameters["rope_theta"] == 150000.0


def test_dflash_target_output_ids_translate_to_neuron_boundaries():
    hf_config = type(
        "Config",
        (),
        {"dflash_config": {"target_layer_ids": [1, 6, 11, 16, 21]}},
    )()

    assert dflash_target_capture_layer_ids(hf_config) == (2, 7, 12, 17, 22)


def test_feature_projection_padding_preserves_feature_boundaries():
    # Use tiny dimensions to prove that padding is inserted after every
    # feature rather than appended once after the concatenation.
    loader = _feature_projection_loader(
        target_hidden_size=2,
        draft_hidden_size=2,
        padded_hidden_size=3,
        num_features=3,
    )
    weight = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.bfloat16,
    )

    padded = loader.load([weight], rank=0)

    assert padded.shape == (3, 9)
    torch.testing.assert_close(
        padded,
        torch.tensor(
            [
                [1, 2, 0, 3, 4, 0, 5, 6, 0],
                [7, 8, 0, 9, 10, 0, 11, 12, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bfloat16,
        ),
    )


def test_dflash_registry_and_bonus_token_selection():
    assert dict(get_models())["DFlashDraftModel"] is not None
    assert NeuronAttentionBackend.supports_non_causal()
    # Row 0 stops after one accepted sample; row 1 has 10 >= vocab_size, so the
    # bonus is the last in-vocab entry rather than the last column.
    samples = torch.tensor([[3, -1, -1], [4, 5, 10]], dtype=torch.int32)
    bonus, _ = resolve_verified_block(
        samples,
        torch.tensor([2, 5], dtype=torch.long),
        vocab_size=10,
        query_len=3,
    )
    torch.testing.assert_close(bonus, torch.tensor([3, 5], dtype=torch.int32))


def test_dflash_all_rejected_row_yields_zero_bonus():
    bonus, _ = resolve_verified_block(
        torch.tensor([[-1, -1]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.long),
        vocab_size=10,
        query_len=2,
    )
    torch.testing.assert_close(bonus, torch.tensor([0], dtype=torch.int32))


def test_dflash_partial_rejection_rewinds_context_boundary():
    # Block of 8: row 0 keeps 1 of 8 (rewind 7), row 1 keeps 3 of 8 (rewind 5).
    samples = torch.tensor(
        [
            [101, -1, -1, -1, -1, -1, -1, -1],
            [201, 202, 203, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    _, rewound = resolve_verified_block(
        samples,
        torch.tensor([7, 15], dtype=torch.long),
        vocab_size=1000,
        query_len=8,
    )
    torch.testing.assert_close(rewound, torch.tensor([0, 10], dtype=torch.long))


def test_dflash_prefill_shape_does_not_rewind():
    # A [bs, 1] sample tensor is the post-prefill case: nothing to rewind.
    _, rewound = resolve_verified_block(
        torch.tensor([[101], [201]], dtype=torch.int32),
        torch.tensor([7, 15], dtype=torch.long),
        vocab_size=1000,
        query_len=8,
    )
    torch.testing.assert_close(rewound, torch.tensor([7, 15], dtype=torch.long))


def test_dflash_active_block_is_bidirectional():
    query_len = 8
    positions = torch.arange(32, 32 + query_len, dtype=torch.float32).view(1, -1)
    active_mask = torch.ones(query_len, 1, 1, query_len)

    bidirectional = gen_attention_decode_mask(
        positions,
        bs=1,
        q_head=1,
        s_active=query_len,
        s_prior=256,
        block_len=8,
        active_mask=active_mask,
    )
    causal = gen_attention_decode_mask(
        positions,
        bs=1,
        q_head=1,
        s_active=query_len,
        s_prior=256,
        block_len=8,
    )

    assert torch.all(bidirectional[-query_len:] == 1)
    assert not torch.equal(bidirectional[-query_len:], causal[-query_len:])
