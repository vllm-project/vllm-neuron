# SPDX-License-Identifier: Apache-2.0
"""The model must name its DeltaNet layers and describe one state row.

A hybrid model holds a paged KV cache for its attention layers and a recurrent
state for its linear attention layers. ``get_kv_spec`` names both groups, so the
framework allocates the state and the KV cache manager knows the state exists.
``get_mamba_state_shape_from_config`` gives the shape of one state row.

One test compares the shapes against the upstream generic calculator. The port
keeps its own recurrent-state axis order, and the two agree in size only while
``linear_key_head_dim == linear_value_head_dim``. That test therefore fails
loudly if a later config breaks the equality, instead of moving wrong bytes.

CPU-only; no Neuron device.
"""

import os
import types

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch

from vllm_neuron.model.kv_cache import HybridKVSpec, KVSpec
from vllm_neuron.model.qwen3_5.model_bf16 import Qwen3_5ForCausalLM

# Qwen3.6/3.8-27B: 64 layers, one full attention layer after every three linear
# attention layers, at TP=8.
TP_SIZE = 8
NUM_LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
# One state row for each concurrent request, as `max(num_seqs_buckets)` sets it.
STATE_ROWS = 4


def _hf_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


def _model_stub(num_layers: int = NUM_LAYERS) -> types.SimpleNamespace:
    """Build the smallest object that ``get_kv_spec`` reads.

    Use a namespace, not a mock. ``getattr`` on a mock answers every lookup, so
    a mock would hide a wrong attribute name.
    """
    layers = []
    for i in range(num_layers):
        is_full = (i + 1) % FULL_ATTENTION_INTERVAL == 0
        self_attn = types.SimpleNamespace(
            num_key_value_heads_per_rank=4 // 1,
            head_dim=256,
            dtype=torch.bfloat16,
            # One row for each request, with the shapes of
            # get_mamba_state_shape_from_config at TP=8.
            conv_state=torch.zeros(STATE_ROWS, 1280, 3, dtype=torch.bfloat16),
            recurrent_state=torch.zeros(STATE_ROWS, 6, 128, 128, dtype=torch.bfloat16),
        )
        layers.append(
            types.SimpleNamespace(is_full_attention=is_full, self_attn=self_attn)
        )
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


def _vllm_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=_hf_config(), dtype=torch.bfloat16
        ),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=TP_SIZE),
    )


# ── get_kv_spec ────────────────────────────────────────────────────────────


def test_the_spec_names_both_layer_groups():
    spec = Qwen3_5ForCausalLM.get_kv_spec(_model_stub())
    assert isinstance(spec, HybridKVSpec)
    assert len(spec.layers) == 16
    assert len(spec.stateful_layer_names) == 48


def test_a_hybrid_spec_is_still_a_kv_spec():
    # The attention-only consumers read `layers` and must not change.
    spec = Qwen3_5ForCausalLM.get_kv_spec(_model_stub())
    assert isinstance(spec, KVSpec)


def test_the_two_groups_never_share_a_layer():
    spec = Qwen3_5ForCausalLM.get_kv_spec(_model_stub())
    attention_names = {layer.name for layer in spec.layers}
    assert attention_names.isdisjoint(set(spec.stateful_layer_names))
    assert len(set(spec.stateful_layer_names)) == 48


def test_the_names_follow_the_module_tree():
    # Both layer types live at layers.{i}.self_attn, so the runner can find the
    # module from the name.
    spec = Qwen3_5ForCausalLM.get_kv_spec(_model_stub(num_layers=4))
    assert [layer.name for layer in spec.layers] == ["layers.3.self_attn"]
    assert spec.stateful_layer_names == [
        "layers.0.self_attn",
        "layers.1.self_attn",
        "layers.2.self_attn",
    ]


def test_a_model_of_only_linear_layers_names_no_attention_layer():
    stub = _model_stub(num_layers=3)  # layers 0,1,2 are all linear
    spec = Qwen3_5ForCausalLM.get_kv_spec(stub)
    assert spec.layers == []
    assert len(spec.stateful_layer_names) == 3


# ── get_mamba_state_shape_from_config ──────────────────────────────────────


def test_the_state_shapes_of_one_row():
    shapes = Qwen3_5ForCausalLM.get_mamba_state_shape_from_config(
        hf_config=_hf_config(), tp_size=TP_SIZE
    )
    conv_shape, recurrent_shape = shapes
    # qkv dim of one rank: (16/8)*128*2 + (48/8)*128 = 512 + 768 = 1280
    assert conv_shape == (1280, 3)
    assert recurrent_shape == (6, 128, 128)


def test_the_shapes_come_from_a_vllm_config_too():
    from_kwargs = Qwen3_5ForCausalLM.get_mamba_state_shape_from_config(
        hf_config=_hf_config(), tp_size=TP_SIZE
    )
    from_config = Qwen3_5ForCausalLM.get_mamba_state_shape_from_config(_vllm_config())
    assert from_config == from_kwargs


def test_the_state_row_is_small_against_the_kv_cache():
    # One row of state holds 74.8 MiB across 48 layers and 8 ranks, and it does
    # not grow with the prompt. A 131072-token request holds 8 GiB of KV.
    conv_shape, recurrent_shape = Qwen3_5ForCausalLM.get_mamba_state_shape_from_config(
        hf_config=_hf_config(), tp_size=TP_SIZE
    )
    elements = conv_shape[0] * conv_shape[1]
    elements += recurrent_shape[0] * recurrent_shape[1] * recurrent_shape[2]
    total_bytes = elements * 2 * 48 * TP_SIZE  # bf16, 48 layers, every rank
    assert 74.0 < total_bytes / 1024 / 1024 < 75.5


def test_the_shapes_agree_with_the_upstream_calculator():
    """Guard against drift from the upstream generic calculator.

    The conv shape must match exactly. The recurrent shape matches in size only,
    because upstream returns (heads, value_dim, key_dim) and this port keeps
    (heads, key_dim, value_dim). The sizes agree while the two head dims are
    equal. This test fails if a config breaks that, which is the point: the
    transfer of Phase 2 moves these bytes between two nodes.
    """
    pytest.importorskip("vllm.model_executor.layers.mamba.mamba_utils")
    from vllm.model_executor.layers.mamba import mamba_utils

    hf = _hf_config()
    # The upstream conv shape depends on a global layout choice. This port stores
    # the sharded dimension first, which upstream calls DS.
    monkey_layout = getattr(mamba_utils, "get_conv_state_layout", None)
    assert monkey_layout is not None, "upstream moved get_conv_state_layout"
    original = mamba_utils.get_conv_state_layout
    mamba_utils.get_conv_state_layout = lambda: "DS"
    try:
        upstream = mamba_utils.MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_world_size=TP_SIZE,
            num_k_heads=hf.linear_num_key_heads,
            num_v_heads=hf.linear_num_value_heads,
            head_k_dim=hf.linear_key_head_dim,
            head_v_dim=hf.linear_value_head_dim,
            conv_kernel_size=hf.linear_conv_kernel_dim,
        )
    finally:
        mamba_utils.get_conv_state_layout = original

    ours = Qwen3_5ForCausalLM.get_mamba_state_shape_from_config(
        hf_config=hf, tp_size=TP_SIZE
    )
    upstream_conv, upstream_recurrent = upstream
    our_conv, our_recurrent = ours

    assert our_conv == upstream_conv, (
        "The conv state shape must equal the upstream shape under the DS layout."
    )
    assert hf.linear_key_head_dim == hf.linear_value_head_dim, (
        "This config gives the key head dim a different size from the value head "
        "dim. The recurrent state axis order of this port then no longer matches "
        "the upstream order in size. Decide which order the transfer uses before "
        "you move these bytes between two nodes."
    )

    def _size(shape):
        total = 1
        for dim in shape:
            total *= dim
        return total

    assert _size(our_recurrent) == _size(upstream_recurrent)


# ── get_mamba_state_dtype_from_config ──────────────────────────────────────


def test_both_states_use_the_model_dtype():
    dtypes = Qwen3_5ForCausalLM.get_mamba_state_dtype_from_config(_vllm_config())
    assert dtypes == (torch.bfloat16, torch.bfloat16)
