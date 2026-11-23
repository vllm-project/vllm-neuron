# SPDX-License-Identifier: Apache-2.0
"""CPU loading checks against the released HyperCLOVAX-SEED-Think-14B checkpoint.

The tests read the config, the weight index and the safetensors headers of the
real checkpoint at the pinned revision. Only the last test reads tensor values,
and only for a few small tensors. No test builds a model.

- The checkpoint config gives the expected HyperCLOVAXConfig, and the factory
  accepts it.
- The checkpoint has each of the 421 keys that get_weight_mappings uses, and
  no other key.
- Each tensor has the dtype float32 and the shape that the config gives.
- Sampled tensor values are finite, also after the cast to bfloat16.

Without ``HYPERCLOVAX_CHECKPOINT_PATH`` the first three tests download
``config.json`` and ``model.safetensors.index.json`` (about 35 KB) and read the
safetensors headers from the hub with ranged requests; the last test skips.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hyperclovax/test/unit/test_checkpoint.py -v
"""

import pytest
import torch
from _local_src import src_module
from hyperclovax_test_utils import (
    expected_checkpoint_keys,
    load_checkpoint_tensors,
    local_checkpoint_or_skip,
    metadata_dir,
    read_weight_map,
    safetensors_tensor_info,
)

HIDDEN = 6144
INTERMEDIATE = 14336
VOCAB = 110592
Q_SIZE = 48 * 128
KV_SIZE = 8 * 128


@pytest.fixture(scope="module")
def checkpoint_metadata_dir():
    return metadata_dir()


@pytest.fixture(scope="module")
def weight_map(checkpoint_metadata_dir):
    return read_weight_map(checkpoint_metadata_dir)


def _expected_shape(key: str) -> tuple[int, ...]:
    if key in ("model.embed_tokens.weight", "lm_head.weight"):
        return (VOCAB, HIDDEN)
    # The peri-layer norm keys end with "post_norm1.weight" and "post_norm2.weight".
    if key.endswith(("norm.weight", "post_norm1.weight", "post_norm2.weight")):
        return (HIDDEN,)
    if key.endswith("q_proj.weight"):
        return (Q_SIZE, HIDDEN)
    if key.endswith("k_proj.weight") or key.endswith("v_proj.weight"):
        return (KV_SIZE, HIDDEN)
    if key.endswith("o_proj.weight"):
        return (HIDDEN, Q_SIZE)
    if key.endswith("gate_proj.weight") or key.endswith("up_proj.weight"):
        return (INTERMEDIATE, HIDDEN)
    if key.endswith("down_proj.weight"):
        return (HIDDEN, INTERMEDIATE)
    raise AssertionError(f"No expected shape for key {key}.")


def test_config_parses(checkpoint_metadata_dir):
    from transformers import AutoConfig

    HyperCLOVAXConfig = src_module("config").HyperCLOVAXConfig
    HyperCLOVAXFactory = src_module("factory").HyperCLOVAXForCausalLM

    hf_config = AutoConfig.from_pretrained(checkpoint_metadata_dir)
    HyperCLOVAXFactory._validate_config(hf_config, None)

    cfg = HyperCLOVAXConfig.from_configs(hf_config)
    assert (cfg.hidden_size, cfg.intermediate_size) == (HIDDEN, INTERMEDIATE)
    assert cfg.num_hidden_layers == 38
    assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (48, 8)
    assert cfg.head_dim == 128
    assert cfg.vocab_size == VOCAB
    assert cfg.rope_theta == pytest.approx(1e8)
    assert cfg.attention_multiplier == pytest.approx(0.0078125)
    assert cfg.embedding_multiplier == pytest.approx(10.0)
    assert cfg.logits_scaling == pytest.approx(0.125)
    assert cfg.residual_multiplier == pytest.approx(1.0)
    assert cfg.use_post_norm is True
    assert cfg.tie_word_embeddings is False
    assert cfg.torch_dtype == torch.bfloat16


def test_checkpoint_keys_match_mappings(weight_map):
    expected = expected_checkpoint_keys(38)
    actual = set(weight_map)
    assert actual == expected, (
        f"Missing keys: {sorted(expected - actual)[:8]}. "
        f"Unused keys: {sorted(actual - expected)[:8]}."
    )


def test_checkpoint_dtypes_and_shapes():
    info = safetensors_tensor_info()
    assert set(info) == expected_checkpoint_keys(38)
    bad = [
        (key, shape, dtype)
        for key, (dtype, shape) in sorted(info.items())
        if shape != _expected_shape(key) or dtype != "F32"
    ]
    assert not bad, f"Tensors with an unexpected shape or dtype: {bad[:8]}"


def test_sampled_values_are_finite():
    checkpoint_path = local_checkpoint_or_skip()
    keys = [
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_norm1.weight",
        "model.layers.37.post_norm2.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.37.self_attn.v_proj.weight",
    ]
    tensors = load_checkpoint_tensors(checkpoint_path, keys)
    for key, tensor in tensors.items():
        assert torch.isfinite(tensor).all(), f"{key} has a value that is not finite."
        bf16 = tensor.to(torch.bfloat16)
        assert torch.isfinite(bf16).all(), f"{key} overflows in bfloat16."
        assert tensor.abs().max() > 0, f"{key} is all zeros."
