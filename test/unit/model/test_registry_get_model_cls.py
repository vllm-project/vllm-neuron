# SPDX-License-Identifier: Apache-2.0
"""``get_model_cls`` and the ``mm_utils`` capability lookups for out-of-tree models.

A model registered with ``ModelRegistry.register_model`` (for example a contrib
model) is not in ``get_models()``. The multimodal capability lookups must still
find its class, or the runner sizes the encoder-cache gather to 0 blocks.

    VLLM_NEURON_CPU_MODE=1 pytest test/unit/model/test_registry_get_model_cls.py -v
"""

import math
import sys
import types

import pytest
import torch.nn as nn

pytest.importorskip("vllm")

from transformers import PretrainedConfig  # noqa: E402
from vllm import ModelRegistry  # noqa: E402

from vllm_neuron.model.interfaces import (  # noqa: E402
    SupportsEncoderBudget,
    SupportsSpatialMerge,
)
from vllm_neuron.model.registry import get_model_cls, get_models  # noqa: E402
from vllm_neuron.utils import mm_utils  # noqa: E402

ARCH = "OutOfTreeTestForConditionalGeneration"
MODULE = "vllm_neuron_test_out_of_tree_model"


class OutOfTreeModel(nn.Module, SupportsSpatialMerge, SupportsEncoderBudget):
    """Minimal out-of-tree multimodal model class (capability classmethods only)."""

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config):
        return 4

    @classmethod
    def get_max_encoder_embed_budget(cls, hf_config, encoder_neuron_config):
        return max(encoder_neuron_config.num_encoder_input_buckets) // 4

    @classmethod
    def get_max_encoder_blocks_per_request(cls, hf_config, encoder_neuron_config):
        per_block = encoder_neuron_config.encoder_attention_block_size // 4
        return math.ceil(cls.get_max_encoder_embed_budget(hf_config, encoder_neuron_config) / per_block)


@pytest.fixture
def registered():
    """Register ``ARCH`` lazily (``module:Class``), as a vLLM general plugin would."""
    mod = types.ModuleType(MODULE)
    mod.OutOfTreeModel = OutOfTreeModel
    sys.modules[MODULE] = mod
    ModelRegistry.register_model(ARCH, f"{MODULE}:OutOfTreeModel")
    yield
    ModelRegistry.models.pop(ARCH, None)
    sys.modules.pop(MODULE, None)


def _hf(arch):
    cfg = PretrainedConfig()
    cfg.architectures = [arch]
    return cfg


def _encoder_config():
    return types.SimpleNamespace(num_encoder_input_buckets=[4096], encoder_attention_block_size=4096)


def test_builtin_models_resolve_from_get_models():
    for arch, cls in get_models():
        assert get_model_cls(arch) is cls
        assert get_model_cls([arch]) is cls


def test_unknown_architecture_returns_none():
    assert get_model_cls("NoSuchArchitectureForCausalLM") is None
    assert get_model_cls(None) is None
    assert get_model_cls([]) is None


def test_out_of_tree_model_resolves_from_vllm_registry(registered):
    assert ARCH not in dict(get_models())
    assert get_model_cls(ARCH) is OutOfTreeModel
    assert get_model_cls(["NoSuchArchitectureForCausalLM", ARCH]) is OutOfTreeModel


def test_get_models_takes_priority_over_vllm_registry():
    arch, cls = get_models()[0]
    # vLLM's own registry maps most built-in names to its GPU classes.
    assert get_model_cls(arch) is cls


def test_unimportable_registry_entry_returns_none():
    ModelRegistry.register_model("BrokenOutOfTreeForCausalLM", "no_such_module_xyz:Model")
    try:
        assert get_model_cls("BrokenOutOfTreeForCausalLM") is None
    finally:
        ModelRegistry.models.pop("BrokenOutOfTreeForCausalLM", None)


def test_mm_utils_lookups_use_out_of_tree_class(registered):
    hf, enc = _hf(ARCH), _encoder_config()
    assert mm_utils.get_vision_token_merge_factor(hf) == 4
    assert mm_utils.get_encoder_cache_block_size(hf, 4096) == 1024
    assert mm_utils.get_max_encoder_embed_budget(hf, enc) == 1024
    # Before the fix this was 0: the runner then rejected every image request with
    # "request has more cached blocks than max_encoder_blocks_per_request=0".
    assert mm_utils.get_max_encoder_blocks_per_request(hf, enc) == 1


def test_mm_utils_defaults_without_a_registered_class():
    hf, enc = _hf("NoSuchArchitectureForCausalLM"), _encoder_config()
    assert mm_utils.get_vision_token_merge_factor(hf) == 1
    assert mm_utils.get_max_encoder_embed_budget(hf, enc) is None
    assert mm_utils.get_max_encoder_blocks_per_request(hf, enc) == 0
    assert mm_utils.get_max_pixels_token_count(hf, 1000) is None
