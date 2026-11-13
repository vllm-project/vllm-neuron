# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Mistral config, factory checks and registration.

No checkpoint and no Neuron device needed:

    pip install -e contrib/models/mistral
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/mistral/test/unit/test_config.py -v
"""

import copy

import pytest
import torch

pytest.importorskip("vllm")
transformers = pytest.importorskip("transformers")

from mistral_neuron.config import MistralConfig  # noqa: E402
from mistral_neuron.factory import MistralForCausalLM as MistralFactory  # noqa: E402

# config.json of mistralai/Mistral-7B-Instruct-v0.3, copied verbatim. It has
# no head_dim key, which is what the head_dim tests below rely on.
MISTRAL_7B_V03_CONFIG = {
    "architectures": ["MistralForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.42.0.dev0",
    "use_cache": True,
    "vocab_size": 32768,
}


def _hf_config(**overrides) -> "transformers.MistralConfig":
    """The 7B v0.3 config as a transformers object (AutoConfig path)."""
    d = copy.deepcopy(MISTRAL_7B_V03_CONFIG)
    d.update(overrides)
    for key in ("architectures", "model_type", "transformers_version"):
        d.pop(key)
    cfg = transformers.MistralConfig(**d)
    cfg.architectures = ["MistralForCausalLM"]
    return cfg


def _assert_v03_shapes(c: MistralConfig) -> None:
    assert c.vocab_size == 32768
    assert c.hidden_size == 4096
    assert c.intermediate_size == 14336
    assert c.num_hidden_layers == 32
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128
    assert c.max_position_embeddings == 32768
    assert c.rms_norm_eps == 1e-5
    assert c.sliding_window is None
    assert c.tie_word_embeddings is False
    assert c.torch_dtype == torch.bfloat16
    assert c.rope_parameters.get("rope_type", "default") == "default"
    assert c.rope_parameters["rope_theta"] == 1000000.0


# ── config parsing ───────────────────────────────────────────────────────


def test_raw_config_json_parses_to_v03_shapes():
    """The raw config.json path: no head_dim key, top-level rope_theta."""
    _assert_v03_shapes(MistralConfig.from_configs(copy.deepcopy(MISTRAL_7B_V03_CONFIG)))


def test_transformers_config_parses_to_v03_shapes():
    """The AutoConfig path vLLM uses."""
    _assert_v03_shapes(MistralConfig.from_configs(_hf_config()))


def test_head_dim_is_derived_not_llama_default():
    """Without a head_dim key the Llama config would give 64; Mistral needs 128."""
    d = copy.deepcopy(MISTRAL_7B_V03_CONFIG)
    assert "head_dim" not in d
    assert MistralConfig.from_configs(d).head_dim == 4096 // 32


def test_explicit_head_dim_is_kept():
    """Checkpoints that set head_dim (e.g. Mistral Nemo, 128 != 5120 // 32) keep it."""
    d = copy.deepcopy(MISTRAL_7B_V03_CONFIG)
    d.update(hidden_size=5120, head_dim=128)
    assert MistralConfig.from_configs(d).head_dim == 128


def test_defaults_match_v03():
    _assert_v03_shapes(MistralConfig())


def test_quantized_config_is_rejected():
    d = copy.deepcopy(MISTRAL_7B_V03_CONFIG)
    d["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(NotImplementedError, match="Quantized"):
        MistralConfig.from_configs(d)


# ── factory checks ───────────────────────────────────────────────────────


def test_factory_accepts_v03():
    MistralFactory._validate_config(_hf_config(), None)


def test_factory_rejects_sliding_window():
    """Mistral-7B-v0.1 uses a 4096-token window, which this port does not implement."""
    with pytest.raises(NotImplementedError, match="sliding_window"):
        MistralFactory._validate_config(_hf_config(sliding_window=4096), None)


def test_factory_rejects_quantization():
    cfg = _hf_config()
    cfg.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(NotImplementedError, match="Quantized"):
        MistralFactory._validate_config(cfg, None)


# ── registration ─────────────────────────────────────────────────────────


def test_plugin_entry_point_is_installed():
    from importlib.metadata import entry_points

    names = {ep.name: ep.value for ep in entry_points(group="vllm.general_plugins")}
    assert names.get("mistral_neuron") == "mistral_neuron.plugin:register", (
        "Run `pip install -e contrib/models/mistral` first."
    )


def test_register_overrides_the_vllm_gpu_class():
    from vllm import ModelRegistry

    from mistral_neuron.plugin import ARCHITECTURE, register

    register()
    register()  # idempotent
    cls = ModelRegistry.models[ARCHITECTURE].load_model_cls()
    assert cls.__module__ == "mistral_neuron.factory"


def test_vllm_front_end_sees_a_text_generation_model():
    """vLLM's front end inspects the registered class before starting the engine.

    Without the protocol stubs on the factory, vLLM rejects the model with
    "This model does not support `--runner generate`".
    """
    from vllm import ModelRegistry
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
        is_vllm_model,
    )

    from mistral_neuron.plugin import ARCHITECTURE, register

    assert is_vllm_model(MistralFactory)
    assert is_text_generation_model(MistralFactory)

    register()
    info = ModelRegistry.models[ARCHITECTURE].inspect_model_cls()
    assert info.is_text_generation_model
    assert not info.is_pooling_model


def test_factory_is_not_instantiated_directly():
    with pytest.raises(RuntimeError, match="from_configs"):
        MistralFactory(vllm_config=None)
