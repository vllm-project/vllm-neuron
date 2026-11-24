# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Gemma3 (rnj-1) contrib tests.

The HuggingFace reference side uses the built-in ``transformers`` Gemma3 text
classes and does not import the code in ``src/``. The Neuron side builds the
classes in ``src/`` through ``_local_src``.

Two configs are used:

* ``REAL_CONFIG``, the ``config.json`` of ``EssentialAI/rnj-1-instruct`` at
  the pinned revision, so that the config, RoPE and weight-map tests need no
  download (the checkpoint stores about 33 GB of FP32 weights), and
* ``TINY_CONFIG``, a random-weight model with the same layout, which
  :func:`make_tiny_checkpoint` saves in seconds for the CPU end-to-end test.
"""

import copy
import os

import torch

from _local_src import src_module

MODEL_ID = "EssentialAI/rnj-1-instruct"
HF_REVISION = "5ef96d7a82b2bff8009192979432c9bc64e870cd"
CHECKPOINT_ENV = "GEMMA3_CHECKPOINT_PATH"
ARCHITECTURE = "Gemma3ForCausalLM"

# ``config.json`` of EssentialAI/rnj-1-instruct at HF_REVISION. Only
# ``transformers_version`` is left out. The checkpoint stores ``layer_type``
# (singular), which transformers ignores: it derives ``layer_types`` from
# ``sliding_window_pattern`` (1, so every layer is a full-attention layer).
REAL_CONFIG = {
    "architectures": [ARCHITECTURE],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": None,
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "head_dim": 128,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 16384,
    "layer_type": ["full_attention"] * 32,
    "max_position_embeddings": 32768,
    "model_type": "gemma3_text",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pad_token_id": 0,
    "query_pre_attn_scalar": 128,
    "rms_norm_eps": 1e-06,
    "rope_local_base_freq": 10000,
    "rope_scaling": {
        "attn_factor": 1.0,
        "beta_fast": 64.0,
        "beta_slow": 1.0,
        "extrapolation_factor": 1.0,
        "factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "yarn",
    },
    "rope_theta": 10000,
    "sliding_window": 32768,
    "sliding_window_pattern": 1,
    "torch_dtype": "float32",
    "use_cache": True,
    "vocab_size": 128256,
}

# A 2-layer model with the rnj-1 layout: GQA (4 Q heads, 2 KV heads) with
# head_dim 128, full attention in every layer, YaRN RoPE, the final logit soft
# cap, tied embeddings and FP32 weights. ``original_max_position_embeddings``
# is 64, so the YaRN ramp changes the frequencies of a short CPU prompt. The
# vocabulary (256) divides by the TP sizes of the CPU test.
TINY_CONFIG = {
    **REAL_CONFIG,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "layer_type": ["full_attention"] * 2,
    "max_position_embeddings": 512,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rope_scaling": {
        **REAL_CONFIG["rope_scaling"],
        "original_max_position_embeddings": 64,
    },
    "sliding_window": 512,
    "vocab_size": 256,
}


def hf_config_from_dict(config_dict: dict):
    """Build the transformers ``Gemma3TextConfig`` the way ``from_pretrained`` does.

    transformers 5 moves ``rope_theta`` and ``rope_scaling`` to the
    per-layer-type ``rope_parameters`` dict, as it does for the checkpoint's
    ``config.json``.
    """
    from transformers import Gemma3TextConfig

    return Gemma3TextConfig.from_dict(copy.deepcopy(config_dict))


def src_config(hf_config):
    """Return the ``src/`` config for ``hf_config``."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    Gemma3Config = src_module("config").Gemma3Config
    return Gemma3Config.from_configs(hf_config, NeuronConfig())


def real_checkpoint() -> str | None:
    """Return the local real checkpoint directory, or ``None``."""
    path = os.environ.get(CHECKPOINT_ENV)
    if path and os.path.isfile(os.path.join(path, "config.json")):
        return path
    return None


def make_tiny_checkpoint(model_dir: str, seed: int = 42) -> str:
    """Save a random-weight FP32 checkpoint with ``TINY_CONFIG`` to ``model_dir``.

    Gemma3 norm weights are zero-centered (the norm multiplies by ``1 + w``).
    The HF init sets them to zero, so they are replaced with ``0.1 * randn``:
    a norm that drops the ``1 +`` or reads the wrong weight changes the output.
    The weights stay FP32, as in the real checkpoint, so the load path casts
    them to the BF16 served dtype.
    """
    from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM

    torch.manual_seed(seed)
    model = HFGemma3ForCausalLM(hf_config_from_dict(TINY_CONFIG))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(0.1 * torch.randn_like(param))
    model.to(torch.float32).save_pretrained(model_dir)
    return model_dir
