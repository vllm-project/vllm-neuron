# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the OLMo-3 contrib tests.

The HuggingFace reference side uses the ``transformers`` OLMo-3 classes and
does not import the code in ``src/``. The Neuron side builds the classes in
``src/`` through ``_local_src``. The ``*_model_load`` functions are top-level
functions, so that ``functools.partial`` can pickle them for the
``MPExecutor`` worker pool.

Two checkpoints are used:

* a tiny random-weight checkpoint with the OLMo-3 layout
  (:func:`make_tiny_checkpoint`), which every CPU test can build in seconds,
  and
* the real ``allenai/Olmo-3-7B-Instruct`` checkpoint, only when
  ``OLMO3_CHECKPOINT_PATH`` names a local snapshot (:func:`real_checkpoint`).
"""

import copy
import json
import os

import torch
import torch.distributed as dist

from _local_src import src_module

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
HF_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
CHECKPOINT_ENV = "OLMO3_CHECKPOINT_PATH"
ARCHITECTURE = "Olmo3ForCausalLM"

# Layer 0 is a sliding-window layer (default RoPE). Layer 3 is a
# full-attention layer (YaRN RoPE).
SLIDING_LAYER_IDX = 0
FULL_LAYER_IDX = 3

_LAYER_TYPES_7B = (["sliding_attention"] * 3 + ["full_attention"]) * 8

# ``config.json`` of allenai/Olmo-3-7B-Instruct at HF_REVISION, so that the
# config tests need no download. ``transformers_version`` and the training-only
# fields are left out.
REAL_CONFIG = {
    "architectures": [ARCHITECTURE],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "eos_token_id": 100257,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "layer_types": _LAYER_TYPES_7B,
    "max_position_embeddings": 65536,
    "model_type": "olmo3",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pad_token_id": 100277,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "attention_factor": 1.2079441541679836,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "yarn",
    },
    "rope_theta": 500000,
    "sliding_window": 4096,
    "tie_word_embeddings": False,
    "vocab_size": 100278,
}

# A 4-layer model with the OLMo-3 layout: MHA, head_dim 128, three
# sliding-window layers and then one full-attention layer, YaRN RoPE with an
# explicit attention factor on the full-attention layer, an untied LM head.
# The vocabulary (251) does not divide by 2, so TP 2 pads the LM head and the
# embedding by one row. The window (16) is far smaller than the real 4096, so
# that a short CPU prompt reaches it in prefill and in decode.
TINY_WINDOW = 16
TINY_CONFIG = {
    **REAL_CONFIG,
    "eos_token_id": 2,
    "pad_token_id": 1,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
    "max_position_embeddings": 512,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "rope_scaling": {
        **REAL_CONFIG["rope_scaling"],
        "original_max_position_embeddings": 64,
    },
    "sliding_window": TINY_WINDOW,
    "vocab_size": 251,
}


def hf_config_from_dict(config_dict: dict, sliding_window: int | None = None):
    """Build the transformers ``Olmo3Config`` the way ``from_pretrained`` does.

    transformers 5 moves ``rope_theta`` and ``rope_scaling`` to the per-layer-type
    ``rope_parameters`` dict, as it does for the checkpoint's ``config.json``.
    The attention runs in eager mode, so the HF reference applies the
    sliding window exactly.
    """
    from transformers import Olmo3Config as HFOlmo3Config

    config = HFOlmo3Config.from_dict(copy.deepcopy(config_dict))
    config._attn_implementation = "eager"
    if sliding_window is not None:
        config.sliding_window = sliding_window
    return config


def port_config(hf_config, sliding_window: int | None = None):
    """Return the ``src/`` config for ``hf_config``, with the same window."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    Olmo3Config = src_module("config").Olmo3Config
    cfg = Olmo3Config.from_configs(hf_config, NeuronConfig())
    if sliding_window is not None:
        cfg.sliding_window = sliding_window
    return cfg


def real_checkpoint() -> str | None:
    """Return the local real checkpoint directory, or ``None``."""
    path = os.environ.get(CHECKPOINT_ENV)
    if path and os.path.isfile(os.path.join(path, "config.json")):
        return path
    return None


def make_tiny_checkpoint(model_dir: str, seed: int = 42) -> str:
    """Save a random-weight BF16 checkpoint with ``TINY_CONFIG`` to ``model_dir``.

    The norm weights are ``1 + 0.1 * randn`` instead of ones, so that a norm
    that reads the wrong slice of its weight changes the output.
    """
    from transformers import Olmo3ForCausalLM as HFOlmo3ForCausalLM

    torch.manual_seed(seed)
    model = HFOlmo3ForCausalLM(hf_config_from_dict(TINY_CONFIG))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
    model.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir


# ---------------------------------------------------------------------------
# HuggingFace references
# ---------------------------------------------------------------------------


def _weight_map(checkpoint: str) -> dict[str, str]:
    """Return {checkpoint key: safetensors file name} for a checkpoint dir."""
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return json.load(f)["weight_map"]
    from safetensors import safe_open

    with safe_open(os.path.join(checkpoint, "model.safetensors"), "pt") as f:
        return {key: "model.safetensors" for key in f.keys()}


def load_hf_layer(checkpoint: str, hf_config, layer_idx: int):
    """Load one HF ``Olmo3DecoderLayer`` (BF16) from the safetensors files."""
    from safetensors import safe_open
    from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer as HFLayer

    prefix = f"model.layers.{layer_idx}."
    state = {}
    for key, filename in _weight_map(checkpoint).items():
        if not key.startswith(prefix):
            continue
        with safe_open(os.path.join(checkpoint, filename), framework="pt") as f:
            state[key[len(prefix) :]] = f.get_tensor(key)
    layer = HFLayer(hf_config, layer_idx).to(torch.bfloat16)
    missing, unexpected = layer.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return layer.eval()


def hf_rotary(hf_config):
    """Return the HF OLMo-3 rotary embedding (independent reference)."""
    from transformers.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding

    return Olmo3RotaryEmbedding(hf_config)


def additive_mask(seq_len: int, sliding_window: int | None) -> torch.Tensor:
    """Return the [1, 1, T, T] additive causal mask, with an optional window.

    A key ``k`` is visible to a query ``q`` when ``k <= q`` and, for a
    sliding-window layer, ``k > q - sliding_window`` (HF semantics).
    """
    q = torch.arange(seq_len).unsqueeze(1)
    k = torch.arange(seq_len).unsqueeze(0)
    blocked = k > q
    if sliding_window is not None:
        blocked = blocked | (k <= q - sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    mask = mask.masked_fill(blocked, torch.finfo(torch.float32).min)
    return mask[None, None]


def sample_hidden(sample_idx: int, seq_len: int, hidden: int) -> torch.Tensor:
    """Return a seeded BF16 hidden-state sample [T, H]."""
    torch.manual_seed(sample_idx)
    return torch.randn(seq_len, hidden, dtype=torch.bfloat16)


def compute_hf_layer_goldens(
    checkpoint: str,
    hf_config,
    layer_idx: int,
    seq_len: int,
    n_samples: int,
) -> dict:
    """Compute FP32 and BF16 HF goldens for one decoder layer on random inputs.

    The layer runs on the full sequence with the HF eager attention and the
    window of ``hf_config`` on a sliding-window layer.
    """
    hf_layer = load_hf_layer(checkpoint, hf_config, layer_idx)
    hf_layer_fp32 = copy.deepcopy(hf_layer).float()
    layer_type = hf_config.layer_types[layer_idx]
    window = hf_config.sliding_window if layer_type == "sliding_attention" else None
    mask = additive_mask(seq_len, window)
    rotary = hf_rotary(hf_config)
    pos = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)

    fp32_list, bf16_list = [], []
    for s in range(n_samples):
        h = sample_hidden(s, seq_len, hf_config.hidden_size).unsqueeze(0)
        with torch.no_grad():
            pe32 = rotary(h.float(), pos, layer_type)
            pe16 = rotary(h, pos, layer_type)
            y32 = hf_layer_fp32(
                h.float(), attention_mask=mask, position_embeddings=pe32
            )
            y16 = hf_layer(
                h, attention_mask=mask.to(torch.bfloat16), position_embeddings=pe16
            )
        fp32_list.append(y32[0].detach().float())
        bf16_list.append(y16[0].detach().float())
    return {"all_fp32": fp32_list, "all_bf16": bf16_list}


# ---------------------------------------------------------------------------
# Neuron-side model_load functions (top-level, picklable with partial)
# ---------------------------------------------------------------------------


def layer_mappings(layer_idx: int) -> dict:
    """Return the weight map of one decoder layer, relative to the layer."""
    olmo3_weight_mappings = src_module("model").olmo3_weight_mappings

    prefix = f"model.layers.{layer_idx}."
    full = olmo3_weight_mappings(layer_idx + 1)
    return {k[len(prefix) :]: v for k, v in full.items() if k.startswith(prefix)}


class _SPPrefillSlice(torch.nn.Module):
    """Give each rank its sequence-parallel chunk of the full prefill input.

    The decoder layer takes the SP layout in prefill (each rank holds
    ``T / tp`` rows) and gives the SP layout back.
    """

    def __init__(self, layer, layer_idx: int):
        super().__init__()
        self.layer = layer
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        meta = attn_metadata[self.layer_name]
        if meta["max_query_len"] > meta["decode_token_threshold"]:
            chunk = hidden_states.shape[0] // self.world_size
            start = self.rank * chunk
            hidden_states = hidden_states[start : start + chunk]
        return self.layer(
            hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )


def decoder_layer_model_load(
    neuron_cfg,
    checkpoint_path: str,
    layer_idx: int,
    num_blocks: int,
    block_size: int,
    device: str,
):
    """Build and load one ``Olmo3DecoderLayer`` with a zero KV cache (CPU mode)."""
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    Olmo3DecoderLayer = src_module("model").Olmo3DecoderLayer

    layer = Olmo3DecoderLayer(neuron_cfg, layer_idx)
    rank, ws = dist.get_rank(), dist.get_world_size()
    shards = SafetensorsCheckpoint(checkpoint_path, None).load_sharded(
        rank, ws, layer, layer_mappings(layer_idx), device
    )
    layer.load_state_dict(shards.state_dict, strict=False, assign=True)
    nkv = neuron_cfg.num_key_value_heads // ws
    shape = (num_blocks, nkv, block_size, neuron_cfg.head_dim)
    layer.self_attn.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    layer.self_attn.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    return _SPPrefillSlice(layer, layer_idx)


def prefill_metadata(layer_idx: int, seq_len: int, block_size: int, num_blocks: int):
    """Return the prefill attention metadata for one sequence from position 0."""
    return {
        f"layers.{layer_idx}.self_attn": {
            "max_query_len": seq_len,
            "decode_token_threshold": 1,
            "slot_mapping": torch.arange(seq_len, dtype=torch.int64),
            "block_size": block_size,
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32).unsqueeze(
                0
            ),
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }
