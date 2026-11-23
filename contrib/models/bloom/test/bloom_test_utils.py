# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Bloom (bigscience/bloomz-560m) contrib tests.

This file travels with the model: it is the port's own test helper, vendored
into the contrib folder so that nothing outside ``contrib/models/bloom/`` is
needed. It sits at the ``test/`` root because ``test/unit`` and
``test/module`` both import it (``test/conftest.py`` puts this directory on
``sys.path``). The implementation under ``src/`` is reached through
``_local_src``.

The helpers give:
- The checkpoint path of the pinned revision.
- Independent HuggingFace references for layer 0 (attention and MLP). The
  references come from the HF ``Bloom`` modules only. They do not use the port
  code.
- Picklable ``model_load`` functions and harness modules for ``MPExecutor``.
- Attention metadata builders for prefill and decode.

The checkpoint comes from the environment: ``BLOOM_CHECKPOINT_PATH`` points at
a checkpoint that is already on disk. Without it, the pinned revision comes
from the HuggingFace hub (about 1.1 GB: ``*.json`` and ``model.safetensors``).
"""

from __future__ import annotations

import glob
import json
import os
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from vllm_neuron import envs

from _local_src import src_module

MODEL_ID = "bigscience/bloomz-560m"
HF_REVISION = "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05"
ARCHITECTURE = "BloomForCausalLM"
LAYER_IDX = 0
N_SAMPLES = 4
RANDOM_SEED = 42

CHECKPOINT_ENV = "BLOOM_CHECKPOINT_PATH"

# The decode kernel reads the cache in groups of LNC x 128 blocks.
BLOCK_SIZE = 32
BLOCKS_ALIGNMENT = 2 * 128

COLLECT_TIMEOUT_S = 1800


# =============================================================================
# Checkpoint
# =============================================================================


def get_checkpoint() -> str:
    """Return the local checkpoint directory of the pinned Bloom revision.

    ``BLOOM_CHECKPOINT_PATH`` points at a checkpoint that is already on disk.
    Without it, download the pinned revision from the HuggingFace hub.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise FileNotFoundError(
                f"{CHECKPOINT_ENV}={path} has no config.json. Set the variable to "
                "a Bloom checkpoint directory."
            )
        return path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.json", "*.safetensors"],
    )


def load_hf_config_dict(checkpoint: str) -> dict:
    """Read config.json as a dict. The dict is picklable for MPExecutor."""
    with open(os.path.join(checkpoint, "config.json")) as f:
        return json.load(f)


# HF BloomConfig accepts the old and the new field names (attribute_map).
# config.json of bigscience/bloomz-560m uses n_embed and n_layer, but it
# uses num_attention_heads and not n_head.
_HF_CONFIG_ALIASES = {
    "hidden_size": ("hidden_size", "n_embed"),
    "num_attention_heads": ("num_attention_heads", "n_head"),
    "num_hidden_layers": ("num_hidden_layers", "n_layer"),
}


def hf_config_value(hf_config_dict: dict, name: str):
    """Read a field from config.json. Accept each HF alias of the field."""
    for key in _HF_CONFIG_ALIASES.get(name, (name,)):
        if key in hf_config_dict:
            return hf_config_dict[key]
    raise KeyError(
        f"config.json has no {name} field (aliases: {_HF_CONFIG_ALIASES.get(name)})."
    )


def get_device() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


# =============================================================================
# Checkpoint keys
# =============================================================================


def checkpoint_prefix(checkpoint: str) -> str:
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    names = SafetensorsCheckpoint(checkpoint, None).get_tensor_names()
    return src_module("model").BloomForCausalLM.detect_checkpoint_prefix(names)


def layer_key(prefix: str, name: str, layer_idx: int = LAYER_IDX) -> str:
    return f"{prefix}h.{layer_idx}.{name}"


def attention_mappings(prefix: str, layer_idx: int = LAYER_IDX) -> dict:
    """Map the BloomAttention parameters to the checkpoint keys of one layer."""
    return {
        "qkv_proj_weight": [
            layer_key(prefix, "self_attention.query_key_value.weight", layer_idx),
            layer_key(prefix, "self_attention.query_key_value.bias", layer_idx),
        ],
        "o_proj_weight": layer_key(prefix, "self_attention.dense.weight", layer_idx),
        "o_proj_bias": layer_key(prefix, "self_attention.dense.bias", layer_idx),
    }


def mlp_mappings(prefix: str, layer_idx: int = LAYER_IDX) -> dict:
    """Map the BloomMLP parameters to the checkpoint keys of one layer."""
    return {
        "up_proj_weight": layer_key(prefix, "mlp.dense_h_to_4h.weight", layer_idx),
        "up_proj_bias": layer_key(prefix, "mlp.dense_h_to_4h.bias", layer_idx),
        "down_proj_weight": layer_key(prefix, "mlp.dense_4h_to_h.weight", layer_idx),
        "down_proj_bias": layer_key(prefix, "mlp.dense_4h_to_h.bias", layer_idx),
    }


def checkpoint_tensor_names(checkpoint: str) -> set[str]:
    """Return the tensor names of the safetensors files of the checkpoint."""
    from safetensors import safe_open

    keys = set()
    for path in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            keys.update(f.keys())
    return keys


def read_checkpoint_tensors(checkpoint: str, keys: list[str]) -> dict:
    """Read the given tensors from the safetensors files of the checkpoint."""
    from safetensors import safe_open

    wanted = set(keys)
    out = {}
    for path in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for key in wanted.intersection(f.keys()):
                out[key] = f.get_tensor(key)
    missing = wanted - set(out)
    assert not missing, f"Keys {sorted(missing)} are not in the checkpoint."
    return out


# =============================================================================
# Independent HF references
# =============================================================================


def _hf_layer_state(checkpoint: str, prefix: str, sub: str, names: list[str]) -> dict:
    keys = [layer_key(prefix, f"{sub}.{n}") for n in names]
    tensors = read_checkpoint_tensors(checkpoint, keys)
    return {n: tensors[k] for n, k in zip(names, keys)}


def build_hf_attention(checkpoint: str, dtype: torch.dtype):
    """Build HF BloomAttention of layer LAYER_IDX with the checkpoint weights."""
    from transformers import BloomConfig as HFBloomConfig
    from transformers.models.bloom.modeling_bloom import BloomAttention as HFAttn

    hf_config = HFBloomConfig.from_pretrained(checkpoint)
    prefix = checkpoint_prefix(checkpoint)
    attn = HFAttn(hf_config, layer_idx=LAYER_IDX)
    state = _hf_layer_state(
        checkpoint,
        prefix,
        "self_attention",
        [
            "query_key_value.weight",
            "query_key_value.bias",
            "dense.weight",
            "dense.bias",
        ],
    )
    attn.load_state_dict({k: v.to(torch.float32) for k, v in state.items()})
    return attn.to(dtype).eval()


def build_hf_mlp(checkpoint: str, dtype: torch.dtype):
    """Build HF BloomMLP of layer LAYER_IDX with the checkpoint weights."""
    from transformers import BloomConfig as HFBloomConfig
    from transformers.models.bloom.modeling_bloom import BloomMLP as HFMLP

    hf_config = HFBloomConfig.from_pretrained(checkpoint)
    prefix = checkpoint_prefix(checkpoint)
    mlp = HFMLP(hf_config)
    state = _hf_layer_state(
        checkpoint,
        prefix,
        "mlp",
        [
            "dense_h_to_4h.weight",
            "dense_h_to_4h.bias",
            "dense_4h_to_h.weight",
            "dense_4h_to_h.bias",
        ],
    )
    mlp.load_state_dict({k: v.to(torch.float32) for k, v in state.items()})
    return mlp.to(dtype).eval()


@torch.no_grad()
def hf_attention_forward(attn, hidden: torch.Tensor) -> torch.Tensor:
    """Run HF BloomAttention with causal ALiBi on ``hidden`` [T, H].

    The HF module adds the residual. The residual is zero, so the result is
    the attention output only.
    """
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor

    dtype = attn.dense.weight.dtype
    seq_len = hidden.shape[0]
    x = hidden.to(dtype).unsqueeze(0)
    alibi = build_alibi_tensor(
        torch.ones(1, seq_len, dtype=torch.long), attn.num_heads, dtype=dtype
    )
    mask = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype),
        diagonal=1,
    )[None, None]
    out, _ = attn(x, torch.zeros_like(x), alibi, mask)
    return out[0].detach()


@torch.no_grad()
def hf_mlp_forward(mlp, hidden: torch.Tensor) -> torch.Tensor:
    dtype = mlp.dense_h_to_4h.weight.dtype
    x = hidden.to(dtype).unsqueeze(0)
    return mlp(x, torch.zeros_like(x))[0].detach()


def make_hidden(num_tokens: int, hidden_size: int, sample: int) -> torch.Tensor:
    """Return a seeded LayerNorm-scale input in bfloat16."""
    torch.manual_seed(RANDOM_SEED + sample)
    return torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)


def compute_attention_goldens(checkpoint, hf_config_dict, seq_len, n_samples):
    """Compute FP32 and BF16 HF attention outputs over ``seq_len`` tokens."""
    attn_fp32 = build_hf_attention(checkpoint, torch.float32)
    attn_bf16 = build_hf_attention(checkpoint, torch.bfloat16)
    hidden_size = hf_config_value(hf_config_dict, "hidden_size")
    fp32, bf16 = [], []
    for s in range(n_samples):
        h = make_hidden(seq_len, hidden_size, s)
        fp32.append(hf_attention_forward(attn_fp32, h.float()))
        bf16.append(hf_attention_forward(attn_bf16, h))
    return {"all_fp32": fp32, "all_bf16": bf16}


def compute_mlp_goldens(checkpoint, hf_config_dict, num_tokens, n_samples):
    """Compute FP32 and BF16 HF MLP outputs for ``num_tokens`` tokens."""
    mlp_fp32 = build_hf_mlp(checkpoint, torch.float32)
    mlp_bf16 = build_hf_mlp(checkpoint, torch.bfloat16)
    hidden_size = hf_config_value(hf_config_dict, "hidden_size")
    fp32, bf16 = [], []
    for s in range(n_samples):
        h = make_hidden(num_tokens, hidden_size, s)
        fp32.append(hf_mlp_forward(mlp_fp32, h.float()))
        bf16.append(hf_mlp_forward(mlp_bf16, h))
    return {"all_fp32": fp32, "all_bf16": bf16}


def golden_key(kind: str, **fields) -> dict:
    key = {
        "model_id": MODEL_ID,
        "revision": HF_REVISION,
        "layer_idx": LAYER_IDX,
        "kind": kind,
        "n_samples": N_SAMPLES,
        "seed": RANDOM_SEED,
    }
    key.update(fields)
    return key


# =============================================================================
# Attention metadata
# =============================================================================


def aligned_num_blocks(total_tokens: int) -> int:
    min_blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    return max(
        BLOCKS_ALIGNMENT,
        ((min_blocks + BLOCKS_ALIGNMENT - 1) // BLOCKS_ALIGNMENT) * BLOCKS_ALIGNMENT,
    )


def prefill_metadata(start: int, num_tokens: int, num_blocks: int) -> dict:
    """Build the prefill attn_metadata of one sequence (tokens start..start+n)."""
    return {
        f"layers.{LAYER_IDX}.self_attn": {
            "max_query_len": num_tokens,
            "decode_token_threshold": 1,
            "slot_mapping": torch.arange(start, start + num_tokens, dtype=torch.long),
            "block_size": BLOCK_SIZE,
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32).unsqueeze(
                0
            ),
            "max_blocks_per_seq": num_blocks,
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def decode_metadata(position: int, num_blocks: int) -> dict:
    """Build the decode attn_metadata of one token at ``position``."""
    return {
        f"layers.{LAYER_IDX}.self_attn": {
            "max_query_len": 1,
            "decode_token_threshold": 1,
            "slot_mapping": torch.tensor([position], dtype=torch.long),
            "block_size": BLOCK_SIZE,
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32).unsqueeze(
                0
            ),
            "max_blocks_per_seq": num_blocks,
        }
    }


# =============================================================================
# MPExecutor harnesses (top-level, picklable model_load functions)
# =============================================================================


class AttentionHarness(nn.Module):
    """Give each rank its SP slice on prefill. Decode gets the full input."""

    def __init__(self, attn: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.attn = attn
        self.rank = rank
        self.world_size = world_size

    def forward(self, hidden_states, positions, attn_metadata, is_prefill: bool):
        if is_prefill:
            per_rank = hidden_states.shape[0] // self.world_size
            start = self.rank * per_rank
            hidden_states = hidden_states[start : start + per_rank]
        return self.attn(hidden_states, positions, attn_metadata)


class MLPHarness(nn.Module):
    """Give each rank its SP slice on prefill. Decode gets the full input."""

    def __init__(self, mlp: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.mlp = mlp
        self.rank = rank
        self.world_size = world_size

    def forward(self, hidden_states, is_prefill: bool):
        if is_prefill:
            per_rank = hidden_states.shape[0] // self.world_size
            start = self.rank * per_rank
            hidden_states = hidden_states[start : start + per_rank]
        return self.mlp(hidden_states, is_prefill=is_prefill)


def _load_into(module, checkpoint, mappings, device):
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    rank = dist.get_rank()
    ws = dist.get_world_size()
    ckpt = SafetensorsCheckpoint(checkpoint, None)
    if device == "cpu":
        shards = ckpt.load_sharded(rank, ws, module, mappings, device)
    else:
        shards = ckpt.load_sharded_pipelined(rank, ws, module, mappings, device)
    module.load_state_dict(shards.state_dict, strict=False, assign=True)


def load_attention(checkpoint, hf_config_dict, prefix, num_blocks, device):
    """model_load for MPExecutor: one BloomAttention layer with a KV cache."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    config_mod = src_module("config")
    model_mod = src_module("model")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    config = config_mod.BloomConfig.from_configs(hf_config_dict, NeuronConfig())
    attn = model_mod.BloomAttention(config, layer_idx=LAYER_IDX)
    _load_into(attn, checkpoint, attention_mappings(prefix), device)

    cache_shape = (
        num_blocks,
        attn.num_key_value_heads_per_rank,
        BLOCK_SIZE,
        attn.padded_head_dim,
    )
    attn.k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    attn.v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    return AttentionHarness(attn, rank, ws)


def load_mlp(checkpoint, hf_config_dict, prefix, device):
    """model_load for MPExecutor: one BloomMLP layer."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    config_mod = src_module("config")
    model_mod = src_module("model")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    config = config_mod.BloomConfig.from_configs(hf_config_dict, NeuronConfig())
    mlp = model_mod.BloomMLP(config)
    _load_into(mlp, checkpoint, mlp_mappings(prefix), device)
    return MLPHarness(mlp, rank, ws)


def make_attention_executor(checkpoint, hf_config_dict, num_blocks, world_size):
    from vllm_neuron.utils.executor import MPExecutor

    device = get_device()
    model_load = partial(
        load_attention,
        checkpoint,
        hf_config_dict,
        checkpoint_prefix(checkpoint),
        num_blocks,
        device,
    )
    return MPExecutor(world_size=world_size, model_load=model_load, device=device)


def make_mlp_executor(checkpoint, hf_config_dict, world_size):
    from vllm_neuron.utils.executor import MPExecutor

    device = get_device()
    model_load = partial(
        load_mlp,
        checkpoint,
        hf_config_dict,
        checkpoint_prefix(checkpoint),
        device,
    )
    return MPExecutor(world_size=world_size, model_load=model_load, device=device)
