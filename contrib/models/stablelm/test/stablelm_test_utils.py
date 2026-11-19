# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the StableLM (stabilityai/stablelm-zephyr-3b) contrib tests.

This file travels with the model. It sits at the ``test/`` root because
``test/unit``, ``test/module`` and ``test/integration`` all import it
(``test/conftest.py`` puts this directory on ``sys.path``). The implementation
under ``src/`` is reached through ``_local_src``.

The helpers give:

- The pinned checkpoint (``MODEL_ID`` at ``HF_REVISION``) and its geometry.
  ``STABLELM_CHECKPOINT_PATH`` points at a local copy. Without it,
  ``resolve_checkpoint`` downloads the pinned revision from the HuggingFace hub.
- A tiny random-weight checkpoint (2 layers, hidden size 1280, head_dim 80,
  rotary dim 20, and at TP 2 the 864 MLP columns per rank of the real model
  at TP 8). The CPU tests use it, so they need no download.
- Independent HuggingFace references for layer 0 (attention and MLP). They
  come from the HF ``StableLm`` modules and the checkpoint tensors only. They
  do not use the code under ``src/``.
- Picklable ``model_load`` functions for ``MPExecutor`` and the attention
  metadata of a prefill step and a decode step.
- ``run_*`` functions that run one component on all ranks and return the
  FP32 HF, the BF16 HF and the port outputs for a three-way comparison. The
  unit tests run them on CPU with the tiny checkpoint, and the module tests
  run them on Neuron with the real checkpoint.
"""

from __future__ import annotations

import copy
import json
import os
from functools import partial

import torch
import torch.distributed as dist

from vllm_neuron import envs

from _local_src import src_module

MODEL_ID = "stabilityai/stablelm-zephyr-3b"
HF_REVISION = "fe1fd5e36ccb8cb2cb1dc225ab6d2962692b9837"
CHECKPOINT_ENV = "STABLELM_CHECKPOINT_PATH"
ARCHITECTURE = "StableLmForCausalLM"

# config.json of MODEL_ID at HF_REVISION, verbatim.
PINNED_CONFIG = {
    "architectures": ["StableLmForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 0,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 6912,
    "max_position_embeddings": 4096,
    "model_type": "stablelm",
    "layer_norm_eps": 1e-05,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.38.0",
    "use_cache": True,
    "use_qkv_bias": False,
    "vocab_size": 50304,
}

# stablelm-zephyr-3b geometry (from PINNED_CONFIG).
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 6912
NUM_LAYERS = 32
NUM_HEADS = 32
NUM_KV_HEADS = 32
HEAD_DIM = 80
ROTARY_DIM = 20
VOCAB_SIZE = 50304
# 11 tensors in each layer (q, k, v, o, gate, up, down, 2 LayerNorms with a
# weight and a bias) and 4 outside the layers (embedding, final LayerNorm
# weight and bias, lm_head).
TENSORS_PER_LAYER = 11
TOP_LEVEL_TENSORS = 4
NUM_CHECKPOINT_TENSORS = TOP_LEVEL_TENSORS + TENSORS_PER_LAYER * NUM_LAYERS

# The tensor-parallel degree of the port's device tests and of the
# integration test.
DEVICE_TP = 8

# The tiny checkpoint keeps head_dim 80, the 0.25 rotary factor and, at TP 2,
# the 864 MLP columns per rank of the real model at TP 8. The hidden size is
# 1280 (16 heads), because the decode QKV kernel splits hidden / 128 over the
# 2 logical cores of an LNC 2 NeuronCore, so hidden / 128 must be even (the
# real model has 2560 / 128 = 20).
TINY_OVERRIDES = {
    "hidden_size": 1280,
    "intermediate_size": 1728,
    "num_hidden_layers": 2,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "vocab_size": 512,
}
TINY_TP = 2

LAYER_IDX = 0
BLOCK_SIZE = 32
COLLECT_TIMEOUT_S = 3600.0


# =============================================================================
# Checkpoints
# =============================================================================


def resolve_checkpoint() -> str:
    """Return the local directory of the pinned checkpoint.

    ``STABLELM_CHECKPOINT_PATH`` points at a checkpoint that is already on
    disk. Without it, download the pinned revision from the HuggingFace hub
    (about 5.6 GB).
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise FileNotFoundError(
                f"{CHECKPOINT_ENV}={path!r} has no config.json. Set "
                f"{CHECKPOINT_ENV} to a directory with the checkpoint."
            )
        return path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    )


def write_pinned_config(directory: str) -> str:
    """Write ``PINNED_CONFIG`` as ``config.json`` into ``directory``."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(PINNED_CONFIG, f, indent=2)
    return directory


def tiny_hf_config(**overrides):
    """Return the HF ``StableLmConfig`` of the tiny checkpoint."""
    from transformers import StableLmConfig as HFStableLmConfig

    values = {
        k: v
        for k, v in PINNED_CONFIG.items()
        if k not in ("architectures", "model_type", "transformers_version")
    }
    values.update(TINY_OVERRIDES)
    values.update(overrides)
    config = HFStableLmConfig(**values)
    config.architectures = [ARCHITECTURE]
    return config


def make_tiny_checkpoint(directory: str, seed: int = 42, **overrides) -> str:
    """Save a random-weight tiny StableLM checkpoint in BF16 into ``directory``.

    The LayerNorm weights and biases are random too, so that a missing bias
    or a wrong norm shows up in the outputs.
    """
    from transformers import StableLmForCausalLM as HFStableLmForCausalLM

    torch.manual_seed(seed)
    model = HFStableLmForCausalLM(tiny_hf_config(**overrides))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "layernorm" in name or name.startswith("model.norm."):
                if name.endswith(".weight"):
                    param.copy_(1.0 + 0.1 * torch.randn_like(param))
                else:
                    param.copy_(0.1 * torch.randn_like(param))
    model.to(torch.bfloat16).save_pretrained(directory)
    return directory


def load_hf_config(checkpoint: str):
    """Load the HF config with the eager attention implementation."""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(checkpoint)
    hf_config._attn_implementation = "eager"
    return hf_config


def checkpoint_weight_map(checkpoint: str) -> dict[str, str]:
    """Return ``{tensor name: safetensors file}`` for a checkpoint directory."""
    from safetensors import safe_open

    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return json.load(f)["weight_map"]
    with safe_open(
        os.path.join(checkpoint, "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        return {name: "model.safetensors" for name in f.keys()}


def read_checkpoint_tensors(checkpoint: str, names: list[str]) -> dict:
    """Read the named tensors from the checkpoint without the port loaders."""
    from safetensors import safe_open

    weight_map = checkpoint_weight_map(checkpoint)
    out = {}
    for file_name in sorted({weight_map[n] for n in names}):
        with safe_open(
            os.path.join(checkpoint, file_name), framework="pt", device="cpu"
        ) as f:
            for name in names:
                if weight_map[name] == file_name:
                    out[name] = f.get_tensor(name)
    return out


# =============================================================================
# HuggingFace references
# =============================================================================


def load_hf_attention(checkpoint: str, layer_idx: int, dtype: torch.dtype):
    """Build the HF StableLmAttention for one layer with the checkpoint weights."""
    from transformers.models.stablelm.modeling_stablelm import (
        StableLmAttention as HFAttention,
    )

    hf_config = load_hf_config(checkpoint)
    prefix = f"model.layers.{layer_idx}.self_attn"
    names = [f"{prefix}.{p}_proj.weight" for p in ("q", "k", "v", "o")]
    tensors = read_checkpoint_tensors(checkpoint, names)
    attn = HFAttention(hf_config, layer_idx=layer_idx)
    attn.load_state_dict(
        {k.replace(f"{prefix}.", ""): v for k, v in tensors.items()}, strict=True
    )
    return attn.to(dtype).eval()


def load_hf_mlp(checkpoint: str, layer_idx: int, dtype: torch.dtype):
    """Build the HF StableLmMLP for one layer with the checkpoint weights."""
    from transformers.models.stablelm.modeling_stablelm import StableLmMLP as HFMLP

    hf_config = load_hf_config(checkpoint)
    prefix = f"model.layers.{layer_idx}.mlp"
    names = [f"{prefix}.{p}_proj.weight" for p in ("gate", "up", "down")]
    tensors = read_checkpoint_tensors(checkpoint, names)
    mlp = HFMLP(hf_config)
    mlp.load_state_dict(
        {k.replace(f"{prefix}.", ""): v for k, v in tensors.items()}, strict=True
    )
    return mlp.to(dtype).eval()


def hf_rotary(checkpoint: str):
    """Return the HF partial-RoPE module for this checkpoint."""
    from transformers.models.stablelm.modeling_stablelm import (
        StableLmRotaryEmbedding as HFRotaryEmbedding,
    )

    return HFRotaryEmbedding(config=load_hf_config(checkpoint))


def _causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(torch.float32).min), diagonal=1
    )[None, None]


def hf_attention_prefill(checkpoint, hidden, cache_bf16=None, cache_fp32=None):
    """Run the HF attention prefill in BF16 and FP32. Return (bf16, fp32)."""
    seq_len = hidden.shape[1]
    hf_attn = load_hf_attention(checkpoint, LAYER_IDX, torch.bfloat16)
    hf_attn_fp32 = copy.deepcopy(hf_attn).float()
    positions = torch.arange(seq_len, dtype=torch.int64)[None]
    mask = _causal_mask(seq_len)
    with torch.no_grad():
        pe = hf_rotary(checkpoint)(hidden, positions)
        out_bf16, _ = hf_attn(
            hidden,
            attention_mask=mask.to(torch.bfloat16),
            position_ids=positions,
            past_key_values=cache_bf16,
            position_embeddings=pe,
        )
        out_fp32, _ = hf_attn_fp32(
            hidden.float(),
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache_fp32,
            position_embeddings=(pe[0].float(), pe[1].float()),
        )
    return out_bf16[0], out_fp32[0]


def hf_attention_decode(checkpoint, hidden, pos, cache, dtype):
    """Run one HF attention decode step against a filled DynamicCache."""
    hf_attn = load_hf_attention(checkpoint, LAYER_IDX, dtype)
    positions = torch.tensor([[pos]], dtype=torch.int64)
    with torch.no_grad():
        pe = hf_rotary(checkpoint)(hidden.to(dtype), positions)
        out, _ = hf_attn(
            hidden.to(dtype),
            attention_mask=None,
            position_ids=positions,
            past_key_values=cache,
            position_embeddings=pe,
        )
    return out[0]


# =============================================================================
# Port modules on MPExecutor ranks
# =============================================================================


def component_device() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


def num_blocks_for(total_tokens: int) -> int:
    """Return the KV block count for ``total_tokens`` (plus the null block).

    On Neuron the count is a multiple of LNC 2 * 128, as in the plugin's other
    decode cycle tests.
    """
    blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    if envs.VLLM_NEURON_CPU_MODE:
        return blocks
    alignment = 2 * 128
    return ((blocks + alignment - 1) // alignment) * alignment


def attn_metadata(slot_mapping: torch.Tensor, num_blocks: int, max_query_len: int):
    """Return the attention metadata of one step for layer ``LAYER_IDX``."""
    return {
        f"layers.{LAYER_IDX}.self_attn": {
            "max_query_len": max_query_len,
            "decode_token_threshold": 1,
            "slot_mapping": slot_mapping,
            "block_size": BLOCK_SIZE,
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32)[None],
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def attention_mappings(layer_idx: int) -> dict:
    """Return the checkpoint mappings for one StableLmAttention module."""
    prefix = f"model.layers.{layer_idx}.self_attn"
    return {
        "qkv_proj_weight": [
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
        ],
        "o_proj_weight": f"{prefix}.o_proj.weight",
    }


def mlp_mappings(layer_idx: int) -> dict:
    """Return the checkpoint mappings for one StableLmMLP module."""
    prefix = f"model.layers.{layer_idx}.mlp"
    return {
        "gate_proj_weight": f"{prefix}.gate_proj.weight",
        "up_proj_weight": f"{prefix}.up_proj.weight",
        "down_proj_weight": f"{prefix}.down_proj.weight",
    }


def port_config(checkpoint: str):
    """Return the port ``StableLmConfig`` of a checkpoint."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    config_cls = src_module("config").StableLmConfig
    return config_cls.from_configs(load_hf_config(checkpoint), NeuronConfig())


def port_rope(checkpoint: str, positions: torch.Tensor):
    """Return the port cos/sin tables (BF16) for ``positions``."""
    rotary_cls = src_module("model").StableLmRotaryEmbedding
    return rotary_cls(port_config(checkpoint))(
        positions, device=torch.device("cpu"), dtype=torch.bfloat16
    )


def _maybe_compile(module: torch.nn.Module) -> torch.nn.Module:
    """Compile for Neuron. CPU mode runs the torch fallbacks eagerly."""
    if envs.VLLM_NEURON_CPU_MODE:
        return module
    return torch.compile(
        module, fullgraph=True, backend=envs.get_compile_backend_name()
    )


def attention_load(checkpoint: str, prefill_len: int, num_blocks: int, device: str):
    """Top-level picklable model load: one StableLmAttention for this rank.

    The forward takes a ``phase``. In the prefill phase each rank gets its
    sequence-parallel slice of the prompt. The decode phase uses the full input.
    """
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    attention_cls = src_module("model").StableLmAttention
    rank = dist.get_rank()
    ws = dist.get_world_size()
    attn = attention_cls(port_config(checkpoint), layer_idx=LAYER_IDX)
    sd = (
        SafetensorsCheckpoint(checkpoint, None)
        .load_sharded_pipelined(
            rank, ws, attn, attention_mappings(LAYER_IDX), torch.device(device)
        )
        .state_dict
    )
    attn.load_state_dict(sd, strict=False, assign=True)

    nkh = attn.num_key_value_heads_per_rank
    cache_shape = (num_blocks, nkh, BLOCK_SIZE, attn.head_dim)
    attn.k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    attn.v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    per_rank = prefill_len // ws
    original_forward = attn.forward

    def wrapped_forward(
        hidden_states, positions, position_embeddings, attn_metadata, phase
    ):
        if phase == "prefill":
            start = rank * per_rank
            hidden_states = hidden_states[start : start + per_rank]
        return original_forward(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    attn.forward = wrapped_forward
    return _maybe_compile(attn)


def mlp_load(checkpoint: str, num_tokens: int, device: str):
    """Top-level picklable model load: one StableLmMLP for this rank."""
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    mlp_cls = src_module("model").StableLmMLP
    rank = dist.get_rank()
    ws = dist.get_world_size()
    mlp = mlp_cls(port_config(checkpoint))
    sd = (
        SafetensorsCheckpoint(checkpoint, None)
        .load_sharded_pipelined(
            rank, ws, mlp, mlp_mappings(LAYER_IDX), torch.device(device)
        )
        .state_dict
    )
    mlp.load_state_dict(sd, strict=False, assign=True)

    per_rank = num_tokens // ws
    original_forward = mlp.forward

    def wrapped_forward(hidden_states, phase):
        is_prefill = phase == "prefill"
        if is_prefill:
            start = rank * per_rank
            hidden_states = hidden_states[start : start + per_rank]
        return original_forward(hidden_states, is_prefill=is_prefill)

    mlp.forward = wrapped_forward
    return _maybe_compile(mlp)


def _executor(model_load, world_size: int):
    from vllm_neuron.utils.executor import MPExecutor

    device = component_device()
    return MPExecutor(
        world_size=world_size,
        model_load=partial(model_load, device=device),
        device=device,
    )


def _assert_ranks_equal(outputs: list[torch.Tensor], name: str) -> None:
    from vllm_neuron.accuracy.testing import assert_close

    for i in range(1, len(outputs)):
        assert_close(outputs[i], outputs[0], name=f"{name}/rank_{i}")


def run_attention_prefill(checkpoint: str, world_size: int, seq_len: int, seed: int):
    """Run the attention prefill on all ranks.

    Returns:
        ``(fp32 HF, bf16 HF, port)`` outputs, each ``[seq_len, hidden]``.
    """
    torch.manual_seed(seed)
    hidden_size = load_hf_config(checkpoint).hidden_size
    num_blocks = num_blocks_for(seq_len)
    hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(seq_len, dtype=torch.int64)

    ref_bf16, ref_fp32 = hf_attention_prefill(checkpoint, hidden)

    executor = _executor(
        partial(
            attention_load,
            checkpoint=checkpoint,
            prefill_len=seq_len,
            num_blocks=num_blocks,
        ),
        world_size,
    )
    executor.dispatch(
        hidden[0],
        positions,
        port_rope(checkpoint, positions),
        attn_metadata(positions.clone(), num_blocks, seq_len),
        "prefill",
    )
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
    actual = torch.cat([torch.as_tensor(o).cpu() for o in outputs], dim=0)
    return ref_fp32, ref_bf16, actual


def run_attention_prefill_to_decode(
    checkpoint: str, world_size: int, prefill_len: int, seed: int
):
    """Run a prefill that fills the paged KV cache, then one decode step.

    The HF reference fills a ``DynamicCache`` with the same prompt and runs one
    decode step against it.

    Returns:
        ``(fp32 HF, bf16 HF, port)`` decode outputs, each ``[1, hidden]``.
    """
    from transformers import DynamicCache

    torch.manual_seed(seed)
    hidden_size = load_hf_config(checkpoint).hidden_size
    num_blocks = num_blocks_for(prefill_len + 1)
    prompt = torch.randn(1, prefill_len, hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)

    cache_bf16, cache_fp32 = DynamicCache(), DynamicCache()
    hf_attention_prefill(checkpoint, prompt, cache_bf16, cache_fp32)
    ref_bf16 = hf_attention_decode(
        checkpoint, decode_hidden, prefill_len, cache_bf16, torch.bfloat16
    )
    ref_fp32 = hf_attention_decode(
        checkpoint, decode_hidden, prefill_len, cache_fp32, torch.float32
    )

    executor = _executor(
        partial(
            attention_load,
            checkpoint=checkpoint,
            prefill_len=prefill_len,
            num_blocks=num_blocks,
        ),
        world_size,
    )
    prefill_positions = torch.arange(prefill_len, dtype=torch.int64)
    executor.dispatch(
        prompt[0],
        prefill_positions,
        port_rope(checkpoint, prefill_positions),
        attn_metadata(prefill_positions.clone(), num_blocks, prefill_len),
        "prefill",
    )
    executor.collect(timeout=COLLECT_TIMEOUT_S)

    decode_positions = torch.tensor([prefill_len], dtype=torch.int64)
    executor.dispatch(
        decode_hidden[0],
        decode_positions,
        port_rope(checkpoint, decode_positions),
        attn_metadata(decode_positions.clone(), num_blocks, 1),
        "decode",
    )
    outputs = [
        torch.as_tensor(o).cpu() for o in executor.collect(timeout=COLLECT_TIMEOUT_S)
    ]
    # The all-reduce gives the same decode output on each rank.
    _assert_ranks_equal(outputs, "attention_decode")
    return ref_fp32, ref_bf16, outputs[0].reshape(ref_bf16.shape)


def run_mlp(checkpoint: str, world_size: int, phase: str, num_tokens: int, seed: int):
    """Run the MLP on all ranks for the prefill or the decode phase.

    Returns:
        ``(fp32 HF, bf16 HF, port)`` outputs, each ``[num_tokens, hidden]``.
    """
    torch.manual_seed(seed)
    hidden_size = load_hf_config(checkpoint).hidden_size
    hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)

    hf_mlp = load_hf_mlp(checkpoint, LAYER_IDX, torch.bfloat16)
    hf_mlp_fp32 = copy.deepcopy(hf_mlp).float()
    with torch.no_grad():
        ref_bf16 = hf_mlp(hidden)
        ref_fp32 = hf_mlp_fp32(hidden.float())

    executor = _executor(
        partial(mlp_load, checkpoint=checkpoint, num_tokens=num_tokens), world_size
    )
    executor.dispatch(hidden, phase)
    outputs = [
        torch.as_tensor(o).cpu() for o in executor.collect(timeout=COLLECT_TIMEOUT_S)
    ]
    if phase == "prefill":
        actual = torch.cat(outputs, dim=0)
    else:
        _assert_ranks_equal(outputs, "mlp_decode")
        actual = outputs[0].reshape(ref_bf16.shape)
    return ref_fp32, ref_bf16, actual
