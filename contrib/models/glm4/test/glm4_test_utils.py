# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the GLM-4-0414 (zai-org/GLM-4-32B-0414) contrib tests.

This file holds the checkpoint helpers, the independent references and the
weight-loader drivers that the tests in this folder share. It sits at the
``test/`` root because ``test/unit`` and ``test/integration`` both import it
(``test/conftest.py`` puts this directory on ``sys.path``). The implementation
under ``src/`` is reached through ``_local_src``.

Checkpoint resolution:

* ``GLM4_0414_CHECKPOINT_PATH`` points at a local copy of the pinned revision.
  The CPU tests that read the weights (about 65 GB on disk) run only when it is
  set. Without it, the integration test downloads the pinned revision.
* Without it, the CPU tests that read only the checkpoint metadata
  (``config.json``, ``model.safetensors.index.json`` and the tokenizer files,
  about 20 MB) download those files of the pinned revision from the
  HuggingFace hub. They skip when the hub cannot be reached.

The module also holds ``load_tiny_decoder_layer``, the picklable model-load
function that the ``MPExecutor`` decoder-layer test sends to its CPU workers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

from _local_src import src_module

MODEL_ID = "zai-org/GLM-4-32B-0414"
# The revision that the tests and the serve example in README.md pin.
HF_REVISION = "077b5c2f5c43bd3239fd605a0600229e8facbd4a"
# Namespaces the cached HF goldens by checkpoint revision.
GOLDEN_VARIANT = f"rev_{HF_REVISION[:12]}"
ARCHITECTURE = "Glm4ForCausalLM"

CHECKPOINT_ENV = "GLM4_0414_CHECKPOINT_PATH"

_METADATA_PATTERNS = ["*.json", "tokenizer*"]

LAYER_IDX = 0


# =============================================================================
# Checkpoint resolution
# =============================================================================


def _checked(path: str) -> str:
    if not (Path(path) / "config.json").is_file():
        raise FileNotFoundError(
            f"{CHECKPOINT_ENV}={path!r} has no config.json. Set {CHECKPOINT_ENV} "
            f"to a local copy of {MODEL_ID} at revision {HF_REVISION}."
        )
    return path


def metadata_path() -> str:
    """Return a directory with the checkpoint metadata of the pinned revision.

    The directory holds ``config.json``, ``model.safetensors.index.json`` and
    the tokenizer files, and possibly the weights. Skip the calling test when
    ``GLM4_0414_CHECKPOINT_PATH`` is not set and the hub cannot be reached.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        return _checked(path)

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            MODEL_ID, revision=HF_REVISION, allow_patterns=_METADATA_PATTERNS
        )
    except Exception as e:  # noqa: BLE001 - any hub failure skips the test
        pytest.skip(
            f"Cannot download the {MODEL_ID} metadata ({type(e).__name__}: {e}). "
            f"Set {CHECKPOINT_ENV} to a local copy of the checkpoint."
        )


def weights_path() -> str:
    """Return a local checkpoint directory with the weights.

    The weights are about 65 GB, so the unit tests do not download them. Skip
    the calling test when ``GLM4_0414_CHECKPOINT_PATH`` is not set.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        pytest.skip(
            f"This test reads the {MODEL_ID} weights. Set {CHECKPOINT_ENV} "
            f"to a local copy of revision {HF_REVISION}."
        )
    return _checked(path)


def resolve_checkpoint() -> str:
    """Return a local checkpoint directory with the weights, downloading if needed.

    The integration test uses this: ``GLM4_0414_CHECKPOINT_PATH`` if set,
    otherwise the pinned revision from the HuggingFace hub.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        return _checked(path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )


def read_weight_map(checkpoint: str) -> dict[str, str]:
    """Return the tensor-name to file-name map of a sharded checkpoint."""
    index_path = Path(checkpoint) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    from safetensors import safe_open

    weight_map = {}
    for file in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = file.name
    return weight_map


def read_tensors(checkpoint: str, names: list[str]) -> dict[str, torch.Tensor]:
    """Read the named tensors from the checkpoint. Do not read other tensors."""
    from safetensors import safe_open

    weight_map = read_weight_map(checkpoint)
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    tensors = {}
    for file, file_names in by_file.items():
        with safe_open(str(Path(checkpoint) / file), framework="pt", device="cpu") as f:
            for name in file_names:
                tensors[name] = f.get_tensor(name)
    return tensors


# =============================================================================
# Independent references and weight-loader drivers
# =============================================================================


def reference_rope_permutation(head_dim: int, rotary_dim: int) -> list[int]:
    """Return the expected column order of a permuted Q/K head.

    This is an independent statement of the layout: the even rotary columns,
    then the first half of the pass-through columns, then the odd rotary
    columns, then the second half of the pass-through columns.
    """
    num_pass_half = (head_dim - rotary_dim) // 2
    return (
        list(range(0, rotary_dim, 2))
        + list(range(rotary_dim, rotary_dim + num_pass_half))
        + list(range(1, rotary_dim, 2))
        + list(range(rotary_dim + num_pass_half, head_dim))
    )


class FakeSafeSlice:
    """Stand-in for a safetensors ``PySafeSlice`` backed by a ``torch.Tensor``.

    The weight-loader transforms take checkpoint slices. This class lets a test
    drive them without a checkpoint file. It supports slicing and the shape and
    dtype queries only.
    """

    __slots__ = ("_t",)

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor

    def __getitem__(self, idx):
        return self._t[idx]

    def get_shape(self):
        return list(self._t.shape)

    def get_dtype(self):
        return str(self._t.dtype).replace("torch.", "")


def load_weights_from_slices(
    module: nn.Module,
    slice_map: dict[str, FakeSafeSlice],
    mappings: dict,
    rank: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Run the weight loader of each parameter of ``module`` on synthetic slices.

    Returns a rank-local state dict for ``module.load_state_dict(..., assign=True)``.
    """
    from vllm_neuron.utils.weight_loader import get_weight_loader

    state_dict: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        checkpoint_keys = mappings.get(name, name)
        if not isinstance(checkpoint_keys, list):
            checkpoint_keys = [checkpoint_keys]
        slices = [slice_map[k] for k in checkpoint_keys]
        tensor = get_weight_loader(param).load(slices, rank)
        state_dict[name] = tensor.to(device)
    return state_dict


def layer_mappings(prefix: str, layer_idx: int = LAYER_IDX) -> dict:
    """Return the GLM-4 mappings of one layer, relative to ``prefix``.

    ``prefix`` is ``"self_attn."``, ``"mlp."`` or ``""`` (the full decoder
    layer). The result also maps the norm weights of the layer.
    """
    model_cls = src_module("model").Glm4ForCausalLM
    layer_prefix = f"model.layers.{layer_idx}."
    full = model_cls.checkpoint_mappings(layer_idx + 1)
    for norm in (
        "input_layernorm",
        "post_self_attn_layernorm",
        "post_attention_layernorm",
        "post_mlp_layernorm",
    ):
        full[f"{layer_prefix}{norm}.weight"] = f"{layer_prefix}{norm}.weight"
    start = layer_prefix + prefix
    return {k.removeprefix(start): v for k, v in full.items() if k.startswith(start)}


# =============================================================================
# MPExecutor model_load for the tiny decoder-layer test
# =============================================================================


def _allocate_kv_cache(attn: nn.Module, num_blocks: int, block_size: int, device):
    shape = (num_blocks, attn.num_key_value_heads_per_rank, block_size, attn.head_dim)
    attn.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    attn.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)


class DecoderLayerSlice(nn.Module):
    """Give each rank its sequence-parallel chunk of a prefill input.

    The GLM-4 decoder layer expects the SP layout on prefill. The test sends
    the full sequence to each rank, and this wrapper keeps only the chunk of
    the rank. On decode, the wrapper does not change the input.
    """

    def __init__(self, layer: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.layer = layer
        self.rank = rank
        self.world_size = world_size

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        meta = attn_metadata[f"layers.{LAYER_IDX}.self_attn"]
        if meta["max_query_len"] > meta["decode_token_threshold"]:
            per_rank = hidden_states.shape[0] // self.world_size
            start = self.rank * per_rank
            hidden_states = hidden_states[start : start + per_rank]
        return self.layer(hidden_states, positions, position_embeddings, attn_metadata)


def load_tiny_decoder_layer(
    hf_config_dict: dict,
    hf_layer_state: dict,
    num_blocks: int,
    block_size: int,
    device: str,
) -> nn.Module:
    """Build a tiny ``Glm4DecoderLayer`` from an HF layer state dict.

    ``hf_layer_state`` uses the HF names of one decoder layer (for example
    ``self_attn.q_proj.weight``). The real GLM-4 loaders shard and permute it.
    This is a top-level function so that ``functools.partial`` of it can be
    sent to a forked ``MPExecutor`` worker.
    """
    config_cls = src_module("config").Glm4Config
    layer_cls = src_module("model").Glm4DecoderLayer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    config = config_cls.from_configs(hf_config_dict, None)
    layer = layer_cls(config, LAYER_IDX)

    slice_map = {
        f"model.layers.{LAYER_IDX}.{name}": FakeSafeSlice(tensor)
        for name, tensor in hf_layer_state.items()
    }
    state_dict = load_weights_from_slices(
        layer, slice_map, layer_mappings(""), rank, device
    )
    layer.load_state_dict(state_dict, strict=True, assign=True)
    _allocate_kv_cache(layer.self_attn, num_blocks, block_size, device)
    return DecoderLayerSlice(layer, rank, world_size)
