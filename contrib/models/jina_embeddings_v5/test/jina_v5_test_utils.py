# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Jina Embeddings v5 text nano contrib tests.

This file travels with the model: it is the port's own test helper, vendored
into the contrib folder so that nothing outside
``contrib/models/jina_embeddings_v5/`` is needed. It sits at the ``test/``
root because ``test/unit``, ``test/module`` and ``test/integration`` all import
it (``test/conftest.py`` puts this directory on ``sys.path``). The
implementation under ``src/`` is reached through ``_local_src``.

The references are independent of the port:

- The HF reference is ``EuroBertModel`` from the ``modeling_eurobert.py`` file
  of the checkpoint (remote code). The tests load it with eager attention.
- The HF reference runs each sequence without padding. The port runs the same
  sequence in a padded bucket. The tests compare only the real rows.
- The pooled reference is the ``custom_st`` pooling of the checkpoint: the
  hidden state of the last real token, then L2 normalization.

The worker loaders are top-level functions. Use them with
``functools.partial``, so that ``MPExecutor`` can pickle them.
"""

import hashlib
import json
import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from vllm_neuron import envs

from _local_src import src_module

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano"
HF_REVISION = "8a7f00aac812071b69403df470f1038ec85f8925"
CHECKPOINT_ENV = "JINA_V5_CHECKPOINT"

BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
# Draw random token IDs below the special-token range.
MAX_RANDOM_TOKEN_ID = 128000

LAYER_IDX = 0
BLOCK_SIZE = 16
# The serving bucket: --max-model-len 1024 and one 1024-token prefill bucket.
DEVICE_BUCKET = 1024


def _without_adapters(path: str) -> str:
    """Return ``path``, or a view of it without the ``adapters/`` directory.

    The implementation loads the base encoder weights only, and it rejects a
    local checkpoint that has the task LoRA adapters. A full snapshot of the
    repository has them, so the tests use a directory of symbolic links to
    every other entry of the snapshot.
    """
    if not os.path.isdir(os.path.join(path, "adapters")):
        return path
    source = os.path.realpath(path)
    digest = hashlib.sha1(source.encode()).hexdigest()[:12]
    view = os.path.join(tempfile.gettempdir(), f"jina_v5_base_{digest}")
    os.makedirs(view, exist_ok=True)
    for name in os.listdir(source):
        link = os.path.join(view, name)
        if name != "adapters" and not os.path.lexists(link):
            os.symlink(os.path.join(source, name), link)
    return view


def get_checkpoint() -> str:
    """Return the local directory of the base encoder checkpoint.

    Use the directory in ``JINA_V5_CHECKPOINT`` when it is set. Otherwise
    download the pinned revision of ``MODEL_ID`` without the ``adapters/``
    directory into a directory of its own under ``HF_HOME``. Either way, the
    returned directory has no ``adapters/`` directory (see
    :func:`_without_adapters`).
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise FileNotFoundError(
                f"{CHECKPOINT_ENV}={path} has no config.json. Set "
                f"{CHECKPOINT_ENV} to the {MODEL_ID} checkpoint directory."
            )
        return _without_adapters(path)
    from huggingface_hub import constants, snapshot_download

    local_dir = os.path.join(
        constants.HF_HOME, "contrib_jina_embeddings_v5", HF_REVISION
    )
    path = snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        ignore_patterns=["adapters/*"],
        local_dir=local_dir,
    )
    return _without_adapters(path)


def get_device() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


def load_config_dict(checkpoint: str) -> dict:
    """Return the raw ``config.json`` dict. A dict is picklable."""
    with open(os.path.join(checkpoint, "config.json")) as f:
        return json.load(f)


# =============================================================================
# HF reference (independent of the port)
# =============================================================================


def load_hf_encoder(checkpoint: str, dtype: torch.dtype) -> nn.Module:
    """Load the checkpoint ``EuroBertModel`` with eager attention.

    Do not use ``AutoModel``. It gives the PEFT wrapper, which needs the task
    adapters. The base encoder weights have no key prefix, so they load
    directly into ``EuroBertModel``.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    hf_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    encoder_cls = get_class_from_dynamic_module(
        "modeling_eurobert.EuroBertModel", checkpoint
    )
    model = encoder_cls.from_pretrained(
        checkpoint,
        config=hf_config,
        dtype=dtype,
        attn_implementation="eager",
    )
    return model.eval()


@torch.no_grad()
def hf_last_hidden(model: nn.Module, token_ids: list[int]) -> torch.Tensor:
    """Return the ``[L, H]`` final hidden states of one unpadded sequence."""
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    return out.last_hidden_state[0].detach()


@torch.no_grad()
def hf_layer_attention(
    model: nn.Module, hidden: torch.Tensor, layer_idx: int = LAYER_IDX
) -> torch.Tensor:
    """Return the ``[L, H]`` output of one HF attention module.

    ``hidden`` is ``[L, H]`` with no padding. A ``None`` mask gives full
    bidirectional attention over the ``L`` tokens.
    """
    dtype = next(model.parameters()).dtype
    h = hidden.to(dtype).unsqueeze(0)
    pos = torch.arange(h.shape[1], dtype=torch.long).unsqueeze(0)
    position_embeddings = model.rotary_emb(h, pos)
    out, _ = model.layers[layer_idx].self_attn(
        h, position_embeddings=position_embeddings, attention_mask=None
    )
    return out[0].detach()


@torch.no_grad()
def hf_layer_mlp(
    model: nn.Module, hidden: torch.Tensor, layer_idx: int = LAYER_IDX
) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    return model.layers[layer_idx].mlp(hidden.to(dtype)).detach()


def pool_last_normalize(hidden: torch.Tensor, real_len: int) -> torch.Tensor:
    """Apply the checkpoint ``custom_st`` pooling to ``[T, H]`` hidden states."""
    return F.normalize(hidden[real_len - 1].float(), p=2, dim=-1)


# =============================================================================
# Inputs
# =============================================================================


def random_token_ids(seed: int, length: int) -> list[int]:
    """Return ``length`` seeded token IDs that start with the BOS token."""
    generator = torch.Generator().manual_seed(seed)
    body = torch.randint(0, MAX_RANDOM_TOKEN_ID, (length - 1,), generator=generator)
    return [BOS_TOKEN_ID] + body.tolist()


def padded_positions(real_len: int, bucket: int) -> torch.Tensor:
    """Return the runner positions of one sequence in a padded bucket.

    The runner gives each pad token the last real position.
    """
    positions = torch.arange(bucket, dtype=torch.int32)
    positions[real_len:] = real_len - 1
    return positions


def padded_token_ids(token_ids: list[int], bucket: int) -> torch.Tensor:
    ids = torch.full((bucket,), PAD_TOKEN_ID, dtype=torch.int32)
    ids[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.int32)
    return ids


def build_attn_metadata(
    num_layers: int, num_tokens: int, max_query_len: int, num_seqs: int = 1
) -> dict:
    """Return the per-layer attention metadata that the runner gives.

    ``max_query_len`` larger than 1 selects the prefill path. A value of 1
    selects the decode-classified path.
    """
    blocks_per_seq = max((num_tokens // num_seqs + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)
    layer = {
        "max_query_len": max_query_len,
        "decode_token_threshold": 1,
        "slot_mapping": torch.arange(num_tokens, dtype=torch.int64),
        "block_size": BLOCK_SIZE,
        "block_table_tensor": torch.arange(
            num_seqs * blocks_per_seq, dtype=torch.int64
        ).view(num_seqs, blocks_per_seq),
        "cached_seq_len": None,
        "kv_segment_size": None,
        "max_blocks_per_seq": blocks_per_seq,
    }
    return {f"layers.{i}.self_attn": dict(layer) for i in range(num_layers)}


# =============================================================================
# Port modules under test (run inside MPExecutor workers)
# =============================================================================


class AttentionUnderTest(nn.Module):
    """Run the port attention on the SP slice of this rank.

    The wrapper computes RoPE and the key bounds with the port code, as the
    encoder does.
    """

    def __init__(self, attn: nn.Module, rotary: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.attn = attn
        self.rotary = rotary
        self.rank = rank
        self.world_size = world_size

    def forward(self, hidden_states, positions, attn_metadata):
        build_key_bounds = src_module("model").build_key_bounds

        per_rank = hidden_states.shape[0] // self.world_size
        start = self.rank * per_rank
        local = hidden_states[start : start + per_rank]
        position_embeddings = self.rotary(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        key_bounds = build_key_bounds(positions, self.attn.num_attention_heads_per_rank)
        return self.attn(
            local, positions, position_embeddings, attn_metadata, key_bounds=key_bounds
        )


class MLPUnderTest(nn.Module):
    """Run the port prefill MLP on the SP slice of this rank."""

    def __init__(self, mlp: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.mlp = mlp
        self.rank = rank
        self.world_size = world_size

    def forward(self, hidden_states):
        per_rank = hidden_states.shape[0] // self.world_size
        start = self.rank * per_rank
        return self.mlp(hidden_states[start : start + per_rank], is_prefill=True)


class EncoderUnderTest(nn.Module):
    """Run the full port encoder (all layers) with the runner inputs."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, positions, attn_metadata):
        return self.encoder(input_ids, positions.to(torch.int32), attn_metadata)


def _port_config(config_dict: dict):
    from vllm_neuron.model.neuron_config import NeuronConfig

    jina_v5_config_from_hf = src_module("config").jina_v5_config_from_hf

    return jina_v5_config_from_hf(config_dict, NeuronConfig(), torch.bfloat16)


def _load_module(module, checkpoint, mappings, device, cpu_mode):
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    rank, world_size = dist.get_rank(), dist.get_world_size()
    ckpt = SafetensorsCheckpoint(checkpoint, None)
    if cpu_mode:
        result = ckpt.load_sharded(rank, world_size, module, mappings, device)
    else:
        result = ckpt.load_sharded_pipelined(rank, world_size, module, mappings, device)
    module.load_state_dict(result.state_dict, strict=False, assign=True)


def check_kernel_dispatch(name: str, eligible: bool) -> None:
    """Log one kernel dispatch decision, and fail if the kernel is refused.

    This records the dispatch condition only. It is not evidence that the
    kernel executed.
    """
    print(f"[jina_v5_kernels] {name} eligible={eligible}", flush=True)
    if not eligible:
        raise RuntimeError(
            f"The {name} NKI kernel gate refused the test geometry. The port "
            "then uses the torch fallback. Examine the gate conditions."
        )


def _finish(module: nn.Module, device: str, cpu_mode: bool) -> nn.Module:
    module = module.to(device).eval()
    if cpu_mode:
        return module
    return torch.compile(
        module, fullgraph=True, backend=envs.get_compile_backend_name()
    )


def load_attention(checkpoint: str, config_dict: dict, device: str) -> nn.Module:
    """Build and load the layer-0 port attention (``MPExecutor`` model_load)."""
    model = src_module("model")

    cpu_mode = device == "cpu"
    config = _port_config(config_dict)
    attn = model.JinaEmbeddingsV5Attention(config, layer_idx=LAYER_IDX)
    prefix = f"layers.{LAYER_IDX}.self_attn"
    mappings = {
        "qkv_proj_weight": [
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
        ],
        "o_proj_weight": f"{prefix}.o_proj.weight",
    }
    _load_module(attn, checkpoint, mappings, device, cpu_mode)
    if not cpu_mode:
        from vllm_neuron.functional.attention.attention_cte import (
            _can_use_flash_attention_kernel,
        )

        v = torch.empty(
            attn.num_attention_heads_per_rank,
            DEVICE_BUCKET,
            attn.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        check_kernel_dispatch(
            f"attention_cte[{tuple(v.shape)}]", _can_use_flash_attention_kernel(v)
        )
    wrapper = AttentionUnderTest(
        attn,
        model.JinaEmbeddingsV5RotaryEmbedding(config),
        dist.get_rank(),
        dist.get_world_size(),
    )
    return _finish(wrapper, device, cpu_mode)


def load_mlp(checkpoint: str, config_dict: dict, device: str) -> nn.Module:
    """Build and load the layer-0 port MLP (``MPExecutor`` model_load)."""
    cpu_mode = device == "cpu"
    config = _port_config(config_dict)
    mlp = src_module("model").JinaEmbeddingsV5MLP(config)
    prefix = f"layers.{LAYER_IDX}.mlp"
    mappings = {
        "gate_proj_weight": f"{prefix}.gate_proj.weight",
        "up_proj_weight": f"{prefix}.up_proj.weight",
        "down_proj_weight": f"{prefix}.down_proj.weight",
    }
    _load_module(mlp, checkpoint, mappings, device, cpu_mode)
    if not cpu_mode:
        from vllm_neuron.functional.mlp import _can_use_kernel

        hidden = torch.empty(
            DEVICE_BUCKET, config.hidden_size, dtype=torch.bfloat16, device=device
        )
        check_kernel_dispatch(
            f"mlp_prefill[{tuple(hidden.shape)}x{tuple(mlp.gate_proj_weight.shape)}]",
            _can_use_kernel(hidden, mlp.gate_proj_weight),
        )
    wrapper = MLPUnderTest(mlp, dist.get_rank(), dist.get_world_size())
    return _finish(wrapper, device, cpu_mode)


def load_encoder(checkpoint: str, config_dict: dict, device: str) -> nn.Module:
    """Build and load the full port encoder (``MPExecutor`` model_load).

    The load uses the same loader as the serving path. It fails if a
    checkpoint tensor is not used.
    """
    model = src_module("model")
    cpu_mode = device == "cpu"
    config = _port_config(config_dict)
    with torch.device("meta"):
        encoder = model.JinaEmbeddingsV5Encoder(config)
    result = model.load_encoder_weights(
        encoder,
        checkpoint,
        device,
        None,
        prefix="",
        rank=dist.get_rank(),
        pipelined=not cpu_mode,
    )
    if result.unexpected_keys:
        raise RuntimeError(
            f"The port did not use {len(result.unexpected_keys)} checkpoint "
            f"tensors, for example {sorted(result.unexpected_keys)[:3]}."
        )
    meta = [n for n, p in encoder.named_parameters() if p.device.type == "meta"]
    if meta:
        raise RuntimeError(f"Parameters stay on the meta device: {meta[:3]}.")
    return _finish(EncoderUnderTest(encoder), device, cpu_mode)
