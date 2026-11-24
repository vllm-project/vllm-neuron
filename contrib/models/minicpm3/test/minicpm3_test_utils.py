# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the MiniCPM3 (openbmb/MiniCPM3-4B) tests.

This file travels with the model, so nothing outside
``contrib/models/minicpm3/`` is needed. It sits at the ``test/`` root because
``test/unit`` and ``test/integration`` both import it (``test/conftest.py``
puts this directory on ``sys.path``).

The independent reference is the MiniCPM3 implementation built into
``transformers`` (``transformers.MiniCPM3ForCausalLM``). The helpers never use
``trust_remote_code``, so the remote ``modeling_minicpm.py`` of the checkpoint
(written for transformers 4.41) takes no part in a reference.

The checkpoint comes from one of two sources:

1. ``MINICPM3_CHECKPOINT_PATH``, a local directory that holds the checkpoint.
2. Without it, ``huggingface_hub.snapshot_download`` at the pinned revision
   (about 8 GB, one ``pytorch_model.bin``).

The unit tests never download the checkpoint. They build small random-weight
checkpoints with the real MiniCPM3-4B geometry (``MINICPM3_4B_CONFIG``) and
save them as ``pytorch_model.bin``, the format of the published checkpoint.
Only the tests that need the real weights call :func:`local_checkpoint_or_skip`
and skip when ``MINICPM3_CHECKPOINT_PATH`` is not set.

Every helper that an ``MPExecutor`` worker calls is a top-level function, so
that ``functools.partial`` can pickle it.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from _local_src import src_module

MODEL_ID = "openbmb/MiniCPM3-4B"
# Every test pins this HuggingFace revision of MODEL_ID.
HF_REVISION = "d6b14ddaefdb11c624dd75c3c779549bc90b08cb"
ARCHITECTURE = "MiniCPM3ForCausalLM"
CHECKPOINT_ENV = "MINICPM3_CHECKPOINT_PATH"

# The topology of the integration test and of the serving configuration in
# README.md: TP 8 on one trn2.48xlarge. 40 heads give 5 heads on each rank.
WORLD_SIZE = 8
# Finite wait for one MPExecutor collect.
COLLECT_TIMEOUT_S = 1800

# ``config.json`` of MODEL_ID at HF_REVISION, without ``auto_map`` (the remote
# code) and ``_name_or_path``.
MINICPM3_4B_CONFIG = {
    "architectures": ["MiniCPM3ForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": [2, 73440],
    "hidden_act": "silu",
    "initializer_range": 0.1,
    "hidden_size": 2560,
    "num_hidden_layers": 62,
    "intermediate_size": 6400,
    "max_position_embeddings": 32768,
    "model_type": "minicpm3",
    "num_attention_heads": 40,
    "num_key_value_heads": 40,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 32,
    "q_lora_rank": 768,
    "kv_lora_rank": 256,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "type": "longrope",
        "long_factor": [
            1.0591234137867171,
            1.1241891283591912,
            1.2596935748670968,
            1.5380380402321725,
            2.093982484148734,
            3.1446935121267696,
            4.937952647693647,
            7.524541999994549,
            10.475458000005451,
            13.062047352306353,
            14.85530648787323,
            15.906017515851266,
            16.461961959767827,
            16.740306425132907,
            16.87581087164081,
            16.940876586213285,
        ],
        "short_factor": [
            1.0591234137867171,
            1.1241891283591912,
            1.2596935748670968,
            1.5380380402321725,
            2.093982484148734,
            3.1446935121267696,
            4.937952647693647,
            7.524541999994549,
            10.475458000005451,
            13.062047352306353,
            14.85530648787323,
            15.906017515851266,
            16.461961959767827,
            16.740306425132907,
            16.87581087164081,
            16.940876586213285,
        ],
        "original_max_position_embeddings": 32768,
    },
    "torch_dtype": "bfloat16",
    "transformers_version": "4.41.0",
    "use_cache": True,
    "vocab_size": 73448,
    "scale_emb": 12,
    "dim_model_base": 256,
    "scale_depth": 1.4,
}

# Layer 0 is the layer that the component tests compare.
LAYER_IDX = 0


# =============================================================================
# Configs
# =============================================================================


def set_longrope_factor(config) -> None:
    """Set the optional longrope ``factor`` key to the value that HF derives.

    The checkpoint gives no ``factor``. The HF longrope init then uses
    max_position_embeddings / original_max_position_embeddings. The HF
    MiniCPM3Attention reads ``rope_parameters["factor"]`` for each RoPE type
    that is not "default" and raises KeyError without it. It uses the value
    only when ``mscale_all_dim`` is set, so the derived value does not change
    the HF result.
    """
    rope = config.rope_parameters
    if rope.get("rope_type") != "longrope" or rope.get("factor") is not None:
        return
    rope["factor"] = (
        config.max_position_embeddings / rope["original_max_position_embeddings"]
    )


def hf_config(overrides: dict | None = None, attn_implementation: str = "eager"):
    """Return the HF MiniCPM3 config of MiniCPM3-4B with ``overrides`` applied."""
    from transformers import MiniCPM3Config as HFMiniCPM3Config

    config = HFMiniCPM3Config.from_dict({**MINICPM3_4B_CONFIG, **(overrides or {})})
    config._attn_implementation = attn_implementation
    set_longrope_factor(config)
    return config


def load_hf_config(checkpoint: str, attn_implementation: str = "eager"):
    """Load the built-in transformers MiniCPM3 config of a checkpoint directory."""
    from transformers import MiniCPM3Config as HFMiniCPM3Config

    config = HFMiniCPM3Config.from_pretrained(checkpoint)
    config._attn_implementation = attn_implementation
    set_longrope_factor(config)
    return config


def neuron_model_config(hf):
    """Build this folder's MiniCPM3Config from an HF config."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    return src_module("config").MiniCPM3Config.from_configs(hf, NeuronConfig())


def native_hf_model_loader(model_checkpoint: str, dtype: torch.dtype, config=None):
    """Load the reference with the ``MiniCPM3ForCausalLM`` built into transformers.

    The plugin's default golden loader sets ``trust_remote_code=True``. The
    checkpoint's ``config.json`` maps to its remote ``modeling_minicpm.py``,
    which targets transformers 4.41 and does not run on the transformers 5.17
    of vLLM Neuron 0.29.0.1.2.0. The built-in class implements the same
    architecture and reads the same checkpoint.
    """
    from transformers import MiniCPM3ForCausalLM as HFModel

    if config is None:
        config = load_hf_config(model_checkpoint, attn_implementation="sdpa")
    set_longrope_factor(config)
    return HFModel.from_pretrained(
        model_checkpoint, config=config, dtype=dtype, attn_implementation="sdpa"
    ).eval()


# =============================================================================
# Checkpoints
# =============================================================================


def checkpoint_file(checkpoint: str) -> str:
    """Return the single PyTorch weight file of a MiniCPM3 checkpoint."""
    path = os.path.join(checkpoint, "pytorch_model.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File {path} does not exist. A MiniCPM3-4B checkpoint has one "
            f"pytorch_model.bin file."
        )
    return path


def load_checkpoint_state(checkpoint: str) -> dict[str, torch.Tensor]:
    """Memory-map the full checkpoint state dict (the data stays on disk)."""
    return torch.load(
        checkpoint_file(checkpoint), map_location="cpu", mmap=True, weights_only=True
    )


def load_prefixed_state(checkpoint: str, prefix: str) -> dict[str, torch.Tensor]:
    """Return a copy of the checkpoint tensors under ``prefix``, prefix removed."""
    state = load_checkpoint_state(checkpoint)
    return {
        k[len(prefix) :]: v.clone() for k, v in state.items() if k.startswith(prefix)
    }


def write_random_checkpoint(path: str, config, seed: int = 0) -> str:
    """Save a random-weight BF16 MiniCPM3 checkpoint as ``pytorch_model.bin``.

    The file has the key set and the format of the published checkpoint: one
    zip-format ``torch.save`` file, BF16 tensors, and no ``lm_head.weight``
    (the embeddings are tied). The RMSNorm weights get random values around 1,
    so a wrong norm mapping shows in the outputs.

    Args:
        path: Directory to write. It must exist.
        config: HF MiniCPM3 config (geometry and layer count).
        seed: Seed of the random weights.

    Returns:
        ``path``.
    """
    from transformers import MiniCPM3ForCausalLM as HFModel

    torch.manual_seed(seed)
    model = HFModel(config).to(torch.bfloat16)
    state = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == "lm_head.weight" and config.tie_word_embeddings:
                continue
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
            state[name] = param.detach().clone().contiguous()
    torch.save(state, os.path.join(path, "pytorch_model.bin"))
    config.save_pretrained(path)
    return path


def _checkpoint_from_env() -> str | None:
    """Return the checkpoint in ``MINICPM3_CHECKPOINT_PATH``, or None if unset."""
    value = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not value:
        return None
    root = Path(value)
    missing = [
        name
        for name in ("config.json", "pytorch_model.bin", "tokenizer_config.json")
        if not (root / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{CHECKPOINT_ENV}={value!r} holds no complete {MODEL_ID} checkpoint "
            f"(missing {missing}). Set it to a snapshot of revision {HF_REVISION}."
        )
    return str(root)


def local_checkpoint_or_skip() -> str:
    """Return the local checkpoint, or skip the test if none is configured."""
    import pytest

    path = _checkpoint_from_env()
    if path is None:
        pytest.skip(
            f"Set {CHECKPOINT_ENV} to a local snapshot of {MODEL_ID} at revision "
            f"{HF_REVISION} (about 8 GB) to run this test."
        )
    return path


def resolve_checkpoint() -> str:
    """Return a local directory that holds the pinned checkpoint.

    Use ``MINICPM3_CHECKPOINT_PATH`` if it is set. Otherwise download the
    pinned revision from the HuggingFace hub (about 8 GB). The download skips
    the remote modeling code, which the tests do not use.
    """
    path = _checkpoint_from_env()
    if path is not None:
        return path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.json", "pytorch_model.bin", "tokenizer.model"],
    )


# =============================================================================
# HF references (built-in transformers MiniCPM3)
# =============================================================================


def hf_attention_reference(
    checkpoint: str, seq_len: int, seed: int
) -> dict[str, torch.Tensor]:
    """Run the HF MiniCPM3 attention of layer LAYER_IDX in FP32 and in BF16.

    The input is one random sequence of ``seq_len`` tokens. The caller uses rows
    [0, seq_len - 1) as the prefill reference and row seq_len - 1 as the decode
    reference.
    """
    from transformers.models.minicpm3.modeling_minicpm3 import (
        MiniCPM3Attention as HFAttention,
        MiniCPM3RotaryEmbedding as HFRotary,
    )

    config = load_hf_config(checkpoint)
    attn = HFAttention(config, layer_idx=LAYER_IDX)
    state = load_prefixed_state(checkpoint, f"model.layers.{LAYER_IDX}.self_attn.")
    attn.load_state_dict(state, strict=True)
    attn = attn.to(torch.bfloat16).eval()
    attn_fp32 = copy.deepcopy(attn).float()
    rotary = HFRotary(config)

    torch.manual_seed(seed)
    h = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    pos = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    mask = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(torch.float32).min), diagonal=1
    )[None, None]
    with torch.no_grad():
        cos32, sin32 = rotary(h.float(), pos)
        out32, _ = attn_fp32(h.float(), (cos32, sin32), attention_mask=mask)
        cos16, sin16 = rotary(h, pos)
        out16, _ = attn(h, (cos16, sin16), attention_mask=mask.to(torch.bfloat16))
    return {"input": h[0], "fp32": out32[0], "bf16": out16[0]}


def hf_mlp_reference(
    checkpoint: str, num_tokens: int, seed: int
) -> dict[str, torch.Tensor]:
    """Run the HF MiniCPM3 MLP of layer LAYER_IDX in FP32 and in BF16."""
    from transformers.models.minicpm3.modeling_minicpm3 import MiniCPM3MLP as HFMLP

    config = load_hf_config(checkpoint)
    mlp = HFMLP(config)
    mlp.load_state_dict(
        load_prefixed_state(checkpoint, f"model.layers.{LAYER_IDX}.mlp."), strict=True
    )
    mlp = mlp.to(torch.bfloat16).eval()
    mlp_fp32 = copy.deepcopy(mlp).float()

    torch.manual_seed(seed)
    h = torch.randn(num_tokens, config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        out32 = mlp_fp32(h.float())
        out16 = mlp(h)
    return {"input": h, "fp32": out32, "bf16": out16}


def hf_full_model_logits(
    checkpoint: str, input_ids: list[int]
) -> dict[str, torch.Tensor]:
    """Run the full HF MiniCPM3 model in FP32 and in BF16. Return two logit rows.

    The HF flow is the same as the flow under test. First, HF runs a prefill of
    len - 1 tokens. Row 0 is the logits at position len - 2 (the prefill
    reference). Then HF runs one decode step for the last token. This step reads
    the HF KV cache of the prefill. Row 1 is the logits of this step at position
    len - 1 (the decode reference).
    """
    from transformers import DynamicCache

    config = load_hf_config(checkpoint, attn_implementation="sdpa")
    n = len(input_ids) - 1
    ids = torch.tensor([input_ids], dtype=torch.long)
    result = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        model = native_hf_model_loader(checkpoint, dtype, config=config)
        with torch.no_grad():
            cache = DynamicCache(config=config)
            prefill = model(
                input_ids=ids[:, :n],
                position_ids=torch.arange(n, dtype=torch.long).unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
            )
            decode = model(
                input_ids=ids[:, n:],
                position_ids=torch.tensor([[n]], dtype=torch.long),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
        result[name] = torch.cat(
            [prefill.logits[0, -1:, :], decode.logits[0, -1:, :]], dim=0
        ).float()
        del model
    return result


# =============================================================================
# Neuron harnesses (run inside MPExecutor workers)
# =============================================================================


def _cache_shape(config, num_blocks: int, block_size: int) -> tuple[int, ...]:
    layout = src_module("model").minicpm3_head_layout(config)
    num_heads = config.num_attention_heads // dist.get_world_size()
    return (num_blocks, num_heads, block_size, layout.d_head)


class AttentionHarness(nn.Module):
    """Run one MiniCPM3Attention on the SP slice of this rank (prefill) or on all
    tokens (decode). The KV cache stays in the module between calls."""

    def __init__(self, attn: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.attn = attn
        self.rank = rank
        self.world_size = world_size

    def forward(self, hidden_states, positions, cos, sin, attn_metadata, is_prefill):
        if is_prefill:
            per_rank = hidden_states.shape[0] // self.world_size
            start = self.rank * per_rank
            hidden_states = hidden_states[start : start + per_rank]
        return self.attn(hidden_states, positions, (cos, sin), attn_metadata)


def attention_model_load(
    checkpoint: str, num_blocks: int, block_size: int, device: str
) -> nn.Module:
    """Load layer LAYER_IDX attention of the checkpoint on this rank.

    The load goes through this folder's ``SafetensorsCheckpoint``, which reads
    the ``pytorch_model.bin`` of the checkpoint.
    """
    model = src_module("model")
    SafetensorsCheckpoint = src_module("checkpoints").SafetensorsCheckpoint

    config = neuron_model_config(load_hf_config(checkpoint))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    attn = model.MiniCPM3Attention(config, layer_idx=LAYER_IDX)
    prefix = f"model.layers.{LAYER_IDX}.self_attn"
    mappings = {
        "latent_proj_weight": [
            f"{prefix}.q_a_proj.weight",
            f"{prefix}.kv_a_proj_with_mqa.weight",
        ],
        "qkv_proj_weight": [f"{prefix}.q_b_proj.weight", f"{prefix}.kv_b_proj.weight"],
        "o_proj_weight": f"{prefix}.o_proj.weight",
        "q_a_layernorm.weight": f"{prefix}.q_a_layernorm.weight",
        "kv_a_layernorm.weight": f"{prefix}.kv_a_layernorm.weight",
    }
    shards = SafetensorsCheckpoint(checkpoint, None).load_sharded(
        rank, world_size, attn, mappings, "cpu"
    )
    attn.load_state_dict(shards.state_dict, strict=True, assign=True)

    shape = _cache_shape(config, num_blocks, block_size)
    attn.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    attn.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    return AttentionHarness(attn, rank, world_size)


def attention_inputs(
    checkpoint: str,
    hidden: torch.Tensor,
    start: int,
    num_tokens: int,
    num_blocks: int,
    block_size: int,
    max_query_len: int,
    kv_segment_size: int | None = None,
):
    """Build (hidden, positions, cos, sin, attn_metadata) for tokens [start, start+n).

    With ``kv_segment_size`` set, the metadata selects the segmented-prefill
    attention, as the vLLM Neuron model runner sets it for each prefill chunk
    when ``kv_segment_size_buckets`` is configured: ``cached_seq_len`` is a
    ``[1, 1]`` int32 tensor that holds ``start``, the number of tokens that the
    earlier chunks wrote to the KV cache.
    """
    model = src_module("model")

    config = neuron_model_config(load_hf_config(checkpoint))
    rotary = model.MiniCPM3RotaryEmbedding(config, model.minicpm3_head_layout(config))
    positions = torch.arange(start, start + num_tokens, dtype=torch.int64)
    cos, sin = rotary(positions, device=torch.device("cpu"), dtype=torch.bfloat16)
    metadata = {
        f"layers.{LAYER_IDX}.self_attn": {
            "max_query_len": max_query_len,
            "decode_token_threshold": 1,
            "slot_mapping": positions.clone(),
            "block_size": block_size,
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32)[None],
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }
    if kv_segment_size is not None:
        metadata[f"layers.{LAYER_IDX}.self_attn"].update(
            cached_seq_len=torch.tensor([[start]], dtype=torch.int32),
            kv_segment_size=kv_segment_size,
        )
    return hidden[start : start + num_tokens], positions, cos, sin, metadata


class MLPHarness(nn.Module):
    """Run one MiniCPM3MLP on the SP slice of this rank (prefill) or on all tokens."""

    def __init__(self, mlp: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.mlp = mlp
        self.rank = rank
        self.world_size = world_size

    def forward(self, hidden_states, is_prefill):
        if is_prefill:
            per_rank = hidden_states.shape[0] // self.world_size
            start = self.rank * per_rank
            hidden_states = hidden_states[start : start + per_rank]
        return self.mlp(hidden_states, is_prefill=is_prefill)


def mlp_model_load(checkpoint: str, device: str) -> nn.Module:
    """Load layer LAYER_IDX MLP of the checkpoint on this rank."""
    model = src_module("model")
    SafetensorsCheckpoint = src_module("checkpoints").SafetensorsCheckpoint

    config = neuron_model_config(load_hf_config(checkpoint))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    mlp = model.MiniCPM3MLP(config)
    prefix = f"model.layers.{LAYER_IDX}.mlp"
    mappings = {
        "gate_proj_weight": f"{prefix}.gate_proj.weight",
        "up_proj_weight": f"{prefix}.up_proj.weight",
        "down_proj_weight": f"{prefix}.down_proj.weight",
    }
    shards = SafetensorsCheckpoint(checkpoint, None).load_sharded(
        rank, world_size, mlp, mappings, "cpu"
    )
    mlp.load_state_dict(shards.state_dict, strict=True, assign=True)
    return MLPHarness(mlp, rank, world_size)


class FullModelHarness(nn.Module):
    """Run the full MiniCPM3ForCausalLM and return the gathered logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, positions, attn_metadata, sampling_positions):
        return self.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=sampling_positions,
        )


def full_model_load(
    checkpoint: str, num_blocks: int, block_size: int, device: str
) -> nn.Module:
    """Load the full model with all checkpoint weights on this rank.

    The load goes through ``MiniCPM3ForCausalLM.load_weights``, the path that
    vLLM uses. Stop with an error if a parameter stays on the meta device, has
    the wrong dtype, or holds a value that is not finite.
    """
    from vllm_neuron.model.neuron_config import NeuronConfig

    MiniCPM3ForCausalLM = src_module("model").MiniCPM3ForCausalLM
    model = MiniCPM3ForCausalLM.from_configs(
        load_hf_config(checkpoint), NeuronConfig(on_device_sampling_config=None)
    )
    param_names = {name for name, _ in model.named_parameters()}
    mapped = set(model.checkpoint_mappings())
    if param_names != mapped:
        raise RuntimeError(
            f"The checkpoint mappings do not cover the model parameters. Unmapped: "
            f"{sorted(param_names - mapped)[:8]}. Unknown: {sorted(mapped - param_names)[:8]}."
        )
    model.load_weights(checkpoint, torch.device("cpu"), None)
    for name, param in model.named_parameters():
        if param.is_meta:
            raise RuntimeError(f"Parameter {name} is on the meta device after load.")
        if param.dtype != torch.bfloat16:
            raise RuntimeError(f"Parameter {name} has dtype {param.dtype}, not bf16.")
        if not torch.isfinite(param.float()).all():
            raise RuntimeError(f"Parameter {name} has a value that is not finite.")

    shape = _cache_shape(model.config, num_blocks, block_size)
    model.bind_kv_cache(
        {
            f"layers.{i}.self_attn": [
                torch.zeros(shape, dtype=torch.bfloat16, device=device),
                torch.zeros(shape, dtype=torch.bfloat16, device=device),
            ]
            for i in range(model.config.num_hidden_layers)
        }
    )
    return FullModelHarness(model)


def full_model_metadata(
    num_layers: int,
    start: int,
    num_tokens: int,
    num_blocks: int,
    block_size: int,
) -> dict:
    """Build the attention metadata of all layers for tokens [start, start + n)."""
    slots = torch.arange(start, start + num_tokens, dtype=torch.int64)
    table = torch.arange(num_blocks, dtype=torch.int32)[None]
    return {
        f"layers.{i}.self_attn": {
            "max_query_len": num_tokens,
            "decode_token_threshold": 1,
            "slot_mapping": slots,
            "block_size": block_size,
            "block_table_tensor": table,
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
        for i in range(num_layers)
    }


# =============================================================================
# Shared runners (the parent process calls them)
# =============================================================================


def run_attention_prefill_decode(
    executor,
    checkpoint: str,
    hidden: torch.Tensor,
    num_blocks: int,
    block_size: int,
    decode_steps: int = 1,
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """Run a prefill of hidden[:-decode_steps], then decode the last rows one by one.

    Each decode step reads the KV cache that the prefill and the earlier decode
    steps wrote. Return the per-rank prefill outputs, and for each rank the list
    of its ``decode_steps`` decode outputs.
    """
    prefill_len = hidden.shape[0] - decode_steps
    executor.dispatch(
        *attention_inputs(
            checkpoint, hidden, 0, prefill_len, num_blocks, block_size, prefill_len
        ),
        True,
    )
    prefill = executor.collect(timeout=COLLECT_TIMEOUT_S)
    decode: list[list[torch.Tensor]] = []
    for step in range(decode_steps):
        executor.dispatch(
            *attention_inputs(
                checkpoint, hidden, prefill_len + step, 1, num_blocks, block_size, 1
            ),
            False,
        )
        outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
        if not decode:
            decode = [[] for _ in outputs]
        for rank, out in enumerate(outputs):
            decode[rank].append(out)
    return prefill, decode


def run_attention_chunked_prefill(
    executor,
    checkpoint: str,
    hidden: torch.Tensor,
    chunk_tokens: int,
    num_blocks: int,
    block_size: int,
) -> list[list[torch.Tensor]]:
    """Prefill all rows of ``hidden`` in chunks of ``chunk_tokens`` tokens.

    Each chunk goes through the segmented-prefill attention with
    ``kv_segment_size=chunk_tokens``, and its queries attend to the KV cache
    that the earlier chunks wrote. Return, for each chunk, the per-rank outputs
    (the sequence-parallel slices of the chunk, in rank order).
    """
    num_tokens = hidden.shape[0]
    assert num_tokens % chunk_tokens == 0, (num_tokens, chunk_tokens)
    chunks = []
    for start in range(0, num_tokens, chunk_tokens):
        executor.dispatch(
            *attention_inputs(
                checkpoint,
                hidden,
                start,
                chunk_tokens,
                num_blocks,
                block_size,
                chunk_tokens,
                kv_segment_size=chunk_tokens,
            ),
            True,
        )
        chunks.append(executor.collect(timeout=COLLECT_TIMEOUT_S))
    return chunks


def run_mlp_prefill_decode(
    executor, hidden: torch.Tensor, decode_tokens: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run the MLP prefill on all rows, then the decode on the first rows."""
    executor.dispatch(hidden, True)
    prefill = executor.collect(timeout=COLLECT_TIMEOUT_S)
    executor.dispatch(hidden[:decode_tokens].contiguous(), False)
    decode = executor.collect(timeout=COLLECT_TIMEOUT_S)
    return prefill, decode
