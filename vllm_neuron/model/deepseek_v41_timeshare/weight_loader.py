# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash (timeshare) weight loading — STAGE 1d SCAFFOLD
==================================================================

Separated from `model.py` because this checkpoint's loading is not a key
rename. Four things make it unlike the Qwen3/Llama3 loaders in this repo, and
each is called out at the stub that owns it:

  1. **Mixed storage dtypes.** Attention is stored `fp8`, expert weights `fp4`,
     arithmetic runs `bf16`. Dequantization scales travel with the weights.
  2. **Engram tables exceed a bank.** `layers.1.engram` and `layers.14.engram`
     are 94.6 GiB each — 3.9x a 24 GiB HBM bank — so they cannot be loaded
     whole on any rank. A 64-way split still leaves 1.5 GiB per unit.
  3. **Two expert pools.** 384 routed experts for the main MoE plus 128 for
     DSpark, with different activation counts (6 vs 3).
  4. **Hyper-connection streams.** Sublayer weights are indexed by
     `hc_mult=4` residual streams, which the checkpoint stores flattened.

Weights live at `partition-artifact/hf/` (48 safetensors shards, 474.5 GiB).
Per `references/INDEX.md`: do not re-download from HuggingFace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import DeepseekV41ForCausalLM

_STAGE = "DeepSeek-V4.1-Flash timeshare port is at stage 1d (scaffold only)"


def build_weight_mappings(
    model: DeepseekV41ForCausalLM,
) -> dict[str, str | list[str]]:
    """Map this model's parameter names to checkpoint key(s).

    One entry per parameter. A `list` value fuses several checkpoint tensors
    into one parameter (as Qwen3 does for QKV); a `str` value is 1:1.

    Returns:
        `{parameter_name: checkpoint_key | [checkpoint_key, ...]}`
    """
    raise NotImplementedError(
        f"{_STAGE}. build_weight_mappings() needs the checkpoint key layout for: "
        "low-rank attention (q_lora_rank=1280, o_lora_rank=1024, o_groups=8, "
        "num_key_value_heads=1 latent KV at head_dim=512); "
        "the hyper-connection streams (hc_mult=4); the two Engram tables "
        "(layers 1, 14); the 384-expert MoE; and the 128-expert DSpark pool "
        "(layers 37-39). Read a shard's key list before writing these."
    )


def register_weight_loaders(model: DeepseekV41ForCausalLM) -> None:
    """Attach per-parameter TP sharding transforms.

    Uses `vllm_neuron.utils.weight_loader`'s `set_weight_loader` with
    `sharding_weight_loader` / `fused_qkv_weight_loader`, plus custom
    transforms for the cases those do not cover (Engram row sharding, fp4
    expert unpacking).
    """
    raise NotImplementedError(
        f"{_STAGE}. register_weight_loaders() must cover fp8 attention and fp4 "
        "expert dequantization, and an Engram strategy: the 94.6 GiB tables are "
        "3.9x a 24 GiB bank, so they must be split, tiered, or placed off-HBM "
        "before any rank can load one."
    )


def load_weights(
    model: DeepseekV41ForCausalLM,
    checkpoint_path: str,
    device: torch.device,
    cache_dir: str | None = None,
) -> None:
    """Load the sharded checkpoint into `model`, in place.

    Args:
        model: The model to populate.
        checkpoint_path: Directory of safetensors shards.
        device: Target device for the materialized rank-local weights.
        cache_dir: Optional directory for the checkpoint index cache.
    """
    raise NotImplementedError(
        f"{_STAGE}. load_weights() wiring, once the two stubs above are real:\n"
        "    mappings = build_weight_mappings(model)\n"
        "    register_weight_loaders(model)\n"
        "    ckpt = SafetensorsCheckpoint(checkpoint_path, cache_dir)\n"
        "    sharded = ckpt.load_sharded_pipelined(\n"
        "        model.rank, model.world_size, model, mappings, device\n"
        "    ).state_dict\n"
        "    model.load_state_dict(sharded, strict=False, assign=True)"
    )
