# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Qwen3-MoE (Qwen/Qwen3-30B-A3B) contrib tests.

This file travels with the model. It sits at the ``test/`` root because
``test/unit`` and ``test/integration`` both import it (``test/conftest.py``
puts this directory on ``sys.path``). The implementation under ``src/`` is
reached through ``_local_src``.

The helpers give:

- The pinned checkpoint (``MODEL_ID`` at ``HF_REVISION``), its ``config.json``
  and a fingerprint of the tensor names in its ``model.safetensors.index.json``.
  ``QWEN3_MOE_CHECKPOINT_PATH`` points at a local copy. Without it,
  ``resolve_checkpoint`` downloads the pinned revision from the Hugging Face
  Hub (about 61 GB).
- A tiny random-weight checkpoint in the layout of the real one (one
  ``gate_proj`` / ``up_proj`` / ``down_proj`` tensor per expert). It keeps
  head_dim 128 and, at TP 2, 2 query heads and 1 KV head per rank, as the
  real model has at TP 16. The CPU tests use it, so they need no download.
- The HF ``Qwen3MoeForCausalLM`` of transformers as the independent
  reference. It does not use the code under ``src/``.
- A mock of vLLM's current config for ``MPExecutor`` ranks, which run without
  a vLLM engine.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch

MODEL_ID = "Qwen/Qwen3-30B-A3B"
HF_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
CHECKPOINT_ENV = "QWEN3_MOE_CHECKPOINT_PATH"
ARCHITECTURE = "Qwen3MoeForCausalLM"
# The golden cache key of the integration test includes the pinned revision.
GOLDEN_VARIANT = f"rev-{HF_REVISION[:12]}"

# config.json of MODEL_ID at HF_REVISION, verbatim.
PINNED_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 48,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
    "num_attention_heads": 32,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "output_router_logits": False,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "router_aux_loss_coef": 0.001,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

# The tensor names of model.safetensors.index.json at HF_REVISION: their
# count, and the SHA-256 of the sorted names joined with "\n". 48 layers of
# 9 attention / norm / router tensors and 3 * 128 expert tensors, plus the
# embedding, the final norm and the LM head.
NUM_CHECKPOINT_TENSORS = 18867
CHECKPOINT_NAMES_SHA256 = (
    "2dc20e7e2721e2db383851257df6af98df3705c28db667bd2222f7310a7b97a8"
)

# The tiny checkpoint. hidden_size is a multiple of 256, as the decode MoE
# kernel needs. 4 query heads and 2 KV heads give 2 query heads and 1 KV head
# per rank at TP 2. At EP 2 each rank holds 4 complete experts; without EP
# each rank holds a 128-column shard of the 8 experts.
TINY_OVERRIDES = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "max_window_layers": 2,
    "max_position_embeddings": 4096,
    "vocab_size": 512,
    # The real ids (151643, 151645) are outside the tiny vocabulary.
    "bos_token_id": 510,
    "eos_token_id": 511,
}
TINY_TP = 2
# Standard deviation of the tiny router weight. With inputs of size ~1 it
# gives router logits of about 2, so the top-k choice is clear of near-ties
# that BF16 rounding could flip.
TINY_ROUTER_STD = 0.2

COLLECT_TIMEOUT_S = 3600.0


# =============================================================================
# Checkpoints
# =============================================================================


def resolve_checkpoint() -> str:
    """Return the local directory of the pinned checkpoint.

    ``QWEN3_MOE_CHECKPOINT_PATH`` points at a checkpoint that is already on
    disk. Without it, download the pinned revision from the Hugging Face Hub
    (about 61 GB).
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


def names_fingerprint(names) -> tuple[int, str]:
    """Return ``(count, sha256)`` of a set of tensor names, as pinned above."""
    names = sorted(names)
    return len(names), hashlib.sha256("\n".join(names).encode()).hexdigest()


def tiny_hf_config(**overrides):
    """Return the HF ``Qwen3MoeConfig`` of the tiny checkpoint."""
    from transformers import Qwen3MoeConfig as HFQwen3MoeConfig

    values = {
        k: v
        for k, v in PINNED_CONFIG.items()
        if k not in ("architectures", "model_type", "transformers_version")
    }
    values.update(TINY_OVERRIDES)
    values.update(overrides)
    config = HFQwen3MoeConfig(**values)
    config.architectures = [ARCHITECTURE]
    return config


def _random_hf_model(seed: int, **overrides):
    """Return an HF ``Qwen3MoeForCausalLM`` with random weights in FP32.

    The RMSNorm weights are random too (HF initializes them to 1), so a norm
    in the wrong place shows up in the outputs.
    """
    from transformers import Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM

    torch.manual_seed(seed)
    model = HFQwen3MoeForCausalLM(tiny_hf_config(**overrides))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
            elif name.endswith("mlp.gate.weight"):
                param.copy_(TINY_ROUTER_STD * torch.randn_like(param))
    return model.eval()


def hub_layout_state_dict(model) -> dict[str, torch.Tensor]:
    """Return the state dict of an HF model in the layout of the Hub checkpoint.

    transformers keeps the experts of a layer as two fused tensors
    (``experts.gate_up_proj`` ``[E, 2 * I, H]`` with the gate rows first, and
    ``experts.down_proj`` ``[E, H, I]``). The Hub checkpoint stores one
    ``gate_proj``, ``up_proj`` and ``down_proj`` tensor per expert, which is
    the layout that the implementation loads.
    """
    out = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("mlp.experts.gate_up_proj"):
            prefix = name[: -len("gate_up_proj")]
            inter = tensor.shape[1] // 2
            for e in range(tensor.shape[0]):
                out[f"{prefix}{e}.gate_proj.weight"] = tensor[e, :inter]
                out[f"{prefix}{e}.up_proj.weight"] = tensor[e, inter:]
        elif name.endswith("mlp.experts.down_proj"):
            prefix = name[: -len("down_proj")]
            for e in range(tensor.shape[0]):
                out[f"{prefix}{e}.down_proj.weight"] = tensor[e]
        else:
            out[name] = tensor
    return {k: v.detach().contiguous().clone() for k, v in out.items()}


def make_tiny_checkpoint(directory: str, seed: int = 42, **overrides) -> str:
    """Save a random-weight tiny Qwen3-MoE checkpoint in BF16 into ``directory``.

    The file has the tensor names of the Hub checkpoint (see
    ``hub_layout_state_dict``) and a ``config.json`` with the pinned keys and
    ``TINY_OVERRIDES``.
    """
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok=True)
    model = _random_hf_model(seed, **overrides)
    tensors = {k: v.to(torch.bfloat16) for k, v in hub_layout_state_dict(model).items()}
    save_file(
        tensors, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"}
    )
    config = dict(PINNED_CONFIG)
    config.update(TINY_OVERRIDES)
    config.update(overrides)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return directory


def read_checkpoint_tensors(checkpoint: str, names=None) -> dict[str, torch.Tensor]:
    """Read tensors from the tiny single-file checkpoint with safetensors only."""
    from safetensors import safe_open

    with safe_open(
        os.path.join(checkpoint, "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        keys = list(f.keys()) if names is None else list(names)
        return {k: f.get_tensor(k) for k in keys}


def load_hf_config(checkpoint: str):
    """Load the HF config of a checkpoint directory with eager attention."""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(checkpoint)
    hf_config._attn_implementation = "eager"
    return hf_config


def load_hf_reference(checkpoint: str, dtype: torch.dtype):
    """Build the HF ``Qwen3MoeForCausalLM`` from the tiny checkpoint tensors.

    The BF16 checkpoint tensors go into the fused HF expert layout by hand, so
    the reference does not depend on the checkpoint conversion of
    ``from_pretrained``.
    """
    from transformers import Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM

    hf_config = load_hf_config(checkpoint)
    tensors = read_checkpoint_tensors(checkpoint)
    state = {}
    for name, tensor in tensors.items():
        if ".mlp.experts." in name:
            continue
        state[name] = tensor
    for layer in range(hf_config.num_hidden_layers):
        p = f"model.layers.{layer}.mlp.experts."
        experts = range(hf_config.num_experts)
        state[p + "gate_up_proj"] = torch.stack(
            [
                torch.cat(
                    [
                        tensors[f"{p}{e}.gate_proj.weight"],
                        tensors[f"{p}{e}.up_proj.weight"],
                    ]
                )
                for e in experts
            ]
        )
        state[p + "down_proj"] = torch.stack(
            [tensors[f"{p}{e}.down_proj.weight"] for e in experts]
        )
    # Build on CPU, not on the meta device: the RoPE inv_freq buffer is not in
    # the state dict. The FP32 parameters take the BF16 values exactly.
    model = HFQwen3MoeForCausalLM(hf_config)
    model.load_state_dict(state, strict=True)
    return model.to(dtype).eval()


# =============================================================================
# MPExecutor ranks
# =============================================================================


def set_mock_vllm_config(ep_enabled: bool) -> None:
    """Make ``vllm.config.get_current_vllm_config`` return a config with EP set.

    ``MPExecutor`` ranks run without a vLLM engine, and the MoE block reads
    ``parallel_config.enable_expert_parallel`` and
    ``parallel_config.data_parallel_size`` from the current vLLM config. Call
    this in the rank process, before the model is built.
    """
    from unittest.mock import MagicMock

    import vllm.config as vllm_config_module

    mock_config = MagicMock()
    mock_config.parallel_config.enable_expert_parallel = ep_enabled
    mock_config.parallel_config.data_parallel_size = 1
    mock_config.cache_config.cache_dtype = "auto"
    vllm_config_module.get_current_vllm_config = lambda: mock_config
