# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3-MoE model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


def _get(hf_config, name: str, default=None):
    if isinstance(hf_config, dict):
        value = hf_config.get(name)
    else:
        value = getattr(hf_config, name, None)
        if value is None and hasattr(hf_config, "to_dict"):
            value = hf_config.to_dict().get(name)
    return default if value is None else value


class Qwen3MoeForCausalLM(nn.Module):
    """Factory that validates the config and selects the Qwen3-MoE implementation.

    This class extends nn.Module to satisfy the vLLM ModelRegistry requirements.
    The factory keeps the selected implementation and sends forward() calls to it.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *args,
        **kwargs,
    ):
        """Send the forward pass to the selected implementation."""
        return self._model(*args, input_ids=input_ids, positions=positions, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create the model from the configs. Return the selected implementation."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config, then create the BF16 implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import Qwen3MoeForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig | dict, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the Qwen3-MoE config is supported.

        Stop at construction with a clear error. An unsupported config must not
        run as a silently different model. The implementation supports BF16
        checkpoints in which each decoder layer has a sparse MoE block, full
        attention in every layer, the default RoPE and no attention bias. The
        model checks the tensor-parallel and expert-parallel sizes, because
        only the model knows the process groups.
        """
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for Qwen3-MoE. "
                "Use None or 'bf16'."
            )
        if _get(hf_config, "quantization_config") is not None:
            raise ValueError(
                "Quantized Qwen3-MoE checkpoints are not supported. "
                "Use a BF16 checkpoint."
            )

        # transformers >= 5 names the checkpoint dtype ``dtype``; a raw
        # config.json (transformers < 5) names it ``torch_dtype``.
        torch_dtype = _get(hf_config, "dtype") or _get(hf_config, "torch_dtype")
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"Qwen3-MoE on Neuron supports bfloat16 only. The config dtype "
                f"is {torch_dtype!r}. Use a bfloat16 checkpoint."
            )

        mlp_only_layers = _get(hf_config, "mlp_only_layers", [])
        sparse_step = _get(hf_config, "decoder_sparse_step", 1)
        if mlp_only_layers or sparse_step != 1:
            raise ValueError(
                f"Dense MLP layers are not supported for Qwen3-MoE "
                f"(mlp_only_layers={mlp_only_layers}, "
                f"decoder_sparse_step={sparse_step}). Use a checkpoint in which "
                "each layer has a sparse MoE block."
            )

        # NF.router has no "softmax over all experts, then top-k without
        # renorm" order, so norm_topk_prob=False would route with renormalized
        # weights in prefill and differ from the HF model.
        if not _get(hf_config, "norm_topk_prob", True):
            raise ValueError(
                "Qwen3-MoE on Neuron supports norm_topk_prob=True only (the "
                "top-k routing weights are renormalized to sum to 1)."
            )

        if _get(hf_config, "use_sliding_window", False):
            raise ValueError(
                "Qwen3-MoE sliding-window attention is not supported. Set "
                "use_sliding_window to false in the model config."
            )
        layer_types = _get(hf_config, "layer_types")
        if layer_types and any(t != "full_attention" for t in layer_types):
            raise ValueError(
                "Qwen3-MoE on Neuron supports full attention in every layer "
                f"only. The config has layer_types {sorted(set(layer_types))}."
            )

        if _get(hf_config, "attention_bias", False):
            raise ValueError(
                "Qwen3-MoE on Neuron has no attention bias. Use a checkpoint "
                "with attention_bias set to false."
            )

        hidden_act = _get(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Qwen3-MoE on Neuron supports hidden_act 'silu' only, not "
                f"{hidden_act!r}."
            )

        rope_scaling = _get(hf_config, "rope_scaling") or {}
        rope_parameters = _get(hf_config, "rope_parameters") or {}
        rope_type = (
            rope_scaling.get("rope_type")
            or rope_scaling.get("type")
            or rope_parameters.get("rope_type")
        )
        if rope_type not in (None, "default"):
            raise ValueError(
                f"Qwen3-MoE on Neuron supports the default RoPE only, not "
                f"{rope_type!r}. Remove rope_scaling from the model config."
            )
