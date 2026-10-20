# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3 model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .quantization import QuantScheme, QuantizationSpec


class Qwen3ForCausalLM(nn.Module):
    """Factory that validates config and selects the Qwen3 implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        quant_spec = cls._get_quantization_spec(hf_config)

        # Qwen3-Embedding-8B's HF architectures field is "Qwen3ForCausalLM", so
        # both the generative and pooling models resolve to this factory.
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        if model_config.runner_type == "pooling":
            # Two pooling cases share this arch, split by the checkpoint:
            #   - native sentence-transformers embedding checkpoint (has
            #     modules.json, e.g. Qwen3-Embedding-8B) -> dedicated hand-written
            #     Qwen3ForEmbedding.
            #   - a plain generative checkpoint run with --convert embed -> the
            #     generic pooling adapter (applied at the load_model seam), so
            #     the factory just builds the generative model here.
            from vllm.transformers_utils.config import get_pooling_config

            is_native_st_embedding = (
                get_pooling_config(model_config.model, model_config.revision)
                is not None
            )
            if is_native_st_embedding:
                from .model_embedding import Qwen3ForEmbedding as Model
            else:
                Model = cls._get_generative_model_cls(quant_spec, neuron_config)
        else:
            Model = cls._get_generative_model_cls(quant_spec, neuron_config)

        return Model.from_configs(hf_config, neuron_config)

    @staticmethod
    def _get_quantization_spec(
        hf_config: PretrainedConfig | None,
    ) -> QuantizationSpec | None:
        quant_cfg = None
        if hf_config is not None:
            quant_cfg = getattr(hf_config, "quantization_config", None)
            if quant_cfg is None and hasattr(hf_config, "to_dict"):
                quant_cfg = hf_config.to_dict().get("quantization_config")
        return QuantizationSpec.from_hf_quantization_config(quant_cfg)

    @staticmethod
    def _get_generative_model_cls(
        quant_spec: QuantizationSpec | None,
        neuron_config: NeuronConfig | None,
    ):
        quantization = neuron_config.quantization if neuron_config else None
        if quantization == "mxfp8" or (
            quant_spec is not None
            and quant_spec.get_scheme(layer_index=None, prefix="")
            is QuantScheme.MXFP8_DYNAMIC_PER_TOKEN_CHANNELWISE
        ):
            from .model_mxfp8 import Qwen3ForCausalLMMxfp8 as Model
        else:
            from .model import Qwen3ForCausalLM as Model
        return Model

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported.

        Qwen3-32B dense supports BF16 checkpoints and compressed-tensors MXFP8
        checkpoints for the kernel-enabled POC path.
        """
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16", "mxfp8"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for Qwen3-32B "
                "dense. Supported values are None, 'bf16', and 'mxfp8'."
            )

        cls._get_quantization_spec(hf_config)


class DFlashQwen3ForCausalLM(nn.Module):
    """Factory that validates config and selects the DFlash Qwen3 implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls to it.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            config=config,
            start_layer_idx=start_layer_idx,
            neuron_config=neuron_config,
        )

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(
            config=config,
            start_layer_idx=start_layer_idx,
            neuron_config=neuron_config,
        )

    @classmethod
    def _select_implementation(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(config)

        from .dflash_model import DFlashQwen3ForCausalLM as Model

        return Model.from_configs(
            config=config,
            start_layer_idx=start_layer_idx,
            neuron_config=neuron_config,
        )

    @classmethod
    def _validate_config(cls, config: PretrainedConfig) -> None:
        """Validate that the DFlash configuration is supported."""
        quantization_config = getattr(config, "quantization_config", None)
        if quantization_config is None and hasattr(config, "to_dict"):
            quantization_config = config.to_dict().get("quantization_config")
        if quantization_config is not None:
            raise ValueError(
                "Quantized DFlash draft checkpoints are not supported. "
                "Use an unquantized BF16 draft model."
            )
