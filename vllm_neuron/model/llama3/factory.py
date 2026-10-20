# SPDX-License-Identifier: Apache-2.0
"""Factory for Llama model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class LlamaForCausalLM(nn.Module):
    """Factory that validates config and selects the appropriate Llama implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls to it.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        # TODO: Below is a preliminary logic and needs to be made more robust
        from .model import LlamaForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported. Add rules as needed.

        Parses ``hf_config.quantization_config`` through
        :meth:`~vllm_neuron.model.llama3.quantization.QuantizationSpec.from_hf_quantization_config`
        so unsupported quantization schemes fail fast — at model
        construction, with a clear message — instead of at the first
        forward call inside a kernel launch.

        The parse result is discarded; :class:`LlamaConfig.from_configs`
        re-parses it and attaches the :class:`QuantizationSpec` to the
        config object. Re-parsing costs nothing (pure-Python dict walk)
        and keeps the factory stateless.
        """
        del neuron_config  # reserved for future validation rules
        from .quantization import QuantizationSpec

        quant_cfg = None
        if hf_config is not None:
            quant_cfg = getattr(hf_config, "quantization_config", None)
            # HuggingFace configs also store the dict under __dict__ when
            # the attribute wasn't declared on the pretrained class; fall
            # back to dict form so ModelOpt-injected configs are caught.
            if quant_cfg is None and hasattr(hf_config, "to_dict"):
                quant_cfg = hf_config.to_dict().get("quantization_config")
        # Raises ValueError on unsupported scheme / malformed config;
        # returns None when the checkpoint is unquantized.
        QuantizationSpec.from_hf_quantization_config(quant_cfg)


class Eagle3LlamaForCausalLM(nn.Module):
    """Factory that validates config and selects the appropriate Eagle3 Llama implementation.

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

        # TODO: Below is a preliminary logic and needs to be made more robust
        from .eagle3_model import Eagle3LlamaForCausalLM as Model

        return Model.from_configs(
            config=config,
            start_layer_idx=start_layer_idx,
            neuron_config=neuron_config,
        )

    @classmethod
    def _validate_config(cls, config: PretrainedConfig) -> None:
        """Validate that the configuration is supported. Add rules as needed."""
        # TODO: Add validation rules
        pass


class PEagleLlamaForCausalLM(nn.Module):
    """Factory for P-EAGLE (Parallel-Drafting EAGLE) Llama implementation.

    Routes to PEagleLlamaForCausalLM from peagle_model.py.
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
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
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
        from .peagle_model import PEagleLlamaForCausalLM as Model

        return Model.from_configs(
            config=config,
            start_layer_idx=start_layer_idx,
            neuron_config=neuron_config,
        )
