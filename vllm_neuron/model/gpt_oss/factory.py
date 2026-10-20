# SPDX-License-Identifier: Apache-2.0
"""Factory for GptOss model selection based on platform and configuration."""

import logging

import torch.nn as nn
from transformers import PretrainedConfig

from libtorch_neuronx_lite.compile.platform import get_platform_target
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.hardware_config import get_logical_nc_config

logger = logging.getLogger(__name__)


def should_use_gk_decode(
    quantization: str | None, platform_target: str | None, kv_role: str | None
) -> bool:
    """Return True when the GPT-OSS decode giga-kernel is the right path.

    vLLM-Neuron owns this choice; the customer does not set it.

    The giga-kernel runs only when all four conditions hold. Quantization is mxfp4. The platform
    is trn3. The logical-NC config is 1, because the kernel is traced at grid[1]. And this is the
    decode engine of a disaggregated deployment, meaning a kv_transfer role that is not
    kv_producer. A fused or prefill engine needs lnc2 per-layer prefill instead.

    Args:
        quantization: The resolved quantization, for example "mxfp4".
        platform_target: The platform target, for example "trn3".
        kv_role: The kv_transfer role, or None when kv_transfer is not set.

    Returns:
        bool: True to select the giga-kernel decode model.

    Examples:
        >>> should_use_gk_decode("mxfp4", "trn3", "kv_consumer")  # doctest: +SKIP
        True
    """
    if quantization != "mxfp4":
        return False
    if not (platform_target or "").startswith("trn3"):
        return False
    if get_logical_nc_config() != 1:
        return False
    if kv_role is None or kv_role == "kv_producer":
        return False
    return True


def _log_model_choice(module: str, quantization, platform: str, why: str) -> None:
    """Name the gpt-oss model module actually selected, once, greppably.

    The three modules are not interchangeable: mxfp4 shuffles the hidden dim, so its padding
    lanes are interleaved and its RMSNorm variance convention differs from bf16's. Which one
    loaded was previously inferable only from the quantization value plus this function's
    source: mxfp4 names itself only once the giga-kernel path is taken, and bf16 not at all.
    """
    logger.info(
        "[gpt_oss] model=%s quantization=%r platform=%s reason=%s",
        module,
        quantization,
        platform,
        why,
    )


class GptOssForCausalLM(nn.Module):
    """Factory that validates config and selects the appropriate GptOss implementation.

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
        platform = get_platform_target()
        quantization = neuron_config.quantization if neuron_config else None

        # Every path below names the module it picked. Grep "[gpt_oss] model=".
        # gk_decode is derived by should_use_gk_decode() in platform
        # check_and_update_config, not set by the customer.
        gk_decode = bool(neuron_config and neuron_config.gk_decode)
        if gk_decode:
            from .model_mxfp4_perf import GptOssForCausalLM as Model

            _log_model_choice(
                "model_mxfp4_perf", quantization, platform, "gk_decode=True"
            )
            return Model.from_configs(hf_config, neuron_config)

        if quantization == "mxfp4":
            from .model_mxfp4 import GptOssForCausalLM as Model

            _log_model_choice(
                "model_mxfp4", quantization, platform, "quantization=mxfp4"
            )
            return Model.from_configs(hf_config, neuron_config)

        if quantization == "bf16":
            from .model_bf16 import GptOssForCausalLM as Model

            _log_model_choice("model_bf16", quantization, platform, "quantization=bf16")
            return Model.from_configs(hf_config, neuron_config)

        if platform.startswith("trn3"):
            from .model_mxfp4 import GptOssForCausalLM as Model

            _log_model_choice(
                "model_mxfp4", quantization, platform, "trn3 platform default"
            )
            return Model.from_configs(hf_config, neuron_config)

        from .model_bf16 import GptOssForCausalLM as Model

        _log_model_choice("model_bf16", quantization, platform, "fallback")
        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported. Add rules as needed."""
        # TODO: Below is one example rule, but we need to add more and make user errors clear

        quantization = neuron_config.quantization if neuron_config else None
        platform = get_platform_target()

        if quantization == "mxfp4" and platform == "trn2":
            raise ValueError(
                "quantization='mxfp4' is not supported on TRN2. "
                "Please use quantization='bf16'"
            )
