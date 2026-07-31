# SPDX-License-Identifier: Apache-2.0
"""Factory for MiMo-V2.5 implementation selection."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.compile.platform import get_platform_target
from vllm_neuron.model.neuron_config import NeuronConfig


class MiMoV2ForCausalLM(nn.Module):
    """Validates the config and selects a MiMo-V2.5 implementation.

    Extends ``nn.Module`` so vLLM's ``ModelRegistry`` accepts the entry. Only a
    BF16 runtime exists today: the released checkpoint stores its matmul weights
    as 128x128-blockwise FP8-e4m3 and the weight loaders dequantize to BF16
    host-side at load time (see ``weight_loaders_bf16``), so both
    ``quantization=None|"bf16"`` and ``quantization="fp8"`` land on the same
    implementation. There is no on-device FP8 path: Neuron has no 128x128
    blockwise-FP8 MoE kernel.
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
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: object | None = None,
    ) -> nn.Module:
        """Create the model from configs, returning the implementation directly.

        Accepts the multimodal call signature (``text_neuron_config=`` plus
        ``vision_neuron_config=``) as well as the positional text-only one, and
        ignores the vision half. That is not defensive padding -- it is forced by
        this checkpoint:

        ``configuration_mimo_v2.py`` sets ``self.vision_config`` unconditionally,
        so ``hasattr(hf_config, "vision_config")`` is True even when the key is
        absent from ``config.json``. That hasattr is what makes
        ``NeuronPlatform.check_and_update_config`` synthesize a
        ``vision_neuron_config``, which in turn makes ``load_model`` take the
        multimodal ``from_configs`` branch. The offline example can dodge it with
        a callable ``hf_overrides`` that ``delattr``s the attribute, but ``vllm
        serve`` only accepts a JSON dict -- and a None-valued key still satisfies
        hasattr, so there is no CLI-side spelling that works. Absorbing the
        kwargs here is the one fix both entrypoints share.

        Ignoring ``vision_neuron_config`` is sound because this port implements
        the TEXT decoder only: the vision/audio towers are never built and their
        checkpoint tensors are never referenced. Serve with
        ``--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'`` so no request
        can carry multimodal input the model cannot consume.
        """
        return cls._select_implementation(
            hf_config, neuron_config if neuron_config is not None else text_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        from .model import MiMoV2ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16", "fp8"):
            raise ValueError(
                f"MiMo-V2.5 supports quantization=None/'bf16'/'fp8' (the FP8 "
                f"checkpoint is dequantized to BF16 at load time), got "
                f"{quantization!r}."
            )

        # Platform detection raises on a non-Neuron host unless
        # NEURON_PLATFORM_TARGET_OVERRIDE is set; treat that as "unknown" rather
        # than an error so config-only construction still works off-instance.
        try:
            platform = get_platform_target()
        except Exception:
            platform = None
        if platform in ("trn1", "trn1n", "inf2"):
            raise ValueError(
                f"MiMo-V2.5 needs 24 GB+ per device for its 256 BF16 experts; "
                f"{platform} has 16 GB. Use trn2 or trn3."
            )

        # Attention is eager because the 192-wide Q/K heads exceed every fused
        # attention kernel's 128 cap — a perf note, not a correctness issue.
        # What IS a correctness trap is a malformed hybrid pattern: a wrong
        # length would silently turn SWA layers into full-attention ones.
        pattern = getattr(hf_config, "hybrid_layer_pattern", None)
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if pattern is not None and num_layers is not None and len(pattern) != num_layers:
            raise ValueError(
                f"hybrid_layer_pattern has {len(pattern)} entries but "
                f"num_hidden_layers={num_layers}."
            )
