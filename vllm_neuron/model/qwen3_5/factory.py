# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.5, registered under the checkpoint's HF architecture name.

The checkpoint declares ``Qwen3_5ForConditionalGeneration``, so vLLM's frontend
supplies the config and the multimodal processor for free and only *execution* is
replaced here.

Two implementations sit behind this: the text-only decoder (``model.py``) and the
vision-language model (``vl.py``). Which one is built depends on whether the
runner supplied a ``VisionNeuronConfig`` — see ``_select_implementation``.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.interfaces import SupportsMaxPixels, SupportsSpatialMerge
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


class Qwen3_5ForConditionalGeneration(
    nn.Module, SupportsSpatialMerge, SupportsMaxPixels
):
    """Selects the text-only or the vision-language implementation.

    The spatial-merge and max-pixels classmethods have to live **here**, not on
    the implementation: ``vision_utils`` resolves them through the *registry*,
    which holds this factory. Putting them only on the VL class silently yields a
    merge factor of 1, which mis-sizes the encoder cache blocks by 4x and fails
    at graph capture with an ``aten::index_put`` lowering error.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        vision_neuron_config: VisionNeuronConfig | None,
    ) -> nn.Module:
        # The runner only builds a VisionNeuronConfig when the engine was
        # configured to serve images or video: the platform skips vision bucket
        # resolution when every ``limit_mm_per_prompt`` count is 0, and without
        # that dict the runner leaves this None. So it is the signal for "vision
        # is wanted", and a text-only launch pays neither the tower's weights nor
        # its compile time.
        if vision_neuron_config is None:
            from .model import Qwen3_5ForCausalLM

            return Qwen3_5ForCausalLM.from_configs(
                hf_config,
                text_neuron_config=text_neuron_config,
                vision_neuron_config=None,
            )

        from .vl import Qwen3_5VLForConditionalGeneration

        return Qwen3_5VLForConditionalGeneration.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

    # ── vLLM's hybrid-model contract ─────────────────────────────────────
    # vLLM asks the *registered* class — which is this one, not its own
    # implementation — for the recurrent state geometry, and uses it to size the
    # state pages the block planner allocates. Delegating to
    # ``Mamba*Calculator`` is what keeps that sizing identical to what
    # ``Qwen3_5TextConfig.state_shapes`` reports to the model runner; a second
    # copy of the arithmetic would alias memory rather than raise.

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> tuple[tuple[int, ...], ...]:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        hf_text_config = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf_text_config.linear_num_key_heads,
            hf_text_config.linear_num_value_heads,
            hf_text_config.linear_key_head_dim,
            hf_text_config.linear_value_head_dim,
            hf_text_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )

        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    # Vision support is selected in ``_select_implementation`` above, not guarded
    # here: the two implementations differ only in whether the tower exists.

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        """Raw vision tokens that collapse into one embedding token."""
        return hf_config.vision_config.spatial_merge_size**2

    @classmethod
    def get_max_pixels_token_count(
        cls, hf_config: PretrainedConfig, max_pixels: int
    ) -> int:
        """A ``max_pixels`` cap expressed as a raw (pre-merge) token count."""
        return max_pixels // (hf_config.vision_config.patch_size**2)


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The sparse checkpoints (``model_type: qwen3_5_moe``, e.g. 35B-A3B).

    A separate registry entry only because the checkpoint declares a different
    architecture name. The decoder is the same hybrid stack with the same state
    geometry, and the dense MLP is swapped for ``Qwen3_5SparseMoeBlock`` inside
    ``Qwen3_5DecoderLayer`` off ``config.is_moe`` -- so everything inherited
    here, including the recurrent-state and vision classmethods vLLM resolves
    through the registry, is correct unchanged.
    """
