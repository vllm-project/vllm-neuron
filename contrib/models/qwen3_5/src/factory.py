# SPDX-License-Identifier: Apache-2.0
"""The classes vLLM's ``ModelRegistry`` builds for Qwen3.5 checkpoints.

The dense checkpoints (2B, 27B) declare ``Qwen3_5ForConditionalGeneration`` and
the sparse ones (35B-A3B, 397B-A17B) declare
``Qwen3_5MoeForConditionalGeneration``. Both build the same text-only hybrid
decoder in ``model.py``; ``Qwen3_5DecoderLayer`` picks the dense MLP or the MoE
block from the checkpoint config.

Register these classes before the engine starts, either from your script
(``ModelRegistry.register_model``) or with the ``vllm.general_plugins`` entry
point in this folder's ``pyproject.toml``. See ``plugin.py``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Factory for the dense Qwen3.5 checkpoints.

    Registering this class replaces vLLM's own ``Qwen3_5ForConditionalGeneration``
    in every process, including the frontend. So it must also carry vLLM's
    hybrid-model contract (``is_hybrid`` and the two ``get_mamba_state_*``
    classmethods): vLLM reads it from the REGISTERED class to set
    ``ModelConfig.is_hybrid``, ``mamba_block_size`` and the state page size.
    Without it vLLM plans the model as attention-only.

    The class does not implement ``SupportsMultiModal``, so vLLM treats the
    checkpoint as text-only and builds no multimodal processor.

    vLLM also inspects the registered class for its text-generation protocol
    (``is_text_generation_model``): ``__init__`` must take ``vllm_config``, and
    the class needs ``embed_input_ids``, ``compute_logits`` and a ``forward``
    with ``input_ids``/``positions``. Without them vLLM does not treat the
    architecture as a generative model. The Neuron runner never calls
    ``__init__``, ``embed_input_ids`` or ``compute_logits``; it builds the model
    with ``from_configs`` and calls ``forward``.
    """

    # vLLM's ``IsHybrid`` / ``HasInnerState`` protocol flags.
    is_hybrid: ClassVar[Literal[True]] = True
    has_inner_state: ClassVar[Literal[True]] = True

    def __init__(
        self,
        hf_config: PretrainedConfig | None = None,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        # ``vllm_config``/``prefix`` exist only for vLLM's protocol check (see
        # the class docstring).
        if hf_config is None:
            raise TypeError(f"build {type(self).__name__} with from_configs()")
        super().__init__()
        self._model = self.from_configs(
            hf_config, text_neuron_config, encoder_neuron_config
        )

    def forward(self, input_ids, positions, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids):
        return self._model.language_model.embed_tokens(input_ids, scatter_tokens=False)

    def compute_logits(self, hidden_states):
        return self._model.lm_head(hidden_states)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
    ) -> nn.Module:
        """Build the text-only implementation.

        The runner passes ``encoder_neuron_config`` when the platform resolved
        vision buckets for the checkpoint's ``vision_config``. It is ignored:
        this port builds no vision tower.
        """
        from .model import Qwen3_5ForCausalLM

        # Single-shot prefill only: a later chunk would restart DeltaNet from a
        # zero state and attend only to itself, with no error. The 0.29 runner
        # turns segmented prefill on whenever max_num_batched_tokens <
        # max_model_len, so refuse that here, at load.
        segments = getattr(text_neuron_config, "kv_segment_size_buckets", None)
        if segments:
            raise ValueError(
                f"Qwen3.5 does not support chunked prefill, but "
                f"kv_segment_size_buckets={segments}: set max_num_batched_tokens "
                f"== max_model_len (and keep enable_prefix_caching=False)."
            )

        return Qwen3_5ForCausalLM.from_configs(
            hf_config, text_neuron_config=text_neuron_config
        )

    # ── vLLM's hybrid-model contract ─────────────────────────────────────
    # vLLM uses these to size the recurrent-state pages that the block planner
    # allocates. Delegating to ``MambaState*Calculator`` keeps that size equal to
    # what ``Qwen3_5TextConfig.state_shapes`` reports to the model runner. A
    # second copy of the arithmetic would alias memory instead of raising.

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config
    ) -> tuple[tuple[int, ...], ...]:
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

    @classmethod
    def get_mamba_state_copy_func(cls):
        # Only prefix caching over recurrent state (mamba_cache_mode "align" or
        # "all") copies states between blocks, and the plugin rejects it.
        raise NotImplementedError(
            "Qwen3.5 on Neuron does not support prefix caching; set "
            "enable_prefix_caching=False"
        )


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Factory for the sparse Qwen3.5 checkpoints (``model_type: qwen3_5_moe``).

    A separate class only because the checkpoint declares a different
    architecture name. The decoder and the state geometry are the same.
    """


__all__ = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
]
