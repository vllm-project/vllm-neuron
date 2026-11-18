# SPDX-License-Identifier: Apache-2.0
"""Factory registered with vLLM's ModelRegistry for LiquidAI/d1-3B."""

from __future__ import annotations

import math

from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.lfm2_vl import (
    Lfm2VLDummyInputsBuilder,
    Lfm2VLMultiModalProcessor,
    Lfm2VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_neuron.model.interfaces import SupportsEncoderBudget, SupportsSpatialMerge
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig

ARCHITECTURE = "Lfm2VlForConditionalGeneration"


@MULTIMODAL_REGISTRY.register_processor(
    Lfm2VLMultiModalProcessor,
    info=Lfm2VLProcessingInfo,
    dummy_inputs=Lfm2VLDummyInputsBuilder,
)
class D1ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsSpatialMerge, SupportsEncoderBudget
):
    """Registry entry for the ``Lfm2VlForConditionalGeneration`` architecture.

    vLLM inspects this class in the main process. It must not be the upstream
    ``Lfm2VLForConditionalGeneration``: that class declares the hybrid (mamba)
    interface, which makes vLLM allocate mamba state the Neuron runner cannot use.
    This class declares the multimodal interface (with the upstream LFM2-VL
    processor, so prompts and images are processed exactly as upstream) but not
    the hybrid one, so vLLM manages every layer through the plain KV spec returned
    by ``D1NeuronForCausalLM.get_kv_spec`` (attention K/V plus the conv state).

    The Neuron runner never instantiates it; it calls :meth:`from_configs`.
    """

    merge_by_field_config = True

    def __init__(self, *, vllm_config=None, prefix: str = "", **kwargs):
        super().__init__()
        raise NotImplementedError("Built through from_configs by the Neuron runner")

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        return None

    def embed_multimodal(self, **kwargs):
        raise NotImplementedError

    def embed_input_ids(self, input_ids, *args, **kwargs):
        raise NotImplementedError

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError

    def compute_logits(self, hidden_states):
        raise NotImplementedError

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> nn.Module:
        """Build the Neuron implementation (text-only without an encoder config)."""
        from .model import D1NeuronForCausalLM

        return D1NeuronForCausalLM.from_configs(
            hf_config,
            neuron_config or text_neuron_config,
            encoder_neuron_config=encoder_neuron_config,
        )

    # ── Plugin encoder-sizing protocols (vllm_neuron.model.interfaces) ──
    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        return getattr(hf_config, "downsample_factor", 2) ** 2

    @classmethod
    def get_max_encoder_embed_budget(
        cls, hf_config: PretrainedConfig, encoder_neuron_config: EncoderNeuronConfig
    ) -> int | None:
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return None
        return max(buckets) // cls.get_vision_token_merge_factor(hf_config)

    @classmethod
    def get_max_encoder_blocks_per_request(
        cls, hf_config: PretrainedConfig, encoder_neuron_config: EncoderNeuronConfig
    ) -> int:
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return 0
        # One tile per encoder block, one cache block per tile.
        return math.ceil(
            max(buckets) / encoder_neuron_config.encoder_attention_block_size
        )


def register() -> None:
    """Register d1-3B with vLLM (and the plugin's model list). Call before ``LLM(...)``.

    Two registrations:
      * vLLM's ``ModelRegistry``: resolves the architecture to this class.
      * The plugin's ``get_models()`` list: the plugin's multimodal sizing helpers
        (``vllm_neuron/utils/mm_utils.py``: merge factor, encoder budget, blocks per
        request) look the class up there, not in vLLM's registry. Without this entry
        a contrib vision model gets a merge factor of 1 and an encoder budget of 0.
        This patches the function in this process only; it is idempotent.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, D1ForConditionalGeneration)

    import vllm_neuron.model.registry as plugin_registry

    if getattr(plugin_registry.get_models, "_d1_3b", False):
        return
    original = plugin_registry.get_models

    def get_models():
        models = [(a, c) for a, c in original() if a != ARCHITECTURE]
        return models + [(ARCHITECTURE, D1ForConditionalGeneration)]

    get_models._d1_3b = True
    plugin_registry.get_models = get_models
