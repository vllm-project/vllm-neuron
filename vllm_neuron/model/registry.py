# SPDX-License-Identifier: Apache-2.0
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .llama3 import Eagle3LlamaForCausalLM, PEagleLlamaForCausalLM
from .qwen3 import DFlashQwen3ForCausalLM, Qwen3ForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .qwen3_asr import Qwen3ASRForConditionalGeneration
from .voxtral import VoxtralForConditionalGeneration


def get_models() -> list[tuple[str, type]]:
    """Return a list of available model classes.

    Returns:
        list[tuple[str, type]]: A list of tuples containing model names and their corresponding classes.
            Each tuple contains (model_name, model_class) where:
            - model_name (str): The string identifier for the model, compatible with Hugging Face transformers architecture
            - model_class (type): The actual model class implementation
    """
    models = [
        ("LlamaForCausalLM", LlamaForCausalLM),
        ("GptOssForCausalLM", GptOssForCausalLM),
        ("Eagle3LlamaForCausalLM", Eagle3LlamaForCausalLM),
        ("PEagleLlamaForCausalLM", PEagleLlamaForCausalLM),
        ("DFlashDraftModel", DFlashQwen3ForCausalLM),
        ("Qwen3ForCausalLM", Qwen3ForCausalLM),
        ("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),
        ("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration),
        ("VoxtralForConditionalGeneration", VoxtralForConditionalGeneration),
    ]

    # SyntheticNeuronModel is a testing-only model that replaces real neural
    # network computation with deterministic KV cache fill/verify. Useful for
    # validating infrastructure (KV transfer, sharding, block management)
    # without requiring model weights or compilation.
    # Not for production inference — gated to avoid exposing to customers.
    if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
        from .synthetic import SyntheticNeuronModel
        from .synthetic.synthetic_stacked import SyntheticStackedNeuronModel

        models.append(("SyntheticNeuronModel", SyntheticNeuronModel))
        # SyntheticStackedNeuronModel — synthetic subclass with the GK stacked-KV
        # layout, for the non-GK-prefill -> GK-decode KV-cache adaptor DI test.
        models.append(("SyntheticStackedNeuronModel", SyntheticStackedNeuronModel))

    return models
