# SPDX-License-Identifier: Apache-2.0
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .llama3 import Eagle3LlamaForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .qwen3 import Qwen3ForCausalLM


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
        ("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),
        # Dense and sparse Qwen3 configs share one conditional implementation.
        ("Qwen3ForCausalLM", Qwen3ForCausalLM),
        ("Qwen3MoeForCausalLM", Qwen3ForCausalLM),
    ]

    # SyntheticNeuronModel is a testing-only model that replaces real neural
    # network computation with deterministic KV cache fill/verify. Useful for
    # validating infrastructure (KV transfer, sharding, block management)
    # without requiring model weights or compilation.
    # Not for production inference — gated to avoid exposing to customers.
    if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
        from .synthetic import SyntheticNeuronModel

        models.append(("SyntheticNeuronModel", SyntheticNeuronModel))

    return models
