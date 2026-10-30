# SPDX-License-Identifier: Apache-2.0
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .llama3 import Eagle3LlamaForCausalLM, PEagleLlamaForCausalLM
from .qwen3 import DFlashQwen3ForCausalLM, Qwen3ForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .qwen3_asr import Qwen3ASRForConditionalGeneration


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


def get_model_cls(architectures) -> type | None:
    """Return the Neuron model class registered for the first known architecture, or None.

    Looks in ``get_models()`` first, then in vLLM's ``ModelRegistry`` for models
    registered out of tree (for example contrib models, through
    ``ModelRegistry.register_model``). The plugin core reads model capabilities
    (``vllm_neuron.model.interfaces``) from this class before the model is built.

    Args:
        architectures: Architecture name, or list of names (``hf_config.architectures``).

    Returns:
        The model (factory) class, or None when no architecture is registered or the
        registered class cannot be imported here.
    """
    if isinstance(architectures, str):
        architectures = [architectures]
    architectures = list(architectures or [])
    models = dict(get_models())
    for arch in architectures:
        if arch in models:
            return models[arch]
    from vllm.model_executor.models.registry import ModelRegistry

    for arch in architectures:
        registered = ModelRegistry.models.get(arch)
        if registered is None:
            continue
        # An entry that cannot be imported here (e.g. a GPU-only model) is not a Neuron model.
        try:
            return registered.load_model_cls()
        except Exception:
            return None
    return None
