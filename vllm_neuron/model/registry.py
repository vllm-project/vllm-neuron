# SPDX-License-Identifier: Apache-2.0
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .llama3 import Eagle3LlamaForCausalLM, PEagleLlamaForCausalLM
from .qwen3 import DFlashQwen3ForCausalLM, Qwen3ForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .qwen3_asr import Qwen3ASRForConditionalGeneration


# Models registered at runtime by plugins (``vllm.general_plugins`` entry
# points), e.g. contrib models and their speculative-decoding draft models.
_EXTERNAL_MODELS: dict[str, type] = {}


def register_neuron_model(architecture: str, model_cls: type) -> None:
    """Register an out-of-tree Neuron model class under an architecture name.

    ``get_models()`` returns it alongside the built-in models, so the Neuron
    worker registers it with vLLM's ``ModelRegistry`` and the speculative
    decoding proposers can resolve it as a draft model. Call it from a
    ``vllm.general_plugins`` hook so it runs in every process; later calls for
    the same architecture replace the class.

    Args:
        architecture: The ``architectures`` entry of the (draft) model's HF
            config, for example ``"HYV3MTPModel"``.
        model_cls: The model class. It must provide ``from_configs``.

    Example:
        >>> register_neuron_model("MyDraftModel", MyDraftModel)  # doctest: +SKIP
    """
    _EXTERNAL_MODELS[architecture] = model_cls


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

    models.extend(_EXTERNAL_MODELS.items())

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
