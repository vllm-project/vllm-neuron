# SPDX-License-Identifier: Apache-2.0
"""Hy3 (``HYV3ForCausalLM``) static-FP8 model for vLLM on Neuron (contrib)."""

from .config import HYV3Config
from .factory import HYV3ForCausalLM

__all__ = ["HYV3Config", "HYV3ForCausalLM", "register"]

ARCHITECTURE = "HYV3ForCausalLM"
MTP_ARCHITECTURE = "HYV3MTPModel"


def register() -> None:
    """Register Hy3 with vLLM. Safe to call more than once.

    vLLM ships its own (GPU) ``HYV3ForCausalLM``, and the engine core and the
    Neuron workers run in spawned processes, so the registration has to run
    in every process. Installing this package (``pip install -e
    contrib/models/hy_v3``) exposes this function as a ``vllm.general_plugins``
    entry point, which vLLM calls in each process at startup.

    It also adds ``"fp8"`` (the checkpoint's ``quant_method``) to the Neuron
    platform's supported quantizations. The model consumes the FP8 weights and
    scales itself, so vLLM's own FP8 linear methods are never used. With a
    vllm-neuron that supports MTP it registers the MTP draft model as well.
    """
    from vllm import ModelRegistry

    from vllm_neuron.vllm.platform import NeuronPlatform

    if "fp8" not in NeuronPlatform.supported_quantization:
        NeuronPlatform.supported_quantization.append("fp8")
    ModelRegistry.register_model(ARCHITECTURE, HYV3ForCausalLM)

    # MTP draft (--speculative-config '{"method": "mtp", ...}'), resolved by
    # the Neuron MTP proposer. Needs a vllm-neuron with register_neuron_model.
    try:
        from vllm_neuron.model.registry import register_neuron_model
    except ImportError:
        return
    from .model_mtp import HYV3MTPForCausalLM

    register_neuron_model(MTP_ARCHITECTURE, HYV3MTPForCausalLM)
