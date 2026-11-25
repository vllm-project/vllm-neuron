# SPDX-License-Identifier: Apache-2.0
"""Stable import path for the Qwen3.5 implementation class.

The runner's model is ``model_bf16.Qwen3_5ForCausalLM`` (``factory`` selects
it). External tools import the class from here, for example a vLLM worker
extension that refreshes the resident weights in place for RL training. It
checks ``type(runner.model._orig_mod) is Qwen3_5ForCausalLM``, builds a staging
copy with ``Qwen3_5ForCausalLM(model.config)`` and calls ``load_weights`` on it.
Keep this name pointing at the class the factory returns.

Example:
    >>> from vllm_neuron.model.qwen3_5.model import Qwen3_5ForCausalLM  # doctest: +SKIP
"""

from vllm_neuron.model.qwen3_5.model_bf16 import Qwen3_5ForCausalLM

__all__ = ["Qwen3_5ForCausalLM"]
