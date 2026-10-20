# SPDX-License-Identifier: Apache-2.0
"""Prompt analysis plugins for the accuracy debugger.

Each plugin implements a single analysis step (logit_val, kv_cache, etc.)
against a loaded vLLM LLM instance and pre-computed reference goldens.
"""

from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.base import PromptPlugin
from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.logit_val import (
    LogitValPlugin,
)
from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.kv_cache import KvCachePlugin
from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.tensor_compare import (
    TensorComparePlugin,
)

PLUGIN_REGISTRY: dict[str, type[PromptPlugin]] = {
    "logit_val": LogitValPlugin,
    "kv_cache": KvCachePlugin,
    "tensor_compare": TensorComparePlugin,
}

__all__ = [
    "PromptPlugin",
    "LogitValPlugin",
    "KvCachePlugin",
    "TensorComparePlugin",
    "PLUGIN_REGISTRY",
]
