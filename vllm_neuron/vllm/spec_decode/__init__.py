# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding module for vllm_neuron backend.

`eagle3` is built in: a separate draft model, loaded and compiled beside the target. A model
whose draft path is *inside itself* -- DeepSeek V4.1 Flash's MTP stages are checkpoint layers
40-42 of the same weights -- has nothing for `EagleProposer` to load, so it supplies its own
proposer instead and registers it here at import time.

The registry exists so that proposer can live with the model it belongs to rather than in this
package. That is not only tidiness: the integrate loop may edit `vllm_neuron/model/<model>/`
and may not edit this directory, so a drafter kept here is one no agent can work on.

A proposer is duck-typed. `NeuronModelRunner` calls `num_speculative_tokens`, `load_model`,
`warmup`, `graph_extract`, `propose`, `validate_same_kv_cache_group`, and reads `.model`; see
`eagle.EagleProposer` for the reference shape. One whose drafts come from the target model also
gets `target_model_provider`, a callable returning the loaded target, since the runner builds
the proposer before any model exists.
"""

from typing import Any

#: Speculative method name -> proposer class, for methods not built into this package.
PROPOSERS: dict[str, Any] = {}


def register_proposer(method: str, proposer_cls: Any) -> None:
    """Make `proposer_cls` available as `speculative_config.method == method`.

    Called from a model package's `__init__`, which the platform plugin imports when the model
    is registered, so the entry exists before the runner reads `speculative_config`.
    """
    PROPOSERS[method] = proposer_cls


def get_proposer(method: str) -> Any | None:
    return PROPOSERS.get(method)
