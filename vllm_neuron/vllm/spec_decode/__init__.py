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

    Registering also installs the patch that lets vLLM build a `SpeculativeConfig` for a
    drafter with no draft model — see `patches/in_model_proposer_patch.py` for why that is
    needed and why it is armed from here rather than unconditionally.
    """
    PROPOSERS[method] = proposer_cls

    from vllm_neuron.vllm.patches.in_model_proposer_patch import (
        apply_in_model_proposer_patch,
    )

    apply_in_model_proposer_patch()


def get_proposer(method: str) -> Any | None:
    return PROPOSERS.get(method)


def resolve_proposer(method: str) -> Any | None:
    """The proposer for `method`, accepting any spelling vLLM folds into the same method.

    `SpeculativeConfig` rewrites every name in its `MTPModelTypes` list to `"mtp"` — logging
    that the specific spelling is deprecated — so by the time the runner reads
    `speculative_config.method`, a model that registered `"deepseek_mtp"` no longer matches it
    by name. Both spellings resolve to the same proposer here.
    """
    found = PROPOSERS.get(method)
    if found is not None:
        return found

    from vllm_neuron.vllm.patches.in_model_proposer_patch import mtp_aliases

    aliases = mtp_aliases()
    if method in aliases:
        for alias in aliases:
            found = PROPOSERS.get(alias)
            if found is not None:
                return found
    return None


def attach_speculative_config(vllm_config, method: str, num_speculative_tokens: int) -> Any:
    """Give `vllm_config` a speculative config for an in-model drafter. Returns it.

    For a model package to use from its `check_and_update_config` hook, which is the one point
    where it can see a `VllmConfig` — and which runs *after* `VllmConfig.__post_init__` has
    already made its speculative-decoding decisions. Only one of those decisions matters here
    and it is redone: `_set_max_num_scheduled_tokens` holds back the token slots a drafter may
    add to a batch, and it is a no-op when no speculative config exists yet.

    Raises if nothing is registered for `method`, because the alternative is an engine that
    builds fine and then fails several minutes later inside the runner.
    """
    from vllm.config.speculative import SpeculativeConfig

    if resolve_proposer(method) is None:
        raise ValueError(
            f"no speculative proposer registered for method {method!r}. "
            f"Registered: {sorted(PROPOSERS) or 'none'}. Call register_proposer() from the "
            "model package's __init__ before its platform hook runs."
        )

    vllm_config.speculative_config = SpeculativeConfig(
        target_model_config=vllm_config.model_config,
        target_parallel_config=vllm_config.parallel_config,
        method=method,
        num_speculative_tokens=num_speculative_tokens,
    )

    recompute = getattr(vllm_config, "_set_max_num_scheduled_tokens", None)
    if recompute is not None:
        recompute()

    return vllm_config.speculative_config
