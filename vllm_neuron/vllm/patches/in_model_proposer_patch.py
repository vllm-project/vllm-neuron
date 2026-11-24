# SPDX-License-Identifier: Apache-2.0
"""Let a model's own layers be its speculative drafter, with no second checkpoint.

`SpeculativeConfig.__post_init__` assumes a draft *model*. For an MTP method with
`model=None` it points `model` at the target checkpoint, builds a second `ModelConfig` from
it, and then re-detects the method from *that* config's `model_type`. A model whose MTP
stages are layers of the target's own checkpoint fails the re-detection: only `model_type`s
in vLLM's `MTPModelTypes` survive it (plus the handful `hf_config_override` rewrites), so a
port whose `model_type` is its own -- `deepseek_v41` -- ends the chain at

    NotImplementedError: Unsupported speculative method: 'mtp'

before any Neuron code runs. The second `ModelConfig` would be wrong anyway: there is no
separate draft to configure, and nothing on this backend would load it, because the runner
builds the proposer the model registered.

So for exactly the in-model case -- `model is None`, an MTP method name, and a proposer
registered under that name in `vllm_neuron.vllm.spec_decode` -- this sets the fields
downstream actually reads and skips the draft-model machinery. Every other configuration,
including every real draft model, takes the original path untouched.

The method is canonicalised to `"mtp"`. That is what upstream does with every MTP alias, and
it is what `SpeculativeConfig.use_eagle()` keys on -- which is what makes the v1 scheduler
reserve `num_speculative_tokens` lookahead slots per request, and what keeps the method out
of the "unsupported speculative method" lists in `VllmConfig`. `PROPOSERS` stays keyed by the
name the model chose; `resolve_proposer` accepts either spelling.

Installed by `register_proposer`, so a build where no model brings a drafter is upstream
exactly.
"""

from typing import get_args

from vllm.logger import init_logger

logger = init_logger(__name__)

_applied = False
_original_post_init = None


def mtp_aliases() -> frozenset[str]:
    """Every method name vLLM folds into `"mtp"`, including `"mtp"` itself."""
    try:
        from vllm.config.speculative import MTPModelTypes
    except ImportError:  # pragma: no cover - vLLM version without the alias list
        return frozenset({"mtp"})
    return frozenset(get_args(MTPModelTypes))


def _in_model_drafter(cfg) -> bool:
    """Is this a model drafting from its own weights, rather than a draft model?"""
    if cfg.model is not None or cfg.method not in mtp_aliases():
        return False
    from vllm_neuron.vllm.spec_decode import resolve_proposer

    return resolve_proposer(cfg.method) is not None


def _patched_post_init(self):
    if not _in_model_drafter(self):
        return _original_post_init(self)

    if self.target_model_config is None:
        raise ValueError(
            "an in-model speculative drafter needs target_model_config: the draft is the "
            "target model's own layers, so there is nothing else to describe it"
        )
    if self.num_speculative_tokens is None:
        raise ValueError(
            f"speculative method {self.method!r} is registered by the model, but "
            "num_speculative_tokens was not given. It cannot be inferred: there is no "
            "draft model config to read `n_predict` from."
        )

    requested = self.method
    self.method = "mtp"
    self.prompt_lookup_max = 0
    self.prompt_lookup_min = 0
    # The draft *is* the target, so these are the target's. Set rather than left None
    # because `compute_hash` and the eagle3 aux-layer lookup both dereference them.
    self.draft_model_config = self.target_model_config
    self.draft_parallel_config = self.target_parallel_config
    self.draft_tensor_parallel_size = self.target_parallel_config.tensor_parallel_size

    logger.info(
        "Speculative method %r is drafted inside the target model: keeping "
        "num_speculative_tokens=%d and skipping draft-model configuration.",
        requested,
        self.num_speculative_tokens,
    )
    return self


def apply_in_model_proposer_patch() -> None:
    """Idempotent; safe to call from every `register_proposer`."""
    global _applied, _original_post_init
    if _applied:
        return
    try:
        from vllm.config.speculative import SpeculativeConfig
    except ImportError:  # pragma: no cover
        logger.warning(
            "Could not import SpeculativeConfig; an in-model proposer will only work if "
            "vLLM accepts its method name unaided."
        )
        return
    _original_post_init = SpeculativeConfig.__post_init__
    SpeculativeConfig.__post_init__ = _patched_post_init
    _applied = True
    logger.debug("In-model speculative proposer patch applied.")
