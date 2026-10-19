# SPDX-License-Identifier: Apache-2.0
"""
Patch to support grammar rollback for xgrammar with jump-forward decoding.
We must rollback the grammar after the JF NEFF is used or else we will double accept
jump-forward tokens. Trying to add logic to only accept each JF token once, without having
to rollback would cause upstream changes to vLLM.
"""

import logging

logger = logging.getLogger(__name__)


def apply_jf_grammar_rollback_patch(structured_output_manager, jf_max: int) -> None:
    """Wrap ``grammar_init`` so compiled matchers get ``jf_max`` rollback budget.

    Args:
        structured_output_manager: The manager owned by the NeuronScheduler.
        jf_max: Maximum jump-forward tokens per request (the rollback budget).
    """
    mgr = structured_output_manager
    if mgr is None or getattr(mgr, "_jf_rollback_patched", False):
        return
    orig_create_grammar = mgr._create_grammar

    # xgrammar sets the matcher's max_rollback_tokens from
    # backend.num_speculative_tokens when it builds the GrammarMatcher inside
    # compile_grammar. We must widen that budget to jf_max BEFORE compile runs,
    # otherwise the matcher can only roll back num_spec_tokens (0 when JF is on).
    # _create_grammar is the caller of compile_grammar and always runs after
    # grammar_init created the backend — inline in sync mode, on the executor
    # thread in async mode. Setting the budget here (same thread, right before
    # the read) is race-free, unlike setting it after grammar_init returns.
    def create_grammar_with_jf_rollback(request):
        backend = getattr(mgr, "backend", None)
        if backend is not None:
            cur = getattr(backend, "num_speculative_tokens", 0)
            if cur < jf_max:
                backend.num_speculative_tokens = jf_max
        return orig_create_grammar(request)

    mgr._create_grammar = create_grammar_with_jf_rollback
    mgr._jf_rollback_patched = True
    logger.info(
        "JF: patched structured_output_manager._create_grammar to set "
        "num_speculative_tokens=%d (matcher rollback budget) before compile.",
        jf_max,
    )
