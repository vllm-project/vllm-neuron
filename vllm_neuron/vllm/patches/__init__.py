# SPDX-License-Identifier: Apache-2.0
"""
Centralized patch management for vllm-neuron.

All monkey-patches to upstream vLLM are registered here. Startup-critical
patches use vLLM's general-plugin hook so they are applied in spawned engine,
API, and worker processes.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def apply_patches():
    """Apply all vllm-neuron patches to upstream vLLM."""
    from vllm_neuron.vllm.patches.dp_load_balancer_patch import (
        apply_dp_load_balancer_patch,
    )

    apply_dp_load_balancer_patch()

    from vllm_neuron.vllm.patches.pp_draft_patch import (
        apply_pp_draft_patch,
        apply_pp_mq_depth_patch,
    )

    apply_pp_draft_patch()
    apply_pp_mq_depth_patch()

    # patches/jf_grammar_patch.py wraps a specific
    # StructuredOutputManager and is applied from NeuronScheduler.__init__.

    # patches/shm_reader_spin_patch.py is applied from NeuronWorker.init_device and from the
    # Neuron run_engine_core wrapper, not here: it has to take effect before a process builds
    # a MessageQueue reader, and importing shm_broadcast this early can raise. It is
    # idempotent, so both sites are safe.
