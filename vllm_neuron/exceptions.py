# SPDX-License-Identifier: Apache-2.0
"""Exceptions shared across the vllm_neuron package.

Kept dependency-free (no torch/vllm/transformers) so it can be imported from
both the ``vllm_neuron`` source tree and the test harness without pulling in
heavy runtime deps.
"""


class CPUCompilationComplete(Exception):
    """Raised when CPU compile mode finishes graph capture/compilation successfully.

    No inference is possible without Neuron devices, so the test should be
    marked as passed (compilation was the goal).
    """

    pass
