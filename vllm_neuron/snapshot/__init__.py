# SPDX-License-Identifier: Apache-2.0
"""Regression-artifact snapshot capture for the Neuron inference stack.

Owns configuration resolution, proactive capture selection (call index, token,
request), per-forward identity, and the metadata tag written beside the
device->host input dump.
"""

from vllm_neuron.snapshot.config import SnapshotConfig

__all__ = ["SnapshotConfig"]
