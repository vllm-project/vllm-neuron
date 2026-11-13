# SPDX-License-Identifier: Apache-2.0
"""Neuron weight-transfer backend.

Implements vLLM's ``WeightTransferEngine`` / ``TrainerWeightTransferEngine``
contracts for AWS Trainium/Neuron so an RL trainer can push updated weights
into a *serving* vLLM Neuron rollout in place, without restarting the engine
or invalidating the compiled graph.

The backend is registered under the name ``"neuron"`` in vLLM's lazy
``WeightTransferEngineFactory`` / ``WeightTransferTrainerFactory`` from
``vllm_neuron/vllm/platform.py`` (see ``pre_register_and_update``), so no
upstream vLLM patch is required.
"""

from vllm_neuron.vllm.weight_transfer.neuron_engine import (
    NeuronWeightTransferEngine,
    NeuronWeightTransferInitInfo,
    NeuronWeightTransferUpdateInfo,
)
from vllm_neuron.vllm.weight_transfer.neuron_trainer import (
    NeuronTrainerInitInfo,
    NeuronTrainerWeightTransferEngine,
)

__all__ = [
    "NeuronWeightTransferEngine",
    "NeuronWeightTransferInitInfo",
    "NeuronWeightTransferUpdateInfo",
    "NeuronTrainerWeightTransferEngine",
    "NeuronTrainerInitInfo",
]
