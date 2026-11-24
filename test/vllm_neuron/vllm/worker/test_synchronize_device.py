# SPDX-License-Identifier: Apache-2.0
"""The Neuron worker waits for its device without torch.accelerator.

vLLM's EngineCore ends every pause with ``collective_rpc("synchronize_device")``.
The WorkerBase default calls ``torch.accelerator.synchronize()``, which the
Neuron backend does not implement.

CPU-only; no Neuron device.
"""

import types

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker


@pytest.fixture
def neuron_module(monkeypatch):
    """Give torch a fake ``torch.neuron`` that counts synchronize calls."""
    calls = []
    module = getattr(torch, "neuron", None)
    if module is None:
        module = types.ModuleType("torch.neuron")
        monkeypatch.setattr(torch, "neuron", module, raising=False)
    monkeypatch.setattr(
        module, "synchronize", lambda device=None: calls.append(device), raising=False
    )
    return calls


@pytest.fixture
def device_mode(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)


def _worker() -> NeuronWorker:
    return object.__new__(NeuronWorker)


def test_synchronize_device_waits_for_the_neuron_device(
    neuron_module, device_mode, monkeypatch
):
    def fail(*args, **kwargs):
        raise RuntimeError(
            "Backend doesn't support synchronizing all streams on device."
        )

    monkeypatch.setattr(torch.accelerator, "synchronize", fail)
    _worker().synchronize_device()
    assert neuron_module == [None]


@pytest.mark.parametrize("mode", ["VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE"])
def test_synchronize_device_is_a_no_op_without_a_device(
    neuron_module, monkeypatch, mode
):
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.setenv(mode, "1")
    _worker().synchronize_device()
    assert neuron_module == []
