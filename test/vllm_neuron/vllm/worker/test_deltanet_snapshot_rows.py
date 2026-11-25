# SPDX-License-Identifier: Apache-2.0
"""The runner gives each prefill step the snapshot rows that the scheduler chose.

The scheduler puts ``deltanet_snapshot_ops`` (request id -> ``(restore_row,
save_row)``) on the scheduler output. The runner keeps the map of the step and
builds the ``deltanet_snapshot_rows`` input of the prefill graph from it. Row 0
means "none".

CPU-only; no Neuron device.
"""

from types import SimpleNamespace

import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner


def _runner(req_ids: list[str]) -> NeuronModelRunner:
    """Build the smallest runner that the builder reads. Skip `__init__`."""
    runner = object.__new__(NeuronModelRunner)
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids))
    return runner


def test_the_rows_of_the_prefill_request_are_given():
    runner = _runner(["req-b"])
    runner._deltanet_snapshot_ops = {"req-a": (1, 0), "req-b": (2, 3)}
    assert runner._build_deltanet_snapshot_rows().tolist() == [2, 3]


def test_a_request_with_no_op_gets_row_0():
    runner = _runner(["req-c"])
    runner._deltanet_snapshot_ops = {"req-a": (1, 0)}
    assert runner._build_deltanet_snapshot_rows().tolist() == [0, 0]


def test_a_runner_without_the_map_gets_row_0():
    """A test can build a runner that skips __init__, so the map can be missing."""
    runner = _runner(["req-a"])
    assert runner._build_deltanet_snapshot_rows().tolist() == [0, 0]


def test_the_rows_are_int64():
    runner = _runner(["req-a"])
    runner._deltanet_snapshot_ops = {"req-a": (1, 2)}
    assert runner._build_deltanet_snapshot_rows().dtype == torch.long
