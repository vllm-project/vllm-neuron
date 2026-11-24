# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for the Ouro contrib tests.

Three things live here:

* This directory goes on ``sys.path``, so that ``test/unit``,
  ``test/module`` and ``test/integration`` can import ``_local_src`` (the
  loader for this model's ``src/``), ``ouro_reference`` and
  ``ouro_test_utils``. pytest puts the directory of a test file on
  ``sys.path``, not the directory above it.
* ``dist_cleanup`` destroys the default process group after a test, so
  that the next test can create its own.
* ``_worker_pool_cleanup`` shuts down the plugin's ``MPExecutor`` worker pool
  at the end of the session.
"""

import os
import sys
from pathlib import Path

import pytest

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)


@pytest.fixture(scope="function")
def dist_cleanup():
    """Destroy the default process group that a test created, if any."""
    yield
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="session", autouse=True)
def _worker_pool_cleanup():
    """Shut down the plugin's ``MPExecutor`` worker pool when the session ends.

    ``MPExecutor`` keeps its worker processes in a module-level pool so later
    tests can reuse them. Those workers outlive the last test, so without this
    shutdown pytest prints its summary and then never exits.
    """
    pid = os.getpid()
    yield
    if os.getpid() == pid:
        from vllm_neuron.utils.executor import _worker_pool

        _worker_pool.shutdown_all()
