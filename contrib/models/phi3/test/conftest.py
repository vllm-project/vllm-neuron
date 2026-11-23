# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for the Phi3 contrib tests."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _worker_pool_cleanup():
    """Shut down the plugin's ``MPExecutor`` worker pool when the session ends.

    ``MPExecutor`` keeps its worker processes in a module-level pool so later
    tests can reuse them. Those workers outlive the last test, so without this
    shutdown pytest prints its summary and then never exits. The plugin's own
    test suite shuts the pool down the same way.
    """
    pid = os.getpid()
    yield
    if os.getpid() == pid:
        from vllm_neuron.utils.executor import _worker_pool

        _worker_pool.shutdown_all()
