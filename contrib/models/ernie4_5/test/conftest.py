# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for the ERNIE 4.5 contrib tests.

This implementation runs on the ``neuron_native`` route only, so the backend is
forced here, once for the whole model folder, instead of once at the top of
every test module. pytest imports a conftest before the test modules beneath
it, so this runs before anything in ``vllm_neuron`` reads the variable. The
in-tree port used a session-wide ``conftest.py`` for the same purpose; a
contrib model's own test tree is the scope the contrib layout allows for it.

Nothing else belongs here. In particular this file does not touch
``sys.path``: ``test/_local_src.py`` resolves this model's ``src/`` package and
the sibling test helpers by file location, so the tests keep working from any
working directory and after the folder is copied out of the repository.
"""

import os

import pytest

os.environ["VLLM_NEURON_BACKEND"] = "neuron_native"


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
