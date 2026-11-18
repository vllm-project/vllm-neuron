# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture(scope="session", autouse=True)
def _shutdown_worker_pool():
    """MPExecutor keeps its worker processes alive between tests; stop them at the end."""
    yield
    from vllm_neuron.utils.executor import _worker_pool

    _worker_pool.shutdown_all()
