# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for the OLMo2 contrib tests.

Two things live here:

* This directory goes on ``sys.path``, so that ``test/unit``,
  ``test/module`` and ``test/integration`` can import ``_local_src`` (the
  loader for this model's ``src/``) and ``olmo2_test_utils``. pytest puts the
  directory of a test file on ``sys.path``, not the directory above it.
* ``dist_cleanup`` is a copy of the fixture in the plugin's
  ``test/vllm_neuron/conftest.py``. A test under ``contrib/`` does not pick up
  that conftest, and ``contrib/README.md`` allows copying a shared test helper
  into the model folder, so the fixture is vendored here.
"""

import sys
from pathlib import Path

import pytest

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)


@pytest.fixture(scope="function")
def dist_cleanup():
    """Close the Neuron runtime before a test that starts its own workers."""
    import torch

    from vllm_neuron import envs

    if not envs.is_native_backend():
        try:
            torch.classes.neuron.Runtime().unsafe_close()
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Runtime close failed: {e}")
    yield
