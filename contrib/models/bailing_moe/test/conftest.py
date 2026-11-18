# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the BailingMoeV2 (inclusionAI/Ling-mini-2.0) tests.

``dist_cleanup`` is vendored from the plugin's ``test/vllm_neuron/conftest.py``.
That conftest only applies under ``test/vllm_neuron/``, so a test in this
folder that asks for the fixture by name would error with "fixture
'dist_cleanup' not found". Vendoring it is what ``contrib/README.md``
prescribes ("if two contrib models need the same helper, copy it into each"),
and it keeps the folder runnable when it is copied out of the repository.
Several of the plugin's own model tests define the same fixture locally, for
example ``test/vllm_neuron/model/qwen3_embedding/modules/test_attention.py``.
"""

import pytest
import torch

from vllm_neuron import envs


@pytest.fixture(scope="function")
def dist_cleanup():
    """Close the Neuron runtime before a test that needs a clean runtime."""
    if not envs.is_native_backend():
        try:
            torch.classes.neuron.Runtime().unsafe_close()
        except Exception as e:
            print(f"Warning: Runtime close failed: {e}")
    yield
