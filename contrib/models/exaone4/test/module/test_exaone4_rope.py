# SPDX-License-Identifier: Apache-2.0
"""Module test for the EXAONE 4.0 rotary embedding.

This is a module-level test in the sense of
``docs/design/accuracy/module_test_guidelines.md``: it validates one component
of the model in isolation against its HuggingFace reference, before the full
model is assembled. The component is ``Exaone4RotaryEmbedding`` from
``../../src/model.py``, and the reference is
``transformers.models.exaone4.modeling_exaone4.Exaone4RotaryEmbedding``.

RoPE carries no checkpoint weights, so the test needs only the checkpoint's
``config.json`` — it is the Llama3 frequency scaling (factor 16, original
context 8192, theta 1.0e6) that the comparison exercises, and those values come
from the config. The test reads the real config rather than a hand-written one
so that a config-mapping change cannot silently pass it. Set
``EXAONE4_CHECKPOINT_PATH`` to the checkpoint directory; without it the test
skips.

The comparison is two-way against an FP32 reference, not the three-way rule of
the guidelines: both implementations evaluate the same closed-form expression
in FP32 on CPU, so there is no dtype-error reference to normalize against. The
tolerances are therefore tight (1e-6 relative on the inverse frequencies, 1e-4
absolute on cos/sin), not the 1% band that a BF16 matmul component would need.

Run with:
    VLLM_NEURON_CPU_MODE=1 EXAONE4_CHECKPOINT_PATH=/path/to/EXAONE-4.0-32B \
        pytest contrib/models/exaone4/test/module/test_exaone4_rope.py -v
"""

import os
import sys

import pytest
import torch


def _load_local_src():
    """Load this model's ``test/_local_src.py`` helper by file location.

    Loading the helper by location, under a name unique to this model, keeps
    two contrib models collected in one pytest run from resolving each other's
    helper (a plain ``import _local_src`` would return whichever copy Python
    cached first). Nothing is added to ``sys.path``.
    """
    import importlib.util
    from pathlib import Path

    name = "contrib_exaone4_local_src"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_local_src = _load_local_src()

pytestmark = [pytest.mark.fast]

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-32B"

# Positions on both sides of the 4,096-token sliding window and of the 8,192
# original context length, where the Llama3 scaling changes regime.
POSITIONS = (0, 1, 127, 4095, 4096, 8191, 8192)


@pytest.fixture(scope="module")
def checkpoint_dir():
    """Return the local checkpoint directory, or skip without one.

    The port resolved the checkpoint through a shared test helper that does
    not ship here. A contrib model keeps its tests self-contained, so this
    copy reads ``EXAONE4_CHECKPOINT_PATH`` and skips when it is not a usable
    checkpoint directory.
    """
    path = os.environ.get("EXAONE4_CHECKPOINT_PATH")
    if not path or not os.path.isfile(os.path.join(path, "config.json")):
        pytest.skip(
            f"Set EXAONE4_CHECKPOINT_PATH to a local directory holding the "
            f"{MODEL_ID} checkpoint."
        )
    return path


@pytest.fixture(scope="module")
def hf_config(checkpoint_dir):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(checkpoint_dir)


@pytest.fixture(scope="module")
def model_config(hf_config):
    from vllm_neuron.model.neuron_config import NeuronConfig

    config_module = _local_src.load("config")

    return config_module.Exaone4Config.from_configs(hf_config, NeuronConfig())


def test_rope_matches_hf(hf_config, model_config):
    """The Llama3-scaled RoPE must agree with the HF RoPE implementation."""
    from transformers.models.exaone4.modeling_exaone4 import (
        Exaone4RotaryEmbedding as HFRotary,
    )

    Exaone4RotaryEmbedding = _local_src.load("model").Exaone4RotaryEmbedding

    ours = Exaone4RotaryEmbedding(model_config)
    ref = HFRotary(hf_config)
    assert ref.attention_scaling == 1.0

    inv_ours = ours._compute_inv_freq(torch.device("cpu"))
    torch.testing.assert_close(inv_ours, ref.inv_freq.float(), rtol=1e-6, atol=0.0)

    positions = torch.tensor(POSITIONS, dtype=torch.int64)
    cos, sin = ours(positions, torch.device("cpu"), torch.float32)
    x = torch.zeros(1, positions.numel(), hf_config.hidden_size)
    ref_cos, ref_sin = ref(x, positions[None, :])
    torch.testing.assert_close(cos, ref_cos[0], rtol=0.0, atol=1e-4)
    torch.testing.assert_close(sin, ref_sin[0], rtol=0.0, atol=1e-4)
