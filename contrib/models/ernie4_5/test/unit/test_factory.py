# SPDX-License-Identifier: Apache-2.0
"""Architecture, config and validation test for ERNIE 4.5 dense.

This test reads the ``config.json`` of the pinned checkpoint. It makes sure
that:

- The architecture string that a user registers with vLLM is the string in the
  checkpoint, and it names the factory class of this folder.
- ``Ernie4_5Config`` reads each geometry field of the real config.
- The factory refuses each configuration that the implementation cannot run.

``test/conftest.py`` forces ``VLLM_NEURON_BACKEND=neuron_native`` (the only
route this implementation supports) before pytest imports this module. The
helper modules come from ``test/_local_src.py``, by file location.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/ernie4_5/test/unit -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` helper by file location.

    The tests must keep importing after the folder is copied out of the
    repository, so they resolve ``src/`` and the sibling test helpers from
    their own paths instead of through a ``contrib.models.ernie4_5...``
    namespace path that needs the repository root on ``sys.path``. A plain
    ``import _local_src`` would need this directory on ``sys.path`` and would
    resolve to whichever contrib model got imported first in the process. See
    ``test/_local_src.py``.
    """
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    name = "contrib_ernie4_5_local_src"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the local-src helper at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


_local_src = _load_local_src()
_checkpoint = _local_src.load_helper("checkpoint")
ARCHITECTURE = _checkpoint.ARCHITECTURE
get_ernie_checkpoint = _checkpoint.get_ernie_checkpoint
load_hf_config_dict = _checkpoint.load_hf_config_dict

pytestmark = [pytest.mark.fast]


def test_ernie4_5_registry_config_and_validation():
    """Check the architecture string, the config fields and the validation."""
    src = _local_src.load()
    Ernie4_5Config = src.Ernie4_5Config
    Ernie4_5ForCausalLM = src.Ernie4_5ForCausalLM

    raw = load_hf_config_dict(get_ernie_checkpoint())

    # A contrib model is not in ``vllm_neuron.model.registry.get_models()`` by
    # design, so there is no registry wiring to assert. What a user has to get
    # right is the pair they register: the architecture string of the
    # checkpoint and the factory class of this folder.
    assert raw["architectures"] == [ARCHITECTURE], raw["architectures"]
    assert Ernie4_5ForCausalLM.__name__ == ARCHITECTURE

    config = Ernie4_5Config.from_configs(raw)
    for field in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "use_bias",
        "hidden_act",
        "tie_word_embeddings",
    ):
        assert getattr(config, field) == raw[field], field
    expected_head_dim = raw.get("head_dim") or (
        raw["hidden_size"] // raw["num_attention_heads"]
    )
    assert config.head_dim == expected_head_dim
    assert config.torch_dtype == torch.bfloat16
    assert config.rope_scaling is None

    # The real config must pass the validation.
    Ernie4_5ForCausalLM._validate_config(raw, None)

    rejected = {
        "float16": {**raw, "dtype": "float16"},
        "use_bias": {**raw, "use_bias": True},
        "rope_scaling": {**raw, "rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        "odd_head_dim": {**raw, "head_dim": 127},
        "gelu": {**raw, "hidden_act": "gelu"},
    }
    accepted = []
    for name, bad in rejected.items():
        try:
            Ernie4_5ForCausalLM._validate_config(bad, None)
        except ValueError:
            continue
        accepted.append(name)
    assert not accepted, f"The factory accepted these bad configs: {accepted}"
