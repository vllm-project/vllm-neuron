# SPDX-License-Identifier: Apache-2.0
"""Ouro registration and config tests (CPU, no weights).

1. ``test_plugin_register``: ``src/plugin.py``'s ``register()`` registers
   the ``"ouro"`` model type with ``AutoConfig`` and the ``OuroForCausalLM``
   architecture with vLLM.
2. ``test_installed_plugin_resolves``: with ``src/`` importable as
   ``ouro_neuron`` (what ``pip install -e`` gives), vLLM resolves and inspects
   the lazy registration in a fresh interpreter, and the entry point in
   ``pyproject.toml`` names ``register()``.
3. ``test_config_parse``: the ``config.json`` of the pinned revision parses
   without remote code, through transformers and through vLLM, and gives the
   expected Neuron config.
4. ``test_factory_validation``: the factory accepts the pinned config and
   stops on each configuration that the implementation does not support.
5. ``test_vllm_model_protocol``: the factory satisfies vLLM's model protocol.

Only ``config.json`` (1.5 kB) is downloaded. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/ouro/test/unit/ -v
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import torch

from _local_src import src_module, src_package
from ouro_test_utils import ARCHITECTURE, EXPECTED_CONFIG, pinned_config_dir

_FOLDER = Path(__file__).resolve().parents[2]
_PACKAGE = "ouro_neuron"


@pytest.fixture(scope="module")
def config_dir():
    return pinned_config_dir()


def test_plugin_register():
    """register() adds the "ouro" config type and the architecture."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from vllm.model_executor.models.registry import ModelRegistry

    src_module("plugin").register()

    assert ARCHITECTURE in ModelRegistry.get_supported_archs()
    registered = ModelRegistry.models[ARCHITECTURE]
    # The lazy form keeps register() cheap: vLLM imports the factory only
    # when it builds the model.
    assert getattr(registered, "module_name", None) == f"{_PACKAGE}.factory"
    assert getattr(registered, "class_name", None) == ARCHITECTURE

    assert "ouro" in CONFIG_MAPPING
    assert CONFIG_MAPPING["ouro"].__name__ == "OuroConfig"
    assert CONFIG_MAPPING["ouro"].model_type == "ouro"

    # The registration is safe to repeat (vLLM calls the plugin in every
    # process, and a process can load it more than once).
    src_module("plugin").register()
    assert ARCHITECTURE in ModelRegistry.get_supported_archs()


def test_installed_plugin_resolves(tmp_path):
    """The installed package resolves, and the entry point names register()."""
    with open(_FOLDER / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    entry_points = pyproject["project"]["entry-points"]["vllm.general_plugins"]
    assert entry_points == {_PACKAGE: f"{_PACKAGE}.plugin:register"}
    assert pyproject["tool"]["setuptools"]["package-dir"] == {_PACKAGE: "src"}

    # The installed layout: src/ importable as ``ouro_neuron``. A symlink
    # gives the same layout as ``pip install -e`` without changing the
    # environment. vLLM inspects a lazy registration in a subprocess, which
    # inherits PYTHONPATH.
    (tmp_path / _PACKAGE).symlink_to(_FOLDER / "src", target_is_directory=True)
    script = f"""
from {_PACKAGE}.plugin import register
from vllm.model_executor.models.registry import ModelRegistry

register()
cls = ModelRegistry._try_load_model_cls({ARCHITECTURE!r})
info = ModelRegistry._try_inspect_model_cls({ARCHITECTURE!r})
print("RESULT", cls.__module__, cls.__name__, info.is_text_generation_model)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT")]
    assert proc.returncode == 0 and lines, (
        f"The subprocess failed with exit code {proc.returncode}.\n"
        f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
    assert lines[-1].split()[1:] == [f"{_PACKAGE}.factory", ARCHITECTURE, "True"]


def test_config_parse(config_dir):
    """The pinned config.json parses without remote code."""
    from transformers import AutoConfig
    from vllm.transformers_utils.config import get_config

    hf_config_cls = src_module("hf_config").OuroConfig
    OuroConfig = src_module("config").OuroConfig

    hf_config = AutoConfig.from_pretrained(config_dir)
    assert isinstance(hf_config, hf_config_cls)
    assert hf_config.model_type == "ouro"
    for key, value in EXPECTED_CONFIG.items():
        assert getattr(hf_config, key) == value, f"hf_config.{key} is wrong."

    # vLLM parses the config the same way, with trust_remote_code off.
    vllm_config = get_config(config_dir, trust_remote_code=False)
    assert isinstance(vllm_config, hf_config_cls)

    config = OuroConfig.from_configs(hf_config)
    assert config.total_ut_steps == 4
    assert config.num_kv_layers == 96
    assert config.early_exit_threshold == 1.0
    assert config.torch_dtype == torch.bfloat16
    assert config.tie_word_embeddings is False
    assert config.rms_norm_eps == pytest.approx(1e-6)
    assert config.rope_theta == pytest.approx(1e6)
    assert config.head_dim == 128

    # A config path gives the same result as a PretrainedConfig.
    from_path = OuroConfig.from_configs(os.path.join(config_dir, "config.json"))
    assert from_path.num_kv_layers == config.num_kv_layers
    assert from_path.rope_theta == config.rope_theta


def test_factory_validation(config_dir):
    """The factory accepts the pinned config and rejects unsupported ones."""
    from transformers import AutoConfig

    Factory = src_package().OuroForCausalLM
    hf_config = AutoConfig.from_pretrained(config_dir)
    Factory._validate_config(hf_config, None)

    unsupported = [
        {"torch_dtype": torch.float16},
        {"total_ut_steps": 0},
        {"early_exit_step": 2},
        {"rope_scaling": {"rope_type": "yarn", "factor": 2.0}},
        {"rope_parameters": {"rope_type": "linear", "factor": 2.0}},
        {"use_sliding_window": True},
        {"layer_types": ["sliding_attention"] * 24},
        {"hidden_act": "gelu"},
    ]
    base = hf_config.to_dict()
    base["torch_dtype"] = torch.bfloat16
    for override in unsupported:
        bad = dict(base, **override)
        with pytest.raises(ValueError):
            Factory._validate_config(bad, None)


def test_vllm_model_protocol():
    """The factory is a vLLM text-generation model."""
    from vllm.model_executor.models.interfaces_base import (
        is_pooling_model,
        is_text_generation_model,
        is_vllm_model,
    )

    Factory = src_package().OuroForCausalLM
    assert getattr(src_package(), ARCHITECTURE) is Factory
    assert is_vllm_model(Factory)
    assert is_text_generation_model(Factory)
    assert not is_pooling_model(Factory)


def test_kv_layer_names():
    """Each UT step of each layer has its own KV cache layer name."""
    kv_layer_name = src_module("model").kv_layer_name
    names = [kv_layer_name(u, i, 24) for u in range(4) for i in range(24)]
    assert names == [f"layers.{n}.self_attn" for n in range(96)]
