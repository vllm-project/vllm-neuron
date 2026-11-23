# SPDX-License-Identifier: Apache-2.0
"""OLMo loading tests (CPU).

These tests check the loading path of ``allenai/OLMo-1B-hf`` at the recorded
topology (TP 8) without the 4.7 GB of FP32 weights:

1. ``test_vllm_model_protocol``: the factory satisfies vLLM's model protocol
   for a text-generation model.
2. ``test_plugin_entry_point``: ``pyproject.toml`` points vLLM's
   ``vllm.general_plugins`` group at ``src/plugin.py``, and ``register()``
   maps the checkpoint's architecture string to the factory.
3. ``test_config_parse`` and ``test_factory_validation``: the config parse of
   the real ``config.json`` and the factory checks.
4. ``test_tp_validation``: the TP sizes that the model accepts and rejects.
5. ``test_checkpoint_mapping_coverage``: each model parameter has a mapping,
   and the mappings use each checkpoint tensor (read from the safetensors
   header) exactly one time, with the expected shape.
6. ``test_tp8_weight_loader_shards``: the loaders give the expected TP 8 shard
   of each weight, on a small random checkpoint with the OLMo tensor names.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo/test/unit/test_loading.py -v
"""

import copy
import tomllib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from _local_src import src_module, src_package
from olmo_test_utils import (
    WORLD_SIZE,
    checkpoint_tensor_shapes,
    config_dir,
    read_config_json,
)

ARCHITECTURE = "OlmoForCausalLM"
# A rank in the second half of the TP group. Its shards start at a non-zero
# offset, so a wrong offset gives a wrong shard.
SHARD_RANK = 5


class _FakeTPGroup:
    """Give the TP size and the rank without a process group."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = None


class MockSlice:
    """Mock checkpoint slice with __getitem__ and get_shape."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]

    def get_shape(self):
        return self.tensor.shape


def _patch_tp_group(monkeypatch, world_size: int, rank: int) -> None:
    monkeypatch.setattr(
        src_module("model"), "get_tp_group", lambda: _FakeTPGroup(world_size, rank)
    )


@pytest.fixture(scope="module")
def checkpoint_dir() -> str:
    return config_dir()


@pytest.fixture(scope="module")
def hf_config(checkpoint_dir):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(checkpoint_dir)


# =============================================================================
# 1-2. vLLM model protocol and plugin registration
# =============================================================================


def test_vllm_model_protocol():
    """The factory is a vLLM text-generation model (vLLM 0.29 protocol)."""
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
        is_vllm_model,
    )

    factory = getattr(src_package(), ARCHITECTURE)
    assert is_vllm_model(factory)
    assert is_text_generation_model(factory)


def test_plugin_entry_point(checkpoint_dir, monkeypatch):
    """The general-plugin entry point registers the factory for the checkpoint."""
    from vllm import ModelRegistry

    assert read_config_json(checkpoint_dir)["architectures"] == [ARCHITECTURE]

    folder = Path(__file__).resolve().parents[2]
    with open(folder / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)
    entry_points = project["project"]["entry-points"]["vllm.general_plugins"]
    assert entry_points == {"olmo_neuron": "olmo_neuron.plugin:register"}
    assert project["tool"]["setuptools"]["package-dir"] == {"olmo_neuron": "src"}

    # register() must not leak into the other tests of this session.
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    src_module("plugin").register()
    registered = ModelRegistry.models[ARCHITECTURE]
    # The installed package is ``olmo_neuron``, which maps to ``src/``.
    assert (registered.module_name, registered.class_name) == (
        "olmo_neuron.factory",
        ARCHITECTURE,
    )
    assert hasattr(src_module("factory"), ARCHITECTURE)
    assert getattr(src_package(), ARCHITECTURE) is getattr(
        src_module("factory"), ARCHITECTURE
    )


# =============================================================================
# 3. Config parse and factory validation
# =============================================================================


def test_config_parse(checkpoint_dir, hf_config):
    """The config dataclass reads every field of the real config.json."""
    OlmoConfig = src_module("config").OlmoConfig
    raw = read_config_json(checkpoint_dir)

    # Both the HF config object (transformers 5 moves rope_theta into
    # rope_parameters) and the raw JSON path must give the same fields.
    for source in (hf_config, str(Path(checkpoint_dir) / "config.json")):
        config = OlmoConfig.from_configs(source, None)
        for field in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "tie_word_embeddings",
            "clip_qkv",
            "attention_bias",
        ):
            assert getattr(config, field) == raw[field], field
        assert config.head_dim == raw["hidden_size"] // raw["num_attention_heads"]
        # HF OlmoLayerNorm uses a fixed epsilon that the config does not hold.
        assert config.layer_norm_eps == 1e-5
        # The checkpoint stores FP32. The port computes in BF16.
        assert config.torch_dtype == torch.bfloat16


def test_factory_validation(hf_config):
    """The factory accepts the real config and rejects unsupported features."""
    factory = getattr(src_package(), ARCHITECTURE)
    factory._validate_config(hf_config, None)

    for field, value in (
        ("clip_qkv", 8.0),
        ("attention_bias", True),
        ("rope_scaling", {"rope_type": "linear", "factor": 2.0}),
        ("torch_dtype", torch.float16),
    ):
        bad = copy.deepcopy(hf_config)
        setattr(bad, field, value)
        with pytest.raises(ValueError, match="OLMo"):
            factory._validate_config(bad, None)


# =============================================================================
# 4. TP size checks
# =============================================================================


def test_tp_validation(monkeypatch, hf_config):
    """TP 1, 2, 4 and 8 are legal. TP 3 and TP 16 are rejected at construction."""
    OlmoConfig = src_module("config").OlmoConfig
    OlmoForCausalLM = src_module("model").OlmoForCausalLM
    config = OlmoConfig.from_configs(hf_config, None)

    for world_size in (1, 2, 4, 8):
        _patch_tp_group(monkeypatch, world_size, 0)
        with torch.device("meta"):
            OlmoForCausalLM(config)
    # 16 heads: TP 3 does not divide them, and TP 16 gives one head per rank,
    # while the decode kernel needs an even head count on each rank.
    for world_size in (3, 16):
        _patch_tp_group(monkeypatch, world_size, 0)
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            with torch.device("meta"):
                OlmoForCausalLM(config)


# =============================================================================
# 5. Checkpoint mapping coverage
# =============================================================================


def test_checkpoint_mapping_coverage(monkeypatch, hf_config):
    """Each parameter maps to a checkpoint tensor, and each tensor is used once."""
    OlmoConfig = src_module("config").OlmoConfig
    OlmoForCausalLM = src_module("model").OlmoForCausalLM
    config = OlmoConfig.from_configs(hf_config, None)

    _patch_tp_group(monkeypatch, WORLD_SIZE, 0)
    with torch.device("meta"):
        model = OlmoForCausalLM(config)
    mappings = OlmoForCausalLM.checkpoint_mappings(
        config.num_hidden_layers, config.tie_word_embeddings
    )
    param_names = {name for name, _ in model.named_parameters()}
    assert param_names == set(mappings), (
        f"Parameters without a mapping: {sorted(param_names - set(mappings))}. "
        f"Mappings without a parameter: {sorted(set(mappings) - param_names)}."
    )
    # The checkpoint has no norm tensors, so the model must have no norm weight.
    assert not any("norm" in name for name in param_names)

    referenced: list[str] = []
    for keys in mappings.values():
        referenced.extend(keys if isinstance(keys, list) else [keys])
    # The tied lm_head reads the embedding a second time. Every other key is
    # read exactly once.
    assert config.tie_word_embeddings
    assert referenced.count("model.embed_tokens.weight") == 2
    assert len(set(referenced)) == len(referenced) - 1

    shapes = checkpoint_tensor_shapes()
    missing = sorted(set(referenced) - set(shapes))
    assert not missing, f"Mapped keys that the checkpoint does not have: {missing}"
    unused = sorted(set(shapes) - set(referenced))
    assert not unused, f"Checkpoint tensors that no parameter loads: {unused}"

    H = hf_config.hidden_size
    inter = hf_config.intermediate_size
    kv_dim = hf_config.num_key_value_heads * (H // hf_config.num_attention_heads)
    expected = {"model.embed_tokens.weight": (hf_config.vocab_size, H)}
    for i in range(hf_config.num_hidden_layers):
        p = f"model.layers.{i}"
        expected[f"{p}.self_attn.q_proj.weight"] = (H, H)
        expected[f"{p}.self_attn.k_proj.weight"] = (kv_dim, H)
        expected[f"{p}.self_attn.v_proj.weight"] = (kv_dim, H)
        expected[f"{p}.self_attn.o_proj.weight"] = (H, H)
        expected[f"{p}.mlp.gate_proj.weight"] = (inter, H)
        expected[f"{p}.mlp.up_proj.weight"] = (inter, H)
        expected[f"{p}.mlp.down_proj.weight"] = (H, inter)
    assert shapes == expected


# =============================================================================
# 6. TP 8 weight-loader shards
# =============================================================================


def _small_config():
    """A small OLMo geometry with the 1B head layout (16 Q and 16 KV heads)."""
    OlmoConfig = src_module("config").OlmoConfig
    return OlmoConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        tie_word_embeddings=True,
    )


def test_tp8_weight_loader_shards(monkeypatch, tmp_path):
    """The TP 8 loaders give the same shards as an independent slice."""
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
    from vllm_neuron.utils.weight_loader import get_weight_loader

    model_mod = src_module("model")
    config = _small_config()
    H, inter, head_dim = config.hidden_size, config.intermediate_size, config.head_dim

    torch.manual_seed(42)
    prefix = "model.layers.0."
    raw = {
        f"{prefix}self_attn.q_proj.weight": torch.randn(H, H),
        f"{prefix}self_attn.k_proj.weight": torch.randn(H, H),
        f"{prefix}self_attn.v_proj.weight": torch.randn(H, H),
        f"{prefix}self_attn.o_proj.weight": torch.randn(H, H),
        f"{prefix}mlp.gate_proj.weight": torch.randn(inter, H),
        f"{prefix}mlp.up_proj.weight": torch.randn(inter, H),
        f"{prefix}mlp.down_proj.weight": torch.randn(H, inter),
        "model.embed_tokens.weight": torch.randn(config.vocab_size, H),
    }
    save_file(raw, str(tmp_path / "model.safetensors"))

    _patch_tp_group(monkeypatch, WORLD_SIZE, SHARD_RANK)
    layer = model_mod.OlmoDecoderLayer(config, layer_idx=0)
    mappings = {
        name[len(prefix) :]: keys
        for name, keys in model_mod.OlmoForCausalLM.checkpoint_mappings(1, True).items()
        if name.startswith(prefix)
    }
    shards = (
        SafetensorsCheckpoint(str(tmp_path), None)
        .load_sharded(SHARD_RANK, WORLD_SIZE, layer, mappings, torch.device("cpu"))
        .state_dict
    )
    params = dict(layer.named_parameters())
    assert set(shards) == set(params)

    # Independent slices. HF stores [out_features, in_features]; the model
    # stores [in_features, out_features].
    heads = config.num_attention_heads // WORLD_SIZE
    rows = slice(SHARD_RANK * heads * head_dim, (SHARD_RANK + 1) * heads * head_dim)
    inter_rows = slice(
        SHARD_RANK * inter // WORLD_SIZE, (SHARD_RANK + 1) * inter // WORLD_SIZE
    )
    expected = {
        "self_attn.qkv_proj_weight": torch.cat(
            [raw[f"{prefix}self_attn.{p}_proj.weight"][rows].T for p in "qkv"], dim=1
        ),
        "self_attn.o_proj_weight": raw[f"{prefix}self_attn.o_proj.weight"][:, rows].T,
        "mlp.gate_proj_weight": raw[f"{prefix}mlp.gate_proj.weight"][inter_rows].T,
        "mlp.up_proj_weight": raw[f"{prefix}mlp.up_proj.weight"][inter_rows].T,
        "mlp.down_proj_weight": raw[f"{prefix}mlp.down_proj.weight"][:, inter_rows].T,
    }
    for name, ref in expected.items():
        got = shards[name]
        assert got.shape == params[name].shape, (
            f"{name}: shard shape {tuple(got.shape)}, parameter shape "
            f"{tuple(params[name].shape)}"
        )
        assert torch.equal(got.float(), ref.float()), f"{name} shard is wrong"

    # The tied lm_head reads the vocab rows of the embedding at this rank. (The
    # embedding takes its shard size from the real process group, which this
    # test does not start; test_tiny_e2e.py loads it at TP 2.)
    with torch.device("meta"):
        model = model_mod.OlmoForCausalLM(config)
    embed = raw["model.embed_tokens.weight"]
    vocab_rows = config.vocab_size // WORLD_SIZE
    got = get_weight_loader(model.lm_head.weight).load([MockSlice(embed)], SHARD_RANK)
    assert torch.equal(
        got, embed[SHARD_RANK * vocab_rows : (SHARD_RANK + 1) * vocab_rows]
    )
