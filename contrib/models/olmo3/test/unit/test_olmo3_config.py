# SPDX-License-Identifier: Apache-2.0
"""OLMo-3 registration, config, RoPE and weight-mapping tests (CPU, no download).

The tests read the pinned ``config.json`` of ``allenai/Olmo-3-7B-Instruct``
from ``olmo3_test_utils.REAL_CONFIG``, so they need no checkpoint:

1. ``test_registration_contract``: the plugin entry point, the class it
   registers, and the vLLM model protocol of the factory.
2. ``test_config_parse``: the config parse from the transformers 5 config
   (per-layer-type ``rope_parameters``), from a raw ``config.json`` file and
   from a plain dict.
3. ``test_rope_tables_ignore_default_device``: the RoPE tables are FP32 CPU
   buffers, also when the runner builds the model on the meta device.
4. ``test_factory_guards``: the factory rejects a configuration that the
   implementation does not support, and the attention rejects an illegal TP
   degree.
5. ``test_rope_matches_hf``: the default RoPE and the YaRN RoPE against the HF
   ``Olmo3RotaryEmbedding``, over the full 65,536-position range.
6. ``test_weight_mapping_coverage``: each model parameter maps to checkpoint
   keys, and the map uses each key of the HF model exactly one time.
7. ``test_weight_loader_transforms``: the loaders that the model attaches give
   the expected shard on each TP16 rank, with the vocabulary padding.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo3/test/unit/ -v
"""

import json
import tomllib
from pathlib import Path

import pytest
import torch

from _local_src import src_module, src_package
from olmo3_test_utils import (
    ARCHITECTURE,
    REAL_CONFIG,
    hf_config_from_dict,
    hf_rotary,
    port_config,
)

TP_SIZE = 16
FOLDER = Path(__file__).resolve().parents[2]


class _FakeTPGroup:
    """Give the TP size and the rank without a process group."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = None


class _MockSlice:
    """Checkpoint slice stand-in that wraps a real tensor."""

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


def test_registration_contract(monkeypatch):
    """The entry point registers the factory in ``src/`` under the checkpoint arch."""
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
        is_vllm_model,
    )

    assert REAL_CONFIG["architectures"] == [ARCHITECTURE]

    # pyproject.toml installs src/ as ``olmo3_neuron`` and points the
    # vllm.general_plugins entry point at ``olmo3_neuron.plugin:register``.
    with open(FOLDER / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)
    package_dir = project["tool"]["setuptools"]["package-dir"]
    assert package_dir == {"olmo3_neuron": "src"}
    entry_points = project["project"]["entry-points"]["vllm.general_plugins"]
    assert entry_points == {"olmo3_neuron": "olmo3_neuron.plugin:register"}

    # register() hands vLLM the factory by its installed module path.
    calls = []
    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, target: calls.append((arch, target)),
    )
    src_module("plugin").register()
    assert calls == [(ARCHITECTURE, "olmo3_neuron.factory:Olmo3ForCausalLM")]

    # That path names the class that src/ exports, and the class satisfies the
    # vLLM 0.29 model protocol for a text generation model.
    factory_cls = src_module("factory").Olmo3ForCausalLM
    assert getattr(src_package(), ARCHITECTURE) is factory_cls
    assert is_vllm_model(factory_cls)
    assert is_text_generation_model(factory_cls)


def _check_7b_fields(cfg) -> None:
    assert cfg.num_hidden_layers == 32
    assert cfg.num_attention_heads == cfg.num_key_value_heads == 32
    assert cfg.hidden_size == 4096
    assert cfg.intermediate_size == 11008
    assert cfg.head_dim == 128
    assert cfg.vocab_size == 100278
    assert cfg.max_position_embeddings == 65536
    assert cfg.torch_dtype == torch.bfloat16
    assert cfg.sliding_window == 4096
    assert cfg.rope_theta == 500000
    assert cfg.rope_scaling["rope_type"] == "yarn"
    assert cfg.rope_scaling["factor"] == 8.0
    assert cfg.rope_scaling["original_max_position_embeddings"] == 8192
    assert cfg.rope_scaling["attention_factor"] == pytest.approx(1.2079441541679836)
    assert cfg.tie_word_embeddings is False
    # Three sliding-window layers, then one full-attention layer.
    full = [i for i in range(32) if not cfg.is_sliding(i)]
    assert full == list(range(3, 32, 4))


def test_config_parse(tmp_path):
    """Parse the config from transformers 5, from config.json and from a dict."""
    hf_config = hf_config_from_dict(REAL_CONFIG)
    # transformers 5 moves the RoPE settings into one dict per layer type.
    assert set(hf_config.rope_parameters) == {"sliding_attention", "full_attention"}
    assert hf_config.rope_parameters["sliding_attention"]["rope_type"] == "default"
    assert hf_config.rope_parameters["full_attention"]["rope_type"] == "yarn"
    _check_7b_fields(port_config(hf_config))

    Olmo3Config = src_module("config").Olmo3Config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(REAL_CONFIG))
    _check_7b_fields(Olmo3Config.from_configs(str(path)))
    _check_7b_fields(Olmo3Config.from_configs(dict(REAL_CONFIG)))

    # A YaRN setting on the sliding-window layers is not supported.
    bad = json.loads(json.dumps(hf_config.to_dict()))
    bad["rope_parameters"]["sliding_attention"]["rope_type"] = "yarn"
    with pytest.raises(ValueError, match="sliding-window layers"):
        Olmo3Config.from_configs(bad)


def test_rope_tables_ignore_default_device():
    """The RoPE tables are FP32 CPU buffers, outside the state dict."""
    model = src_module("model")
    cfg = port_config(hf_config_from_dict(REAL_CONFIG))
    for layer_type in (model.SLIDING_ATTENTION, model.FULL_ATTENTION):
        ref = model.Olmo3RotaryEmbedding(cfg, layer_type)
        # The runner builds the model with the meta default device.
        with torch.device("meta"):
            on_meta = model.Olmo3RotaryEmbedding(cfg, layer_type)
        assert list(on_meta.parameters()) == []
        assert on_meta.state_dict() == {}
        assert (on_meta.yarn is not None) == (layer_type == model.FULL_ATTENTION)
        assert on_meta.yarn == ref.yarn
        assert on_meta.attention_scaling == ref.attention_scaling
        for name in ("cos_half", "sin_half"):
            table = getattr(on_meta, name)
            assert table.device.type == "cpu"
            assert table.dtype == torch.float32
            assert table.shape == (cfg.max_position_embeddings, cfg.head_dim // 2)
            torch.testing.assert_close(table, getattr(ref, name), rtol=0, atol=0)


def test_factory_guards(monkeypatch):
    """The factory and the attention reject what the implementation does not do."""
    Factory = src_module("factory").Olmo3ForCausalLM
    Olmo3Attention = src_module("model").Olmo3Attention

    Factory._validate_config(hf_config_from_dict(REAL_CONFIG), None)
    for override, match in (
        ({"dtype": "float16"}, "dtype"),
        ({"attention_bias": True}, "attention_bias"),
        ({"layer_types": _short_layer_types()}, "layer_types"),
        ({"layer_types": ["chunked_attention"] * 32}, "layer types"),
    ):
        bad = {**REAL_CONFIG, **override}
        hf_config = _hf_config_unchecked(bad)
        with pytest.raises(ValueError, match=match):
            Factory._validate_config(hf_config, None)

    # 32 Q heads and 32 KV heads: TP 1, 2, 4, 8 and 16 are legal. TP 32 gives
    # one Q head on each rank, and TP 3 does not divide the heads.
    cfg = port_config(hf_config_from_dict(REAL_CONFIG))
    for world_size in (1, 2, 4, 8, 16):
        _patch_tp_group(monkeypatch, world_size, 0)
        with torch.device("meta"):
            Olmo3Attention(cfg, layer_idx=0)
    for world_size in (3, 32):
        _patch_tp_group(monkeypatch, world_size, 0)
        with pytest.raises(ValueError), torch.device("meta"):
            Olmo3Attention(cfg, layer_idx=0)


def _short_layer_types() -> list[str]:
    return REAL_CONFIG["layer_types"][:-1]


def _hf_config_unchecked(config_dict: dict):
    """Build an HF config, then set the fields that its validation would reject."""
    clean = {k: v for k, v in config_dict.items() if k not in ("layer_types", "dtype")}
    hf_config = hf_config_from_dict(clean)
    hf_config.layer_types = config_dict["layer_types"]
    dtype = config_dict.get("dtype")
    if dtype is not None:
        hf_config.dtype = getattr(torch, dtype)
    return hf_config


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_rope_matches_hf(layer_type):
    """Compare the default RoPE and the YaRN RoPE with the HF reference."""
    Olmo3RotaryEmbedding = src_module("model").Olmo3RotaryEmbedding

    hf_config = hf_config_from_dict(REAL_CONFIG)
    cfg = port_config(hf_config)
    positions = torch.arange(0, 65536, 61, dtype=torch.int64)
    x = torch.zeros(1, positions.numel(), 128, dtype=torch.float32)
    hf_cos, hf_sin = hf_rotary(hf_config)(x, positions.unsqueeze(0), layer_type)
    port = Olmo3RotaryEmbedding(cfg, layer_type)
    cos, sin = port(positions, device=torch.device("cpu"), dtype=torch.float32)
    expected_scale = 1.0 if layer_type == "sliding_attention" else 1.2079441541679836
    assert port.attention_scaling == pytest.approx(expected_scale)
    torch.testing.assert_close(cos, hf_cos[0], rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(sin, hf_sin[0], rtol=1e-5, atol=1e-4)


def test_weight_mapping_coverage(monkeypatch):
    """Each parameter maps to checkpoint keys. Each HF key is used one time."""
    from transformers import Olmo3ForCausalLM as HFOlmo3ForCausalLM

    model = src_module("model")
    hf_config = hf_config_from_dict(REAL_CONFIG)
    with torch.device("meta"):
        hf_keys = set(HFOlmo3ForCausalLM(hf_config).state_dict())
    assert len(hf_keys) == 3 + 11 * 32

    mappings = model.olmo3_weight_mappings(32)
    used = []
    for value in mappings.values():
        used += value if isinstance(value, list) else [value]
    assert len(used) == len(set(used)), "A checkpoint key is mapped twice."
    assert set(used) == hf_keys, (
        f"not mapped: {sorted(hf_keys - set(used))[:5]}, "
        f"not in the HF model: {sorted(set(used) - hf_keys)[:5]}"
    )

    # The mapping keys are the parameter names of the model at TP16. The RoPE
    # tables are buffers, so they need no mapping.
    _patch_tp_group(monkeypatch, TP_SIZE, 0)
    cfg = port_config(hf_config)
    with torch.device("meta"):
        layer = model.Olmo3DecoderLayer(cfg, layer_idx=0)
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(32):
        names |= {f"model.layers.{i}.{n}" for n, _ in layer.named_parameters()}
    assert set(mappings) == names


def test_weight_loader_transforms(monkeypatch):
    """Check the attached loaders on each TP16 rank with mock slices.

    The HF layout is [out_features, in_features]. The implementation stores
    the projections transposed, [in_features, out_features]. QK-norm gammas
    follow the head shard. The LM head pads 100278 rows to 16 x 6268 = 100288
    with zero rows on the last rank.
    """
    from vllm_neuron.utils.weight_loader import get_weight_loader

    model = src_module("model")
    cfg = port_config(hf_config_from_dict(REAL_CONFIG))
    # A reduced hidden size keeps the random tensors small. The head and
    # vocabulary layout stays the 7B one.
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    hidden, inter, head_dim = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    q_full = cfg.num_attention_heads * head_dim
    kv_full = cfg.num_key_value_heads * head_dim

    torch.manual_seed(0)
    w_q, w_k, w_v = (torch.randn(n, hidden) for n in (q_full, kv_full, kv_full))
    w_o = torch.randn(hidden, q_full)
    q_norm, k_norm = torch.randn(q_full), torch.randn(kv_full)
    w_gate, w_up = torch.randn(inter, hidden), torch.randn(inter, hidden)
    w_down = torch.randn(hidden, inter)
    lm_head = torch.randn(cfg.vocab_size, 8)

    q_shard, kv_shard, i_shard = q_full // TP_SIZE, kv_full // TP_SIZE, inter // TP_SIZE
    vocab_shard = -(-cfg.vocab_size // TP_SIZE)
    assert vocab_shard == 6268

    def load(param, tensors, rank):
        return get_weight_loader(param).load([_MockSlice(t) for t in tensors], rank)

    for rank in range(TP_SIZE):
        _patch_tp_group(monkeypatch, TP_SIZE, rank)
        with torch.device("meta"):
            attn = model.Olmo3Attention(cfg, layer_idx=0)
            mlp = model.Olmo3MLP(cfg)
        q_rows = slice(rank * q_shard, (rank + 1) * q_shard)
        kv_rows = slice(rank * kv_shard, (rank + 1) * kv_shard)
        i_rows = slice(rank * i_shard, (rank + 1) * i_shard)
        expected_qkv = torch.cat((w_q[q_rows], w_k[kv_rows], w_v[kv_rows]), dim=0).T
        cases = {
            "qkv": (load(attn.qkv_proj_weight, [w_q, w_k, w_v], rank), expected_qkv),
            "o": (load(attn.o_proj_weight, [w_o], rank), w_o[:, q_rows].T),
            "q_norm": (load(attn.q_norm.weight, [q_norm], rank), q_norm[q_rows]),
            "k_norm": (load(attn.k_norm.weight, [k_norm], rank), k_norm[kv_rows]),
            "gate": (load(mlp.gate_proj_weight, [w_gate], rank), w_gate[i_rows].T),
            "up": (load(mlp.up_proj_weight, [w_up], rank), w_up[i_rows].T),
            "down": (load(mlp.down_proj_weight, [w_down], rank), w_down[:, i_rows].T),
        }
        for name, (actual, want) in cases.items():
            assert tuple(actual.shape) == tuple(want.shape), (name, rank)
            assert torch.equal(actual, want), (name, rank)

    # The LM head and the embedding use the same padded vocabulary shard.
    from vllm_neuron.utils.weight_loader import sharding_weight_loader

    lm_loader = sharding_weight_loader(
        shard_dim=0,
        shard_size=vocab_shard,
        num_shards=TP_SIZE,
        is_storage_transposed=False,
        pad_shard=True,
    )
    last = lm_loader.load([_MockSlice(lm_head)], TP_SIZE - 1)
    real_rows = cfg.vocab_size - (TP_SIZE - 1) * vocab_shard
    assert last.shape == (vocab_shard, 8)
    assert torch.equal(last[:real_rows], lm_head[(TP_SIZE - 1) * vocab_shard :])
    assert torch.count_nonzero(last[real_rows:]) == 0
