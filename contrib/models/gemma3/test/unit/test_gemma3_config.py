# SPDX-License-Identifier: Apache-2.0
"""Gemma3 (rnj-1) registration, config, RoPE and weight-mapping tests (CPU, no download).

The tests read the pinned ``config.json`` of ``EssentialAI/rnj-1-instruct``
from ``gemma3_test_utils.REAL_CONFIG``, so they need no checkpoint:

1. ``test_registration_contract``: the plugin entry point, the class it
   registers, and the vLLM model protocol of the factory.
2. ``test_config_parse``: the config parse from the transformers 5 config
   (per-layer-type ``rope_parameters``), from a raw ``config.json`` file and
   from a plain dict (top-level ``rope_scaling``, ``sliding_window_pattern``).
3. ``test_factory_guards``: the factory rejects a configuration that the
   implementation does not support.
4. ``test_yarn_rotary_matches_hf`` and ``test_yarn_rotary_builds_on_meta``:
   the YaRN inverse frequencies, the attention factor and the cos/sin tables
   against the HF ``Gemma3RotaryEmbedding``, also when the runner builds the
   model on the meta device.
5. ``test_weight_mapping_coverage``: each model parameter maps to checkpoint
   keys, and the map uses each key of the HF model exactly one time (the tied
   LM head reads the embedding).
6. ``test_per_rank_shapes`` and ``test_weight_loader_transforms``: the head
   split and the shard that each loader gives on each TP 8 rank, and the
   replicated KV head at TP 16.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/gemma3/test/unit/ -v
"""

import inspect
import json
import math
import tomllib
from pathlib import Path

import pytest
import torch

from _local_src import src_module, src_package
from gemma3_test_utils import (
    ARCHITECTURE,
    REAL_CONFIG,
    TINY_CONFIG,
    hf_config_from_dict,
    make_tiny_checkpoint,
    port_config,
)

FOLDER = Path(__file__).resolve().parents[2]
NUM_LAYERS = REAL_CONFIG["num_hidden_layers"]


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


# =============================================================================
# 1. Registration
# =============================================================================


def test_registration_contract(monkeypatch):
    """The entry point registers the factory in ``src/`` under the checkpoint arch."""
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
        is_vllm_model,
    )

    assert REAL_CONFIG["architectures"] == [ARCHITECTURE]

    # pyproject.toml installs src/ as ``gemma3_neuron`` and points the
    # vllm.general_plugins entry point at ``gemma3_neuron.plugin:register``.
    with open(FOLDER / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)
    assert project["tool"]["setuptools"]["package-dir"] == {"gemma3_neuron": "src"}
    entry_points = project["project"]["entry-points"]["vllm.general_plugins"]
    assert entry_points == {"gemma3_neuron": "gemma3_neuron.plugin:register"}

    # register() hands vLLM the factory by its installed module path.
    calls = []
    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, target: calls.append((arch, target)),
    )
    src_module("plugin").register()
    assert calls == [(ARCHITECTURE, "gemma3_neuron.factory:Gemma3ForCausalLM")]

    # That path names the class that src/ exports, and the class satisfies the
    # vLLM 0.29 model protocol for a text generation model.
    factory_cls = src_module("factory").Gemma3ForCausalLM
    assert getattr(src_package(), ARCHITECTURE) is factory_cls
    assert is_vllm_model(factory_cls)
    assert is_text_generation_model(factory_cls)


# =============================================================================
# 2. Config
# =============================================================================


def _check_rnj1_fields(cfg) -> None:
    assert cfg.num_hidden_layers == NUM_LAYERS
    assert cfg.layer_types == ["full_attention"] * NUM_LAYERS
    assert (cfg.hidden_size, cfg.intermediate_size) == (4096, 16384)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (
        32,
        8,
        128,
    )
    assert cfg.vocab_size == 128256
    assert cfg.max_position_embeddings == 32768
    assert cfg.tie_word_embeddings is True
    assert cfg.final_logit_softcapping == 30.0
    assert cfg.attn_logit_softcapping is None
    assert cfg.hidden_activation == "gelu_pytorch_tanh"
    assert cfg.attention_bias is False
    assert cfg.rope_theta == 10000.0
    assert cfg.rope_type == "yarn"
    assert cfg.rope_factor == 4.0
    assert cfg.rope_original_max_position_embeddings == 8192
    assert cfg.rope_beta_fast == 64.0
    assert cfg.rope_beta_slow == 1.0
    # The checkpoint's ``attn_factor`` is not a YaRN key that transformers
    # reads, so the attention factor comes from the scale factor.
    assert cfg.rope_attention_factor is None
    # The checkpoint is FP32; the served dtype outside a vLLM config is BF16.
    assert cfg.torch_dtype == torch.bfloat16
    assert math.isclose(cfg.attn_scaling, 128.0**-0.5)
    assert cfg.embed_scale == 64.0


def test_config_parse(tmp_path):
    """Parse the config from transformers 5, from config.json and from a dict."""
    hf_config = hf_config_from_dict(REAL_CONFIG)
    # transformers 5 moves the RoPE settings into one dict per layer type and
    # derives the layer types from sliding_window_pattern (1: all full).
    assert hf_config.rope_parameters["full_attention"]["rope_type"] == "yarn"
    assert hf_config.layer_types == ["full_attention"] * NUM_LAYERS
    _check_rnj1_fields(port_config(hf_config))

    Gemma3Config = src_module("config").Gemma3Config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(REAL_CONFIG))
    _check_rnj1_fields(Gemma3Config.from_configs(str(path)))
    _check_rnj1_fields(Gemma3Config.from_configs(dict(REAL_CONFIG)))


def test_factory_guards(monkeypatch):
    """The factory rejects what the implementation does not do."""
    Factory = src_module("factory").Gemma3ForCausalLM

    Factory._validate_config(hf_config_from_dict(REAL_CONFIG), None)
    for override, match in (
        # Gemma 3 default: five sliding-window layers, then one full layer.
        ({"sliding_window_pattern": 6}, "full attention"),
        ({"attn_logit_softcapping": 50.0}, "attn_logit_softcapping"),
        ({"hidden_activation": "gelu", "hidden_act": "gelu"}, "Activation"),
        ({"attention_bias": True}, "attention_bias"),
    ):
        bad = hf_config_from_dict({**REAL_CONFIG, **override})
        with pytest.raises(ValueError, match=match):
            Factory._validate_config(bad, None)

    # Only the BF16 served dtype is implemented.
    monkeypatch.setattr(
        src_module("config"), "resolve_served_dtype", lambda: torch.float16
    )
    with pytest.raises(ValueError, match="bfloat16"):
        Factory._validate_config(hf_config_from_dict(REAL_CONFIG), None)


# =============================================================================
# 3. RoPE
# =============================================================================

BF16_EPS = 2.0**-7
_POSITIONS = torch.cat(
    [torch.arange(0, 1536), torch.tensor([2047, 4095, 8191, 8192, 16383, 32767])]
)


def _hf_rotary(hf_config, positions: torch.Tensor):
    """Return cos, sin, inv_freq and the scale of the HF full-attention RoPE."""
    from transformers.models.gemma3 import modeling_gemma3 as hf

    rope = hf.Gemma3RotaryEmbedding(config=hf_config)
    x = torch.zeros(1, positions.numel(), 8, dtype=torch.float32)
    fwd_params = inspect.signature(rope.forward).parameters
    with torch.no_grad():
        if "layer_type" in fwd_params:
            cos, sin = rope(x, positions[None], layer_type="full_attention")
        else:
            cos, sin = rope(x, positions[None])
    inv_freq = getattr(rope, "full_attention_inv_freq", None)
    if inv_freq is None:
        inv_freq = rope.inv_freq
    scale = getattr(rope, "full_attention_attention_scaling", None)
    if scale is None:
        scale = rope.attention_scaling
    return cos[0], sin[0], inv_freq, float(scale)


def test_yarn_rotary_matches_hf():
    """The YaRN inverse frequencies, scale and cos/sin tables match HF in fp32."""
    model = src_module("model")
    hf_config = hf_config_from_dict(REAL_CONFIG)
    cfg = port_config(hf_config)
    hf_cos, hf_sin, hf_inv, hf_scale = _hf_rotary(hf_config, _POSITIONS)

    inv, scale = model.compute_rope_inv_freq(cfg)
    torch.testing.assert_close(inv, hf_inv.float().cpu(), rtol=1e-6, atol=0)
    assert math.isclose(scale, hf_scale, rel_tol=1e-7), (scale, hf_scale)
    assert math.isclose(scale, 0.1 * math.log(4.0) + 1.0, rel_tol=1e-7)

    rope = model.Gemma3RotaryEmbedding(cfg)
    cos, sin = rope(_POSITIONS, device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(cos, hf_cos.float(), rtol=0, atol=2e-5)
    torch.testing.assert_close(sin, hf_sin.float(), rtol=0, atol=2e-5)

    cos_bf16, sin_bf16 = rope(
        _POSITIONS, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    torch.testing.assert_close(
        cos_bf16, hf_cos.to(torch.bfloat16), rtol=0, atol=BF16_EPS
    )
    torch.testing.assert_close(
        sin_bf16, hf_sin.to(torch.bfloat16), rtol=0, atol=BF16_EPS
    )


def test_yarn_rotary_builds_on_meta():
    """The rotary table has real values when the model is built on meta.

    The model runner builds the model in a meta device context. The rotary
    table must not become a meta tensor there.
    """
    model = src_module("model")
    hf_config = hf_config_from_dict(REAL_CONFIG)
    cfg = port_config(hf_config)
    hf_cos, hf_sin, hf_inv, _ = _hf_rotary(hf_config, _POSITIONS)

    with torch.device("meta"):
        rope = model.Gemma3RotaryEmbedding(cfg)
    reference = model.Gemma3RotaryEmbedding(cfg)
    # The table is a non-persistent CPU buffer. Module.to() moves it to the
    # device, and the weight load does not touch it.
    assert rope.inv_freq.device.type == "cpu"
    assert "inv_freq" not in rope.state_dict()
    torch.testing.assert_close(rope.inv_freq, reference.inv_freq, rtol=0, atol=0)
    assert rope.attention_scaling == reference.attention_scaling
    torch.testing.assert_close(rope.inv_freq, hf_inv.float().cpu(), rtol=1e-6, atol=0)

    cos, sin = rope(_POSITIONS, device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(cos, hf_cos.float(), rtol=0, atol=2e-5)
    torch.testing.assert_close(sin, hf_sin.float(), rtol=0, atol=2e-5)


# =============================================================================
# 4. Weight mapping and loaders
# =============================================================================


def _mapped_sources(mappings: dict) -> list[str]:
    used = []
    for value in mappings.values():
        used += value if isinstance(value, list) else [value]
    return used


def test_weight_mapping_coverage(monkeypatch, tmp_path):
    """Each parameter maps to checkpoint keys. Each HF key is used one time."""
    from safetensors import safe_open
    from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM

    model = src_module("model")
    hf_config = hf_config_from_dict(REAL_CONFIG)
    with torch.device("meta"):
        hf_keys = set(HFGemma3ForCausalLM(hf_config).state_dict())
    # The LM head is tied to the embedding, so a checkpoint does not store it.
    hf_keys.discard("lm_head.weight")
    assert len(hf_keys) == 2 + 13 * NUM_LAYERS

    mappings = model.build_weight_mappings(NUM_LAYERS, tie_word_embeddings=True)
    assert mappings["lm_head.weight"] == "model.embed_tokens.weight"
    used = _mapped_sources(mappings)
    # The tied LM head reads the embedding key a second time; nothing else may.
    assert used.count("model.embed_tokens.weight") == 2
    used.remove("model.embed_tokens.weight")
    assert len(used) == len(set(used)), "A checkpoint key is mapped twice."
    assert set(used) == hf_keys, (
        f"not mapped: {sorted(hf_keys - set(used))[:5]}, "
        f"not in the HF model: {sorted(set(used) - hf_keys)[:5]}"
    )

    # The mapping keys are the parameter names of the model at TP 8. The RoPE
    # table is a buffer, so it needs no mapping.
    _patch_tp_group(monkeypatch, 8, 0)
    cfg = port_config(hf_config)
    with torch.device("meta"):
        full = model.Gemma3ForCausalLM(cfg)
    assert {n for n, _ in full.named_parameters()} == set(mappings)

    # A checkpoint that save_pretrained writes has exactly the mapped keys.
    ckpt = make_tiny_checkpoint(str(tmp_path))
    with safe_open(str(Path(ckpt) / "model.safetensors"), framework="pt") as f:
        saved = set(f.keys())
    tiny_map = model.build_weight_mappings(
        TINY_CONFIG["num_hidden_layers"], tie_word_embeddings=True
    )
    assert set(_mapped_sources(tiny_map)) == saved


@pytest.mark.parametrize("world_size", [8, 16])
def test_per_rank_shapes(monkeypatch, world_size):
    """The head split, the weight shapes and the KV spec at TP 8 and TP 16."""
    model = src_module("model")
    cfg = port_config(hf_config_from_dict(REAL_CONFIG))
    _patch_tp_group(monkeypatch, world_size, 0)
    with torch.device("meta"):
        full = model.Gemma3ForCausalLM(cfg)

    q_heads = 32 // world_size
    attn = full.model.layers[0].self_attn
    assert attn.num_attention_heads_per_rank == q_heads
    # 8 KV heads: one KV head on each rank, replicated on 2 ranks at TP 16.
    assert attn.num_key_value_heads_per_rank == 1
    assert attn.num_kv_replicas == world_size // 8
    assert tuple(attn.qkv_proj_weight.shape) == (4096, (q_heads + 2) * 128)
    assert tuple(attn.o_proj_weight.shape) == (q_heads * 128, 4096)
    mlp = full.model.layers[0].mlp
    assert tuple(mlp.gate_proj_weight.shape) == (4096, 16384 // world_size)
    assert tuple(mlp.down_proj_weight.shape) == (16384 // world_size, 4096)

    spec = full.get_kv_spec()
    assert len(spec.layers) == NUM_LAYERS
    for i, layer in enumerate(spec.layers):
        assert layer.name == f"layers.{i}.self_attn"
        assert (layer.num_kv_heads, layer.head_size) == (1, 128)
        assert layer.sliding_window_size is None


def test_weight_loader_transforms(monkeypatch):
    """Check the attached loaders on each TP 8 rank, and the KV replicas at TP 16.

    The HF layout is [out_features, in_features]. The implementation stores
    the projections transposed, [in_features, out_features]. At TP 16 the
    two ranks of a KV head get the same K and V rows.
    """
    from vllm_neuron.utils.weight_loader import get_weight_loader

    model = src_module("model")
    cfg = port_config(hf_config_from_dict(REAL_CONFIG))
    # A reduced hidden size keeps the random tensors small. The head layout
    # stays the rnj-1 one (32 Q heads, 8 KV heads, head_dim 128).
    cfg.hidden_size = 256
    cfg.intermediate_size = 1024
    hidden, inter, head_dim = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    q_full = cfg.num_attention_heads * head_dim
    kv_full = cfg.num_key_value_heads * head_dim

    torch.manual_seed(0)
    w_q, w_k, w_v = (torch.randn(n, hidden) for n in (q_full, kv_full, kv_full))
    w_o = torch.randn(hidden, q_full)
    w_gate, w_up = torch.randn(inter, hidden), torch.randn(inter, hidden)
    w_down = torch.randn(hidden, inter)
    embed = torch.randn(cfg.vocab_size, 8)

    def load(param, tensors, rank):
        return get_weight_loader(param).load([_MockSlice(t) for t in tensors], rank)

    for world_size in (8, 16):
        q_shard, i_shard = q_full // world_size, inter // world_size
        vocab_shard = cfg.vocab_size // world_size
        replicas = world_size // cfg.num_key_value_heads
        for rank in range(world_size):
            _patch_tp_group(monkeypatch, world_size, rank)
            with torch.device("meta"):
                attn = model.Gemma3Attention(cfg, layer_idx=0)
                mlp = model.Gemma3MLP(cfg)
            q_rows = slice(rank * q_shard, (rank + 1) * q_shard)
            kv_head = rank // replicas
            kv_rows = slice(kv_head * head_dim, (kv_head + 1) * head_dim)
            i_rows = slice(rank * i_shard, (rank + 1) * i_shard)
            expected_qkv = torch.cat((w_q[q_rows], w_k[kv_rows], w_v[kv_rows]), dim=0).T
            cases = {
                "qkv": (
                    load(attn.qkv_proj_weight, [w_q, w_k, w_v], rank),
                    expected_qkv,
                ),
                "o": (load(attn.o_proj_weight, [w_o], rank), w_o[:, q_rows].T),
                "gate": (load(mlp.gate_proj_weight, [w_gate], rank), w_gate[i_rows].T),
                "up": (load(mlp.up_proj_weight, [w_up], rank), w_up[i_rows].T),
                "down": (
                    load(mlp.down_proj_weight, [w_down], rank),
                    w_down[:, i_rows].T,
                ),
            }
            for name, (actual, want) in cases.items():
                assert tuple(actual.shape) == tuple(want.shape), (name, rank)
                assert torch.equal(actual, want), (name, world_size, rank)

        # The tied LM head: each rank reads its vocabulary rows of the
        # embedding (128256 divides by 8 and by 16, so there is no padding).
        _patch_tp_group(monkeypatch, world_size, 0)
        with torch.device("meta"):
            full = model.Gemma3ForCausalLM(
                port_config(hf_config_from_dict(REAL_CONFIG))
            )
        for rank in (0, world_size - 1):
            got = load(full.lm_head.weight, [embed], rank)
            want = embed[rank * vocab_shard : (rank + 1) * vocab_shard]
            assert torch.equal(got, want), (world_size, rank)
