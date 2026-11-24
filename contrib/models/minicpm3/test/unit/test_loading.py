# SPDX-License-Identifier: Apache-2.0
"""MiniCPM3 loading tests (CPU, one process, no Neuron hardware).

Coverage:
  - Registration: the checkpoint's architecture string reaches this folder's
    factory through ``src/plugin.py`` (the ``pyproject.toml`` entry point) and
    through a direct ``ModelRegistry.register_model`` call, and the factory
    satisfies vLLM 0.29's model protocol.
  - The factory rejects the configurations that this implementation does not
    support.
  - Config parse and padded MLA head layout of MiniCPM3-4B.
  - RoPE buffers: finite CPU buffers under a meta-device build, and LongRoPE
    values against an independent FP64 formula.
  - Checkpoint key coverage: the mappings read each checkpoint tensor once.
  - ``src/checkpoints.py``: the ``pytorch_model*.bin`` fallback of the
    checkpoint loader (the published checkpoint has no ``.safetensors`` file).
  - Fused MLA weight math in FP32 against the HF attention (no kernels), at
    the real MiniCPM3-4B geometry with random weights.

The tests build small random-weight checkpoints with the MiniCPM3-4B geometry,
so they download nothing. ``test_checkpoint_key_coverage_real`` also checks the
real checkpoint when ``MINICPM3_CHECKPOINT_PATH`` is set and skips otherwise.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/minicpm3/test/unit/test_loading.py -v
"""

import json
import math
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from _local_src import src_module, src_package
import minicpm3_test_utils as U

_FOLDER = Path(__file__).resolve().parents[2]
SEED = 42
# The geometry of MiniCPM3-4B with one layer and a small vocabulary.
ONE_LAYER = {"num_hidden_layers": 1, "vocab_size": 1024, "initializer_range": 0.02}


def _factory():
    return src_package().MiniCPM3ForCausalLM


@pytest.fixture(scope="module")
def one_layer_checkpoint(tmp_path_factory):
    """A one-layer random-weight checkpoint with the MiniCPM3-4B geometry."""
    path = str(tmp_path_factory.mktemp("minicpm3_one_layer"))
    return U.write_random_checkpoint(path, U.hf_config(ONE_LAYER), seed=SEED)


# =============================================================================
# Registration and factory
# =============================================================================


class TestRegistration:
    """The checkpoint's architecture string must reach this folder's factory.

    A contrib model is deliberately absent from the plugin's
    ``vllm_neuron.model.registry.get_models()``. It reaches vLLM through the
    ``vllm.general_plugins`` entry point of ``pyproject.toml`` (option 3 of
    ``contrib/README.md``) or through a ``ModelRegistry.register_model`` call in
    the user's own script (option 1).
    """

    def test_src_exports_factory(self):
        assert getattr(src_package(), U.ARCHITECTURE) is _factory()
        assert U.MINICPM3_4B_CONFIG["architectures"] == [U.ARCHITECTURE]

    def test_factory_satisfies_vllm_model_protocol(self):
        """vLLM's front end rejects a class that fails these checks."""
        from vllm.model_executor.models.interfaces_base import (
            is_pooling_model,
            is_text_generation_model,
            is_vllm_model,
        )

        assert is_vllm_model(_factory())
        assert is_text_generation_model(_factory())
        assert not is_pooling_model(_factory())

    def test_registered_factory_replaces_vllm_class(self, monkeypatch):
        """vLLM ships its own MiniCPM3ForCausalLM; registration replaces it."""
        from vllm import ModelRegistry

        monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
        ModelRegistry.register_model(U.ARCHITECTURE, _factory())
        info = ModelRegistry.models[U.ARCHITECTURE].inspect_model_cls()
        assert info.is_text_generation_model
        assert not info.is_pooling_model
        assert ModelRegistry.models[U.ARCHITECTURE].load_model_cls() is _factory()

    def test_plugin_entry_point_registers_factory(self, monkeypatch):
        """``pyproject.toml`` and ``src/plugin.py`` must name the same package."""
        from vllm import ModelRegistry

        with open(_FOLDER / "pyproject.toml", "rb") as f:
            project = tomllib.load(f)
        package_dir = project["tool"]["setuptools"]["package-dir"]
        assert list(package_dir.values()) == ["src"]
        (package,) = package_dir
        entry_points = project["project"]["entry-points"]["vllm.general_plugins"]
        assert entry_points == {package: f"{package}.plugin:register"}

        monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
        plugin = src_module("plugin")
        plugin.register()
        plugin.register()  # idempotent
        registered = ModelRegistry.models[U.ARCHITECTURE]
        assert registered.module_name == f"{package}.factory"
        assert registered.class_name == U.ARCHITECTURE


@pytest.mark.parametrize(
    "override, message",
    [
        ({"dtype": torch.float16}, "bfloat16"),
        ({"q_lora_rank": None}, "q_lora_rank"),
        ({"attention_bias": True}, "attention_bias"),
        ({"mlp_bias": True}, "mlp_bias"),
    ],
    ids=["fp16", "no_q_lora", "attention_bias", "mlp_bias"],
)
def test_factory_rejects_unsupported_config(override, message):
    """The factory stops with a clear error before it builds a module."""
    hf = U.hf_config()
    for name, value in override.items():
        setattr(hf, name, value)
    with pytest.raises(ValueError, match=message):
        _factory()._validate_config(hf, None)


def test_factory_accepts_checkpoint_config():
    _factory()._validate_config(U.hf_config(), None)


# =============================================================================
# Config and head layout
# =============================================================================


def test_config_and_head_layout():
    """The derived config and the padded MLA head layout agree with the checkpoint."""
    minicpm3_head_layout = src_module("model").minicpm3_head_layout

    hf = U.hf_config()
    cfg = U.neuron_model_config(hf)
    layout = minicpm3_head_layout(cfg)

    assert cfg.num_hidden_layers == 62
    assert cfg.torch_dtype == torch.bfloat16
    assert cfg.rope_theta == 10000.0
    assert cfg.rope_scaling["rope_type"] == "longrope"
    assert cfg.v_head_dim == hf.hidden_size // hf.num_attention_heads == 64
    assert cfg.qk_head_dim == hf.qk_nope_head_dim + hf.qk_rope_head_dim
    assert math.isclose(cfg.residual_scale, hf.scale_depth / math.sqrt(62))
    assert math.isclose(cfg.logits_scaling, hf.hidden_size / hf.dim_model_base)
    assert cfg.scale_emb == hf.scale_emb
    assert cfg.tie_word_embeddings

    assert (layout.d_head, layout.half_d, layout.latent_dim) == (128, 64, 1280)
    # The server builds the model under a meta device context. The layout
    # index tensors must stay on the CPU, because the weight builders read them.
    with torch.device("meta"):
        meta_layout = minicpm3_head_layout(cfg)
    assert meta_layout.q_index.device.type == "cpu"
    assert meta_layout.rope_index.device.type == "cpu"
    assert torch.equal(meta_layout.q_index, layout.q_index)
    # Each HF Q/K channel has one distinct destination in the padded head.
    assert len(set(layout.q_index.tolist())) == cfg.qk_head_dim
    assert int(layout.q_index.max()) < layout.d_head
    # RoPE pairs HF rotary channel r with r + rope/2. The padded layout must
    # keep that pair at a distance of half_d.
    half = hf.qk_rope_head_dim // 2
    for r in range(half):
        assert layout.rope_index[r + half] - layout.rope_index[r] == layout.half_d
    # The integration TP divides the heads, the MLP and the vocabulary.
    assert cfg.num_attention_heads % U.WORLD_SIZE == 0
    assert cfg.intermediate_size % U.WORLD_SIZE == 0
    assert cfg.vocab_size % U.WORLD_SIZE == 0


def test_config_from_json_path(tmp_path):
    """``from_configs`` also accepts the path of a raw (pre-v5) config.json."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(U.MINICPM3_4B_CONFIG))
    cfg = src_module("config").MiniCPM3Config.from_configs(str(path))
    assert cfg.torch_dtype == torch.bfloat16
    assert len(cfg.rope_short_factor) == U.MINICPM3_4B_CONFIG["qk_rope_head_dim"] // 2
    assert cfg.rope_original_max_position_embeddings == 32768


def test_rotary_buffers_under_meta_context():
    """The rotary frequencies are finite CPU buffers when the model builds on meta.

    The values agree with an independent FP64 LongRoPE formula. The pad
    channels give cos = 1 and sin = 0.
    """
    model = src_module("model")

    cfg = U.neuron_model_config(U.hf_config())
    layout = model.minicpm3_head_layout(cfg)
    with torch.device("meta"):
        rotary = model.MiniCPM3RotaryEmbedding(cfg, layout)
    for name, buf in rotary.named_buffers():
        assert buf.device.type == "cpu", f"Buffer {name} is on {buf.device}."
        assert torch.isfinite(buf).all(), (
            f"Buffer {name} has a value that is not finite."
        )

    # Read the RoPE values from the raw config.json of the checkpoint. The
    # transformers config class moves these keys between versions.
    raw = U.MINICPM3_4B_CONFIG
    scaling = raw["rope_scaling"]
    rope = raw["qk_rope_head_dim"]
    half = rope // 2
    # The checkpoint gives no rope_theta. The checkpoint's remote code and
    # transformers then use the base 10000.
    theta = raw.get("rope_theta", 10000.0)
    assert cfg.rope_theta == theta
    short = torch.tensor(scaling["short_factor"], dtype=torch.float64)
    base = theta ** (torch.arange(0, rope, 2, dtype=torch.float64) / rope)
    positions = torch.arange(0, 2048, 7, dtype=torch.int64)
    freqs = positions.double()[:, None] * (1.0 / (short * base))[None, :]
    original = scaling.get(
        "original_max_position_embeddings", raw["max_position_embeddings"]
    )
    factor = raw["max_position_embeddings"] / original
    attn_factor = (
        math.sqrt(1 + math.log(factor) / math.log(original)) if factor > 1.0 else 1.0
    )

    cos, sin = rotary(positions, torch.device("cpu"), torch.float32)
    assert cos.shape == (positions.shape[0], layout.half_d)
    torch.testing.assert_close(
        cos[:, :half].double(), freqs.cos() * attn_factor, rtol=0, atol=1e-3
    )
    torch.testing.assert_close(
        sin[:, :half].double(), freqs.sin() * attn_factor, rtol=0, atol=1e-3
    )
    assert torch.equal(cos[:, half:], torch.ones_like(cos[:, half:]))
    assert torch.equal(sin[:, half:], torch.zeros_like(sin[:, half:]))


# =============================================================================
# Checkpoint key coverage
# =============================================================================


def _mapped_keys(num_layers: int, tie_word_embeddings: bool) -> set[str]:
    stub = SimpleNamespace(
        model=SimpleNamespace(layers=[None] * num_layers),
        config=SimpleNamespace(tie_word_embeddings=tie_word_embeddings),
    )
    mappings = src_module("model").MiniCPM3ForCausalLM.checkpoint_mappings(stub)
    used: list[str] = []
    for value in mappings.values():
        used.extend([value] if isinstance(value, str) else value)
    return used


def test_checkpoint_key_coverage(one_layer_checkpoint):
    """The mappings read each tensor of the checkpoint, and no other key.

    ``lm_head.weight`` and ``embed_tokens`` share the one checkpoint key
    ``model.embed_tokens.weight`` (tied embeddings), so that key is the only
    one that two parameters read.
    """
    used = _mapped_keys(1, tie_word_embeddings=True)
    keys = set(U.load_checkpoint_state(one_layer_checkpoint))
    assert set(used) == keys, (
        f"Unread keys: {sorted(keys - set(used))[:8]}. "
        f"Missing keys: {sorted(set(used) - keys)[:8]}."
    )
    repeated = {k for k in used if used.count(k) > 1}
    assert repeated == {"model.embed_tokens.weight"}
    # 12 tensors per layer, plus the embedding and the final norm.
    assert len(keys) == 12 * 1 + 2


def test_checkpoint_key_coverage_real():
    """The real checkpoint has the 746 BF16 tensors that the mappings read."""
    checkpoint = U.local_checkpoint_or_skip()
    used = set(_mapped_keys(62, tie_word_embeddings=True))
    state = U.load_checkpoint_state(checkpoint)
    assert len(state) == 746
    assert used == set(state)
    assert all(t.dtype == torch.bfloat16 for t in state.values())


# =============================================================================
# src/checkpoints.py: the pytorch_model*.bin fallback
# =============================================================================


def _bin_dir(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, tmp_path / "pytorch_model.bin")
    return tmp_path


def test_checkpoint_loader_falls_back_to_torch_bin(tmp_path):
    """Without .safetensors files the loader reads pytorch_model*.bin only."""
    ckpt = src_module("checkpoints")
    w = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    root = _bin_dir(tmp_path / "bin", {"w": w, "b": torch.ones(3)})
    # A second .bin file that is not a weight file must stay out.
    torch.save({"lr": 1e-3}, root / "training_args.bin")

    loader = ckpt.SafetensorsCheckpoint(str(root))
    assert loader.get_num_files() == 1
    assert loader.get_tensor_names() == {"w", "b"}
    torch.testing.assert_close(loader._get_slice("w")[1:3, 2:], w[1:3, 2:])
    assert tuple(loader._get_slice("w").get_shape()) == (4, 6)


def test_checkpoint_loader_prefers_safetensors(tmp_path):
    """With a .safetensors file present, the loader ignores pytorch_model.bin."""
    ckpt = src_module("checkpoints")
    root = _bin_dir(tmp_path / "both", {"from_bin": torch.zeros(2)})
    save_file({"from_safetensors": torch.ones(2)}, root / "model.safetensors")

    loader = ckpt.SafetensorsCheckpoint(str(root))
    assert loader.get_tensor_names() == {"from_safetensors"}


def test_checkpoint_loader_without_weights(tmp_path):
    ckpt = src_module("checkpoints")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(AssertionError, match="pytorch_model"):
        ckpt.SafetensorsCheckpoint(str(tmp_path))


def test_torch_bin_file_rejects_non_state_dict(tmp_path):
    ckpt = src_module("checkpoints")
    path = tmp_path / "pytorch_model.bin"
    torch.save([torch.ones(2)], path)
    with pytest.raises(TypeError, match="state dict"):
        ckpt._TorchBinFile(str(path))


def test_load_sharded_from_torch_bin(tmp_path):
    """``load_sharded`` gives each rank its shard of a .bin tensor."""
    from vllm_neuron.utils.weight_loader import (
        set_weight_loader,
        sharding_weight_loader,
    )

    ckpt = src_module("checkpoints")
    world_size = 4
    full = torch.randn(8, 16, dtype=torch.bfloat16)
    root = _bin_dir(tmp_path / "shard", {"layer.weight": full})

    for rank in range(world_size):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.empty(8, 16 // world_size))
        set_weight_loader(
            module.weight,
            sharding_weight_loader(
                shard_dim=1, shard_size=16 // world_size, num_shards=world_size
            ),
        )
        result = ckpt.SafetensorsCheckpoint(str(root)).load_sharded(
            rank, world_size, module, {"weight": "layer.weight"}, "cpu"
        )
        assert result.missing_keys == [] and result.unexpected_keys == []
        expected = full[:, rank * 4 : (rank + 1) * 4]
        assert torch.equal(result.state_dict["weight"], expected)


# =============================================================================
# CPU component: fused MLA weight math (FP32, no kernels)
# =============================================================================


def _fused_attention_fp32(checkpoint: str, h: torch.Tensor) -> torch.Tensor:
    """Run layer LAYER_IDX attention with this folder's fused weights in FP32 torch."""
    model = src_module("model")

    cfg = U.neuron_model_config(U.load_hf_config(checkpoint))
    layout = model.minicpm3_head_layout(cfg)
    nh, d = cfg.num_attention_heads, layout.d_head
    st = {
        k: v.float()
        for k, v in U.load_prefixed_state(
            checkpoint, f"model.layers.{U.LAYER_IDX}.self_attn."
        ).items()
    }
    w_latent = model.build_latent_proj_weight(
        st["q_a_proj.weight"], st["kv_a_proj_with_mqa.weight"], layout
    )
    w_qkv = model.build_fused_qkv_weight(
        st["q_b_proj.weight"], st["kv_b_proj.weight"], layout, nh, cfg.qk_nope_head_dim
    )
    w_o = model.build_o_proj_weight(st["o_proj.weight"], layout, nh)

    def rms(x, w):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))

    tokens = h.shape[0]
    x = h @ w_latent
    c0, c1 = layout.c_offset, layout.k_pe_offset
    x = torch.cat(
        [
            rms(x[:, :c0], st["q_a_layernorm.weight"]),
            rms(x[:, c0:c1], st["kv_a_layernorm.weight"]),
            x[:, c1:],
        ],
        dim=-1,
    )
    q, k, v = (
        t.view(tokens, nh, d).transpose(0, 1) for t in (x @ w_qkv).split(nh * d, -1)
    )
    rotary = model.MiniCPM3RotaryEmbedding(cfg, layout)
    cos, sin = rotary(torch.arange(tokens), torch.device("cpu"), torch.float32)
    cos, sin = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)

    def rope(t):
        rot = torch.cat([-t[..., d // 2 :], t[..., : d // 2]], -1)
        return t * cos + rot * sin

    out = F.scaled_dot_product_attention(
        rope(q), rope(k), v, is_causal=True, scale=cfg.qk_head_dim**-0.5
    )
    return out.transpose(0, 1).reshape(tokens, nh * d) @ w_o


def test_fused_mla_weight_math_fp32(one_layer_checkpoint):
    """The fused latent, QKV and O weights reproduce the HF attention in FP32."""
    ref = U.hf_attention_reference(one_layer_checkpoint, seq_len=64, seed=SEED)
    with torch.no_grad():
        out = _fused_attention_fp32(one_layer_checkpoint, ref["input"].float())
    torch.testing.assert_close(out, ref["fp32"], rtol=1e-3, atol=1e-3)


def test_random_checkpoint_matches_published_format(one_layer_checkpoint):
    """The test checkpoint has the format of the published one (one zip .bin)."""
    names = sorted(os.listdir(one_layer_checkpoint))
    assert "pytorch_model.bin" in names
    assert not [n for n in names if n.endswith(".safetensors")]
    state = U.load_checkpoint_state(one_layer_checkpoint)
    assert all(t.dtype == torch.bfloat16 for t in state.values())
