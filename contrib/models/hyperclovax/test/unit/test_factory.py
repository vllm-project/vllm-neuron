# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the HyperCLOVAX factory, registration, config and model helpers.

The tests load no checkpoint and need no Neuron hardware. They check:

- The registration contract: ``src/`` exports the factory, the factory
  satisfies vLLM's model protocol, and the plugin entry point in
  ``pyproject.toml`` registers it.
- The factory selects the BF16 implementation and rejects an unsupported config.
- HyperCLOVAXConfig reads the MuP multipliers and the peri-layer norm flag. The
  compute dtype stays bfloat16 for the float32 checkpoint.
- get_weight_mappings covers each of the 421 checkpoint keys of the released
  checkpoint (38 layers x 11 keys + 3 keys) one time.
- The vendored ``hoist_rope_parameters`` in ``src/compat.py``.
- The O projection in ``src/o_proj.py``: the torch fallback accepts an fp32
  input with a bf16 weight and agrees with the NKI kernel (in the CPU
  simulator) on it, and everything else matches the plugin's ``NF.o_proj``.
- The decode path gives the decode kernel uint32 KV cache write slots.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hyperclovax/test/unit/test_factory.py -v
"""

import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from _local_src import src_module, src_package
from hyperclovax_test_utils import (
    ARCHITECTURE,
    HF_REVISION,
    TOP_LEVEL_KEYS,
    expected_checkpoint_keys,
)

_FOLDER = Path(__file__).resolve().parents[2]

# The fields of config.json of the released checkpoint at revision HF_REVISION.
# The model is text-only.
RELEASED_CONFIG = {
    "architectures": ["HyperCLOVAXForCausalLM"],
    "model_type": "hyperclovax",
    "vocab_size": 110592,
    "hidden_size": 6144,
    "intermediate_size": 14336,
    "num_hidden_layers": 38,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-05,
    "rope_theta": 100000000,
    "rope_scaling": None,
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,
    "attention_multiplier": 0.0078125,
    "embedding_multiplier": 10.0,
    "logits_scaling": 0.125,
    "residual_multiplier": 1.0,
    "use_post_norm": True,
    "torch_dtype": "float32",
}


def _hf_config(**overrides):
    values = dict(RELEASED_CONFIG)
    values.update(overrides)
    return SimpleNamespace(**values)


def _factory():
    return src_module("factory").HyperCLOVAXForCausalLM


class _FakeTPGroup:
    """Give the TP size and the rank without a process group."""

    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = None


# =============================================================================
# Registration
# =============================================================================


class TestHyperCLOVAXRegistration:
    """The checkpoint's architecture string must reach this folder's factory.

    A contrib model is deliberately absent from the plugin's
    ``vllm_neuron.model.registry.get_models()``. It reaches vLLM through the
    ``vllm.general_plugins`` entry point of ``pyproject.toml`` (option 3 of
    ``contrib/README.md``) or through a ``ModelRegistry.register_model`` call in
    the user's own script (option 1). These tests check both paths.
    """

    def test_src_exports_factory(self):
        assert getattr(src_package(), ARCHITECTURE) is _factory()
        assert RELEASED_CONFIG["architectures"] == [ARCHITECTURE]

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

    def test_registered_factory_is_text_generation_model(self, monkeypatch):
        """The registration replaces vLLM's own HyperCLOVAXForCausalLM."""
        from vllm import ModelRegistry

        monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
        ModelRegistry.register_model(ARCHITECTURE, _factory())
        info = ModelRegistry.models[ARCHITECTURE].inspect_model_cls()
        assert info.is_text_generation_model
        assert not info.is_pooling_model
        assert ModelRegistry.models[ARCHITECTURE].load_model_cls() is _factory()

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
        registered = ModelRegistry.models[ARCHITECTURE]
        assert registered.module_name == f"{package}.factory"
        assert registered.class_name == ARCHITECTURE
        assert hasattr(src_module("factory"), registered.class_name)

    def test_factory_forward_delegates(self):
        """The factory's forward passes the call through to the implementation."""
        factory = _factory().__new__(_factory())
        torch.nn.Module.__init__(factory)
        impl = MagicMock(return_value="out")
        factory._model = impl
        input_ids = torch.zeros(4, dtype=torch.long)
        positions = torch.arange(4)
        assert factory(input_ids, positions, attn_metadata="meta") == "out"
        impl.assert_called_once_with(
            input_ids=input_ids, positions=positions, attn_metadata="meta"
        )


# =============================================================================
# Factory
# =============================================================================


class TestHyperCLOVAXFactory:
    """The factory validates the config and selects the BF16 implementation."""

    @pytest.mark.parametrize("quantization", ["bf16", None])
    def test_selects_bf16_implementation(self, quantization):
        neuron_config = MagicMock()
        neuron_config.quantization = quantization
        hf_config = MagicMock()
        model_cls = src_module("model").HyperCLOVAXForCausalLM

        with (
            patch.object(_factory(), "_validate_config"),
            patch.object(model_cls, "from_configs") as mock_bf16,
        ):
            mock_bf16.return_value = MagicMock()
            _factory().from_configs(hf_config, neuron_config)
            mock_bf16.assert_called_once_with(hf_config, neuron_config)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, "float32"])
    def test_accepts_released_config(self, dtype):
        _factory()._validate_config(_hf_config(torch_dtype=dtype), None)

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"torch_dtype": torch.float16}, "float32 and bfloat16"),
            ({"attention_bias": True}, "attention_bias"),
            ({"mlp_bias": True}, "mlp_bias"),
            ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "default RoPE"),
            ({"rope_parameters": {"rope_type": "llama3"}}, "default RoPE"),
            ({"hidden_act": "gelu"}, "silu"),
        ],
    )
    def test_rejects_unsupported_config(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _factory()._validate_config(_hf_config(**overrides), None)


# =============================================================================
# Config
# =============================================================================


class TestHyperCLOVAXConfig:
    """HyperCLOVAXConfig reads the released config correctly."""

    def test_released_config_fields(self):
        HyperCLOVAXConfig = src_module("config").HyperCLOVAXConfig

        cfg = HyperCLOVAXConfig.from_configs(dict(RELEASED_CONFIG))
        assert cfg.num_hidden_layers == 38
        assert cfg.hidden_size == 6144
        assert cfg.intermediate_size == 14336
        assert cfg.num_attention_heads == 48
        assert cfg.num_key_value_heads == 8
        assert cfg.head_dim == 128
        assert cfg.vocab_size == 110592
        assert cfg.rope_theta == 100000000
        assert cfg.rms_norm_eps == 1e-5
        assert cfg.tie_word_embeddings is False
        assert cfg.attention_multiplier == 0.0078125
        assert cfg.embedding_multiplier == 10.0
        assert cfg.logits_scaling == 0.125
        assert cfg.residual_multiplier == 1.0
        assert cfg.use_post_norm is True
        # The checkpoint is float32. The model computes in bfloat16.
        assert cfg.torch_dtype == torch.bfloat16

    def test_attention_multiplier_default(self):
        HyperCLOVAXConfig = src_module("config").HyperCLOVAXConfig

        values = dict(RELEASED_CONFIG)
        values.pop("attention_multiplier")
        cfg = HyperCLOVAXConfig.from_configs(values)
        assert cfg.attention_multiplier == pytest.approx(128**-0.5)

    def test_hf_config_object(self):
        """transformers 5 keeps rope_theta under rope_parameters."""
        from transformers import AutoConfig

        HyperCLOVAXConfig = src_module("config").HyperCLOVAXConfig

        values = {
            k: v
            for k, v in RELEASED_CONFIG.items()
            if k not in ("architectures", "model_type", "torch_dtype", "rope_theta")
        }
        hf_config = AutoConfig.for_model(
            "hyperclovax",
            rope_parameters={"rope_type": "default", "rope_theta": 100000000.0},
            **values,
        )
        cfg = HyperCLOVAXConfig.from_configs(hf_config)
        assert cfg.rope_theta == 100000000.0
        assert cfg.attention_multiplier == 0.0078125
        assert cfg.use_post_norm is True

    def test_pinned_revision_is_a_full_sha(self):
        assert len(HF_REVISION) == 40
        int(HF_REVISION, 16)


# =============================================================================
# Weight mappings
# =============================================================================


class TestHyperCLOVAXWeightMappings:
    """get_weight_mappings covers each checkpoint key one time."""

    @staticmethod
    def _mappings(num_layers: int, use_post_norm: bool = True, tie: bool = False):
        model_cls = src_module("model").HyperCLOVAXForCausalLM

        fake = SimpleNamespace(
            model=SimpleNamespace(layers=[None] * num_layers),
            config=SimpleNamespace(
                use_post_norm=use_post_norm, tie_word_embeddings=tie
            ),
        )
        return model_cls.get_weight_mappings(fake)

    def test_covers_released_checkpoint_keys(self):
        mappings = self._mappings(38)
        sources: list[str] = []
        for value in mappings.values():
            sources.extend(value if isinstance(value, list) else [value])
        # A parameter without a map entry loads from the key with its own name.
        sources.extend(TOP_LEVEL_KEYS)

        expected = expected_checkpoint_keys(38)
        assert len(expected) == 421
        assert len(sources) == len(set(sources)), "A checkpoint key maps two times."
        assert set(sources) == expected

    def test_fused_qkv_order(self):
        mappings = self._mappings(1)
        assert mappings["model.layers.0.self_attn.qkv_proj_weight"] == [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ]

    def test_tied_embeddings_map_lm_head(self):
        mappings = self._mappings(1, tie=True)
        assert mappings["lm_head.weight"] == "model.embed_tokens.weight"

    def test_no_post_norm_keys_without_post_norm(self):
        mappings = self._mappings(1, use_post_norm=False)
        assert not any("post_norm" in k for k in mappings)


# =============================================================================
# Vendored 0.29 helper (src/compat.py)
# =============================================================================


class TestCompat:
    """``hoist_rope_parameters`` gives the pre-v5 RoPE keys to the dataclass."""

    def test_hoists_rope_theta(self):
        hoist = src_module("compat").hoist_rope_parameters

        out = hoist({"rope_parameters": {"rope_type": "default", "rope_theta": 1e8}})
        assert out["rope_theta"] == 1e8
        assert out.get("rope_scaling") is None

    def test_top_level_keys_win(self):
        hoist = src_module("compat").hoist_rope_parameters

        out = hoist({"rope_theta": 5.0, "rope_parameters": {"rope_theta": 1e8}})
        assert out["rope_theta"] == 5.0

    def test_non_default_type_becomes_rope_scaling(self):
        hoist = src_module("compat").hoist_rope_parameters

        params = {"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0}
        out = hoist({"rope_parameters": params}, ("yarn",))
        assert out["rope_scaling"] == {"rope_type": "yarn", "factor": 4.0}

    def test_no_rope_parameters_is_unchanged(self):
        hoist = src_module("compat").hoist_rope_parameters

        config = {"rope_theta": 1e8}
        assert hoist(config) is config


# =============================================================================
# O projection (src/o_proj.py)
# =============================================================================


def _o_proj_inputs(dtype: torch.dtype, B=1, N=6, D=128, S=16, H=256, seed=0):
    torch.manual_seed(seed)
    active = torch.randn(B, N, D, S, dtype=torch.float32).to(dtype)
    weight = (torch.randn(N * D, H, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    return active, weight


class TestOProj:
    """``src/o_proj.py`` is the plugin's ``NF.o_proj`` with an fp32-input fallback."""

    def test_fallback_accepts_fp32_input(self):
        local = src_module("o_proj")

        active, weight = _o_proj_inputs(torch.float32)
        out = local._torch_o_proj_impl(active, weight)
        assert out.dtype == torch.float32
        assert out.shape == (1, 16, 256)

        # Cast the input to the weight dtype, then multiply in fp32.
        x = active.to(torch.bfloat16).float().permute(0, 3, 1, 2).reshape(1, 16, -1)
        expected = x @ weight.float()
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_plugin_fallback_rejects_fp32_input(self):
        """The reason for the local copy: the plugin's fallback fails on it."""
        from vllm_neuron.functional.attention import o_proj as plugin

        active, weight = _o_proj_inputs(torch.float32)
        with pytest.raises(RuntimeError):
            plugin._torch_o_proj_impl(active, weight)

    @pytest.mark.parametrize("shape", [(1, 6, 128, 16), (2, 4, 64, 8)])
    def test_same_dtype_matches_plugin(self, shape):
        from vllm_neuron.functional.attention import o_proj as plugin

        local = src_module("o_proj")
        B, N, D, S = shape
        active, weight = _o_proj_inputs(torch.bfloat16, B=B, N=N, D=D, S=S)
        bias = torch.randn(1, 256, dtype=torch.bfloat16)
        torch.testing.assert_close(
            local._torch_o_proj_impl(active, weight, bias),
            plugin._torch_o_proj_impl(active, weight, bias),
            rtol=0,
            atol=0,
        )
        flat = active.permute(0, 3, 1, 2).reshape(B, S, N * D)
        torch.testing.assert_close(
            local._torch_o_proj_impl(flat, weight),
            plugin._torch_o_proj_impl(flat, weight),
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize(
        "active_shape, weight_shape",
        [
            ((1, 6, 128, 2048), (768, 6144)),  # TP 8 prefill of the 14B model
            ((1, 48, 128, 128), (6144, 6144)),  # TP 1: too many heads
            ((1, 2, 256, 128), (512, 512)),  # head_dim > 128
            ((1, 6, 128, 16), (640, 6144)),  # N * D != weight rows
        ],
    )
    def test_kernel_eligibility_matches_plugin(
        self, monkeypatch, active_shape, weight_shape
    ):
        """The local dispatch selects the kernel exactly when the plugin does."""
        from vllm_neuron.functional.attention import o_proj as plugin

        local = src_module("o_proj")
        monkeypatch.setattr(local, "can_run_kernel", lambda *_: True)
        monkeypatch.setattr(plugin, "can_run_kernel", lambda *_: True)
        active = torch.empty(active_shape, dtype=torch.float32, device="meta")
        weight = torch.empty(weight_shape, dtype=torch.bfloat16, device="meta")
        assert local._can_use_kernel(active, weight) == plugin._can_use_kernel(
            active, weight
        )

    @pytest.mark.parametrize("shape", [(2, 128, 512), (6, 256, 1024)])
    def test_kernel_matches_fallback_in_simulator(self, monkeypatch, shape):
        """The NKI kernel (CPU simulator) and the fallback agree on an fp32 input.

        This is the input that the model gives the O projection in prefill. The
        kernel writes an fp32 output, and so does the fallback.
        """
        import os

        if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
            pytest.skip("The NKI simulator runs in CPU mode (VLLM_NEURON_CPU_MODE=1).")
        monkeypatch.setenv("NKI_SIMULATOR", "1")
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
        local = src_module("o_proj")

        N, S, H = shape
        torch.manual_seed(0)
        active = torch.randn(1, N, 128, S, dtype=torch.float32) * 0.05
        weight = (torch.randn(N * 128, H, dtype=torch.float32) * 0.02).to(
            torch.bfloat16
        )
        assert local._can_use_kernel(active, weight)
        kernel_out = local.o_proj(active, weight, None)
        assert kernel_out.dtype == torch.float32
        torch.testing.assert_close(
            kernel_out, local._torch_o_proj_impl(active, weight), rtol=1e-5, atol=1e-6
        )

    def test_o_proj_falls_back_without_kernel(self, monkeypatch):
        local = src_module("o_proj")
        monkeypatch.setattr(local, "can_run_kernel", lambda *_: False)

        active, weight = _o_proj_inputs(torch.float32)
        out = local.o_proj(active, weight, None)
        torch.testing.assert_close(out, local._torch_o_proj_impl(active, weight))


# =============================================================================
# Decode kernel write slots
# =============================================================================


def _tiny_model_config():
    HyperCLOVAXConfig = src_module("config").HyperCLOVAXConfig

    return HyperCLOVAXConfig(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        attention_multiplier=0.0078125,
    )


def test_decode_passes_uint32_cache_update_idx(monkeypatch):
    """The decode NEFF rejects int64 write slots, so the model casts to uint32."""
    model = src_module("model")
    monkeypatch.setattr(model, "get_tp_group", lambda: _FakeTPGroup())

    attn = model.HyperCLOVAXAttention(_tiny_model_config(), layer_idx=0)
    attn.k_cache = torch.zeros(4, 2, 32, 128, dtype=torch.bfloat16)
    attn.v_cache = torch.zeros(4, 2, 32, 128, dtype=torch.bfloat16)

    captured = {}

    def fake_attention_decode(**kwargs):
        captured.update(kwargs)
        return torch.zeros(kwargs["X"].shape, dtype=torch.bfloat16)

    monkeypatch.setattr(model.NF, "attention_decode", fake_attention_decode)
    monkeypatch.setattr(
        model.NF, "gen_attention_decode_mask", lambda **kwargs: torch.zeros(1)
    )

    B = 2
    attn_metadata = {
        "layers.0.self_attn": {
            "slot_mapping": torch.tensor([5, 40], dtype=torch.int64),
            "block_size": 32,
            "max_blocks_per_seq": 2,
            "block_table_tensor": torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
            "max_query_len": 1,
            "decode_token_threshold": 1,
        }
    }
    hidden = torch.randn(B, 512, dtype=torch.bfloat16)
    positions = torch.tensor([5, 8], dtype=torch.int32)
    rotary = model.HyperCLOVAXRotaryEmbedding(_tiny_model_config())
    cos_sin = rotary(positions, device=hidden.device, dtype=hidden.dtype)

    out = attn(hidden, positions, cos_sin, attn_metadata=attn_metadata)
    assert out.shape == (B, 1, 512)
    idx = captured["kv_cache_update_idx"]
    assert idx.dtype == torch.uint32
    assert idx.shape == (B, 1)
    assert idx.to(torch.int64).flatten().tolist() == [5, 40]
    assert captured["update_cache"] is True
    assert captured["softmax_scale"] == pytest.approx(0.0078125)
