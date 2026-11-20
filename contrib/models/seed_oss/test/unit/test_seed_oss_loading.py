# SPDX-License-Identifier: Apache-2.0
"""Seed-OSS registration, config and loading tests (CPU group).

These tests need no Neuron device and no Seed-OSS-36B weights. They check:

1. Registration. Registering the contrib factory with vLLM makes the
   architecture ``SeedOssForCausalLM`` resolve to it, and the factory resolves
   to the implementation in ``src/model.py``.
2. Config mapping. The published Seed-OSS-36B ``config.json`` maps to
   ``SeedOssConfig`` with all checkpoint semantics.
3. Construction at the device geometry. The full 64-layer model is built on
   the meta device at TP8 for each rank. The test checks the parameter shapes
   and that the checkpoint mappings cover the 771 published checkpoint keys.
4. Loader layout. The fused QKV bias loader agrees with the fused QKV weight
   loader for each TP8 rank.

The per-component accuracy checks against a HuggingFace reference live next
door in ``test/module/test_seed_oss_components.py``, which is where
``contrib/README.md`` puts component tests.

Run with:
    VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native pytest \
        contrib/models/seed_oss/test/unit/ -v
"""

import contextlib
import importlib.util
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` helper by file location.

    The tests must keep importing after the folder is copied out of the
    repository, so they resolve ``src/`` from its own path instead of through
    a ``contrib.models.seed_oss.src`` namespace path that needs the repository
    root on ``sys.path``. See ``test/_local_src.py``.
    """
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    name = "contrib_seed_oss_local_src"
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

# The module name the loader registers the shipped implementation under.
# ``mock.patch`` needs it as a string, and so does the ``__module__`` check.
_SRC = _local_src.SRC_PACKAGE
_SRC_MODEL = f"{_SRC}.model"

pytestmark = [pytest.mark.fast]

RANDOM_SEED = 42
DEVICE_TP = 8

# Published config.json of ByteDance-Seed/Seed-OSS-36B-Instruct (revision
# 497f1dca95ebdec98e41d517b9f060ee753c902f).
SEED_OSS_36B_CONFIG = {
    "architectures": ["SeedOssForCausalLM"],
    "attention_bias": True,
    "attention_dropout": 0.1,
    "attention_out_bias": False,
    "bos_token_id": 0,
    "pad_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 27648,
    "max_position_embeddings": 524288,
    "mlp_bias": False,
    "model_type": "seed_oss",
    "num_attention_heads": 80,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "residual_dropout": 0.1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {"rope_type": "default"},
    "rope_theta": 10000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 155136,
}

# Number of tensors in model.safetensors.index.json of the published checkpoint.
SEED_OSS_36B_NUM_CHECKPOINT_KEYS = 771


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def _published_checkpoint_keys(num_layers: int) -> set[str]:
    """Return the tensor names of the published Seed-OSS checkpoint."""
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(num_layers):
        p = f"model.layers.{i}"
        keys |= {
            f"{p}.input_layernorm.weight",
            f"{p}.post_attention_layernorm.weight",
            f"{p}.mlp.gate_proj.weight",
            f"{p}.mlp.up_proj.weight",
            f"{p}.mlp.down_proj.weight",
            f"{p}.self_attn.q_proj.weight",
            f"{p}.self_attn.k_proj.weight",
            f"{p}.self_attn.v_proj.weight",
            f"{p}.self_attn.o_proj.weight",
            f"{p}.self_attn.q_proj.bias",
            f"{p}.self_attn.k_proj.bias",
            f"{p}.self_attn.v_proj.bias",
        }
    return keys


@contextlib.contextmanager
def _fake_tp(world_size: int, rank: int):
    """Give the model a TP group of ``world_size`` without a process group."""
    group = MagicMock()
    group.world_size = world_size
    group.rank_in_group = rank
    group.device_group = MagicMock()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(f"{_SRC_MODEL}.get_tp_group", return_value=group))
        if world_size > 1:
            stack.enter_context(
                patch("torch.distributed.is_initialized", return_value=True)
            )
            stack.enter_context(
                patch("torch.distributed.get_world_size", return_value=world_size)
            )
            stack.enter_context(patch("torch.distributed.get_rank", return_value=rank))
        yield group


class _TensorSlice:
    """Minimal PySafeSlice stand-in over a torch tensor."""

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor

    def get_shape(self):
        return list(self._t.shape)

    def __getitem__(self, idx):
        return self._t[idx]


# =============================================================================
# 1. Registration and factory
# =============================================================================


class TestSeedOssRegistration:
    def test_arch_resolves_after_registration(self):
        """A contrib model is registered by the user, not by the plugin.

        ``vllm_neuron.model.registry.get_models()`` does not list a contrib
        model by design, so the architecture resolves only after the user
        registers the factory (contrib/README.md, "How do I register a custom
        model with vLLM?"). This test does that registration, the same way the
        integration test does.
        """
        from vllm import ModelRegistry

        SeedOssForCausalLM = _local_src.load().SeedOssForCausalLM

        ModelRegistry.register_model("SeedOssForCausalLM", SeedOssForCausalLM)
        resolved = ModelRegistry._try_load_model_cls("SeedOssForCausalLM")
        assert resolved is SeedOssForCausalLM, f"Resolved class is {resolved!r}"

    def test_factory_resolves_explicit_neuron_class(self):
        """The factory must build the Neuron implementation class."""
        Factory = _local_src.load("factory").SeedOssForCausalLM
        Impl = _local_src.load("model").SeedOssForCausalLM

        with _fake_tp(DEVICE_TP, 0), torch.device("meta"):
            model = Factory.from_configs(dict(SEED_OSS_36B_CONFIG), None)
        assert type(model) is Impl
        assert type(model).__module__ == _SRC_MODEL
        assert len(model.model.layers) == SEED_OSS_36B_CONFIG["num_hidden_layers"]

    @pytest.mark.parametrize(
        "override, match",
        [
            ({"attention_out_bias": True}, "attention_out_bias"),
            ({"mlp_bias": True}, "mlp_bias"),
            ({"hidden_act": "gelu"}, "hidden_act"),
            ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "rope_type"),
            ({"torch_dtype": "float16"}, "bfloat16"),
        ],
    )
    def test_factory_rejects_unsupported_config(self, override, match):
        Factory = _local_src.load("factory").SeedOssForCausalLM

        cfg = dict(SEED_OSS_36B_CONFIG)
        cfg.update(override)
        with pytest.raises(ValueError, match=match):
            Factory._validate_config(cfg, None)


# =============================================================================
# 2. Config mapping
# =============================================================================


class TestSeedOssConfig:
    def test_published_config_maps_all_fields(self):
        SeedOssConfig = _local_src.load("config").SeedOssConfig

        cfg = SeedOssConfig.from_configs(dict(SEED_OSS_36B_CONFIG))
        assert cfg.vocab_size == 155136
        assert cfg.hidden_size == 5120
        assert cfg.intermediate_size == 27648
        assert cfg.num_hidden_layers == 64
        assert cfg.num_attention_heads == 80
        assert cfg.num_key_value_heads == 8
        assert cfg.head_dim == 128
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.rope_theta == 10000000.0
        assert cfg.attention_bias is True
        assert cfg.attention_out_bias is False
        assert cfg.mlp_bias is False
        assert cfg.tie_word_embeddings is False
        assert cfg.torch_dtype == torch.bfloat16

    def test_hf_config_object_maps_rope_theta(self):
        """A transformers config object must give the same RoPE theta."""
        from transformers import PretrainedConfig

        SeedOssConfig = _local_src.load("config").SeedOssConfig

        hf_cfg = PretrainedConfig(**SEED_OSS_36B_CONFIG)
        cfg = SeedOssConfig.from_configs(hf_cfg)
        assert cfg.rope_theta == 10000000.0
        assert cfg.attention_bias is True
        assert cfg.head_dim == 128


# =============================================================================
# 3. Construction at the device geometry (TP8, meta device)
# =============================================================================


class TestSeedOssTP8Construction:
    @pytest.mark.parametrize("rank", [0, DEVICE_TP - 1])
    def test_tp8_shapes_and_checkpoint_coverage(self, rank):
        SeedOssForCausalLM = _local_src.load("model").SeedOssForCausalLM

        with _fake_tp(DEVICE_TP, rank), torch.device("meta"):
            model = SeedOssForCausalLM.from_configs(dict(SEED_OSS_36B_CONFIG), None)

        H, I, V, Dh = 5120, 27648, 155136, 128  # noqa: E741
        q_per_rank = 80 // DEVICE_TP * Dh
        kv_per_rank = 1 * Dh
        qkv = q_per_rank + 2 * kv_per_rank

        attn = model.model.layers[0].self_attn
        assert attn.num_attention_heads_per_rank == 10
        assert attn.num_key_value_heads_per_rank == 1
        assert attn.num_kv_replicas == 1
        assert tuple(attn.qkv_proj_weight.shape) == (H, qkv)
        assert tuple(attn.qkv_proj_bias.shape) == (qkv,)
        assert tuple(attn.o_proj_weight.shape) == (q_per_rank, H)

        mlp = model.model.layers[0].mlp
        assert tuple(mlp.gate_proj_weight.shape) == (H, I // DEVICE_TP)
        assert tuple(mlp.up_proj_weight.shape) == (H, I // DEVICE_TP)
        assert tuple(mlp.down_proj_weight.shape) == (I // DEVICE_TP, H)

        assert tuple(model.model.embed_tokens.weight.shape) == (V // DEVICE_TP, H)
        assert tuple(model.lm_head.weight.shape) == (V // DEVICE_TP, H)

        # Every parameter has a mapping, and every mapping names a parameter.
        mappings = model.checkpoint_mappings()
        param_names = {n for n, _ in model.named_parameters()}
        assert set(mappings) == param_names

        # The mapped checkpoint keys are exactly the published checkpoint keys.
        mapped_keys = set()
        for v in mappings.values():
            mapped_keys |= set(v) if isinstance(v, list) else {v}
        expected = _published_checkpoint_keys(64)
        assert len(expected) == SEED_OSS_36B_NUM_CHECKPOINT_KEYS
        assert mapped_keys == expected

        # The dispatch log covers the first and the last decoder layer.
        logged = [
            i
            for i, layer in enumerate(model.model.layers)
            if layer.self_attn.log_dispatch
        ]
        assert logged == [0, 63]

        kv_spec = model.get_kv_spec()
        assert len(kv_spec.layers) == 64
        assert kv_spec.layers[0].num_kv_heads == 1
        assert kv_spec.layers[0].head_size == Dh

    def test_tp_size_that_does_not_divide_heads_raises(self):
        SeedOssForCausalLM = _local_src.load("model").SeedOssForCausalLM

        with _fake_tp(3, 0), torch.device("meta"):
            with pytest.raises(ValueError, match="not divisible"):
                SeedOssForCausalLM.from_configs(dict(SEED_OSS_36B_CONFIG), None)


# =============================================================================
# 4. Loader layout
# =============================================================================


class TestSeedOssLoaders:
    def test_bias_loader_matches_weight_loader_rows(self):
        """Rank r gets the same Q/K/V rows for the bias and for the weight."""
        from vllm_neuron.utils.weight_loader import fused_qkv_weight_loader

        fused_qkv_bias_loader = _local_src.load("model").fused_qkv_bias_loader

        Dh, nq, nkv, hidden = 128, 80, 8, 4
        q_size, kv_size = nq // DEVICE_TP * Dh, nkv // DEVICE_TP * Dh

        q_w = torch.randn(nq * Dh, hidden)
        k_w = torch.randn(nkv * Dh, hidden)
        v_w = torch.randn(nkv * Dh, hidden)
        q_b = torch.randn(nq * Dh)
        k_b = torch.randn(nkv * Dh)
        v_b = torch.randn(nkv * Dh)

        w_loader = fused_qkv_weight_loader(
            q_size=q_size,
            kv_size=kv_size,
            shard_dim=1,
            num_shards=DEVICE_TP,
            is_storage_transposed=True,
        )
        b_loader = fused_qkv_bias_loader(q_size=q_size, kv_size=kv_size)

        for rank in range(DEVICE_TP):
            w = w_loader.load([_TensorSlice(t) for t in (q_w, k_w, v_w)], rank)
            b = b_loader.load([_TensorSlice(t) for t in (q_b, k_b, v_b)], rank)
            assert tuple(w.shape) == (hidden, q_size + 2 * kv_size)
            assert tuple(b.shape) == (q_size + 2 * kv_size,)

            # Bias row j must belong to the same output channel as weight col j.
            q_rows = slice(rank * q_size, (rank + 1) * q_size)
            kv_rows = slice(rank * kv_size, (rank + 1) * kv_size)
            ref_w = torch.cat([q_w[q_rows], k_w[kv_rows], v_w[kv_rows]], dim=0).T
            ref_b = torch.cat([q_b[q_rows], k_b[kv_rows], v_b[kv_rows]], dim=0)
            torch.testing.assert_close(w, ref_w)
            torch.testing.assert_close(b, ref_b)

    def test_bias_loader_kv_replicas(self):
        """With KV replicas, ranks that share a KV head get the same K/V bias."""
        fused_qkv_bias_loader = _local_src.load("model").fused_qkv_bias_loader

        q_b, k_b, v_b = torch.arange(16.0), torch.arange(4.0), torch.arange(4.0) + 10
        loader = fused_qkv_bias_loader(q_size=4, kv_size=2, num_kv_replicas=2)
        slices = [_TensorSlice(t) for t in (q_b, k_b, v_b)]
        r0, r1, r2 = (loader.load(slices, r) for r in (0, 1, 2))
        torch.testing.assert_close(r0[4:], r1[4:])
        torch.testing.assert_close(r0[:4], q_b[0:4])
        torch.testing.assert_close(r1[:4], q_b[4:8])
        torch.testing.assert_close(r2[4:6], k_b[2:4])
