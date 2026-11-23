# SPDX-License-Identifier: Apache-2.0
"""Loading and component tests for EXAONE 4.0 on the native Neuron route.

These tests use the real ``LGAI-EXAONE/EXAONE-4.0-32B`` checkpoint. They do
not run a forward pass of the full model and they do not need Neuron hardware.

The tests check these items:

- Registering the contrib factory with the vLLM model registry makes the HF
  architecture string resolve to it, in place of vLLM's own EXAONE 4.0 class.
- The parsed config agrees with the HF config: hybrid layer types, the NoPE
  full layers, Llama3 RoPE scaling and the untied LM head.
- The weight mappings use each checkpoint tensor one time. Each mapped key
  is in the checkpoint, with the expected shape and dtype.
- The TP8 loaders that the model attaches give the same tensors as an
  independent slice of the real checkpoint tensors. This includes the zero
  padding of the MLP intermediate dimension (3424 to 3456 for each rank).
- The padded MLP gives the same output as the unpadded checkpoint MLP.

The RoPE component is checked against the HuggingFace implementation in
``../module/test_exaone4_rope.py``.

The tests read the real checkpoint, so they need it on disk. Set
``EXAONE4_CHECKPOINT_PATH`` to its directory; without it the tests skip.

Run with:
    VLLM_NEURON_CPU_MODE=1 EXAONE4_CHECKPOINT_PATH=/path/to/EXAONE-4.0-32B \
        pytest contrib/models/exaone4/test/unit/test_exaone4_loading.py -v
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open


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

# The package name the shipped modeling code (``../../src/``) is loaded under.
# Everything that names a module path derives from it.
_SRC = _local_src.SRC_PACKAGE

pytestmark = [pytest.mark.fast]

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-32B"
TP = 8
# One sliding-window layer (RoPE) and one full layer (NoPE).
LAYERS_UNDER_TEST = (0, 3)
RANKS_UNDER_TEST = (0, 3, 7)


@pytest.fixture(scope="module")
def checkpoint_dir():
    """Return the local checkpoint directory, or skip without one.

    The port resolved the checkpoint through a shared test helper that does
    not ship here. A contrib model keeps its tests self-contained, so this
    copy reads ``EXAONE4_CHECKPOINT_PATH`` and skips when it is not a usable
    checkpoint directory. These tests read real tensors, so they cannot run
    without it.
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
def neuron_config_obj(hf_config):
    from vllm_neuron.model.neuron_config import NeuronConfig

    config_module = _local_src.load("config")

    return config_module.Exaone4Config.from_configs(hf_config, NeuronConfig())


@pytest.fixture(scope="module")
def weight_map(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


class _CheckpointSlices:
    """Open each checkpoint tensor through safetensors slices."""

    def __init__(self, checkpoint_dir: str, weight_map: dict[str, str]):
        self._dir = checkpoint_dir
        self._weight_map = weight_map
        self._handles = {}

    def slice(self, key: str):
        fname = self._weight_map[key]
        if fname not in self._handles:
            self._handles[fname] = safe_open(
                os.path.join(self._dir, fname), framework="pt"
            )
        return self._handles[fname].get_slice(key)

    def tensor(self, key: str) -> torch.Tensor:
        return self.slice(key)[:]


@pytest.fixture(scope="module")
def ckpt(checkpoint_dir, weight_map):
    return _CheckpointSlices(checkpoint_dir, weight_map)


def _fake_tp_group(rank: int):
    # The constructors read only the world size and the rank from the group.
    return SimpleNamespace(world_size=TP, rank_in_group=rank)


# =============================================================================
# Registry and config
# =============================================================================


def test_registry_resolves_neuron_class():
    """The architecture string must resolve to the contrib factory class.

    A contrib model is not in ``vllm_neuron.model.registry.get_models()`` by
    design, so the user registers it (contrib/README.md, "How do I register a
    custom model with vLLM?"). vLLM ships its own ``Exaone4ForCausalLM``, so
    the registration must replace it. This test registers the local factory
    the same way the integration test does.
    """
    from vllm import ModelRegistry

    Factory = _local_src.load().Exaone4ForCausalLM

    assert Factory is _local_src.load("factory").Exaone4ForCausalLM

    ModelRegistry.register_model("Exaone4ForCausalLM", Factory)
    resolved = ModelRegistry._try_load_model_cls("Exaone4ForCausalLM")
    assert resolved is Factory, f"Resolved class is {resolved!r}"
    assert resolved.__module__ == f"{_SRC}.factory"


def test_config_matches_hf(hf_config, neuron_config_obj):
    """The parsed config must keep the hybrid layout and the HF semantics."""
    cfg = neuron_config_obj
    assert hf_config.architectures == ["Exaone4ForCausalLM"]
    assert cfg.num_hidden_layers == hf_config.num_hidden_layers == 64
    assert cfg.hidden_size == hf_config.hidden_size == 5120
    assert cfg.num_attention_heads == 40
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == hf_config.hidden_size // hf_config.num_attention_heads
    assert cfg.intermediate_size == hf_config.intermediate_size == 27392
    assert cfg.vocab_size == hf_config.vocab_size
    assert cfg.rms_norm_eps == hf_config.rms_norm_eps
    assert cfg.tie_word_embeddings is False
    assert cfg.torch_dtype == torch.bfloat16

    assert cfg.sliding_window == hf_config.sliding_window == 4096
    assert cfg.layer_types == list(hf_config.layer_types)
    for idx, layer_type in enumerate(hf_config.layer_types):
        hf_sliding = layer_type == "sliding_attention"
        assert cfg.is_sliding(idx) == hf_sliding
        # HF applies RoPE when there is no window or on a sliding layer.
        assert cfg.uses_rope(idx) == (hf_config.sliding_window is None or hf_sliding)
    assert [i for i in range(64) if not cfg.is_sliding(i)] == list(range(3, 64, 4))

    rope = hf_config.rope_parameters
    assert cfg.rope_theta == rope["rope_theta"]
    assert cfg.rope_scaling["rope_type"] == rope["rope_type"] == "llama3"
    for key in (
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
    ):
        assert cfg.rope_scaling[key] == rope[key], key


# =============================================================================
# Weight coverage
# =============================================================================


def test_weight_mappings_cover_checkpoint(neuron_config_obj, weight_map, ckpt):
    """Each checkpoint tensor must have one mapping, with the expected shape."""
    get_weight_mappings = _local_src.load("model").get_weight_mappings

    cfg = neuron_config_obj
    mappings = get_weight_mappings(cfg.num_hidden_layers)

    mapped_keys = []
    for value in mappings.values():
        mapped_keys.extend(value if isinstance(value, list) else [value])
    assert len(mapped_keys) == len(set(mapped_keys)), "A key is mapped two times"

    checkpoint_keys = set(weight_map)
    missing = sorted(set(mapped_keys) - checkpoint_keys)
    unused = sorted(checkpoint_keys - set(mapped_keys))
    assert not missing, f"Mapped keys are not in the checkpoint: {missing[:10]}"
    assert not unused, f"Checkpoint keys have no mapping: {unused[:10]}"
    assert len(checkpoint_keys) == 707

    h, d = cfg.hidden_size, cfg.head_dim
    q_out = cfg.num_attention_heads * d
    kv_out = cfg.num_key_value_heads * d
    expected = {
        "model.embed_tokens.weight": (cfg.vocab_size, h),
        "lm_head.weight": (cfg.vocab_size, h),
        "model.norm.weight": (h,),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        expected.update(
            {
                f"{p}.self_attn.q_proj.weight": (q_out, h),
                f"{p}.self_attn.k_proj.weight": (kv_out, h),
                f"{p}.self_attn.v_proj.weight": (kv_out, h),
                f"{p}.self_attn.o_proj.weight": (h, q_out),
                f"{p}.self_attn.q_norm.weight": (d,),
                f"{p}.self_attn.k_norm.weight": (d,),
                f"{p}.post_attention_layernorm.weight": (h,),
                f"{p}.post_feedforward_layernorm.weight": (h,),
                f"{p}.mlp.gate_proj.weight": (cfg.intermediate_size, h),
                f"{p}.mlp.up_proj.weight": (cfg.intermediate_size, h),
                f"{p}.mlp.down_proj.weight": (h, cfg.intermediate_size),
            }
        )
    assert set(expected) == checkpoint_keys
    for key, shape in expected.items():
        sl = ckpt.slice(key)
        assert tuple(sl.get_shape()) == shape, f"{key}: {sl.get_shape()} != {shape}"
        assert sl.get_dtype() == "BF16", f"{key}: dtype {sl.get_dtype()}"


# =============================================================================
# TP8 loader transforms against independent slices
# =============================================================================


@pytest.mark.parametrize("layer_idx", LAYERS_UNDER_TEST)
def test_tp8_attention_loaders_match_checkpoint(
    neuron_config_obj, ckpt, monkeypatch, layer_idx
):
    """The fused QKV and o_proj loaders must give the TP8 head shards."""
    from vllm_neuron.utils.weight_loader import get_weight_loader

    # ``m`` is the module object, so ``monkeypatch.setattr`` below can replace
    # the ``get_tp_group`` that the component constructors call.
    m = _local_src.load("model")

    cfg = neuron_config_obj
    d = cfg.head_dim
    p = f"model.layers.{layer_idx}.self_attn"
    q = ckpt.tensor(f"{p}.q_proj.weight")
    k = ckpt.tensor(f"{p}.k_proj.weight")
    v = ckpt.tensor(f"{p}.v_proj.weight")
    o = ckpt.tensor(f"{p}.o_proj.weight")
    q_heads = cfg.num_attention_heads // TP
    # 8 KV heads at TP8: each rank holds KV head ``rank``.
    assert cfg.num_key_value_heads == TP

    for rank in RANKS_UNDER_TEST:
        monkeypatch.setattr(m, "get_tp_group", lambda r=rank: _fake_tp_group(r))
        attn = m.Exaone4Attention(cfg, layer_idx)
        assert attn.num_attention_heads_per_rank == q_heads == 5
        assert attn.num_key_value_heads_per_rank == 1
        assert attn.sliding_window == (4096 if layer_idx % 4 != 3 else None)

        qkv_slices = [ckpt.slice(f"{p}.{n}_proj.weight") for n in ("q", "k", "v")]
        got_qkv = get_weight_loader(attn.qkv_proj_weight).load(qkv_slices, rank)
        want_qkv = torch.cat(
            [
                q[rank * q_heads * d : (rank + 1) * q_heads * d],
                k[rank * d : (rank + 1) * d],
                v[rank * d : (rank + 1) * d],
            ],
            dim=0,
        ).T
        assert got_qkv.shape == attn.qkv_proj_weight.shape
        assert torch.equal(got_qkv, want_qkv), f"qkv rank {rank}"

        got_o = get_weight_loader(attn.o_proj_weight).load(
            [ckpt.slice(f"{p}.o_proj.weight")], rank
        )
        want_o = o[:, rank * q_heads * d : (rank + 1) * q_heads * d].T
        assert got_o.shape == attn.o_proj_weight.shape
        assert torch.equal(got_o, want_o), f"o_proj rank {rank}"


@pytest.mark.parametrize("layer_idx", LAYERS_UNDER_TEST)
def test_tp8_padded_mlp_matches_checkpoint(
    neuron_config_obj, ckpt, monkeypatch, layer_idx
):
    """The padded MLP shards must hold the checkpoint values and zero padding.

    The test also runs the padded rank MLP and the unpadded checkpoint MLP in
    FP32 on the same input. The zero padding must not change the output.
    """
    from vllm_neuron.utils.weight_loader import get_weight_loader

    # ``m`` is the module object, so ``monkeypatch.setattr`` below can replace
    # the ``get_tp_group`` that the component constructors call.
    m = _local_src.load("model")

    cfg = neuron_config_obj
    p = f"model.layers.{layer_idx}.mlp"
    gate = ckpt.tensor(f"{p}.gate_proj.weight")
    up = ckpt.tensor(f"{p}.up_proj.weight")
    down = ckpt.tensor(f"{p}.down_proj.weight")
    shard = cfg.intermediate_size // TP
    padded = m.padded_intermediate_size(cfg.intermediate_size, TP)
    assert (shard, padded) == (3424, 3456)

    torch.manual_seed(42)
    x = torch.randn(4, cfg.hidden_size, dtype=torch.float32)

    for rank in RANKS_UNDER_TEST:
        monkeypatch.setattr(m, "get_tp_group", lambda r=rank: _fake_tp_group(r))
        mlp = m.Exaone4MLP(cfg, layer_idx)
        rows = slice(rank * shard, (rank + 1) * shard)

        got = {}
        for name in ("gate", "up", "down"):
            param = getattr(mlp, f"{name}_proj_weight")
            got[name] = get_weight_loader(param).load(
                [ckpt.slice(f"{p}.{name}_proj.weight")], rank
            )
            assert got[name].shape == param.shape, name

        assert torch.equal(got["gate"][:, :shard], gate[rows].T)
        assert torch.equal(got["up"][:, :shard], up[rows].T)
        assert torch.equal(got["down"][:shard], down[:, rows].T)
        assert torch.count_nonzero(got["gate"][:, shard:]) == 0
        assert torch.count_nonzero(got["up"][:, shard:]) == 0
        assert torch.count_nonzero(got["down"][shard:]) == 0

        def swiglu(xx, g, u, dn):
            return (torch.nn.functional.silu(xx @ g) * (xx @ u)) @ dn

        out_padded = swiglu(
            x, got["gate"].float(), got["up"].float(), got["down"].float()
        )
        out_ref = swiglu(
            x, gate[rows].T.float(), up[rows].T.float(), down[:, rows].T.float()
        )
        torch.testing.assert_close(out_padded, out_ref, rtol=1e-5, atol=1e-5)
