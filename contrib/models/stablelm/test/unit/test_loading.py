# SPDX-License-Identifier: Apache-2.0
"""StableLM loading tests (CPU).

1. ``test_registration_contract``: the plugin entry point registers the
   architecture string of the checkpoint, and the factory satisfies vLLM's
   model protocol for a text generation model.
2. ``test_config_parse``: the port config of the pinned ``config.json`` (read
   through transformers, and read as raw JSON).
3. ``test_factory_rejects_unsupported``: the factory refuses the options that
   this implementation does not have.
4. ``test_checkpoint_mapping_coverage``: each model parameter has a mapping,
   and the mappings use each tensor of the HF model exactly one time. The HF
   tensor names come from an HF model built on the meta device, so the test
   needs no download.
5. ``test_qkv_loader_permutes_rotary_dims``: the QKV loader shards by head and
   permutes the Q and K head dims for partial RoPE. The test uses mock slices.
6. ``test_full_model_load``: a TP 2 CPU load of a tiny random-weight
   checkpoint with head_dim 80 and rotary dim 20.

No Neuron hardware and no checkpoint download are needed. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/stablelm/test/unit/test_loading.py -v
"""

import json
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from vllm_neuron.model.neuron_config import NeuronConfig

from _local_src import src_module, src_package
from stablelm_test_utils import (
    ARCHITECTURE,
    COLLECT_TIMEOUT_S,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_CHECKPOINT_TENSORS,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    PINNED_CONFIG,
    ROTARY_DIM,
    TENSORS_PER_LAYER,
    TINY_OVERRIDES,
    TINY_TP,
    TOP_LEVEL_TENSORS,
    VOCAB_SIZE,
    checkpoint_weight_map,
    load_hf_config,
    make_tiny_checkpoint,
    read_checkpoint_tensors,
    write_pinned_config,
)

# The HF partial RoPE pairs dim j with dim j + 10 for j < 10. The port moves
# this pair to (j, j + 40). Write the expected order by hand, so that the test
# does not use the port function as its own reference.
EXPECTED_PERM_80_20 = (
    list(range(0, 10)) + list(range(20, 50)) + list(range(10, 20)) + list(range(50, 80))
)


class MockSlice:
    """Mock checkpoint slice with __getitem__ and get_shape."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]

    def get_shape(self):
        return self.tensor.shape


class _FakeTPGroup:
    """Give the TP size and the rank without a process group."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = None


def _patch_tp_group(monkeypatch, world_size: int, rank: int) -> None:
    monkeypatch.setattr(
        src_module("model"), "get_tp_group", lambda: _FakeTPGroup(world_size, rank)
    )


@pytest.fixture(scope="module")
def pinned_config_dir(tmp_path_factory) -> str:
    """A directory with the pinned ``config.json`` and no weights."""
    return write_pinned_config(str(tmp_path_factory.mktemp("stablelm_config")))


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_stablelm")))


# =============================================================================
# 1. Registration
# =============================================================================


def test_registration_contract(monkeypatch):
    """The entry point registers the checkpoint architecture with the factory."""
    from vllm import ModelRegistry
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
        is_vllm_model,
    )

    factory_cls = src_package().StableLmForCausalLM
    plugin = src_module("plugin")

    assert PINNED_CONFIG["architectures"] == [ARCHITECTURE]
    assert getattr(src_package(), ARCHITECTURE) is factory_cls, (
        f"src/__init__.py must export {ARCHITECTURE}."
    )

    # The entry point registers "<package>.factory:<class>". The package name
    # comes from pyproject.toml, so check the module path and the class name.
    calls = []
    monkeypatch.setattr(
        ModelRegistry, "register_model", lambda *args: calls.append(args)
    )
    plugin.register()
    assert calls == [(ARCHITECTURE, "stablelm_neuron.factory:StableLmForCausalLM")]
    with open(os.path.join(os.path.dirname(__file__), "../../pyproject.toml")) as f:
        pyproject = f.read()
    assert 'stablelm_neuron = "stablelm_neuron.plugin:register"' in pyproject
    assert 'package-dir = {"stablelm_neuron" = "src"}' in pyproject

    # vLLM 0.29 checks the model protocol when the class is registered.
    assert is_vllm_model(factory_cls)
    assert is_text_generation_model(factory_cls)


# =============================================================================
# 2. Config
# =============================================================================


def test_config_parse(pinned_config_dir):
    """The port config of the pinned config.json has the real geometry."""
    config_cls = src_module("config").StableLmConfig

    hf_config = load_hf_config(pinned_config_dir)
    from_hf = config_cls.from_configs(hf_config, NeuronConfig())
    from_json = config_cls.from_configs(
        os.path.join(pinned_config_dir, "config.json"), NeuronConfig()
    )
    for config in (from_hf, from_json):
        assert config.vocab_size == VOCAB_SIZE
        assert config.hidden_size == HIDDEN_SIZE
        assert config.intermediate_size == INTERMEDIATE_SIZE
        assert config.num_hidden_layers == NUM_LAYERS
        assert config.num_attention_heads == NUM_HEADS
        assert config.num_key_value_heads == NUM_KV_HEADS
        assert config.head_dim == HEAD_DIM
        assert config.rotary_dim == ROTARY_DIM
        assert config.partial_rotary_factor == pytest.approx(0.25)
        assert config.rope_theta == pytest.approx(10000.0)
        assert config.layer_norm_eps == pytest.approx(1e-5)
        assert config.torch_dtype == torch.bfloat16
        assert config.tie_word_embeddings is False
        assert config.neuron_config is not None

    # transformers >= 5 moves rope_theta and partial_rotary_factor into
    # rope_parameters. The config must find both there.
    nested = {
        k: v
        for k, v in PINNED_CONFIG.items()
        if k not in ("rope_theta", "partial_rotary_factor")
    }
    nested["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 500.0,
        "partial_rotary_factor": 0.5,
    }
    config = config_cls.from_configs(nested, None)
    assert config.rope_theta == pytest.approx(500.0)
    assert config.rotary_dim == HEAD_DIM // 2


# =============================================================================
# 3. Factory checks
# =============================================================================


def test_factory_rejects_unsupported(pinned_config_dir):
    """The factory accepts the pinned config and refuses unsupported options."""
    factory_cls = src_package().StableLmForCausalLM

    factory_cls._validate_config(load_hf_config(pinned_config_dir), NeuronConfig())

    overrides = [
        {"use_qkv_bias": True},
        {"qk_layernorm": True},
        {"use_parallel_residual": True},
        {"hidden_act": "gelu"},
        {"torch_dtype": torch.float16},
        {"rope_parameters": {"rope_type": "linear", "factor": 2.0}},
        {"rope_scaling": {"type": "dynamic", "factor": 2.0}},
    ]
    for override in overrides:
        bad = load_hf_config(pinned_config_dir)
        for key, value in override.items():
            setattr(bad, key, value)
        with pytest.raises(ValueError):
            factory_cls._validate_config(bad, NeuronConfig())

    # An odd rotary dim cannot pair rotate_half dims.
    bad = load_hf_config(pinned_config_dir)
    bad.partial_rotary_factor = 0.1625  # 80 * 0.1625 = 13
    if isinstance(getattr(bad, "rope_parameters", None), dict):
        bad.rope_parameters = {**bad.rope_parameters, "partial_rotary_factor": 0.1625}
    with pytest.raises(ValueError, match="Rotary dim"):
        factory_cls._validate_config(bad, NeuronConfig())


# =============================================================================
# 4. Checkpoint mapping coverage
# =============================================================================


def test_checkpoint_mapping_coverage(monkeypatch, pinned_config_dir):
    """Each parameter maps to HF tensors. Each HF tensor is used one time."""
    from transformers import AutoModelForCausalLM

    model_cls = src_module("model").StableLmForCausalLM
    decoder_cls = src_module("model").StableLmDecoderLayer
    config = src_module("config").StableLmConfig.from_configs(
        load_hf_config(pinned_config_dir), NeuronConfig()
    )

    # The HF model on meta gives the checkpoint tensor names.
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(load_hf_config(pinned_config_dir))
    hf_keys = set(hf_model.state_dict())
    assert len(hf_keys) == NUM_CHECKPOINT_TENSORS

    mappings = model_cls.checkpoint_mappings(config.num_hidden_layers)
    used = []
    for value in mappings.values():
        used += value if isinstance(value, list) else [value]
    # The parameters without a mapping use their own name as the key.
    passthrough = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.norm.bias",
        "lm_head.weight",
    }
    used += sorted(passthrough)
    assert len(used) == len(set(used)), "a checkpoint key is used twice"
    assert set(used) == hf_keys, (
        f"not mapped: {sorted(hf_keys - set(used))[:5]}, "
        f"not in checkpoint: {sorted(set(used) - hf_keys)[:5]}"
    )

    # The port parameter names are the mapping keys plus the passthrough names.
    _patch_tp_group(monkeypatch, 8, 0)
    with torch.device("meta"):
        layer = decoder_cls(config, layer_idx=0)
    layer_names = [name for name, _ in layer.named_parameters()]
    assert len(layer_names) == TENSORS_PER_LAYER - 2  # fused QKV: 3 keys -> 1
    param_names = set(passthrough)
    for i in range(config.num_hidden_layers):
        param_names |= {f"model.layers.{i}.{name}" for name in layer_names}
    assert set(mappings) | passthrough == param_names
    assert len(passthrough) == TOP_LEVEL_TENSORS


# =============================================================================
# 5. QKV loader with the partial-RoPE permutation
# =============================================================================


@pytest.mark.parametrize("tp_size", [8])
def test_qkv_loader_permutes_rotary_dims(tp_size: int):
    """The QKV loader shards by head and permutes the Q and K head dims only."""
    from vllm_neuron.utils.weight_loader import fused_qkv_weight_loader

    model_module = src_module("model")
    partial_rotary_qkv_weight_loader = model_module.partial_rotary_qkv_weight_loader
    rotary_head_dim_permutation = model_module.rotary_head_dim_permutation

    torch.manual_seed(42)
    assert rotary_head_dim_permutation(HEAD_DIM, ROTARY_DIM).tolist() == (
        EXPECTED_PERM_80_20
    )

    q = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)
    k = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)
    v = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)
    q_size = NUM_HEADS * HEAD_DIM // tp_size
    kv_size = NUM_KV_HEADS * HEAD_DIM // tp_size
    base = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=1,
        num_shards=tp_size,
        is_storage_transposed=True,
        num_kv_replicas=1,
    )
    loader = partial_rotary_qkv_weight_loader(
        base, q_size=q_size, kv_size=kv_size, head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM
    )
    perm = torch.tensor(EXPECTED_PERM_80_20)
    heads_per_rank = NUM_HEADS // tp_size

    for rank in (0, tp_size - 1):
        out = loader.load([MockSlice(q), MockSlice(k), MockSlice(v)], rank)
        assert out.shape == (HIDDEN_SIZE, q_size + 2 * kv_size)
        for local_head in range(heads_per_rank):
            head = rank * heads_per_rank + local_head
            rows = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
            q_ref = q[rows].t()[:, perm]
            k_ref = k[rows].t()[:, perm]
            v_ref = v[rows].t()
            c = local_head * HEAD_DIM
            torch.testing.assert_close(out[:, c : c + HEAD_DIM], q_ref, rtol=0, atol=0)
            c = q_size + local_head * HEAD_DIM
            torch.testing.assert_close(out[:, c : c + HEAD_DIM], k_ref, rtol=0, atol=0)
            c = q_size + kv_size + local_head * HEAD_DIM
            torch.testing.assert_close(out[:, c : c + HEAD_DIM], v_ref, rtol=0, atol=0)

    # The model runner builds the model, and so the loaders, inside a
    # torch.device("meta") context. A loader built there must give the same
    # result as a loader built on the CPU.
    with torch.device("meta"):
        meta_loader = partial_rotary_qkv_weight_loader(
            base,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
        )
    for rank in (0, tp_size - 1):
        expected = loader.load([MockSlice(q), MockSlice(k), MockSlice(v)], rank)
        actual = meta_loader.load([MockSlice(q), MockSlice(k), MockSlice(v)], rank)
        assert not actual.is_meta, (
            "The QKV loader built under meta gives a meta tensor."
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# =============================================================================
# 6. Full-model TP 2 load of the tiny checkpoint
# =============================================================================


def _check_full_model_load(checkpoint: str, device: str) -> torch.Tensor:
    """Load all weights for this rank the way the runner does. Raise on a problem.

    Return the counts [checkpoint keys, layers, loaded parameters].
    """
    factory_cls = src_package().StableLmForCausalLM

    rank = dist.get_rank()
    ws = dist.get_world_size()
    hf_config = load_hf_config(checkpoint)
    # The model runner builds the model inside a meta device context and then
    # calls load_weights.
    with torch.device("meta"):
        model = factory_cls.from_configs(hf_config, NeuronConfig())
    model.load_weights(checkpoint, torch.device(device), None)

    params = dict(model.named_parameters())
    problems = {
        "on_meta": [n for n, p in params.items() if p.is_meta],
        "non_bf16": [n for n, p in params.items() if p.dtype != torch.bfloat16],
        "non_finite": [
            n for n, p in params.items() if not torch.isfinite(p.float()).all().item()
        ],
    }
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        raise AssertionError(f"Rank {rank}: the weight load has problems: {problems}")

    # Independent checks: slice the HF tensors here, without the port loaders.
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    heads = hf_config.num_attention_heads // ws
    inter = hf_config.intermediate_size // ws
    vocab = hf_config.vocab_size // ws
    perm = torch.tensor(EXPECTED_PERM_80_20)
    prefix = "model.layers.1"
    hf = read_checkpoint_tensors(
        checkpoint,
        [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.down_proj.weight",
            f"{prefix}.input_layernorm.bias",
            "model.norm.bias",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ],
    )

    def per_head(w: torch.Tensor, permute: bool) -> torch.Tensor:
        rows = w[rank * heads * head_dim : (rank + 1) * heads * head_dim]
        cols = rows.t().reshape(-1, heads, head_dim)
        if permute:
            cols = cols[:, :, perm]
        return cols.reshape(-1, heads * head_dim)

    layer = model.model.layers[1]
    expected = {
        "qkv": torch.cat(
            [
                per_head(hf[f"{prefix}.self_attn.q_proj.weight"], True),
                per_head(hf[f"{prefix}.self_attn.k_proj.weight"], True),
                per_head(hf[f"{prefix}.self_attn.v_proj.weight"], False),
            ],
            dim=1,
        ),
        "o": hf[f"{prefix}.self_attn.o_proj.weight"][
            :, rank * heads * head_dim : (rank + 1) * heads * head_dim
        ].t(),
        "gate": hf[f"{prefix}.mlp.gate_proj.weight"][
            rank * inter : (rank + 1) * inter
        ].t(),
        "down": hf[f"{prefix}.mlp.down_proj.weight"][
            :, rank * inter : (rank + 1) * inter
        ].t(),
        "ln_bias": hf[f"{prefix}.input_layernorm.bias"],
        "norm_bias": hf["model.norm.bias"],
        "embed": hf["model.embed_tokens.weight"][rank * vocab : (rank + 1) * vocab],
        "lm_head": hf["lm_head.weight"][rank * vocab : (rank + 1) * vocab],
    }
    actual = {
        "qkv": layer.self_attn.qkv_proj_weight,
        "o": layer.self_attn.o_proj_weight,
        "gate": layer.mlp.gate_proj_weight,
        "down": layer.mlp.down_proj_weight,
        "ln_bias": layer.input_layernorm.bias,
        "norm_bias": model.model.norm.bias,
        "embed": model.model.embed_tokens.weight,
        "lm_head": model.lm_head.weight,
    }
    for name, want in expected.items():
        got = actual[name].detach().cpu()
        if got.shape != want.shape or not torch.equal(got, want):
            raise AssertionError(f"Rank {rank}: {name} does not match the checkpoint.")

    return torch.tensor(
        [len(checkpoint_weight_map(checkpoint)), len(model.model.layers), len(params)],
        dtype=torch.int64,
    )


class _LoadReport(torch.nn.Module):
    """Load the model at construction. Return the counts at each call."""

    def __init__(self, checkpoint: str, device: str):
        super().__init__()
        self.error = None
        self.counts = None
        try:
            self.counts = _check_full_model_load(checkpoint, device)
        except Exception as exc:  # noqa: BLE001 - report at the RUN step
            self.error = exc

    def forward(self):
        if self.error is not None:
            raise self.error
        return self.counts


def _full_model_load(checkpoint: str, device: str) -> torch.nn.Module:
    """Top-level picklable model load for the full-checkpoint test."""
    return _LoadReport(checkpoint, device)


@pytest.mark.usefixtures("dist_cleanup")
def test_full_model_load(tiny_checkpoint):
    """Load every layer of the tiny checkpoint at TP 2 through load_weights."""
    from vllm_neuron.utils.executor import MPExecutor

    with open(os.path.join(tiny_checkpoint, "config.json")) as f:
        assert json.load(f)["architectures"] == [ARCHITECTURE]

    executor = MPExecutor(
        world_size=TINY_TP,
        model_load=partial(_full_model_load, checkpoint=tiny_checkpoint, device="cpu"),
        device="cpu",
    )
    executor.dispatch()
    counts = executor.collect(timeout=COLLECT_TIMEOUT_S)

    num_layers = TINY_OVERRIDES["num_hidden_layers"]
    num_keys = TOP_LEVEL_TENSORS + TENSORS_PER_LAYER * num_layers
    # The fused QKV parameter takes 3 checkpoint keys.
    num_params = TOP_LEVEL_TENSORS + (TENSORS_PER_LAYER - 2) * num_layers
    assert len(counts) == TINY_TP
    for rank, c in enumerate(counts):
        c = torch.as_tensor(c).tolist()
        assert c == [num_keys, num_layers, num_params], f"rank {rank}: counts {c}"
