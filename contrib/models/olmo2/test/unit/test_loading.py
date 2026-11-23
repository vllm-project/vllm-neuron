# SPDX-License-Identifier: Apache-2.0
"""OLMo2 loading tests (CPU).

These tests check the loading path of allenai/OLMo-2-0425-1B at the registered
topology (TP 8):

1. ``test_registry_and_config``: the registration contract (the architecture
   string of the checkpoint and the factory that ``src/`` exports), the config
   parse, the factory checks and the TP size checks.
2. ``test_checkpoint_mapping_coverage``: each model parameter has a mapping, and
   the mappings use each checkpoint key exactly one time.
3. ``test_weight_loader_transforms``: the loaders that the model attaches give
   the expected shard on each rank. The test uses mock slices.
4. ``test_full_model_tp8_load``: an 8-rank CPU load of the full model. The test
   compares the loaded shards with slices that it reads from the safetensors
   files.

The tests need 8 CPU-mode ranks, and no Neuron hardware. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo2/test/unit/test_loading.py -v
"""

import json
import os
from functools import partial

import pytest
import torch
from safetensors import safe_open

from vllm_neuron import envs

from _local_src import src_module, src_package
from olmo2_test_utils import (
    COLLECT_TIMEOUT_S,
    WORLD_SIZE,
    full_model_load,
    local_global_ranks,
    make_executor,
    read_config_json,
    report_spec,
    resolve_checkpoint,
)

ARCHITECTURE = "Olmo2ForCausalLM"


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
    olmo2_model = src_module("model")

    monkeypatch.setattr(
        olmo2_model, "get_tp_group", lambda: _FakeTPGroup(world_size, rank)
    )


def _olmo2_config(checkpoint_path: str):
    Olmo2Config = src_module("config").Olmo2Config

    return Olmo2Config.from_configs(os.path.join(checkpoint_path, "config.json"))


# =============================================================================
# 1. Registry and config
# =============================================================================


def test_registry_and_config(monkeypatch):
    """Check the registration contract, the config parse and the config checks."""
    from transformers import AutoConfig

    Olmo2Factory = src_package().Olmo2ForCausalLM
    Olmo2Attention = src_module("model").Olmo2Attention

    checkpoint_path = resolve_checkpoint()
    raw = read_config_json(checkpoint_path)
    assert raw["architectures"] == [ARCHITECTURE]

    # A contrib model is deliberately absent from
    # ``vllm_neuron.model.registry.get_models()``. The user registers it
    # (contrib/README.md, "How do I register a custom model with vLLM?"), so
    # what this test can check is that the architecture string of the
    # checkpoint reaches this factory through the package export that
    # ``ModelRegistry.register_model`` is given.
    assert getattr(src_package(), ARCHITECTURE) is Olmo2Factory, (
        f"src/__init__.py must export {ARCHITECTURE}."
    )

    config = _olmo2_config(checkpoint_path)
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
        "tie_word_embeddings",
    ):
        assert getattr(config, field) == raw[field], field
    assert config.head_dim == raw["hidden_size"] // raw["num_attention_heads"]
    # The checkpoint stores FP32. The port computes in BF16.
    assert config.torch_dtype == torch.bfloat16

    # The factory accepts the real config and rejects a feature that the port
    # does not implement.
    hf_config = AutoConfig.from_pretrained(checkpoint_path)
    Olmo2Factory._validate_config(hf_config, None)
    for override in (
        {"tie_word_embeddings": True},
        {"attention_bias": True},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
    ):
        bad = AutoConfig.from_pretrained(checkpoint_path)
        for key, value in override.items():
            setattr(bad, key, value)
        with pytest.raises(ValueError):
            Olmo2Factory._validate_config(bad, None)

    # 16 heads and 16 KV heads: TP 1, 2, 4 and 8 are legal. TP 16 gives one
    # head per rank (odd), and TP 3 does not divide the heads.
    for world_size in (1, 2, 4, 8):
        _patch_tp_group(monkeypatch, world_size, 0)
        with torch.device("meta"):
            Olmo2Attention(config, layer_idx=0)
    for world_size in (3, 16):
        _patch_tp_group(monkeypatch, world_size, 0)
        with pytest.raises(ValueError), torch.device("meta"):
            Olmo2Attention(config, layer_idx=0)


# =============================================================================
# 2. Checkpoint mapping coverage
# =============================================================================


def test_checkpoint_mapping_coverage(monkeypatch):
    """Each parameter maps to checkpoint keys. Each key is used one time."""
    model_module = src_module("model")
    Olmo2DecoderLayer = model_module.Olmo2DecoderLayer
    olmo2_checkpoint_mappings = model_module.olmo2_checkpoint_mappings

    checkpoint_path = resolve_checkpoint()
    config = _olmo2_config(checkpoint_path)
    with open(os.path.join(checkpoint_path, "model.safetensors.index.json")) as f:
        checkpoint_keys = set(json.load(f)["weight_map"])

    mappings = olmo2_checkpoint_mappings(config.num_hidden_layers)

    used_keys = []
    for value in mappings.values():
        used_keys += value if isinstance(value, list) else [value]
    assert len(used_keys) == len(set(used_keys)), "a checkpoint key is used twice"
    assert set(used_keys) == checkpoint_keys, (
        f"not mapped: {sorted(checkpoint_keys - set(used_keys))[:5]}, "
        f"not in checkpoint: {sorted(set(used_keys) - checkpoint_keys)[:5]}"
    )

    # Build one decoder layer on meta to get the parameter names.
    _patch_tp_group(monkeypatch, WORLD_SIZE, 0)
    with torch.device("meta"):
        layer = Olmo2DecoderLayer(config, layer_idx=0)
    layer_names = [name for name, _ in layer.named_parameters()]
    param_names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for i in range(config.num_hidden_layers):
        param_names |= {f"model.layers.{i}.{name}" for name in layer_names}
    assert set(mappings) == param_names


# =============================================================================
# 3. Weight loader transforms
# =============================================================================


def test_weight_loader_transforms(monkeypatch):
    """Check the attached loaders on each of the 8 ranks with mock slices.

    The expected shard comes from the HF layout: [out_features, in_features].
    The port stores the projections transposed: [in_features, out_features].
    """
    from vllm_neuron.utils.weight_loader import get_weight_loader

    model_module = src_module("model")
    Olmo2Attention = model_module.Olmo2Attention
    Olmo2MLP = model_module.Olmo2MLP

    torch.manual_seed(42)
    checkpoint_path = resolve_checkpoint()
    config = _olmo2_config(checkpoint_path)
    hidden = config.hidden_size
    inter = config.intermediate_size
    head_dim = config.head_dim
    q_full = config.num_attention_heads * head_dim
    kv_full = config.num_key_value_heads * head_dim

    w_q = torch.randn(q_full, hidden)
    w_k = torch.randn(kv_full, hidden)
    w_v = torch.randn(kv_full, hidden)
    w_o = torch.randn(hidden, q_full)
    q_norm = torch.randn(q_full)
    k_norm = torch.randn(kv_full)
    w_gate = torch.randn(inter, hidden)
    w_up = torch.randn(inter, hidden)
    w_down = torch.randn(hidden, inter)

    q_shard = q_full // WORLD_SIZE
    kv_shard = kv_full // WORLD_SIZE
    i_shard = inter // WORLD_SIZE

    def load(param, tensors, rank):
        return get_weight_loader(param).load([MockSlice(t) for t in tensors], rank)

    for rank in range(WORLD_SIZE):
        _patch_tp_group(monkeypatch, WORLD_SIZE, rank)
        with torch.device("meta"):
            attn = Olmo2Attention(config, layer_idx=0)
            mlp = Olmo2MLP(config)

        q_rows = slice(rank * q_shard, (rank + 1) * q_shard)
        kv_rows = slice(rank * kv_shard, (rank + 1) * kv_shard)
        i_rows = slice(rank * i_shard, (rank + 1) * i_shard)

        expected_qkv = torch.cat((w_q[q_rows], w_k[kv_rows], w_v[kv_rows]), dim=0).T
        expected = {
            "qkv": (load(attn.qkv_proj_weight, [w_q, w_k, w_v], rank), expected_qkv),
            "o": (load(attn.o_proj_weight, [w_o], rank), w_o[:, q_rows].T),
            "q_norm": (load(attn.q_norm_weight, [q_norm], rank), q_norm[q_rows]),
            "k_norm": (load(attn.k_norm_weight, [k_norm], rank), k_norm[kv_rows]),
            "gate": (load(mlp.gate_proj_weight, [w_gate], rank), w_gate[i_rows].T),
            "up": (load(mlp.up_proj_weight, [w_up], rank), w_up[i_rows].T),
            "down": (load(mlp.down_proj_weight, [w_down], rank), w_down[:, i_rows].T),
        }
        for name, (actual, want) in expected.items():
            assert tuple(actual.shape) == tuple(want.shape), (name, rank)
            assert torch.equal(actual, want), (name, rank)
        assert tuple(expected["qkv"][0].shape) == tuple(attn.qkv_proj_weight.shape)


# =============================================================================
# 4. Full-model TP 8 load
# =============================================================================


def _expected_report(checkpoint_path: str, rank: int) -> list[torch.Tensor]:
    """Read the expected values for ``report_spec()`` from the safetensors files.

    This reads the HF tensors and does the TP slicing here, without the port
    loaders.
    """
    config = read_config_json(checkpoint_path)
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    q_shard = config["num_attention_heads"] * head_dim // WORLD_SIZE
    kv_shard = config["num_key_value_heads"] * head_dim // WORLD_SIZE
    i_shard = config["intermediate_size"] // WORLD_SIZE
    v_shard = config["vocab_size"] // WORLD_SIZE
    q_rows = slice(rank * q_shard, (rank + 1) * q_shard)
    kv_rows = slice(rank * kv_shard, (rank + 1) * kv_shard)
    i_rows = slice(rank * i_shard, (rank + 1) * i_shard)
    v_rows = slice(rank * v_shard, (rank + 1) * v_shard)

    with open(os.path.join(checkpoint_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {}

    def hf(key: str) -> torch.Tensor:
        file_name = weight_map[key]
        if file_name not in handles:
            handles[file_name] = safe_open(
                os.path.join(checkpoint_path, file_name), framework="pt", device="cpu"
            )
        return handles[file_name].get_tensor(key)

    def port_layout(name: str) -> torch.Tensor:
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return hf(name)[v_rows]
        if name == "model.norm.weight":
            return hf(name)
        prefix, _, leaf = name.rpartition(".")
        if leaf == "qkv_proj_weight":
            return torch.cat(
                (
                    hf(f"{prefix}.q_proj.weight")[q_rows],
                    hf(f"{prefix}.k_proj.weight")[kv_rows],
                    hf(f"{prefix}.v_proj.weight")[kv_rows],
                ),
                dim=0,
            ).T
        if leaf == "o_proj_weight":
            return hf(f"{prefix}.o_proj.weight")[:, q_rows].T
        if leaf == "q_norm_weight":
            return hf(f"{prefix}.q_norm.weight")[q_rows]
        if leaf == "k_norm_weight":
            return hf(f"{prefix}.k_norm.weight")[kv_rows]
        if leaf in ("gate_proj_weight", "up_proj_weight"):
            return hf(f"{prefix}.{leaf[: -len('_weight')]}.weight")[i_rows].T
        if leaf == "down_proj_weight":
            return hf(f"{prefix}.down_proj.weight")[:, i_rows].T
        # The decoder norms keep the HF name.
        return hf(name)

    expected = []
    for name, rows, cols in report_spec():
        tensor = port_layout(name).to(torch.bfloat16)
        expected.append(tensor[rows] if tensor.dim() == 1 else tensor[rows, cols])
    return expected


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [WORLD_SIZE])
def test_full_model_tp8_load(world_size: int):
    """Load all 16 layers at TP 8 on CPU. Compare the shards with the checkpoint."""
    assert envs.VLLM_NEURON_CPU_MODE, "Run this test with VLLM_NEURON_CPU_MODE=1."
    checkpoint_path = resolve_checkpoint()

    executor = make_executor(
        partial(full_model_load, checkpoint_path=checkpoint_path, device="cpu")
    )
    executor.dispatch()
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)

    spec = report_spec()
    for rank, output in zip(local_global_ranks(), outputs):
        status, *actual = output
        num_params, num_meta, num_nonfinite, num_wrong_dtype, num_meta_buffers = (
            status.tolist()
        )
        # 3 top-level parameters + 9 parameters in each of the 16 layers. The
        # fused QKV parameter takes 3 checkpoint keys, so 179 keys give 147.
        assert num_params == 3 + 9 * 16, f"rank {rank}: {num_params} parameters"
        assert num_meta == 0, f"rank {rank}: {num_meta} parameters on meta"
        assert num_meta_buffers == 0, f"rank {rank}: {num_meta_buffers} meta buffers"
        assert num_nonfinite == 0, f"rank {rank}: {num_nonfinite} non-finite"
        assert num_wrong_dtype == 0, f"rank {rank}: {num_wrong_dtype} not BF16"

        expected = _expected_report(checkpoint_path, rank)
        for (name, _, _), got, want in zip(spec, actual, expected):
            assert tuple(got.shape) == tuple(want.shape), (rank, name)
            assert torch.equal(got, want), f"rank {rank}: {name} does not match"
