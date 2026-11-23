# SPDX-License-Identifier: Apache-2.0
"""ModernBERT CPU tests: config, registry, attention bounds, weight loading, CPU layers.

Model: ibm-granite/granite-embedding-small-english-r2 (BF16, 12 layers).

These tests run on one node in CPU mode. The weight-loading and layer tests
use MPExecutor with TP 2 on CPU (gloo), so each rank loads its own shard of
the original checkpoint.

Run with:
    VLLM_NEURON_CPU_MODE=1 MODERNBERT_CHECKPOINT=/path/to/checkpoint \
        pytest contrib/models/modernbert/test/unit/test_loading.py -v
"""

import importlib.util
import json
import os
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from vllm_neuron.accuracy.testing import (
    assert_close_three_way,
    assert_pooling_close_three_way,
)
from vllm_neuron.utils.executor import MPExecutor

# Load this folder's own ``src/`` by file location, not through a package path
# rooted at the repository: a contrib model has to keep working when the folder
# is copied into an application. ``test/_local_src.py`` is the loader, and it is
# loaded the same way, under a name no other contrib model can collide with.
_LOADER_NAME = "contrib_modernbert_local_src"
if _LOADER_NAME in sys.modules:
    _local_src = sys.modules[_LOADER_NAME]
else:
    _loader_spec = importlib.util.spec_from_file_location(
        _LOADER_NAME, Path(__file__).resolve().parents[1] / "_local_src.py"
    )
    _local_src = importlib.util.module_from_spec(_loader_spec)
    sys.modules[_LOADER_NAME] = _local_src
    _loader_spec.loader.exec_module(_local_src)

_src_config = _local_src.load("config")
_src_model = _local_src.load("model")
common = _local_src.load_test_module("module/common.py")

FULL_ATTENTION = _src_config.FULL_ATTENTION
SLIDING_ATTENTION = _src_config.SLIDING_ATTENTION
modernbert_attention_bounds = _src_model.modernbert_attention_bounds

SEED = common.SEED
TP_SIZE = common.TP_SIZE
cls_embedding = common.cls_embedding
expected_param_names = common.expected_param_names
hf_component_references = common.hf_component_references
hf_hidden_states = common.hf_hidden_states
load_config = common.load_config
load_hf_models = common.load_hf_models
make_token_ids = common.make_token_ids
model_load = common.model_load
pack = common.pack
resolve_checkpoint = common.resolve_checkpoint

DEVICE = "cpu"
SEQ_LEN = 1024
COLLECT_TIMEOUT_S = 1800.0


@pytest.fixture(scope="module")
def checkpoint_path():
    return resolve_checkpoint()


def _read_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    tensors = {}
    for name in sorted(os.listdir(checkpoint_path)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(checkpoint_path, name), framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


# =============================================================================
# Config and registry
# =============================================================================


def test_config_and_registry(checkpoint_path, monkeypatch):
    """The port config agrees with config.json, and vLLM resolves the architecture."""
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        raw = json.load(f)
    config = load_config(checkpoint_path)

    assert raw["architectures"] == ["ModernBertModel"]
    assert config.hidden_size == raw["hidden_size"] == 384
    assert config.intermediate_size == raw["intermediate_size"] == 1536
    assert config.num_hidden_layers == raw["num_hidden_layers"] == 12
    assert config.num_attention_heads == raw["num_attention_heads"] == 12
    assert config.head_dim == 32
    assert config.vocab_size == raw["vocab_size"]
    assert config.sliding_window == raw["local_attention"] // 2
    assert config.torch_dtype == torch.bfloat16

    expected_types = raw.get("layer_types") or [
        SLIDING_ATTENTION if i % raw["global_attn_every_n_layers"] else FULL_ATTENTION
        for i in range(raw["num_hidden_layers"])
    ]
    assert config.layer_types == expected_types
    assert [i for i, t in enumerate(config.layer_types) if t == FULL_ATTENTION] == [
        0,
        3,
        6,
        9,
    ]

    rope = raw.get("rope_parameters")
    if isinstance(rope, dict):
        assert config.global_rope_theta == rope[FULL_ATTENTION]["rope_theta"]
        assert config.local_rope_theta == rope[SLIDING_ATTENTION]["rope_theta"]
    else:
        assert config.global_rope_theta == raw["global_rope_theta"]
        assert config.local_rope_theta == raw["local_rope_theta"]

    # The TP degree divides the heads and the MLP channels.
    assert config.num_attention_heads % TP_SIZE == 0
    assert config.intermediate_size % TP_SIZE == 0

    from vllm import ModelRegistry

    ModernBertModel = _local_src.load().ModernBertModel

    # A contrib model is not in ``vllm_neuron.model.registry``. The user
    # registers the architecture with vLLM (see contrib/README.md, "How do I
    # register a custom model with vLLM?"), so this test registers it the same
    # way and checks that vLLM holds it under the architecture string of the
    # checkpoint. The copy of ``ModelRegistry.models`` keeps the registry of
    # the pytest session unchanged.
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    ModelRegistry.register_model("ModernBertModel", ModernBertModel)
    assert "ModernBertModel" in ModelRegistry.models
    assert ModernBertModel.is_pooling_model
    assert ModernBertModel.default_seq_pooling_type == "CLS"
    assert ModernBertModel.attn_type == "encoder_only"


# =============================================================================
# Attention bounds (pure torch)
# =============================================================================


def _reference_mask(lengths: list[int], pad: int, window: int | None) -> torch.Tensor:
    """Build the allowed-key mask from the sequence lengths (independent of positions)."""
    seq_id, pos = [], []
    for s, length in enumerate(lengths):
        seq_id += [s] * length
        pos += list(range(length))
    for p in range(pad):
        seq_id.append(len(lengths) + p)
        pos.append(0)
    seq_id = torch.tensor(seq_id)
    pos = torch.tensor(pos)
    mask = seq_id[:, None] == seq_id[None, :]
    if window is not None:
        mask &= (pos[:, None] - pos[None, :]).abs() <= window
    return mask


@pytest.mark.parametrize("lengths", [[1024], [700, 300], [1, 63, 200, 5], [130]])
def test_attention_bounds(lengths):
    """The KV bounds give the HF bidirectional and sliding-window masks of packed sequences."""
    torch.manual_seed(SEED)
    window = 64
    total = sum(lengths)
    pad = (-total) % 128
    positions = torch.cat(
        [torch.arange(n) for n in lengths] + [torch.zeros(pad, dtype=torch.int64)]
    )
    full_min, full_max, sliding_min, sliding_max = modernbert_attention_bounds(
        positions, window
    )
    keys = torch.arange(total + pad)
    for (lo, hi), win in (
        ((full_min, full_max), None),
        ((sliding_min, sliding_max), window),
    ):
        got = (keys[None, :] >= lo) & (keys[None, :] < hi)
        torch.testing.assert_close(got, _reference_mask(lengths, pad, win))
        assert (hi > lo).all()


# =============================================================================
# Weight loading (MPExecutor, TP 2 on CPU)
# =============================================================================


@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_weight_shards_reconstruct_checkpoint(checkpoint_path, world_size):
    """Each rank loads all parameters, and the TP shards rebuild each original tensor exactly."""
    config = load_config(checkpoint_path)
    names = expected_param_names(config)
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(model_load, checkpoint_path, "shards", 0, DEVICE),
        device=DEVICE,
    )
    executor.dispatch(torch.zeros(1))
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
    assert len(outputs) == world_size
    shards = [dict(zip(names, out, strict=True)) for out in outputs]

    ckpt = _read_checkpoint(checkpoint_path)
    prefix = "model." if "model.embeddings.tok_embeddings.weight" in ckpt else ""
    used = set()

    def original(key):
        used.add(prefix + key)
        return ckpt[prefix + key].to(torch.bfloat16)

    hidden, inter = config.hidden_size, config.intermediate_size
    s, si = hidden // world_size, inter // world_size

    for name in (
        "embeddings.tok_embeddings.weight",
        "embeddings.norm.weight",
        "final_norm.weight",
    ):
        for r in range(world_size):
            torch.testing.assert_close(shards[r][name], original(name), rtol=0, atol=0)

    for i in range(config.num_hidden_layers):
        lp = f"layers.{i}"
        if i != 0:
            for r in range(world_size):
                torch.testing.assert_close(
                    shards[r][f"{lp}.attn_norm.weight"],
                    original(f"{lp}.attn_norm.weight"),
                    rtol=0,
                    atol=0,
                )
        for r in range(world_size):
            torch.testing.assert_close(
                shards[r][f"{lp}.mlp_norm.weight"],
                original(f"{lp}.mlp_norm.weight"),
                rtol=0,
                atol=0,
            )

        qkv = [shards[r][f"{lp}.attn.qkv_proj_weight"].T for r in range(world_size)]
        rebuilt_qkv = torch.cat(
            [
                torch.cat([qkv[r][j * s : (j + 1) * s] for r in range(world_size)])
                for j in range(3)
            ]
        )
        torch.testing.assert_close(
            rebuilt_qkv, original(f"{lp}.attn.Wqkv.weight"), rtol=0, atol=0
        )

        rebuilt_o = torch.cat(
            [shards[r][f"{lp}.attn.o_proj_weight"] for r in range(world_size)]
        ).T
        torch.testing.assert_close(
            rebuilt_o, original(f"{lp}.attn.Wo.weight"), rtol=0, atol=0
        )

        rebuilt_wi = torch.cat(
            [
                torch.cat(
                    [
                        shards[r][f"{lp}.mlp.{p}_proj_weight"].T
                        for r in range(world_size)
                    ]
                )
                for p in ("gate", "up")
            ]
        )
        torch.testing.assert_close(
            rebuilt_wi, original(f"{lp}.mlp.Wi.weight"), rtol=0, atol=0
        )

        rebuilt_down = torch.cat(
            [shards[r][f"{lp}.mlp.down_proj_weight"] for r in range(world_size)]
        ).T
        torch.testing.assert_close(
            rebuilt_down, original(f"{lp}.mlp.Wo.weight"), rtol=0, atol=0
        )
        for r in range(world_size):
            assert shards[r][f"{lp}.attn.qkv_proj_weight"].shape == (hidden, 3 * s)
            assert shards[r][f"{lp}.mlp.gate_proj_weight"].shape == (hidden, si)

    assert used == set(ckpt), f"Unused checkpoint keys: {sorted(set(ckpt) - used)}"
    for r in range(world_size):
        for name, tensor in shards[r].items():
            assert tensor.dtype == torch.bfloat16, name
            assert torch.isfinite(tensor.float()).all(), name


# =============================================================================
# CPU layers and CPU full model against HF (MPExecutor, TP 2 on CPU)
# =============================================================================


@pytest.mark.parametrize(
    "layer_idx", [0, 1], ids=["full_attention", "sliding_attention"]
)
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_layer_cpu(checkpoint_path, world_size, layer_idx):
    """One TP 2 encoder layer on CPU matches the HF layer (three-way)."""
    (input_ids,) = make_token_ids(checkpoint_path, [SEQ_LEN])
    hidden, baseline, expected = hf_component_references(
        checkpoint_path, f"layers.{layer_idx}", input_ids
    )
    positions = torch.arange(SEQ_LEN, dtype=torch.int64)

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(model_load, checkpoint_path, "layer", layer_idx, DEVICE),
        device=DEVICE,
    )
    executor.dispatch(hidden, positions)
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)

    for rank, actual in enumerate(outputs):
        result = assert_close_three_way(
            baseline,
            expected,
            actual.float(),
            name=f"modernbert/cpu/layer{layer_idx}/rank{rank}",
        )
        assert result.passed


@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_full_model_cpu_packed(checkpoint_path, world_size):
    """The TP 2 backbone on CPU matches HF for two packed sequences and a padding tail."""
    sequences = make_token_ids(checkpoint_path, [700, 300])
    input_ids, positions = pack(sequences)

    hf_fp32, hf_bf16 = load_hf_models(checkpoint_path)
    baseline = [hf_hidden_states(hf_fp32, s).float() for s in sequences]
    expected = [hf_hidden_states(hf_bf16, s).float() for s in sequences]

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(model_load, checkpoint_path, "full", 0, DEVICE),
        device=DEVICE,
    )
    executor.dispatch(input_ids, positions)
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)

    lengths = [s.shape[0] for s in sequences]
    hidden_size = load_config(checkpoint_path).hidden_size
    for rank, out in enumerate(outputs):
        assert out.shape == (sum(lengths), hidden_size)
        actual = list(out.float().split(lengths))
        result = assert_close_three_way(
            torch.cat(baseline),
            torch.cat(expected),
            torch.cat(actual),
            name=f"modernbert/cpu/full_packed/rank{rank}",
        )
        assert result.passed
        pooled = assert_pooling_close_three_way(
            baseline=torch.stack([cls_embedding(h) for h in baseline]),
            expected=torch.stack([cls_embedding(h) for h in expected]),
            actual=torch.stack([cls_embedding(h) for h in actual]),
            task="embed",
            name=f"modernbert/cpu/embed_packed/rank{rank}",
            min_cos=0.99,
        )
        print(
            f"rank {rank}: worst cos {pooled.min_tgt_cos:.6f}, "
            f"BF16-HF worst cos {pooled.min_base_cos:.6f}"
        )
        assert pooled.passed
