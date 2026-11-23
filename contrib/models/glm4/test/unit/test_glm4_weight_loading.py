# SPDX-License-Identifier: Apache-2.0
"""GLM-4-32B-0414 checkpoint loading tests (CPU only).

The tests use the TP 8 sharding of the recorded runs. They replace the TP group
with a stub, so they do not start a process group and they do not need a
Neuron device.

1. Key coverage: the parameters of the full 61-layer model refer to each
   checkpoint tensor, and each referenced key is in the checkpoint. This test
   reads only the weight index (``model.safetensors.index.json``).
2. Full rank load: the loaders of TP rank 7 give a tensor of the correct
   shape, BF16 dtype, and finite values for each parameter.
3. Meta build: a QKV loader that a meta-device build makes gives the same
   shard as a CPU build (regression test for the RoPE column permutation).
4. Transforms: the fused QKV (with the RoPE column permutation), O, gate, up,
   down, embedding and lm_head shards of TP rank 7 are equal to an independent
   slice of the checkpoint tensors.

Tests 2 to 4 read the weights (about 65 GB on disk, about 10 GB of host memory
for the rank-7 shards) and run only when ``GLM4_0414_CHECKPOINT_PATH`` is set.

Run with:
    VLLM_NEURON_CPU_MODE=1 GLM4_0414_CHECKPOINT_PATH=/path/to/GLM-4-32B-0414 \
        pytest contrib/models/glm4/test/unit/test_glm4_weight_loading.py -v
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig

from _local_src import src_module
from glm4_test_utils import (
    FakeSafeSlice,
    metadata_path,
    read_tensors,
    read_weight_map,
    reference_rope_permutation,
    weights_path,
)

TP = 8
RANK = 7


def _config(checkpoint: str):
    hf_config = AutoConfig.from_pretrained(checkpoint)
    return src_module("config").Glm4Config.from_configs(hf_config, None)


@pytest.fixture(scope="module")
def weights_and_config():
    checkpoint = weights_path()
    return checkpoint, _config(checkpoint)


@contextlib.contextmanager
def _stub_tp(world_size: int, rank: int):
    """Give the GLM-4 modules and the nn layers a TP group without a process group."""
    group = SimpleNamespace(
        world_size=world_size, rank_in_group=rank, device_group=None
    )
    with (
        patch.object(src_module("model"), "get_tp_group", return_value=group),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=rank),
    ):
        yield


def _build_meta_model(config, world_size: int, rank: int):
    model_cls = src_module("model").Glm4ForCausalLM
    with _stub_tp(world_size, rank), torch.device("meta"):
        return model_cls(config)


def _model_mappings(config) -> dict:
    return src_module("model").Glm4ForCausalLM.checkpoint_mappings(
        config.num_hidden_layers
    )


def test_checkpoint_key_coverage():
    """The model refers to each checkpoint tensor, and to no other key."""
    checkpoint = metadata_path()
    config = _config(checkpoint)
    model = _build_meta_model(config, TP, 0)
    mappings = _model_mappings(config)

    referenced = set()
    for name, _ in model.named_parameters():
        keys = mappings.get(name, name)
        referenced.update(keys if isinstance(keys, list) else [keys])

    checkpoint_keys = set(read_weight_map(checkpoint))
    missing = sorted(referenced - checkpoint_keys)
    unused = sorted(checkpoint_keys - referenced)
    assert not missing, (
        f"Keys that the model needs but the checkpoint does not have: {missing[:20]}"
    )
    assert not unused, f"Checkpoint keys that the model does not load: {unused[:20]}"

    # 61 layers x (q, k, v, o, gate_up, down, 4 norms) + embed + norm + lm_head.
    assert len(checkpoint_keys) == config.num_hidden_layers * 10 + 3


def test_full_rank_load(weights_and_config):
    """Each loader of TP rank 7 gives a finite BF16 tensor of the parameter shape."""
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    checkpoint, config = weights_and_config
    model = _build_meta_model(config, TP, RANK)
    mappings = _model_mappings(config)

    with _stub_tp(TP, RANK):
        result = SafetensorsCheckpoint(checkpoint, None).load_sharded(
            RANK, TP, model, mappings, torch.device("cpu"), strict=True
        )

    assert not result.missing_keys
    assert not result.unexpected_keys
    params = dict(model.named_parameters())
    assert set(result.state_dict) == set(params)
    for name, tensor in result.state_dict.items():
        assert tuple(tensor.shape) == tuple(params[name].shape), (
            f"{name}: loader shape {tuple(tensor.shape)} is not the parameter "
            f"shape {tuple(params[name].shape)}"
        )
        assert tensor.dtype == torch.bfloat16, f"{name}: dtype is {tensor.dtype}"
        assert torch.isfinite(tensor).all(), f"{name}: tensor has non-finite values"
        assert tensor.abs().sum() > 0, f"{name}: tensor is all zeros"


def test_meta_built_qkv_loader_matches_cpu_built(weights_and_config):
    """A QKV loader made under the meta device gives the CPU-built result.

    This is a regression test. The serving runner and test_full_rank_load build
    the model under ``torch.device("meta")``. The RoPE column permutation of
    the QKV loader was then a meta tensor without index values, and the loaded
    Q/K columns were not finite.
    """
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    attention_cls = src_module("model").Glm4Attention
    checkpoint, config = weights_and_config

    layer_idx = 0
    prefix = f"model.layers.{layer_idx}.self_attn."
    mappings = {
        k.removeprefix(prefix): v
        for k, v in _model_mappings(config).items()
        if k.startswith(prefix)
    }
    ckpt = SafetensorsCheckpoint(checkpoint, None)
    with _stub_tp(TP, RANK):
        with torch.device("meta"):
            meta_attn = attention_cls(config, layer_idx)
        cpu_attn = attention_cls(config, layer_idx)
        meta_sd = ckpt.load_sharded(RANK, TP, meta_attn, mappings, "cpu").state_dict
        cpu_sd = ckpt.load_sharded(RANK, TP, cpu_attn, mappings, "cpu").state_dict

    qkv = meta_sd["qkv_proj_weight"]
    assert qkv.device.type == "cpu"
    assert torch.isfinite(qkv).all(), "meta-built qkv_proj_weight is not finite"
    assert torch.equal(qkv, cpu_sd["qkv_proj_weight"])


def test_transforms_match_reference(weights_and_config):
    """The rank-7 shards are equal to an independent slice of the checkpoint."""
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
    from vllm_neuron.utils.weight_loader import get_weight_loader

    model = src_module("model")
    checkpoint, config = weights_and_config
    layer_idx = 0
    prefix = f"model.layers.{layer_idx}"
    full_mappings = _model_mappings(config)
    attn_mappings = {
        k.removeprefix(f"{prefix}.self_attn."): v
        for k, v in full_mappings.items()
        if k.startswith(f"{prefix}.self_attn.")
    }
    mlp_mappings = {
        k.removeprefix(f"{prefix}.mlp."): v
        for k, v in full_mappings.items()
        if k.startswith(f"{prefix}.mlp.")
    }

    ckpt = SafetensorsCheckpoint(checkpoint, None)
    with _stub_tp(TP, RANK):
        attn = model.Glm4Attention(config, layer_idx)
        mlp = model.Glm4MLP(config)
        attn_sd = ckpt.load_sharded(RANK, TP, attn, attn_mappings, "cpu").state_dict
        mlp_sd = ckpt.load_sharded(RANK, TP, mlp, mlp_mappings, "cpu").state_dict

    raw = read_tensors(
        checkpoint,
        [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.mlp.gate_up_proj.weight",
            f"{prefix}.mlp.down_proj.weight",
        ],
    )

    hd = config.head_dim
    q_heads = config.num_attention_heads // TP  # 6
    kv_head = RANK // (TP // config.num_key_value_heads)  # replica owner: 1
    perm = reference_rope_permutation(hd, config.rotary_dim)

    def permute_heads(w_t: torch.Tensor, num_heads: int) -> torch.Tensor:
        rows = w_t.shape[0]
        return w_t.reshape(rows, num_heads, hd)[:, :, perm].reshape(rows, -1)

    q = raw[f"{prefix}.self_attn.q_proj.weight"]
    k = raw[f"{prefix}.self_attn.k_proj.weight"]
    v = raw[f"{prefix}.self_attn.v_proj.weight"]
    q_ref = permute_heads(q[RANK * q_heads * hd : (RANK + 1) * q_heads * hd].T, q_heads)
    k_ref = permute_heads(k[kv_head * hd : (kv_head + 1) * hd].T, 1)
    v_ref = v[kv_head * hd : (kv_head + 1) * hd].T
    qkv_ref = torch.cat([q_ref, k_ref, v_ref], dim=1)
    assert torch.equal(attn_sd["qkv_proj_weight"], qkv_ref)

    o = raw[f"{prefix}.self_attn.o_proj.weight"]
    o_ref = o[:, RANK * q_heads * hd : (RANK + 1) * q_heads * hd].T
    assert torch.equal(attn_sd["o_proj_weight"], o_ref)

    inter = config.intermediate_size
    i_rank = inter // TP
    gate_up = raw[f"{prefix}.mlp.gate_up_proj.weight"]
    assert gate_up.shape == (2 * inter, config.hidden_size)
    gate_ref = gate_up[:inter][RANK * i_rank : (RANK + 1) * i_rank].T
    up_ref = gate_up[inter:][RANK * i_rank : (RANK + 1) * i_rank].T
    down = raw[f"{prefix}.mlp.down_proj.weight"]
    down_ref = down[:, RANK * i_rank : (RANK + 1) * i_rank].T
    assert torch.equal(mlp_sd["gate_proj_weight"], gate_ref)
    assert torch.equal(mlp_sd["up_proj_weight"], up_ref)
    assert torch.equal(mlp_sd["down_proj_weight"], down_ref)

    # Vocab-sharded embedding and lm_head (untied, so two different tensors).
    vocab_rank = config.vocab_size // TP
    raw_vocab = read_tensors(
        checkpoint, ["model.embed_tokens.weight", "lm_head.weight"]
    )
    meta_model = _build_meta_model(config, TP, RANK)
    params = dict(meta_model.named_parameters())
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        loader = get_weight_loader(params[name])
        tensor = loader.load([FakeSafeSlice(raw_vocab[name])], RANK)
        expected = raw_vocab[name][RANK * vocab_rank : (RANK + 1) * vocab_rank]
        assert torch.equal(tensor, expected), f"{name}: rank {RANK} shard differs"
    assert not torch.equal(
        raw_vocab["model.embed_tokens.weight"], raw_vocab["lm_head.weight"]
    )
