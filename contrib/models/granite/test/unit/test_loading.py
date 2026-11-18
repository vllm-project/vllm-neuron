# SPDX-License-Identifier: Apache-2.0
"""Granite (ibm/PowerLM-3b) loading tests. CPU only.

These tests check:

1. The architecture and the factory: the architecture string of the checkpoint
   names the factory class of this folder, the factory accepts the real config
   and rejects the configurations that the port does not support.
2. The config: GraniteConfig keeps the four Granite scalars of the real
   config, and runs the FP32 checkpoint in bfloat16.
3. The revision: the checkpoint is the pinned HF revision.
4. The full-checkpoint TP4 load: each of 4 CPU ranks builds the full model and
   runs the production ``load_weights``. The test compares the shards with the
   raw checkpoint tensors. Every checkpoint tensor must have a mapping, and
   every parameter must load.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/granite/test/unit -v
"""

import importlib.util
import pathlib
import sys
from functools import partial

import pytest
import torch
import torch.distributed as dist

from vllm_neuron import envs
from vllm_neuron.utils.executor import MPExecutor


def _load_local_src():
    """Load this model folder's ``test/_local_src.py`` by file location.

    Every contrib model ships a file with this name, so a plain
    ``import _local_src`` would bind whichever copy entered ``sys.modules``
    first. The per-model name used here cannot collide.
    """
    name = "contrib_granite_local_src"
    if name not in sys.modules:
        path = pathlib.Path(__file__).resolve().parents[1] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()
_reference = _local_src.helper("granite_reference")

HF_REVISION = _reference.HF_REVISION
MODEL_ID = _reference.MODEL_ID
NUM_CHECKPOINT_TENSORS = _reference.NUM_CHECKPOINT_TENSORS
TP_SIZE = _reference.TP_SIZE
fingerprint_mismatches = _reference.fingerprint_mismatches
full_mappings = _reference.full_mappings
index_keys = _reference.index_keys
load_hf_config = _reference.load_hf_config
read_tensors = _reference.read_tensors
resolve_checkpoint = _reference.resolve_checkpoint

pytestmark = [
    pytest.mark.skipif(
        not envs.VLLM_NEURON_CPU_MODE,
        reason="Granite loading tests run in CPU mode (VLLM_NEURON_CPU_MODE=1).",
    ),
]

# Layers that the value check compares with the raw checkpoint.
CHECKED_LAYERS = (0, 39)


@pytest.fixture(scope="module")
def checkpoint():
    torch.manual_seed(42)
    return resolve_checkpoint()


# =============================================================================
# 1. Registry and factory
# =============================================================================


def test_registry_and_factory_validation(checkpoint):
    """Match the architecture string. Accept the real config. Reject bad ones."""
    Factory = _local_src.load().GraniteForCausalLM

    # A contrib model is not in ``vllm_neuron.model.registry.get_models()`` by
    # design, so there is no registry wiring to assert. What a user has to get
    # right is the pair they register: the architecture string of the
    # checkpoint and the factory class of this folder.
    hf_config = load_hf_config(checkpoint)
    assert hf_config.architectures == ["GraniteForCausalLM"]
    assert Factory.__name__ == "GraniteForCausalLM"
    Factory._validate_config(hf_config, None)

    bad_act = load_hf_config(checkpoint)
    bad_act.hidden_act = "gelu"
    with pytest.raises(ValueError, match="does not support hidden_act"):
        Factory._validate_config(bad_act, None)

    bad_rope = load_hf_config(checkpoint)
    bad_rope.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    bad_rope.rope_parameters = {"rope_type": "linear", "factor": 2.0}
    with pytest.raises(ValueError, match="does not support RoPE type"):
        Factory._validate_config(bad_rope, None)


# =============================================================================
# 2. Config
# =============================================================================


def test_config_from_real_checkpoint(checkpoint):
    """Keep the Granite scalars and the geometry of the real config."""
    GraniteConfig = _local_src.load("config").GraniteConfig

    hf_config = load_hf_config(checkpoint)
    cfg = GraniteConfig.from_configs(hf_config)

    assert cfg.hidden_size == 2304
    assert cfg.intermediate_size == 9216
    assert cfg.num_hidden_layers == 40
    assert cfg.num_attention_heads == 36
    assert cfg.num_key_value_heads == 36
    assert cfg.head_dim == 64
    assert cfg.vocab_size == 49152
    assert cfg.attention_bias is True
    assert cfg.mlp_bias is True
    assert cfg.tie_word_embeddings is True
    assert cfg.rope_theta == pytest.approx(10000.0)
    assert cfg.rms_norm_eps == pytest.approx(1e-5)
    assert cfg.embedding_multiplier == pytest.approx(hf_config.embedding_multiplier)
    assert cfg.attention_multiplier == pytest.approx(hf_config.attention_multiplier)
    assert cfg.residual_multiplier == pytest.approx(hf_config.residual_multiplier)
    assert cfg.logits_scaling == pytest.approx(hf_config.logits_scaling)
    assert cfg.embedding_multiplier == pytest.approx(12.0)
    assert cfg.attention_multiplier == pytest.approx(0.015625)
    assert cfg.residual_multiplier == pytest.approx(0.22)
    assert cfg.logits_scaling == pytest.approx(9.0)
    # The FP32 checkpoint runs in bfloat16.
    assert cfg.torch_dtype == torch.bfloat16


# =============================================================================
# 3. Revision
# =============================================================================


def test_checkpoint_is_pinned_revision(checkpoint):
    """The resolved checkpoint must be HF revision HF_REVISION."""
    problems = fingerprint_mismatches(checkpoint)
    assert not problems, (
        f"Checkpoint {checkpoint} of {MODEL_ID} is not revision {HF_REVISION}: "
        f"{problems}"
    )
    keys = index_keys(checkpoint)
    assert len(keys) == NUM_CHECKPOINT_TENSORS
    assert "lm_head.weight" not in keys


# =============================================================================
# 4. Full-checkpoint TP4 load on CPU ranks
# =============================================================================


def _selected_param_names(layer_idx: int) -> list[str]:
    attn = f"model.layers.{layer_idx}.self_attn"
    mlp = f"model.layers.{layer_idx}.mlp"
    names = [
        f"{attn}.qkv_proj_weight",
        f"{attn}.qkv_proj_bias",
        f"{attn}.o_proj_bias",
        f"{mlp}.gate_proj_bias",
        f"{mlp}.up_proj_bias",
        f"{mlp}.down_proj_bias",
        f"model.layers.{layer_idx}.input_layernorm.weight",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight",
    ]
    if layer_idx == 0:
        names += [
            f"{attn}.o_proj_weight",
            f"{mlp}.gate_proj_weight",
            f"{mlp}.up_proj_weight",
            f"{mlp}.down_proj_weight",
        ]
    return names


class _LoadCheck(torch.nn.Module):
    """Hold a loaded model. Check the rank-local parameters in the worker.

    The executor sends only tensors back to the test process. Thus the worker
    does the checks and raises AssertionError with the details on a failure.
    """

    def __init__(self, model: torch.nn.Module, checkpoint_path: str):
        super().__init__()
        self.loaded = model
        self.checkpoint_path = checkpoint_path

    def forward(self):
        rank = dist.get_rank()
        mappings = full_mappings(load_hf_config(self.checkpoint_path))
        named = dict(self.loaded.named_parameters())

        # Every model parameter has a mapping, so the loader loads it.
        assert set(named) == set(mappings), (
            f"rank {rank}: parameters without a mapping: "
            f"{sorted(set(named) - set(mappings))}; mappings without a "
            f"parameter: {sorted(set(mappings) - set(named))}"
        )
        bad = [
            name
            for name, p in named.items()
            if p.dtype != torch.bfloat16
            or p.device.type != "cpu"
            or p.is_meta
            or not bool(torch.isfinite(p).all())
        ]
        assert not bad, f"rank {rank}: parameters with a bad dtype or value: {bad}"

        selected = {}
        for layer_idx in CHECKED_LAYERS:
            for name in _selected_param_names(layer_idx):
                selected[name] = named[name]
        for name in ("model.embed_tokens.weight", "lm_head.weight"):
            # The first and the last rows of the vocabulary shard.
            selected[name + "[head]"] = named[name][:8]
            selected[name + "[tail]"] = named[name][-8:]
        selected["model.norm.weight"] = named["model.norm.weight"]

        raw = read_tensors(
            self.checkpoint_path, _raw_keys_for(sorted(selected), mappings)
        )
        for name, actual in selected.items():
            expected = _expected_shard(raw, name, rank)
            assert actual.shape == expected.shape, (
                f"rank {rank} {name}: shape {tuple(actual.shape)}, "
                f"expected {tuple(expected.shape)}"
            )
            assert torch.equal(actual, expected), (
                f"rank {rank} {name}: the shard is not equal to the raw tensor"
            )
        return torch.tensor([rank, len(named), len(selected)], dtype=torch.int64)


def _load_full_model(checkpoint_path: str):
    """Build the full model with the factory and run the production load."""
    Factory = _local_src.load().GraniteForCausalLM

    hf_config = load_hf_config(checkpoint_path)
    model = Factory.from_configs(hf_config, None)
    model.load_weights(checkpoint_path, torch.device("cpu"), None)
    return _LoadCheck(model, checkpoint_path)


def _expected_shard(raw: dict, name: str, rank: int) -> torch.Tensor:
    """Compute the expected rank shard from the raw FP32 checkpoint tensors."""
    tp = TP_SIZE
    heads_per_rank = 36 // tp
    q_size = heads_per_rank * 64
    i_size = 9216 // tp
    vocab_size = 49152 // tp
    bf16 = torch.bfloat16

    def rows(t, size):
        return t[rank * size : (rank + 1) * size]

    if name.endswith("qkv_proj_weight"):
        p = name[: -len("qkv_proj_weight")]
        parts = [rows(raw[f"{p}{x}_proj.weight"], q_size) for x in "qkv"]
        return torch.cat(parts, dim=0).T.to(bf16)
    if name.endswith("qkv_proj_bias"):
        p = name[: -len("qkv_proj_bias")]
        parts = [rows(raw[f"{p}{x}_proj.bias"], q_size) for x in "qkv"]
        return torch.cat(parts, dim=0).to(bf16)
    if name.endswith("o_proj_weight"):
        w = raw[name.replace("o_proj_weight", "o_proj.weight")]
        return w[:, rank * q_size : (rank + 1) * q_size].T.to(bf16)
    if name.endswith("o_proj_bias") or name.endswith("down_proj_bias"):
        b = raw[name.replace("_bias", ".bias")]
        return (b / tp).to(bf16)
    if name.endswith("gate_proj_weight") or name.endswith("up_proj_weight"):
        w = raw[name.replace("_weight", ".weight")]
        return rows(w, i_size).T.to(bf16)
    if name.endswith("gate_proj_bias") or name.endswith("up_proj_bias"):
        return rows(raw[name.replace("_bias", ".bias")], i_size).to(bf16)
    if name.endswith("down_proj_weight"):
        w = raw[name.replace("down_proj_weight", "down_proj.weight")]
        return w[:, rank * i_size : (rank + 1) * i_size].T.to(bf16)
    if name.endswith("layernorm.weight") or name == "model.norm.weight":
        return raw[name].to(bf16)
    if name.endswith("[head]") or name.endswith("[tail]"):
        # The tied lm_head loads from the embedding tensor.
        shard = rows(raw["model.embed_tokens.weight"], vocab_size)
        return (shard[:8] if name.endswith("[head]") else shard[-8:]).to(bf16)
    raise KeyError(name)


def _raw_keys_for(names: list[str], mappings: dict) -> list[str]:
    keys = set()
    for name in names:
        base = name.split("[")[0]
        ck = mappings[base]
        keys.update(ck if isinstance(ck, list) else [ck])
    return sorted(keys)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_full_checkpoint_tp4_load(checkpoint, world_size):
    """Load all 642 tensors on 4 CPU ranks. Compare the shards with the raw tensors."""
    hf_config = load_hf_config(checkpoint)
    mappings = full_mappings(hf_config)

    # Every checkpoint tensor has a mapping, and every mapping has a tensor.
    referenced = set()
    for ck in mappings.values():
        referenced.update(ck if isinstance(ck, list) else [ck])
    ckpt_keys = set(index_keys(checkpoint))
    assert ckpt_keys - referenced == set(), "Checkpoint tensors without a mapping"
    assert referenced - ckpt_keys == set(), "Mappings without a checkpoint tensor"

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(_load_full_model, checkpoint),
        device="cpu",
        load_timeout=3600,
    )
    executor.dispatch()
    # A worker raises on a failed check. collect() then raises RuntimeError
    # with the worker message.
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == world_size
    ranks = sorted(int(out[0]) for out in outputs)
    assert ranks == list(range(world_size)), ranks
    for out in outputs:
        assert int(out[1]) == len(mappings), (
            f"rank {int(out[0])}: {int(out[1])} parameters, expected {len(mappings)}"
        )
