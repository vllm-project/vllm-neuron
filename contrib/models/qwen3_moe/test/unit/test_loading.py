# SPDX-License-Identifier: Apache-2.0
"""Qwen3-MoE loading tests (CPU).

1. ``test_registration_contract``: the plugin entry point registers the
   architecture string of the checkpoint, and the factory satisfies vLLM's
   model protocol for a text generation model.
2. ``test_config_parse``: the config of ``src/`` for the pinned
   ``config.json``, read through transformers and read as raw JSON, keeps the
   MoE, attention and RoPE fields. ``rope_theta`` also comes through the
   transformers 5 ``rope_parameters`` form.
3. ``test_factory_rejects_unsupported``: the factory refuses the options that
   this implementation does not have.
4. ``test_checkpoint_mapping_coverage``: the mappings of the real geometry
   (48 layers, 128 experts) use each tensor name of the pinned checkpoint
   index exactly one time. The test compares with a fingerprint of the index
   names, so it needs no download.
5. ``test_expert_loader_layout``: the expert loaders fuse gate/up and stack
   down in the kernel layout, with and without the EP selection.
6. ``test_full_model_load``: a TP 2 CPU load of the tiny random-weight
   checkpoint through ``load_weights``, with EP 2 and without EP, checked
   against slices that the test takes from the checkpoint itself.

No Neuron hardware and no checkpoint download are needed. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen3_moe/test/unit/test_loading.py -v
"""

import json
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from vllm_neuron import envs
from vllm_neuron.model.neuron_config import NeuronConfig

from _local_src import src_module, src_package
from qwen3_moe_test_utils import (
    ARCHITECTURE,
    CHECKPOINT_NAMES_SHA256,
    COLLECT_TIMEOUT_S,
    NUM_CHECKPOINT_TENSORS,
    PINNED_CONFIG,
    TINY_OVERRIDES,
    TINY_TP,
    load_hf_config,
    make_tiny_checkpoint,
    names_fingerprint,
    read_checkpoint_tensors,
    set_mock_vllm_config,
    tiny_hf_config,
    write_pinned_config,
)

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE, reason="Set VLLM_NEURON_CPU_MODE=1."
)

# The parameters without a mapping load the checkpoint tensor of the same name.
PASSTHROUGH = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}


class MockSlice:
    """Mock checkpoint slice with __getitem__ and get_shape."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]

    def get_shape(self):
        return self.tensor.shape


@pytest.fixture(scope="module")
def pinned_config_dir(tmp_path_factory) -> str:
    """A directory with the pinned ``config.json`` and no weights."""
    return write_pinned_config(str(tmp_path_factory.mktemp("qwen3_moe_config")))


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_qwen3_moe")))


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

    factory_cls = src_package().Qwen3MoeForCausalLM
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
    assert calls == [(ARCHITECTURE, "qwen3_moe_neuron.factory:Qwen3MoeForCausalLM")]
    with open(os.path.join(os.path.dirname(__file__), "../../pyproject.toml")) as f:
        pyproject = f.read()
    assert 'qwen3_moe_neuron = "qwen3_moe_neuron.plugin:register"' in pyproject
    assert 'package-dir = {"qwen3_moe_neuron" = "src"}' in pyproject

    # vLLM 0.29 checks the model protocol when the class is registered.
    assert is_vllm_model(factory_cls)
    assert is_text_generation_model(factory_cls)


# =============================================================================
# 2. Config
# =============================================================================


def _check_real_geometry(config) -> None:
    assert config.num_hidden_layers == 48
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 4
    assert config.head_dim == 128
    assert config.num_experts == 128
    assert config.num_experts_per_tok == 8
    assert config.moe_intermediate_size == 768
    assert config.norm_topk_prob is True
    assert config.decoder_sparse_step == 1
    assert config.mlp_only_layers == []
    assert config.vocab_size == 151936
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 1_000_000.0
    assert config.max_position_embeddings == 40960
    assert config.tie_word_embeddings is False
    assert config.torch_dtype == torch.bfloat16


def test_config_parse(pinned_config_dir):
    """The config of src/ keeps the fields of the pinned config.json."""
    config_cls = src_module("config").Qwen3MoeConfig
    neuron_config = NeuronConfig()

    # Through transformers (transformers 5 moves rope_theta into
    # rope_parameters), as the model runner reads it.
    hf_config = load_hf_config(pinned_config_dir)
    config = config_cls.from_configs(hf_config, neuron_config)
    _check_real_geometry(config)
    assert config.neuron_config is neuron_config

    # As raw JSON.
    config = config_cls.from_configs(
        os.path.join(pinned_config_dir, "config.json"), neuron_config
    )
    _check_real_geometry(config)

    # The real geometry equals the dataclass defaults, so a field that the
    # conversion drops would still pass above. Values that differ from every
    # default must come through: the tiny config through transformers
    # (transformers 5 stores num_experts as num_local_experts), as its
    # to_dict(), and as raw JSON.
    tiny_hf = tiny_hf_config(rope_theta=5.0e5)
    for source in (tiny_hf, tiny_hf.to_dict(), dict(PINNED_CONFIG, **TINY_OVERRIDES)):
        config = config_cls.from_configs(source, neuron_config)
        for key, value in TINY_OVERRIDES.items():
            if hasattr(config, key):
                assert getattr(config, key) == value, (key, type(source))
        assert config.num_experts == TINY_OVERRIDES["num_experts"]
    assert config_cls.from_configs(tiny_hf).rope_theta == 5.0e5

    # A theta that only rope_parameters carries is not replaced by the default.
    raw = dict(PINNED_CONFIG, rope_theta=None)
    raw["rope_parameters"] = {"rope_type": "default", "rope_theta": 5.0e5}
    assert config_cls.from_configs(raw).rope_theta == 5.0e5


# =============================================================================
# 3. Factory validation
# =============================================================================


def test_factory_rejects_unsupported(pinned_config_dir):
    """The factory refuses each option that the implementation does not have."""
    factory_cls = src_package().Qwen3MoeForCausalLM
    base = dict(PINNED_CONFIG)

    # The pinned config passes, as a dict and through transformers.
    factory_cls._validate_config(base, NeuronConfig())
    factory_cls._validate_config(load_hf_config(pinned_config_dir), NeuronConfig())

    rejected = {
        "quantized checkpoint": dict(base, quantization_config={"quant_method": "fp8"}),
        "float16 checkpoint": dict(base, torch_dtype="float16"),
        "dense MLP layer": dict(base, mlp_only_layers=[3]),
        "sparse step 2": dict(base, decoder_sparse_step=2),
        "no top-k renorm": dict(base, norm_topk_prob=False),
        "sliding window": dict(base, use_sliding_window=True, sliding_window=4096),
        "non-full layer type": dict(
            base, layer_types=["sliding_attention"] * base["num_hidden_layers"]
        ),
        "attention bias": dict(base, attention_bias=True),
        "gelu experts": dict(base, hidden_act="gelu"),
        "yarn rope": dict(
            base,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        ),
    }
    for what, bad in rejected.items():
        with pytest.raises(ValueError):
            factory_cls._validate_config(bad, NeuronConfig())
            pytest.fail(f"The factory accepted a config with {what}.")
    with pytest.raises(ValueError):
        factory_cls._validate_config(base, NeuronConfig(quantization="fp8"))


# =============================================================================
# 4. Checkpoint mappings
# =============================================================================


def test_checkpoint_mapping_coverage():
    """The mappings use each tensor name of the pinned checkpoint exactly once."""
    mappings_fn = src_module("model").qwen3_moe_checkpoint_mappings
    mappings = mappings_fn(
        PINNED_CONFIG["num_hidden_layers"], PINNED_CONFIG["num_experts"]
    )
    mapped = []
    for keys in mappings.values():
        mapped.extend(keys if isinstance(keys, list) else [keys])

    assert len(mapped) == len(set(mapped)), "A checkpoint key maps twice."
    assert names_fingerprint(mapped) == (
        NUM_CHECKPOINT_TENSORS,
        CHECKPOINT_NAMES_SHA256,
    ), "The mapped keys are not the tensor names of the pinned checkpoint."


# =============================================================================
# 5. Expert loaders
# =============================================================================


def test_expert_loader_layout():
    """The loaders give [E_L, H, 2 * I_r] and [E_L, I_r, H], with the EP selection."""
    from vllm_neuron.utils.weight_loader import expert_parallel_grouped_loader

    model_mod = src_module("model")
    num_experts, hidden, inter = 8, 16, 12
    gates = [torch.randn(inter, hidden) for _ in range(num_experts)]
    ups = [torch.randn(inter, hidden) for _ in range(num_experts)]
    downs = [torch.randn(hidden, inter) for _ in range(num_experts)]
    gate_up_slices = [MockSlice(t) for t in gates + ups]
    down_slices = [MockSlice(t) for t in downs]

    # No EP: the intermediate dim is split over 2 ranks.
    tp = 2
    r = inter // tp
    for rank in range(tp):
        gate_up = model_mod.expert_gate_up_loader(tp).transform(gate_up_slices, rank)
        assert gate_up.shape == (num_experts, hidden, 2 * r)
        rows = slice(rank * r, (rank + 1) * r)
        for e in range(num_experts):
            torch.testing.assert_close(gate_up[e, :, :r], gates[e][rows].T)
            torch.testing.assert_close(gate_up[e, :, r:], ups[e][rows].T)
        down = model_mod.expert_down_loader(tp).transform(down_slices, rank)
        assert down.shape == (num_experts, r, hidden)
        for e in range(num_experts):
            torch.testing.assert_close(down[e], downs[e][:, rows].T)

    # EP 4: complete experts on each rank.
    ep = 4
    local = num_experts // ep
    for ep_rank in range(ep):
        indices = list(range(ep_rank * local, (ep_rank + 1) * local))
        gate_up = expert_parallel_grouped_loader(
            indices, model_mod.expert_gate_up_loader(1), num_experts
        ).transform(gate_up_slices, ep_rank)
        down = expert_parallel_grouped_loader(
            indices, model_mod.expert_down_loader(1), num_experts
        ).transform(down_slices, ep_rank)
        assert gate_up.shape == (local, hidden, 2 * inter)
        assert down.shape == (local, inter, hidden)
        for i, e in enumerate(indices):
            torch.testing.assert_close(gate_up[i, :, :inter], gates[e].T)
            torch.testing.assert_close(gate_up[i, :, inter:], ups[e].T)
            torch.testing.assert_close(down[i], downs[e].T)


# =============================================================================
# 6. Full-model TP 2 load of the tiny checkpoint
# =============================================================================


def _check_full_model_load(checkpoint: str, device: str, ep: bool) -> torch.Tensor:
    """Load all weights for this rank the way the runner does. Raise on a problem.

    Return the counts [layers, loaded parameters, local experts].
    """
    set_mock_vllm_config(ep_enabled=ep)
    factory_cls = src_package().Qwen3MoeForCausalLM
    model_cls = src_module("model").Qwen3MoeForCausalLM

    rank = dist.get_rank()
    ws = dist.get_world_size()
    hf_config = load_hf_config(checkpoint)
    # The model runner builds the model inside a meta device context and then
    # calls load_weights.
    with torch.device("meta"):
        model = factory_cls.from_configs(hf_config, NeuronConfig())
    if not isinstance(model, model_cls):
        raise AssertionError(f"The factory built {type(model).__name__}")
    model.load_weights(checkpoint, torch.device(device), None)

    params = dict(model.named_parameters())
    mappings = model.checkpoint_mappings()
    problems = {
        "on_meta": [n for n, p in params.items() if p.is_meta],
        "non_bf16": [n for n, p in params.items() if p.dtype != torch.bfloat16],
        "non_finite": [
            n for n, p in params.items() if not torch.isfinite(p.float()).all().item()
        ],
        "unmapped": sorted(set(params) - set(mappings) - PASSTHROUGH),
        "missing": sorted(set(mappings) - set(params)),
    }
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        raise AssertionError(f"Rank {rank}: the weight load has problems: {problems}")

    # Independent checks: slice the checkpoint tensors here, without the src/
    # loaders.
    num_experts = hf_config.num_experts
    inter = hf_config.moe_intermediate_size
    if ep:
        local = num_experts // ws
        experts = range(rank * local, (rank + 1) * local)
        cols = slice(0, inter)
    else:
        experts = range(num_experts)
        shard = inter // ws
        cols = slice(rank * shard, (rank + 1) * shard)

    head_dim = hf_config.head_dim
    q_size = hf_config.num_attention_heads // ws * head_dim
    # At TP 2 the 2 KV heads of the tiny checkpoint give 1 KV head per rank.
    assert ws == hf_config.num_key_value_heads
    kv_head = rank
    vocab = hf_config.vocab_size // ws

    prefix = "model.layers.1"
    tensors = read_checkpoint_tensors(checkpoint)
    attn = f"{prefix}.self_attn"
    layer = model.model.layers[1]

    def expert(name: str, e: int) -> torch.Tensor:
        return tensors[f"{prefix}.mlp.experts.{e}.{name}.weight"]

    expected = {
        "router": tensors[f"{prefix}.mlp.gate.weight"],
        "gate_up": torch.stack(
            [
                torch.cat(
                    [expert("gate_proj", e)[cols].T, expert("up_proj", e)[cols].T],
                    dim=-1,
                )
                for e in experts
            ]
        ),
        "down": torch.stack([expert("down_proj", e)[:, cols].T for e in experts]),
        "qkv": torch.cat(
            [
                tensors[f"{attn}.q_proj.weight"][rank * q_size : (rank + 1) * q_size].T,
                tensors[f"{attn}.k_proj.weight"][
                    kv_head * head_dim : (kv_head + 1) * head_dim
                ].T,
                tensors[f"{attn}.v_proj.weight"][
                    kv_head * head_dim : (kv_head + 1) * head_dim
                ].T,
            ],
            dim=1,
        ),
        "o": tensors[f"{attn}.o_proj.weight"][:, rank * q_size : (rank + 1) * q_size].T,
        "q_norm": tensors[f"{attn}.q_norm.weight"],
        "k_norm": tensors[f"{attn}.k_norm.weight"],
        "post_norm": tensors[f"{prefix}.post_attention_layernorm.weight"],
        "final_norm": tensors["model.norm.weight"],
        "embed": tensors["model.embed_tokens.weight"][
            rank * vocab : (rank + 1) * vocab
        ],
        "lm_head": tensors["lm_head.weight"][rank * vocab : (rank + 1) * vocab],
    }
    actual = {
        "router": layer.mlp.router_weight,
        "gate_up": layer.mlp.gate_up_proj_weight,
        "down": layer.mlp.down_proj_weight,
        "qkv": layer.self_attn.qkv_proj_weight,
        "o": layer.self_attn.o_proj_weight,
        "q_norm": layer.self_attn.q_norm.weight,
        "k_norm": layer.self_attn.k_norm.weight,
        "post_norm": layer.post_attention_layernorm.weight,
        "final_norm": model.model.norm.weight,
        "embed": model.model.embed_tokens.weight,
        "lm_head": model.lm_head.weight,
    }
    for name, want in expected.items():
        got = actual[name].detach().cpu()
        if got.shape != want.shape or not torch.equal(got, want):
            raise AssertionError(
                f"Rank {rank}: {name} does not match the checkpoint "
                f"(shape {tuple(got.shape)}, expected {tuple(want.shape)})."
            )
    # The LM head is not tied to the embedding.
    if torch.equal(model.lm_head.weight.detach().cpu(), expected["embed"]):
        raise AssertionError(f"Rank {rank}: lm_head is tied to the embedding.")

    return torch.tensor(
        [len(model.model.layers), len(params), layer.mlp.num_local_experts],
        dtype=torch.int64,
    )


class _LoadReport(torch.nn.Module):
    """Load the model at construction. Return the counts at each call."""

    def __init__(self, checkpoint: str, device: str, ep: bool):
        super().__init__()
        self.error = None
        self.counts = None
        try:
            self.counts = _check_full_model_load(checkpoint, device, ep)
        except Exception as exc:  # noqa: BLE001 - report at the RUN step
            self.error = exc

    def forward(self):
        if self.error is not None:
            raise self.error
        return self.counts


def _full_model_load(checkpoint: str, device: str, ep: bool) -> torch.nn.Module:
    """Top-level picklable model load for the full-checkpoint test."""
    return _LoadReport(checkpoint, device, ep)


@pytest.mark.parametrize("ep", [True, False], ids=["ep2", "tp2"])
def test_full_model_load(tiny_checkpoint, ep: bool):
    """Load every layer of the tiny checkpoint at TP 2 through load_weights."""
    from vllm_neuron.utils.executor import MPExecutor

    with open(os.path.join(tiny_checkpoint, "config.json")) as f:
        assert json.load(f)["architectures"] == [ARCHITECTURE]

    executor = MPExecutor(
        world_size=TINY_TP,
        model_load=partial(
            _full_model_load, checkpoint=tiny_checkpoint, device="cpu", ep=ep
        ),
        device="cpu",
        ep_degree=TINY_TP if ep else 1,
    )
    executor.dispatch()
    counts = executor.collect(timeout=COLLECT_TIMEOUT_S)

    num_layers = TINY_OVERRIDES["num_hidden_layers"]
    # Per layer: fused QKV, O, Q/K norms, 2 layer norms, router, gate/up, down.
    num_params = 3 + 9 * num_layers
    num_experts = TINY_OVERRIDES["num_experts"]
    local_experts = num_experts // TINY_TP if ep else num_experts
    assert len(counts) == TINY_TP
    for rank, c in enumerate(counts):
        c = torch.as_tensor(c).tolist()
        assert c == [num_layers, num_params, local_experts], f"rank {rank}: {c}"
