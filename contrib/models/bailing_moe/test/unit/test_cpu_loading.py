# SPDX-License-Identifier: Apache-2.0
"""CPU loading tests for BailingMoeV2 (inclusionAI/Ling-mini-2.0).

These tests run on CPU only (``VLLM_NEURON_CPU_MODE=1``). They check:
  1. The factory, the registration and the config translation.
  2. The weight loaders against independent expected shards.
  3. The partial-RoPE head permutation and the router against the reference.
  4. The full-model ``load_weights`` path on 8 ranks with EP 8.

Run:
    VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
        pytest contrib/models/bailing_moe/test/unit/test_cpu_loading.py -v
"""

import importlib.util
from functools import partial
from pathlib import Path

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.executor import MPExecutor
from vllm_neuron.utils.weight_loader import expert_parallel_grouped_loader

# This folder's own code loads by file location, so the folder keeps working
# when it is copied out of the repository. See ``test/_local_src.py``.
_LOADER = importlib.util.spec_from_file_location(
    "contrib_bailing_moe_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER)
_LOADER.loader.exec_module(_local_src)

BailingMoeV2Config = _local_src.load("config").BailingMoeV2Config
_model = _local_src.load("model")
BailingMoeV2RotaryEmbedding = _model.BailingMoeV2RotaryEmbedding
bailing_qkv_weight_loader = _model.bailing_qkv_weight_loader
bailing_route = _model.bailing_route
expert_down_loader = _model.expert_down_loader
expert_gate_up_loader = _model.expert_gate_up_loader
rope_head_permutation = _model.rope_head_permutation
rope_permuted_vector_loader = _model.rope_permuted_vector_loader

utils = _local_src.load_test_module("utils")
FakeSafeSlice = utils.FakeSafeSlice
_reference = _local_src.load_test_module("reference_model")
load_reference_tensors = _reference.load_reference_tensors
read_reference_config = _reference.read_reference_config
reference_apply_partial_rope = _reference.reference_apply_partial_rope
reference_gate = _reference.reference_gate
reference_rotary = _reference.reference_rotary

RANDOM_SEED = 42


@pytest.fixture(autouse=True)
def require_cpu_mode():
    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.fail(
            "test_cpu_loading.py is a CPU test. Set VLLM_NEURON_CPU_MODE=1 to run it."
        )
    torch.manual_seed(RANDOM_SEED)


@pytest.fixture(scope="module")
def checkpoint():
    return utils.resolve_checkpoint()


def _independent_rope_perm() -> torch.Tensor:
    # Rotary dims [0:32] and [32:64] go to [0:32] and [64:96]. The pass-through
    # dims [64:96] and [96:128] go to [32:64] and [96:128].
    return torch.cat(
        [
            torch.arange(0, 32),
            torch.arange(64, 96),
            torch.arange(32, 64),
            torch.arange(96, 128),
        ]
    )


def test_factory_registry_and_config(checkpoint, monkeypatch):
    """Check the registration, the config translation and the factory guards."""
    from vllm import ModelRegistry

    BailingMoeV2ForCausalLM = _local_src.load("factory").BailingMoeV2ForCausalLM

    hf = utils.read_hf_config(checkpoint)
    assert hf["architectures"] == ["BailingMoeV2ForCausalLM"]
    # A contrib model is not in ``vllm_neuron.model.registry``. The user
    # registers the architecture with vLLM (see contrib/README.md, "How do I
    # register a custom model with vLLM?"), so this test registers it the same
    # way and checks that vLLM holds it under the architecture string of the
    # checkpoint. The copy of ``ModelRegistry.models`` keeps the registry of
    # the pytest session unchanged.
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    ModelRegistry.register_model("BailingMoeV2ForCausalLM", BailingMoeV2ForCausalLM)
    assert "BailingMoeV2ForCausalLM" in ModelRegistry.models

    cfg = BailingMoeV2Config.from_configs(hf, NeuronConfig())
    assert cfg.num_hidden_layers == 20
    assert cfg.hidden_size == 2048
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (
        16,
        4,
        128,
    )
    assert cfg.rotary_dim == 64
    assert cfg.vocab_size == utils.VOCAB_SIZE
    assert cfg.num_experts == 256 and cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 512 and cfg.num_shared_experts == 1
    assert (cfg.n_group, cfg.topk_group) == (8, 4)
    assert cfg.routed_scaling_factor == pytest.approx(2.5)
    assert cfg.norm_topk_prob is True
    assert cfg.torch_dtype == torch.bfloat16
    assert not cfg.is_moe_layer(0)
    assert all(cfg.is_moe_layer(i) for i in range(1, 20))

    # The factory refuses the configurations that have no implementation.
    for override in (
        {"tie_word_embeddings": True},
        {"score_function": "softmax"},
        {"num_nextn_predict_layers": 1},
        {"use_qkv_bias": True},
        {"quantization_config": {"quant_method": "fp8"}},
    ):
        with pytest.raises(ValueError):
            BailingMoeV2ForCausalLM._validate_config({**hf, **override}, NeuronConfig())
    BailingMoeV2ForCausalLM._validate_config(hf, NeuronConfig())


def test_weight_loaders_match_independent_shards():
    """Drive each custom loader on synthetic tensors for all 8 ranks."""
    nq, nkv, hd, hidden = 16, 4, 128, 64
    perm = _independent_rope_perm()
    assert torch.equal(rope_head_permutation(hd, 64), perm)

    w_qkv = torch.randn((nq + 2 * nkv) * hd, hidden)
    q_heads = w_qkv[: nq * hd].view(nq, hd, hidden)
    k_heads = w_qkv[nq * hd : (nq + nkv) * hd].view(nkv, hd, hidden)
    v_heads = w_qkv[(nq + nkv) * hd :].view(nkv, hd, hidden)
    qkv_loader = bailing_qkv_weight_loader(
        nq, nkv, nq // 8, 1, 8 // nkv, hd, rope_head_permutation(hd, 64), 8
    )
    for rank in range(8):
        expected = torch.cat(
            [
                q_heads[2 * rank][perm],
                q_heads[2 * rank + 1][perm],
                k_heads[rank // 2][perm],
                v_heads[rank // 2],
            ],
            dim=0,
        ).T
        actual = qkv_loader.load([FakeSafeSlice(w_qkv)], rank)
        assert actual.shape == (hidden, 4 * hd)
        assert torch.equal(actual, expected), f"qkv shard of rank {rank}"

    gamma = torch.randn(hd)
    assert torch.equal(
        rope_permuted_vector_loader(rope_head_permutation(hd, 64)).load(
            [FakeSafeSlice(gamma)], 3
        ),
        gamma[perm],
    )

    num_experts, inter, ep = 16, 8, 8
    local = num_experts // ep
    gates = [torch.randn(inter, hidden) for _ in range(num_experts)]
    ups = [torch.randn(inter, hidden) for _ in range(num_experts)]
    downs = [torch.randn(hidden, inter) for _ in range(num_experts)]
    for rank in range(ep):
        experts = list(range(rank * local, (rank + 1) * local))
        gate_up = expert_parallel_grouped_loader(
            experts,
            expert_gate_up_loader(inter, 1, local),
            total_num_experts=num_experts,
        ).load([FakeSafeSlice(t) for t in gates + ups], rank)
        down = expert_parallel_grouped_loader(
            experts,
            expert_down_loader(inter, 1, local),
            total_num_experts=num_experts,
        ).load([FakeSafeSlice(t) for t in downs], rank)
        expected_gate_up = torch.stack(
            [torch.cat([gates[e].T, ups[e].T], dim=-1) for e in experts]
        )
        expected_down = torch.stack([downs[e].T for e in experts])
        assert torch.equal(gate_up, expected_gate_up), f"gate_up of rank {rank}"
        assert torch.equal(down, expected_down), f"down of rank {rank}"


def test_partial_rope_and_router_match_reference(checkpoint):
    """Check the permuted full-head RoPE and the router against the reference."""
    hf = utils.read_hf_config(checkpoint)
    cfg = BailingMoeV2Config.from_configs(hf, NeuronConfig())
    ref_cfg = read_reference_config(checkpoint)
    perm = _independent_rope_perm()
    inv_perm = torch.argsort(perm)

    seq = 64
    positions = torch.arange(1000, 1000 + seq)
    x = torch.randn(2, 4, seq, cfg.head_dim, dtype=torch.float32)

    cos_r, sin_r = reference_rotary(ref_cfg, positions, torch.float32)
    expected = reference_apply_partial_rope(x, cos_r, sin_r)

    # The model applies rotate_half over the full head in the permuted layout.
    cos, sin = BailingMoeV2RotaryEmbedding(cfg)(
        positions, device=torch.device("cpu"), dtype=torch.float32
    )
    xp = x[..., perm]
    half = cfg.head_dim // 2
    rotated = torch.cat([-xp[..., half:], xp[..., :half]], dim=-1)
    actual = (xp * cos + rotated * sin)[..., inv_perm]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    tokens = 128
    hidden = torch.randn(tokens, cfg.hidden_size)
    router = torch.randn(cfg.num_experts, cfg.hidden_size) * 0.02
    bias = torch.randn(cfg.num_experts) * 0.01
    ref_idx, ref_w = reference_gate(ref_cfg, hidden, router, bias)
    idx, weights, dense = bailing_route(
        hidden,
        router,
        bias,
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
        top_k=cfg.num_experts_per_tok,
        routed_scaling_factor=cfg.routed_scaling_factor,
        norm_topk_prob=cfg.norm_topk_prob,
    )
    ref_dense = torch.zeros(tokens, cfg.num_experts).scatter_(1, ref_idx, ref_w)
    assert torch.equal(idx.sort(dim=-1).values, ref_idx.sort(dim=-1).values)
    torch.testing.assert_close(dense, ref_dense, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.full((tokens,), cfg.routed_scaling_factor),
        rtol=1e-5,
        atol=1e-5,
    )


def _run_full_model_load(model_load, world_size: int) -> list:
    executor = MPExecutor(
        world_size=world_size,
        model_load=model_load,
        device="cpu",
        ep_degree=utils.EP_DEGREE,
        load_timeout=3600,
    )
    executor.dispatch(torch.zeros(1))
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == world_size
    return outputs


def _assert_full_model_spot_values(checkpoint: str, outputs: list, world_size: int):
    """Compare the spot values of each rank with the checkpoint tensors."""
    p1 = f"model.layers.{utils.MOE_LAYER}"
    experts_per_rank = 256 // utils.EP_DEGREE
    vocab_per_rank = utils.VOCAB_SIZE // world_size
    perm = _independent_rope_perm()
    counts = set()
    for rank, out in enumerate(outputs):
        nan_params, count, qkv, router, gate_up0, down0, lm_head4, dense_gate4 = out[:8]
        assert nan_params.item() == 0, f"rank {rank} has parameters that did not load"
        counts.add(count.item())

        expert = rank * experts_per_rank
        ref = load_reference_tensors(
            checkpoint,
            [
                f"{p1}.attention.query_key_value.weight",
                f"{p1}.mlp.gate.weight",
                f"{p1}.mlp.experts.{expert}.gate_proj.weight",
                f"{p1}.mlp.experts.{expert}.up_proj.weight",
                f"{p1}.mlp.experts.{expert}.down_proj.weight",
                "lm_head.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
            torch.bfloat16,
        )
        w = ref[f"{p1}.attention.query_key_value.weight"].view(24, 128, -1)
        expected_qkv = torch.cat(
            [w[2 * rank][perm], w[2 * rank + 1][perm], w[16 + rank // 2][perm]]
            + [w[20 + rank // 2]],
            dim=0,
        ).T
        assert torch.equal(qkv, expected_qkv), f"qkv of rank {rank}"

        assert router.dtype == torch.float32
        assert torch.equal(router, ref[f"{p1}.mlp.gate.weight"])

        expected_gate_up = torch.cat(
            [
                ref[f"{p1}.mlp.experts.{expert}.gate_proj.weight"].T,
                ref[f"{p1}.mlp.experts.{expert}.up_proj.weight"].T,
            ],
            dim=-1,
        )
        assert torch.equal(gate_up0, expected_gate_up), f"gate_up of rank {rank}"
        assert torch.equal(
            down0, ref[f"{p1}.mlp.experts.{expert}.down_proj.weight"].T
        ), f"down of rank {rank}"

        start = rank * vocab_per_rank
        assert torch.equal(lm_head4, ref["lm_head.weight"][start : start + 4])

        dense_shard = 5120 // world_size
        assert torch.equal(
            dense_gate4,
            ref["model.layers.0.mlp.gate_proj.weight"][
                rank * dense_shard : rank * dense_shard + 4
            ].T,
        )
    assert len(counts) == 1, f"ranks hold different parameter counts: {counts}"


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_full_model_load_weights(checkpoint, world_size):
    """Load all 20 layers and all 256 experts through ``load_weights`` on 8 ranks."""
    outputs = _run_full_model_load(
        partial(utils.full_model_load, checkpoint=checkpoint, device="cpu"),
        world_size,
    )
    _assert_full_model_spot_values(checkpoint, outputs, world_size)


def test_weight_loaders_in_meta_device_context():
    """Make the RoPE loaders in a meta context, as the runner makes the model.

    An earlier port revision: the logit server gave NaN for all 157184 logits of
    each step. The full-model device tests passed. The runner builds the model
    in a ``torch.device("meta")`` context, and the tests build it on CPU. In
    that context the RoPE head permutation was a meta tensor. A CPU weight
    that a meta index selects is memory with no values, and torch gives no
    error. Thus each Q head, each K head and each QK-norm gamma had no values.
    """
    nq, nkv, hd, hidden = 16, 4, 128, 64
    perm = _independent_rope_perm()
    with torch.device("meta"):
        meta_context_perm = rope_head_permutation(hd, 64)
        qkv_loader = bailing_qkv_weight_loader(
            nq, nkv, nq // 8, 1, 8 // nkv, hd, meta_context_perm, 8
        )
        gamma_loader = rope_permuted_vector_loader(meta_context_perm)
    assert meta_context_perm.device.type == "cpu"
    assert torch.equal(meta_context_perm, perm)

    w_qkv = torch.randn((nq + 2 * nkv) * hd, hidden)
    q_heads = w_qkv[: nq * hd].view(nq, hd, hidden)
    k_heads = w_qkv[nq * hd : (nq + nkv) * hd].view(nkv, hd, hidden)
    v_heads = w_qkv[(nq + nkv) * hd :].view(nkv, hd, hidden)
    for rank in range(8):
        expected = torch.cat(
            [
                q_heads[2 * rank][perm],
                q_heads[2 * rank + 1][perm],
                k_heads[rank // 2][perm],
                v_heads[rank // 2],
            ],
            dim=0,
        ).T
        actual = qkv_loader.load([FakeSafeSlice(w_qkv)], rank)
        assert actual.device.type == "cpu"
        assert torch.equal(actual, expected), f"qkv shard of rank {rank}"

    gamma = torch.randn(hd)
    assert torch.equal(gamma_loader.load([FakeSafeSlice(gamma)], 3), gamma[perm])

    # A meta permutation gives an error, not a weight with no values.
    meta_perm = perm.to("meta")
    with pytest.raises(ValueError):
        bailing_qkv_weight_loader(nq, nkv, nq // 8, 1, 8 // nkv, hd, meta_perm, 8)
    with pytest.raises(ValueError):
        rope_permuted_vector_loader(meta_perm)


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_full_model_load_weights_meta_init(checkpoint, world_size):
    """Load the full model with the meta construction of the runner on 8 ranks.

    The assertions of ``test_full_model_load_weights`` apply. The first spot
    value counts the non-finite parameters. The test also compares the
    QK-norm gammas of the MoE layer with the checkpoint.
    """
    outputs = _run_full_model_load(
        partial(utils.full_model_meta_load, checkpoint=checkpoint, device="cpu"),
        world_size,
    )
    _assert_full_model_spot_values(checkpoint, outputs, world_size)

    p1 = f"model.layers.{utils.MOE_LAYER}.attention"
    ref = load_reference_tensors(
        checkpoint,
        [f"{p1}.query_layernorm.weight", f"{p1}.key_layernorm.weight"],
        torch.bfloat16,
    )
    perm = _independent_rope_perm()
    for rank, out in enumerate(outputs):
        q_gamma, k_gamma = out[8:]
        assert torch.equal(q_gamma, ref[f"{p1}.query_layernorm.weight"][perm]), (
            f"query_layernorm of rank {rank}"
        )
        assert torch.equal(k_gamma, ref[f"{p1}.key_layernorm.weight"][perm]), (
            f"key_layernorm of rank {rank}"
        )


class _StampedGroup:
    """Minimal TP group with the stamped rank tensor of MPExecutor."""

    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.rank_in_group_tensor = torch.tensor([rank], dtype=torch.int32)
        self.world_size = world_size
        self.unique_name = "bailing_test_tp"


class _EchoBlock(torch.nn.Module):
    """Return the rows that the harness gives to the block."""

    def forward(self, hidden_states, is_prefill, positions):
        del is_prefill, positions
        return hidden_states * 2


class _EchoAttention(torch.nn.Module):
    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        del positions, position_embeddings, attn_metadata
        return hidden_states * 2


class _EchoRotary(torch.nn.Module):
    def forward(self, positions, device, dtype):
        table = positions.to(device=device, dtype=dtype)[:, None]
        return table, table


def test_component_harness_prefill_graph_is_rank_invariant():
    """Trace the prefill SP shard of the component harnesses on 4 ranks.

    Earlier port revisions: the MoE case with identical pad rows
    and pads at position 0 stopped in an inter-node collective. Seven of the
    8 workers did not stop after SIGTERM. The harness selected the SP rows
    with a Python int rank. Thus each rank traced a different graph and ran a
    different NEFF around the same collectives. The runner and the full-model
    tests select the rows with the stamped rank tensor and share one NEFF.

    This test asserts that each rank traces the same graph, and that each
    rank gets its own SP rows.
    """
    world_size, tokens, hidden = 4, 16, 8
    shard = tokens // world_size
    x = torch.randn(tokens, hidden)
    positions = torch.arange(tokens, dtype=torch.int32)

    def make_moe(group):
        return utils.MoEHarness(_EchoBlock(), group), (x, positions, True)

    def make_attention(group):
        harness = utils.AttentionHarness(_EchoAttention(), _EchoRotary(), group)
        return harness, (x, positions, {}, True)

    for name, make in (("moe", make_moe), ("attention", make_attention)):
        graphs = []

        def capture(gm, example_inputs):
            del example_inputs
            graphs.append(gm.code)
            return gm.forward

        for rank in range(world_size):
            torch._dynamo.reset()
            harness, args = make(_StampedGroup(rank, world_size))
            out = torch.compile(harness, fullgraph=True, backend=capture)(*args)
            torch.testing.assert_close(
                out, x[rank * shard : (rank + 1) * shard] * 2, rtol=0, atol=0
            )
        torch._dynamo.reset()
        assert len(graphs) == world_size, f"{name}: {len(graphs)} graphs traced"
        assert all(code == graphs[0] for code in graphs), (
            f"The {name} harness traces a different prefill graph on each rank. "
            "Select the SP rows with the stamped rank tensor (utils.sp_shard)."
        )
