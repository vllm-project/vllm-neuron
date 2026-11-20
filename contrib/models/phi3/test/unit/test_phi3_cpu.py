# SPDX-License-Identifier: Apache-2.0
"""Phi3 (Phi-4-mini-instruct) loading and CPU component tests.

These tests run in CPU mode on one node with TP=4 (4 MPExecutor ranks). They
use the full checkpoint, all 32 layers and the original BF16 weights.

Coverage:

1. The registration contract (the architecture string of the checkpoint and
   the factory that ``src/`` exports), config parsing and factory validation.
2. Fused QKV and gate/up weight transforms against manual slices.
3. Full weight coverage: strict load, shapes, dtype, finite values, no meta
   tensors, and the tied LM head.
4. Partial LongRoPE against HF ``Phi3RotaryEmbedding`` (short factor and long
   factor), and the Q.K invariance of the kernel head permutation.
5. Attention prefill and decode against HF ``Phi3Attention`` (three-way).
6. Full-model prefill and decode logits against the HF full model (three-way).

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/phi3/test/unit/test_phi3_cpu.py \
        -v --timeout=3600 --tb=short
"""

import functools
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way


def _load_local_src():
    """Return ``test/_local_src.py``, loaded by file location.

    A plain ``import _local_src`` would resolve to whichever contrib model
    pytest collected first, so load the helper under a name that is unique to
    this model. Nothing goes on ``sys.path``, so this test file still imports
    the model's own code after the folder is copied out of the repository.
    """
    name = "contrib_phi3_local_src"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


U = _load_local_src().load_test_utils()

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE,
    reason="Phi3 CPU tests need VLLM_NEURON_CPU_MODE=1.",
)

ARCHITECTURE = "Phi3ForCausalLM"

# HF partial RoPE pair (i, i + 48) must go to kernel pair (j, j + 64).
_EXPECTED_PERMUTATION = (
    list(range(0, 48))
    + list(range(96, 112))
    + list(range(48, 96))
    + list(range(112, 128))
)


@pytest.fixture(scope="module")
def checkpoint() -> str:
    return U.resolve_checkpoint()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


# -----------------------------------------------------------------------------
# 1. Registry, config and factory
# -----------------------------------------------------------------------------


def test_phi3_registry_config_factory(checkpoint):
    import json
    import os

    from transformers import AutoConfig

    Phi3ForCausalLM = U.src_package().Phi3ForCausalLM
    Phi3Config = U.src_module("config").Phi3Config

    config_path = os.path.join(checkpoint, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    # A contrib model is deliberately absent from
    # ``vllm_neuron.model.registry.get_models()``. The user registers it
    # (contrib/README.md, "How do I register a custom model with vLLM?"), so
    # what this test can check is that the architecture string of the
    # checkpoint reaches this factory through the package export that
    # ``ModelRegistry.register_model`` is given.
    assert ARCHITECTURE in raw["architectures"]
    assert getattr(U.src_package(), ARCHITECTURE) is Phi3ForCausalLM, (
        f"src/__init__.py must export {ARCHITECTURE}."
    )

    from_json = Phi3Config.from_configs(config_path, None)
    from_hf = Phi3Config.from_configs(AutoConfig.from_pretrained(checkpoint), None)

    for cfg in (from_json, from_hf):
        assert cfg.num_hidden_layers == 32
        assert cfg.head_dim == 128
        assert cfg.rotary_dim == 96
        assert cfg.rope_type == "longrope"
        assert cfg.original_max_position_embeddings == 4096
        assert cfg.max_position_embeddings == 131072
        assert len(cfg.rope_short_factor) == 48
        assert len(cfg.rope_long_factor) == 48
        assert cfg.tie_word_embeddings is True
        assert cfg.torch_dtype == torch.bfloat16
    # Both config sources must give the same RoPE settings.
    for field in (
        "rope_theta",
        "partial_rotary_factor",
        "rope_short_factor",
        "rope_long_factor",
        "rope_attention_factor",
        "original_max_position_embeddings",
    ):
        assert getattr(from_json, field) == getattr(from_hf, field), field
    assert from_json.rope_short_factor == list(raw["rope_scaling"]["short_factor"])

    # The factory accepts the real config and refuses unsupported settings.
    Phi3ForCausalLM._validate_config(raw, None)
    for override in (
        {"torch_dtype": "float32"},
        {"sliding_window": 2048},
        {"rope_scaling": {**raw["rope_scaling"], "short_factor": [1.0] * 10}},
        {
            "rope_scaling": {
                **raw["rope_scaling"],
                "type": "dynamic",
                "rope_type": "dynamic",
            }
        },
    ):
        with pytest.raises(ValueError):
            Phi3ForCausalLM._validate_config({**raw, **override}, None)


# -----------------------------------------------------------------------------
# 2. Fused weight transforms
# -----------------------------------------------------------------------------


def test_phi3_fused_weight_transforms():
    model_module = U.src_module("model")
    phi3_gate_up_weight_loader = model_module.phi3_gate_up_weight_loader
    phi3_qkv_weight_loader = model_module.phi3_qkv_weight_loader
    rope_head_permutation = model_module.rope_head_permutation

    assert rope_head_permutation(128, 96) == _EXPECTED_PERMUTATION

    tp, nq, nkv, dh, hidden, inter = 4, 24, 8, 128, 16, 64
    nq_r, nkv_r = nq // tp, nkv // tp
    qkv = torch.randn((nq + 2 * nkv) * dh, hidden)
    q_all = qkv[: nq * dh].view(nq, dh, hidden)
    k_all = qkv[nq * dh : (nq + nkv) * dh].view(nkv, dh, hidden)
    v_all = qkv[(nq + nkv) * dh :].view(nkv, dh, hidden)
    perm = torch.tensor(_EXPECTED_PERMUTATION)

    loader = phi3_qkv_weight_loader(nq_r, nkv_r, nq, nkv, dh, 1, _EXPECTED_PERMUTATION)
    for rank in range(tp):
        got = loader.load([qkv], rank)
        q = q_all[rank * nq_r : (rank + 1) * nq_r][:, perm].reshape(-1, hidden)
        k = k_all[rank * nkv_r : (rank + 1) * nkv_r][:, perm].reshape(-1, hidden)
        v = v_all[rank * nkv_r : (rank + 1) * nkv_r].reshape(-1, hidden)
        expected = torch.cat((q, k, v), dim=0).t()
        assert got.shape == (hidden, (nq_r + 2 * nkv_r) * dh)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    gate_up = torch.randn(2 * inter, hidden)
    gate_loader = phi3_gate_up_weight_loader(inter, inter // tp, part=0)
    up_loader = phi3_gate_up_weight_loader(inter, inter // tp, part=1)
    ip = inter // tp
    for rank in range(tp):
        torch.testing.assert_close(
            gate_loader.load([gate_up], rank),
            gate_up[rank * ip : (rank + 1) * ip].t(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            up_loader.load([gate_up], rank),
            gate_up[inter + rank * ip : inter + (rank + 1) * ip].t(),
            rtol=0,
            atol=0,
        )


# -----------------------------------------------------------------------------
# 3. Full weight coverage
# -----------------------------------------------------------------------------


def test_phi3_full_weight_coverage(checkpoint):
    executor = U.make_executor(
        functools.partial(U.load_weight_coverage, checkpoint), load_timeout=1800.0
    )
    executor.dispatch(torch.zeros(1))
    stats = executor.collect(timeout=U.collect_timeout())

    # 32 layers x 7 parameters, plus embedding, final norm and LM head.
    expected_params = 32 * 7 + 3
    assert sorted(int(s[4]) for s in stats) == list(range(U.WORLD_SIZE))
    for s in stats:
        assert int(s[0]) == expected_params, s
        assert int(s[1]) == expected_params, s
        assert int(s[2]) == 32, s
        assert int(s[3]) == 1, s


# -----------------------------------------------------------------------------
# 4. Partial LongRoPE
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start,length",
    [
        pytest.param(0, 1536, id="short-serving-range"),
        pytest.param(0, 4096, id="short-boundary"),
        pytest.param(0, 4097, id="long-boundary"),
        pytest.param(8192, 64, id="long"),
    ],
)
def test_phi3_partial_longrope(checkpoint, start, length):
    from transformers.models.phi3.modeling_phi3 import (
        Phi3RotaryEmbedding as HFRotary,
        apply_rotary_pos_emb as hf_apply_rope,
    )

    model_module = U.src_module("model")
    Phi3RotaryEmbedding = model_module.Phi3RotaryEmbedding
    apply_rotary_pos_emb = model_module.apply_rotary_pos_emb

    positions = torch.arange(start, start + length, dtype=torch.int64)
    port = Phi3RotaryEmbedding(U._port_config(checkpoint))
    cos, sin = port(positions.to(torch.int32), torch.device("cpu"), torch.float32)
    hf_cos, hf_sin = HFRotary(U.hf_config(checkpoint))(
        torch.zeros(1, dtype=torch.float32), positions.unsqueeze(0)
    )
    hf_cos, hf_sin = hf_cos[0], hf_sin[0]  # [T, 96]

    # fp32 phase error grows with the position. 2e-3 covers positions < 9000.
    tol = dict(rtol=0, atol=2e-3)
    for half in (0, 64):
        torch.testing.assert_close(cos[:, half : half + 48], hf_cos[:, :48], **tol)
        torch.testing.assert_close(sin[:, half : half + 48], hf_sin[:, :48], **tol)
        torch.testing.assert_close(
            cos[:, half + 48 : half + 64], torch.ones(length, 16), rtol=0, atol=0
        )
        torch.testing.assert_close(
            sin[:, half + 48 : half + 64], torch.zeros(length, 16), rtol=0, atol=0
        )

    # Q.K must not change when the port permutes the head dimensions.
    rows = min(length, 64)
    q = torch.randn(2, rows, 128)
    k = torch.randn(2, rows, 128)
    q_hf, k_hf = hf_apply_rope(
        q[None], k[None], hf_cos[None, -rows:], hf_sin[None, -rows:]
    )
    perm = torch.tensor(_EXPECTED_PERMUTATION)
    q_p, k_p = apply_rotary_pos_emb(
        q[..., perm], k[..., perm], cos[-rows:], sin[-rows:]
    )
    torch.testing.assert_close(
        q_p @ k_p.transpose(-1, -2),
        q_hf[0] @ k_hf[0].transpose(-1, -2),
        rtol=1e-3,
        atol=5e-2,
    )


# -----------------------------------------------------------------------------
# 5. Attention prefill and decode (CPU fallback paths)
# -----------------------------------------------------------------------------


def test_phi3_attention_cpu(checkpoint):
    U.check_attention(
        checkpoint,
        chunk_len=64,
        num_chunks=1,
        num_decode=3,
        segmented=False,
        name="phi3/cpu/attention/tp4",
    )


# -----------------------------------------------------------------------------
# 6. Full-model logits
# -----------------------------------------------------------------------------


def test_phi3_full_model_logits_cpu(checkpoint):
    from transformers import AutoTokenizer

    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    prompt_len, num_decode = 64, 3
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prompt_ids = U.natural_prompt_ids(tokenizer, prompt_len)
    goldens = get_or_compute_goldens(
        key_config=U.golden_key(
            "full_model_logits", prompt_ids=prompt_ids, num_decode=num_decode
        ),
        compute_fn=lambda: U.hf_full_model_reference(
            checkpoint, prompt_ids, num_decode
        ),
    )
    token_ids = goldens["token_ids"]
    names = U.layer_names(32)

    executor = U.make_executor(
        functools.partial(U.load_full_model, checkpoint), load_timeout=1800.0
    )

    def run(input_ids, positions, metadata, sampling_positions):
        executor.dispatch(
            input_ids, positions, None, metadata, sampling_positions,
            None, None, None, None,
        )  # fmt: skip
        outputs = executor.collect(timeout=U.collect_timeout())
        for out in outputs[1:]:
            torch.testing.assert_close(out, outputs[0], rtol=0, atol=0)
        return outputs[0]

    prefill = run(
        token_ids[:prompt_len],
        torch.arange(prompt_len, dtype=torch.int64),
        U.prefill_metadata(names, 0, prompt_len),
        torch.arange(prompt_len, dtype=torch.int64),
    )
    vocab = goldens["fp32"].shape[-1]
    assert prefill.shape == (prompt_len, vocab)
    assert_close_three_way(
        goldens["fp32"][:prompt_len],
        goldens["bf16"][:prompt_len],
        prefill,
        name="phi3/cpu/full_model/prefill",
    )

    decoded, fp32_rows, bf16_rows = [], [], []
    for step in range(num_decode):
        pos = prompt_len + step
        logits = run(
            token_ids[pos : pos + 1],
            torch.tensor([pos], dtype=torch.int64),
            U.decode_metadata(names, pos),
            torch.tensor([0], dtype=torch.int64),
        )
        assert logits.shape == (1, vocab)
        decoded.append(logits)
        fp32_rows.append(goldens["fp32"][pos : pos + 1])
        bf16_rows.append(goldens["bf16"][pos : pos + 1])
    assert_close_three_way(
        fp32_rows, bf16_rows, decoded, name="phi3/cpu/full_model/decode"
    )
