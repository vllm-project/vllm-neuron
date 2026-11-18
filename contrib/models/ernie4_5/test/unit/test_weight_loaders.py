# SPDX-License-Identifier: Apache-2.0
"""Weight loading tests for ERNIE 4.5 dense (CPU).

ERNIE 4.5 uses interleaved RoPE. The port reorders the Q/K output rows of each
head at load time, so that the rotate-half RoPE of the Neuron kernels gives the
same result. These tests check:

1. The reordered QKV weight and the rotate-half RoPE of ``NF.qkv_proj`` give
   the HF interleaved RoPE result. The reference is the HF function
   ``apply_rotary_pos_emb`` of ``transformers.models.ernie4_5``.
2. The fused QKV loader gives the correct heads and the correct row order on
   rank 5 of TP 8 (2 Q heads and 1 replicated KV head on each rank).
3. ``Ernie4_5ForCausalLM.load_weights`` loads the full pinned checkpoint (all
   18 layers, BF16). Each of the 164 checkpoint tensors has a use, each
   parameter gets a value, and sampled parameters are equal to the checkpoint
   tensors after an independent transform.

``test/conftest.py`` forces ``VLLM_NEURON_BACKEND=neuron_native`` (the only
route this implementation supports) before pytest imports this module. The
helper modules come from ``test/_local_src.py``, by file location.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/ernie4_5/test/unit -v
"""

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from vllm_neuron import envs


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` helper by file location.

    The tests must keep importing after the folder is copied out of the
    repository, so they resolve ``src/`` and the sibling test helpers from
    their own paths instead of through a ``contrib.models.ernie4_5...``
    namespace path that needs the repository root on ``sys.path``. A plain
    ``import _local_src`` would need this directory on ``sys.path`` and would
    resolve to whichever contrib model got imported first in the process. See
    ``test/_local_src.py``.
    """
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    name = "contrib_ernie4_5_local_src"
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
_checkpoint = _local_src.load_helper("checkpoint")
CHECKPOINT_FILE = _checkpoint.CHECKPOINT_FILE
CHECKPOINT_NUM_TENSORS = _checkpoint.CHECKPOINT_NUM_TENSORS
get_ernie_checkpoint = _checkpoint.get_ernie_checkpoint
load_hf_config_dict = _checkpoint.load_hf_config_dict

_src_model = _local_src.load("model")
ernie_fused_qkv_weight_loader = _src_model.ernie_fused_qkv_weight_loader

pytestmark = [pytest.mark.fast]

HIDDEN_SIZE = 1024
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2


class MockSlice:
    """Mock checkpoint slice with ``__getitem__`` and ``get_shape``."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]

    def get_shape(self):
        return self.tensor.shape


@pytest.fixture(autouse=True)
def dist_cleanup():
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _even_then_odd_rows(weight: torch.Tensor, head: int) -> torch.Tensor:
    """Return the rows of one head: the even rows first, then the odd rows.

    This is an independent statement of the reorder. It does not use the port
    helpers.
    """
    rows = weight[head * HEAD_DIM : (head + 1) * HEAD_DIM]
    return torch.cat([rows[0::2], rows[1::2]], dim=0)


@pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE, reason="Uses the CPU fallback of NF.qkv_proj"
)
def test_ernie4_5_rotary_reorder_matches_hf_interleaved_rope():
    """The reordered QKV and rotate-half RoPE give the HF interleaved RoPE result."""
    from transformers import AutoConfig
    from transformers.models.ernie4_5 import modeling_ernie4_5 as hf_ernie

    import vllm_neuron.functional as NF

    Ernie4_5Config = _local_src.load("config").Ernie4_5Config
    Ernie4_5RotaryEmbedding = _src_model.Ernie4_5RotaryEmbedding

    torch.manual_seed(42)
    checkpoint = get_ernie_checkpoint()
    hf_config = AutoConfig.from_pretrained(checkpoint)

    # One TP 8 rank: 2 Q heads, 1 KV head. FP32 removes the BF16 rounding, so
    # a small tolerance shows a layout error clearly.
    nq, nkv, tokens = 2, 1, 64
    wq = torch.randn(nq * HEAD_DIM, HIDDEN_SIZE) / HIDDEN_SIZE**0.5
    wk = torch.randn(nkv * HEAD_DIM, HIDDEN_SIZE) / HIDDEN_SIZE**0.5
    wv = torch.randn(nkv * HEAD_DIM, HIDDEN_SIZE) / HIDDEN_SIZE**0.5
    hidden = torch.randn(tokens, HIDDEN_SIZE)
    # Use small and large positions, so that the high-frequency and the
    # low-frequency pairs both rotate by a large angle.
    positions = torch.cat(
        [torch.arange(tokens // 2), 30000 + 997 * torch.arange(tokens // 2)]
    ).to(torch.int64)

    # HF reference.
    hf_rotary = hf_ernie.Ernie4_5RotaryEmbedding(config=hf_config)
    cos_hf, sin_hf = hf_rotary(hidden.unsqueeze(0), positions.unsqueeze(0))
    q_hf = (hidden @ wq.T).view(1, tokens, nq, HEAD_DIM).transpose(1, 2)
    k_hf = (hidden @ wk.T).view(1, tokens, nkv, HEAD_DIM).transpose(1, 2)
    q_hf, k_hf = hf_ernie.apply_rotary_pos_emb(q_hf, k_hf, cos_hf, sin_hf)
    v_hf = (hidden @ wv.T).view(tokens, nkv, HEAD_DIM)

    # Port path: loader -> NF.qkv_proj with fused rotate-half RoPE.
    loader = ernie_fused_qkv_weight_loader(
        q_size=nq * HEAD_DIM,
        kv_size=nkv * HEAD_DIM,
        head_dim=HEAD_DIM,
        num_shards=1,
    )
    fused = loader.load([MockSlice(wq), MockSlice(wk), MockSlice(wv)], rank=0)
    config = Ernie4_5Config.from_configs(load_hf_config_dict(checkpoint))
    cos, sin = Ernie4_5RotaryEmbedding(config)(
        positions, device=torch.device("cpu"), dtype=torch.float32
    )
    qkv = NF.qkv_proj(
        hidden=hidden.unsqueeze(0),
        qkv_weights=fused,
        bias=None,
        d_head=HEAD_DIM,
        cos_cache=cos.unsqueeze(0),
        sin_cache=sin.unsqueeze(0),
        num_q_heads=nq,
        num_kv_heads=nkv,
    ).squeeze(0)
    q, k, v = torch.split(qkv, [nq * HEAD_DIM, nkv * HEAD_DIM, nkv * HEAD_DIM], -1)

    # Bring the HF result into the reordered layout with plain indexing.
    def to_half_layout(x):  # [heads, T, Hd] -> [T, heads, Hd]
        x = torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)
        return x.transpose(0, 1)

    torch.testing.assert_close(
        q.view(tokens, nq, HEAD_DIM), to_half_layout(q_hf[0]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        k.view(tokens, nkv, HEAD_DIM), to_half_layout(k_hf[0]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(v.view(tokens, nkv, HEAD_DIM), v_hf)

    # The reorder must not change the attention scores.
    q_port = q.view(tokens, nq, HEAD_DIM)
    k_port = k.view(tokens, nkv, HEAD_DIM).expand(tokens, nq, HEAD_DIM)
    scores_port = torch.einsum("thd,shd->hts", q_port, k_port)
    scores_hf = torch.einsum("htd,hsd->hts", q_hf[0], k_hf[0].expand(nq, -1, -1))
    torch.testing.assert_close(scores_port, scores_hf, rtol=1e-4, atol=1e-3)


def test_ernie4_5_fused_qkv_loader_tp8_rank5():
    """Rank 5 of TP 8 gets Q heads 10 and 11 and KV head 1, in the reordered layout."""
    torch.manual_seed(42)
    tp, rank = 8, 5
    replicas = tp // NUM_KV_HEADS
    q_size = NUM_Q_HEADS * HEAD_DIM // tp
    kv_size = HEAD_DIM

    wq = torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)
    wk = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)
    wv = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)

    loader = ernie_fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        head_dim=HEAD_DIM,
        num_shards=tp,
        num_kv_replicas=replicas,
    )
    out = loader.load([MockSlice(wq), MockSlice(wk), MockSlice(wv)], rank=rank)
    assert out.shape == (HIDDEN_SIZE, q_size + 2 * kv_size)

    kv_head = rank // replicas
    assert kv_head == 1
    expected = torch.cat(
        [
            _even_then_odd_rows(wq, 2 * rank),
            _even_then_odd_rows(wq, 2 * rank + 1),
            _even_then_odd_rows(wk, kv_head),
            wv[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM],
        ],
        dim=0,
    ).T
    assert torch.equal(out, expected)


class _LoadReport:
    """Picklable worker callable. It returns the load report of the rank.

    ``MPExecutor.collect`` accepts only tensors. Thus the callable returns the
    report as JSON bytes in a uint8 tensor. Use ``_decode_report`` to read it.
    """

    def __init__(self, report: dict):
        self.report = report

    def __call__(self, *args):
        data = json.dumps(self.report).encode("utf-8")
        return torch.tensor(list(data), dtype=torch.uint8)


def _decode_report(tensor: torch.Tensor) -> dict:
    """Decode the uint8 tensor that ``_LoadReport`` returns."""
    return json.loads(bytes(tensor.to(torch.uint8).tolist()).decode("utf-8"))


def _full_model_load(checkpoint: str, hf_config_dict: dict) -> _LoadReport:
    """Top-level picklable model load. Load the full model and make a report."""
    from safetensors import safe_open

    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    Ernie4_5ForCausalLM = _local_src.load().Ernie4_5ForCausalLM
    get_checkpoint_mappings = _src_model.get_checkpoint_mappings

    rank = dist.get_rank()
    ws = dist.get_world_size()
    model = Ernie4_5ForCausalLM.from_configs(hf_config_dict, NeuronConfig())
    num_layers = len(model.model.layers)

    # Coverage: the mappings must use each checkpoint key and fill each
    # parameter. This uses the same mappings as load_weights.
    mappings = get_checkpoint_mappings(num_layers, model.config.tie_word_embeddings)
    coverage = SafetensorsCheckpoint(checkpoint, None).load_sharded(
        rank, ws, model, mappings, "cpu", strict=True
    )
    param_names = {name for name, _ in model.named_parameters()}

    model.load_weights(checkpoint, torch.device("cpu"), None)

    bad = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_meta:
            bad.append(f"{name}: meta")
        elif tensor.is_floating_point() and not torch.isfinite(tensor).all():
            bad.append(f"{name}: not finite")
        elif tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            bad.append(f"{name}: {tensor.dtype}")

    # Compare sampled parameters with an independent transform of the
    # checkpoint tensors (TP 1: no sharding).
    mismatches = []
    with safe_open(f"{checkpoint}/{CHECKPOINT_FILE}", framework="pt") as f:
        ckpt_keys = set(f.keys())

        def ck(key):
            return f.get_tensor(key).to(torch.bfloat16)

        for layer in sorted({0, num_layers // 2, num_layers - 1}):
            p = f"model.layers.{layer}"
            mod = model.model.layers[layer]
            wq = ck(f"{p}.self_attn.q_proj.weight")
            wk = ck(f"{p}.self_attn.k_proj.weight")
            wv = ck(f"{p}.self_attn.v_proj.weight")
            expected_qkv = torch.cat(
                [_even_then_odd_rows(wq, h) for h in range(NUM_Q_HEADS)]
                + [_even_then_odd_rows(wk, h) for h in range(NUM_KV_HEADS)]
                + [wv],
                dim=0,
            ).T
            checks = {
                f"{p}.qkv": (mod.self_attn.qkv_proj_weight, expected_qkv),
                f"{p}.o_proj": (
                    mod.self_attn.o_proj_weight,
                    ck(f"{p}.self_attn.o_proj.weight").T,
                ),
                f"{p}.gate": (
                    mod.mlp.gate_proj_weight,
                    ck(f"{p}.mlp.gate_proj.weight").T,
                ),
                f"{p}.up": (mod.mlp.up_proj_weight, ck(f"{p}.mlp.up_proj.weight").T),
                f"{p}.down": (
                    mod.mlp.down_proj_weight,
                    ck(f"{p}.mlp.down_proj.weight").T,
                ),
                f"{p}.in_norm": (
                    mod.input_layernorm.weight,
                    ck(f"{p}.input_layernorm.weight"),
                ),
                f"{p}.post_norm": (
                    mod.post_attention_layernorm.weight,
                    ck(f"{p}.post_attention_layernorm.weight"),
                ),
            }
            for name, (actual, expected) in checks.items():
                if actual.shape != expected.shape or not torch.equal(actual, expected):
                    mismatches.append(name)

        embed = ck("model.embed_tokens.weight")
        for name, actual, expected in (
            ("embed_tokens", model.model.embed_tokens.weight, embed),
            ("lm_head (tied)", model.lm_head.weight, embed),
            ("norm", model.model.norm.weight, ck("model.norm.weight")),
        ):
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                mismatches.append(name)

    return _LoadReport(
        {
            "num_layers": num_layers,
            "num_ckpt_keys": len(ckpt_keys),
            "unexpected_keys": sorted(coverage.unexpected_keys),
            "missing_keys": sorted(coverage.missing_keys),
            "unfilled_params": sorted(param_names - set(coverage.state_dict)),
            "bad_tensors": bad,
            "mismatches": mismatches,
        }
    )


@pytest.mark.parametrize("world_size", [1])
def test_ernie4_5_full_model_load(world_size: int):
    """Load the full pinned checkpoint. Check coverage, dtype and values."""
    from vllm_neuron.utils.executor import MPExecutor

    torch.manual_seed(42)
    checkpoint = get_ernie_checkpoint()
    hf_config_dict = load_hf_config_dict(checkpoint)

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            _full_model_load, checkpoint=checkpoint, hf_config_dict=hf_config_dict
        ),
        device="cpu",
        # The worker loads the full checkpoint two times on the CPU.
        load_timeout=1800,
    )
    executor.dispatch()
    reports = executor.collect(timeout=1800)

    assert len(reports) == world_size
    for rank, report_tensor in enumerate(reports):
        report = _decode_report(report_tensor)
        assert report["num_layers"] == hf_config_dict["num_hidden_layers"], rank
        assert report["num_ckpt_keys"] == CHECKPOINT_NUM_TENSORS, rank
        assert report["unexpected_keys"] == [], (
            f"rank {rank}: checkpoint keys without a use: {report['unexpected_keys']}"
        )
        assert report["missing_keys"] == [], (
            f"rank {rank}: missing checkpoint keys: {report['missing_keys']}"
        )
        assert report["unfilled_params"] == [], (
            f"rank {rank}: parameters without a value: {report['unfilled_params']}"
        )
        assert report["bad_tensors"] == [], f"rank {rank}: {report['bad_tensors']}"
        assert report["mismatches"] == [], (
            f"rank {rank}: parameters not equal to the checkpoint: "
            f"{report['mismatches']}"
        )
