# SPDX-License-Identifier: Apache-2.0
"""Seed-OSS module (per-component) accuracy tests against HuggingFace.

These are the component tests that ``contrib/README.md`` describes for
``test/module/``: each one validates a single piece of the model in isolation
against a HuggingFace reference, before the full model is assembled. They run
on CPU (``VLLM_NEURON_CPU_MODE=1``), need no Neuron device and no
Seed-OSS-36B weights.

A tiny random Seed-OSS checkpoint (QKV bias, untied ``lm_head``) is written by
the test itself and goes through the real ``load_weights`` path. Then, in the
staged order of the
`module test guidelines
<../../../../../docs/design/accuracy/module_test_guidelines.md>`_
(module, then full model):

1. Prefill attention against the HuggingFace attention module.
2. Decode attention, with the in-kernel paged-cache update, against the same
   reference.
3. The MLP in both the prefill and the decode phase.
4. The assembled model's prefill logits, plus the ``[seed_oss_dispatch]``
   kernel-dispatch log lines.

Scope and deviations from the guidelines. These tests predate the guidelines'
three-way helpers: they use a two-way relative-L2 bound against an FP32
reference built from the BF16-rounded weights, not
``assert_close_three_way``'s BC / sigma-ratio gate, and they run one input per
case rather than the recommended N=10 aggregate. The weights are random, not
a real checkpoint, so the reference error that the three-way gate adapts to
would not be the 36B error profile either. They are the component cases that
the recorded CPU run covered, shipped unchanged apart from the import style;
see the model README's "Validation report".

Run with:
    VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native pytest \
        contrib/models/seed_oss/test/module/ -v
"""

import contextlib
import copy
import importlib.util
import logging
import random
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` helper by file location.

    The tests must keep importing after the folder is copied out of the
    repository, so they resolve ``src/`` from its own path instead of through
    a ``contrib.models.seed_oss.src`` namespace path that needs the repository
    root on ``sys.path``. See ``test/_local_src.py``.
    """
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    name = "contrib_seed_oss_local_src"
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

# The module name the loader registers the shipped implementation under.
# ``mock.patch`` needs it as a string, and so does the ``caplog`` logger name.
_SRC = _local_src.SRC_PACKAGE
_SRC_MODEL = f"{_SRC}.model"

pytestmark = [pytest.mark.fast]

RANDOM_SEED = 42

# Relative L2 error bound for a BF16 Neuron output against an FP32 reference.
BF16_REL_L2_TOL = 2e-2

# hidden / heads = head_dim = 128. Two KV heads give GQA with 2 Q heads per
# KV head. The checkpoint keeps QKV bias and an untied lm_head.
TINY_SEED_OSS = dict(
    vocab_size=512,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=128,
    max_position_embeddings=1024,
    rms_norm_eps=1e-6,
    attention_bias=True,
    attention_out_bias=False,
    mlp_bias=False,
    tie_word_embeddings=False,
    attention_dropout=0.0,
    residual_dropout=0.0,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
)
BLOCK_SIZE = 32
NUM_BLOCKS = 4  # 4 * 32 = 128 prior slots. The decode mask needs a multiple of 128.
PROMPT_LEN = 24


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


@contextlib.contextmanager
def _fake_tp(world_size: int, rank: int):
    """Give the model a TP group of ``world_size`` without a process group."""
    group = MagicMock()
    group.world_size = world_size
    group.rank_in_group = rank
    group.device_group = MagicMock()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(f"{_SRC_MODEL}.get_tp_group", return_value=group))
        if world_size > 1:
            stack.enter_context(
                patch("torch.distributed.is_initialized", return_value=True)
            )
            stack.enter_context(
                patch("torch.distributed.get_world_size", return_value=world_size)
            )
            stack.enter_context(patch("torch.distributed.get_rank", return_value=rank))
        yield group


@contextlib.contextmanager
def _single_rank_process_group(store_dir: str):
    """Start a single-rank gloo process group for the real checkpoint loader.

    ``load_sharded_pipelined`` reads the default store of the process group. A
    serving worker always has this group. A CPU test does not, so make one here.
    The file init method uses no network port.
    """
    import os

    import torch.distributed as dist

    if dist.is_initialized():
        yield
        return
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{os.path.join(store_dir, 'pg_store')}",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Save a tiny random Seed-OSS checkpoint in BF16. Return (dir, HF model FP32)."""
    from transformers import SeedOssConfig as HFSeedOssConfig
    from transformers import SeedOssForCausalLM as HFSeedOssForCausalLM

    torch.manual_seed(RANDOM_SEED)
    hf_cfg = HFSeedOssConfig(**TINY_SEED_OSS)
    hf_model = HFSeedOssForCausalLM(hf_cfg).eval()
    with torch.no_grad():
        for name, p in hf_model.named_parameters():
            if name.endswith("_proj.bias"):
                # HF initializes the biases to zero. Use large values so that a
                # wrong bias layout gives a large error.
                p.normal_(mean=0.0, std=0.5)
            elif "layernorm" in name or name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(mean=0.0, std=0.05)

    hf_bf16 = hf_model.to(torch.bfloat16)
    model_dir = str(tmp_path_factory.mktemp("tiny_seed_oss"))
    hf_bf16.save_pretrained(model_dir)

    # Use the BF16-rounded weights in FP32 as the reference.
    hf_ref = copy.deepcopy(hf_bf16).float().eval()
    hf_ref.config._attn_implementation = "eager"
    return model_dir, hf_ref


@pytest.fixture(scope="module")
def neuron_tiny(tiny_checkpoint, tmp_path_factory):
    """Build the Neuron Seed-OSS model at TP1 and load the tiny checkpoint."""
    from transformers import AutoConfig

    Factory = _local_src.load("factory").SeedOssForCausalLM
    Impl = _local_src.load("model").SeedOssForCausalLM

    model_dir, _ = tiny_checkpoint
    hf_cfg = AutoConfig.from_pretrained(model_dir)
    store_dir = str(tmp_path_factory.mktemp("seed_oss_pg"))
    with _fake_tp(1, 0):
        with torch.device("meta"):
            model = Factory.from_configs(hf_cfg, None)
        assert type(model) is Impl
        with _single_rank_process_group(store_dir):
            model.load_weights(model_dir, torch.device("cpu"), None)
    for name, p in model.named_parameters():
        assert p.device.type == "cpu", f"{name} is still on {p.device}"
        assert torch.isfinite(p.float()).all(), f"{name} has non-finite values"
    return model


def _metadata(num_layers, query_len, slot_mapping, decode=False):
    entry = {
        "max_query_len": query_len,
        "decode_token_threshold": 1,
        "slot_mapping": slot_mapping,
        "block_size": BLOCK_SIZE,
        "block_table_tensor": torch.arange(NUM_BLOCKS, dtype=torch.int32).view(1, -1),
        "cached_seq_len": None,
        "kv_segment_size": None,
    }
    if decode:
        entry["max_blocks_per_seq"] = NUM_BLOCKS
    return {f"layers.{i}.self_attn": dict(entry) for i in range(num_layers)}


def _fresh_caches(model):
    caches = {}
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        shape = (
            NUM_BLOCKS,
            attn.num_key_value_heads_per_rank,
            BLOCK_SIZE,
            attn.head_dim,
        )
        caches[f"layers.{i}.self_attn"] = [
            torch.zeros(shape, dtype=torch.bfloat16),
            torch.zeros(shape, dtype=torch.bfloat16),
        ]
    model.bind_kv_cache(caches)


def _hf_causal_mask(T):
    return (
        torch.triu(torch.full((T, T), torch.finfo(torch.float32).min), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(0)
    )


def _hf_attention(hf_ref, layer_idx, h):
    T = h.shape[0]
    pos = torch.arange(T).unsqueeze(0)
    x = h.float().unsqueeze(0)
    pe = hf_ref.model.rotary_emb(x, pos)
    out, _ = hf_ref.model.layers[layer_idx].self_attn(
        x, position_embeddings=pe, attention_mask=_hf_causal_mask(T)
    )
    return out[0]


class TestSeedOssTinyNumerics:
    def test_attention_prefill_matches_hf(self, tiny_checkpoint, neuron_tiny):
        _, hf_ref = tiny_checkpoint
        model = neuron_tiny
        _fresh_caches(model)
        attn = model.model.layers[0].self_attn
        T = PROMPT_LEN
        h = torch.randn(T, TINY_SEED_OSS["hidden_size"], dtype=torch.bfloat16)
        positions = torch.arange(T, dtype=torch.int32)
        cos, sin = model.model.rotary_emb(
            positions, torch.device("cpu"), torch.bfloat16
        )
        with _fake_tp(1, 0), torch.no_grad():
            out = attn(
                h,
                positions,
                (cos, sin),
                _metadata(2, T, torch.arange(T, dtype=torch.int64)),
            )
            ref = _hf_attention(hf_ref, 0, h)
        err = _rel_l2(out, ref)
        assert err < BF16_REL_L2_TOL, f"prefill attention rel L2 error {err:.4f}"

        # The bias must matter. Without it the output must differ.
        with torch.no_grad():
            bias = attn.qkv_proj_bias.detach().clone()
            attn.qkv_proj_bias.zero_()
            _fresh_caches(model)
            with _fake_tp(1, 0):
                no_bias = attn(
                    h,
                    positions,
                    (cos, sin),
                    _metadata(2, T, torch.arange(T, dtype=torch.int64)),
                )
            attn.qkv_proj_bias.copy_(bias)
        assert _rel_l2(no_bias, ref) > 5 * BF16_REL_L2_TOL

    def test_attention_decode_matches_hf(self, tiny_checkpoint, neuron_tiny):
        """Prefill T tokens, then decode token T with the in-kernel cache update."""
        _, hf_ref = tiny_checkpoint
        model = neuron_tiny
        _fresh_caches(model)
        attn = model.model.layers[0].self_attn
        T = PROMPT_LEN
        h = torch.randn(T + 1, TINY_SEED_OSS["hidden_size"], dtype=torch.bfloat16)
        all_pos = torch.arange(T + 1, dtype=torch.int32)
        cos, sin = model.model.rotary_emb(all_pos, torch.device("cpu"), torch.bfloat16)
        with _fake_tp(1, 0), torch.no_grad():
            attn(
                h[:T],
                all_pos[:T],
                (cos[:T], sin[:T]),
                _metadata(2, T, torch.arange(T, dtype=torch.int64)),
            )
            dec = attn(
                h[T:],
                all_pos[T:],
                (cos[T:], sin[T:]),
                _metadata(2, 1, torch.tensor([T], dtype=torch.int64), decode=True),
            )
            ref = _hf_attention(hf_ref, 0, h)[T:]
        dec = dec.reshape(1, -1)
        err = _rel_l2(dec, ref)
        assert err < BF16_REL_L2_TOL, f"decode attention rel L2 error {err:.4f}"

        # The decode path writes the new K/V token to slot T of the cache.
        assert attn.k_cache[0, :, T].abs().sum() > 0
        assert attn.v_cache[0, :, T].abs().sum() > 0

    @pytest.mark.parametrize("is_prefill", [True, False], ids=["prefill", "decode"])
    def test_mlp_matches_hf(self, tiny_checkpoint, neuron_tiny, is_prefill):
        _, hf_ref = tiny_checkpoint
        mlp = neuron_tiny.model.layers[1].mlp
        T = PROMPT_LEN if is_prefill else 4
        h = torch.randn(T, TINY_SEED_OSS["hidden_size"], dtype=torch.bfloat16)
        with _fake_tp(1, 0), torch.no_grad():
            out = mlp(h, is_prefill=is_prefill)
            ref = hf_ref.model.layers[1].mlp(h.float())
        err = _rel_l2(out, ref)
        assert err < BF16_REL_L2_TOL, f"MLP rel L2 error {err:.4f}"

    def test_full_model_prefill_logits_match_hf(
        self, tiny_checkpoint, neuron_tiny, caplog
    ):
        """Compare all prefill logits, and check the dispatch log lines."""
        _, hf_ref = tiny_checkpoint
        model = neuron_tiny
        _fresh_caches(model)
        T = PROMPT_LEN
        input_ids = torch.randint(3, TINY_SEED_OSS["vocab_size"], (T,))
        positions = torch.arange(T, dtype=torch.int64)
        with (
            _fake_tp(1, 0),
            torch.no_grad(),
            caplog.at_level(logging.INFO, logger=_SRC_MODEL),
        ):
            logits = model(
                input_ids=input_ids,
                positions=positions,
                attn_metadata=_metadata(2, T, torch.arange(T, dtype=torch.int64)),
                sampling_positions=torch.arange(T),
            )
            ref = hf_ref(input_ids.unsqueeze(0)).logits[0]
        assert tuple(logits.shape) == (T, TINY_SEED_OSS["vocab_size"])
        err = _rel_l2(logits, ref)
        assert err < BF16_REL_L2_TOL, f"full-model logits rel L2 error {err:.4f}"
        agree = (logits.float().argmax(-1) == ref.argmax(-1)).float().mean().item()
        logging.getLogger(__name__).info(
            "Tiny Seed-OSS prefill: logits rel L2 %.5f, argmax agreement %.3f",
            err,
            agree,
        )

        # The dispatch log names each prefill op for the first and the last layer.
        pattern = re.compile(
            r"\[seed_oss_dispatch\] phase=(\w+) op=(\w+) kernel=(\w+) "
            r"layer=(\d+) rank=(-?\d+) world=(-?\d+)"
        )
        seen = set()
        for rec in caplog.records:
            m = pattern.search(rec.getMessage())
            if m:
                seen.add((m.group(1), m.group(2), int(m.group(4)), int(m.group(5))))
        for layer in (0, 1):
            for op in ("qkv_proj", "flash_attention", "o_proj", "mlp"):
                assert ("prefill", op, layer, 0) in seen, (
                    f"No dispatch line for prefill {op} layer {layer}. Seen: {seen}"
                )
