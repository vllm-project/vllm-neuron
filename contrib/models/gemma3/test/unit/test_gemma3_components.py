# SPDX-License-Identifier: Apache-2.0
"""Gemma3 (rnj-1) component tests in CPU mode, against the HF Gemma3 modules.

The tests use random weights (the real checkpoint stores about 33 GB of FP32
weights), with the rnj-1 hidden size where a tensor stays small:

- RMSNorm: the ``(1 + w)`` FP32 norm matches HF, and the kernel gamma is
  ``1 + w`` in BF16.
- Embedding: the token embedding times ``sqrt(hidden_size)`` matches the HF
  scaled embedding bit for bit.
- MLP: the GELU-tanh gated MLP through ``NF.mlp`` matches HF (three-way), as
  one rank and as the sum of the 8 TP shards that the decode all-reduce adds.
- Final soft cap, decoder-layer order and the vendored KV helpers of
  ``src/compat.py``.
- Prefill attention: the segmented-prefill branch (two 512-token steps that
  read the K/V back from the paged cache, the second with 512 prior tokens)
  gives the same per-token output as the full-prefill branch over the whole
  prompt.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/gemma3/test/unit/test_gemma3_components.py -v
"""

import copy
import inspect

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way

from _local_src import src_module
from gemma3_test_utils import REAL_CONFIG, hf_config_from_dict, src_config

BF16_EPS = 2.0**-7
N_SAMPLES = 4


class _FakeTPGroup:
    """Give the TP geometry of one rank without a process group."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = None


class _MockSlice:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]

    def get_shape(self):
        return self.tensor.shape


def _patch_tp_group(monkeypatch, world_size: int, rank: int) -> None:
    monkeypatch.setattr(
        src_module("model"), "get_tp_group", lambda: _FakeTPGroup(world_size, rank)
    )


@pytest.mark.parametrize("dim", [4096, 128])
def test_rmsnorm_matches_hf(dim):
    """The Gemma3 RMSNorm (1 + w, fp32) matches HF on zero-centered weights."""
    from transformers.models.gemma3 import modeling_gemma3 as hf

    model = src_module("model")
    torch.manual_seed(42)
    weight = (0.3 * torch.randn(dim)).to(torch.bfloat16)
    eps = REAL_CONFIG["rms_norm_eps"]

    ours = model.Gemma3RMSNorm(dim, eps, torch.bfloat16)
    ours.weight.data.copy_(weight)
    ref = hf.Gemma3RMSNorm(dim, eps=eps).to(torch.bfloat16)
    ref.weight.data.copy_(weight)

    x = (torch.randn(128, dim) * 3).to(torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(ours(x), ref(x), rtol=BF16_EPS, atol=1e-6)

    # The kernel gamma is 1 + w in bf16, the same as the torch norm multiplier.
    gamma = model.gemma3_norm_gamma(weight, torch.bfloat16)
    assert tuple(gamma.shape) == (1, dim)
    torch.testing.assert_close(
        gamma.float().view(-1), (1.0 + weight.float()).to(torch.bfloat16).float()
    )


def test_embedding_scale():
    """The embedding output times sqrt(H), rounded to bf16 once, matches HF."""
    from transformers.models.gemma3 import modeling_gemma3 as hf

    cfg = src_config(hf_config_from_dict(REAL_CONFIG))
    torch.manual_seed(42)
    vocab, hidden = 1024, cfg.hidden_size
    weight = torch.randn(vocab, hidden).to(torch.bfloat16)
    ids = torch.randint(0, vocab, (64,))

    hf_embed = hf.Gemma3TextScaledWordEmbedding(
        vocab, hidden, 0, embed_scale=hidden**0.5
    ).to(torch.bfloat16)
    hf_embed.weight.data.copy_(weight)
    with torch.no_grad():
        expected = hf_embed(ids)

    # Gemma3Model keeps the bf16-rounded scale as a Python float.
    assert cfg.embed_scale == 64.0
    scale = float(torch.tensor(cfg.embed_scale, dtype=torch.bfloat16))
    got = torch.nn.functional.embedding(ids, weight) * scale
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def _mlp_config():
    cfg = src_config(hf_config_from_dict(REAL_CONFIG))
    # rnj-1 hidden size, a reduced intermediate size (2048, 256 per rank at
    # TP 8) to keep the random weights small.
    cfg.intermediate_size = 2048
    return cfg


def _load_mlp(monkeypatch, cfg, weights, world_size, rank):
    from vllm_neuron.utils.weight_loader import get_weight_loader

    model = src_module("model")
    _patch_tp_group(monkeypatch, world_size, rank)
    mlp = model.Gemma3MLP(cfg)
    state = {}
    for name, param in mlp.named_parameters():
        tensor = weights[name.removesuffix("_weight")]
        state[name] = get_weight_loader(param).load([_MockSlice(tensor)], rank)
    mlp.load_state_dict(
        {k: v.to(torch.bfloat16) for k, v in state.items()}, strict=True, assign=True
    )
    return mlp


def test_mlp_matches_hf(monkeypatch):
    """The GELU-tanh MLP through NF.mlp matches HF (three-way).

    The test runs the single-rank module, and the 8 TP shards with the sum
    that the decode all-reduce does.
    """
    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.skip("This test runs in CPU mode (VLLM_NEURON_CPU_MODE=1).")
    from transformers.models.gemma3 import modeling_gemma3 as hf

    import vllm_neuron.functional as NF

    model = src_module("model")
    cfg = _mlp_config()
    hf_cfg = hf_config_from_dict(
        {**REAL_CONFIG, "intermediate_size": cfg.intermediate_size}
    )
    torch.manual_seed(0)
    hf_mlp = hf.Gemma3MLP(hf_cfg)
    weights = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        w = getattr(hf_mlp, name).weight
        with torch.no_grad():
            w.copy_(torch.randn_like(w) * 0.02)
        weights[name] = w.detach().clone()
    hf_bf16 = copy.deepcopy(hf_mlp).to(torch.bfloat16)
    hf_fp32 = hf_mlp.float()

    tp = 8
    full = _load_mlp(monkeypatch, cfg, weights, 1, 0)
    shards = [_load_mlp(monkeypatch, cfg, weights, tp, r) for r in range(tp)]

    for tokens, is_prefill in ((128, True), (4, False)):
        base, exp, act, act_tp = [], [], [], []
        for s in range(N_SAMPLES):
            torch.manual_seed(s)
            x = (torch.rand(tokens, cfg.hidden_size) * 2 - 1).to(torch.bfloat16)
            with torch.no_grad():
                base.append(hf_fp32(x.float()))
                exp.append(hf_bf16(x))
                act.append(full(x, is_prefill=is_prefill))
                partial = [
                    NF.mlp(
                        x,
                        m.gate_proj_weight,
                        m.up_proj_weight,
                        m.down_proj_weight,
                        act_fn=model.GEMMA3_MLP_ACT_FN,
                    ).float()
                    for m in shards
                ]
                act_tp.append(torch.stack(partial).sum(0).to(torch.bfloat16))
        assert_close_three_way(base, exp, act, name=f"gemma3_mlp_tp1_T{tokens}")
        assert_close_three_way(base, exp, act_tp, name=f"gemma3_mlp_tp8_T{tokens}")


def test_final_softcap():
    """The final soft cap is tanh(x / 30) * 30 in fp32, as in HF."""
    model = src_module("model")
    torch.manual_seed(42)
    logits = (torch.randn(4, 8016) * 40).to(torch.bfloat16)
    expected = (torch.tanh(logits.float() / 30.0) * 30.0).to(torch.bfloat16)
    got = model.apply_final_logit_softcap(logits, 30.0)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert got.float().abs().max() <= 30.0
    assert model.apply_final_logit_softcap(logits, None) is logits


def test_decoder_layer_order(monkeypatch):
    """The layer has the four Gemma3 norms and applies the post norms before the add."""
    model = src_module("model")
    cfg = src_config(hf_config_from_dict(REAL_CONFIG))
    _patch_tp_group(monkeypatch, 8, 0)
    with torch.device("meta"):
        layer = model.Gemma3DecoderLayer(cfg, 0)
    names = {n for n, _ in layer.named_children()}
    assert {
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "self_attn",
        "mlp",
    } <= names
    src = inspect.getsource(model.Gemma3DecoderLayer.forward)
    assert src.index("post_attention_layernorm") < src.index("residual + hidden_states")
    assert src.rindex("post_feedforward_layernorm") < src.rindex(
        "residual + hidden_states"
    )


def test_compat_kv_helpers():
    """The vendored per-head block table and KV scatter of ``src/compat.py``."""
    compat = src_module("compat")

    table = torch.tensor([[0, 1, -1]])
    torch.testing.assert_close(
        compat.build_per_head_block_table(table, kv_heads=2),
        torch.tensor([[[0, 2, -1], [1, 3, -1]]]),
    )

    # 2 blocks of 4 slots, 2 KV heads, d_head 3, 2 batch rows of 1 token.
    nkh, d, bs = 2, 3, 4
    k_cache = torch.zeros(2, nkh, bs, d)
    v_cache = torch.zeros(2, nkh, bs, d)
    K_new = torch.arange(d * 2 * nkh, dtype=torch.float32).view(d, 2, nkh, 1)
    V_new = -torch.arange(2 * nkh * d, dtype=torch.float32).view(2, nkh, 1, d)
    slots = torch.tensor([5, 6])  # block 1, positions 1 and 2
    compat.scatter_new_kv(k_cache, v_cache, K_new, V_new, slots, bs, nkh, d)
    for b, slot in enumerate(slots.tolist()):
        for h in range(nkh):
            torch.testing.assert_close(
                k_cache[slot // bs, h, slot % bs], K_new[:, b, h, 0]
            )
            torch.testing.assert_close(
                v_cache[slot // bs, h, slot % bs], V_new[b, h, 0]
            )
    assert k_cache[0].abs().sum() == 0 and v_cache[0].abs().sum() == 0


def test_segmented_prefill_matches_full_prefill(monkeypatch):
    """The segmented-prefill branch matches the full-prefill branch.

    The device integration test prefills each 8000-8192-token prompt in one
    8192-token step through ``NF.segmented_attention`` (the ``kv_segment_size``
    branch of ``Gemma3Attention.forward_prefill``). That branch writes the new
    K/V to the paged cache and reads the whole context back through the block
    table, so it depends on the cache write, the block table and the prior
    token count, which the full-prefill branch (``NF.flash_attention`` over
    the new tokens) does not use.

    A 700-token prompt runs once through the full-prefill branch, and once in
    two 512-token segmented steps: tokens 0-511 with no prior KV, then tokens
    512-699 padded to 512 (padding slots -1) with 512 prior tokens and a
    512-token KV segment. The sequence owns a shuffled set of cache blocks.
    The attention has the per-rank geometry of rnj-1 at TP 8 (4 Q heads and
    one KV head of size 128, hidden size 4096) on one rank. In CPU mode both
    branches run the plugin's torch reference of the attention kernel; the
    NKI segmented kernel runs on device only.
    """
    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.skip("This test runs in CPU mode (VLLM_NEURON_CPU_MODE=1).")
    model = src_module("model")
    cfg = src_config(hf_config_from_dict(REAL_CONFIG))
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 1
    _patch_tp_group(monkeypatch, 1, 0)
    torch.manual_seed(42)
    attn = model.Gemma3Attention(cfg, layer_idx=0)
    hidden, d_head = cfg.hidden_size, cfg.head_dim
    q_size = cfg.num_attention_heads * d_head
    with torch.no_grad():
        attn.qkv_proj_weight.copy_(
            torch.randn(hidden, q_size + 2 * d_head) * hidden**-0.5
        )
        attn.o_proj_weight.copy_(torch.randn(q_size, hidden) * q_size**-0.5)
        attn.q_norm.weight.copy_(0.1 * torch.randn(d_head))
        attn.k_norm.weight.copy_(0.1 * torch.randn(d_head))
    rope = model.Gemma3RotaryEmbedding(cfg)

    block_size, num_blocks, prompt_len, step = 32, 32, 700, 512
    # Block 0 is the null block; the sequence owns blocks 1-32 in shuffled order.
    blocks = [1 + (5 * i) % num_blocks for i in range(num_blocks)]
    block_table = torch.tensor([blocks], dtype=torch.int32)
    x = torch.randn(prompt_len, hidden).to(torch.bfloat16)

    def slots(positions):
        return torch.tensor(
            [blocks[p // block_size] * block_size + p % block_size for p in positions]
        )

    def prefill(xs, positions, slot_mapping, prior, kv_segment_size):
        metadata = {
            "layers.0.self_attn": {
                "slot_mapping": slot_mapping,
                "block_size": block_size,
                "block_table_tensor": block_table,
                "cached_seq_len": torch.tensor([[prior]], dtype=torch.int32),
                "kv_segment_size": kv_segment_size,
            }
        }
        pos = torch.tensor(positions)
        with torch.no_grad():
            return attn.forward_prefill(
                xs, pos, rope(pos, xs.device, torch.bfloat16), metadata
            )

    def empty_cache():
        attn.k_cache = torch.zeros(
            num_blocks + 1, 1, block_size, d_head, dtype=torch.bfloat16
        )
        attn.v_cache = torch.zeros_like(attn.k_cache)

    empty_cache()
    full = prefill(x, list(range(prompt_len)), slots(range(prompt_len)), 0, 0)

    empty_cache()
    first = prefill(x[:step], list(range(step)), slots(range(step)), 0, step)
    n_rest = prompt_len - step
    pad = step - n_rest
    second = prefill(
        torch.cat([x[step:], torch.zeros(pad, hidden, dtype=x.dtype)]),
        list(range(step, prompt_len)) + [0] * pad,
        torch.cat([slots(range(step, prompt_len)), torch.full((pad,), -1)]),
        step,
        step,
    )
    segmented = torch.cat([first, second[:n_rest]])

    # The two branches round differently (bf16 scores in the full branch,
    # fp32 in the segmented one); allow 2 BF16 eps of the largest output.
    # Passing 0 instead of 512 prior tokens to the second step gives an error
    # of more than 50 times this bound.
    bound = 2 * BF16_EPS * full.float().abs().max().item()
    err = (segmented.float() - full.float()).abs()
    print(
        f"segmented vs full prefill: max err {err.max().item():.5f} "
        f"(first step {err[:step].max().item():.5f}, second step "
        f"{err[step:].max().item():.5f}), bound {bound:.5f}"
    )
    assert err.max().item() <= bound
