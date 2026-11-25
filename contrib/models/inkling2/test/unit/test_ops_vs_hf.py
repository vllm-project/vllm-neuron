# SPDX-License-Identifier: Apache-2.0
"""``ops.py`` (TP 1, contiguous caches) against Hugging Face's ``Inkling2LoopForCausalLM`` on a small random
configuration with the real architecture: sliding and full layers, the 3-pass loop, a window and a relative-bias
extent shorter than the prompt, the log scaling active. Needs a transformers with ``inkling2_loop`` (5.24+); runs on
the CPU in fp32.

Checks: a whole-prompt prefill, then a prefill of all but the last 4 tokens followed by 4 one-token decode steps
through the KV and convolution caches, each against HF's logits; and the dense against the gathered MoE.
"""

import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.inkling2_loop import configuration_inkling2_loop as C
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M
except ImportError:  # pragma: no cover
    pytest.skip("transformers without inkling2_loop", allow_module_level=True)

from inkling2_neuron import ops
from inkling2_neuron.config import Inkling2Config

T = 128
TINY = dict(hidden_size=256, num_hidden_layers=18, num_attention_heads=4, num_key_value_heads=2,
            swa_num_attention_heads=4, swa_num_key_value_heads=2, head_dim=64, swa_head_dim=64, d_rel=8,
            rel_extent=64, sliding_window_size=64, moe_intermediate_size=128, n_routed_experts=8,
            num_experts_per_tok=2, vocab_size=1024, unpadded_vocab_size=1000, loop_layer_index_start=6,
            loop_layer_index_end=12, loop_num_passes=3, loop_gate_rank=16, log_scaling_n_floor=32)


def hf_model():
    cfg = C.Inkling2LoopTextConfig()
    for k, v in TINY.items():
        setattr(cfg, k, v)
    cfg.layer_types = ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(cfg.num_hidden_layers)]
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = M.Inkling2LoopForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 1 or name.endswith(("norm.weight", ".scale")):
                p.copy_(1.0 + 0.05 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
        for name, b in m.named_buffers():
            if "e_score_correction_bias" in name:
                b.copy_(0.05 * torch.randn(b.shape, generator=g))
    return m


def our_config():
    return Inkling2Config(hidden_size=256, num_layers=18, vocab_size=1024, unpadded_vocab_size=1000, num_heads=4,
                          num_kv_heads=2, head_dim=64, d_rel=8, rel_extent=64, window=64, log_floor=32,
                          num_experts=8, top_k=2, moe_inter=128, loop_start=6, loop_end=12, passes=3, gate_rank=16,
                          full_layers=(5, 11, 17), tp_size=1)


def params_from_hf(m, cfg):
    sd = {k: v.detach() for k, v in m.state_dict().items()}
    pre = "model."
    out = {"embed": sd[pre + "embed_tokens.weight"], "embed_norm": sd[pre + "embed_tokens.embed_norm.weight"],
           "norm": sd[pre + "norm.weight"], "final_query": sd[pre + "final_input.query"],
           "message_query": sd[pre + "message_input.query"], "lm_head": sd["lm_head.weight"], "layers": []}
    for L in range(cfg.num_layers):
        s = lambda n: sd[f"{pre}layers.{L}.{n}"]  # noqa: E731
        a = "self_attn."
        p = {"wq": s(a + "q_proj.linear.weight"), "q_scale": s(a + "q_proj.norm.scale"),
             "wk": s(a + "k_proj.linear.weight"), "wv": s(a + "v_proj.linear.weight"),
             "wo": s(a + "o_proj.linear.weight"), "wgate": s(a + "o_proj.gate_proj.weight"),
             "wsink": s(a + "sink_proj.weight"), "wr": s(a + "relative_position_bias.linear.weight"),
             "r_scale": s(a + "relative_position_bias.norm.scale"), "r_proj": s(a + "relative_position_bias.proj"),
             "post_norm": s("post_attention_layernorm.weight"), "mlp_query": s("mlp_input.query"),
             "post_attn_conv": s("post_attention_convolution.conv1d.weight").squeeze(1),
             "post_mlp_conv": s("post_mlp_convolution.conv1d.weight").squeeze(1),
             "router": s("mlp.gate.weight"), "router_bias": s("mlp.gate.e_score_correction_bias"),
             "gate_up": s("mlp.experts.gate_up_proj"), "down": s("mlp.experts.down_proj")}
        if L > 0:
            p["attn_query"] = s("attention_input.query")
        if cfg.is_full(L):
            p.update(k_static=s(a + "k_proj.sconv.static_weight"), k_wproj=s(a + "k_proj.sconv.weight_proj.weight"),
                     v_static=s(a + "v_proj.sconv.static_weight"), v_wproj=s(a + "v_proj.sconv.weight_proj.weight"),
                     q_norm=s("q_input_layernorm.weight"), k_norm=s("k_input_layernorm.weight"),
                     v_norm=s("v_input_layernorm.weight"))
        else:
            p.update(k_conv=s(a + "k_proj.sconv.conv1d.weight").squeeze(1),
                     v_conv=s(a + "v_proj.sconv.conv1d.weight").squeeze(1), in_norm=s("input_layernorm.weight"))
        if cfg.loop_start <= L < cfg.loop_end:
            for g in ("q", "k", "v", "mlp"):
                p[f"gate_{g}"] = tuple(s(f"{g}_message_gate.{n}.weight")
                                       for n in ("down_proj", "forget_gate_proj", "message_gate_proj"))
        out["layers"].append(p)
    return out


class ContiguousIO(ops.LayerIO):
    def __init__(self):
        self.k, self.v, self.win = {}, {}, {}

    def kv(self, c, k, v, window=None):
        if c in self.k:
            k, v = torch.cat([self.k[c], k], 2), torch.cat([self.v[c], v], 2)
        self.k[c], self.v[c] = k, v
        L = k.shape[2]
        return k, v, torch.arange(L).view(1, L).expand(k.shape[0], L)

    def window(self, c, slot):
        if (c, slot) not in self.win:
            return None
        return self.win[(c, slot)]

    def put_window(self, c, slot, seq):
        self.win[(c, slot)] = ops.next_window(seq, seq.shape[1] - ops.CONV_WINDOW)


def _zero_window_io(io, B, cfg, params):
    """Zero windows of the right width for a first step."""
    orig = io.window

    def window(c, slot):
        w = orig(c, slot)
        if w is not None:
            return w
        width = {0: cfg.kv_heads_per_rank * cfg.head_dim, 1: cfg.kv_heads_per_rank * cfg.head_dim}.get(
            slot, cfg.hidden_size)
        return torch.zeros(B, ops.CONV_WINDOW, width)
    io.window = window
    return io


def logits_of(params, cfg, ids, io, start, dense):
    B, S = ids.shape
    io.positions = (torch.arange(S) + start).view(1, S).expand(B, S)
    h = ops.rms_norm(params["embed"][ids], params["embed_norm"], cfg.eps, dtype=torch.float32)
    h = ops.text_model(params, cfg, h, io, dense)
    lg = h.float() @ params["lm_head"].float().t()
    return ops.softcap(lg, cfg.final_cap)


@pytest.fixture(scope="module")
def setup():
    m = hf_model()
    cfg = our_config()
    params = params_from_hf(m, cfg)
    ids = torch.randint(0, 1000, (1, T), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = m(input_ids=ids, use_cache=False).logits.float()
    return m, cfg, params, ids, ref


def _report(name, ours, ref):
    d = (ours - ref).abs()
    agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"{name}: max |diff| {d.max():.2e}, mean {d.mean():.2e}, argmax agree {agree:.4f}", file=sys.stderr)
    return d.max().item(), agree


def test_prefill_matches_hf(setup):
    m, cfg, params, ids, ref = setup
    io = _zero_window_io(ContiguousIO(), 1, cfg, params)
    with torch.no_grad():
        ours = logits_of(params, cfg, ids, io, 0, dense=True)
    dmax, agree = _report("prefill", ours, ref)
    assert agree == 1.0 and dmax < 1e-3


def test_decode_steps_match_hf(setup):
    m, cfg, params, ids, ref = setup
    io = _zero_window_io(ContiguousIO(), 1, cfg, params)
    n = 4
    with torch.no_grad():
        logits_of(params, cfg, ids[:, :T - n], io, 0, dense=True)
        steps = [logits_of(params, cfg, ids[:, t:t + 1], io, t, dense=False) for t in range(T - n, T)]
    ours = torch.cat(steps, dim=1)
    dmax, agree = _report("decode", ours, ref[:, T - n:])
    assert agree == 1.0 and dmax < 1e-3


def test_dense_and_gathered_moe_agree(setup):
    _, cfg, params, _, _ = setup
    p = params["layers"][3]
    x = torch.randn(16, cfg.hidden_size, generator=torch.Generator().manual_seed(3))
    idx, w = ops.route(x, p["router"], p["router_bias"], cfg.top_k, cfg.route_scale)
    a = ops.moe_dense(x, idx, w, p["gate_up"], p["down"], cfg.gate_cap, cfg.up_cap, group=4)
    b = ops.moe_gather(x, idx, w, p["gate_up"], p["down"], cfg.gate_cap, cfg.up_cap)
    assert (a - b).abs().max() < 1e-4


@pytest.mark.parametrize("B,S,start,L,E,kv0", [(1, 64, 0, 256, 32, 0), (1, 64, 100, 256, 32, 0),
                                              (3, 1, 150, 256, 32, 0), (1, 128, 900, 1024, 256, 0),
                                              (2, 1, 1023, 1024, 1024, 0), (1, 256, 5000, 1536, 1024, 3840),
                                              (2, 1, 9000, 1280, 1024, 7936)])
def test_relative_bias_shift_matches_gather(B, S, start, L, E, kv0):
    """Keys from position ``kv0`` (a sliding layer's window) as well as from 0."""
    g = torch.Generator().manual_seed(4)
    rel = torch.randn(B, 3, S, E, generator=g)
    q_start = torch.tensor([start + 7 * b for b in range(B)])
    kv_start = torch.tensor([kv0 + 7 * b for b in range(B)])
    q_pos = q_start.view(-1, 1) + torch.arange(S).view(1, -1)
    kv_pos = kv_start.view(-1, 1) + torch.arange(L).view(1, -1)
    ref = ops.relative_bias(rel, q_pos.unsqueeze(-1) - kv_pos.unsqueeze(1), E)
    ours = ops.relative_bias_shift(rel, q_start, L, E, kv_start)
    assert torch.equal(ref, ours)


@pytest.mark.parametrize("B,S,starts,nb,block,window", [(1, 512, [0], 64, 256, 1024), (1, 512, [5000], 64, 256, 1024),
                                                         (3, 1, [0, 1023, 9000], 64, 256, 1024), (2, 1, [16383, 300], 64, 256, 1024)])
def test_window_blocks_cover_the_window(B, S, starts, nb, block, window):
    g = torch.Generator().manual_seed(5)
    table = torch.stack([torch.randperm(1000, generator=g)[:nb] for _ in range(B)])
    q_start = torch.tensor(starts)
    blocks, first = ops.window_blocks(table, q_start, S, window, block)
    for b in range(B):
        for s_ in (0, S - 1):
            q = starts[b] + s_
            for pos in range(max(0, q - window + 1), q + 1):      # every key the window allows
                j = (pos - first[b].item()) // block
                assert 0 <= j < blocks.shape[1] and blocks[b, j].item() == table[b, pos // block].item()


class WindowedIO(ContiguousIO):
    """As the server reads a sliding layer: only the keys from a block-aligned start before each window."""

    def __init__(self, block=16):
        super().__init__()
        self.block = block

    def kv(self, c, k, v, window=None):
        k_all, v_all, kv_pos = super().kv(c, k, v)
        if window is None:
            return k_all, v_all, kv_pos
        q0 = int(self.positions[0, 0])
        start = max(0, q0 - window + 1) // self.block * self.block
        return k_all[:, :, start:], v_all[:, :, start:], kv_pos[:, start:]


def test_windowed_context_matches_hf(setup):
    """Prefill in 32-token chunks then 4 decode steps, the sliding layers reading only their window (the prompt is
    twice the window), against HF's whole-prompt logits."""
    m, cfg, params, ids, ref = setup
    io = _zero_window_io(WindowedIO(), 1, cfg, params)
    n, chunk = 4, 32
    out = []
    with torch.no_grad():
        for t0 in range(0, T - n, chunk):
            t1 = min(t0 + chunk, T - n)
            out.append(logits_of(params, cfg, ids[:, t0:t1], io, t0, dense=True))
        out += [logits_of(params, cfg, ids[:, t:t + 1], io, t, dense=False) for t in range(T - n, T)]
    ours = torch.cat(out, dim=1)
    dmax, agree = _report("windowed chunked prefill + decode", ours, ref)
    assert agree == 1.0 and dmax < 1e-3
