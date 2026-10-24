# SPDX-License-Identifier: Apache-2.0
"""Module tests: Hy3 FP8 KV-cache attention on one NeuronCore, non-1 scales.

Layer-0 attention of the real checkpoint as rank 0 of TP=64 (1 Q head, 1 KV
head, DCP=1), loaded through the model's weight loaders. Each case runs the
model's own ``forward_prefill`` / ``forward_decode`` on device and compares with
an FP32 reference that applies the same static activation fake-quant and an
exact FP8 round-trip of the cached K/V:

- cache bytes written by prefill (torch scatter, standard and packed K) and by
  decode (in-kernel write of the active token, packed and through the forced
  unpacked fallback + write-back);
- prefill output: full (flash) and segmented with a prefix read from the cache;
- decode output over a non-contiguous block table with odd context lengths, and
  a second decode step that must read the token written by the first;
- distinct non-1 K and V scales sized from the actual K/V range (plus scale 1,
  reported only: it puts Hy3's small V in the FP8 subnormal range).

A negative control checks the tolerances discriminate: the reference with the
V-scale compensation left out (or folded into the weight scale only, which
clips the O-projection input) must miss the device result by a wide margin.

    HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/module/test_fp8_kv_attention.py -v -s
"""

import types

import pytest
import torch
from conftest import TRN2_FP8_MAX, fake_quant, rel_err, rmsnorm
from hy_v3_neuron import model as hy_model
from hy_v3_neuron.config import HYV3Config

from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
)
from vllm_neuron.utils.weight_loader import get_weight_loader

FP8 = torch.float8_e4m3fn
TP, D, BLOCK = 64, 128, 32
LAYER = 0
A = f"model.layers.{LAYER}.self_attn"
DEVICE = "neuron:0"
# Output tolerance vs the FP32 reference (BF16 activations + static FP8 act quant).
OUT_TOL = 3e-2
# A wrong scale contract must miss by at least this much.
WRONG_MIN = 0.2


def _check(err, kind):
    """``unit`` (scale 1, i.e. a direct cast) is reported, not held to OUT_TOL:
    Hy3's V is ~0.1-0.2 in magnitude, so a direct cast lands it in the FP8
    subnormal range -- the reason the model requires a calibrated profile."""
    assert err == err, "NaN output"
    if kind != "unit":
        assert err < OUT_TOL


class _FakeGroup(types.SimpleNamespace):
    def all_reduce(self, x):
        return x

    def reduce_scatter(self, x, dim=0):
        return x

    def all_gather(self, x, dim=0):
        return x


@pytest.fixture(scope="module")
def attn(ckpt):
    """HYV3Attention as TP rank 0 of 64 (collectives are identities)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(hy_model, "get_tp_group", lambda: _FakeGroup(world_size=TP, rank_in_group=0))
    mp.setattr(hy_model, "get_dcp_group", lambda: _FakeGroup(world_size=1, rank_in_group=0))
    cfg = HYV3Config()
    a = hy_model.HYV3Attention(cfg, LAYER)
    keys = {
        "qkv_proj_weight": [f"{A}.{n}_proj.weight" for n in "qkv"],
        "qkv_weight_scale": [f"{A}.{n}_proj.weight_scale" for n in "qkv"],
        "qkv_input_scale": [f"{A}.{n}_proj.input_scale" for n in "qkv"],
        "o_proj_weight": [f"{A}.o_proj.weight"],
        "o_weight_scale": [f"{A}.o_proj.weight_scale"],
        "o_input_scale": [f"{A}.o_proj.input_scale"],
    }
    with torch.no_grad():
        for name, srcs in keys.items():
            p = getattr(a, name)
            p.data = get_weight_loader(p).transform([ckpt.slice(k) for k in srcs], 0)
        a.q_norm.weight.data = ckpt.get(f"{A}.q_norm.weight").to(a.dtype)
        a.k_norm.weight.data = ckpt.get(f"{A}.k_norm.weight").to(a.dtype)
    a.ln_w = ckpt.get(f"model.layers.{LAYER}.input_layernorm.weight").to(a.dtype)
    a.cfg = cfg
    a.requires_grad_(False)  # inference graphs only (the server runs under no_grad)
    yield a.to(DEVICE)
    mp.undo()


# ── FP32 reference ────────────────────────────────────────────────────────


class Ref:
    """FP32 reference of the rank-0 attention with the model's quantizers."""

    def __init__(self, a):
        w = a.qkv_proj_weight.detach().cpu().float()
        s = a.qkv_weight_scale.detach().cpu().float()[0]
        q_size, kv_size = a.q_size, a.kv_size
        self.wq = w[:, :q_size] * s[0]
        self.wk = w[:, q_size : q_size + kv_size] * s[1]
        self.wv = w[:, q_size + kv_size :] * s[2]
        self.qkv_in = a.qkv_input_scale.detach().cpu().float()[0, 0]
        self.wo = a.o_proj_weight.detach().cpu().float() * a.o_weight_scale.detach().cpu().float()[0, 0]
        self.o_in = a.o_input_scale.detach().cpu().float()[0, 0]
        self.o_w_scale = a.o_weight_scale.detach().cpu().float()[0, 0]
        self.o_w_fp8 = a.o_proj_weight.detach().cpu().float()
        self.qn = a.q_norm.weight.detach().cpu().float()
        self.kn = a.k_norm.weight.detach().cpu().float()
        self.eps = a.q_norm.variance_epsilon
        self.theta = a.cfg.rope_theta
        self.scaling = a.scaling

    def rope(self, x, pos):
        inv = 1.0 / (self.theta ** (torch.arange(0, D, 2).float() / D))
        f = pos.float()[:, None] * inv[None]
        emb = torch.cat((f, f), -1)
        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        return x * emb.cos() + torch.cat((-x2, x1), -1) * emb.sin()

    def qkv(self, x_normed, pos):
        """Post-norm hidden [T, H] -> q, k (QK-norm + RoPE), v, all [T, D] fp32."""
        xq = fake_quant(x_normed, self.qkv_in)
        q, k, v = xq @ self.wq, xq @ self.wk, xq @ self.wv
        q = self.rope(rmsnorm(q, self.qn, self.eps), pos)
        k = self.rope(rmsnorm(k, self.kn, self.eps), pos)
        return q, k, v

    @staticmethod
    def kv_roundtrip(x, qscale):
        return (x * qscale).clamp(-TRN2_FP8_MAX, TRN2_FP8_MAX).to(FP8).float() / qscale

    def attend(self, q, k, v, q_pos, kv_pos):
        s = (q @ k.T) * self.scaling
        s = s.masked_fill(kv_pos[None, :] > q_pos[:, None], float("-inf"))
        return torch.softmax(s, -1) @ v

    def o_proj(self, attn, in_mult=1.0, w_div=1.0):
        """STATIC O-proj; in_mult / w_div model the decode scale adjustment."""
        in_s = self.o_in * in_mult
        aq = (attn / in_s).clamp(-TRN2_FP8_MAX, TRN2_FP8_MAX).to(FP8).float()
        return (aq @ self.o_w_fp8) * (self.o_w_scale / w_div) * in_s


@pytest.fixture(scope="module")
def ref(attn):
    return Ref(attn)


def _hidden(attn, n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, attn.hidden_size, generator=g)
    return x


def _scales(ref, attn, kind):
    """(q_k, q_v) quantizing scales sized to use ~200 of the 240 FP8 range."""
    x = rmsnorm(_hidden(attn, 1024, 7), attn.ln_w.cpu().float(), attn.rms_norm_eps)
    _, k, v = ref.qkv(x, torch.arange(1024))
    qk, qv = 200.0 / k.abs().max().item(), 200.0 / v.abs().max().item()
    # k_low / v_low: distinct non-1 K and V scales (one using 1/8 of the range),
    # so a swapped or inverted scale cannot pass.
    return {
        "unit": (1.0, 1.0),
        "kv": (qk, qv),
        "k_low": (qk / 8, qv),
        "v_low": (qk, qv / 8),
    }[kind]


def _set_scales(attn, qk, qv):
    attn.set_kv_scales(qk, qv, DEVICE)  # as the model's profile loader does


# Host-side cache manipulation uses uint8 views (CPU indexing kernels do not
# all cover float8); the device caches are float8_e4m3fn.
def _alloc(num_blocks):
    """Standard-layout uint8 K, V host buffers [blocks, 1, BLOCK, D]."""
    z = torch.zeros(num_blocks, 1, BLOCK, D, dtype=torch.uint8)
    return z.clone(), z.clone()


def _codes(x_fp32, qscale):
    return (x_fp32 * qscale).clamp(-TRN2_FP8_MAX, TRN2_FP8_MAX).to(FP8).view(torch.uint8)


def _bind(attn, k_u8, v_u8, packed):
    k = _swizzle_packed_k(k_u8) if packed else k_u8
    attn.k_cache = k.contiguous().view(FP8).to(DEVICE)
    attn.v_cache = v_u8.contiguous().view(FP8).to(DEVICE)
    attn.fp8_packed = packed


def _host_caches(attn):
    """Device caches -> standard-layout uint8 K, V."""
    k = attn.k_cache.cpu().view(torch.uint8)
    if k.dim() == 5:
        k = _unswizzle_packed_k(k)
    return k, attn.v_cache.cpu().view(torch.uint8)


def _compile(fn):
    from vllm_neuron.envs import get_compile_backend_name

    compiled = torch.compile(fn, backend=get_compile_backend_name(), fullgraph=True)

    def run(*args):
        with torch.no_grad():
            return compiled(*args)

    return run


# ── prefill ───────────────────────────────────────────────────────────────


class _Prefill(torch.nn.Module):
    def __init__(self, attn, kv_segment_size):
        super().__init__()
        self.attn = attn
        self.seg = kv_segment_size

    def forward(self, x, cos, sin, slot_mapping, block_table, cached):
        meta = {
            "slot_mapping": slot_mapping,
            "block_size": BLOCK,
            "block_table_tensor": block_table,
            "cached_seq_len": cached,
            "kv_segment_size": self.seg,
        }
        return self.attn.forward_prefill(x, (cos, sin), {self.attn.layer_name: meta}, None)


def _run_prefill(attn, x_normed, pos, blocks, n_valid, seg, cached=0):
    T = x_normed.shape[0]
    slots = torch.full((T,), -1, dtype=torch.long)
    p = pos[:n_valid]
    slots[:n_valid] = blocks[p // BLOCK] * BLOCK + p % BLOCK
    cos, sin = hy_model.HYV3RotaryEmbedding(attn.cfg)(pos, "cpu", attn.dtype)
    bt = blocks.view(1, -1).to(torch.int32)
    fn = _compile(_Prefill(attn, seg))
    out = fn(
        x_normed.to(attn.dtype).to(DEVICE),
        cos.to(DEVICE),
        sin.to(DEVICE),
        slots.to(DEVICE),
        bt.to(DEVICE),
        torch.tensor([[cached]], dtype=torch.int32).to(DEVICE),
    )
    return out.cpu().float(), slots


def _cache_err(got_u8, x_ref, qscale):
    """Relative error of the dequantized cache vs the unquantized reference.

    Returns (device, reference) errors: the device cache must be about as close
    to the FP32 K/V as an exact FP8 quantization of them (its inputs come from
    BF16 / FP8 kernels, so individual codes may differ by one step). A wrong
    scale value or direction misses by orders of magnitude.
    """
    got = got_u8.contiguous().view(FP8).float() / qscale
    want = _codes(x_ref, qscale).view(FP8).float() / qscale
    return rel_err(got, x_ref), rel_err(want, x_ref)


def _assert_cache(e, tag):
    e_dev, e_ref = e
    print(f"  {tag} cache: dequant rel err {e_dev:.3e} (exact FP8 quantization {e_ref:.3e})")
    assert e_dev < max(1.5 * e_ref, e_ref + 1e-2), (tag, e_dev, e_ref)


@pytest.mark.parametrize("packed", [False, True], ids=["std", "packed"])
@pytest.mark.parametrize("kind", ["unit", "kv"])
def test_prefill_full_writes_cache(attn, ref, kind, packed):
    """Full (flash) prefill: output + quantized cache bytes, odd valid count."""
    qk, qv = _scales(ref, attn, kind)
    _set_scales(attn, qk, qv)
    T, n_valid = 512, 509
    blocks = torch.randperm(63)[: T // BLOCK] + 1  # non-contiguous, never block 0
    k, v = _alloc(64)
    _bind(attn, k, v, packed)
    x = rmsnorm(_hidden(attn, T, 1), attn.ln_w.cpu().float(), attn.rms_norm_eps)
    pos = torch.arange(T)
    out, slots = _run_prefill(attn, x, pos, blocks, n_valid, seg=0)

    q, kr, vr = ref.qkv(x, pos)
    want = ref.o_proj(ref.attend(q, kr, vr, pos, pos))
    err = rel_err(out[:n_valid], want[:n_valid])
    print(f"\n[prefill full {kind} {'packed' if packed else 'std'}] q_k={qk:.4g} q_v={qv:.4g} out rel err {err:.3e}")
    _check(err, kind)

    kc, vc = _host_caches(attn)
    s = slots[:n_valid]
    k_got, v_got = kc[s // BLOCK, 0, s % BLOCK], vc[s // BLOCK, 0, s % BLOCK]
    _assert_cache(_cache_err(k_got, kr[:n_valid], qk), "K")
    _assert_cache(_cache_err(v_got, vr[:n_valid], qv), "V")
    # Padding rows went to the null block only: no other block was touched.
    used = set(blocks.tolist())
    other = [b for b in range(64) if b not in used and b != 0]
    assert kc[other].count_nonzero() == 0
    assert vc[other].count_nonzero() == 0


@pytest.mark.parametrize("kind", ["unit", "kv"])
def test_prefill_segmented_prefix(attn, ref, kind):
    """Packed K, two 512-token chunks; chunk 2 reads chunk 1 from the FP8 cache."""
    qk, qv = _scales(ref, attn, kind)
    _set_scales(attn, qk, qv)
    T = 512
    blocks = torch.randperm(63)[: 2 * T // BLOCK] + 1
    k, v = _alloc(64)
    _bind(attn, k, v, packed=True)
    x = rmsnorm(_hidden(attn, 2 * T, 2), attn.ln_w.cpu().float(), attn.rms_norm_eps)
    pos = torch.arange(2 * T)
    _run_prefill(attn, x[:T], pos[:T], blocks, T, seg=512, cached=0)
    out2, _ = _run_prefill(attn, x[T:], pos[T:], blocks, T, seg=512, cached=T)

    q, kr, vr = ref.qkv(x, pos)
    kd, vd = ref.kv_roundtrip(kr, qk), ref.kv_roundtrip(vr, qv)
    want = ref.o_proj(ref.attend(q[T:], kd, vd, pos[T:], pos))
    err = rel_err(out2, want)
    print(f"\n[prefill segmented {kind}] q_k={qk:.4g} q_v={qv:.4g} chunk-2 rel err {err:.3e}")
    _check(err, kind)
    if kind == "kv":
        # Negative control: without the K / V dequant the result is far off.
        bad = ref.o_proj(ref.attend(q[T:], kd * qk, vd * qv, pos[T:], pos))
        assert rel_err(out2, bad) > WRONG_MIN


# ── decode ────────────────────────────────────────────────────────────────


class _Decode(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, x, positions, cos, sin, slot_mapping, block_table, ln_w):
        meta = {
            "slot_mapping": slot_mapping,
            "block_size": BLOCK,
            "block_table_tensor": block_table,
            "max_blocks_per_seq": block_table.shape[1],
        }
        return self.attn.forward_decode(
            x, positions, (cos, sin), {self.attn.layer_name: meta}, ln_w, None
        )


def _fill_history(ref, attn, lens, tables, qk, qv, packed, seed):
    """Write each sequence's history (K/V of random tokens) into the FP8 cache."""
    k, v = _alloc(256)
    hist = []
    for b, L in enumerate(lens):
        x = rmsnorm(_hidden(attn, L, seed + b), attn.ln_w.cpu().float(), attn.rms_norm_eps)
        pos = torch.arange(L)
        _, kr, vr = ref.qkv(x, pos)
        blk, off = tables[b][pos // BLOCK], pos % BLOCK
        k[blk, 0, off] = _codes(kr, qk)
        v[blk, 0, off] = _codes(vr, qv)
        hist.append((ref.kv_roundtrip(kr, qk), ref.kv_roundtrip(vr, qv)))
    _bind(attn, k, v, packed)
    return hist


def _decode_step(attn, ref, hist, lens, tables, qk, qv, seed):
    B = len(lens)
    x = _hidden(attn, B, seed)
    positions = torch.tensor(lens)
    cos, sin = hy_model.HYV3RotaryEmbedding(attn.cfg)(positions, "cpu", attn.dtype)
    slots = torch.stack([tables[b][lens[b] // BLOCK] * BLOCK + lens[b] % BLOCK for b in range(B)])
    bt = torch.stack(tables).to(torch.int32)
    out = _compile(_Decode(attn))(
        x.to(attn.dtype).to(DEVICE),
        positions.to(DEVICE),
        cos.to(DEVICE),
        sin.to(DEVICE),
        slots.to(DEVICE),
        bt.to(DEVICE),
        attn.ln_w.to(DEVICE),
    ).cpu().float()

    xn = rmsnorm(x, attn.ln_w.cpu().float(), attn.rms_norm_eps)
    wants, bads = [], []
    new_kv = []
    for b in range(B):
        q, kr, vr = ref.qkv(xn[b : b + 1], positions[b : b + 1])
        kd = torch.cat([hist[b][0], ref.kv_roundtrip(kr, qk)])
        vd = torch.cat([hist[b][1], ref.kv_roundtrip(vr, qv)])
        kv_pos = torch.arange(lens[b] + 1)
        attn_out = ref.attend(q, kd, vd, positions[b : b + 1], kv_pos)
        wants.append(ref.o_proj(attn_out))
        # Negative controls for the V contract: no compensation at all, and the
        # kernel docstring's weight-scale-only compensation (clips at 240 / q_v).
        bads.append((ref.o_proj(attn_out * qv), ref.o_proj(attn_out * qv, w_div=qv)))
        new_kv.append((kr, vr, kd, vd))
    no_comp = torch.cat([b[0] for b in bads])
    w_only = torch.cat([b[1] for b in bads])
    return out, torch.cat(wants), (no_comp, w_only), slots, new_kv


@pytest.mark.parametrize(
    "packed,force_unpacked",
    [(False, False), (True, False), (True, True)],
    ids=["std", "packed", "packed_fallback"],
)
@pytest.mark.parametrize("kind", ["unit", "k_low", "v_low", "kv"])
def test_decode_two_steps(attn, ref, kind, packed, force_unpacked):
    """Two decode steps, B=4, odd context lengths, non-contiguous block tables."""
    if kind in ("k_low", "v_low") and packed:
        pytest.skip("scale-direction cases run on the standard layout")
    qk, qv = _scales(ref, attn, kind)
    _set_scales(attn, qk, qv)
    hy_model._FORCE_UNPACKED_DECODE = force_unpacked
    try:
        lens = [1001, 517, 63, 800]
        nblk = 32  # S_ctx 1024 bucket
        perm = torch.randperm(255) + 1
        tables = [perm[b * nblk : (b + 1) * nblk] for b in range(4)]
        hist = _fill_history(ref, attn, lens, tables, qk, qv, packed, seed=100)
        for step in range(2):
            out, want, bad, slots, new_kv = _decode_step(attn, ref, hist, lens, tables, qk, qv, seed=200 + step)
            err = rel_err(out, want)
            tag = f"{kind} {'packed' if packed else 'std'}{' fallback' if force_unpacked else ''}"
            print(f"\n[decode {tag} step {step}] q_k={qk:.4g} q_v={qv:.4g} rel err {err:.3e}")
            _check(err, kind)
            if qv != 1.0:
                e_none, e_wonly = rel_err(out, bad[0]), rel_err(out, bad[1])
                print(f"  vs no V compensation {e_none:.3e}; vs weight-scale-only {e_wonly:.3e}")
                assert e_none > WRONG_MIN
            # The kernel wrote the active token into the (packed) cache.
            kc, vc = _host_caches(attn)
            s = slots
            k_new = torch.cat([new_kv[b][0] for b in range(4)])
            v_new = torch.cat([new_kv[b][1] for b in range(4)])
            _assert_cache(_cache_err(kc[s // BLOCK, 0, s % BLOCK], k_new, qk), "new-token K")
            _assert_cache(_cache_err(vc[s // BLOCK, 0, s % BLOCK], v_new, qv), "new-token V")
            # Next step reads the token just written (stale-cache check).
            for b in range(4):
                kd, vd = new_kv[b][2], new_kv[b][3]
                hist[b] = (kd, vd)
                lens[b] += 1
    finally:
        hy_model._FORCE_UNPACKED_DECODE = False
