# SPDX-License-Identifier: Apache-2.0
"""Probe the MX (block-scaled fp8/fp4) matmul conventions on the device.

    KH_CORE=8 python probe_mx.py

A1  quantize_mx on a known bf16 tile -> dump the x4 codes and the scale tile:
    which contraction element lands in which byte, where the e8m0 scales sit.
A2  fp8 weight packed on the host as [k/4, n] int32 (byte j = k 4p+j) and its
    scales as [k/32, n]: nc_matmul_mx against a CPU fp32 GEMM.
A3  the same with an fp4 (e2m1) weight packed as [k/4, n] int16 (nibble j = k 4p+j).
A4  a partial K chunk (K = 256 -> 64 partitions).
"""

import sys
import traceback

from kharness import compare, run_nki  # noqa: E402 - sets the environment first
import torch

import nki
import nki.isa as nisa
import nki.language as nl

PT = 128


def e8m0(code):
    return torch.pow(2.0, code.float() - 127.0)


def make_act(m, k, seed):
    """bf16 rows whose 32-blocks are exact e4m3 x 2^(s-127) with amax in [128, 256):
    quantize_mx (X = 2^(floor(log2 amax) - 7)) recovers them exactly."""
    g = torch.Generator().manual_seed(seed)
    nb = k // 32
    codes = torch.randint(-200, 200, (m, nb, 32), generator=g).float()
    # force one element per block to +-[128, 256)
    big = torch.randint(128, 256, (m, nb, 1), generator=g).float()
    codes[:, :, 0:1] = big
    q = codes.to(torch.float8_e4m3fn).float()
    s = torch.randint(120, 134, (m, nb, 1), generator=g)
    x = (q * e8m0(s)).reshape(m, k)
    return x.to(torch.bfloat16), q.reshape(m, k), s.reshape(m, nb)


def make_w8(n, k, seed, subnormal=None):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (n, k), generator=g, dtype=torch.int64)
    w[(w & 0x7F) == 0x7F] = 0  # no NaN codes
    if subnormal is False:
        w[(w & 0x78) == 0] = 0      # no subnormal codes (exponent field 0)
    elif subnormal is True:
        w = (w & 0x87)              # only subnormal codes (and zeros)
    w_u8 = w.to(torch.uint8)
    s = torch.randint(120, 134, (n // 32, k // 32), generator=g).to(torch.uint8)
    wf = w_u8.view(torch.float8_e4m3fn).float()
    sf = e8m0(s).repeat_interleave(32, 0).repeat_interleave(32, 1)
    return w_u8, s, wf * sf


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def make_w4(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g)
    vals = E2M1[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)  # low nibble = even k
    s = torch.randint(120, 134, (n, k // 32), generator=g).to(torch.uint8)
    sf = e8m0(s).repeat_interleave(32, 1)
    return packed, s, vals * sf


def pack_w8_mx(w_u8):
    """[n, k] bytes -> [k/4, n] int32 with byte j = k 4p+j (little endian)."""
    n, k = w_u8.shape
    t = w_u8.reshape(n, k // 4, 4).permute(1, 0, 2).contiguous()  # [k/4, n, 4]
    return t.view(torch.int32).reshape(k // 4, n)


def pack_w4_mx(packed):
    """[n, k/2] bytes -> [k/4, n] int16 with byte j = k pair (4p+2j, 4p+2j+1)."""
    n, k2 = packed.shape
    t = packed.reshape(n, k2 // 2, 2).permute(1, 0, 2).contiguous()  # [k/4, n, 2]
    return t.view(torch.int16).reshape(k2 // 2, n)


def scales_w8_mx(s):
    """[n/32, k/32] -> [k/32, n] (one byte per (k-group, n))."""
    return s.t().repeat_interleave(32, 1).contiguous()


def scales_w4_mx(s):
    """[n, k/32] -> [k/32, n]."""
    return s.t().contiguous()


# ----------------------------------------------------------------------------- kernels


def _load_xT(x, m0, k0, kk, xT):
    """xT[p, m, j] = x[m0 + m, k0 + 4p + j] for a chunk of kk (<= 512) columns: the
    rows loaded contiguously, then four strided PE transposes (columns 4p + j) copied
    into the j-minor layout (a DMA transpose cannot write 2-byte elements at an 8-byte
    stride: "transpose dest offset must be 32B aligned")."""
    pk = kk // 4
    k = x.shape[1]
    rows = nl.ndarray((PT, kk), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows, src=x[m0 : m0 + PT, k0 : k0 + kk])
    interleave_rows(rows, kk, xT)


def interleave_rows(rows, kk, xT):
    """rows [128 (m), kk] bf16 SBUF -> xT[p, m, j] = rows[m, 4p + j]."""
    pk = kk // 4
    for j in range(4):
        tp = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:pk, :], data=rows.ap(pattern=[[kk, PT], [4, pk]], offset=j))
        nisa.tensor_copy(dst=xT[0:pk, :, j], src=tp[0:pk, :])


def _scatter_scales(ws, s_mx, g0, ngroups, n0, nt):
    """ws[32q + i, 0:nt] = s_mx[g0 + 4q + i, n0:n0+nt] for the chunk's groups."""
    for q in range(ngroups // 4):
        nisa.dma_copy(dst=ws[32 * q : 32 * q + 4, 0:nt], src=s_mx[g0 + 4 * q : g0 + 4 * q + 4, n0 : n0 + nt])


@nki.jit
def k_quant_dump(x):
    """A1: quantize a [128, 512] bf16 tile through the transposed-interleaved
    layout; return the x4 codes as int32 [128, 128] and the scale tile [128, 128]."""
    m, k = x.shape
    out_q = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.shared_hbm)
    out_s = nl.ndarray((PT, PT), dtype=nl.uint8, buffer=nl.shared_hbm)
    out_t = nl.ndarray((PT, PT * 4), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    _load_xT(x, 0, 0, 512, xT)
    nisa.dma_copy(dst=out_t, src=xT.reshape((PT, PT * 4)))
    xq = nl.ndarray((PT, PT), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    xs = nl.ndarray((PT, PT), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(xs, 0)
    nisa.quantize_mx(dst=xq, src=xT.reshape((PT, PT * 4)), dst_scale=xs)
    nisa.dma_copy(dst=out_q, src=xq.view(nl.int32))
    nisa.dma_copy(dst=out_s, src=xs)
    return out_q, out_s, out_t


@nki.jit
def k_mm_fp8(x, w_mx, s_mx, flags):
    """A2: out[m, n] = x[m, k] @ w[n, k].T with w in the host MX layout."""
    n, = flags
    m, k = x.shape
    nkc = k // 512
    out = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.shared_hbm)
    w4 = w_mx.view(nl.float8_e4m3fn_x4)
    for mi in range(m // PT):
        xq = nl.ndarray((PT, nkc, PT), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        xs = nl.ndarray((PT, nkc, PT), dtype=nl.uint8, buffer=nl.sbuf)
        for kc in range(nkc):
            xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
            _load_xT(x, mi * PT, kc * 512, 512, xT)
            nisa.quantize_mx(dst=xq[:, kc, :], src=xT.reshape((PT, PT * 4)), dst_scale=xs[:, kc, :])
        for nb in range(n // 512):
            acc = nl.ndarray((PT, 512), dtype=nl.float32, buffer=nl.psum)
            for kc in range(nkc):
                wt = nl.ndarray((PT, 512), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
                nisa.dma_copy(dst=wt, src=w4[kc * PT : (kc + 1) * PT, nb * 512 : (nb + 1) * 512])
                ws = nl.ndarray((PT, 512), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.memset(ws, 0)
                _scatter_scales(ws, s_mx, kc * 16, 16, nb * 512, 512)
                nisa.nc_matmul_mx(dst=acc, stationary=xq[:, kc, :], moving=wt,
                                  stationary_scale=xs[:, kc, :], moving_scale=ws, accumulate=(kc > 0))
            o = nl.ndarray((PT, 512), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o, src=acc)
            nisa.dma_copy(dst=out[mi * PT : (mi + 1) * PT, nb * 512 : (nb + 1) * 512], src=o)
    return out


@nki.jit
def k_mm_fp4(x, w_mx, s_mx, flags):
    """A3: the fp4 weight: w_mx [k/4, n] int16 viewed as float4_e2m1fn_x4."""
    n, = flags
    m, k = x.shape
    nkc = k // 512
    out = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.shared_hbm)
    w4 = w_mx.view(nl.float4_e2m1fn_x4)
    for mi in range(m // PT):
        xq = nl.ndarray((PT, nkc, PT), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        xs = nl.ndarray((PT, nkc, PT), dtype=nl.uint8, buffer=nl.sbuf)
        for kc in range(nkc):
            xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
            _load_xT(x, mi * PT, kc * 512, 512, xT)
            nisa.quantize_mx(dst=xq[:, kc, :], src=xT.reshape((PT, PT * 4)), dst_scale=xs[:, kc, :])
        for nb in range(n // 512):
            acc = nl.ndarray((PT, 512), dtype=nl.float32, buffer=nl.psum)
            for kc in range(nkc):
                wt = nl.ndarray((PT, 512), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf)
                nisa.dma_copy(dst=wt, src=w4[kc * PT : (kc + 1) * PT, nb * 512 : (nb + 1) * 512])
                ws = nl.ndarray((PT, 512), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.memset(ws, 0)
                _scatter_scales(ws, s_mx, kc * 16, 16, nb * 512, 512)
                nisa.nc_matmul_mx(dst=acc, stationary=xq[:, kc, :], moving=wt,
                                  stationary_scale=xs[:, kc, :], moving_scale=ws, accumulate=(kc > 0))
            o = nl.ndarray((PT, 512), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o, src=acc)
            nisa.dma_copy(dst=out[mi * PT : (mi + 1) * PT, nb * 512 : (nb + 1) * 512], src=o)
    return out


@nki.jit
def k_mm_fp8_partial(x, w_mx, s_mx, flags):
    """A4: K = 256 (64 partitions), n = 512, one chunk."""
    n, = flags
    m, k = x.shape
    pk = k // 4
    out = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.shared_hbm)
    w4 = w_mx.view(nl.float8_e4m3fn_x4)
    xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    _load_xT(x, 0, 0, k, xT)
    xq = nl.ndarray((PT, PT), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    xs = nl.ndarray((PT, PT), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(xs, 0)
    nisa.quantize_mx(dst=xq[0:pk, :], src=xT[0:pk, :, :].reshape((pk, PT * 4)), dst_scale=xs[0:pk, :])
    wt = nl.ndarray((PT, n), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    nisa.dma_copy(dst=wt[0:pk, :], src=w4[0:pk, 0:n])
    ws = nl.ndarray((PT, n), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(ws, 0)
    _scatter_scales(ws, s_mx, 0, pk // 8, 0, n)
    acc = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul_mx(dst=acc, stationary=xq[0:pk, :], moving=wt[0:pk, :],
                      stationary_scale=xs[0:pk, :], moving_scale=ws[0:pk, :])
    o = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=acc)
    nisa.dma_copy(dst=out, src=o)
    return out


def main():
    which = set(sys.argv[1:]) or {"A1", "A2", "A3", "A4"}
    if which == {"A7"}:
        return
    ok = True
    if "A1" in which:
        try:
            x, q, s = make_act(128, 512, 1)
            out_q, out_s, out_t = run_nki(k_quant_dump, x)
            # expected transposed/interleaved bf16: t[p, m, j] = x[m, 4p+j]
            exp_t = x.reshape(128, 128, 4).permute(1, 0, 2).reshape(128, 512)
            print("A1 transposed load exact:", bool(torch.equal(out_t, exp_t)))
            # codes: expected x4 at [p, m] packs q[m, 4p+j] as byte j
            qb = q.to(torch.float8_e4m3fn).view(torch.uint8).reshape(128, 128, 4).permute(1, 0, 2)  # [p, m, j]
            got_b = out_q.contiguous().view(torch.uint8).reshape(128, 128, 4)
            same = torch.equal(got_b, qb)
            print("A1 x4 byte order (byte j = k 4p+j):", same)
            if not same:
                rev = torch.equal(got_b, qb.flip(-1))
                mm = (got_b != qb).nonzero()
                print("   reversed byte order:", rev, " mismatches:", mm.shape[0], "of", got_b.numel())
                for r in mm[:6].tolist():
                    p_, m_, j_ = r
                    print(f"   at p={p_} m={m_} j={j_}: got {int(got_b[p_, m_, j_])} exp {int(qb[p_, m_, j_])} "
                          f"x={float(x[m_, 4 * p_ + j_])} code={float(q[m_, 4 * p_ + j_])} s={int(s[m_, (4 * p_ + j_) // 32])}")
                gotv = got_b.contiguous().view(torch.float8_e4m3fn).float()
                expv = qb.contiguous().view(torch.float8_e4m3fn).float()
                print("   value-level mismatches:", int((gotv != expv).sum()), " max |diff|:", float((gotv - expv).abs().max()))
            # scales: group g (data partitions 8g..8g+7) of column m: expected s[m, g]
            nz = (out_s != 0).nonzero()
            print("A1 scale tile nonzero partitions:", sorted(set(nz[:, 0].tolist())))
            exp_s = s.t().contiguous()  # [g, m]
            want = torch.zeros(128, 128, dtype=torch.uint8)
            for g in range(16):
                want[32 * (g // 4) + g % 4] = exp_s[g].to(torch.uint8)
            print("A1 scale placement (32*(g//4) + g%4):", bool(torch.equal(out_s, want)))
            if not torch.equal(out_s, want):
                diff = (out_s.int() - want.int())
                print("   scale diffs (got - want) unique:", diff[want != 0].unique().tolist(),
                      " count nonzero:", int((diff != 0).sum()))
                for p in sorted(set(nz[:, 0].tolist()))[:4]:
                    print(f"   partition {p}: got {out_s[p, :6].tolist()} want {want[p, :6].tolist()}")
        except Exception:
            traceback.print_exc()
            ok = False
    if "A2" in which:
        try:
            m, k, n = 256, 1024, 1024
            x, _, _ = make_act(m, k, 2)
            w_u8, s, wf = make_w8(n, k, 3, subnormal=False)
            ref = x.float() @ wf.t()
            got = run_nki(k_mm_fp8, x, pack_w8_mx(w_u8), scales_w8_mx(s), (n,))
            ok &= compare("A2 fp8 MX matmul (normal codes)", got, ref, tol_rel=1e-5)
            w_u8, s, wf = make_w8(n, k, 3, subnormal=True)
            ref = x.float() @ wf.t()
            got = run_nki(k_mm_fp8, x, pack_w8_mx(w_u8), scales_w8_mx(s), (n,))
            compare("A5 fp8 MX matmul (subnormal weight codes only) vs exact", got, ref, tol_rel=1e-5)
            compare("A5 ... vs zero (flushed?)", got, torch.zeros_like(ref), tol_rel=1e-5)
        except Exception:
            traceback.print_exc()
            ok = False
    if "A6" in which:
        try:
            # same-exponent codes: every 4-term product sum fits the internal adder exactly?
            m, k, n = 128, 1024, 512
            x, _, _ = make_act(m, k, 8)
            g = torch.Generator().manual_seed(9)
            codes = torch.randint(64, 127, (n, k), generator=g) | (torch.randint(0, 2, (n, k), generator=g) * 128)
            w_u8 = codes.to(torch.uint8)
            s = torch.full((n // 32, k // 32), 127, dtype=torch.uint8)
            wf = w_u8.view(torch.float8_e4m3fn).float()
            ref = x.float() @ wf.t()
            got = run_nki(k_mm_fp8, x, pack_w8_mx(w_u8), scales_w8_mx(s), (n,))
            compare("A6 fp8 MX matmul, one weight binade", got, ref, tol_rel=1e-5)
            x2, _, _ = make_act(m, k, 10)
            w_u8, s, wf = make_w8(n, k, 11, subnormal=False)
            ref = x2.float() @ wf.t()
            got = run_nki(k_mm_fp8, x2, pack_w8_mx(w_u8), scales_w8_mx(s), (n,))
            err = (got - ref).abs() / ref.abs().clamp_min(1.0)
            print("A6 random fp8: per-element rel err quantiles 50/90/99/max:",
                  [float(torch.quantile(err.flatten()[:100000], qq)) for qq in (0.5, 0.9, 0.99)], float(err.max()))
        except Exception:
            traceback.print_exc()
    if "A3" in which:
        try:
            m, k, n = 128, 1024, 512
            x, _, _ = make_act(m, k, 4)
            packed, s, wf = make_w4(n, k, 5)
            ref = x.float() @ wf.t()
            got = run_nki(k_mm_fp4, x, pack_w4_mx(packed), scales_w4_mx(s), (n,))
            ok &= compare("A3 fp4 MX matmul", got, ref, tol_rel=1e-4)
        except Exception:
            traceback.print_exc()
            ok = False
    if "A4" in which:
        try:
            m, k, n = 128, 256, 512
            x, _, _ = make_act(m, k, 6)
            w_u8, s, wf = make_w8(n, k, 7)
            ref = x.float() @ wf.t()
            got = run_nki(k_mm_fp8_partial, x, pack_w8_mx(w_u8), scales_w8_mx(s), (n,))
            ok &= compare("A4 fp8 MX matmul K=256", got, ref, tol_rel=1e-4)
        except Exception:
            traceback.print_exc()
            ok = False
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- A7: quantize_mx vs emulation


def emulate(x, mode, rule):
    """act_quant_mx variants: rounding `mode` (rne | trunc), scale `rule` (floor-7 | ceil-8)."""
    T, n = x.shape
    xb = x.float().reshape(T, n // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    if rule == "floor-7":
        e = torch.floor(torch.log2(amax.clamp_min(2.0 ** -127)))
        scale = torch.exp2(e - 7.0)
    else:
        scale = torch.exp2(torch.ceil(torch.log2((amax / 448.0).clamp_min(2.0 ** -127))))
    y = xb / scale
    if mode == "rne":
        q = y.to(torch.float8_e4m3fn).float()
    else:  # truncate toward zero to the e4m3 grid
        mag = y.abs().clamp_min(2.0 ** -6)
        ex = torch.floor(torch.log2(mag))
        step = torch.exp2(ex - 3)
        q = torch.sign(y) * torch.floor(y.abs() / step) * step
        q = q.clamp(-448, 448)
    return (q * scale).reshape(T, n)


def main_a7():
    torch.manual_seed(3)
    x = (torch.randn(128, 512) * 0.7).to(torch.bfloat16)
    x[5, :32] = 0.0
    out_q, out_s, _ = run_nki(k_quant_dump, x)
    codes = out_q.contiguous().view(torch.uint8).reshape(128, 128, 4).permute(1, 0, 2).reshape(128, 512)  # [m, k]
    vals = codes.contiguous().view(torch.float8_e4m3fn).float()
    scale_rows = torch.zeros(128, 16, dtype=torch.int64)       # [m, g] e8m0 codes
    for g in range(16):
        scale_rows[:, g] = out_s[32 * (g // 4) + g % 4].to(torch.int64)
    got = (vals.reshape(128, 16, 32) * torch.exp2(scale_rows.float() - 127).unsqueeze(-1)).reshape(128, 512)
    for mode in ("rne", "trunc"):
        for rule in ("floor-7", "ceil-8"):
            ref = emulate(x, mode, rule)
            n_mis = int((ref != got).sum())
            print(f"A7 hardware quantize_mx vs emulation({mode}, {rule}): mismatching elements {n_mis}/65536, "
                  f"max |diff| {float((ref - got).abs().max()):.4g}")
    e = torch.floor(torch.log2(x.float().reshape(128, 16, 32).abs().amax(-1).clamp_min(2.0 ** -127)))
    print("A7 scale codes match floor(log2 amax)-7+127:", bool(torch.equal(scale_rows, (e - 7 + 127).to(torch.int64))),
          " zero block scale code:", int(scale_rows[5, 0]))


if __name__ == "__main__" and "A7" in sys.argv:
    main_a7()
