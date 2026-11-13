# SPDX-License-Identifier: Apache-2.0
"""A bs1 decode block (one span of T = 6 rows) with its residual stream on chip (round 5).

The stream lives in SBUF between the launch's consecutive members of this kind, in the packed
layout `S[p, i, f] = stream[t, i, 512 c + f]`, p = t * 10 + c (the 10 512-column chunks of row t on
partitions 10 t .. 10 t + 9): a row's per-row coefficients are per-partition scalars there, the
all-reduce results load into it with one contiguous DMA, and the MX GEMM inputs stage from it with
four PE transposes. Both cores hold the whole stream (the hyper-connection updates are cheap and
the cores exchange nothing for them); the coefficients (pre / post / comb) are computed on all 60
partitions (each partition its row's values).

    hc_in(attn) -> attention side (`b1attn`) -> AR-8 -> hc_out(attn) -> hc_in(ffn) -> MoE (`moe_tp`)
                -> AR-64 -> hc_out(ffn)

Same arithmetic as the generic body (`layer_kernel.layer_body`): the pre-weighted stream sum in fp32
(streams in order), its bf16 rounding, the RMS norm of the bf16 values, the bf16 norm output; the mix
GEMM's products of the bf16 stream and `fn` (its weights split hi + lo in bf16: the products are exact
and the sum carries fn to ~2^-17); sigmoid / Sinkhorn coefficients in fp32; hc_out's fp32 products and
sums in the generic order, one bf16 rounding. The fp32 sums run in a different order.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from . import b1attn as A
from . import b1moe
from .mk_common import NORM_EPS, PT
from .. import moe_tp

HC = 4
T6 = 6
KC = 512
HC_EPS = 1e-06
SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5
TOPK_EXPERTS = 6
#: the core computing hc_in(attn)'s / hc_in(ffn)'s coefficients (mix GEMM, sigmoids, Sinkhorn): each
#: core one sublayer's 20-iteration Sinkhorn chain (~120 vector-engine instructions), the other core's
#: vector engine free for the critical path; the coefficients swapped where first read
COEF_A = 1
COEF_F = 0
RH = A.RH
#: the stream layout's half offset (compute-engine accesses of 60 partitions start at 0 or 64)
H0 = 64
#: the next sublayer's input as the reference computes it, the pre-weighted sum of the bf16-rounded
#: stream (after hc_out), instead of the shortcut alpha x + sum_i beta_i S_i (made during the
#: all-reduce: ~2 us earlier, but from the unrounded stream)
SHORTCUT_EXACT = True
#: the block's world all-reduce scheduled after its prefetch of the next member's first GEMM weights
PREFETCH_BY_AR2 = True
#: the block's static loads' DMA queue (norm weights, hc constants, the MoE's router / shared expert, the
#: next member's constants): "sync" (idle in the attention side) or "act" (the ACT queue: its in-order
#: stream held the critical path's activations behind ~50 DMA issues and their queue-credit waits)
STATIC_Q = "sync"
#: hc_in(attn)'s mix (core COEF_A) scheduled after its kv path (else free: it interleaved with it)
MIX_AFTER_SIDE = True
#: the coefficient core's mix traced right behind its kv path (`attn_side`'s after_side: the PE idles
#: there until the q exchange) instead of at the first tick (behind its q path)
MIX_AT_SIDE = True
#: the HBM outputs' stores (off the critical path): software DGE (GpSimd, idle at the block's start)
#: instead of the sync queue (~0.6 us of sync issue each, ahead of the block's start loads)
OUT_DMA = dict(dge_mode=nisa.dge_mode.swdge)
#: the next member's prefetched first-GEMM weights read once (one element, GpSimd) right after this
#: block's world all-reduce is triggered: the scheduler otherwise issues that prefetch just before
#: the next block's x_norm (as late as it estimates it can), and the next block's first GEMM waited
#: ~12 us for it
PREF_PIN = False
#: where the next member's first-GEMM weights are prefetched: "down" (before this block's down GEMM, in
#: the MoE's weight stream) or "ar2" (behind this block's world all-reduce trigger: the DMA queues idle)
PREFETCH_AT = "ar1done"
#: per-core overrides of PREFETCH_AT (with `b1attn.A_PERSIST_CORES`: a core whose persistent first-GEMM
#: buffer the compiler spilled -- the SBUF full in the MoE -- prefetches into a fresh tile at the down GEMM)
PREFETCH_AT_CORE = {}
#: every DMA a member traces scheduled after the previous member's all-reduce result load (the next
#: block's loads otherwise run inside the world all-reduce, which then takes ~2x its floor)
GATE_DMAS = True
#: instead of the gate: every DMA a member traces without an explicit priority at this DMA priority (the
#: collectives' transfers ahead of the hoisted loads); None: off
LOW_PRIO = None
#: the MoE's static weights issued during the attention side ("r" router, "s" shared expert)
MST_EARLY = "rs"
#: ... and behind the attention side
MST_LATE_ATTN = ""
BF = nl.bfloat16
F32 = nl.float32


def _hw(eng):
    return dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)


def is_bd_fn(fn):
    """`fn.T` in the block-diagonal hi | lo layout (`layout.py`: `[128, 2, 16, 240]` bf16)."""
    return len(fn.shape) == 4 and tuple(fn.shape[:2]) == (PT, 2)


# --------------------------------------------------------------------------- launch constants
#
# The stream on 120 partitions: `S[P, s, f]`, P = h * 60 + t * 10 + c, holds stream i = 2 s + h of row
# t, columns 512 c + f (two streams a partition half). Its linear maps (the pre-weighted sums, hc_out)
# run on the PE: per-partition coefficients as block-diagonal stationaries (`Ibase[P, p] = (P % 60 ==
# p)` times the coefficient), split hi + lo in bf16 (the products of the bf16 stream are exact, the
# sums fp32: the coefficients to ~2^-17).


def _row(pattern, NP, sentinel):
    """`[PT, 2, 64]` fp32: each partition the same row, `iota(pattern)` over `(half, t, c)` at
    columns `h * 64 + t * C + c`, `sentinel` at the halves' 4 unused columns."""
    ti = nl.ndarray((PT, 2, H0), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(ti, sentinel)
    C = NP // T6
    nisa.iota(dst=ti.ap(pattern=[[2 * H0, PT], [H0, 2], [C, T6], [1, C]], offset=0), pattern=pattern, offset=0,
              channel_multiplier=0)
    r = nl.ndarray((PT, 2 * H0), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=ti.reshape((PT, 2 * H0)))
    return r


def consts(L, C):
    """Shape-only tiles shared by the launch's members (built by the first): the row-sum matrix
    `msum2[P, Q] = (row(P) == row(Q))` (the 128-partition layout; [0:60, 0:60] the packed one's), the
    mix's diagonal mask `bdm[p, c', m] = (chunk(p) == c')`, `Ibase[P, p] = (P % 64 == p)` (zero on the
    unused partitions / columns)."""
    if "msum2" in L:
        return
    NP = T6 * C
    NS = 2 * H0
    trow = _row([[0, 2], [1, T6], [0, C]], NP, -1)
    tr2 = _row([[0, 2], [1, T6], [0, C]], NP, -2)
    crow = _row([[0, 2], [0, T6], [1, C]], NP, -1)
    qrow = _row([[0, 2], [C, T6], [1, C]], NP, -5)
    qr2 = _row([[0, 2], [C, T6], [1, C]], NP, -7)
    # the same values as columns (partition P's row / chunk / P % 64): PE transposes of one row
    tc = nl.ndarray((PT, 3), dtype=F32, buffer=nl.psum)
    nisa.nc_transpose(dst=tc[0:NS, 0:1], data=tr2[0:1, 0:NS])
    nisa.nc_transpose(dst=tc[0:NS, 1:2], data=crow[0:1, 0:NS])
    nisa.nc_transpose(dst=tc[0:NS, 2:3], data=qr2[0:1, 0:NS])
    tcol = nl.ndarray((PT, 3), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tcol[0:NS], src=tc[0:NS])
    msum2 = nl.ndarray((PT, NS), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=msum2[0:NS], data=trow[0:NS], op0=nl.equal, operand0=tcol[0:NS, 0:1])
    mix = 24
    cj = nl.ndarray((PT, C, mix), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=cj, pattern=[[1, C], [0, mix]], offset=0, channel_multiplier=0)
    cjf = nl.ndarray((PT, C * mix), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cjf, src=cj.reshape((PT, C * mix)))
    bdm = nl.ndarray((PT, C * mix), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bdm[0:NP], data=cjf[0:NP], op0=nl.equal, operand0=tcol[0:NP, 1:2])
    ib = nl.ndarray((PT, H0), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ib[0:NS], data=qrow[0:NS, 0:H0], op0=nl.equal, operand0=tcol[0:NS, 2:3])
    L["msum2"] = msum2
    L["bdm"] = bdm
    L["Ibase"] = ib


def rowsum(L, col, NP, n=1, NO=None):
    """`out[p, :] = sum over the partitions q (of `NP`) of p's row of col[q, :]` (fp32 PE, exact 0/1
    weights), on `NO` output partitions (default `NP`; the 120-partition layout's rows repeat)."""
    NO = NO or NP
    ps = nl.ndarray((PT, n), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps[0:NO, 0:n], stationary=L["msum2"][0:NP, 0:NO], moving=col[0:NP, 0:n])
    return ps


def packed_w(row_hbm, C, dtype, eng):
    """A `[1, C * 512]` per-column weight in the packed layout: `w[t * C + c, f] = row[512 c + f]`."""
    w = nl.ndarray((PT, KC), dtype=dtype, buffer=nl.sbuf)
    for t in range(T6):
        nisa.dma_copy(dst=w[t * C : (t + 1) * C, :], src=row_hbm.ap(pattern=[[KC, C], [1, KC]], offset=0), **_hw(eng))
    return w


def bcast60(row_hbm, n, NP, eng):
    """A `[1, n]` row on partitions 0 .. NP - 1 (fp32)."""
    t = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t[0:NP, :], src=row_hbm.ap(pattern=[[0, NP], [1, n]], offset=0), **_hw(eng))
    return t


# --------------------------------------------------------------------------- hyper-connections


def hilo(x, shape):
    """`x` (fp32 SBUF / AP) split `(hi, lo)` bf16: hi = bf16(x), lo = bf16(x - hi)."""
    hi = nl.ndarray(shape, dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=hi, src=x)
    lo = nl.ndarray(shape, dtype=BF, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=lo, data1=x, data2=hi, op=nl.subtract)
    return hi, lo


def load_stream(stream, C):
    """The stream input `[6, 4, C * 512]` (HBM) in the 128-partition layout: `[128, 2, 512]` bf16."""
    NP = T6 * C
    dim = C * KC
    S = nl.ndarray((PT, 2, KC), dtype=BF, buffer=nl.sbuf)
    nisa.memset(S, 0.0)
    for h in range(2):
        for s_ in range(2):
            for t in range(T6):
                nisa.dma_copy(dst=S[h * H0 + t * C : h * H0 + (t + 1) * C, s_, :],
                              src=stream.ap(pattern=[[KC, C], [1, KC]], offset=t * HC * dim + (2 * s_ + h) * dim),
                              **_hw(nisa.engine.sync))
    return S


def wsum_stat(L, w, NP):
    """The stationaries of `sum_i w_i S_i` (`w` `[128, 4]` fp32 per-partition weights): `(hi, lo)`
    `[128, 2, 60]` bf16 with `[h * 64 + q, s, p] = (q == p) w_{2 s + h}(q)`."""
    st = nl.ndarray((PT, 2, NP), dtype=F32, buffer=nl.sbuf)
    nisa.memset(st, 0.0)
    ib = L["Ibase"]
    for h in range(2):
        nisa.tensor_tensor(dst=st[h * H0 : h * H0 + NP], data1=ib.ap(pattern=[[H0, NP], [0, 2], [1, NP]], offset=h * H0 * H0),
                           data2=w.ap(pattern=[[w.shape[1], NP], [2, 2], [0, NP]], offset=h * H0 * w.shape[1] + h),
                           op=nl.multiply)
    return hilo(st, (PT, 2, NP))


def diag_stat(L, v, NP):
    """`(hi, lo)` `[60, 60]` bf16 stationaries of `diag(v)` (`v` `[>= 60, 1]` fp32)."""
    st = nl.ndarray((PT, NP), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=st[0:NP], data=L["Ibase"][0:NP, 0:NP], op0=nl.multiply, operand0=v[0:NP, 0:1])
    return hilo(st[0:NP], (NP, NP))


def wsum_mm(ps, stat, S, NP, first=True):
    """`ps[0:60] (+)= sum_i w_i S_i` (the `wsum_stat` stationaries)."""
    n = 0
    for s_ in range(2):
        for t in stat:
            nisa.nc_matmul(dst=ps[0:NP, :], stationary=t[:, s_, :], moving=S[:, s_, :],
                           accumulate=not (first and n == 0))
            n += 1


def diag_mm(ps, stat, x16, NP):
    """`ps[0:60] += diag(v) x` (the `diag_stat` stationaries; `x16` `[60, 512]` bf16)."""
    for t in stat:
        nisa.nc_matmul(dst=ps[0:NP, :], stationary=t[0:NP, 0:NP], moving=x16[0:NP, :], accumulate=True)


def x_sum(L, S, pre, NP):
    """`sum_i pre_i S_i` (fp32 PSUM `[60, 512]`) of the stream (`pre` `[120, 4]`)."""
    ps = nl.ndarray((PT, KC), dtype=F32, buffer=nl.psum)
    wsum_mm(ps, wsum_stat(L, pre, NP), S, NP)
    return ps


def shortcut(L, S, coef, NP):
    """The next sublayer's input from this sublayer's output without the bf16 stream in between:
    with S' = post x + comb^T S, sum_j pre_j S'_j = alpha x + R, alpha = sum_j pre_j post_j,
    R = sum_i beta_i S_i, beta_i = sum_j pre_j comb_ij (`coef` = [pre | post | comb], 128 partitions):
    R accumulated into a PSUM tile now (while the all-reduce of x is in flight), `(ps, alpha's
    stationaries)` for `x_short`."""
    NS = 2 * H0
    pp = nl.ndarray((PT, HC), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pp[0:NP], data1=coef[0:NP, 0:HC], data2=coef[0:NP, HC : 2 * HC], op=nl.multiply)
    alpha = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=alpha[0:NP], op=nl.add, data=pp[0:NP], axis=(1,), keepdims=True)
    pc = nl.ndarray((PT, HC, HC), dtype=F32, buffer=nl.sbuf)
    w = coef.shape[1]
    nisa.tensor_tensor(dst=pc[0:NS], data1=coef[0:NS, 2 * HC : 2 * HC + HC * HC].reshape((NS, HC, HC)),
                       data2=coef.ap(pattern=[[w, NS], [0, HC], [1, HC]], offset=0), op=nl.multiply)
    beta = nl.ndarray((PT, HC), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=beta[0:NS], op=nl.add, data=pc[0:NS], axis=(2,))
    ps = nl.ndarray((PT, KC), dtype=F32, buffer=nl.psum)
    wsum_mm(ps, wsum_stat(L, beta, NP), S, NP)
    return ps, diag_stat(L, alpha, NP)


def x_short(sc, x16, NP):
    """`alpha x + R` (fp32 PSUM `[60, 512]`), `sc` = `shortcut(..)`."""
    ps, ast = sc
    diag_mm(ps, ast, x16, NP)
    return ps


def x_norm(L, acc, norm_w, C, eng_w, w60=None):
    """`hc_in`'s normed input from the fp32 pre-weighted stream sum `acc`: its bf16 rounding, the RMS
    norm of the bf16 values, bf16: `xn [60, 512]` packed. `w60`: the packed weight (else loaded)."""
    NP = T6 * C
    dim = C * KC
    y16 = nl.ndarray((PT, KC), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=y16[0:NP], src=acc[0:NP])
    # RMS norm of the bf16 row (per-partition sums of squares, summed over the row's partitions)
    junk = nl.ndarray((PT, KC), dtype=F32, buffer=nl.sbuf)
    ss = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=junk[0:NP], op=nl.square, data=y16[0:NP], reduce_op=nl.add,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=ss[0:NP, 0:1])
    ssr = rowsum(L, ss, NP)
    rs = A.rstd_of(ssr, NP, dim)
    if w60 is None:
        w60 = packed_w(norm_w, C, BF, eng_w)
    yv = nl.ndarray((PT, KC), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=yv[0:NP], op=nl.copy, data=y16[0:NP], scale=rs[0:NP, 0:1])
    xn = nl.ndarray((PT, KC), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xn[0:NP], data1=yv[0:NP], data2=w60[0:NP], op=nl.multiply)
    return xn


def hc_consts(fnbd, base, scale, C, eng):
    """A sublayer's static hc_in tensors in SBUF (issued early): `fn.T` (block-diagonal hi | lo),
    the base and the scales broadcast to the 128 partitions."""
    NS = 2 * H0
    mix = fnbd.shape[3] // C
    fsb = nl.ndarray(tuple(fnbd.shape), dtype=BF, buffer=nl.sbuf)
    nisa.dma_copy(dst=fsb, src=fnbd, **_hw(eng))
    return fsb, bcast60(base, mix, NS, eng), bcast60(scale, 3, NS, eng)


def _after(ins, after):
    """`ins` scheduled after `after` (a schedule-only dependency), if given."""
    if after is not None and hasattr(ins, "depends_on") and hasattr(after, "depends_on"):
        ins.depends_on(after, kind=nisa.dependency.schedule)
    return ins


def hc_mix(L, S, consts, C, after=None, parts=False):
    """hc_in's mix of the stream `S` (128-partition layout): `[128, 24]` fp32 (the RMS-scaled
    block-diagonal GEMM; every partition its row's values)."""
    NP = T6 * C
    NS = 2 * H0
    dim = C * KC
    width = HC * dim
    fsb = consts[0]
    # the mix: sum of squares of the whole stream, the block-diagonal GEMM
    junk2 = nl.ndarray((PT, 2 * KC), dtype=F32, buffer=nl.sbuf)
    ss2 = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    _after(nisa.activation(dst=junk2[0:NS], op=nl.square, data=S[0:NS].reshape((NS, 2 * KC)), reduce_op=nl.add,
                           reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=ss2[0:NS, 0:1]), after)
    ss2r = rowsum(L, ss2, NS)
    if parts:
        # (the rstd -- GpSimd's rsqrt, waiting for the sums -- deferred to `hc_mix_end`: ahead of an
        # exchange in GpSimd's queue it held it)
        rs2 = None
    else:
        rs2 = A.rstd_of(ss2r, NS, width)
    # xT[k, s * 4 + q, P] = S[P, s, 128 q + k] (`layout.fn_bd`'s row order)
    tp = nl.ndarray((PT, 8, NS), dtype=BF, buffer=nl.psum)
    for s_ in range(2):
        for q in range(4):
            _after(nisa.nc_transpose(dst=tp[:, s_ * 4 + q, :], data=S.ap(pattern=[[2 * KC, NS], [1, PT]],
                                                                           offset=s_ * KC + q * PT)), after)
    xT = nl.ndarray((PT, 8, NS), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xT, src=tp)
    mix = fsb.shape[3] // C
    mp = nl.ndarray((NP, C * mix), dtype=F32, buffer=nl.psum)
    n = 0
    for hl in range(2):
        for s_ in range(2):
            for q in range(4):
                for h in range(2):
                    i = 2 * s_ + h
                    nisa.nc_matmul(dst=mp, stationary=xT[:, s_ * 4 + q, h * H0 : h * H0 + NP], moving=fsb[:, hl, i * 4 + q, :],
                                   accumulate=(n > 0))
                    n += 1
    dg = nl.ndarray((PT, C * mix), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dg[0:NP], data1=mp, data2=L["bdm"][0:NP], op=nl.multiply)
    part = nl.ndarray((PT, mix), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=part[0:NP], op=nl.add, data=dg.ap(pattern=[[C * mix, NP], [1, mix], [mix, C]], offset=0),
                       axis=(2,))
    mx = rowsum(L, part, NP, mix, NO=NS)
    if parts:
        return (mx, ss2r, NS, width, mix)
    mixv = nl.ndarray((PT, mix), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mixv[0:NS], data=mx[0:NS], op0=nl.multiply, operand0=rs2[0:NS, 0:1])
    return mixv


def hc_mix_end(st):
    """`hc_mix(parts=True)`'s rstd and scaled mix."""
    mx, ss2r, NS, width, mix = st
    rs2 = A.rstd_of(ss2r, NS, width)
    mixv = nl.ndarray((PT, mix), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mixv[0:NS], data=mx[0:NS], op0=nl.multiply, operand0=rs2[0:NS, 0:1])
    return mixv


def sinkhorn_gen(t2, NP, box):
    """`moe_tp._sinkhorn_pad` (each matrix `(hc + 1) x (hc + 1)` with an eps column / row: a reduce is
    `sum + eps`); a generator yielding after the
    softmax and after every iteration (the caller places the steps in the vector engine's order).
    The result `[PT, 1, hc * hc]` is appended to `box`."""
    hc = HC
    H = hc + 1
    G = H * H
    a = nl.ndarray((PT, H, H), dtype=F32, buffer=nl.sbuf)
    b = nl.ndarray((PT, H, H), dtype=F32, buffer=nl.sbuf)
    _pin(nisa.memset(a[0:NP], HC_EPS))
    _pin(nisa.memset(b[0:NP], HC_EPS))

    def mat(x):
        return x.ap(pattern=[[G, NP], [H, hc], [1, hc]], offset=0)

    x3 = t2.reshape((PT, hc, hc))
    mx = nl.ndarray((PT, hc, 1), dtype=F32, buffer=nl.sbuf)
    _pin(nisa.tensor_reduce(dst=mx[0:NP], op=nl.max, data=x3[0:NP], axis=(2,), keepdims=True))
    dif = nl.ndarray((PT, hc, hc), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dif[0:NP], data1=x3[0:NP], data2=nl.broadcast_to(mx[0:NP], (NP, hc, hc)), op=nl.subtract)
    e = nl.ndarray((PT, hc, hc), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=e[0:NP], op=nl.exp, data=dif[0:NP])
    se = nl.ndarray((PT, hc, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=se[0:NP], op=nl.add, data=e[0:NP], axis=(2,), keepdims=True)
    re = nl.ndarray((PT, hc, 1), dtype=F32, buffer=nl.sbuf)
    nisa.reciprocal(dst=re[0:NP], data=se[0:NP])
    nisa.tensor_tensor(dst=mat(a), data1=e[0:NP], data2=nl.broadcast_to(re[0:NP], (NP, hc, hc)), op=nl.multiply)
    nisa.tensor_scalar(dst=mat(a), data=mat(a), op0=nl.add, operand0=HC_EPS)
    yield

    def rowdiv(src, dst):
        sm = nl.ndarray((PT, hc), dtype=F32, buffer=nl.sbuf)
        _pin(nisa.tensor_reduce(dst=sm[0:NP], op=nl.add, data=src.ap(pattern=[[G, NP], [H, hc], [1, H]], offset=0),
                                axis=(2,)))
        r = nl.ndarray((PT, hc), dtype=F32, buffer=nl.sbuf)
        nisa.reciprocal(dst=r[0:NP], data=sm[0:NP])
        nisa.tensor_tensor(dst=mat(dst), data1=mat(src), data2=r.ap(pattern=[[hc, NP], [1, hc], [0, hc]], offset=0),
                           op=nl.multiply)

    def coldiv(src, dst):
        cs = nl.ndarray((PT, hc), dtype=F32, buffer=nl.sbuf)
        # [p, j, i]: column j's hc entries and the eps row
        _pin(nisa.tensor_reduce(dst=cs[0:NP], op=nl.add, data=src.ap(pattern=[[G, NP], [1, hc], [H, H]], offset=0),
                                axis=(2,)))
        d = nl.ndarray((PT, hc), dtype=F32, buffer=nl.sbuf)
        nisa.reciprocal(dst=d[0:NP], data=cs[0:NP])
        nisa.tensor_tensor(dst=dst, data1=mat(src), data2=d.ap(pattern=[[hc, NP], [0, hc], [1, hc]], offset=0),
                           op=nl.multiply)

    out = nl.ndarray((PT, 1, hc * hc), dtype=F32, buffer=nl.sbuf)
    o3 = out.ap(pattern=[[hc * hc, NP], [hc, hc], [1, hc]], offset=0)
    coldiv(a, mat(b))
    yield
    for it in range(moe_tp.SINKHORN_ITERS - 1):
        rowdiv(b, a)
        coldiv(a, o3 if it == moe_tp.SINKHORN_ITERS - 2 else mat(b))
        yield
    box.append(out)


def hc_tail(mixv, consts, C, box):
    """hc_in's coefficients from the mix (sigmoids, Sinkhorn): a generator (see `sinkhorn_gen`);
    appends the `[60, 24]` fp32 tile `[pre 4 | post 4 | comb 16]` to `box`."""
    NP = 2 * H0  # (the 128 partitions)
    _, bb, sc = consts
    hh = HC * HC
    t0 = nl.ndarray((PT, HC), dtype=F32, buffer=nl.sbuf)
    _pin(nisa.scalar_tensor_tensor(dst=t0[0:NP], data=mixv[0:NP, 0:HC], op0=nl.multiply, operand0=sc[0:NP, 0:1],
                                   op1=nl.add, operand1=bb[0:NP, 0:HC]))
    pre_n = nl.ndarray((PT, HC), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=pre_n[0:NP], op=nl.sigmoid, data=t0[0:NP])
    nisa.tensor_scalar(dst=pre_n[0:NP], data=pre_n[0:NP], op0=nl.add, operand0=HC_EPS)
    t1 = nl.ndarray((PT, HC), dtype=F32, buffer=nl.sbuf)
    _pin(nisa.scalar_tensor_tensor(dst=t1[0:NP], data=mixv[0:NP, HC : 2 * HC], op0=nl.multiply, operand0=sc[0:NP, 1:2],
                                   op1=nl.add, operand1=bb[0:NP, HC : 2 * HC]))
    post = nl.ndarray((PT, HC), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=post[0:NP], op=nl.sigmoid, data=t1[0:NP])
    nisa.tensor_scalar(dst=post[0:NP], data=post[0:NP], op0=nl.multiply, operand0=2.0)
    t2 = nl.ndarray((PT, 1, hh), dtype=F32, buffer=nl.sbuf)
    _pin(nisa.scalar_tensor_tensor(dst=t2[0:NP, 0, :], data=mixv[0:NP, 2 * HC : 2 * HC + hh], op0=nl.multiply,
                                   operand0=sc[0:NP, 2:3], op1=nl.add, operand1=bb[0:NP, 2 * HC : 2 * HC + hh]))
    sb = []
    for _ in sinkhorn_gen(t2, NP, sb):
        yield
    comb = sb[0]
    coef = nl.ndarray((PT, 2 * HC + hh), dtype=F32, buffer=nl.sbuf)
    nisa.memset(coef, 0.0)
    nisa.tensor_copy(dst=coef[0:NP, 0:HC], src=pre_n[0:NP], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=coef[0:NP, HC : 2 * HC], src=post[0:NP], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=coef[0:NP, 2 * HC : 2 * HC + hh], src=comb.reshape((PT, hh))[0:NP], engine=nisa.engine.vector)
    box.append(coef)


def hc_coef(L, S, fnbd, base, scale, C, eng_w):
    """`hc_in`'s coefficients of the packed stream `S` (this sublayer's pre / post / comb), on all 60
    partitions: one `[60, 24]` fp32 tile `[pre 4 | post 4 | comb 16]`."""
    consts = hc_consts(fnbd, base, scale, C, eng_w)
    box = []
    for _ in hc_tail(hc_mix(L, S, consts, C), consts, C, box):
        pass
    return box[0]


#: the instruction the current `Ticker` step's first instructions are scheduled after
_STEP_PIN = [None]


def _pin(ins):
    """`ins` scheduled after the current step's pin (`Ticker(after=)`), if any."""
    return _after(ins, _STEP_PIN[0])


def hc_coef_gen(L, S, consts, C, box):
    """`hc_tail(hc_mix(..))` as a generator whose first step is the mix, scheduled after this core's
    q / kv path (`b1attn.ANCH["side"]`, set by then: the first step runs at the first tick)."""
    st = hc_mix(L, S, consts, C, after=(_STEP_PIN[0] or A.ANCH.get("side")) if MIX_AFTER_SIDE else None,
                parts=MIX_AT_SIDE)
    yield
    mixv = hc_mix_end(st) if MIX_AT_SIDE else st
    yield from hc_tail(mixv, consts, C, box)


class Ticker:
    """Steps a generator (`hc_tail`) at chosen points of the instruction stream: `tick(n)` emits its
    next `n` steps, `rest()` all remaining; `box[0]` its result."""

    def __init__(self, gen=None, box=None):
        self.gen, self.box = gen, box

    def __call__(self, n=1, after=None):
        if self.gen is None:
            return
        _STEP_PIN[0] = after
        try:
            for _ in range(n):
                if next(self.gen, StopIteration) is StopIteration:
                    self.gen = None
                    return
        finally:
            _STEP_PIN[0] = None

    def rest(self):
        while self.gen is not None:
            self(1)
        return self.box[0] if self.box else None


def coef_from(pid, owner, coef, pipe, after=None, bufs=None):
    """The coefficients computed on core `owner` (`coef` there; this core's tile is a placeholder on
    the other) on both cores: one `sendrecv` (the owner's tile one way, the other's back, unused).
    `bufs`: a launch-level dict for a persistent receive tile (`b1attn.SWAP_PERSIST`)."""
    key = ("coef", pipe, tuple(coef.shape), pid)
    got = bufs.get(key) if (A.SWAP_PERSIST and bufs is not None) else None
    if got is None:
        got = nl.ndarray(coef.shape, dtype=F32, buffer=nl.sbuf, name=f"cfr{pipe}_{pid}" if bufs is not None else "")
        if A.SWAP_PERSIST and bufs is not None:
            bufs[key] = got
    # (`after`: the all-reduce trigger -- GpSimd triggers it and runs this exchange's wait: ahead of
    # it, the wait for the other core's Sinkhorn held the collective)
    _after(nisa.sendrecv(src=coef, dst=got, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=pipe,
                         dma_engine=nisa.dma_engine.gpsimd_dma), after)
    return coef if pid == owner else got


def hc_out_pre(L, S, coef, NP):
    """hc_out's stream part (`sum_i comb_ij S_i` for every j) into two PSUM tiles (made while x's
    all-reduce is in flight) and the post stationaries: the state for `hc_out_fin`."""
    NS = 2 * H0
    w = coef.shape[1]
    ib = L["Ibase"]
    # stat[h' * 64 + q, s', j, p'] = (q == p') comb_{2 s' + h', j}(q) (p' < 64: the output partition
    # h * 64 + p' of stream j = 2 s + h)
    st = nl.ndarray((PT, 2, HC, H0), dtype=F32, buffer=nl.sbuf)
    nisa.memset(st, 0.0)
    for h in range(2):
        nisa.tensor_tensor(dst=st[h * H0 : h * H0 + NP],
                           data1=ib.ap(pattern=[[H0, NP], [0, 2], [0, HC], [1, H0]], offset=h * H0 * H0),
                           data2=coef.ap(pattern=[[w, NP], [2 * HC, 2], [1, HC], [0, H0]],
                                         offset=h * H0 * w + 2 * HC + HC * h),
                           op=nl.multiply)
    sh, sl = hilo(st, (PT, 2, HC, H0))
    # post: [q, j, p'] = (q == p') post_j(q)
    sx = nl.ndarray((PT, HC, H0), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sx[0:NP], data1=ib.ap(pattern=[[H0, NP], [0, HC], [1, H0]], offset=0),
                       data2=coef.ap(pattern=[[w, NP], [1, HC], [0, H0]], offset=HC), op=nl.multiply)
    xh, xl = hilo(sx[0:NP], (NP, HC, H0))
    ps = []
    for s_ in range(2):
        p_ = nl.ndarray((PT, KC), dtype=F32, buffer=nl.psum)
        n = 0
        for s2 in range(2):
            for t in (sh, sl):
                nisa.nc_matmul(dst=p_[0:NS, :], stationary=t.reshape((PT, 2, HC * H0))[:, s2, 2 * s_ * H0 : (2 * s_ + 2) * H0],
                               moving=S[0:NS, s2, :],
                               accumulate=(n > 0))
                n += 1
        ps.append(p_)
    return ps, (xh, xl)


def hc_out_fin(pre, x16, NP):
    """`S'[:, j] = post_j x + sum_i comb[i, j] S[:, i]` (fp32 PSUM; bf16) from `hc_out_pre`'s state."""
    NS = 2 * H0
    ps, (xh, xl) = pre
    out = nl.ndarray((PT, 2, KC), dtype=BF, buffer=nl.sbuf)
    for s_ in range(2):
        for t in (xh, xl):
            nisa.nc_matmul(dst=ps[s_][0:NS, :], stationary=t.reshape((NP, HC * H0))[:, 2 * s_ * H0 : (2 * s_ + 2) * H0],
                           moving=x16[0:NP, :],
                           accumulate=True)
        nisa.tensor_copy(dst=out[0:NS, s_, :], src=ps[s_][0:NS, :],
                         engine=nisa.engine.scalar if s_ == 0 else nisa.engine.vector)
    return out


def hc_out(L, S, x16, coef, NP):
    return hc_out_fin(hc_out_pre(L, S, coef, NP), x16, NP)


_LP = [None]


def load_packed(cc_dst, NP, eng):
    """An all-reduce result `[6, C * 512]` fp32 (HBM rows) into the packed layout, rounded to bf16."""
    x32 = nl.ndarray((PT, KC), dtype=F32, buffer=nl.sbuf)
    _LP[0] = nisa.dma_copy(dst=x32[0:NP, :], src=cc_dst.ap(pattern=[[KC, NP], [1, KC]], offset=0), **_hw(eng))
    x16 = nl.ndarray((PT, KC), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x16[0:NP], src=x32[0:NP])
    return x16


# --------------------------------------------------------------------------- the block


def block(stream, pre_mix, hca_fn_t, hca_base, hca_scale, attn_norm_w, hcf_fn_t, hcf_base, hcf_scale, ffn_norm_w,
          wq_a4, wq_a_s, q_norm_w, wq_b4, wq_b_s, wkv4, wkv_s, kv_norm_w, wo_a4, wo_a_s, wo_b4, wo_b_s, attn_sink,
          freqs, row_pos, write_slots, win_slots, cmp_slots, wv, topk_in, window_kv_cache, compress_kv_cache,
          gate_w_t, gate_b, expert_lo, w1, s1, w2, s2, sh_w1, sh_s1, sh_w2, sh_s2, groups_attn, groups_world,
          share, out_stream, ffn_pre, barrier, tap_out=None, ix=None, window_only=False, src=None, S_in=None):
    """One r1 / r2 consumer block for a 6-row span (see the module docstring): writes `out_stream` /
    `ffn_pre` (HBM, for the launch's outputs and the next member of another kind) and leaves the stream
    in `share["b1"]` for the next member of this kind. `ix`: an index source's indexer tensors (`b1idx`;
    its top-k into `topk_in`, the block's `topk_out`)."""
    pid = nl.program_id(0)
    L = share.setdefault("b1", {}) if share is not None else {}
    gate = L.get("gate") if (GATE_DMAS and (L.get("S_src") is stream or S_in is not None)) else None
    if gate is None and LOW_PRIO is None:
        return _block(stream, pre_mix, hca_fn_t, hca_base, hca_scale, attn_norm_w, hcf_fn_t, hcf_base, hcf_scale,
                      ffn_norm_w, wq_a4, wq_a_s, q_norm_w, wq_b4, wq_b_s, wkv4, wkv_s, kv_norm_w, wo_a4, wo_a_s, wo_b4,
                      wo_b_s, attn_sink, freqs, row_pos, write_slots, win_slots, cmp_slots, wv, topk_in, window_kv_cache,
                      compress_kv_cache, gate_w_t, gate_b, expert_lo, w1, s1, w2, s2, sh_w1, sh_s1, sh_w2, sh_s2,
                      groups_attn, groups_world, share, out_stream, ffn_pre, barrier, tap_out, ix, window_only, src,
                      S_in)
    real = nisa.dma_copy

    def gated(*a, **k):
        if LOW_PRIO is not None and "priority" not in k and k.get("dge_mode") in (None, nisa.dge_mode.hwdge):
            k["priority"] = LOW_PRIO
        ins = real(*a, **k)
        if gate is not None:
            _after(ins, gate)
        return ins

    nisa.dma_copy = gated
    try:
        return _block(stream, pre_mix, hca_fn_t, hca_base, hca_scale, attn_norm_w, hcf_fn_t, hcf_base, hcf_scale,
                      ffn_norm_w, wq_a4, wq_a_s, q_norm_w, wq_b4, wq_b_s, wkv4, wkv_s, kv_norm_w, wo_a4, wo_a_s, wo_b4,
                      wo_b_s, attn_sink, freqs, row_pos, write_slots, win_slots, cmp_slots, wv, topk_in, window_kv_cache,
                      compress_kv_cache, gate_w_t, gate_b, expert_lo, w1, s1, w2, s2, sh_w1, sh_s1, sh_w2, sh_s2,
                      groups_attn, groups_world, share, out_stream, ffn_pre, barrier, tap_out, ix, window_only, src,
                      S_in)
    finally:
        nisa.dma_copy = real


def _block(stream, pre_mix, hca_fn_t, hca_base, hca_scale, attn_norm_w, hcf_fn_t, hcf_base, hcf_scale, ffn_norm_w,
           wq_a4, wq_a_s, q_norm_w, wq_b4, wq_b_s, wkv4, wkv_s, kv_norm_w, wo_a4, wo_a_s, wo_b4, wo_b_s, attn_sink,
           freqs, row_pos, write_slots, win_slots, cmp_slots, wv, topk_in, window_kv_cache, compress_kv_cache,
           gate_w_t, gate_b, expert_lo, w1, s1, w2, s2, sh_w1, sh_s1, sh_w2, sh_s2, groups_attn, groups_world,
           share, out_stream, ffn_pre, barrier, tap_out, ix, window_only=False, src=None, S_in=None):
    pid = nl.program_id(0)
    L = share.setdefault("b1", {}) if share is not None else {}
    T, hc, dim = stream.shape
    assert T == T6 and hc == HC
    C = dim // KC
    NP = T6 * C
    rg_attn = ncc.ReplicaGroup([list(g) for g in groups_attn])
    rg_world = ncc.ReplicaGroup([list(g) for g in groups_world])
    eng_w = nisa.engine.sync if STATIC_Q == "sync" else nisa.engine.scalar
    consts(L, C)
    NS = 2 * H0
    sync = nisa.engine.sync

    # ---------------------------------------------------------------- static tensors
    # this block's attention-norm weight and hc_in(attn) constants: prefetched by the previous member
    # of this kind (`before_down`), else loaded first
    pre = L.setdefault("pre", {}).pop(id(wq_a_s), None)
    if pre is None:
        w60a = packed_w(attn_norm_w, C, BF, eng_w)
        hca = hc_consts(hca_fn_t, hca_base, hca_scale, C, eng_w) if pid == COEF_A else None
    else:
        w60a, hca = pre

    # ---------------------------------------------------------------- the stream
    # (the previous member of this kind's: SBUF, its input sum in PSUM; else the HBM input)
    if S_in is None and L.get("S_src") is stream:
        S, xin = L["S"], L["xin"]
    else:
        # (`S_in`: the packed stream already in SBUF -- an engram block's gated stream, `b1eng`)
        S = load_stream(stream, C) if S_in is None else S_in
        pre120 = nl.ndarray((PT, HC), dtype=F32, buffer=nl.sbuf)
        nisa.memset(pre120, 0.0)
        for h in range(2):
            for t in range(T6):
                nisa.dma_copy(dst=pre120[h * H0 + t * C : h * H0 + (t + 1) * C, :],
                              src=pre_mix.ap(pattern=[[0, C], [1, HC]], offset=t * HC), **_hw(sync))
        xin = x_sum(L, S, pre120, NP)

    if tap_out is not None:
        # the DSpark tap (r1_tap): the mean of the input stream's hc copies, fp32 (`mk_common.tap_tile`;
        # here 1/4 of each copy summed on the PE, exact products), core 1's job
        if pid == 1:
            if "tapw" not in L:
                q4 = nl.ndarray((PT, HC), dtype=F32, buffer=nl.sbuf)
                nisa.memset(q4, 1.0 / HC)
                L["tapw"] = wsum_stat(L, q4, NP)
            tp_ = nl.ndarray((PT, KC), dtype=F32, buffer=nl.psum)
            wsum_mm(tp_, L["tapw"][0:1], S, NP)
            tsb = nl.ndarray((PT, KC), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tsb[0:NP], src=tp_[0:NP])
            nisa.dma_copy(dst=tap_out.ap(pattern=[[KC, NP], [1, KC]], offset=0), src=tsb[0:NP], **OUT_DMA)

    # ---------------------------------------------------------------- hc_in(attn)
    # both cores the normed input; core COEF_A (the kv core) the coefficients: the mix behind its kv
    # GEMM, the tail (sigmoids, Sinkhorn) through the attention side's waits and the all-reduce's
    xn = x_norm(L, xin, attn_norm_w, C, eng_w, w60=w60a)
    st = {}
    tick_a = Ticker()

    def after_gemm():
        # the ffn sublayer's static tensors
        st["w60f"] = packed_w(ffn_norm_w, C, BF, eng_w)
        if pid == COEF_F:
            st["hcf"] = hc_consts(hcf_fn_t, hcf_base, hcf_scale, C, eng_w)
        # the MoE's static weights (router, shared expert)
        st["mst"] = b1moe.moe_static(pid, C, gate_w_t, sh_w1, sh_s1, sh_w2, sh_s2, eng_w, bufs=L, parts=MST_EARLY)
        if pid == COEF_A:
            box = []
            tick_a.gen, tick_a.box = hc_coef_gen(L, S, hca, C, box), box

    # ---------------------------------------------------------------- attention side -> AR-8
    cc_src = nl.ndarray((T6, dim), dtype=F32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((T6, dim), dtype=F32, buffer=nl.shared_hbm, name="cc_dst")
    A.attn_side(pid, xn, wq_a4, wq_a_s, q_norm_w, wq_b4, wq_b_s, wkv4, wkv_s, kv_norm_w, wo_a4, wo_a_s, wo_b4,
                wo_b_s, attn_sink, freqs, row_pos, write_slots, win_slots, cmp_slots, wv, topk_in, window_kv_cache,
                compress_kv_cache, cc_src, after_gemm=after_gemm, share=share, xn_packed=True, tick=tick_a,
                after_side=(lambda: tick_a(1)) if (MIX_AT_SIDE and pid == COEF_A) else None,
                ix=None if ix is None else dict(ix, rg=rg_attn, barrier=barrier), window_only=window_only,
                src=src)
    if "r" in MST_LATE_ATTN:
        # the router weight behind the attention side (its weights' SBUF free then; the hi | lo split
        # while the all-reduce is in flight)
        st["mst"].update(b1moe.moe_static(pid, C, gate_w_t, sh_w1, sh_s1, sh_w2, sh_s2, eng_w, bufs=L, parts="r"))
    barrier(cc_src)
    ar1 = ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg_attn, op=nl.add)
    pf_at = PREFETCH_AT_CORE.get(pid, PREFETCH_AT)
    if pf_at == "ar1":
        # (the DMA queues idle between the attention's weight loads and the experts')
        pd = A.prefetch_next(pid, wq_a_s, share)
        for d in (pd if isinstance(pd, list) else [pd]):
            if d is not None:
                _after(d, ar1)
    # (while the all-reduce is in flight: the coefficients finished and swapped, the shortcut's and
    # hc_out(attn)'s stream parts, the MoE's static weights)
    if pid == COEF_A:
        coef_a = tick_a.rest()
    else:
        coef_a = nl.ndarray((PT, 2 * HC + HC * HC), dtype=F32, buffer=nl.sbuf)
        nisa.memset(coef_a, 0.0)
    ca = coef_from(pid, COEF_A, coef_a, 31, after=ar1, bufs=L)
    sc_a = None if SHORTCUT_EXACT else shortcut(L, S, ca, NP)
    s2pre = hc_out_pre(L, S, ca, NP)

    # ---------------------------------------------------------------- hc_in(ffn), hc_out(attn), MoE
    a16 = load_packed(cc_dst, NP, sync)
    if pf_at == "ar1done":
        # (behind the attention all-reduce's result: the DMA queues idle until the experts' loads)
        a16_ld = _LP[0]
        pd = A.prefetch_next(pid, wq_a_s, share)
        for d in (pd if isinstance(pd, list) else [pd]):
            if d is not None:
                _after(d, a16_ld)
        st["pf"] = pd
    if SHORTCUT_EXACT:
        # (the reference's order: the bf16 stream first, its pre-weighted sum from it)
        S2 = hc_out_fin(s2pre, a16, NP)
        xn2 = x_norm(L, x_sum(L, S2, ca, NP), ffn_norm_w, C, eng_w, w60=st["w60f"])
    else:
        xn2 = x_norm(L, x_short(sc_a, a16, NP), ffn_norm_w, C, eng_w, w60=st["w60f"])
        S2 = hc_out_fin(s2pre, a16, NP)
    cc2_src = nl.ndarray((T6, dim), dtype=F32, buffer=nl.shared_hbm, name="cc2_src")
    cc2_dst = nl.ndarray((T6, dim), dtype=F32, buffer=nl.shared_hbm, name="cc2_dst")
    xq2, xs2, xT2 = A.stage_xn(None, dim, packed=xn2, want_t=True)
    tick_f = Ticker()

    def after_logits(last_router_mm=None):
        # core COEF_F: hc_in(ffn)'s mix behind the router (scheduled after its last matmul), the tail
        # through the MoE's waits and the all-reduce's
        if pid == COEF_F:
            box = []
            mixv = hc_mix(L, S2, st["hcf"], C, after=last_router_mm)
            tick_f.gen, tick_f.box = hc_tail(mixv, st["hcf"], C, box), box

    def before_down():
        # the next member of this kind's attention weights, attention-norm weight, hc_in(attn) constants
        last = A.prefetch_next(pid, wq_a_s, share) if pf_at == "down" else None
        nxt = L.get("pre_next", {}).get(id(wq_a_s))
        if nxt is not None:
            nw, fn, bs, scl = nxt
            key = id(L["attn_next"][id(wq_a_s)][1])
            L["pre"][key] = (packed_w(nw, C, BF, eng_w),
                             hc_consts(fn, bs, scl, C, eng_w) if pid == COEF_A else None)
        return last

    b1moe.moe(pid, T6, xq2, xs2, xT2, gate_w_t, gate_b, expert_lo, w1, s1, w2, sh_w1, sh_s1, sh_w2, sh_s2, cc2_src,
              SWIGLU_LIMIT, ROUTE_SCALE, TOPK_EXPERTS, static=st["mst"], after_logits=after_logits,
              before_down=before_down, tick=tick_f, bufs=L)
    barrier(cc2_src)
    ar2 = ncc.all_reduce(srcs=[cc2_src], dsts=[cc2_dst], replica_group=rg_world, op=nl.add)
    if PREFETCH_BY_AR2 and st.get("pf"):
        # (the prefetch issued within this block: else the scheduler sank it into the next block, whose
        # first GEMM then waited for the whole weight load)
        for d in (st["pf"] if isinstance(st["pf"], list) else [st["pf"]]):
            if d is not None:
                _after(ar2, d)
    if pf_at == "ar2":
        pd = A.prefetch_next(pid, wq_a_s, share)
        for d in (pd if isinstance(pd, list) else [pd]):
            if d is not None:
                _after(d, ar2)
    if PREF_PIN and id(wq_a_s) in L.get("attn_next", {}):
        Wn = L.get("pref", {}).get(id(L["attn_next"][id(wq_a_s)][1]))
        if Wn:
            for wt_, ws_ in Wn.values():
                tch = nl.ndarray((1, 2), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=tch[0:1, 0:1], src=wt_.view(nl.int32)[0:1, 0, 0:1], engine=nisa.engine.gpsimd)
    if pid == COEF_F:
        coef_f = tick_f.rest()
    else:
        coef_f = nl.ndarray((PT, 2 * HC + HC * HC), dtype=F32, buffer=nl.sbuf)
        nisa.memset(coef_f, 0.0)
    cf = coef_from(pid, COEF_F, coef_f, 32, after=ar2, bufs=L)
    sc_f = None if SHORTCUT_EXACT else shortcut(L, S2, cf, NP)
    s3pre = hc_out_pre(L, S2, cf, NP)

    # ---------------------------------------------------------------- the next block's input, hc_out(ffn)
    f16 = load_packed(cc2_dst, NP, sync)
    L["gate"] = _LP[0]
    if SHORTCUT_EXACT:
        S3 = hc_out_fin(s3pre, f16, NP)
        xin_next = x_sum(L, S3, cf, NP)
    else:
        xin_next = x_short(sc_f, f16, NP)
        S3 = hc_out_fin(s3pre, f16, NP)
    # the HBM outputs (this core's rows): the stream, the next sublayer's pre
    r0 = pid * RH
    for t in range(r0, r0 + RH):
        for h in range(2):
            for s_ in range(2):
                nisa.dma_copy(dst=out_stream.ap(pattern=[[KC, C], [1, KC]], offset=t * HC * dim + (2 * s_ + h) * dim),
                              src=S3[h * H0 + t * C : h * H0 + (t + 1) * C, s_, :], **OUT_DMA)
    for t in range(r0, r0 + RH):
        nisa.dma_copy(dst=ffn_pre[t : t + 1, :], src=cf[t * C : t * C + 1, 0:HC], **OUT_DMA)
    L["S"], L["xin"], L["S_src"] = S3, xin_next, out_stream
    # (True: the next member is of this kind and reads the stream from SBUF: no output barrier)
    return id(wq_a_s) in L.get("next_b1", ())
