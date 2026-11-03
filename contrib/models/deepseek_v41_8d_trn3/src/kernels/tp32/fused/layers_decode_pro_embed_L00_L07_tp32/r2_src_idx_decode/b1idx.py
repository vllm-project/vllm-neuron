# SPDX-License-Identifier: Apache-2.0
"""The indexer of an index-source block for one 6-row span, on chip (round 5): the top-k of
compressed keys `b1attn.attn_side` attends computed in the block (the r1_idx members) instead of
read from an earlier index source.

    qr (staged, both cores) -> qi = qr @ ix_wq_b, rope, fp4 round trip     xn -> w = bf16(xn @ wproj) * scale
    scores[t, j] = sum_h relu(qi[t, h] . k_j) w[t, h]   (this core: half of the keys, all 6 rows)
    -- AR-8 of the scores --
    mask j >= clen, bf16, top-k (this core: its 3 rows) -> topk_out, the attention's key tables

Same arithmetic as `mk_compress` (`index_scores_span_static` / `index_topk`): bf16 PE scores into fp32
PSUM, ReLU, the fp32 head-weighted sum on the PE, the bf16 rounding of the masked scores, the GpSimd
top-k of the 16-partition snake layout, -1 for picks at or beyond clen. The attention's key tables
(`b1attn.group_keys`'s, in the same pick order) come from the picks on chip: the slot lookup by
`nc_n_gather`, the gather columns and the validity bias by 0/1 matmuls (fp32: exact)."""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from . import b1attn as A
from .mk_common import NEG_BOUND, PT, ROPE_DIM
from .mk_compress import FP4_MAX, RNE_MAGIC, fp4_qdq_tile
from .. import moe_tp

F32 = nl.float32
BF = nl.bfloat16
T6, RH, HEADS, HD, DC = A.T6, A.RH, A.HEADS, A.HD, A.DC
#: the snake layout's partitions a row (`nisa.topk`), the fp4 block of the index queries
SN = 16
QBLK = 32
#: the attended keys gathered one row's 4 tiles a DMA (2-D index tile, `gather_rows_batched`'s
#: column-major order) instead of one DMA a tile (SWDGE: ~1.1 us a gather whatever its size)
GATHER_G4 = True
#: the index queries' rope / fp4 round trip on 128 partitions (row t's 32-column block j of head h on
#: partition 32 j + 6 h + t) instead of on the 6 row partitions (~25 dependent DVE / ACT instructions
#: over 512 columns each)
QI_SPREAD = True


def _hw(eng):
    return dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)


def usable(topk_in, cmp_slots, wproj, keys):
    """The shapes this path handles (else the generic body's indexer); `topk_in`: the block's top-k
    output (`[T, min(index_topk, read_len)]`)."""
    read_len = cmp_slots.shape[0]
    nh = wproj.shape[1]
    k = topk_in.shape[1]
    return (k % PT == 0 and 0 < k <= 4 * PT and keys.shape[1] == PT and read_len % PT == 0 and read_len >= 2 * PT
            and read_len >= k and nh * T6 <= PT)


def key_split(pid, read_len):
    """`(k0, nk)`: this core's keys (whole 128-key tiles, core 0 the larger half)."""
    nb = read_len // PT
    nb0 = (nb + 1) // 2
    return (0, nb0 * PT) if pid == 0 else (nb0 * PT, (nb - nb0) * PT)


def _consts(B, pid):
    """Shape-only 0/1 tiles (fp32), built once a launch: `I6 [6, 6]`, `M48[p, q] = (p // 16 == q)`,
    `D48[p, r] = (p % 16 == r)`, `Sel[p, g, m] = (m // 16 == g) D48[p, m % 16]`, `R24[p, (r, h)] =
    M48[p, r]`."""
    key = ("ixc", pid)
    if key in B:
        return B[key]
    q48 = RH * SN
    G = PT // SN
    I6 = nl.ndarray((PT, T6), dtype=F32, buffer=nl.sbuf)
    moe_tp._onehot(I6, T6, [[1, T6]])
    M48 = nl.ndarray((PT, RH), dtype=F32, buffer=nl.sbuf)
    moe_tp._range_onehot(M48, q48, RH, [[-SN, RH]], 1, 0, SN)
    E = nl.ndarray((PT, SN * RH), dtype=F32, buffer=nl.sbuf)
    moe_tp._onehot(E, q48, [[1, SN], [SN, RH]])
    D48 = nl.ndarray((PT, SN), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=D48[0:q48], op=nl.add, data=E.ap(pattern=[[SN * RH, q48], [RH, SN], [1, RH]], offset=0), axis=(2,))
    Sel = nl.ndarray((PT, G, PT), dtype=F32, buffer=nl.sbuf)
    nisa.memset(Sel, 0.0)
    nisa.tensor_copy(dst=Sel.ap(pattern=[[G * PT, q48], [PT + SN, G], [1, SN]], offset=0),
                     src=D48.ap(pattern=[[SN, q48], [0, G], [1, SN]], offset=0))
    R24 = nl.ndarray((PT, RH, HEADS), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=R24[0:q48], src=M48.ap(pattern=[[RH, q48], [1, RH], [0, HEADS]], offset=0))
    # Sel4[p, s, a] = (cls(a) == s) (p % 16 == (a >> 2) & 15), cls(a) = 2 (a & 3) + (a >> 6): the
    # gather index tile of one row's 4-tile batch (`topk_tables`)
    NSC = 8
    ia = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ia[0:q48], pattern=[[1, PT]], offset=0, channel_multiplier=0)
    ra = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ra[0:q48], data=ia[0:q48], op0=nl.right_shift, operand0=2, op1=nl.bitwise_and, operand1=15)
    cl = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=cl[0:q48], data=ia[0:q48], op0=nl.bitwise_and, operand0=3, op1=nl.left_shift, operand1=1)
    hi = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi[0:q48], data=ia[0:q48], op0=nl.right_shift, operand0=6)
    nisa.tensor_tensor(dst=cl[0:q48], data1=cl[0:q48], data2=hi[0:q48], op=nl.add)
    rp = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=rp[0:q48], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=rp[0:q48], data=rp[0:q48], op0=nl.bitwise_and, operand0=15)
    raf = nl.ndarray((PT, PT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=raf[0:q48], src=ra[0:q48])
    clf = nl.ndarray((PT, PT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=clf[0:q48], src=cl[0:q48])
    rpf = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rpf[0:q48], src=rp[0:q48])
    e1 = nl.ndarray((PT, PT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e1[0:q48], data=raf[0:q48], op0=nl.equal, operand0=rpf[0:q48, 0:1])
    sv = nl.ndarray((PT, NSC, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=sv[0:q48], pattern=[[1, NSC], [0, PT]], offset=0, channel_multiplier=0)
    svf = nl.ndarray((PT, NSC, PT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=svf[0:q48], src=sv[0:q48])
    Sel4 = nl.ndarray((PT, NSC, PT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=Sel4[0:q48], data1=svf[0:q48], data2=clf.ap(pattern=[[PT, q48], [0, NSC], [1, PT]], offset=0),
                       op=nl.equal)
    nisa.tensor_tensor(dst=Sel4[0:q48], data1=Sel4[0:q48], data2=e1.ap(pattern=[[PT, q48], [0, NSC], [1, PT]], offset=0),
                       op=nl.multiply)
    B[key] = (I6, M48, D48, Sel, R24, Sel4)
    return B[key]


def _keys(pid, B, keys, cmp_slots, k0, nk, cache=True):
    """This core's index keys (cache rows `cmp_slots[k0 : k0 + nk]`) transposed, `[128 (d), nk]` bf16
    (the same for every index source of the launch reading this cache; `cache=False`: a kv source's,
    gathered after it wrote this step's keys)."""
    key = ("ik", id(keys), id(cmp_slots), pid)
    if cache and key in B:
        return B[key]
    nb = nk // PT
    kd = keys.shape[1]
    ks = nl.ndarray((PT, nb), dtype=nl.int32, buffer=nl.sbuf)
    for b in range(nb):
        nisa.dma_copy(dst=ks[:, b : b + 1], src=cmp_slots[k0 + b * PT : k0 + (b + 1) * PT, 0:1])
    kr = nl.ndarray((PT, nb, kd), dtype=BF, buffer=nl.sbuf)
    for b in range(nb):
        nisa.dma_copy(dst=kr[:, b, :], src=keys.ap(pattern=[[kd, PT], [1, kd]], vector_offset=ks[0:PT, b : b + 1],
                                                   indirect_dim=0))
    kT = nl.ndarray((PT, nk), dtype=BF, buffer=nl.sbuf)
    for b0 in range(0, nb, 4):
        n = min(4, nb - b0)
        tp = nl.ndarray((PT, 4, PT), dtype=BF, buffer=nl.psum)
        for b in range(n):
            nisa.nc_transpose(dst=tp[:, b, :], data=kr[:, b0 + b, :])
        nisa.tensor_copy(dst=kT[:, b0 * PT : (b0 + n) * PT], src=tp[:, 0:n, :])
    if cache:
        B[key] = kT
    return kT


def _rows_tabs(pid, B, cmp_slots, clen):
    """Launch tables of this core's rows: the slot table on the snake rows' partitions `[48, read_len]`
    int32, clen on them `[48, 1]` and as a row on every partition `[128, 3]` (fp32)."""
    key = ("ixr", id(cmp_slots), id(clen), pid)
    if key in B:
        return B[key]
    q48 = RH * SN
    read_len = cmp_slots.shape[0]
    eng = nisa.engine.sync
    stab = nl.ndarray((PT, read_len), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=stab[0:q48, :], src=cmp_slots.ap(pattern=[[0, q48], [1, read_len]], offset=0), **_hw(eng))
    ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    for q in range(RH):
        nisa.dma_copy(dst=ci[q * SN : (q + 1) * SN, :], src=clen.ap(pattern=[[0, SN], [1, 1]], offset=pid * RH + q),
                      **_hw(eng))
    c48 = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=c48[0:q48], src=ci[0:q48])
    cr = nl.ndarray((PT, RH), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cr, src=clen.ap(pattern=[[0, PT], [1, RH]], offset=pid * RH), **_hw(eng))
    crf = nl.ndarray((PT, RH), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=crf, src=cr)
    B[key] = (stab, c48, crf)
    return B[key]


def _qi_consts(pid, B, freqs, row_pos, nh):
    """`QI_SPREAD`'s tables (once a launch): the spread `E1[t, b, m] = (m == 32 (b // nh) + 6 (b % nh) + t)`
    (bf16, b = j * nh + h), the block mask `bm[p, j] = (p // 32 == j)` and `M24[p, n] = (p % 32 == n)`
    (bf16), the rope (cos | sin) of each partition's 16 pairs `[128, 32]` fp32 (1 | 0 off the rope)."""
    key = ("ixq", id(freqs), id(row_pos), nh, pid)
    if key in B:
        return B[key]
    nb = 4 * nh
    v = nl.ndarray((PT, nb * PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=v.ap(pattern=[[nb * PT, T6], [nh * PT, 4], [PT, nh], [1, PT]], offset=0), pattern=[[-32, 4], [-T6, nh], [1, PT]], offset=0,
              channel_multiplier=-1)
    vf = nl.ndarray((PT, nb * PT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vf[0:T6], src=v[0:T6])
    E1 = nl.ndarray((PT, nb, PT), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=E1.ap(pattern=[[nb * PT, T6], [1, nb * PT]], offset=0), data=vf[0:T6], op0=nl.equal, operand0=0.0)
    bmf = nl.ndarray((PT, 4), dtype=F32, buffer=nl.sbuf)
    moe_tp._range_onehot(bmf, PT, 4, [[-32, 4]], 1, 0, 32)
    bm = nl.ndarray((PT, 4), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bm, src=bmf)
    Ef = nl.ndarray((PT, nh * T6 * 4), dtype=F32, buffer=nl.sbuf)
    moe_tp._onehot(Ef, PT, [[1, nh * T6], [32, 4]])
    M24f = nl.ndarray((PT, nh * T6), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=M24f, op=nl.add, data=Ef.reshape((PT, nh * T6, 4)), axis=(2,))
    M24 = nl.ndarray((PT, nh * T6), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=M24, src=M24f)
    # the rope partitions (blocks j = 2, 3: the heads' last 64 columns): the rows' positions, their
    # (cos | sin) rows gathered, each block's 16 pairs selected
    pos = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(pos, 0)
    for j in (2, 3):
        for h in range(nh):
            p0 = 32 * j + T6 * h
            nisa.dma_copy(dst=pos[p0 : p0 + T6, :], src=row_pos.ap(pattern=[[1, T6], [1, 1]], offset=0),
                          **_hw(nisa.engine.sync))
    fr = nl.ndarray((PT, 4 * SN), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=fr, src=freqs.ap(pattern=[[4 * SN, PT], [1, 4 * SN]], vector_offset=pos[0:PT, 0:1], indirect_dim=0))
    cs = nl.ndarray((PT, 2 * SN), dtype=F32, buffer=nl.sbuf)
    nisa.memset(cs[0:64, 0:SN], 1.0)
    nisa.memset(cs[0:64, SN : 2 * SN], 0.0)
    for j in (2, 3):
        o = (j - 2) * SN
        nisa.tensor_copy(dst=cs[32 * j : 32 * j + 32, 0:SN], src=fr[32 * j : 32 * j + 32, o : o + SN])
        nisa.tensor_copy(dst=cs[32 * j : 32 * j + 32, SN : 2 * SN], src=fr[32 * j : 32 * j + 32, 2 * SN + o : 2 * SN + o + SN])
    B[key] = (E1, bm, M24, cs)
    return B[key]


def _fp4_rows(x):
    """`mk_compress.fp4_qdq_tile(x, 128, 32, 32, False)` for one 32-element block a partition (`x`
    `[128, 32]` fp32): the same operations (power-of-two scale, clamp, half-even e2m1 rounding by the
    2^23 magic constant), the per-block values as per-partition scalars. Returns bf16 `[128, 32]`."""
    n = x.shape[1]
    I32 = nl.int32
    ab = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=x)
    am = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=am, op=nl.maximum, data=ab, axis=(1,), keepdims=True)
    nisa.tensor_scalar(dst=am, data=am, op0=nl.maximum, operand0=FP4_MAX * 2.0 ** -126, op1=nl.multiply,
                       operand1=1.0 / FP4_MAX)
    bits = am.view(I32)
    e = nl.ndarray((PT, 1), dtype=I32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e, data=bits, op0=nl.right_shift, operand0=23, op1=nl.bitwise_and, operand1=0xFF)
    bump = nl.ndarray((PT, 1), dtype=I32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bump, data=bits, op0=nl.bitwise_and, operand0=0x7FFFFF)
    nisa.tensor_scalar(dst=bump, data=bump, op0=nl.greater, operand0=0)
    nisa.tensor_tensor(dst=e, data1=e, data2=bump, op=nl.add)
    sc = nl.ndarray((PT, 1), dtype=I32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sc, data=e, op0=nl.left_shift, operand0=23)
    ie = nl.ndarray((PT, 1), dtype=I32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ie, data=e, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=254)
    nisa.tensor_scalar(dst=ie, data=ie, op0=nl.left_shift, operand0=23)
    # nrm = clamp(x / scale, -6, 6)
    nrm = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nrm, data=x, op0=nl.multiply, operand0=ie.view(F32)[:, 0:1])
    nisa.tensor_scalar(dst=nrm, data=nrm, op0=nl.maximum, operand0=-FP4_MAX, op1=nl.minimum, operand1=FP4_MAX)
    # e2m1: step = 2^(floor(log2(max(|n|, 1))) - 1)
    mag = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=mag, op=nl.abs, data=nrm)
    nisa.tensor_scalar(dst=mag, data=mag, op0=nl.maximum, operand0=1.0)
    E = nl.ndarray((PT, n), dtype=I32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=E, data=mag.view(I32), op0=nl.right_shift, operand0=23, op1=nl.bitwise_and, operand1=0xFF)
    stp = nl.ndarray((PT, n), dtype=I32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=stp, data=E, op0=nl.add, operand0=-1)
    nisa.tensor_scalar(dst=stp, data=stp, op0=nl.left_shift, operand0=23)
    istp = nl.ndarray((PT, n), dtype=I32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=istp, data=E, op0=nl.multiply, operand0=-1, op1=nl.add, operand1=255)
    nisa.tensor_scalar(dst=istp, data=istp, op0=nl.left_shift, operand0=23)
    y = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=nrm, data2=istp.view(F32), op=nl.multiply)
    # (two instructions: fused, the add and the subtract cancel unrounded)
    nisa.tensor_scalar(dst=y, data=y, op0=nl.add, operand0=RNE_MAGIC, engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=y, data=y, op0=nl.subtract, operand0=RNE_MAGIC, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=y, data1=y, data2=stp.view(F32), op=nl.multiply)
    out = nl.ndarray((PT, n), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=out, data=y, op0=nl.multiply, operand0=sc.view(F32)[:, 0:1])
    return out


def _qi_spread(q16, qc, nh):
    """`q16 [6, nh * 128]` (bf16 rows) -> the rope'd fp4 round-tripped queries transposed, `[128 (d),
    nh * 6 ((h, t))]` bf16, on 128 partitions: spread (16 one-hot matmuls), rope, fp4, gathered back
    (one block-masked matmul). Exact moves (one nonzero product a sum)."""
    E1, bm, M24, cs = qc
    nb = 4 * nh
    sp = nl.ndarray((PT, 32), dtype=F32, buffer=nl.psum)
    for b in range(nb):
        j, h = b // nh, b % nh
        nisa.nc_matmul(dst=sp, stationary=E1[0:T6, b, :], moving=q16[0:T6, h * PT + 32 * j : h * PT + 32 * j + 32],
                       accumulate=(b > 0))
    t = nl.ndarray((PT, 32), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t, src=sp)
    # rope (`rope_cols`'s arithmetic): pairs (2 i, 2 i + 1), (cos | sin) per partition
    a = t.ap(pattern=[[32, PT], [2, SN]], offset=0)
    b_ = t.ap(pattern=[[32, PT], [2, SN]], offset=1)
    cos = cs[:, 0:SN]
    sin = cs[:, SN : 2 * SN]

    def mul(x, y):
        r = nl.ndarray((PT, SN), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=x, data2=y, op=nl.multiply)
        return r

    ac, bs, as_, bc = mul(a, cos), mul(b_, sin), mul(a, sin), mul(b_, cos)
    nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
    nisa.tensor_tensor(dst=b_, data1=as_, data2=bc, op=nl.add)
    qf = nl.ndarray((PT, 32), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qf, src=t)
    q4 = _fp4_rows(qf)
    stat = nl.ndarray((PT, 4, 32), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=stat, data1=q4.ap(pattern=[[32, PT], [0, 4], [1, 32]], offset=0),
                       data2=bm.ap(pattern=[[4, PT], [1, 4], [0, 32]], offset=0), op=nl.multiply)
    gp = nl.ndarray((PT, nh * T6), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=gp, stationary=stat.reshape((PT, PT)), moving=M24)
    qT = nl.ndarray((PT, nh * T6), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qT, src=gp)
    return qT


def prep(pid, B, ix, cmp_slots, q_lora):
    """The block-start part (issued before the projections): the index keys (once a launch), the
    tables, this block's indexer weights. Returns the state the later stages take."""
    read_len = cmp_slots.shape[0]
    k0, nk = key_split(pid, read_len)
    st = dict(ix)
    st["nk"], st["k0"], st["read_len"], st["B"] = nk, k0, read_len, B
    st["consts"] = _consts(B, pid)
    # (a kv source: its keys gathered once it wrote this step's, in `scores`)
    st["kT"] = None if ix.get("fresh") else _keys(pid, B, ix["keys"], cmp_slots, k0, nk)
    st["cmp_slots"] = cmp_slots
    st["tabs"] = _rows_tabs(pid, B, cmp_slots, ix["clen"])
    wp = ix["wproj"]
    nh = wp.shape[1]
    st["nh"] = nh
    st["wqb"] = A.load_w(ix["wq_b"], ix["wq_b_s"], q_lora, 0, nh * PT, nisa.engine.scalar)
    dim = wp.shape[0]
    nkc = dim // A.KC
    # wpT[p, c, j, h] = wproj[512 c + 4 p + j, h] (the staged xn's row order)
    wpT = nl.ndarray((PT, nkc, 4, nh), dtype=wp.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wpT, src=wp.ap(pattern=[[4 * nh, PT], [A.KC * nh, nkc], [1, 4 * nh]], offset=0),
                  dge_mode=nisa.dge_mode.none)
    st["wpT"] = wpT
    return st


def weights(st, xT):
    """The head weights' stationary from the staged xn (`stage_xn(want_t=True)`'s `xT [128, C, 6, 4]`):
    `Wst [nh * 6, 6]` fp32, `Wst[(h, t'), t] = (t == t') bf16(xn[t] . wproj[:, h]) * scale`."""
    nh = st["nh"]
    wpT = st["wpT"]
    nkc = wpT.shape[1]
    I6 = st["consts"][0]
    if wpT.dtype != BF:
        x32 = nl.ndarray((PT, nkc, T6, 4), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x32, src=xT)
        xT = x32
    ps = nl.ndarray((T6, nh), dtype=F32, buffer=nl.psum)
    i = 0
    for c in range(nkc):
        for j in range(4):
            nisa.nc_matmul(dst=ps, stationary=xT.ap(pattern=[[nkc * T6 * 4, PT], [4, T6]], offset=c * T6 * 4 + j),
                           moving=wpT[:, c, j, :], accumulate=(i > 0))
            i += 1
    w16 = nl.ndarray((T6, nh), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w16, src=ps)
    w = nl.ndarray((T6, nh), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w, data=w16, op0=nl.multiply, operand0=st["scale"])
    wt = nl.ndarray((T6, nh, T6), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=wt, data1=w.ap(pattern=[[nh, T6], [1, nh], [0, T6]], offset=0),
                       data2=I6.ap(pattern=[[T6, T6], [0, nh], [1, T6]], offset=0), op=nl.multiply)
    tp = nl.ndarray((nh * T6, T6), dtype=F32, buffer=nl.psum)
    nisa.nc_transpose(dst=tp, data=wt.reshape((T6, nh * T6)))
    Wst = nl.ndarray((PT, T6), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Wst[0 : nh * T6], src=tp)
    st["Wst"] = Wst


def scores(pid, st, qrq, qrs, qparts, cs6, rg_attn, barrier, freqs=None, row_pos=None):
    """The index queries from the staged qr, this core's half of every row's scores into the scores'
    all-reduce (issued here): `st["sc_dst"]` `[6, read_len]` fp32 (HBM) once it completes."""
    nh = st["nh"]
    n = nh * PT
    nq = nh * T6
    qp = A.mx_gemm(qrq, qrs, T6, T6, qparts, *st["wqb"], 0, n)
    q16 = nl.ndarray((T6, n), dtype=BF, buffer=nl.sbuf)
    A.evac(q16, qp)
    if QI_SPREAD and freqs is not None:
        qT2 = _qi_spread(q16, _qi_consts(pid, st["B"], freqs, row_pos, nh), nh)
    else:
        A.rope_cols(q16, T6, PT - ROPE_DIM, PT, nh, cs6, inverse=False)
        qf = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qf[0:T6], src=q16)
        q4 = fp4_qdq_tile(qf, T6, n, QBLK, False)
        tq = nl.ndarray((PT, nh, T6), dtype=BF, buffer=nl.psum)
        for h in range(nh):
            nisa.nc_transpose(dst=tq[:, h, :], data=q4[0:T6, h * PT : (h + 1) * PT])
        qT = nl.ndarray((PT, nh, T6), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qT, src=tq)
        qT2 = qT.reshape((PT, nq))
    nk, k0, kT, Wst = st["nk"], st["k0"], st["kT"], st["Wst"]
    if kT is None:
        kT = _keys(pid, st["B"], st["keys"], st["cmp_slots"], k0, nk, cache=False)
    sc = nl.ndarray((T6, nk), dtype=F32, buffer=nl.sbuf)
    for i, b0 in enumerate(range(0, nk, 512)):
        w = min(512, nk - b0)
        sp = nl.ndarray((nq, 512), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=sp[:, 0:w], stationary=qT2, moving=kT[:, b0 : b0 + w])
        rect = nl.ndarray((nq, 512), dtype=F32, buffer=nl.sbuf)
        if i % 2 == 0:
            nisa.tensor_scalar(dst=rect[:, 0:w], data=sp[:, 0:w], op0=nl.maximum, operand0=0.0)
        else:
            nisa.activation(dst=rect[:, 0:w], op=nl.relu, data=sp[:, 0:w])
        wps = nl.ndarray((T6, 512), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=wps[:, 0:w], stationary=Wst[0:nq, 0:T6], moving=rect[:, 0:w])
        nisa.tensor_copy(dst=sc[:, b0 : b0 + w], src=wps[:, 0:w], engine=nisa.engine.scalar if i % 2 else nisa.engine.vector)
    read_len = st["read_len"]
    sc_src = nl.ndarray((T6, read_len), dtype=F32, buffer=nl.shared_hbm, name="ix_sc_src")
    sc_dst = nl.ndarray((T6, read_len), dtype=F32, buffer=nl.shared_hbm, name="ix_sc_dst")
    nisa.dma_copy(dst=sc_src[0:T6, k0 : k0 + nk], src=sc, **_hw(nisa.engine.sync))
    barrier(sc_src)
    st["ar"] = ncc.all_reduce(srcs=[sc_src], dsts=[sc_dst], replica_group=rg_attn, op=nl.add,
                              priority=moe_tp.AR_PRIORITY)
    st["sc_dst"] = sc_dst


def topk_tables(pid, st, compress_kv_cache):
    """This core's rows' top-k from the reduced scores (`index_topk`'s), stored to `st["out"]`, and the
    attention's key tables of its rows `(vc, kTc, cb24)` (`b1attn.group_keys`'s, the same pick order)."""
    read_len = st["read_len"]
    topk = st["out"].shape[1]
    sc_dst = st["sc_dst"]
    stab, c48, crf = st["tabs"]
    I6, M48, D48, Sel, R24, Sel4 = st["consts"]
    n16 = read_len // SN
    k16 = topk // SN
    q48 = RH * SN
    G = PT // SN
    nct = topk // PT
    nq = RH * HEADS
    sync = nisa.engine.sync
    # 1. masked bf16 scores in the snake layout: snake[16 q + r, c] = s[q, 16 c + r] (this core's rows q)
    snake = nl.ndarray((PT, n16), dtype=BF, buffer=nl.sbuf)
    nisa.memset(snake, NEG_BOUND)
    for c0 in range(0, n16, PT):
        pc = min(PT, n16 - c0)
        m = nl.ndarray((PT, RH, SN), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m[0:pc], src=sc_dst.ap(pattern=[[SN, pc], [read_len, RH], [1, SN]],
                                                 offset=pid * RH * read_len + c0 * SN), dge_mode=nisa.dge_mode.none)
        j = nl.ndarray((PT, RH, SN), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=j[0:pc], pattern=[[0, RH], [1, SN]], offset=SN * c0, channel_multiplier=SN)
        jf = nl.ndarray((PT, RH, SN), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jf[0:pc], src=j[0:pc])
        live = nl.ndarray((PT, RH, SN), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=live[0:pc], data1=jf[0:pc], data2=crf.ap(pattern=[[RH, pc], [1, RH], [0, SN]], offset=0),
                           op=nl.less)
        nisa.tensor_tensor(dst=m[0:pc], data1=m[0:pc], data2=live[0:pc], op=nl.multiply)
        dead = nl.ndarray((PT, RH, SN), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dead[0:pc], data=live[0:pc], op0=nl.add, operand0=-1.0, op1=nl.multiply,
                           operand1=-NEG_BOUND)
        nisa.tensor_tensor(dst=m[0:pc], data1=m[0:pc], data2=dead[0:pc], op=nl.add)
        s16 = nl.ndarray((PT, RH * SN), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s16[0:pc], src=m.ap(pattern=[[RH * SN, pc], [1, RH * SN]], offset=0))
        tp = nl.ndarray((PT, PT), dtype=BF, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:q48, 0:pc], data=s16[0:pc, 0:q48])
        nisa.tensor_copy(dst=snake[0:q48, c0 : c0 + pc], src=tp[0:q48, 0:pc])
    # 2. the top-k: pick 16 c + r of row q on [16 q + r, c]; -1 at or beyond clen
    vals = nl.ndarray((PT, topk), dtype=BF, buffer=nl.sbuf)
    idx = nl.ndarray((PT, topk), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.topk(val_dst=vals, idx_dst=idx, src=snake, n=read_len, sorted=False)
    X = nl.ndarray((PT, k16), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=X[0:q48], src=idx[0:q48, 0:k16])
    lv = nl.ndarray((PT, k16), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lv[0:q48], data=X[0:q48], op0=nl.less, operand0=c48[0:q48, 0:1])
    nisa.tensor_tensor(dst=X[0:q48], data1=X[0:q48], data2=lv[0:q48], op=nl.multiply)
    nisa.tensor_scalar(dst=lv[0:q48], data=lv[0:q48], op0=nl.add, operand0=-1.0)
    nisa.tensor_tensor(dst=X[0:q48], data1=X[0:q48], data2=lv[0:q48], op=nl.add)
    # 3. the slot lookup (on the snake partitions), the validity bias
    Xc = nl.ndarray((PT, k16), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=Xc[0:q48], data=X[0:q48], op0=nl.maximum, operand0=0.0)
    Xu = nl.ndarray((PT, k16), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Xu[0:q48], src=Xc[0:q48])
    sl = nl.ndarray((PT, k16), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=sl[0:q48, :], data=stab[0:q48, :], indices=Xu[0:q48, :])
    slf = nl.ndarray((PT, k16), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slf[0:q48], src=sl[0:q48])
    nrc = RH * nct
    vc = nl.ndarray((PT, nrc, HD), dtype=BF, buffer=nl.sbuf)
    if GATHER_G4 and nct == 4:
        # 4. one row's 4 tiles a gather: vc[p, 4 q + g] = row q's pick 128 g + p; the index tile read
        #    column-major (flat k = 4 p + g at [k % 128, k // 128]): J[a, (q, b)] = slot of row q's pick
        #    128 (a & 3) + 32 b + (a >> 2) = snake [16 q + ((a >> 2) & 15), 8 (a & 3) + (a >> 6) + 2 b]
        #    (sum over the snake partitions p of Sel4[p, cls(a), a] x slf[p, 8 x + y + 2 b] (p // 16 == q))
        J = nl.ndarray((PT, RH * nct), dtype=F32, buffer=nl.psum)
        i = 0
        for y in range(2):
            mv = nl.ndarray((PT, 4, RH, nct), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mv[0:q48], data1=slf.ap(pattern=[[k16, q48], [8, 4], [0, RH], [2, nct]], offset=y),
                               data2=M48.ap(pattern=[[RH, q48], [0, 4], [1, RH], [0, nct]], offset=0), op=nl.multiply)
            for x in range(4):
                nisa.nc_matmul(dst=J, stationary=Sel4.ap(pattern=[[8 * PT, q48], [1, PT]], offset=(2 * x + y) * PT),
                               moving=mv.ap(pattern=[[4 * RH * nct, q48], [1, RH * nct]], offset=x * RH * nct),
                               accumulate=(i > 0))
                i += 1
        Ji = nl.ndarray((PT, RH * nct), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Ji, src=J)
        for q in range(RH):
            nisa.dma_copy(dst=vc[:, q * nct : (q + 1) * nct, :],
                          src=compress_kv_cache.ap(pattern=[[HD, PT * nct], [1, HD]], offset=0,
                                                   vector_offset=Ji[0:PT, q * nct : (q + 1) * nct], indirect_dim=0))
    else:
        # 4. the gather columns: ixc[16 g + r, q * 4 + c] = slot of pick 128 c + 16 g + r of row q
        #    (sum over the snake partitions p of Sel[p, g, :] x slf[p, 8 c + g] (p // 16 == q))
        mov = nl.ndarray((PT, G, RH, nct), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mov[0:q48], data1=slf.ap(pattern=[[k16, q48], [1, G], [0, RH], [G, nct]], offset=0),
                           data2=M48.ap(pattern=[[RH, q48], [0, G], [1, RH], [0, nct]], offset=0), op=nl.multiply)
        ixp = nl.ndarray((PT, RH * nct), dtype=F32, buffer=nl.psum)
        for g in range(G):
            nisa.nc_matmul(dst=ixp, stationary=Sel.ap(pattern=[[G * PT, q48], [1, PT]], offset=g * PT),
                           moving=mov.ap(pattern=[[G * RH * nct, q48], [1, RH * nct]], offset=g * RH * nct),
                           accumulate=(g > 0))
        ixc = nl.ndarray((PT, RH * nct), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ixc, src=ixp)
        # 5. the keys (`group_keys`'s gathers and transposes)
        for i in range(nrc):
            nisa.dma_copy(dst=vc[:, i, :], src=compress_kv_cache.ap(pattern=[[HD, PT], [1, HD]],
                                                                    vector_offset=ixc[0:PT, i : i + 1], indirect_dim=0))
    # the validity bias on the (row, head) partitions: cb24[(r, h), 16 c + r'] = NEG (pick < 0)
    bias = nl.ndarray((PT, k16), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bias[0:q48], data=X[0:q48], op0=nl.less, operand0=0.0, op1=nl.multiply, operand1=A.NEG)
    mv2 = nl.ndarray((PT, k16, SN), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mv2[0:q48], data1=bias.ap(pattern=[[k16, q48], [1, k16], [0, SN]], offset=0),
                       data2=D48.ap(pattern=[[SN, q48], [0, k16], [1, SN]], offset=0), op=nl.multiply)
    cbp = nl.ndarray((nq, topk), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=cbp, stationary=R24.ap(pattern=[[nq, q48], [1, nq]], offset=0),
                   moving=mv2.ap(pattern=[[topk, q48], [1, topk]], offset=0))
    cb24 = nl.ndarray((PT, topk), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cb24[0:nq], src=cbp, engine=nisa.engine.scalar)
    kTc = nl.ndarray((PT, DC, RH, topk), dtype=BF, buffer=nl.sbuf)
    for r in range(RH):
        for c in range(nct):
            tps = nl.ndarray((PT, DC, PT), dtype=BF, buffer=nl.psum)
            for dc in range(DC):
                nisa.nc_transpose(dst=tps[:, dc, :], data=vc[:, r * nct + c, dc * PT : (dc + 1) * PT])
            nisa.tensor_copy(dst=kTc[:, :, r, c * PT : (c + 1) * PT], src=tps)
    # 6. topk_out (this core's rows; HBM order: [q, 16 c + r] = pick on [16 q + r, c])
    tpo = nl.ndarray((PT, q48), dtype=F32, buffer=nl.psum)
    nisa.nc_transpose(dst=tpo[0:k16, 0:q48], data=X[0:q48, 0:k16])
    oi = nl.ndarray((PT, q48), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=oi[0:k16], src=tpo[0:k16])
    nisa.dma_copy(dst=st["out"].ap(pattern=[[SN, k16], [topk, RH], [1, SN]], offset=pid * RH * topk),
                  src=oi.ap(pattern=[[q48, k16], [SN, RH], [1, SN]], offset=0))
    return vc, kTc, cb24
