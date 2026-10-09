"""`layers.24.attention` on 4 ranks: a sequence-parallel split, with every rank holding all 64 heads.

A rank owns a quarter of the **sequence** and the whole of every weight, rather than a quarter of
the heads. `wo_b` stops being row-parallel, so a rank writes *finished* rows of the module's output
and the four ranks rejoin by `all_gather` instead of `all_reduce`.

**The tiles a rank owns are interleaved, and that is what makes one NEFF schedule four ranks
identically.** Global `PT`-row query tile `WORLD_SIZE * j + rank` is this rank's local tile `j`.
Global tile `g` reaches `g // WORLD_SIZE + 1` blocks of `NT` compressed positions, so with
`g = WORLD_SIZE * j + rank` and `rank < WORLD_SIZE` that block count is `j + 1` for *every* rank -- a
trace-time constant. Only the valid width inside the last block is rank-dependent,
`(rank + 1) * PT` of `NT`, and that is a runtime causal compare rather than a trace constant. A
contiguous quarter would instead make the block count itself rank-dependent, forcing the one shared
NEFF to schedule every tile at rank 3's reach; the interleave pays only for last blocks evaluated at
full `NT` width, about 4.6% of the attention area, and buys a 4x cut in the replicated prefix and in
the indexer's relu'd QK.

A rank's identity lives entirely in the values `inference.py` hands this kernel, never in the code.
It reaches the hardware in exactly three places, all through `ncc.rank_id()`: the staging DMAs that
pull this rank's `x` and `freqs_cis` tiles down to a contiguous local range (`_rtile`), the causal
compare on the last block of the index score, and the window-KV read. Everything else indexes
trace-time constants -- including the copy that undoes the gather's rank-major concatenation, since
after an `all_gather` every slot's provenance is known at trace time.

Three collectives, all `all_gather` with `collective_dim=0` (the only value HBM operands support):

1. the **window KV latent**, because local tile `j`'s window reaches into global tile
   `WORLD_SIZE * j + rank - 1`, which belongs to another rank -- 8.4 MB, before the attention;
2. and 3. the **output**, in two halves of its columns, 10.5 MB in and 42 MB out a rank each. The
   halves are separate so the first one's collective is issued before the second half's `wo_b`
   GEMM and runs underneath it: unsplit, the gather and the copy that undoes the tile interleave
   were a 565 us tail with every compute engine idle, because neither can start until the GEMM
   before it has finished.

Measured, all three together cost 0.341 ms -- 1.5% of the wall, and less than the single 83.9 MB
`all_reduce` of the head-split version they replaced. On this module a collective is not what
needs optimizing; redundant compute is.

Both obey the rules the `all_reduce` this replaced had to obey, each of which otherwise fails with an
internal compiler error naming something else: `kernel[2]` because this host is LNC=2 and a LNC=1
NEFF leaves the collective's buffer unallocated on the second physical core (`NCC_ILLC059`); `name=`
on every operand or DRAM allocation fails (`NCC_IBIR440`); operands are `nl.shared_hbm` scratch and
never an IO tensor (`NCC_INLA001`), hence the copy out -- plain DMA is not restricted this way, which
is what makes the scalar-indirect staging reads of the IO tensors legal; buffer kinds may not be
mixed; the `ReplicaGroup` is built by `inference.py` because the tracer rejects `range` in a traced
body; and an operand must be a whole buffer of exactly the collective's shape, not a view of
something larger, which is why `cc_src_a`/`cc_dst_a`/`cc_src_b`/`cc_dst_b`/`kv_src`/`kv_dst` are
dedicated.

The top-`INDEX_TOPK` selection is `nisa.topk` on the GpSimd engine rather than a bisection on the
Vector engine. `_index_mask` documents the snake layout it needs, why any bijection onto that snake
is as good as the documented one when only the threshold is wanted, and why the threshold is read as
the minimum of a `+inf`-prefilled `val_dst` rather than from one slot.

One more, not about collectives: `inference.py` sets `NKI_COMPILER_FRONTEND=tracer`. NKI's default
frontend rejects the list and dict comprehensions `_core_runs`' static schedule is built out of.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

SEQ = 8192
DIM = 5120
HEAD_DIM = 512
# A rank now holds *all* 64 query heads and a quarter of the sequence, not a quarter of the heads.
N_HEADS = 64
ROPE_DIM = 64
Q_LORA = 1280
O_LORA = 1024
O_GROUPS = 8
WINDOW = 128
RATIO = 1
CSEQ = SEQ // RATIO
CACHE_ROWS = 4 * 16384
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 512
EPS = 1e-20
INDEX_SCORE_SCALE = 1.0 / 64.0
NEG_INF = -1.0e30
SM_SCALE = HEAD_DIM ** -0.5
NEG_BOUND = -1.0e30
HEADS_PER_GROUP = N_HEADS // O_GROUPS
OB = 4

PT = 128
NT = 512
WIN_SPAN = 2 * PT
CHUNK = 1

FP8_BLK = 32
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4
LN2 = 0.6931471805599453
QDQ_CHUNK = 1280

MGROUP_BYTES = 96 * 1024

CORES = 2
QG = NT
NG = SEQ // QG

# --- the sequence split -------------------------------------------------------------------------
# A rank owns SEQ // WORLD_SIZE query rows, taken as every fourth `PT`-row tile: global tile
# `WORLD_SIZE * j + rank` is this rank's local tile `j`. That interleave, and not a contiguous
# quarter, is what keeps the four ranks' *schedules* identical while their work stays balanced.
# Global tile `g` reaches `g * PT + PT` compressed positions, i.e. `g // WORLD_SIZE + 1` blocks of
# `NT`; with `g = WORLD_SIZE * j + rank` and `rank < WORLD_SIZE` that count is `j + 1` for every
# rank -- a trace-time constant. Only the *valid width inside the last block* is rank-dependent
# (`(rank + 1) * PT` of `NT`), and that is handled by a runtime causal compare rather than by the
# trace. A contiguous quarter would instead make the block count itself rank-dependent, forcing the
# one shared NEFF to schedule every tile at rank 3's reach.
SEQ_LOC = SEQ // 4
MT_LOC = SEQ_LOC // PT
# `nisa.topk`'s selection count, and the `NEG_INF` clamp that keeps a causally-masked position out
# of the mask when a query has fewer than `INDEX_TOPK` candidates and the threshold is `NEG_INF`.
THR_FLOOR = -1.0e29

# The four ranks of the projected floorplan. `N_HEADS`, `O_GROUPS` and `attn_sink`'s width above
# are already this rank's share; these constants are what turn the whole recorded weight into that
# share. Every shard boundary -- 16 heads, 2048 rows, 2048 columns, 8192 rows -- is a multiple of
# the 32-wide fp8 scale block, so no block straddles two ranks.
WORLD_SIZE = 4
# The module's output dtype: the recorded output's own, so the chain rounds to bfloat16 exactly
# once, at the end, however the partials are carried.
OUT_DTYPE = nl.bfloat16
# The dtype a rank's `wo_b` partial is written in and the four partials are reduced in. Both
# settings were measured on this host and `ASSEMBLY.md` has the table; bfloat16 ships because the
# collective is bandwidth-bound at this size, so the dtype is the whole lever on it.
CC_DTYPE = nl.bfloat16
# Rows per DMA of the post-collective copy. 1024 x 5120 bfloat16 = 10 MiB a descriptor.
CC_ROWS = 1024
CC_HALF = DIM // 2


def _blocks(n, step):
    out = []
    off = 0
    while off < n:
        out.append((off, min(step, n - off)))
        off += step
    return out


def _core_runs(pid, cores=CORES):
    """This physical core's local `PT`-row tiles, front-and-back so the two cores balance.

    Local tile `j` sweeps `j + 1` blocks of `NT` compressed positions, so pairing the front and the
    back of the range gives both cores 68 block-units of 1..16 + 16..1.
    """
    per = MT_LOC // (2 * cores)
    a = (pid * per, pid * per + per)
    b = (MT_LOC - (pid + 1) * per, MT_LOC - pid * per)
    runs = [(a[0], b[1])] if a[1] == b[0] else [a, b]
    return ([(t0 * PT, (t1 - t0) * PT) for t0, t1 in runs],
            [t for t0, t1 in runs for t in range(t0, t1)])


def _rank_id():
    """This rank's index as a 1x1 int32 SBUF scalar, usable as a DMA `scalar_offset`."""
    rid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rid, ncc.rank_id())
    return rid


def _shard0(t, rank_rows, cols, rid, row, rows):
    """`rows` rows at `row` of this rank's `(rank_rows, cols)` block of a dim-0 sharded weight.

    The whole recorded tensor is `(WORLD_SIZE * rank_rows, cols)`; viewed as
    `(WORLD_SIZE, rank_rows, cols)` the rank index scales by `rank_rows * cols`, which is what
    `indirect_dim=0` does with the scalar offset. The returned view is the `src` of one DMA;
    everything downstream of it indexes trace-time constants again.
    """
    return t.reshape((WORLD_SIZE, rank_rows, cols)).ap(
        pattern=[[cols, rows], [1, cols]], offset=row * cols,
        scalar_offset=rid, indirect_dim=0)


def _shard1(t, rows_all, rank_cols, rid, row, rows):
    """`rows` rows at `row` of this rank's `(rows_all, rank_cols)` block of a dim-1 sharded weight.

    A column shard is a strided row selection once the tensor is viewed as
    `(rows_all * WORLD_SIZE, rank_cols)`: element `(i, rank * rank_cols + j)` of the recorded
    `(rows_all, WORLD_SIZE * rank_cols)` tensor is row `i * WORLD_SIZE + rank` of that view. So the
    rank index scales by `rank_cols` -- dim 0's stride -- and consecutive rows are `WORLD_SIZE`
    apart.
    """
    return t.reshape((rows_all * WORLD_SIZE, rank_cols)).ap(
        pattern=[[WORLD_SIZE * rank_cols, rows], [1, rank_cols]],
        offset=row * WORLD_SIZE * rank_cols, scalar_offset=rid, indirect_dim=0)


def _rtile(t, cols, rid, tile, extra=0, rows=PT):
    """`rows` rows of a `(_, cols)` HBM tensor at global `PT`-row tile `WORLD_SIZE * tile + rid`.

    The sequence split's one piece of runtime addressing. Viewed as `(_, PT, cols)` the rank index
    scales by `PT * cols`, which is what `indirect_dim=0` does with the scalar offset; `extra`
    shifts by whole tiles on top of that. The returned view is the `src` of one DMA, and everything
    downstream of it indexes trace-time constants again.
    """
    return t.reshape((t.shape[0] // PT, PT, cols)).ap(
        pattern=[[cols, rows], [1, cols]],
        offset=(WORLD_SIZE * tile + extra) * PT * cols, scalar_offset=rid, indirect_dim=0)


def _rank_col(rid, rows, scale=1.0):
    """This rank's index as a `(rows, 1)` float32 column, `scale`d -- a per-partition operand."""
    r1 = nl.copy(rid.reshape((1, 1)), dtype=nl.float32)
    return nl.multiply(nl.broadcast_to(r1, (rows, 1)), scale)


def _load_bcast(vec_hbm, n, rows):
    return nl.broadcast_to(nl.load(vec_hbm.reshape((1, n)), dtype=nl.float32), (rows, n))


def _load_bcast_shard(vec_hbm, n, rows, rid):
    """`_load_bcast` of this rank's `n` of `WORLD_SIZE * n` elements -- `attn_sink`, one per head."""
    v = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=v, src=vec_hbm.reshape((WORLD_SIZE, n)).ap(
        pattern=[[n, 1], [1, n]], offset=0, scalar_offset=rid, indirect_dim=0))
    return nl.broadcast_to(v, (rows, n))


def _rmsnorm(tile, wb, n):
    sq = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=tile, reduce_op=nl.add, reduce_res=ss,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    ms = nl.multiply(ss, 1.0 / n)
    ms = nl.add(ms, EPS)
    y = nl.rsqrt(ms)
    y = nl.multiply(y, nl.subtract(1.5, nl.multiply(nl.multiply(ms, 0.5), nl.multiply(y, y))))
    return nl.multiply(nl.multiply(tile, y), wb)


def _pow2_ceil_log2(x):
    bits = nl.copy(x, dtype=nl.float32).view(nl.int32)
    exp = nl.bitwise_and(nl.right_shift(bits, 23), 0xFF)
    man = nl.bitwise_and(bits, 0x7FFFFF)
    bump = nl.copy(nl.greater(man, 0), dtype=nl.int32)
    e = nl.add(exp, bump)
    scale = nl.copy(nl.left_shift(e, 23)).view(nl.float32)
    inv = nl.copy(nl.left_shift(nl.subtract(254, e), 23)).view(nl.float32)
    return scale, inv


def _i32(shape):
    return nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)


def _decode_e4m3(bytes_u8):
    shape = bytes_u8.shape

    def u8():
        return nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf)

    m7 = u8()
    nisa.tensor_scalar(dst=m7, data=bytes_u8, op0=nl.bitwise_and, operand0=0x7F)
    dn = u8()
    nisa.tensor_scalar(dst=dn, data=m7, op0=nl.maximum, operand0=8, op1=nl.subtract, operand1=8)
    sgn = u8()
    nisa.tensor_scalar(dst=sgn, data=bytes_u8, op0=nl.bitwise_and, operand0=0x80)
    b2 = u8()
    nisa.tensor_tensor(dst=b2, data1=dn, data2=sgn, op=nl.bitwise_or)
    # bf16, not fp32: an fp8_e4m3 value has 4 significant bits and the block scale is a power of
    # two, so the product below is exact in bf16 and the widened tile costs half the Vector
    # bandwidth. The dequantizer was 12.4% of the wall with the Vector engine 99% busy.
    return nl.copy(b2.view(nl.float8_e4m3fn), dtype=nl.bfloat16)


def _block_amax(blocked, floor):
    ab = nl.ndarray(blocked.shape, dtype=blocked.dtype, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=blocked)
    amax = nl.max(ab, axis=2, keepdims=True, dtype=nl.float32)
    return nl.maximum(amax, floor)


def _qdq_fp8(tile, rows, n):
    nb = n // FP8_BLK
    blocked = tile.reshape((rows, nb, FP8_BLK))
    amax = _block_amax(blocked, FP8_AMAX_FLOOR)
    scale, rcp = _pow2_ceil_log2(nl.multiply(amax, 2.0 / FP8_MAX))
    inv = rcp.broadcast(2, FP8_BLK)
    wide = scale.broadcast(2, FP8_BLK)
    q8 = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q8, data1=blocked, data2=inv, op=nl.multiply)
    out = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=q8, data2=wide, op=nl.multiply)
    return out.reshape((rows, n))


def _scale_selector():
    sper = PT // FP8_BLK
    pcol = nl.ndarray((sper, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pcol, pattern=[[1, PT]], offset=0, channel_multiplier=0)
    srow = nl.ndarray((sper, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=srow, pattern=[[1, 1]], offset=0, channel_multiplier=FP8_BLK)
    d = nl.ndarray((sper, PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=d, data=nl.copy(pcol, dtype=nl.float32), op0=nl.subtract,
                       operand0=nl.copy(srow, dtype=nl.float32))
    hit = nl.multiply(nl.copy(nl.greater_equal(d, 0.0), dtype=nl.float32),
                      nl.copy(nl.less(d, float(FP8_BLK)), dtype=nl.float32))
    return nl.copy(hit, dtype=nl.bfloat16)


def _dequant_weight(w_u8, s_u8, dst_hbm, n, k, dst_row=0, src_row=0, shard=None, rid=None,
                    shard_rows=None):
    """Widen a block-scaled fp8 weight to bfloat16 in HBM, `PT` rows at a time.

    `shard` is `None` for a replicated weight -- `w_u8` is then exactly the `(n, k)` block to
    widen. `shard=0` means the recorded tensor is `(WORLD_SIZE * n, k)` and this rank owns rows
    `rank * n ...`; `shard=1` means it is `(n, WORLD_SIZE * k)` and this rank owns columns
    `rank * k ...`. In both sharded cases the row range is a runtime quantity, so the two loads
    become explicit DMAs with `rid` as their `scalar_offset`; the scale shard follows its weight,
    one e8m0 exponent per 32x32 block.

    The sharded path's `dma_copy` into an int32 scale tile is the same widening conversion
    `nl.load(..., dtype=nl.int32)` performs on the replicated path, so the two agree bit for bit.
    """
    kb = k // FP8_BLK
    sper = PT // FP8_BLK
    # One rank's row shard, which `shard_rows` names when this call widens only part of it.
    sh = n if shard_rows is None else shard_rows
    sel = _scale_selector()
    for nb in nl.affine_range(n // PT):
        r0 = src_row + nb * PT
        s0 = r0 // FP8_BLK
        if shard is None:
            wq = _decode_e4m3(nl.load(w_u8[r0:r0 + PT, 0:k]))
            sb = nl.load(s_u8[s0:s0 + sper, 0:kb], dtype=nl.int32)
        else:
            wb = nl.ndarray((PT, k), dtype=nl.uint8, buffer=nl.sbuf)
            sb = nl.ndarray((sper, kb), dtype=nl.int32, buffer=nl.sbuf)
            if shard == 0:
                nisa.dma_copy(dst=wb, src=_shard0(w_u8, sh, k, rid, r0, PT))
                nisa.dma_copy(dst=sb, src=_shard0(s_u8, sh // FP8_BLK, kb, rid, s0, sper))
            else:
                nisa.dma_copy(dst=wb, src=_shard1(w_u8, n, k, rid, r0, PT))
                nisa.dma_copy(dst=sb, src=_shard1(s_u8, n // FP8_BLK, kb, rid, s0, sper))
            wq = _decode_e4m3(wb)
        ex = nl.ndarray((PT, kb), dtype=nl.float32, buffer=nl.psum)
        sbf = nl.ndarray((sper, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbf, data=sb, op0=nl.add, operand0=1)
        nisa.nc_matmul(dst=ex, stationary=sel, moving=sbf, accumulate=False)
        # `2**(e-127)` as one Activation-engine instruction reading psum directly, in place of the
        # int32 copy / `left_shift` / view / bf16 copy chain. These tiles are `k / FP8_BLK` wide,
        # and at that width a Vector instruction costs ~1.7 us whatever its width -- fixed
        # overhead, not throughput -- so the scale path was costing more than the full-width
        # weight passes. `exp` is not exact, but its relative error is far below bf16's 2**-9, so
        # the rounded scale is the same bit pattern the shift produced, and the engine it runs on
        # was 14% busy while Vector was at 99%.
        rep = nl.ndarray((PT, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=rep, op=nl.exp, data=ex, bias=-127.0 * LN2, scale=LN2)
        rep = rep.reshape((PT, kb, 1)).broadcast(2, FP8_BLK)
        prod = nl.ndarray((PT, kb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=wq.reshape((PT, kb, FP8_BLK)), data2=rep,
                           op=nl.multiply)
        nl.store(dst_hbm[dst_row + nb * PT:dst_row + (nb + 1) * PT, 0:k],
                 value=prod.reshape((PT, k)))


FP4_MAX = 6.0


def _qdq_fp4(tile, rows, n, blk):
    nb = n // blk
    shape = (rows, nb, blk)
    blocked = tile.reshape(shape)
    amax = _block_amax(blocked, 6.0 * (2.0 ** -126))
    sbits = nl.copy(nl.multiply(amax, 1.0 / FP4_MAX), dtype=nl.float32).view(nl.int32)
    e = nl.add(nl.bitwise_and(nl.right_shift(sbits, 23), 0xFF),
               nl.copy(nl.greater(nl.bitwise_and(sbits, 0x7FFFFF), 0), dtype=nl.int32))
    ef = _i32(shape)
    nisa.tensor_scalar(dst=ef, data=blocked.view(nl.int32), op0=nl.bitwise_and,
                       operand0=0x7FFFFFFF, op1=nl.right_shift, operand1=23)
    sf = _i32(shape)
    nisa.tensor_tensor(dst=sf, data1=ef, data2=e.broadcast(2, blk), op=nl.maximum)
    ki = _i32(shape)
    nisa.tensor_scalar(dst=ki, data=sf, op0=nl.left_shift, operand0=23)
    kf = nl.multiply(ki.view(nl.float32), 6291456.0)
    y = nl.add(blocked, kf)
    out = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=y, data2=kf, op=nl.subtract)
    return out.reshape((rows, n))


def _rope_inplace(t_hbm, freqs_hbm, m, width, pos_stride, inverse, heads=1, row=0, hstride=0):
    half = ROPE_DIM // 2
    lo = width - ROPE_DIM
    for hb in nl.affine_range(heads):
        for mb in nl.affine_range(m // PT):
            r0 = hb * hstride + row + mb * PT
            t = nl.load(t_hbm[r0:r0 + PT, lo:width], dtype=nl.float32).reshape((PT, half, 2))
            a = t[0:PT, 0:half, 0]
            b = t[0:PT, 0:half, 1]
            p0 = (row + mb * PT) * pos_stride
            p1 = p0 + PT * pos_stride
            cos = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 0], dtype=nl.float32)
            sin = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 1], dtype=nl.float32)
            o = nl.ndarray((PT, half, 2), dtype=nl.float32, buffer=nl.sbuf)
            ac = nl.multiply(a, cos)
            bs = nl.multiply(b, sin)
            as_ = nl.multiply(a, sin)
            bc = nl.multiply(b, cos)
            if inverse:
                nisa.tensor_copy(dst=o[0:PT, 0:half, 0], src=nl.add(ac, bs))
                nisa.tensor_copy(dst=o[0:PT, 0:half, 1], src=nl.subtract(bc, as_))
            else:
                nisa.tensor_copy(dst=o[0:PT, 0:half, 0], src=nl.subtract(ac, bs))
                nisa.tensor_copy(dst=o[0:PT, 0:half, 1], src=nl.add(as_, bc))
            nl.store(t_hbm[r0:r0 + PT, lo:width],
                     value=nl.copy(o.reshape((PT, ROPE_DIM)), dtype=nl.bfloat16))


def _qdq_rows(src_hbm, dst_hbm, m, n, src_row=0, dst_row=0):
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[src_row + mb * PT:src_row + (mb + 1) * PT, 0:n],
                    dtype=nl.bfloat16)
        nl.store(dst_hbm[dst_row + mb * PT:dst_row + (mb + 1) * PT, 0:n],
                 value=_qdq_fp8(t, PT, n))


def _mgroup(mtiles, ktiles):
    mg = 1
    while mg * 2 <= mtiles and (mg * 2) * ktiles * PT * 2 <= MGROUP_BYTES:
        mg *= 2
    return mg


def _stage_x(xt, x_hbm, ktiles, mg, row, x_col=0):
    sg = 4 if mg % 4 == 0 else (2 if mg % 2 == 0 else 1)
    flat = xt.reshape((PT, ktiles, mg * PT))
    x3 = x_hbm.reshape((x_hbm.shape[0], x_hbm.shape[1] // PT, PT))
    c0 = x_col // PT
    for mi in nl.static_range(mg // sg):
        r0 = row + mi * sg * PT
        nisa.dma_transpose(dst=flat[0:PT, 0:ktiles, mi * sg * PT:(mi + 1) * sg * PT],
                           src=x3[r0:r0 + sg * PT, c0:c0 + ktiles, 0:PT])


def _gemm(x_hbm, w_hbm, dst_hbm, m, k, n, out_dtype=nl.bfloat16,
          x_row=0, x_col=0, w_row=0, dst_row=0, dst_col=0, qdq=False):
    ktiles = k // PT
    if n <= NT:
        nt = n
    elif n % NT == 0:
        nt = NT
    elif n % 256 == 0:
        nt = 256
    else:
        nt = PT
    mg = _mgroup(m // PT, ktiles)
    w3 = w_hbm.reshape((w_hbm.shape[0], ktiles, PT))
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, ktiles, mg, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        _stage_x(xt, x_hbm, ktiles, mg, x_row + mgb * mg * PT, x_col=x_col)
        for nb in nl.affine_range(n // nt):
            wt = nl.ndarray((PT, ktiles, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(dst=wt[0:PT, 0:ktiles, 0:nt],
                               src=w3[w_row + nb * nt:w_row + (nb + 1) * nt, 0:ktiles, 0:PT])
            for mi in nl.static_range(mg):
                acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, mi, 0:PT],
                                   moving=wt[0:PT, kb, 0:nt], accumulate=(kb > 0))
                r = dst_row + (mgb * mg + mi) * PT
                if qdq:
                    rb = nl.copy(acc, dtype=nl.bfloat16)
                    val = _qdq_fp8(rb, PT, nt)
                else:
                    val = nl.copy(acc, dtype=out_dtype)
                nl.store(dst_hbm[r:r + PT, dst_col + nb * nt:dst_col + (nb + 1) * nt],
                         value=val)


def _gemm_head_major(x_hbm, w_hbm, dst_hbm, m, k, n, hdim, row=0, hstride=0, freqs_hbm=None):
    ktiles = k // PT
    half = ROPE_DIM // 2
    rg = hdim // ROPE_DIM
    mg = _mgroup(m // PT, ktiles)
    w3 = w_hbm.reshape((w_hbm.shape[0], ktiles, PT))
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, ktiles, mg, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        _stage_x(xt, x_hbm, ktiles, mg, row + mgb * mg * PT)
        if freqs_hbm is not None:
            cs = nl.ndarray((PT, mg, 2, half), dtype=nl.float32, buffer=nl.sbuf)
            for mi in nl.static_range(mg):
                r0 = row + (mgb * mg + mi) * PT
                nisa.tensor_copy(dst=cs[0:PT, mi, 0, 0:half],
                                 src=nl.load(freqs_hbm[r0:r0 + PT, 0:half, 0],
                                             dtype=nl.float32))
                nisa.tensor_copy(dst=cs[0:PT, mi, 1, 0:half],
                                 src=nl.load(freqs_hbm[r0:r0 + PT, 0:half, 1],
                                             dtype=nl.float32))
        for nb in nl.affine_range(n // hdim):
            wt = nl.ndarray((PT, ktiles, hdim), dtype=nl.bfloat16, buffer=nl.sbuf)
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=wt[0:PT, kb, 0:hdim],
                    src=w_hbm[nb * hdim:(nb + 1) * hdim, kb * PT:(kb + 1) * PT])
            for mi in nl.static_range(mg):
                acc = nl.ndarray((PT, hdim), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, mi, 0:PT],
                                   moving=wt[0:PT, kb, 0:hdim], accumulate=(kb > 0))
                r = nb * hstride + row + (mgb * mg + mi) * PT
                if freqs_hbm is None:
                    nl.store(dst_hbm[r:r + PT, 0:hdim], value=nl.copy(acc, dtype=nl.bfloat16))
                else:
                    tb = nl.ndarray((PT, hdim), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=tb, src=acc)
                    v = tb.reshape((PT, rg, half, 2))
                    a = v[0:PT, rg - 1, 0:half, 0]
                    b = v[0:PT, rg - 1, 0:half, 1]
                    c = cs[0:PT, mi, 0, 0:half]
                    s = cs[0:PT, mi, 1, 0:half]
                    ac = nl.multiply(a, c, dtype=nl.float32)
                    bs = nl.multiply(b, s, dtype=nl.float32)
                    as_ = nl.multiply(a, s, dtype=nl.float32)
                    bc = nl.multiply(b, c, dtype=nl.float32)
                    nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
                    nisa.tensor_tensor(dst=b, data1=as_, data2=bc, op=nl.add)
                    nl.store(dst_hbm[r:r + PT, 0:hdim], value=tb)


def _index_q_build(qr_hbm, w_hbm, freqs_hbm, dst_hbm, m, row=0, mg=16, hg_heads=4):
    half = ROPE_DIM // 2
    ktiles = Q_LORA // PT
    hd = hg_heads * INDEX_DIM
    mtiles = m // PT
    # The sequence split made a core's run shorter than the 16-tile group this was written for.
    while mg > mtiles:
        mg //= 2
    for mgb in nl.affine_range(mtiles // mg):
        xt = nl.ndarray((PT, ktiles, mg, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        _stage_x(xt, qr_hbm, ktiles, mg, row + mgb * mg * PT)
        cos4 = nl.ndarray((PT, mg, hg_heads, half), dtype=nl.float32, buffer=nl.sbuf)
        sin4 = nl.ndarray((PT, mg, hg_heads, half), dtype=nl.float32, buffer=nl.sbuf)
        for mi in nl.static_range(mg):
            r0 = row + (mgb * mg + mi) * PT
            cos = nl.load(freqs_hbm[r0:r0 + PT, 0:half, 0], dtype=nl.float32)
            sin = nl.load(freqs_hbm[r0:r0 + PT, 0:half, 1], dtype=nl.float32)
            for j in nl.static_range(hg_heads):
                nisa.tensor_copy(dst=cos4[0:PT, mi, j, 0:half], src=cos)
                nisa.tensor_copy(dst=sin4[0:PT, mi, j, 0:half], src=sin)
        for hg in nl.affine_range(INDEX_HEADS // hg_heads):
            wt = nl.ndarray((PT, ktiles, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=wt[0:PT, kb, 0:hd],
                    src=w_hbm[hg * hd:(hg + 1) * hd, kb * PT:(kb + 1) * PT])
            for mi in nl.static_range(mg):
                r0 = row + (mgb * mg + mi) * PT
                acc = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, mi, 0:PT],
                                   moving=wt[0:PT, kb, 0:hd], accumulate=(kb > 0))
                tb = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=tb, src=acc)
                v = tb.reshape((PT, hg_heads, 2, half, 2))
                a = v[0:PT, 0:hg_heads, 1, 0:half, 0]
                b = v[0:PT, 0:hg_heads, 1, 0:half, 1]
                c = cos4[0:PT, mi, 0:hg_heads, 0:half]
                s = sin4[0:PT, mi, 0:hg_heads, 0:half]
                ac = nl.multiply(a, c, dtype=nl.float32)
                bs = nl.multiply(b, s, dtype=nl.float32)
                as_ = nl.multiply(a, s, dtype=nl.float32)
                bc = nl.multiply(b, c, dtype=nl.float32)
                nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
                nisa.tensor_tensor(dst=b, data1=as_, data2=bc, op=nl.add)
                tf = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=tf, src=tb)
                nl.store(dst_hbm[r0:r0 + PT, hg * hd:(hg + 1) * hd],
                         value=_qdq_fp4(tf, PT, hd, 32))


def _norm_rows(src_hbm, w_hbm, dst_hbm, m, n, row=0, src_col=0):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, src_col:src_col + n],
                    dtype=nl.float32)
        nl.store(dst_hbm[row + mb * PT:row + (mb + 1) * PT, 0:n],
                 value=nl.copy(_rmsnorm(t, wb, n), dtype=nl.bfloat16))


def _norm_quant_rows(src_hbm, w_hbm, dst_hbm, m, n, row=0, src_col=0):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, src_col:src_col + n],
                    dtype=nl.float32)
        tb = nl.copy(_rmsnorm(t, wb, n), dtype=nl.bfloat16)
        nl.store(dst_hbm[row + mb * PT:row + (mb + 1) * PT, 0:n],
                 value=_qdq_fp8(tb, PT, n))


def _norm_rows_both(src_hbm, w_hbm, dst_b, dst_q, m, n, row=0):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, 0:n], dtype=nl.float32)
        tb = nl.copy(_rmsnorm(t, wb, n), dtype=nl.bfloat16)
        nl.store(dst_b[row + mb * PT:row + (mb + 1) * PT, 0:n], value=tb)
        nl.store(dst_q[row + mb * PT:row + (mb + 1) * PT, 0:n],
                 value=_qdq_fp8(tb, PT, n))


def _index_mask(index_q, ik_hbm, iw_hbm, msk_hbm, scr_hbm, thr_hbm, tiles, rid):
    """The indexer's score and its top-`INDEX_TOPK` selection, for this core's local query tiles.

    The selection is `nisa.isa.topk` on the GpSimd engine rather than a bisection on the Vector
    engine. `topk` reads one problem per GpSimd core out of a 16-partition snake, so a
    `(PT, 128)`-partition source tile carries 8 queries; the snake's index mapping does not matter
    here because only `val_dst[:, 0]` -- the `INDEX_TOPK`-th largest value, the threshold -- is
    read, so any bijection between a query's `n` scores and its 16 x `n/16` region is as good as
    the documented one. The cheap bijection is "partition `p` holds the `p`-th contiguous chunk",
    which makes the staging a plain contiguous load once the score rows are written out chunk-major
    (`scr_hbm` viewed as `(SEQ_LOC, 16, NT)`).

    A query with fewer than `INDEX_TOPK` causally-reachable positions gets `NEG_INF` back as its
    threshold, because the padding the causal compare wrote is what fills out the snake. Clamping
    the threshold at `THR_FLOOR` turns that into "take every reachable position", which is what the
    reference's `topk` over a shorter row does.
    """
    iq3 = index_q.reshape((SEQ_LOC, INDEX_HEADS, INDEX_DIM))

    kblk = max(tiles) + 1
    ikt_all = nl.ndarray((INDEX_DIM, kblk, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gb in nl.affine_range(kblk):
        nisa.dma_transpose(dst=ikt_all[0:INDEX_DIM, gb, 0:NT],
                           src=ik_hbm[gb * NT:(gb + 1) * NT, 0:INDEX_DIM])

    bpos_i = nl.ndarray((PT, NT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=bpos_i, pattern=[[1, NT]], offset=0, channel_multiplier=0)
    bpos = nl.copy(bpos_i, dtype=nl.float32)
    prow_i = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=prow_i, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    prow = nl.copy(prow_i, dtype=nl.float32)
    neg = nl.full((PT, NT), NEG_INF, dtype=nl.bfloat16)
    # Local tile `j` row `i` is global query `(WORLD_SIZE * j + rid) * PT + i`, so inside the last
    # block of `NT` the reachable columns are `0 .. rid * PT + i`. The rank's contribution is the
    # only runtime term in the whole score pass.
    clim = nl.add(nl.add(prow, 1.0), _rank_col(rid, PT, float(PT)))
    for j in tiles:
        gbn = j + 1
        q0 = j * PT
        wrow = nl.multiply(nl.load(iw_hbm[q0:q0 + PT, 0:INDEX_HEADS], dtype=nl.float32),
                           INDEX_SCORE_SCALE)
        iqt = nl.ndarray((INDEX_DIM, INDEX_HEADS, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_transpose(dst=iqt[0:INDEX_DIM, 0:INDEX_HEADS, 0:PT],
                           src=iq3[q0:q0 + PT, 0:INDEX_HEADS, 0:INDEX_DIM])
        rows = nl.ndarray((PT, gbn, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
        for gb in nl.static_range(gbn):
            ikt = ikt_all[0:INDEX_DIM, gb, 0:NT]
            tgt = rows[0:PT, gb, 0:NT]
            for hb in nl.static_range(INDEX_HEADS):
                ps = nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=iqt[0:INDEX_DIM, hb, 0:PT], moving=ikt,
                               accumulate=False)
                rect = nl.ndarray((PT, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=rect, op=nl.relu, data=ps)
                if hb == 0:
                    nisa.tensor_scalar(dst=tgt, data=rect, op0=nl.multiply,
                                       operand0=wrow[0:PT, hb:hb + 1])
                else:
                    nisa.scalar_tensor_tensor(dst=tgt, data=rect, op0=nl.multiply,
                                              operand0=wrow[0:PT, hb:hb + 1], op1=nl.add,
                                              operand1=tgt)
            if gb == gbn - 1:
                reach = nl.ndarray((PT, NT), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=reach, data=bpos[0:PT, 0:NT], op0=nl.less, operand0=clim)
                nisa.tensor_copy(dst=tgt, src=nl.where(reach, tgt, neg))

        # Out chunk-major and straight back in as the 8-queries-per-tile `topk` snake. `scr_hbm`
        # and `thr_hbm` are written and read through exactly one view each, and every `topk`
        # operand is a whole tile of its own -- a sliced tile and a second view of the same buffer
        # are both things this instruction was not observed to tolerate.
        src_x = (gbn * NT) // 16
        nl.store(scr_hbm[q0:q0 + PT, 0:16, 0:src_x], value=rows.reshape((PT, 16, src_x)))
        # All three `topk` operands are allocated per call. Sharing them across the 256 calls is
        # 0.37 ms faster and gives a wrong threshold on a handful of queries, and hoisting the
        # staging tile alone buys nothing -- so the cost is the price of a correct result, not slack.
        for gi in nl.static_range(PT // 8):
            a0 = q0 + gi * 8
            tsrc = nl.ndarray((PT, src_x), dtype=nl.bfloat16, buffer=nl.sbuf)
            # Pre-filled with +inf so the minimum below is over the slots `topk` actually wrote,
            # wherever in the K region it chose to write them. `val_dst` holds exactly the top-K
            # values, so that minimum is the K-th largest -- the threshold -- under any ordering,
            # and a slot left untouched cannot drag it down.
            vdst = nl.ndarray((PT, INDEX_TOPK), dtype=nl.bfloat16, buffer=nl.sbuf)
            vk = vdst[0:PT, 0:INDEX_TOPK // 16]
            nisa.memset(dst=vk, value=-NEG_INF)
            idst = nl.ndarray((PT, INDEX_TOPK), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tsrc, src=scr_hbm[a0:a0 + 8, 0:16, 0:src_x])
            nisa.topk(val_dst=vdst, idx_dst=idst, src=tsrc, n=gbn * NT)
            # `val_dst` holds exactly the top-K values, in the snake's `16 x K/16` region, so the
            # minimum over that region is the K-th largest whichever end the hardware writes
            # first. Per-partition min here, the cross-partition fold after the round trip.
            vmin = nl.min(vk, axis=1, keepdims=True)
            nisa.dma_copy(dst=thr_hbm[a0:a0 + 8, 0:16], src=vmin)
        thr = nl.maximum(nl.min(nl.load(thr_hbm[q0:q0 + PT, 0:16], dtype=nl.float32),
                                axis=1, keepdims=True), THR_FLOOR)
        for gb in nl.affine_range(gbn):
            mt_ = nl.ndarray((PT, NT), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mt_, data=rows[0:PT, gb, 0:NT],
                               op0=nl.greater_equal, operand0=thr)
            nl.store(msk_hbm[q0:q0 + PT, gb * NT:(gb + 1) * NT], value=mt_)


def _sparse_attention(q_hbm, kvp_hbm, ckv_hbm, msk_hbm, sink_hbm, freqs_hbm, og_hbm, winl, tiles,
                      rid):
    dt = HEAD_DIM // PT
    st = NT // PT
    cmax = (max(tiles) + 1) * NT
    q3 = q_hbm.reshape((N_HEADS * SEQ_LOC, dt, PT))
    ckv3 = ckv_hbm.reshape((CACHE_ROWS, dt, PT))
    winl3 = winl.reshape((MT_LOC * 2 * PT, dt, PT))

    jio_i = nl.ndarray((PT, WIN_SPAN), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=jio_i, pattern=[[1, WIN_SPAN]], offset=0, channel_multiplier=0)
    jio = nl.copy(jio_i, dtype=nl.float32)
    pio_i = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pio_i, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    pio = nl.copy(pio_i, dtype=nl.float32)
    dio = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dio, data=jio, op0=nl.subtract, operand0=pio)
    inwin = nl.multiply(nl.copy(nl.greater_equal(dio, 1.0), dtype=nl.float32),
                        nl.copy(nl.less_equal(dio, float(PT)), dtype=nl.float32))
    nonneg = nl.copy(nl.greater_equal(jio, float(PT)), dtype=nl.float32)
    pwin_gen = nl.copy(nl.greater(inwin, 0.5), dtype=nl.uint8)
    # Local tile 0 is global tile `rid`, so only rank 0's tile 0 can see `kvp`'s zero-padded rows
    # below position 0. `nonneg` must be applied there and must *not* be applied on ranks 1-3, where
    # the same columns carry real tokens -- so the switch is a runtime 0/1 taken from `rid`.
    lowok = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lowok, data=nonneg, op0=nl.maximum,
                       operand0=nl.copy(nl.greater(_rank_col(rid, PT), 0.5), dtype=nl.float32))
    pwin_zero = nl.copy(nl.greater(nl.multiply(lowok, inwin), 0.5), dtype=nl.uint8)
    # All 64 recorded sinks: a rank holds every head now.
    sinkb = _load_bcast(sink_hbm, N_HEADS, PT)

    ckv_t = nl.ndarray((PT, dt, cmax), dtype=nl.bfloat16, buffer=nl.sbuf)
    ckv_n = nl.ndarray((PT, cmax // PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gp in nl.affine_range(cmax // PT):
        nisa.dma_copy(dst=ckv_n[0:PT, gp, 0:HEAD_DIM],
                      src=ckv_hbm[gp * PT:(gp + 1) * PT, 0:HEAD_DIM])
        nisa.dma_transpose(dst=ckv_t[0:PT, 0:dt, gp * PT:(gp + 1) * PT],
                           src=ckv3[gp * PT:(gp + 1) * PT, 0:dt, 0:PT])

    if True:
        for j in tiles:
            gbn = j + 1
            q0 = j * PT
            pwin = pwin_zero if j == 0 else pwin_gen
            win_t = nl.ndarray((PT, dt, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
            win_n = nl.ndarray((PT, 2, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
            # The window's two source tiles sit at a runtime row in the gathered `kvp`, so they are
            # staged once with `rid` as the DMA's scalar offset and re-read from `winl` at a
            # trace-time row for the transpose.
            for wt in nl.static_range(2):
                nisa.dma_copy(dst=win_n[0:PT, wt, 0:HEAD_DIM],
                              src=_rtile(kvp_hbm, HEAD_DIM, rid, j, extra=wt))
                nl.store(winl[(j * 2 + wt) * PT:(j * 2 + wt + 1) * PT, 0:HEAD_DIM],
                         value=win_n[0:PT, wt, 0:HEAD_DIM])
            for wt in nl.static_range(2):
                w0 = (j * 2 + wt) * PT
                nisa.dma_transpose(dst=win_t[0:PT, 0:dt, wt * PT:(wt + 1) * PT],
                                   src=winl3[w0:w0 + PT, 0:dt, 0:PT])
            kp = nl.load(msk_hbm[q0:q0 + PT, 0:gbn * NT], dtype=nl.uint8)
            rcos = nl.load(freqs_hbm[q0:q0 + PT, 0:ROPE_DIM // 2, 0], dtype=nl.float32)
            rsin = nl.load(freqs_hbm[q0:q0 + PT, 0:ROPE_DIM // 2, 1], dtype=nl.float32)
            rc8 = rcos.reshape((PT, 1, ROPE_DIM // 2)).broadcast(1, OB)
            rs8 = rsin.reshape((PT, 1, ROPE_DIM // 2)).broadcast(1, OB)


            for g in nl.affine_range(O_GROUPS):
              for hbat in nl.static_range(HEADS_PER_GROUP // OB):
                obuf = nl.ndarray((PT, OB, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
                for hi in nl.affine_range(OB):
                    hh = hbat * OB + hi
                    h = g * HEADS_PER_GROUP + hh
                    qt = nl.ndarray((PT, dt, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_transpose(dst=qt[0:PT, 0:dt, 0:PT],
                                       src=q3[h * SEQ_LOC + q0:h * SEQ_LOC + q0 + PT, 0:dt, 0:PT])
                    wmax = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                    lwp = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.psum)
                    for dc in nl.static_range(dt):
                        nisa.nc_matmul(dst=lwp, stationary=qt[0:PT, dc, 0:PT],
                                       moving=win_t[0:PT, dc, 0:WIN_SPAN],
                                       accumulate=(dc > 0))
                    lw = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.select_reduce(dst=lw, predicate=pwin, on_true=lwp,
                                       on_false=NEG_BOUND, reduce_res=wmax,
                                       reduce_cmd=nisa.reduce_cmd.reset_reduce,
                                       reduce_op=nl.maximum)
                    lcs = nl.ndarray((PT, gbn * NT), dtype=nl.float32, buffer=nl.sbuf)
                    bmax = nl.ndarray((PT, gbn), dtype=nl.float32, buffer=nl.sbuf)
                    for gb in nl.static_range(gbn):
                        bw = NT
                        keep = kp[0:PT, gb * NT:gb * NT + bw]
                        lcp = nl.ndarray((PT, bw), dtype=nl.float32, buffer=nl.psum)
                        for dc in nl.static_range(dt):
                            nisa.nc_matmul(dst=lcp, stationary=qt[0:PT, dc, 0:PT],
                                           moving=ckv_t[0:PT, dc, gb * NT:gb * NT + bw],
                                           accumulate=(dc > 0))
                        nisa.select_reduce(dst=lcs[0:PT, gb * NT:gb * NT + bw],
                                           predicate=keep, on_true=lcp, on_false=NEG_BOUND,
                                           reduce_res=bmax[0:PT, gb:gb + 1],
                                           reduce_cmd=nisa.reduce_cmd.reset_reduce,
                                           reduce_op=nl.maximum)
                    m = nl.multiply(nl.maximum(wmax, nl.max(bmax, axis=1, keepdims=True)),
                                    SM_SCALE)
                    negm = nl.multiply(m, -1.0)
                    nch = (gbn + CHUNK - 1) // CHUNK
                    dcol = nl.ndarray((PT, nch + 1), dtype=nl.float32, buffer=nl.sbuf)
                    acc = nl.ndarray((PT, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
                    pw = nl.ndarray((PT, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=pw, op=nl.exp, data=lw, bias=negm, scale=SM_SCALE,
                                    reduce_op=nl.add, reduce_res=dcol[0:PT, 0:1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                    pwt = nl.ndarray((PT, 2, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_transpose(dst=pwt[0:PT, 0:2, 0:PT],
                                       src=pw.reshape((PT, 2, PT))[0:PT, 0:2, 0:PT])
                    for wt in nl.static_range(2):
                        nisa.nc_matmul(dst=acc, stationary=pwt[0:PT, wt, 0:PT],
                                       moving=win_n[0:PT, wt, 0:HEAD_DIM],
                                       accumulate=(wt > 0))
                    for ci in range(nch):
                        g0 = ci * CHUNK
                        gk = min(CHUNK, gbn - g0)
                        cw = gk * NT
                        nb = cw // PT
                        pc = nl.ndarray((PT, cw), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.activation(dst=pc, op=nl.exp,
                                        data=lcs[0:PT, g0 * NT:g0 * NT + cw],
                                        bias=negm, scale=SM_SCALE,
                                        reduce_op=nl.add,
                                        reduce_res=dcol[0:PT, ci + 1:ci + 2],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
                        pct = nl.ndarray((PT, nb, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_transpose(dst=pct[0:PT, 0:nb, 0:PT],
                                           src=pc.reshape((PT, nb, PT))[0:PT, 0:nb, 0:PT])
                        for sb in nl.static_range(nb):
                            nisa.nc_matmul(dst=acc, stationary=pct[0:PT, sb, 0:PT],
                                           moving=ckv_n[0:PT, g0 * st + sb, 0:HEAD_DIM],
                                           accumulate=True)
                    esink = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=esink, op=nl.exp, data=sinkb[0:PT, h:h + 1],
                                    bias=negm, scale=1.0)
                    den = nl.add(nl.sum(dcol, axis=1, keepdims=True), esink)
                    nisa.tensor_scalar(dst=obuf[0:PT, hi, 0:HEAD_DIM], data=acc,
                                       op0=nl.multiply, operand0=nl.reciprocal(den))
                ov = obuf.reshape((PT, OB, HEAD_DIM // ROPE_DIM, ROPE_DIM // 2, 2))
                ra = ov[0:PT, 0:OB, HEAD_DIM // ROPE_DIM - 1, 0:ROPE_DIM // 2, 0]
                rb = ov[0:PT, 0:OB, HEAD_DIM // ROPE_DIM - 1, 0:ROPE_DIM // 2, 1]
                rac = nl.multiply(ra, rc8)
                rbs = nl.multiply(rb, rs8)
                ras = nl.multiply(ra, rs8)
                rbc = nl.multiply(rb, rc8)
                nisa.tensor_tensor(dst=ra, data1=rac, data2=rbs, op=nl.add)
                nisa.tensor_tensor(dst=rb, data1=rbc, data2=ras, op=nl.subtract)
                nl.store(og_hbm[g * SEQ_LOC + q0:g * SEQ_LOC + q0 + PT,
                                hbat * OB * HEAD_DIM:(hbat + 1) * OB * HEAD_DIM], value=obuf)


@nki.jit
def kernel(hidden_states, state_shared_attn_candidates, state_shared_attn_compress_kv, state_shared_attn_index_k, state_shared_attn_topk_idxs, layers_24_attn_attn_sink, layers_24_attn_freqs_cis, layers_24_attn_indexer_weights_proj_weight, layers_24_attn_indexer_wq_b_scale, layers_24_attn_indexer_wq_b_weight, layers_24_attn_kv_norm_weight, layers_24_attn_q_norm_weight, layers_24_attn_window_kv_cache, layers_24_attn_wkv_scale, layers_24_attn_wkv_weight, layers_24_attn_wo_a_scale, layers_24_attn_wo_a_weight, layers_24_attn_wo_b_scale, layers_24_attn_wo_b_weight, layers_24_attn_wq_a_scale, layers_24_attn_wq_a_weight, layers_24_attn_wq_b_scale, layers_24_attn_wq_b_weight, layers_24_attn_norm_weight, replica_group):
    x_hbm = hidden_states.reshape((SEQ, DIM))
    ckv_hbm = state_shared_attn_compress_kv.reshape((CACHE_ROWS, HEAD_DIM))
    ik_hbm = state_shared_attn_index_k.reshape((CACHE_ROWS, INDEX_DIM))
    # The module's output, and the only IO tensor the collective may not touch.
    out = nl.ndarray((1, SEQ, DIM), dtype=OUT_DTYPE, buffer=nl.shared_hbm)
    out2 = out.reshape((SEQ, DIM))
    # The seam's buffers. Named, all nl.shared_hbm, none an IO tensor, and each the whole of
    # exactly the collective's shape. Two collectives now: the window KV latent is produced
    # sequence-parallel and all-gathered before the attention needs a neighbour's rows, and the
    # output is all-gathered at the end instead of all-reduced -- a rank owns whole rows of the
    # output now, not a partial sum of all of them.
    # The output gather is split in two column halves so the first half's collective overlaps the
    # second half's `wo_b` GEMM. Unsplit, the gather and its copy out were a 565 us tail with every
    # compute engine 99.9% idle -- the collective cannot start before the whole GEMM is done and
    # the copy cannot start before the collective is.
    cc_src_a = nl.ndarray((SEQ_LOC, CC_HALF), dtype=CC_DTYPE, buffer=nl.shared_hbm,
                          name="cc_src_a")
    cc_dst_a = nl.ndarray((SEQ, CC_HALF), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_dst_a")
    cc_src_b = nl.ndarray((SEQ_LOC, CC_HALF), dtype=CC_DTYPE, buffer=nl.shared_hbm,
                          name="cc_src_b")
    cc_dst_b = nl.ndarray((SEQ, CC_HALF), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_dst_b")
    kv_src = nl.ndarray((SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="kv_src")
    kv_dst = nl.ndarray((SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kv_dst")

    pid = nl.program_id(0)
    runs, tiles = _core_runs(pid)
    atiles = sorted(tiles, reverse=True)

    # Which quarter of the *sequence* this rank owns. One NEFF runs on all four ranks, so this
    # cannot be a trace-time constant; `ncc.rank_id()` is the runtime scalar that makes it one, and
    # it reaches the hardware as the `scalar_offset` of the staging DMAs. Both physical cores take
    # their own copy of it. No weight is sharded any more -- a rank holds all 64 heads.
    rid = _rank_id()

    QKV_N = Q_LORA + HEAD_DIM
    wqkv = nl.ndarray((QKV_N, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="wqkv")
    wqb = nl.ndarray((N_HEADS * HEAD_DIM, Q_LORA), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="wqb")
    iwqb = nl.ndarray((INDEX_HEADS * INDEX_DIM, Q_LORA), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="iwqb")
    wo_a = nl.ndarray((O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_a")
    wo_b = nl.ndarray((DIM, O_GROUPS * O_LORA), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_b")
    # --- the dequantizer, a quarter per rank ----------------------------------------------------
    # Every weight is replicated, so all four ranks were widening the same 132 MB of fp8 into the
    # same 264 MB of bfloat16 -- 12.4% of the wall, and at its HBM bandwidth floor (396 MB at the
    # profile's 118 GB/s), so no rearrangement of the compute helps it. A rank only has to widen
    # a quarter of a weight if a collective supplies the other three quarters.
    #
    # `wq_b`, `wo_a` and `wo_b` -- 118 of the 132 MB -- are therefore widened a quarter at a time
    # into dedicated `nl.shared_hbm` staging buffers, each of exactly the collective's source
    # shape, and each filled from *both* physical cores in disjoint row halves, which is the
    # property every collective source that already worked in this kernel has. The gather's
    # destinations *are* the weight buffers, so there is no copy-out. `wqkv` and the indexer's
    # `wq_b` stay replicated: `_index_mask` consumes their readers' output, so those readers
    # cannot move below it, and the next paragraph is why that matters.
    #
    # Two measured constraints shape the rest:
    #   * **One coalesced collective, not three.** Two `ncc.all_gather` calls in a row whose
    #     destinations are both read later silently corrupt the first destination
    #     (`max_abs_err 2.389` against a 0.5125 bar); one call with three `srcs`/`dsts` is
    #     bit-exact.
    #   * **It cannot be issued before the sequence-parallel prefix.** Every attempt to put a
    #     collective in the dequant region or between the barriers and the prefix died on
    #     `[NCC_ISCH719] topological order violations`. That is why `_gemm_head_major` -- the only
    #     prologue reader of a sharded weight -- moves below `_index_mask`, which is also what
    #     gives the 235 MB gather 2.4 ms of masking to hide behind.
    WQB_N = N_HEADS * HEAD_DIM // WORLD_SIZE
    WOA_N = O_GROUPS * O_LORA // WORLD_SIZE
    WOB_N = DIM // WORLD_SIZE
    wqb_src = nl.ndarray((WQB_N, Q_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                         name="wqb_src")
    woa_src = nl.ndarray((WOA_N, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                         buffer=nl.shared_hbm, name="woa_src")
    wob_src = nl.ndarray((WOB_N, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                         name="wob_src")
    if pid == 0:
        _dequant_weight(layers_24_attn_wq_a_weight, layers_24_attn_wq_a_scale, wqkv,
                        Q_LORA, DIM)
        _dequant_weight(layers_24_attn_wkv_weight, layers_24_attn_wkv_scale, wqkv,
                        HEAD_DIM, DIM, dst_row=Q_LORA)
    else:
        _dequant_weight(layers_24_attn_indexer_wq_b_weight,
                        layers_24_attn_indexer_wq_b_scale, iwqb,
                        INDEX_HEADS * INDEX_DIM, Q_LORA)
    _dequant_weight(layers_24_attn_wq_b_weight, layers_24_attn_wq_b_scale, wqb_src,
                    WQB_N // CORES, Q_LORA, dst_row=pid * (WQB_N // CORES),
                    src_row=pid * (WQB_N // CORES), shard=0, rid=rid, shard_rows=WQB_N)
    _dequant_weight(layers_24_attn_wo_a_weight, layers_24_attn_wo_a_scale, woa_src,
                    WOA_N // CORES, HEADS_PER_GROUP * HEAD_DIM,
                    dst_row=pid * (WOA_N // CORES), src_row=pid * (WOA_N // CORES),
                    shard=0, rid=rid, shard_rows=WOA_N)
    _dequant_weight(layers_24_attn_wo_b_weight, layers_24_attn_wo_b_scale, wob_src,
                    WOB_N // CORES, O_GROUPS * O_LORA, dst_row=pid * (WOB_N // CORES),
                    src_row=pid * (WOB_N // CORES), shard=0, rid=rid, shard_rows=WOB_N)
    for wbuf in (wqkv, iwqb, wqb_src, woa_src, wob_src):
        nisa.core_barrier(data=wbuf, cores=tuple(range(CORES)))

    # Everything from here to `wo_b` is on this rank's own SEQ_LOC rows. `xloc` and `floc` are the
    # rank's interleaved tiles staged down to a contiguous local range, which is what lets every
    # helper below keep indexing trace-time constants.
    xloc = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    floc = nl.ndarray((SEQ_LOC, ROPE_DIM // 2, 2), dtype=nl.float32, buffer=nl.private_hbm)
    xn = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    xn_q = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    qakv = nl.ndarray((SEQ_LOC, QKV_N), dtype=nl.bfloat16, buffer=nl.private_hbm)
    kvr = nl.ndarray((SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    qr_q = nl.ndarray((SEQ_LOC, Q_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)
    kvp = nl.ndarray((SEQ + PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    winl = nl.ndarray((MT_LOC * 2 * PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    index_q = nl.ndarray((SEQ_LOC, INDEX_HEADS * INDEX_DIM), dtype=nl.bfloat16,
                         buffer=nl.private_hbm)
    iw = nl.ndarray((SEQ_LOC, INDEX_HEADS), dtype=nl.bfloat16, buffer=nl.private_hbm)
    q = nl.ndarray((N_HEADS * SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    msk = nl.ndarray((SEQ_LOC, CSEQ), dtype=nl.uint8, buffer=nl.private_hbm)
    scr = nl.ndarray((SEQ_LOC, 16, NT), dtype=nl.bfloat16, buffer=nl.private_hbm)
    thr = nl.ndarray((SEQ_LOC, 16), dtype=nl.bfloat16, buffer=nl.private_hbm)
    og = nl.ndarray((O_GROUPS * SEQ_LOC, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                    buffer=nl.private_hbm)
    z_q = nl.ndarray((SEQ_LOC, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)

    xt = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    ft = nl.ndarray((PT, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
    for j in tiles:
        nisa.dma_copy(dst=xt, src=_rtile(x_hbm, DIM, rid, j))
        nl.store(xloc[j * PT:(j + 1) * PT, 0:DIM], value=xt)
        nisa.dma_copy(dst=ft, src=_rtile(layers_24_attn_freqs_cis.reshape(
                                             (layers_24_attn_freqs_cis.shape[0], ROPE_DIM)),
                                         ROPE_DIM, rid, j))
        nl.store(floc.reshape((SEQ_LOC, ROPE_DIM))[j * PT:(j + 1) * PT, 0:ROPE_DIM], value=ft)

    for r0, m in runs:
        _norm_rows_both(xloc, layers_24_attn_norm_weight, xn, xn_q, m, DIM, row=r0)
        _gemm(xn_q, wqkv, qakv, m, DIM, QKV_N, x_row=r0, dst_row=r0)
        _norm_rows(qakv, layers_24_attn_kv_norm_weight, kvr, m, HEAD_DIM,
                   row=r0, src_col=Q_LORA)
        _rope_inplace(kvr, floc, m, HEAD_DIM, 1, False, row=r0)
        _qdq_rows(kvr, kv_src, m, HEAD_DIM, src_row=r0, dst_row=r0)

        _norm_quant_rows(qakv, layers_24_attn_q_norm_weight, qr_q, m, Q_LORA, row=r0)

        _index_q_build(qr_q, iwqb, floc, index_q, m, row=r0)

        _gemm(xn, layers_24_attn_indexer_weights_proj_weight, iw, m, DIM, INDEX_HEADS,
              x_row=r0, dst_row=r0)

    # The window KV latent is the one thing a rank cannot finish alone: query tile `j`'s window
    # reaches into global tile `WORLD_SIZE * j + rid - 1`, which belongs to the rank below. One
    # all-gather of 8.4 MB rejoins all four quarters; the scatter that follows puts gathered slot
    # `s * SEQ_LOC + j * PT` at global tile `WORLD_SIZE * j + s`, all of it trace-time.
    nisa.core_barrier(data=kv_src, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[kv_src], dsts=[kv_dst], replica_group=replica_group, collective_dim=0)
    if pid == 0:
        nl.store(kvp[0:PT, 0:HEAD_DIM], value=nl.zeros((PT, HEAD_DIM), dtype=nl.bfloat16))
    # Every core scatters the *whole* gathered latent, not just its own tiles. Rank 0's local tile
    # `j` takes its window from global tile `4*j - 1`, which is rank 3's local tile `j - 1`, and at
    # a core-run boundary (`j` = 4 and 12) that tile belongs to the other physical core. Reading it
    # across the `core_barrier` did not work -- global tiles 16 and 48 came back wrong and nothing
    # else did -- so the dependency is removed instead of synchronised. 8.4 MB duplicated.
    for j in range(MT_LOC):
        for s in range(WORLD_SIZE):
            g = WORLD_SIZE * j + s
            nisa.dma_copy(dst=kvp[(g + 1) * PT:(g + 2) * PT, 0:HEAD_DIM],
                          src=kv_dst[s * SEQ_LOC + j * PT:s * SEQ_LOC + (j + 1) * PT, 0:HEAD_DIM])
    nisa.core_barrier(data=kvp, cores=tuple(range(CORES)))

    ncc.all_gather(srcs=[wqb_src, woa_src, wob_src], dsts=[wqb, wo_a, wo_b],
                   replica_group=replica_group, collective_dim=0)

    _index_mask(index_q, ik_hbm, iw, msk, scr, thr, atiles, rid)
    # Below `_index_mask`, not inside the prologue: the gather above cannot be issued any earlier
    # (`[NCC_ISCH719]`), so its only reader has to be moved down to give it something to hide
    # behind. It needs only `qr_q` and `floc`, both finished in the prologue.
    for r0, m in runs:
        _gemm_head_major(qr_q, wqb, q, m, Q_LORA, N_HEADS * HEAD_DIM, HEAD_DIM,
                         row=r0, hstride=SEQ_LOC, freqs_hbm=floc)
    _sparse_attention(q, kvp, ckv_hbm, msk, layers_24_attn_attn_sink,
                      floc, og, winl, atiles, rid)

    for r0, m in runs:
        for g in range(O_GROUPS):
            _gemm(og, wo_a, z_q, m, HEADS_PER_GROUP * HEAD_DIM, O_LORA,
                  x_row=g * SEQ_LOC + r0, w_row=g * O_LORA,
                  dst_row=r0, dst_col=g * O_LORA, qdq=True)
    # `wo_b` is no longer row-parallel: a rank holds every reduction column and writes whole,
    # finished rows of the module's output -- its own quarter of the sequence. The two halves of
    # its output columns are separate passes so that the first half's collective has the second
    # half's GEMM to hide behind.
    for r0, m in runs:
        _gemm(z_q, wo_b, cc_src_a, m, O_GROUPS * O_LORA, CC_HALF, x_row=r0, dst_row=r0,
              out_dtype=CC_DTYPE)
    # The two physical cores of this rank's logical core filled disjoint row runs of `cc_src_*`.
    # `core_barrier` between those writes into shared HBM and the collective's read of them is
    # what makes the other core's rows visible.
    nisa.core_barrier(data=cc_src_a, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[cc_src_a], dsts=[cc_dst_a], replica_group=replica_group,
                   collective_dim=0)
    for r0, m in runs:
        _gemm(z_q, wo_b, cc_src_b, m, O_GROUPS * O_LORA, CC_HALF, x_row=r0, dst_row=r0,
              w_row=CC_HALF, out_dtype=CC_DTYPE)
    nisa.core_barrier(data=cc_src_b, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[cc_src_b], dsts=[cc_dst_b], replica_group=replica_group,
                   collective_dim=0)
    # After the gathers every rank holds all four quarters, rank 0 included. The copy out undoes
    # the tile interleave: gathered slot `s * SEQ_LOC + j * PT` is global tile
    # `WORLD_SIZE * j + s`.
    for half, cc_dst in ((0, cc_dst_a), (1, cc_dst_b)):
        for j in tiles:
            for s in range(WORLD_SIZE):
                g = WORLD_SIZE * j + s
                nisa.dma_copy(
                    dst=out2[g * PT:(g + 1) * PT, half * CC_HALF:(half + 1) * CC_HALF],
                    src=cc_dst[s * SEQ_LOC + j * PT:s * SEQ_LOC + (j + 1) * PT, 0:CC_HALF])
    nisa.core_barrier(data=out, cores=tuple(range(CORES)))
    return out
