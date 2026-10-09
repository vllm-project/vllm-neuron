"""`layers.21.attention` on 4 ranks: head-parallel attention over a sequence-parallel prefix.

The module is split two different ways, in the two places where a different axis is free:

* **The prefix is sequence-parallel by four.** `attn_norm`, the fused `wqkv` GEMM, `kv_norm`, the
  KV RoPE and its activation quantization, and `q_norm` have *no head axis to cut* —
  `head_dim = 512` is one shared KV latent for every head — so a head split leaves all four ranks
  computing the same thing, about 19% of a rank's wall, four times over. A rank instead takes
  2048 contiguous token rows of it and the two latents (`qr`, 1280 wide, and the quantized KV
  latent, 512 wide) are rejoined by **one coalesced `ncc.all_gather`**. Measured worth 1.42 ms of
  non-collective time against 0.19 ms of extra collective: 19.045 -> 17.758 ms, with bit-identical
  numerics, since it is the same computation redistributed.
* **Everything after it is head-parallel**, per `FLOORPLAN.md`'s `headx4(allreduce)`: `wq_b`, the
  attention itself over this rank's 16 of 64 heads and all 8192 queries, the block-diagonal
  `wo_a`, and the `wo_b` partial that one `ncc.all_reduce` sums.

The shard is *contiguous* and not interleaved on purpose. `all_gather` with `collective_dim=0`
concatenates rank-major, so a contiguous shard makes the gathered buffer already be the module's
own token order: no permute, and the sliding window's 128-row pre-window stops being a tile every
core recomputes and becomes an index into a complete buffer.

Why the attention is *not* also sequence-parallel, measured rather than assumed: a dense
compressed sweep is causal, so query group `g` sweeps `QT * (g + 1)` position blocks and that trip
count is a trace-time constant while a rank's identity is not, so a rank-dependent set of query
rows costs the worst case over the four ranks. The fix for that is a gathered sweep — exactly 512
positions for every query, hence identical on every rank — and an indirect-DMA gather of the
selected rows costs one software-generated descriptor per row, **measured 16 ns a row, 8.4 ms a
rank**, against a ceiling of 19.2131. `.autohelix/notes/iter-2.md` has the full ledger, including
the two alternative gathers that were measured and are dead.

Three things the seam adds, and nothing else:

1. **The output seam.** The rank's `wo_b` partial goes into a named `nl.shared_hbm` scratch buffer,
   `nki.collectives.all_reduce` sums the four ranks' partials *inside the traced graph*, and the
   sum is copied into the returned tensor. `FLOORPLAN.md` is binding on the split —
   `headx4(allreduce)` — and `submodule/SUBMODULE.md` is what reads out of
   `reference/reference_torch.py` that a rank's output is a partial sum of `wo_b` and that the
   four partials reassemble by addition.

2. **The rank-indexed weight slicing and the rank-indexed input rows, both here and not in the
   validator.** Every rank
   is handed the module's *whole* recorded weight set and this kernel picks its own quarter of
   `wq_b`, `wo_a`, `wo_b` and `attn_sink` out of it. The dimensions are the ones
   `reference/vendor/convert.py`'s `mapping` shards the checkpoint on — `wq_b` dim 0, `wo_a`
   dim 0, `wo_b` dim 1, `attn_sink` dim 0, each rank taking `narrow(dim, rank * size, size)` —
   and each `scale` follows its weight, one e8m0 exponent per 32x32 block. Every boundary (16,
   64, 256, 2048, 8192) is a multiple of 32, so no scale block straddles two ranks.

   This is deliberate. `inference.py` is frozen for the whole optimization loop that follows, so
   slicing the weights there would hard-wire the head split into the one file the loop cannot
   edit: an iteration that wanted to divide the module some other way — sequence-parallel by
   four, with every rank holding all 64 heads — would be blocked by the validator rather than by
   the hardware. An argument a kernel does not read costs nothing at run time.

3. **A bfloat16 partial** instead of the single-rank kernel's float32 one, because the collective
   is bandwidth-bound at this size and the dtype is the whole lever on it. `ASSEMBLY.md` records
   both measurements.

A rank's identity lives entirely in the *values* `inference.py` hands this kernel, never in the
code: one NEFF runs on all four ranks. That is what lets `neuron-explorer capture` replay the same
NEFF on four workers, and it keeps the four ranks' schedules identical. A rank's row range is
therefore *not* a trace-time constant, unlike a physical core's `nl.program_id(0)`: it comes from
`ncc.rank_id()`, a runtime register, used as the `scalar_offset` of the staging DMAs that feed the
dequantizer and, for the sequence-parallel prefix, the two reads of a whole-module tensor it needs
(`hidden_states` and `freqs_cis`, through `_rap`). Everything downstream of those DMAs indexes
trace-time constants again — every prefix intermediate is a compact per-rank buffer addressed by
*local* row — so no other helper in the file had to learn about dynamic addressing.

A collective's operands are whole buffers, so the four involved here come in two pairs: `qr_src` /
`kv_src`, exactly one rank's 2048 rows of each latent, and `qr_dst` / `kv_dst`, exactly all 8192.
The two latents are gathered by **one** call with two `srcs` and two `dsts`: two `all_gather`
calls back to back whose destinations are both read later leave the first destination corrupt.

Six things about `nki.collectives` on this toolchain, each of which otherwise fails with an
internal compiler error that names something else:

1. the kernel is launched as `kernel[2]`, because this host is LNC=2 and a LNC=1 NEFF leaves the
   collective's buffer unallocated on the logical core's second physical core (`NCC_ILLC059`);
2. `cc_src` / `cc_dst` carry `name=`, or DRAM allocation fails (`NCC_IBIR440`);
3. they are scratch and not the returned tensor, because a collective may not read or write an IO
   tensor (`NCC_INLA001`) — hence the copy out;
4. both are `nl.shared_hbm`, because buffer kinds may not be mixed;
5. the `ReplicaGroup` is built by `inference.py` and passed in, because the tracer rejects `range`
   in a traced body;
6. the inputs are built on CPU and `.to(device)`'d, because `torch.full(..., device=xla)` emits a
   broadcast HLO that hits `NCC_ISMP902` — and that error is the broadcast, not the collective.

A collective operand must also be a *whole* buffer: a destination sliced at a non-zero row is
refused with `Coalesced ccop: AP offsets must be zero` and a view at offset zero with `Coalesced
ccop Output does not write full tensor`. That is why `cc_src` and `cc_dst` are each exactly the
collective's shape rather than windows onto something larger.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

SEQ = 8192
DIM = 5120
HEAD_DIM = 512
# This rank's share: `n_heads // world_size` and `o_groups // world_size`. `HEADS_PER_GROUP` stays
# 8 because both divide by 4 together.
N_HEADS = 16
ROPE_DIM = 64
Q_LORA = 1280
O_LORA = 1024
O_GROUPS = 2
WINDOW = 128
RATIO = 1
CSEQ = SEQ // RATIO
CACHE_ROWS = 4 * (16384 // RATIO)
INDEX_TOPK = 512
EPS = 1e-20
SM_SCALE = HEAD_DIM ** -0.5
NEG_BOUND = -1.0e30
HEADS_PER_GROUP = N_HEADS // O_GROUPS

PT = 128
NT = 512
WIN_SPAN = 2 * PT

FP8_BLK = 32
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4
QDQ_CHUNK = 1280

MGROUP_BYTES = 64 * 1024
WO_B_SPLIT = 1920
XLOOK = 2

MASK_DT = nl.float8_e4m3fn

# --- the four ranks -----------------------------------------------------------------------------
# `N_HEADS`, `O_GROUPS` and `attn_sink`'s width above are already this rank's share; `WORLD_SIZE`
# is what turns a whole recorded weight into that share.
WORLD_SIZE = 4
# The prefix is sequence-parallel by four where everything after it is head-parallel: a rank
# owns 2048 of the 8192 token rows of `attn_norm`, the fused `wqkv` GEMM, `kv_norm`, the KV RoPE
# and its quantization, and `q_norm`. Contiguous and not interleaved, because `all_gather`
# concatenates rank-major and a contiguous shard makes that concatenation already be global
# token order -- no permute afterwards, and the sliding window reads a complete buffer.
LROWS = SEQ // WORLD_SIZE
# One physical core's share of that. The literal 2 is `CORES`, which is declared further down
# next to the thing that consumes it; this is the same 2.
PCROWS = LROWS // 2
# The module's output dtype: the recorded output's own, so the chain rounds to bfloat16 exactly
# once however the partials are carried.
OUT_DTYPE = nl.bfloat16
# The dtype a rank's `wo_b` partial is written in and the four partials are reduced in. A literal
# reading of `RowParallelLinear` reduces in float32 (`y = y.float(); dist.all_reduce(y)`), and that
# is the more accurate of the two by 1.5e-6 of cosine against the recorded output. bfloat16 ships
# because the collective is bandwidth-bound at 83.9 MB and float32 doubles it; `ASSEMBLY.md` has
# both numbers and the headroom each leaves under the bar.
CC_DTYPE = nl.bfloat16
# Rows per DMA of the post-collective copy. 1024 x 5120 bfloat16 = 10 MiB a descriptor.
CC_ROWS = 1024


def _blocks(n, step):
    out = []
    off = 0
    while off < n:
        out.append((off, min(step, n - off)))
        off += step
    return out


def _rank_id():
    """This rank's index as a 1x1 int32 SBUF scalar, usable as a DMA `scalar_offset`."""
    rid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rid, ncc.rank_id())
    return rid


def _shard0(t, rank_rows, cols, rid, row, rows):
    """`rows` rows at `row` of this rank's `(rank_rows, cols)` block of a dim-0 sharded tensor.

    The whole recorded tensor is `(WORLD_SIZE * rank_rows, cols)`; viewed as
    `(WORLD_SIZE, rank_rows, cols)` the rank index scales by `rank_rows * cols`, which is what
    `indirect_dim=0` does with the scalar offset. The returned view is the `src` of one DMA.
    """
    return t.reshape((WORLD_SIZE, rank_rows, cols)).ap(
        pattern=[[cols, rows], [1, cols]], offset=row * cols,
        scalar_offset=rid, indirect_dim=0)


def _shard1(t, rows_all, rank_cols, rid, row, rows):
    """`rows` rows at `row` of this rank's `(rows_all, rank_cols)` block of a dim-1 sharded tensor.

    A column shard needs no new mechanism: element `(i, rank * rank_cols + j)` of the recorded
    `(rows_all, WORLD_SIZE * rank_cols)` tensor is row `i * WORLD_SIZE + rank` of the same storage
    viewed as `(rows_all * WORLD_SIZE, rank_cols)`, so the rank index again scales by dim 0's
    stride and consecutive rows are `WORLD_SIZE` apart. The run length per row is unchanged —
    `rank_cols` contiguous elements either way — so this costs the DMA no extra descriptors.
    """
    return t.reshape((rows_all * WORLD_SIZE, rank_cols)).ap(
        pattern=[[WORLD_SIZE * rank_cols, rows], [1, rank_cols]],
        offset=row * WORLD_SIZE * rank_cols, scalar_offset=rid, indirect_dim=0)


def _shard_src(t, shard, n, k, rid, row, rows):
    """`_shard0` or `_shard1`, chosen by which dimension the checkpoint shards this tensor on."""
    if shard == 0:
        return _shard0(t, n, k, rid, row, rows)
    return _shard1(t, n, k, rid, row, rows)


def _live_pos_blocks(t):
    return -(-(PT * (t + 1) // RATIO) // PT)


def _live_rank_tiles(t):
    return min(INDEX_TOPK // PT, -(-(PT * (t + 1) // RATIO) // PT))


def _rap(t, cols, rid, lrow, rows=PT):
    """`rows` rows of a whole-module tensor at local row `lrow` of *this rank's* row block.

    `ncc.rank_id()` is a register, not a trace-time constant, so a rank's row range cannot be a
    slice. Viewed as `(rows_all / LROWS, LROWS, cols)` the rank index scales by `LROWS * cols`,
    which is what `indirect_dim=0` does with a scalar offset, so the row this reads is
    `rank * LROWS + lrow`. One NEFF still serves all four ranks.
    """
    rows_all = 1
    for d in t.shape:
        rows_all *= d
    rows_all //= cols
    return t.reshape((rows_all // LROWS, LROWS, cols)).ap(
        pattern=[[cols, rows], [1, cols]], offset=lrow * cols,
        scalar_offset=rid, indirect_dim=0)


def _load_bcast(vec_hbm, n, rows, dt=nl.float32):
    wb = nl.ndarray((rows, n), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wb, src=nl.broadcast_to(
        nl.load(vec_hbm.reshape((1, n)), dtype=dt), (rows, n)))
    return wb


def _load_bcast_shard(vec_hbm, n, rows, rid, dt=nl.float32):
    """`_load_bcast` of this rank's `n` of `WORLD_SIZE * n` elements — `attn_sink`, one per head."""
    v = nl.ndarray((1, n), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=v, src=vec_hbm.reshape((WORLD_SIZE, n)).ap(
        pattern=[[n, 1], [1, n]], offset=0, scalar_offset=rid, indirect_dim=0))
    wb = nl.ndarray((rows, n), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wb, src=nl.broadcast_to(v, (rows, n)))
    return wb


def _rmsnorm(tile, wb, n, rows=PT, out_dtype=nl.float32):
    sq = nl.ndarray((rows, n), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    ssq = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=tile, bias=0.0, scale=1.0,
                    reduce_op=nl.add, reduce_res=ssq,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    ms = nl.add(nl.multiply(ssq, 1.0 / n), EPS)
    y = nl.rsqrt(ms)
    y = nl.multiply(y, nl.subtract(1.5, nl.multiply(nl.multiply(ms, 0.5), nl.multiply(y, y))))
    out = nl.ndarray((rows, n), dtype=out_dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=out, data=tile, op0=nl.multiply, operand0=y,
                              op1=nl.multiply, operand1=wb)
    return out


def _pow2_ceil_log2(x):
    bits = nl.copy(x, dtype=nl.float32).view(nl.int32)
    exp = nl.bitwise_and(nl.right_shift(bits, 23), 0xFF)
    man = nl.bitwise_and(bits, 0x7FFFFF)
    bump = nl.copy(nl.greater(man, 0), dtype=nl.int32)
    e = nl.add(exp, bump)
    scale = nl.copy(nl.left_shift(e, 23)).view(nl.float32)
    inv = nl.copy(nl.left_shift(nl.subtract(254, e), 23)).view(nl.float32)
    return scale, inv


FP8_PEAK = FP8_MAX / 2.0


def _decode_e4m3(wbytes, rows, k):
    dn = nl.ndarray((rows, k), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dn, data=nl.bitwise_and(wbytes, 0x7F), op0=nl.maximum,
                       operand0=8, op1=nl.subtract, operand1=8)
    b2 = nl.ndarray((rows, k), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=b2, data=wbytes, op0=nl.bitwise_and, operand0=0x80,
                              op1=nl.bitwise_or, operand1=dn)
    val = nl.ndarray((rows, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=val, op=nl.copy, data=b2.view(nl.float8_e4m3fn),
                    bias=0.0, scale=1.0)
    return val


def _qdq_fp8(tile, rows, n):
    nb = n // FP8_BLK
    blocked = tile.reshape((rows, nb, FP8_BLK))
    ab = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=blocked, bias=0.0, scale=1.0)
    amax = nl.maximum(nl.copy(nl.max(ab, axis=2, keepdims=True), dtype=nl.float32),
                      FP8_AMAX_FLOOR)
    scale, rcp = _pow2_ceil_log2(nl.multiply(amax, 1.0 / FP8_PEAK))
    rcp_b = nl.copy(rcp, dtype=nl.bfloat16)
    scale_b = nl.copy(scale, dtype=nl.bfloat16)
    q8 = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q8, data1=blocked, op=nl.multiply,
                       data2=nl.broadcast_to(rcp_b, (rows, nb, FP8_BLK)))
    deq = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=deq, data1=q8, op=nl.multiply,
                       data2=nl.broadcast_to(scale_b, (rows, nb, FP8_BLK)))
    return deq.reshape((rows, n))


def _act_chunk(k):
    for c in (QDQ_CHUNK, 1024, 512, 256, 128):
        if k % c == 0:
            return c
    return k


def _scale_table(s_u8, n, k, shard=None, rid=None):
    """The `(n/32, k/32)` block scales, widened to bfloat16 powers of two in HBM.

    `shard` is `None` for a replicated weight — `s_u8` is then exactly this rank's scale block.
    Otherwise `s_u8` is the whole recorded scale and the rank's rows (`shard=0`) or columns
    (`shard=1`) arrive through one `scalar_offset` DMA per tile. The sharded path reads the bytes
    as `uint8` and widens them on-chip rather than asking the DMA to widen, which is the one form
    with no doubt about what the transfer does with a narrower source.
    """
    kb = k // FP8_BLK
    nsb = n // FP8_BLK
    out = nl.ndarray((nsb, kb), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for r0 in range(0, nsb, PT):
        rr = min(PT, nsb - r0)
        if shard is None:
            sb = nl.load(s_u8[r0:r0 + rr, 0:kb], dtype=nl.int32)
        else:
            raw = nl.ndarray((rr, kb), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=_shard_src(s_u8, shard, nsb, kb, rid, r0, rr))
            sb = nl.copy(raw, dtype=nl.int32)
        one = nl.copy(nl.left_shift(nl.add(sb, 1), 23)).view(nl.float32)
        nl.store(out[r0:r0 + rr, 0:kb], value=nl.copy(one, dtype=nl.bfloat16))
    return out


def _dequant_weight(w_u8, s_u8, dst_hbm, n, k, dst_row=0, n0=0, n1=None,
                    shard=None, rid=None):
    """Widen a block-scaled fp8 weight to bfloat16 in HBM, `rows` rows at a time.

    `shard=0` means the recorded tensor is `(WORLD_SIZE * n, k)` and this rank owns rows
    `rank * n ...`; `shard=1` means it is `(n, WORLD_SIZE * k)` and this rank owns columns
    `rank * k ...`. In both cases the row range is a runtime quantity, so the weight-byte load
    becomes an explicit DMA with `rid` as its `scalar_offset`. `n`/`k` always describe *this
    rank's* block, so every index below — and every caller — is unchanged by the split.
    """
    kb = k // FP8_BLK
    rows = PT if n % PT == 0 else FP8_BLK
    nsb = rows // FP8_BLK
    n1 = n if n1 is None else n1
    sf_hbm = _scale_table(s_u8, n, k, shard=shard, rid=rid)
    jq_i = nl.ndarray((nsb, rows), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=jq_i, pattern=[[1, rows]], offset=0, channel_multiplier=0)
    pi_i = nl.ndarray((nsb, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pi_i, pattern=[[1, 1]], offset=0, channel_multiplier=FP8_BLK)
    dq = nl.ndarray((nsb, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dq, data=nl.copy(jq_i, dtype=nl.float32), op0=nl.subtract,
                       operand0=nl.copy(pi_i, dtype=nl.float32))
    sel = nl.multiply(nl.copy(nl.greater_equal(dq, 0.0), dtype=nl.bfloat16),
                      nl.copy(nl.less(dq, float(FP8_BLK)), dtype=nl.bfloat16))
    for nb in nl.affine_range((n1 - n0) // rows):
        rb = nb + n0 // rows
        if shard is None:
            wraw = nl.load(w_u8[rb * rows:rb * rows + rows, 0:k], dtype=nl.uint8)
        else:
            wraw = nl.ndarray((rows, k), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=wraw,
                          src=_shard_src(w_u8, shard, n, k, rid, rb * rows, rows))
        wq = _decode_e4m3(wraw, rows, k)
        sp = nl.ndarray((rows, kb), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=sp, stationary=sel,
                       moving=nl.load(sf_hbm[rb * nsb:rb * nsb + nsb, 0:kb]),
                       is_stationary_onezero=True, accumulate=False)
        sf = nl.copy(sp, dtype=nl.bfloat16)
        rep = nl.broadcast_to(sf.reshape((rows, kb, 1)), (rows, kb, FP8_BLK))
        prod = nl.ndarray((rows, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=wq, data2=rep.reshape((rows, k)), op=nl.multiply)
        nl.store(dst_hbm[dst_row + rb * rows:dst_row + rb * rows + rows, 0:k], value=prod)


def _rope_inplace(t_hbm, freqs_hbm, m, width, pos_stride, inverse, heads=1,
                  row=0, hstride=None):
    half = ROPE_DIM // 2
    lo = width - ROPE_DIM
    hst = m if hstride is None else hstride
    for mb in nl.affine_range(m // PT):
        p0 = (row + mb * PT) * pos_stride
        p1 = p0 + PT * pos_stride
        cs = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 0:2], dtype=nl.float32)
        cos = nl.ndarray((PT, heads, half), dtype=nl.float32, buffer=nl.sbuf)
        sin = nl.ndarray((PT, heads, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cos, src=nl.broadcast_to(
            nl.copy(cs[0:PT, 0:half, 0]).reshape((PT, 1, half)), (PT, heads, half)))
        nisa.tensor_copy(dst=sin, src=nl.broadcast_to(
            nl.copy(cs[0:PT, 0:half, 1]).reshape((PT, 1, half)), (PT, heads, half)))
        t = nl.ndarray((PT, heads, ROPE_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        for hb in nl.static_range(heads):
            r0 = hb * hst + row + mb * PT
            nisa.tensor_copy(dst=t[0:PT, hb, 0:ROPE_DIM],
                             src=nl.load(t_hbm[r0:r0 + PT, lo:width]))
        t4 = t.reshape((PT, heads, half, 2))
        a = t4[0:PT, 0:heads, 0:half, 0]
        b = t4[0:PT, 0:heads, 0:half, 1]
        o = nl.ndarray((PT, heads, ROPE_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        o4 = o.reshape((PT, heads, half, 2))

        def _mulf(x, y):
            r = nl.ndarray((PT, heads, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=x, data2=y, op=nl.multiply)
            return r

        ac = _mulf(a, cos)
        bs = _mulf(b, sin)
        as_ = _mulf(a, sin)
        bc = _mulf(b, cos)
        if inverse:
            nisa.tensor_copy(dst=o4[0:PT, 0:heads, 0:half, 0], src=nl.add(ac, bs))
            nisa.tensor_copy(dst=o4[0:PT, 0:heads, 0:half, 1], src=nl.subtract(bc, as_))
        else:
            nisa.tensor_copy(dst=o4[0:PT, 0:heads, 0:half, 0], src=nl.subtract(ac, bs))
            nisa.tensor_copy(dst=o4[0:PT, 0:heads, 0:half, 1], src=nl.add(as_, bc))
        for hb in nl.static_range(heads):
            r0 = hb * hst + row + mb * PT
            nl.store(t_hbm[r0:r0 + PT, lo:width], value=o[0:PT, hb, 0:ROPE_DIM])


def _qdq_rows(src_hbm, dst_hbm, m, n, dst_row=0, src_row=0):
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[src_row + mb * PT:src_row + (mb + 1) * PT, 0:n],
                    dtype=nl.bfloat16)
        nl.store(dst_hbm[dst_row + mb * PT:dst_row + (mb + 1) * PT, 0:n],
                 value=_qdq_fp8(t, PT, n))


def _mgroup(mtiles, ktiles, budget=None):
    cap = MGROUP_BYTES if budget is None else budget
    mg = 1
    while mg * 2 <= mtiles and (mg * 2) * ktiles * PT * 2 <= cap:
        mg *= 2
    return mg


def _gemm(x_hbm, w_hbm, dst_hbm, m, k, n, out_dtype=nl.bfloat16,
          x_row=0, x_col=0, w_row=0, dst_col=0, dst_row=0, qdq=False, mg_bytes=None):
    ktiles = k // PT
    if n <= NT:
        nt = n
    elif n % NT == 0:
        nt = NT
    elif n % 256 == 0:
        nt = 256
    else:
        nt = PT
    mg = _mgroup(m // PT, ktiles, mg_bytes)
    ntiles = n // nt
    x3 = x_hbm.reshape((x_hbm.shape[0], x_hbm.shape[1] // PT, PT))
    w3 = w_hbm.reshape((w_hbm.shape[0], w_hbm.shape[1] // PT, PT))
    xc0 = x_col // PT
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)

        def stage_x(mi):
            r0 = x_row + (mgb * mg + mi) * PT
            nisa.dma_transpose(dst=xt[0:PT, mi, 0:ktiles, 0:PT],
                               src=x3[r0:r0 + PT, xc0:xc0 + ktiles, 0:PT])

        def ntile(nb, jit):
            wt = nl.ndarray((PT, ktiles, nt), dtype=nl.bfloat16, buffer=nl.sbuf)

            nisa.dma_transpose(dst=wt[0:PT, 0:ktiles, 0:nt],
                               src=w3[w_row + nb * nt:w_row + (nb + 1) * nt, 0:ktiles, 0:PT])
            for mi in nl.static_range(mg):
                if jit and mi + XLOOK < mg:
                    stage_x(mi + XLOOK)
                acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, mi, kb, 0:PT],
                                   moving=wt[0:PT, kb, 0:nt], accumulate=(kb > 0))
                row = nl.copy(acc, dtype=out_dtype)
                nl.store(dst_hbm[dst_row + (mgb * mg + mi) * PT:
                                 dst_row + (mgb * mg + mi + 1) * PT,
                                 dst_col + nb * nt:dst_col + (nb + 1) * nt],
                         value=_qdq_fp8(row, PT, nt) if qdq else row)

        for mi0 in nl.static_range(min(XLOOK, mg)):
            stage_x(mi0)
        ntile(0, mg > XLOOK)
        if ntiles > 1:
            for nb in nl.affine_range(ntiles - 1):
                ntile(nb + 1, False)


def _gemm_head_major(x_hbm, w_hbm, dst_hbm, m, k, n, hdim, row=0, hstride=None,
                     mg_bytes=None):
    ktiles = k // PT
    hst = m if hstride is None else hstride
    mg = _mgroup(m // PT, ktiles, mg_bytes)
    ntiles = n // hdim
    x3 = x_hbm.reshape((x_hbm.shape[0], x_hbm.shape[1] // PT, PT))
    w3 = w_hbm.reshape((w_hbm.shape[0], w_hbm.shape[1] // PT, PT))
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)

        def stage_x(mi):
            r0 = row + (mgb * mg + mi) * PT
            nisa.dma_transpose(dst=xt[0:PT, mi, 0:ktiles, 0:PT],
                               src=x3[r0:r0 + PT, 0:ktiles, 0:PT])

        def ntile(nb, jit):
            wt = nl.ndarray((PT, ktiles, hdim), dtype=nl.bfloat16, buffer=nl.sbuf)

            nisa.dma_transpose(dst=wt[0:PT, 0:ktiles, 0:hdim],
                               src=w3[nb * hdim:(nb + 1) * hdim, 0:ktiles, 0:PT])
            for mi in nl.static_range(mg):
                if jit and mi + XLOOK < mg:
                    stage_x(mi + XLOOK)
                acc = nl.ndarray((PT, hdim), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, mi, kb, 0:PT],
                                   moving=wt[0:PT, kb, 0:hdim], accumulate=(kb > 0))
                r = nb * hst + row + (mgb * mg + mi) * PT
                nl.store(dst_hbm[r:r + PT, 0:hdim], value=nl.copy(acc, dtype=nl.bfloat16))

        for mi0 in nl.static_range(min(XLOOK, mg)):
            stage_x(mi0)
        ntile(0, mg > XLOOK)
        if ntiles > 1:
            for nb in nl.affine_range(ntiles - 1):
                ntile(nb + 1, False)


def _norm_rows(src_hbm, w_hbm, dst_hbm, m, n, src_col=0, row=0):
    wb = _load_bcast(w_hbm, n, PT, dt=nl.bfloat16)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, src_col:src_col + n],
                    dtype=nl.bfloat16)
        nl.store(dst_hbm[row + mb * PT:row + (mb + 1) * PT, 0:n],
                 value=_rmsnorm(t, wb, n, out_dtype=nl.bfloat16))


def _norm_quant_rows(src_hbm, w_hbm, dst_hbm, m, n, src_col=0, row=0):
    wb = _load_bcast(w_hbm, n, PT, dt=nl.bfloat16)
    chunk = _act_chunk(n)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, src_col:src_col + n],
                    dtype=nl.bfloat16)
        y = _rmsnorm(t, wb, n, out_dtype=nl.bfloat16)
        for cb in nl.static_range(n // chunk):
            nl.store(dst_hbm[row + mb * PT:row + (mb + 1) * PT,
                             cb * chunk:(cb + 1) * chunk],
                     value=_qdq_fp8(y[0:PT, cb * chunk:(cb + 1) * chunk], PT, chunk))


def _norm_quant_rows_r(src_hbm, w_hbm, dst_hbm, m, n, rid, lrow):
    """`attn_norm` over this rank's rows: the one read of a module input that is rank-addressed.

    The destination is a compact per-rank buffer addressed by *local* row, so no helper below
    this one learns that the sequence is split.
    """
    wb = _load_bcast(w_hbm, n, PT, dt=nl.bfloat16)
    chunk = _act_chunk(n)
    for mb in range(m // PT):
        t = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=_rap(src_hbm, n, rid, lrow + mb * PT))
        y = _rmsnorm(t, wb, n, out_dtype=nl.bfloat16)
        for cb in nl.static_range(n // chunk):
            nl.store(dst_hbm[lrow + mb * PT:lrow + (mb + 1) * PT,
                             cb * chunk:(cb + 1) * chunk],
                     value=_qdq_fp8(y[0:PT, cb * chunk:(cb + 1) * chunk], PT, chunk))


def _rope_og(og3_hbm, freqs_hbm, m=SEQ, row=0):
    half = ROPE_DIM // 2
    lo = HEAD_DIM - ROPE_DIM
    for mb in nl.affine_range(m // PT):
        q0 = row + mb * PT
        cs = nl.load(freqs_hbm[q0:q0 + PT, 0:half, 0:2], dtype=nl.float32)
        cos = nl.ndarray((PT, N_HEADS, half), dtype=nl.float32, buffer=nl.sbuf)
        sin = nl.ndarray((PT, N_HEADS, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cos, src=nl.broadcast_to(
            nl.copy(cs[0:PT, 0:half, 0]).reshape((PT, 1, half)), (PT, N_HEADS, half)))
        nisa.tensor_copy(dst=sin, src=nl.broadcast_to(
            nl.copy(cs[0:PT, 0:half, 1]).reshape((PT, 1, half)), (PT, N_HEADS, half)))
        t = nl.load(og3_hbm[q0:q0 + PT, 0:N_HEADS, lo:HEAD_DIM],
                    dtype=nl.bfloat16).reshape((PT, N_HEADS, half, 2))
        a = t[0:PT, 0:N_HEADS, 0:half, 0]
        b = t[0:PT, 0:N_HEADS, 0:half, 1]
        o = nl.ndarray((PT, N_HEADS, half, 2), dtype=nl.bfloat16, buffer=nl.sbuf)

        def _mulf(x, y):
            r = nl.ndarray((PT, N_HEADS, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=x, data2=y, op=nl.multiply)
            return r

        nisa.tensor_tensor(dst=o[0:PT, 0:N_HEADS, 0:half, 0],
                           data1=_mulf(a, cos), data2=_mulf(b, sin), op=nl.add)
        nisa.tensor_tensor(dst=o[0:PT, 0:N_HEADS, 0:half, 1],
                           data1=_mulf(b, cos), data2=_mulf(a, sin), op=nl.subtract)
        nl.store(og3_hbm[q0:q0 + PT, 0:N_HEADS, lo:HEAD_DIM],
                 value=o.reshape((PT, N_HEADS, ROPE_DIM)))


QT = 2
QG = QT * PT
GQ = 8
HD2 = HEAD_DIM // 2


def _sparse_attention_impl(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs, rid):
    dt = HEAD_DIM // PT
    WB = QT + 1
    ncs = [_live_pos_blocks((gb + 1) * QT - 1) for gb in gbs]
    ncmax = max(ncs)
    nrtmax = max(_live_rank_tiles((gb + 1) * QT - 1) for gb in gbs)

    q3 = q_hbm.reshape((N_HEADS * SEQ, dt, PT))
    kvp3 = kvp_hbm.reshape((kvp_hbm.shape[0], dt, PT))
    ckv3 = ckv_hbm.reshape((ckv_hbm.shape[0], dt, PT))

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
    eye = nl.copy(nl.equal(dio[0:PT, 0:PT], 0.0), dtype=nl.bfloat16)
    # This rank's 16 of 64 sink values — one per local head, `attn_sink` sharded on dim 0.
    sinkb = _load_bcast_shard(sink_hbm, N_HEADS, PT, rid)

    pio2_i = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pio2_i, pattern=[[1, PT]], offset=0, channel_multiplier=0)
    pio2 = nl.copy(pio2_i, dtype=nl.bfloat16)
    pio2g = nl.ndarray((PT, GQ, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pio2g, src=pio2.ap(pattern=[[PT, PT], [0, GQ], [1, PT]]))
    cio2_i = nl.ndarray((PT, ncmax), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=cio2_i, pattern=[[PT, ncmax]], offset=SEQ, channel_multiplier=0)
    cio2 = nl.copy(cio2_i, dtype=nl.bfloat16)

    ckv_t = nl.ndarray((PT, dt, ncmax * PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    ckv_n = nl.ndarray((PT, ncmax, HEAD_DIM + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    ones_c = nl.copy(nl.greater_equal(cio2_i, 0), dtype=nl.bfloat16)
    nisa.tensor_copy(dst=ckv_n[0:PT, 0:ncmax, HD2:HD2 + 1],
                     src=ones_c.reshape((PT, ncmax, 1)))
    for gp in nl.affine_range(ncmax):
        nisa.tensor_copy(dst=ckv_n[0:PT, gp, 0:HD2],
                         src=nl.load(ckv_hbm[gp * PT:(gp + 1) * PT, 0:HD2]))
        nisa.tensor_copy(dst=ckv_n[0:PT, gp, HD2 + 1:HEAD_DIM + 1],
                         src=nl.load(ckv_hbm[gp * PT:(gp + 1) * PT, HD2:HEAD_DIM]))
        nisa.dma_transpose(dst=ckv_t[0:PT, 0:dt, gp * PT:(gp + 1) * PT],
                           src=ckv3[gp * PT:(gp + 1) * PT, 0:dt, 0:PT])

    wadds = []
    for first in range(2):
        live = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=live, data=nonneg, op0=nl.maximum,
                           operand0=float(1 - first))
        wadds.append(nl.multiply(nl.subtract(nl.multiply(live, inwin), 1.0), -NEG_BOUND))

    win_t = nl.ndarray((PT, dt, WB * PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    win_n = nl.ndarray((PT, WB, HEAD_DIM + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=win_n[0:PT, 0:WB, HD2:HD2 + 1],
                     src=nl.copy(nl.less(cio2_i[0:PT, 0:WB], 0),
                                 dtype=nl.bfloat16).reshape((PT, WB, 1)))
    loT = nl.ndarray((PT, nrtmax, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    hiT = nl.ndarray((PT, nrtmax, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    maskTs = [nl.ndarray((PT, ncmax, QG), dtype=MASK_DT, buffer=nl.sbuf) for _ in range(2)]
    qt4 = nl.ndarray((PT, dt, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    pexps = [nl.ndarray((PT, QG), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    pblks = [nl.ndarray((PT, QG), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    pwT = nl.ndarray((PT, 2, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    pall = nl.ndarray((PT, nrtmax, GQ, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    call = nl.ndarray((PT, nrtmax, GQ, ncmax), dtype=nl.bfloat16, buffer=nl.sbuf)

    def win_prep(gb):
        qg0 = gb * QG
        for wb in nl.static_range(WB):
            r0 = qg0 + wb * PT
            nisa.tensor_copy(dst=win_n[0:PT, wb, 0:HD2],
                             src=nl.load(kvp_hbm[r0:r0 + PT, 0:HD2]))
            nisa.tensor_copy(dst=win_n[0:PT, wb, HD2 + 1:HEAD_DIM + 1],
                             src=nl.load(kvp_hbm[r0:r0 + PT, HD2:HEAD_DIM]))
            nisa.dma_transpose(dst=win_t[0:PT, 0:dt, wb * PT:(wb + 1) * PT],
                               src=kvp3[r0:r0 + PT, 0:dt, 0:PT])

    def key_prep(gb, nrt_t):
        qg0 = gb * QG
        kw = nrt_t * PT
        for ti in nl.static_range(QT):
            q0 = qg0 + ti * PT
            tki = nl.load(topk_hbm[q0:q0 + PT, 0:kw], dtype=nl.int32)
            lo_i = nl.bitwise_and(tki, PT - 1)
            lo_b = nl.copy(lo_i, dtype=nl.bfloat16)
            hi_b = nl.ndarray((PT, kw), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=hi_b, data1=tki, data2=lo_i, op=nl.subtract)
            for rt in nl.static_range(nrt_t):
                tp = nl.ndarray((PT, 2, PT), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=tp[0:PT, 0, 0:PT],
                               stationary=hi_b[0:PT, rt * PT:(rt + 1) * PT],
                               moving=eye, accumulate=False)
                nisa.nc_matmul(dst=tp[0:PT, 1, 0:PT],
                               stationary=lo_b[0:PT, rt * PT:(rt + 1) * PT],
                               moving=eye, accumulate=False)
                nisa.tensor_copy(dst=hiT[0:PT, rt, ti * PT:(ti + 1) * PT],
                                 src=tp[0:PT, 0, 0:PT])
                nisa.tensor_copy(dst=loT[0:PT, rt, ti * PT:(ti + 1) * PT],
                                 src=tp[0:PT, 1, 0:PT])

    def mask_slice(nrt_t, nc, qblk, mdst):
        qb = qblk * GQ
        for rt in nl.static_range(nrt_t):
            # The broadcast of one query group's low index bits is read as an access pattern
            # rather than materialized: the `call` build two lines down already takes a
            # zero-stride AP on *both* operands, so a `tensor_copy` into a staging tile first was
            # one (128, 8, 128) write a rank tile a slice -- about 40% of the whole mask build's
            # element traffic, and 2048 instructions a core -- for nothing.
            nisa.tensor_tensor(
                dst=pall[0:PT, rt, 0:GQ, 0:PT], op=nl.equal,
                data1=pio2g[0:PT, 0:GQ, 0:PT],
                data2=loT.ap(pattern=[[nrtmax * QG, PT], [1, GQ], [0, PT]],
                             offset=rt * QG + qb))
            nisa.tensor_tensor(
                dst=call[0:PT, rt, 0:GQ, 0:nc], op=nl.equal,
                data1=cio2.ap(pattern=[[ncmax, PT], [0, GQ], [1, nc]]),
                data2=hiT.ap(pattern=[[nrtmax * QG, PT], [1, GQ], [0, nc]],
                             offset=rt * QG + qb))
        for g in nl.static_range(GQ):
            mp = nl.ndarray((PT, nc), dtype=nl.float32, buffer=nl.psum)
            for rt in nl.static_range(nrt_t):
                nisa.nc_matmul(dst=mp, stationary=pall[0:PT, rt, g, 0:PT],
                               moving=call[0:PT, rt, g, 0:nc], accumulate=(rt > 0))
            nisa.tensor_copy(dst=mdst[0:PT, 0:nc, qb + g], src=mp,
                             engine=nisa.scalar_engine)

    def head_group(gb, h, nc, wadd0, maskT, inject=None):
        qg0 = gb * QG
        nisa.dma_transpose(dst=qt4[0:PT, 0:dt, 0:QG],
                           src=q3[h * SEQ + qg0:h * SEQ + qg0 + QG, 0:dt, 0:PT])
        esink = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=esink, op=nl.exp, data=sinkb[0:PT, h:h + 1],
                        bias=0.0, scale=1.0)
        dws = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.sbuf)
        accs = []
        for ti in nl.static_range(QT):
            wadd = wadds[1] if (gb == 0 and ti == 0) else wadd0
            lwp = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.psum)
            for dc in nl.static_range(dt):
                nisa.nc_matmul(dst=lwp, stationary=qt4[0:PT, dc, ti * PT:(ti + 1) * PT],
                               moving=win_t[0:PT, dc, ti * PT:(ti + 2) * PT],
                               accumulate=(dc > 0))
            lw = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=lw, data=lwp, op0=nl.multiply,
                                      operand0=SM_SCALE, op1=nl.add, operand1=wadd)
            pw = nl.ndarray((PT, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=pw, op=nl.exp, data=lw, bias=0.0, scale=1.0,
                            reduce_op=nl.add, reduce_res=dws[0:PT, ti:ti + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            twp = nl.ndarray((PT, 2, PT), dtype=nl.float32, buffer=nl.psum)
            for wt in nl.static_range(2):
                nisa.nc_matmul(dst=twp[0:PT, wt, 0:PT],
                               stationary=pw[0:PT, wt * PT:(wt + 1) * PT],
                               moving=eye, accumulate=False)
            for wt in nl.static_range(2):
                nisa.tensor_copy(dst=pwT[0:PT, wt, 0:PT], src=twp[0:PT, wt, 0:PT],
                                 engine=nisa.scalar_engine)
            acc_lo = nl.ndarray((PT, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
            acc_hi = nl.ndarray((PT, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
            for wt in nl.static_range(2):
                nisa.nc_matmul(dst=acc_lo[0:PT, 0:HD2 + 1], stationary=pwT[0:PT, wt, 0:PT],
                               moving=win_n[0:PT, ti + wt, 0:HD2 + 1],
                               accumulate=(wt > 0))
                nisa.nc_matmul(dst=acc_hi[0:PT, 0:HD2], stationary=pwT[0:PT, wt, 0:PT],
                               moving=win_n[0:PT, ti + wt, HD2 + 1:HEAD_DIM + 1],
                               accumulate=(wt > 0))
            accs.append((acc_lo, acc_hi))

        for c in nl.static_range(nc):
            pexp = pexps[c % 2]
            pblk = pblks[c % 2]
            lcp = nl.ndarray((PT, QG), dtype=nl.float32, buffer=nl.psum)
            for dc in nl.static_range(dt):
                nisa.nc_matmul(dst=lcp,
                               stationary=ckv_t[0:PT, dc, c * PT:(c + 1) * PT],
                               moving=qt4[0:PT, dc, 0:QG], accumulate=(dc > 0))
            nisa.activation(dst=pexp, op=nl.exp, data=lcp, bias=0.0, scale=SM_SCALE)
            nisa.tensor_tensor(dst=pblk, data1=pexp, data2=maskT[0:PT, c, 0:QG],
                               op=nl.multiply)
            for ti in nl.static_range(QT):
                nisa.nc_matmul(dst=accs[ti][0][0:PT, 0:HD2 + 1],
                               stationary=pblk[0:PT, ti * PT:(ti + 1) * PT],
                               moving=ckv_n[0:PT, c, 0:HD2 + 1], accumulate=True)
                nisa.nc_matmul(dst=accs[ti][1][0:PT, 0:HD2],
                               stationary=pblk[0:PT, ti * PT:(ti + 1) * PT],
                               moving=ckv_n[0:PT, c, HD2 + 1:HEAD_DIM + 1], accumulate=True)
            if inject is not None:
                inject(c)
        for ti in nl.static_range(QT):
            rq = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rq, data=nl.add(nl.add(accs[ti][0][0:PT, HD2:HD2 + 1],
                                                      dws[0:PT, ti:ti + 1]), esink))
            o = nl.ndarray((PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o[0:PT, 0:HD2], data=accs[ti][0][0:PT, 0:HD2],
                               op0=nl.multiply, operand0=rq)
            nisa.tensor_scalar(dst=o[0:PT, HD2:HEAD_DIM], data=accs[ti][1][0:PT, 0:HD2],
                               op0=nl.multiply, operand0=rq)
            nl.store(og3_hbm[qg0 + ti * PT:qg0 + (ti + 1) * PT, h:h + 1, 0:HEAD_DIM],
                     value=o.reshape((PT, 1, HEAD_DIM)))

    NQB = QG // GQ
    QPH = NQB // N_HEADS
    ncof = {gb: _live_pos_blocks((gb + 1) * QT - 1) for gb in gbs}
    nrtof = {gb: _live_rank_tiles((gb + 1) * QT - 1) for gb in gbs}

    key_prep(gbs[0], nrtof[gbs[0]])
    for s0 in nl.static_range(NQB):
        mask_slice(nrtof[gbs[0]], ncof[gbs[0]], s0, maskTs[0])

    for i, gb in enumerate(gbs):
        win_prep(gb)
        nxt = gbs[i + 1] if i + 1 < len(gbs) else None
        if nxt is not None:
            key_prep(nxt, nrtof[nxt])
        ncur = ncof[gb]
        for h in nl.affine_range(N_HEADS):
            if nxt is None:
                head_group(gb, h, ncur, wadds[0], maskTs[i % 2])
                continue

            def inject(c, _nrt=nrtof[nxt], _nc=ncof[nxt], _m=maskTs[(i + 1) % 2],
                       _h=h, _ncur=ncur):
                for k in range(QPH):
                    if (k * _ncur) // QPH == c:
                        mask_slice(_nrt, _nc, _h * QPH + k, _m)

            head_group(gb, h, ncur, wadds[0], maskTs[i % 2], inject=inject)


def _sparse_attention(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs, rid):
    return _sparse_attention_impl(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs, rid)


CORES = 2


def _core_runs(pid, cores=CORES):
    """This physical core's query-group runs, paired front-and-back so the two cores balance.

    The attention is causal, so query group `g` of `QG` queries sweeps `QT * (g + 1)` compressed
    position blocks: 1056 blocks over the 32 groups. Handing each core a contiguous half would be
    272 blocks against 784; pairing group `g` with group `31 - g` puts exactly 528 on each.
    """
    ng = SEQ // QG
    per = ng // (2 * cores)
    a = (pid * per, pid * per + per)
    b = (ng - (pid + 1) * per, ng - pid * per)
    runs = [(a[0], b[1])] if a[1] == b[0] else [a, b]
    return [(g0 * QG, (g1 - g0) * QG) for g0, g1 in runs], \
           [list(range(g0, g1)) for g0, g1 in runs]


@nki.jit
def kernel(hidden_states, state_shared_attn_candidates, state_shared_attn_compress_kv,
           state_shared_attn_index_k, state_shared_attn_topk_idxs,
           layers_21_attn_attn_sink, layers_21_attn_freqs_cis,
           layers_21_attn_kv_norm_weight, layers_21_attn_q_norm_weight,
           layers_21_attn_window_kv_cache, layers_21_attn_wkv_scale,
           layers_21_attn_wkv_weight, layers_21_attn_wo_a_scale, layers_21_attn_wo_a_weight,
           layers_21_attn_wo_b_scale, layers_21_attn_wo_b_weight,
           layers_21_attn_wq_a_scale, layers_21_attn_wq_a_weight,
           layers_21_attn_wq_b_scale, layers_21_attn_wq_b_weight,
           layers_21_attn_norm_weight, replica_group):
    # `candidates`, `index_k` and `window_kv_cache` are the module's recorded arguments that
    # nothing on this layer's path reads: the first two are read only inside `Indexer.forward` and
    # layer 21 has no `Indexer`, and the third is written and never read at `start_pos == 0`. They
    # are carried so the signature matches the module's and ignored exactly as the module's own
    # kernel ignores them.
    x_hbm = hidden_states.reshape((SEQ, DIM))
    ckv_hbm = state_shared_attn_compress_kv.reshape((CACHE_ROWS, HEAD_DIM))
    topk_hbm = state_shared_attn_topk_idxs.reshape((SEQ, INDEX_TOPK))
    # The module's output, and the only IO tensor the collective may not touch.
    out = nl.ndarray((1, SEQ, DIM), dtype=OUT_DTYPE, buffer=nl.shared_hbm)
    out2 = out.reshape((SEQ, DIM))
    # The output seam's two buffers. Named, both nl.shared_hbm, neither an IO tensor, and each
    # exactly the collective's shape. The prefix seam's four are below, with the prefix.
    cc_src = nl.ndarray((SEQ, DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((SEQ, DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_dst")

    pid = nl.program_id(0)
    runs, run_gbs = _core_runs(pid)

    # This rank's index, as the one runtime scalar the staging DMAs below address through. Both
    # physical cores take their own copy, because both dequantize a sharded weight.
    rid = _rank_id()

    QKV_N = Q_LORA + HEAD_DIM
    wqkv = nl.ndarray((QKV_N, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="wqkv")
    wqb = nl.ndarray((N_HEADS * HEAD_DIM, Q_LORA), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="wqb")
    wo_a = nl.ndarray((O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_a")
    wo_b = nl.ndarray((DIM, O_GROUPS * O_LORA), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_b")
    # The dequantizer, split across the two physical cores by total weight bytes. `wq_a` and `wkv`
    # are plain `Linear`s and replicated; `wq_b`, `wo_a` and `wo_b` are the sharded three and each
    # names the dimension the checkpoint shards it on, so `n`/`k` below are this rank's block.
    if pid == 0:
        _dequant_weight(layers_21_attn_wq_a_weight, layers_21_attn_wq_a_scale, wqkv,
                        Q_LORA, DIM)
        _dequant_weight(layers_21_attn_wkv_weight, layers_21_attn_wkv_scale, wqkv,
                        HEAD_DIM, DIM, dst_row=Q_LORA)
        _dequant_weight(layers_21_attn_wo_b_weight, layers_21_attn_wo_b_scale, wo_b,
                        DIM, O_GROUPS * O_LORA, n1=WO_B_SPLIT, shard=1, rid=rid)
    else:
        _dequant_weight(layers_21_attn_wq_b_weight, layers_21_attn_wq_b_scale, wqb,
                        N_HEADS * HEAD_DIM, Q_LORA, shard=0, rid=rid)
        _dequant_weight(layers_21_attn_wo_a_weight, layers_21_attn_wo_a_scale, wo_a,
                        O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM, shard=0, rid=rid)
        _dequant_weight(layers_21_attn_wo_b_weight, layers_21_attn_wo_b_scale, wo_b,
                        DIM, O_GROUPS * O_LORA, n0=WO_B_SPLIT, shard=1, rid=rid)
    for wbuf in (wqkv, wqb):
        nisa.core_barrier(data=wbuf, cores=tuple(range(CORES)))

    # The prefix's own buffers, 2048 rows a rank and addressed by local row.
    xn_q = nl.ndarray((LROWS, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    qakv = nl.ndarray((LROWS, QKV_N), dtype=nl.bfloat16, buffer=nl.private_hbm)
    kvr = nl.ndarray((LROWS, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    # This rank's 2048 RoPE angle rows, staged once so `_rope_inplace` stays rank-oblivious.
    fql = nl.ndarray((LROWS, ROPE_DIM // 2, 2), dtype=nl.float32, buffer=nl.private_hbm)
    # The seam's first pair: this rank's two latents, and all four ranks' of them. Named, both
    # `nl.shared_hbm`, neither an IO tensor, and each exactly the collective's own shape.
    qr_src = nl.ndarray((LROWS, Q_LORA), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="qr_src")
    kv_src = nl.ndarray((LROWS, HEAD_DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="kv_src")
    qr_q = nl.ndarray((SEQ, Q_LORA), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="qr_dst")
    kv_all = nl.ndarray((SEQ, HEAD_DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="kv_dst")
    q = nl.ndarray((N_HEADS * SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    # `nl.shared_hbm`, because the gathered latent is in global token order and both physical
    # cores read all of it: `nl.private_hbm` is not visible to the other physical core of a
    # logical NeuronCore, so a copy split over the two cores into a private buffer leaves each
    # core's half uninitialized.
    kvp = nl.ndarray((SEQ + PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvp")
    og3 = nl.ndarray((SEQ, N_HEADS, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    og = og3.reshape((SEQ, N_HEADS * HEAD_DIM))
    z_q = nl.ndarray((SEQ, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)
    if pid == 0:
        nl.store(kvp[0:PT, 0:HEAD_DIM], value=nl.zeros((PT, HEAD_DIM), dtype=nl.bfloat16))

    # The prefix: `attn_norm`, the fused `wqkv` GEMM, `kv_norm`, the KV RoPE and its activation
    # quantization, and `q_norm`. `head_dim = 512` is one shared KV latent for every head, so
    # there is no head axis to cut here and the head split left all four ranks computing the
    # same thing -- about 19% of a rank's wall, four times over. Cut by *sequence* instead:
    # 2048 rows a rank, 1024 a physical core. Only two reads are rank-addressed, `hidden_states`
    # and `freqs_cis`; everything downstream of them is a compact per-rank buffer indexed by
    # local row, so no other helper learns about the split.
    lr0 = pid * PCROWS
    fql2 = fql.reshape((LROWS, ROPE_DIM))
    for mb in range(PCROWS // PT):
        tcs = nl.ndarray((PT, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tcs, src=_rap(layers_21_attn_freqs_cis, ROPE_DIM, rid,
                                        lr0 + mb * PT))
        nl.store(fql2[lr0 + mb * PT:lr0 + (mb + 1) * PT, 0:ROPE_DIM], value=tcs)

    _norm_quant_rows_r(x_hbm, layers_21_attn_norm_weight, xn_q, PCROWS, DIM, rid, lr0)
    _gemm(xn_q, wqkv, qakv, PCROWS, DIM, QKV_N, x_row=lr0, dst_row=lr0, mg_bytes=128 * 1024)
    _norm_rows(qakv, layers_21_attn_kv_norm_weight, kvr, PCROWS, HEAD_DIM,
               src_col=Q_LORA, row=lr0)
    _rope_inplace(kvr, fql, PCROWS, HEAD_DIM, 1, False, row=lr0)
    _qdq_rows(kvr, kv_src, PCROWS, HEAD_DIM, src_row=lr0, dst_row=lr0)
    _norm_quant_rows(qakv, layers_21_attn_q_norm_weight, qr_src, PCROWS, Q_LORA, row=lr0)

    # One call with both latents: two `all_gather`s back to back whose destinations are both
    # read later leave the first destination corrupt. `collective_dim=0` concatenates rank-major,
    # and a contiguous shard makes that order the module's own token order.
    nisa.core_barrier(data=qr_src, cores=tuple(range(CORES)))
    nisa.core_barrier(data=kv_src, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[qr_src, kv_src], dsts=[qr_q, kv_all],
                   replica_group=replica_group, collective_dim=0)
    # `kvp` is the gathered KV latent with the 128 zero rows in front that the sliding window of
    # the first query tile reads. A collective may not write a view, so this is a copy and not
    # the gather's own destination; split over the two cores, 64 blocks of 128 rows.
    for b in range(SEQ // PT):
        if b % CORES != pid:
            continue
        nisa.dma_copy(dst=kvp[PT + b * PT:PT + (b + 1) * PT, 0:HEAD_DIM],
                      src=kv_all[b * PT:(b + 1) * PT, 0:HEAD_DIM])
    nisa.core_barrier(data=kvp, cores=tuple(range(CORES)))

    # From here the rank is head-parallel: its own 16 of 64 heads, over all 8192 queries, split
    # over the two physical cores by the same runs the attention uses.
    for r0, m in runs:
        _gemm_head_major(qr_q, wqb, q, m, Q_LORA, N_HEADS * HEAD_DIM, HEAD_DIM,
                         row=r0, hstride=SEQ, mg_bytes=128 * 1024)
        _rope_inplace(q, layers_21_attn_freqs_cis, m, HEAD_DIM, 1, False, heads=N_HEADS,
                      row=r0, hstride=SEQ)

    allgbs = sorted([gb for gl in run_gbs for gb in gl], reverse=True)
    _sparse_attention(q, kvp, ckv_hbm, topk_hbm, layers_21_attn_attn_sink, og3, allgbs, rid)

    for wbuf in (wo_a, wo_b):
        nisa.core_barrier(data=wbuf, cores=tuple(range(CORES)))

    for r0, m in runs:
        _rope_og(og3, layers_21_attn_freqs_cis, m=m, row=r0)

        for g in range(O_GROUPS):
            _gemm(og, wo_a, z_q, m, HEADS_PER_GROUP * HEAD_DIM, O_LORA,
                  x_row=r0, x_col=g * HEADS_PER_GROUP * HEAD_DIM, w_row=g * O_LORA,
                  dst_row=r0, dst_col=g * O_LORA, qdq=True)
        # `wo_b` is `RowParallelLinear`: this writes *this rank's partial sum* of the module's
        # output, over the 2048 of 8192 reduction columns this rank holds.
        _gemm(z_q, wo_b, cc_src, m, O_GROUPS * O_LORA, DIM, x_row=r0, dst_row=r0,
              out_dtype=CC_DTYPE, mg_bytes=128 * 1024)

    # The two physical cores of this rank's logical core filled disjoint row runs of `cc_src`.
    # `core_barrier` between those writes into shared HBM and the collective's read of them is
    # what makes the other core's rows visible.
    nisa.core_barrier(data=cc_src, cores=tuple(range(CORES)))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)
    # `wo_b` is constructed with `bias = False`, so there is no post-reduction term: after the
    # reduce every rank holds the whole module's output, rank 0 included. The copy out is split
    # the same way the compute was, so each core moves half of it.
    for r0, m in runs:
        for off, sz in _blocks(m, CC_ROWS):
            nisa.dma_copy(dst=out2[r0 + off:r0 + off + sz, :],
                          src=cc_dst[r0 + off:r0 + off + sz, :])
    nisa.core_barrier(data=out, cores=tuple(range(CORES)))
    return out
