"""`layers.3.attention` on 4 ranks: the optimized single-rank kernel plus the seam that rejoins it.

The body below is `submodule/source.py` at its best accepted commit, unchanged. What is added is
the seam: the rank's `wo_b` partial goes into a named `nl.shared_hbm` scratch buffer,
`nki.collectives.all_reduce` sums the four ranks' partials inside the traced graph, and the sum is
copied into the returned tensor. `FLOORPLAN.md` is binding on the split -- `headx4(allreduce)` --
and `submodule/SUBMODULE.md` is what says a rank's output is a partial sum of `wo_b` and that the
four partials reassemble by addition.

A rank's identity lives entirely in the *values* `inference.py` hands this kernel, never in the
code: one NEFF runs on all four ranks. That is what lets `neuron-explorer capture` replay the same
NEFF on four workers, and it keeps the four ranks' schedules identical.

**The prefix is sequence-parallel by four, and the window KV is distributed across all ranks.**
`attn_norm`, the fused `wqkv` GEMM, `kv_norm`, the KV RoPE, the KV activation quantization and the
`q_norm` of the q latent used to be computed on all 8192 tokens by all four ranks -- the same work
four times, ~19% of a rank's wall. Each rank now computes exactly `SP_SEQ = 8192/4 = 2048` of the
tokens and one coalesced `ncc.all_gather` puts the two latents (`qr`, the 512-wide window-KV
latent) back together, so every rank holds the whole sequence again before the head-parallel part
begins. Two consequences:

* the rank's row range of `hidden_states` and of the RoPE table is *not* a trace-time constant --
  one NEFF runs on all four ranks -- so it comes from `ncc.rank_id()`, a runtime scalar, used as
  the `scalar_offset` of one staging `nisa.dma_copy` per input. Downstream indexing stays static;
* the sliding window's PT overlap rows disappear. They existed because each physical core computed
  only its own row runs of the window-KV latent in *private* HBM; the gathered latent is in shared
  HBM and complete, so a core reads any row it likes.

Attention itself stays head-parallel (16 of 64 heads, all 8192 queries), and the rejoin stays an
`all_reduce` of the `wo_b` partial. Making attention sequence-parallel too would need every rank to
hold all 64 heads' `wq_b`/`wo_a`/`wo_b` -- a second, much larger gather, +0.84 ms -- and, worse, a
*rank-dependent* causal schedule: `_live_pos_blocks` is a trace-time constant and one NEFF serves
all four ranks, so a query-row split has to be sized for the worst rank's reach at every slot.
Measured against the best possible (interleaved) assignment that is **+17.6% on `lcp` + `accs`**,
about +0.85 ms, which is more than the 4x-replicated mask work the split would delete. The full
ledger is in `.autohelix/notes/iter-2.md` section 1.

**None of the three RoPE passes round-trips HBM.** They existed only to rewrite 64 of every 512
columns of a tensor their producer had just written to private HBM: the KV RoPE between `kv_norm`
and `act_quant`, the q RoPE after the `wq_b` GEMM, and the inverse output RoPE before `wo_a`. Each
read a 64-wide window of a 512-wide row -- 128-byte runs at 1024-byte stride -- and wrote it back.
All three are now folded into their producer's drain: `_kv_rope_quant` is one loop that never lets
the tile leave SBUF, `_gemm_head_major`'s `drain()` rotates between the PSUM-to-bfloat16 copy and
the store, and `head_group` rotates all `QT` query tiles at once just before storing them, using a
cos/sin table `rope_prep` stages once per query group because it is head-independent. In all three,
the four products have to be materialized before either half is written back: `a` and `b` are
stride-2 *views* of the tile being rewritten. Worth -0.35 ms, and bit-identical -- the rotation
sees the same bfloat16 values the round trip used to.

**Nothing stages a transposed tile through a temporary any more, and that is the biggest single
win in the file.** Seven sites read an HBM tile transposed into SBUF as
`nisa.tensor_copy(dst=X, src=nl.load_transpose2d(Y), engine=nisa.scalar_engine)` -- both projection
GEMM helpers' `stage_x`/`stage_w`, `head_group`'s `qt4`, the `ckv_t` staging and `win_prep`'s
`win_t`. But `nl.load_transpose2d` *is* `nisa.dma_transpose` into a freshly allocated tile, so the
`tensor_copy` was moving data that had already arrived: about **5300 Activation-engine instructions
a core**, 128 elements each, spread over five regions of the kernel. `nisa.dma_transpose(dst=X,
src=Y)` writes the destination directly. Worth **-6.0%** on its own, bit-identical.

**An `nc_matmul`'s marginal cost is `max(N_moving_free * 0.42 ns, ~25 ns)`,** measured on this exact
NEFF as the median issue period between consecutive MATMULs on one PE: 215 ns at N=512 against
0.42*512 = 215.0, 102 at N=240 against 100.8, 25 at every N below ~60. `LoadStationary` is
**completely** hidden -- the union of MATMUL and LDWEIGHTS busy time exceeds MATMUL's alone by 7 us
in 8.8 ms -- so the 96-98 ns LDWEIGHTS duration, flat in N, never appears in the wall. Two earlier
models were wrong in ways that matter and are recorded so nobody refits them: `max(N, 197)` has the
right shape but a floor 3.3x too high, and `N * 0.42 + 38` has the wrong shape and overcharges the
two matmuls that dominate the kernel by 18%. Consequences, all of them measured rather than argued:
`lcp` and `accs` (K=128, M=128, N=512, 17408 instructions each a rank) are **pure arithmetic with
zero recoverable overhead**; `mask_scatter` (31744 at N<=32) sits on the 25 ns floor; and shrinking
`QT` to 2 so the denominator can ride along in a split PV matmul buys nothing because it trades
8192 *extra* `lcp` matmuls for it -- measured at +3.3%. The softmax denominator used to be 17408
more matmuls at N=1 and is now 640: summing the probability blocks on Vector and reducing along the
partition axis once per head deleted them for -1.23% (see `head_group`'s block loop).

**The engine balance, from the same profile, per physical core.** Whole execution: PE 72.7%,
Vector 49.5%, Activation 43.4%, GpSimd 4.6%, Sync 4.6%, all DMA queues 36.8%. By phase, with
duration as a share of the 12.14 ms wall:

    weight dequant  0.63 ms   5.2%   PE  0.4%   Vec 77.7%   Act 48.4%   DMA 51.7%
    SP prefix       0.66 ms   5.5%   PE 47.9%   Vec 55.8%   Act 29.3%   DMA 47.2%
    all_gather      0.20 ms   1.6%   PE  0.0%   Vec  0.0%   Act  0.0%   DMA 81.8%
    q projection    1.16 ms   9.5%   PE 96.6%   Vec 78.9%   Act 25.1%   DMA 53.0%
    sparse attn     6.47 ms  53.3%   PE 83.2%   Vec 52.2%   Act 66.3%   DMA 17.3%
    wo_a + wo_b     2.14 ms  17.6%   PE 93.8%   Vec 39.7%   Act  8.6%   DMA 51.4%
    all_reduce      0.62 ms   5.1%   PE  0.0%   Vec  0.0%   Act  0.0%   DMA 94.0%
    final copy      0.26 ms   2.2%   PE  0.2%   Vec  0.2%   Act  0.1%   DMA 97.3%

So the two GEMM phases are tensor-bound and done, and **the attention region -- half the wall -- has
no saturated engine at all**: it is bound by the four-stage dependency chain (PE logits, Act `exp`,
Vector mask multiply, PE accumulate), which is why deeper `pexp`/`pblk` buffering was flat and why
the useful moves there have been ones that shorten a stage rather than delete work from it.
`mask_compare`'s broadcast materialization is on the Activation engine for that reason (-1.35%, and
its site says why); GpSimd looks idler still at 4.6% and is ~2.3 ns an element against Act's 0.68
and Vector's 0.47, so the same copy there costs +37%.

**Two thirds of HBM traffic is compiler spill** -- 923.6 MB of 1407.2 MB, 508.3 read plus 415.2
written, against 483.6 MB of real need -- and it is concentrated where `_gemm`'s `xt` tile is
biggest: 138.7 MB of the `wo_a`/`wo_b` phase's 198.1 MB and 49.3 of the q projection's 122.6 move on
`qSyncSpillReload0`. Nothing is bandwidth-bound (MBU 0.16%), so this costs latency only where a
spill sits in a dependency chain. `MGROUP_BYTES` is the knob that sets it and **the intuitive
setting -- as large as the tile can be -- was costing 1.0% of the wall**: the sweep at that constant
is written out at its definition, and 64 KB rather than 192 KB is what ships.

**Instructions cost about as much to issue as a 400-element pass costs to run, and in this kernel
that is the dominant lever left.** Measured on these engines: a bfloat16 `equal` over 1024 elements
a partition is 481 ns, so ~0.42 ns an element plus a fixed **~150-200 ns**. Four separate
experiments this iteration turn on that number, and they do **not** all point the same way:

* deleting an instruction outright wins, and wins more than its issue cost -- the softmax
  denominator's per-block matmul (-1.23%), `mask_scatter`'s PSUM staging copy (-0.16%), and
  `head_group`'s per-head `esink` `exp` plus `key_prep`'s ten-instruction index decode, together
  -1.56% for ~220 instructions a core, which is ~0.8 us apiece rather than 0.175;
* merging instructions wins **off** the dependency chain: the four per-query-tile denominator
  reciprocals became two wide passes at the end of `head_group`, -0.84%;
* merging instructions **loses on** the chain: two compressed blocks an iteration makes the three
  passes in the block loop 1024 elements instead of 512 and halves how many there are -- 0.57 ms a
  core of issue time -- and measures **+1.8%**, because a 1024-element `exp` gates eight `accs`
  matmuls where two 512-element ones each gate four;
* and merging *independent* work still loses if it costs PSUM: batching the four query tiles'
  window-probability drains into one 1024-element copy holds two extra PSUM banks across the whole
  window loop and measures **+1.67%**.
* **shortening** instructions to delete a pass is the worst of all: replacing `mask_compare`'s
  broadcast-then-compare with eight 128-element `tensor_scalar` compares removes 1.39 ms a core of
  Activation and costs **+4.1%**.

Six things about `nki.collectives` on this toolchain, each of which otherwise fails with an
internal compiler error that names something else:

1. the kernel is launched as `kernel[2]`, because this host is LNC=2 and a LNC=1 NEFF leaves the
   collective's buffer unallocated on the logical core's second physical core (`NCC_ILLC059`);
2. `cc_src` / `cc_dst` carry `name=`, or DRAM allocation fails (`NCC_IBIR440`);
3. they are scratch and not the returned tensor, because a collective may not read or write an IO
   tensor (`NCC_INLA001`) -- hence the copy out;
4. both are `nl.shared_hbm`, because buffer kinds may not be mixed;
5. the `ReplicaGroup` is built by `inference.py` and passed in, because the tracer rejects `range`
   in a traced body;
6. the inputs are built on CPU and `.to(device)`'d, because `torch.full(..., device=xla)` emits a
   broadcast HLO that hits `NCC_ISMP902` -- and that error is the broadcast, not the collective.

The reduction is **bfloat16**, which is a measured choice and the one place the numerics differ
from a literal reading of `RowParallelLinear`. `ASSEMBLY.md` records both measurements.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

SEQ = 8192
DIM = 5120
HEAD_DIM = 512
N_HEADS = 16
ROPE_DIM = 64
Q_LORA = 1280
O_LORA = 1024
O_GROUPS = 2
WINDOW = 128
RATIO = 2
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
CHUNK = 4
GATHER_UNROLL = 2

FP8_BLK = 32
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4
QDQ_CHUNK = 1280

# `_mgroup`'s budget for the `xt` staging tile, in bytes *per SBUF partition*: the tile is
# `(PT, mg, ktiles, PT)`, so `mg * ktiles * 128 * 2` bytes a partition against 192 KB available.
# A bigger m-group means fewer m-groups and so fewer reloads of the same weight, so the obvious
# setting is as large as the tile can be -- and that is **wrong**, by 1.0% of the wall. Swept:
#
#     192 KB (wo_b mg=32, wo_a mg=16)   12.1015 ms
#      96 KB (wo_b mg=16, wo_a mg= 8)   11.9769 ms   <- shipped, -1.03%
#      48 KB (wo_b mg= 8, wo_a mg= 4)   12.2642 ms   +1.3%
#
# 128 KB is bit-for-bit the same NEFF as 192 KB (it still admits mg=32 at ktiles=16 and mg=16 at
# ktiles=32) and 64/80 KB are the same as 96 KB for both `wo` GEMMs, so 96 KB is the only distinct
# step between them. The extra weight traffic a smaller m-group costs is real -- 96 KB reads
# `wo_b` twice and `wo_a` four times instead of once and twice, about +55 MB a core -- but at
# 192 KB the `xt` tile alone is 128 KB of a 192 KB partition, which is what forces the 923.6 MB of
# compiler spill the profile found, 138.7 MB of it in this phase. Trading 55 MB of coalesced weight
# reads for that is the win; 48 KB overshoots because it quadruples the weight traffic instead.
MGROUP_BYTES = 64 * 1024
WLOOK = 8
XLOOK = 2

# The four ranks of the projected floorplan, and the dtype they reduce in.
WORLD_SIZE = 4
CC_DTYPE = nl.bfloat16
# Rows per DMA of the post-collective copy. 1024 x 5120 bfloat16 = 10 MiB a descriptor.
CC_ROWS = 1024

# Sequence parallel: the prefix -- attn_norm, the wqkv GEMM, the whole window-KV path and the
# q-latent norm -- is cut by *sequence* across the four ranks, so each rank computes SP_SEQ of
# the 8192 tokens instead of all of them, and one all_gather redistributes the two latents.
SP_SEQ = SEQ // WORLD_SIZE
FREQ_HALF = ROPE_DIM // 2
FREQ_WIDTH = 2 * FREQ_HALF
FREQ_ROWS = 16384


def _blocks(n, step):
    out = []
    off = 0
    while off < n:
        sz = min(step, n - off)
        out.append((off, sz))
        off += sz
    return out


def _live_pos_blocks(t):
    return -(-(PT * (t + 1) // RATIO) // PT)


def _live_rank_tiles(t):
    return min(INDEX_TOPK // PT, -(-(PT * (t + 1) // RATIO) // PT))


def _load_bcast(vec_hbm, n, rows, dt=nl.float32):
    wb = nl.ndarray((rows, n), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wb, src=nl.broadcast_to(
        nl.load(vec_hbm.reshape((1, n)), dtype=dt), (rows, n)))
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


# The e4m3fn decode, built as a bfloat16 bit pattern rather than cast by the hardware. e4m3fn
# reserves exponent field 15 for inf/NaN, so the whole top binade -- bytes with
# (b & 0x7f) >= 0x78, magnitudes 256..448, which these checkpoint weights do occupy -- bitcasts
# back as inf/NaN. The decode this replaces handled that by detecting the binade, decrementing its
# exponent, casting, and doubling to compensate: five full-width passes at one weight per lane.
# Reading the pattern ((b & 0x80) << 8) | ((b & 0x7f) << 4) as bfloat16 instead gives the value
# times 2**-120, with no branch and no special case, normals and subnormals alike -- verified
# exact against the old decode for all 256 byte values -- and the top binade needs nothing,
# because nothing is reading exponent 15 as inf. Two of the three passes run on a uint32 view of
# uint16 pairs, so they carry two weights per lane. The 2**-120 is folded back into the e8m0
# block scale in `_dequant_weight`, where the multiply already has to happen.
BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00      # (b & 0x7f) magnitude mask, doubled for two weights per uint32 lane
E4M3_SGN2 = -2147450880     # 0x80008000, the doubled sign mask, as a signed int32
SC_EXP_ADD = 120            # the 2**120 the block scale carries to cancel the bit-build's 2**-120


def _decode_e4m3(byte_tile, rows, k):
    t = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=byte_tile, op0=nl.multiply, operand0=BYTE_HI,
                       engine=nisa.engine.scalar)
    hi = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi.view(nl.uint32), data=t.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=E4M3_MAG2,
                       op1=nl.right_shift, operand1=4, engine=nisa.engine.vector)
    bits = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=bits.view(nl.uint32), data=t.view(nl.uint32),
                              op0=nl.bitwise_and, operand0=E4M3_SGN2,
                              op1=nl.bitwise_or, operand1=hi.view(nl.uint32))
    return bits.view(nl.bfloat16)


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


def _dequant_weight(w_u8, s_u8, dst_hbm, n, k, dst_row=0):
    kb = k // FP8_BLK
    rows = PT if n % PT == 0 else FP8_BLK
    nsb = rows // FP8_BLK
    for nb in nl.affine_range(n // rows):
        wq = _decode_e4m3(nl.load(w_u8[nb * rows:(nb + 1) * rows, 0:k], dtype=nl.uint8),
                          rows, k)
        sf = nl.ndarray((rows, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in nl.static_range(nsb):
            sb = nl.load(s_u8[nb * nsb + i:nb * nsb + i + 1, 0:kb], dtype=nl.int32)
            # 2**(scale_byte - 127 + 120): the +120 is the bit-build's 2**-120, folded in here.
            # The recorded scale bytes are 114..117, so the biased exponent stays under 254.
            one = nl.copy(nl.left_shift(nl.add(sb, SC_EXP_ADD), 23)).view(nl.float32)
            nisa.tensor_copy(dst=sf[i * FP8_BLK:(i + 1) * FP8_BLK, 0:kb],
                             src=nl.broadcast_to(one, (FP8_BLK, kb)))
        rep = nl.broadcast_to(sf.reshape((rows, kb, 1)), (rows, kb, FP8_BLK))
        prod = nl.ndarray((rows, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=wq, data2=rep.reshape((rows, k)),
                           op=nl.multiply)
        nl.store(dst_hbm[dst_row + nb * rows:dst_row + (nb + 1) * rows, 0:k],
                 value=prod)


def _kv_rope_quant(src_hbm, w_hbm, freqs3, dst_hbm, m, src_col=0, row=0, dst_row=0):
    """`kv_norm` -> the KV RoPE -> `act_quant`, in one pass over the row tiles.

    These were three separate loops, each loading a (PT, HEAD_DIM) tile out of a private-HBM
    `kvr`, rewriting it and storing it back for the next loop to read: a tile written by one
    stage and read by exactly the next one, twice over. What the round trips cost here is not
    their 4 MB a core of DMA but the two extra stage boundaries -- the profile calls this region
    latency-bound rather than engine-bound, and every row tile has to drain through each boundary
    before the next stage starts. Fused, the tile never leaves SBUF and `kvr` disappears.
    """
    wb = _load_bcast(w_hbm, HEAD_DIM, PT, dt=nl.bfloat16)
    half = ROPE_DIM // 2
    lo = HEAD_DIM - ROPE_DIM
    lo2 = lo // 2
    for mb in nl.affine_range(m // PT):
        r0 = row + mb * PT
        t = nl.load(src_hbm[r0:r0 + PT, src_col:src_col + HEAD_DIM], dtype=nl.bfloat16)
        y = _rmsnorm(t, wb, HEAD_DIM, out_dtype=nl.bfloat16)
        cs = nl.load(freqs3[r0:r0 + PT, 0:half, 0:2], dtype=nl.float32)
        cos = nl.copy(cs[0:PT, 0:half, 0])
        sin = nl.copy(cs[0:PT, 0:half, 1])
        # `a` and `b` are stride-2 *views* of `y`'s trailing ROPE_DIM columns, so all four
        # products have to be materialized before either half is written back.
        y2 = y.reshape((PT, HEAD_DIM // 2, 2))
        a = y2[0:PT, lo2:HEAD_DIM // 2, 0]
        b = y2[0:PT, lo2:HEAD_DIM // 2, 1]

        def _mulf(x, z):
            r = nl.ndarray((PT, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=x, data2=z, op=nl.multiply)
            return r

        ac = _mulf(a, cos)
        bs = _mulf(b, sin)
        as_ = _mulf(a, sin)
        bc = _mulf(b, cos)
        o = nl.ndarray((PT, half, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o[0:PT, 0:half, 0], src=nl.subtract(ac, bs))
        nisa.tensor_copy(dst=o[0:PT, 0:half, 1], src=nl.add(as_, bc))
        nisa.tensor_copy(dst=y[0:PT, lo:HEAD_DIM], src=o.reshape((PT, ROPE_DIM)))
        nl.store(dst_hbm[dst_row + mb * PT:dst_row + (mb + 1) * PT, 0:HEAD_DIM],
                 value=_qdq_fp8(y, PT, HEAD_DIM))


def _mgroup(mtiles, ktiles):
    mg = 1
    while mg * 2 <= mtiles and (mg * 2) * ktiles * PT * 2 <= MGROUP_BYTES:
        mg *= 2
    return mg


def _gemm(x_hbm, w_hbm, dst_hbm, m, k, n, out_dtype=nl.bfloat16,
          x_row=0, x_col=0, w_row=0, dst_col=0, dst_row=0, qdq=False):
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
    ntiles = n // nt
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)

        def stage_x(mi):
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=xt[0:PT, mi, kb, 0:PT],
                    src=x_hbm[x_row + (mgb * mg + mi) * PT:x_row + (mgb * mg + mi + 1) * PT,
                              x_col + kb * PT:x_col + (kb + 1) * PT])

        def ntile(nb, jit):
            wt = nl.ndarray((PT, ktiles, nt), dtype=nl.bfloat16, buffer=nl.sbuf)

            def stage_w(kb):
                nisa.dma_transpose(
                    dst=wt[0:PT, kb, 0:nt],
                    src=w_hbm[w_row + nb * nt:w_row + (nb + 1) * nt,
                              kb * PT:(kb + 1) * PT])

            for kb in nl.static_range(min(WLOOK, ktiles)):
                stage_w(kb)
            for mi in nl.static_range(mg):
                if jit and mi + XLOOK < mg:
                    stage_x(mi + XLOOK)
                acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    if mi == 0 and kb + WLOOK < ktiles:
                        stage_w(kb + WLOOK)
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
                     rope_freqs=None):
    """`rope_freqs` folds the q RoPE into the GEMM's drain.

    It used to be a separate in-place RoPE pass: every one of the 16 heads' row tiles was
    written to private HBM as bf16, read straight back, rotated in its trailing ROPE_DIM columns
    and written again -- 16.8 MB a core of round trip on a tile this loop already holds in SBUF,
    plus a whole stage boundary in a 1.43 ms phase. The rotation is a two-multiply-two-add pass
    over 64 of 512 columns, so it costs the drain almost nothing.

    The cos/sin table is staged once per row tile up front (8 KB a partition for 32 tiles) rather
    than per head inside the loop, because the loop is head-outer: re-loading it per head would
    have traded the 16.8 MB saved for 15.8 MB of fresh `freqs_cis` reads.
    """
    ktiles = k // PT
    hst = m if hstride is None else hstride
    mg = _mgroup(m // PT, ktiles)
    ntiles = n // hdim
    half = ROPE_DIM // 2
    lo = hdim - ROPE_DIM
    if rope_freqs is not None:
        mt = m // PT
        cosb = nl.ndarray((PT, mt, half), dtype=nl.float32, buffer=nl.sbuf)
        sinb = nl.ndarray((PT, mt, half), dtype=nl.float32, buffer=nl.sbuf)
        for mi in nl.affine_range(mt):
            cs = nl.load(rope_freqs[row + mi * PT:row + (mi + 1) * PT, 0:half, 0:2],
                         dtype=nl.float32)
            nisa.tensor_copy(dst=cosb[0:PT, mi, 0:half], src=cs[0:PT, 0:half, 0])
            nisa.tensor_copy(dst=sinb[0:PT, mi, 0:half], src=cs[0:PT, 0:half, 1])

    def drain(acc, mii):
        rw = nl.copy(acc, dtype=nl.bfloat16)
        if rope_freqs is None:
            return rw
        r4 = rw.reshape((PT, hdim // 2, 2))
        a = r4[0:PT, lo // 2:hdim // 2, 0]
        b = r4[0:PT, lo // 2:hdim // 2, 1]

        def _mulf(x, z):
            r = nl.ndarray((PT, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=x, data2=z, op=nl.multiply)
            return r

        # All four products first: `a` and `b` are stride-2 views of `rw`, so writing either half
        # back before the last read would clobber the other half's input.
        ac = _mulf(a, cosb[0:PT, mii, 0:half])
        bs = _mulf(b, sinb[0:PT, mii, 0:half])
        as_ = _mulf(a, sinb[0:PT, mii, 0:half])
        bc = _mulf(b, cosb[0:PT, mii, 0:half])
        o = nl.ndarray((PT, half, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o[0:PT, 0:half, 0], src=nl.subtract(ac, bs))
        nisa.tensor_copy(dst=o[0:PT, 0:half, 1], src=nl.add(as_, bc))
        nisa.tensor_copy(dst=rw[0:PT, lo:hdim], src=o.reshape((PT, ROPE_DIM)))
        return rw

    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)

        def stage_x(mi):
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=xt[0:PT, mi, kb, 0:PT],
                    src=x_hbm[row + (mgb * mg + mi) * PT:row + (mgb * mg + mi + 1) * PT,
                              kb * PT:(kb + 1) * PT])

        def ntile(nb, jit):
            wt = nl.ndarray((PT, ktiles, hdim), dtype=nl.bfloat16, buffer=nl.sbuf)

            def stage_w(kb):
                nisa.dma_transpose(
                    dst=wt[0:PT, kb, 0:hdim],
                    src=w_hbm[nb * hdim:(nb + 1) * hdim, kb * PT:(kb + 1) * PT])

            for kb in nl.static_range(min(WLOOK, ktiles)):
                stage_w(kb)
            for mi in nl.static_range(mg):
                if jit and mi + XLOOK < mg:
                    stage_x(mi + XLOOK)
                acc = nl.ndarray((PT, hdim), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    if mi == 0 and kb + WLOOK < ktiles:
                        stage_w(kb + WLOOK)
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, mi, kb, 0:PT],
                                   moving=wt[0:PT, kb, 0:hdim], accumulate=(kb > 0))
                r = nb * hst + row + (mgb * mg + mi) * PT
                nl.store(dst_hbm[r:r + PT, 0:hdim], value=drain(acc, mgb * mg + mi))

        for mi0 in nl.static_range(min(XLOOK, mg)):
            stage_x(mi0)
        ntile(0, mg > XLOOK)
        if ntiles > 1:
            for nb in nl.affine_range(ntiles - 1):
                ntile(nb + 1, False)


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


QT = 4
QG = QT * PT
GQ = 8
# Compressed blocks handled per iteration of the block loop, and it is **1**. CPAIR = 2 halves how
# many `exp`/mask/denominator passes the loop issues -- 3 instructions per 2 blocks instead of 6,
# about 0.57 ms a core of the ~150-200 ns fixed issue cost these engines charge per instruction --
# and measured **+1.8%** (11.942 against 11.731). The region is latency-bound at a finer grain than
# that: a 1024-element `exp` gates all eight of the pair's `accs` matmuls, where two 512-element
# ones each gate four, so the tensor engine loses more to coarser dependency steps than the issue
# saving is worth. Merging instructions pays *off* the chain (see the denominator batch at the end
# of `head_group`, -0.84%) and costs *on* it. CPAIR > 2 does not fit anyway: `lcp`'s PSUM
# destination is a full 2 KB bank a block and the four `accs` accumulators hold four of the eight.
CPAIR = 1


def _sparse_attention_impl(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs,
                           ogf_hbm):
    dt = HEAD_DIM // PT
    nrt = INDEX_TOPK // PT
    ncb = CSEQ // PT
    WB = QT + 1
    ncs = [_live_pos_blocks((gb + 1) * QT - 1) for gb in gbs]
    ncmax = max(ncs)
    nrtmax = max(_live_rank_tiles((gb + 1) * QT - 1) for gb in gbs)

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
    onec = nl.copy(nl.greater_equal(jio[0:PT, 0:1], 0.0), dtype=nl.bfloat16)
    zeroq = nl.copy(nl.less(jio[0:PT, 0:QT], 0.0), dtype=nl.bfloat16)
    # Zeroes wide enough to clear `mask_scatter`'s whole batched PSUM tile in one matmul.
    zblk = nl.zeros((PT, GQ * ncmax), dtype=nl.bfloat16, buffer=nl.sbuf)
    sinkb = _load_bcast(sink_hbm, N_HEADS, PT)
    # exp of the attention sink, for all N_HEADS at once and once for the whole kernel rather
    # than one (PT, 1) `activation` per head_group -- 1 instruction instead of 128 a core, and
    # the value depends on neither the query group nor anything computed inside the loop.
    esinks = nl.ndarray((PT, N_HEADS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=esinks, op=nl.exp, data=sinkb[0:PT, 0:N_HEADS], bias=0.0, scale=1.0)

    pio2_i = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pio2_i, pattern=[[1, PT]], offset=0, channel_multiplier=0)
    pio2 = nl.copy(pio2_i, dtype=nl.bfloat16)
    pio2g = nl.ndarray((PT, GQ, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pio2g, src=pio2.ap(pattern=[[PT, PT], [0, GQ], [1, PT]]))
    cio2_i = nl.ndarray((PT, ncmax), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=cio2_i, pattern=[[1, ncmax]], offset=0, channel_multiplier=0)
    cio2 = nl.copy(cio2_i, dtype=nl.bfloat16)

    ckv_t = nl.ndarray((PT, dt, ncmax * PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    ckv_n = nl.ndarray((PT, ncmax, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gp in nl.affine_range(ncmax):
        nisa.tensor_copy(dst=ckv_n[0:PT, gp, 0:HEAD_DIM],
                         src=nl.load(ckv_hbm[gp * PT:(gp + 1) * PT, 0:HEAD_DIM]))
        for dc in nl.static_range(dt):
            nisa.dma_transpose(dst=ckv_t[0:PT, dc, gp * PT:(gp + 1) * PT],
                               src=ckv_hbm[gp * PT:(gp + 1) * PT, dc * PT:(dc + 1) * PT])

    wadds = []
    for first in range(2):
        live = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=live, data=nonneg, op0=nl.maximum,
                           operand0=float(1 - first))
        wadds.append(nl.multiply(nl.subtract(nl.multiply(live, inwin), 1.0), -NEG_BOUND))

    win_t = nl.ndarray((PT, dt, WB * PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    win_n = nl.ndarray((PT, WB, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    loT = nl.ndarray((PT, nrtmax, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    hiT = nl.ndarray((PT, nrtmax, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    maskTs = [nl.ndarray((PT, ncmax, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
              for _ in range(2)]
    qt4 = nl.ndarray((PT, dt, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Two deep, because the compressed-block loop is a strict chain -- PE logits, Act exp, Vector
    # mask, PE accumulate -- and with one buffer each, block `c`'s mask multiply cannot write
    # `pblk` until all eight of block `c - 1`'s matmuls have read it. That write-after-read is the
    # only thing stopping the two blocks from overlapping; two buffers remove it and leave the
    # ordering to the scheduler.
    PDEPTH = 4
    pexps = [nl.ndarray((PT, CPAIR, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
             for _ in range(PDEPTH)]
    pblks = [nl.ndarray((PT, CPAIR, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
             for _ in range(PDEPTH)]
    # The running sum of `pblk` over the compressed blocks, which is what the softmax denominator
    # needs reduced along the *partition* axis. Two tiles rather than one read-modify-written tile,
    # so no instruction has the same tile as both source and destination.
    pacs = [nl.ndarray((PT, CPAIR, QG), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    pwT = nl.ndarray((PT, 2, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    loTb = [nl.ndarray((PT, GQ, PT), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    palls = [nl.ndarray((PT, nrtmax, GQ, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
             for _ in range(2)]
    calls = [nl.ndarray((PT, nrtmax, GQ, ncmax), dtype=nl.bfloat16, buffer=nl.sbuf)
             for _ in range(2)]

    # The inverse output RoPE, folded into `head_group`'s store (see `og_rope` below). Its cos/sin
    # is per query row and *head-independent*, so it is staged once per query group here and reused
    # by all N_HEADS calls, exactly as `win_prep`'s window tiles are.
    RHALF = ROPE_DIM // 2
    OGLO = HEAD_DIM - ROPE_DIM
    ogcos = nl.ndarray((PT, QT, RHALF), dtype=nl.float32, buffer=nl.sbuf)
    ogsin = nl.ndarray((PT, QT, RHALF), dtype=nl.float32, buffer=nl.sbuf)

    def rope_prep(gb):
        qg0 = gb * QG
        for ti in nl.static_range(QT):
            cs = nl.load(ogf_hbm[qg0 + ti * PT:qg0 + (ti + 1) * PT, 0:RHALF, 0:2],
                         dtype=nl.float32)
            nisa.tensor_copy(dst=ogcos[0:PT, ti, 0:RHALF], src=cs[0:PT, 0:RHALF, 0])
            nisa.tensor_copy(dst=ogsin[0:PT, ti, 0:RHALF], src=cs[0:PT, 0:RHALF, 1])

    def win_prep(gb):
        qg0 = gb * QG
        for wb in nl.static_range(WB):
            # The window of query `q` is positions [q-PT, q), so window block `wb` of query group
            # `gb` is latent rows [qg0 + (wb-1)*PT, qg0 + wb*PT). That -PT used to be a *buffer*
            # offset: the gathered latent was shifted PT rows into a `kvp` whose first PT rows were
            # a zero pre-window, an 8.4 MB copy plus a `core_barrier` per rank. It is a trace-time
            # constant, so it belongs in the index instead, and only `gb == 0`'s first block ever
            # falls off the front.
            r0 = qg0 + (wb - 1) * PT
            if r0 < 0:
                nisa.tensor_copy(dst=win_n[0:PT, wb, 0:HEAD_DIM],
                                 src=nl.zeros((PT, HEAD_DIM), dtype=nl.bfloat16,
                                              buffer=nl.sbuf))
                for dc in nl.static_range(dt):
                    nisa.tensor_copy(dst=win_t[0:PT, dc, wb * PT:(wb + 1) * PT],
                                     src=nl.zeros((PT, PT), dtype=nl.bfloat16,
                                                  buffer=nl.sbuf))
                continue
            nisa.tensor_copy(dst=win_n[0:PT, wb, 0:HEAD_DIM],
                             src=nl.load(kvp_hbm[r0:r0 + PT, 0:HEAD_DIM]))
            for dc in nl.static_range(dt):
                nisa.dma_transpose(
                    dst=win_t[0:PT, dc, wb * PT:(wb + 1) * PT],
                    src=kvp_hbm[r0:r0 + PT, dc * PT:(dc + 1) * PT])

    def key_prep(gb, nrt_t):
        qg0 = gb * QG
        kw = nrt_t * PT
        for ti in nl.static_range(QT):
            q0 = qg0 + ti * PT
            # Seven instructions, not ten. Every `nl.*` above was its own pass, and at
            # kw <= 512 elements roughly half of each pass is the ~150-200 ns these engines
            # charge to issue an instruction at all. `tensor_scalar` carries two ops, so
            # `max(tki - SEQ, 0)` and `(tki < SEQ) * CSEQ` are one instruction each. The two
            # field extractions cannot fold their bf16 cast in -- a bitwise `tensor_scalar`
            # requires an integer `dst` ("'nisa.tensor_scalar_bitvec' op 'dst' must be an
            # integer type") -- so those keep their separate copy.
            tki = nl.load(topk_hbm[q0:q0 + PT, 0:kw], dtype=nl.int32)
            cand = nl.ndarray((PT, kw), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=cand, data=tki, op0=nl.subtract, operand0=SEQ,
                               op1=nl.maximum, operand1=0)
            base = nl.ndarray((PT, kw), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=base, data=tki, op0=nl.less, operand0=SEQ,
                               op1=nl.multiply, operand1=CSEQ)
            rel_i = nl.ndarray((PT, kw), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=rel_i, data1=cand, data2=base, op=nl.add)
            hi_b = nl.copy(nl.right_shift(rel_i, 7), dtype=nl.bfloat16)
            lo_b = nl.copy(nl.bitwise_and(rel_i, PT - 1), dtype=nl.bfloat16)
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

    def mask_compare(nrt_t, nc, qblk, slot):
        qb = qblk * GQ
        for rt in nl.static_range(nrt_t):
            # On the Activation engine, not Vector. This broadcast materialization is 1024 elements
            # a partition per (qblk, rt) -- 2.0 M a core, written once and read once -- and the two
            # `equal` passes that consume it were, with it, 60% of all Vector time in the attention
            # region. Vector was the near-wall there and Act had a third of itself free, so the
            # copy moved: -1.35%, bit-identical. It is now the Activation engine's largest single
            # consumer in the region (2.77 ms of 10.39, 26.7%) and the two `equal` passes are
            # Vector's second and third (3.08 ms of 7.76, 39.7%, behind `pblk`'s mask multiply).
            #
            # Two things this cost to learn. GpSimd looks like the obvious dumping ground at 4.6%
            # busy; the same copy there measured 12.29 -> 16.89 ms, which puts GpSimd at ~2.3 ns an
            # element against Act's 0.68 and Vector's 0.47, so its whole usable budget in this
            # region is ~1 M elements -- a quarter of what `mask_compare` writes. And splitting the
            # copy 50/50 between Act and Vector is *worse* than all of it on Act (12.156 against a
            # 12.100-12.122 same-NEFF band), so Vector was the tighter engine by more than the
            # occupancy numbers suggest.
            #
            # Deleting the copy instead and reading `loT` broadcast in place through a stride-0 AP
            # -- the form `calls` uses three lines down -- is iter-2's measured +1.5%: a 128-wide
            # stride-0 innermost run costs the reading engine more than one contiguous pass plus
            # one contiguous read. `calls` gets away with it because its run is only `nc` <= 32.
            # Deleting this broadcast entirely also does not work, and the reason is a number
            # worth keeping: `loT[p, rt, q]` is one scalar *per partition* for a fixed query,
            # which is exactly what `tensor_scalar`'s tensor `operand0` broadcasts along the free
            # axis for free, out of a register rather than out of SBUF. Replacing the pair below
            # with GQ = 8 `tensor_scalar(op0=nl.equal, operand0=loT[:, rt, qb+g])` compares of 128
            # elements each -- no broadcast tile at all, and `loT` promoted to fp32 because the
            # MLIR verifier requires a float32 tensor `operand0` -- removes ~1.39 ms a core of
            # Activation and measured **+4.1%** (12.318 against 11.830). Eight 128-element Vector
            # passes cost more than one 1024-element pass *plus* the 1024-element Act pass that
            # feeds it, which puts Vector's fixed per-instruction overhead at ~150-200 ns, not the
            # ~50 ns a 481 ns / 1024-element rate implies. **Do not shorten Vector instructions in
            # this region to save a pass.**
            nisa.tensor_copy(
                dst=loTb[rt % 2],
                src=loT.ap(pattern=[[nrtmax * QG, PT], [1, GQ], [0, PT]],
                           offset=rt * QG + qb),
                engine=nisa.scalar_engine)
            nisa.tensor_tensor(
                dst=palls[slot][0:PT, rt, 0:GQ, 0:PT], op=nl.equal,
                data1=pio2g[0:PT, 0:GQ, 0:PT],
                data2=loTb[rt % 2][0:PT, 0:GQ, 0:PT])
            nisa.tensor_tensor(
                dst=calls[slot][0:PT, rt, 0:GQ, 0:nc], op=nl.equal,
                data1=cio2.ap(pattern=[[ncmax, PT], [0, GQ], [1, nc]]),
                data2=hiT.ap(pattern=[[nrtmax * QG, PT], [1, GQ], [0, nc]],
                             offset=rt * QG + qb))

    def mask_scatter(nrt_t, nc, qblk, slot, mdst):
        qb = qblk * GQ
        # One PSUM tile for all GQ queries and one drain, instead of a (PT, nc) tile and a copy per
        # query. The per-query form issued one Scalar copy of nc <= 32 elements per query -- 4096 a
        # core, essentially all per-instruction overhead -- on the *same* engine as the `pexp`
        # activation that sits in the compressed-block loop's dependency chain (PE logits -> Act
        # exp -> Vector mask -> PE accumulate). The profile puts the tensor engine at only 77.8%
        # busy in this region, so what gates the loop is the chain, not PE work, and taking
        # instructions off the chain's Act stage is worth more than taking matmuls off the PE.
        # Measured -3.8%.
        #
        # One `accumulate=False` per query is WRONG once the queries share a PSUM tile: starting a
        # fresh accumulation group resets the whole bank and wipes the queries already written
        # (cosine 0.985, pass_fraction 0.91 -- it fails quietly, it does not error). So the tile is
        # zeroed once up front with the same `nc_matmul(stationary=eye, moving=<zeros>)` idiom
        # `dcol` uses, and every scatter matmul accumulates into it.
        mp = nl.ndarray((PT, GQ * nc), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=mp, stationary=eye, moving=zblk[0:PT, 0:GQ * nc],
                       accumulate=False)
        for g in nl.static_range(GQ):
            for rt in nl.static_range(nrt_t):
                nisa.nc_matmul(dst=mp[0:PT, g * nc:(g + 1) * nc],
                               stationary=palls[slot][0:PT, rt, g, 0:PT],
                               moving=calls[slot][0:PT, rt, g, 0:nc], accumulate=True)
        # One copy, with the (query, block) -> (block, query) permutation `mdst`'s layout wants
        # folded into the PSUM read. This used to be two: a contiguous PSUM drain into SBUF on Act
        # and then the permuting SBUF->SBUF copy on Vector. The staging copy is pure issue
        # overhead -- these engines charge ~150-200 ns an instruction however few elements it
        # touches, and this one is GQ * nc <= 256 -- and `mask_scatter` runs 512 times a core.
        nisa.tensor_copy(dst=mdst[0:PT, 0:nc, qb:qb + GQ],
                         src=mp.ap(pattern=[[GQ * nc, PT], [1, nc], [nc, GQ]]),
                         engine=nisa.engine.vector)

    def mask_slice(nrt_t, nc, qblk, mdst):
        mask_compare(nrt_t, nc, qblk, 0)
        mask_scatter(nrt_t, nc, qblk, 0, mdst)

    def head_group(gb, h, nc, wadd0, maskT, inject=None):
        qg0 = gb * QG
        for ti in nl.static_range(QT):
            q0 = qg0 + ti * PT
            for dc in nl.static_range(dt):
                nisa.dma_transpose(
                    dst=qt4[0:PT, dc, ti * PT:(ti + 1) * PT],
                    src=q_hbm[h * SEQ + q0:h * SEQ + q0 + PT, dc * PT:(dc + 1) * PT])
        dws = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.sbuf)
        accs = []
        dacc = []
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
            # One 256-element drain, not two 128-element ones: `twp` and `pwT` are both
            # (PT, 2, PT) and contiguous, and a short instruction on these engines is nearly all
            # fixed overhead (~150-200 ns -- see `mask_compare`).
            nisa.tensor_copy(dst=pwT, src=twp, engine=nisa.scalar_engine)
            acc = nl.ndarray((PT, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
            for wt in nl.static_range(2):
                nisa.nc_matmul(dst=acc, stationary=pwT[0:PT, wt, 0:PT],
                               moving=win_n[0:PT, ti + wt, 0:HEAD_DIM],
                               accumulate=(wt > 0))
            accs.append(acc)

        # CPAIR compressed blocks an iteration, not one. The three engine passes in this loop --
        # `exp` off PSUM on Activation, the mask multiply on Vector, the running denominator sum on
        # Vector -- are the loop's whole non-matmul cost, and at QG = 512 elements each they are
        # about 45% fixed issue overhead (~150-200 ns an instruction on these engines against
        # 512 * 0.42 ns of data). Pairing the blocks makes each pass 1024 elements and halves how
        # many there are, for the same bytes touched: three instructions per two blocks instead of
        # six. `nc` is `2 * (gb + 1)` for every query group, so it is always even and there is no
        # odd tail. The matmul count does not change -- `ckv_t`'s stationary and `ckv_n`'s moving
        # are still per block -- only the PSUM destination widens to two banks.
        for cb in nl.static_range(nc // CPAIR):
            c0 = cb * CPAIR
            sl = cb % PDEPTH
            pexp = pexps[sl]
            pblk = pblks[sl]
            lcp = nl.ndarray((PT, CPAIR, QG), dtype=nl.float32, buffer=nl.psum)
            for cj in nl.static_range(CPAIR):
                for dc in nl.static_range(dt):
                    nisa.nc_matmul(dst=lcp[0:PT, cj, 0:QG],
                                   stationary=ckv_t[0:PT, dc, (c0 + cj) * PT:(c0 + cj + 1) * PT],
                                   moving=qt4[0:PT, dc, 0:QG], accumulate=(dc > 0))
            # An additive mask instead -- fold `SM_SCALE` and a 0 / NEG_BOUND mask into one Vector
            # `scalar_tensor_tensor` off PSUM, then a pure SBUF `exp` on Activation, the chain the
            # window path uses -- is **exactly bit-identical** (the mask is exactly 0 where the
            # position is selected and exp(NEG_BOUND) is exactly 0 where it is not) and measured
            # **+0.4%**: 12.163 against a 12.100-12.122 same-NEFF band. So the 1.48 ms COPY the
            # profile attributes to this line is not removed by moving the fp32 PSUM read to
            # Vector, and this region's Act:Vector balance is already right. Recorded because the
            # additive form is a prerequisite for anything that wants the mask pre-exp.
            nisa.activation(dst=pexp, op=nl.exp, data=lcp, bias=0.0, scale=SM_SCALE)
            nisa.tensor_tensor(dst=pblk, data1=pexp,
                               data2=maskT[0:PT, c0:c0 + CPAIR, 0:QG], op=nl.multiply)
            # The softmax denominator used to be a fifth `nc_matmul` right here, `pblk` against a
            # ones column, once per query tile per block: 8704 matmuls a core at N=1, every one of
            # them on the 25 ns issue floor, 0.22 ms of the tensor engine in the one region where
            # PE is the resource everything else stalls behind (83.2% busy, and its own wait time
            # over 46513 matmuls is 78 us). Summing over blocks first on the Vector engine and
            # reducing along the partition axis *once* at the end of the head instead costs one
            # 512-element Vector pass a block -- 0.24 us against a 2.97 us block period, on an
            # engine that is 52.2% busy here -- and leaves 5 matmuls a head in place of 4*nc + 1.
            if cb == 0:
                nisa.tensor_copy(dst=pacs[1], src=pblk, engine=nisa.engine.vector)
            else:
                nisa.tensor_tensor(dst=pacs[(cb + 1) % 2], data1=pacs[cb % 2], data2=pblk,
                                   op=nl.add, engine=nisa.engine.vector)
            for cj in nl.static_range(CPAIR):
                for ti in nl.static_range(QT):
                    nisa.nc_matmul(dst=accs[ti],
                                   stationary=pblk[0:PT, cj, ti * PT:(ti + 1) * PT],
                                   moving=ckv_n[0:PT, c0 + cj, 0:HEAD_DIM], accumulate=True)
                if inject is not None:
                    inject(c0 + cj)
        # `accumulate=False` on one column of `dcol` would reset the whole PSUM bank and wipe the
        # columns already written, so the tile is zeroed once by the `eye`/`zeroq` matmul and all
        # four reductions accumulate -- the same rule `mask_scatter`'s batched tile lives under.
        dcol = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=dcol[0:PT, 0:QT], stationary=eye, moving=zeroq,
                       accumulate=False)
        pfin = pacs[(nc // CPAIR) % 2]
        for cj in nl.static_range(CPAIR):
            for ti in nl.static_range(QT):
                nisa.nc_matmul(dst=dcol[0:PT, ti:ti + 1],
                               stationary=pfin[0:PT, cj, ti * PT:(ti + 1) * PT],
                               moving=onec, accumulate=True)
        # All QT query tiles of this head are normalized into one SBUF tile, the inverse RoPE is
        # applied to their trailing ROPE_DIM columns in a single batch of passes, and then they are
        # stored. That deletes the separate output-RoPE pass entirely: it used to read all of `og3`
        # back out of private HBM, rotate 64 of every 512 columns and write it again -- 16.8 MB
        # a core of round trip on tiles this loop just produced, plus a stage boundary. Batching
        # the QT tiles matters: per tile the rotated slice is 32 elements a partition, which is
        # all per-instruction overhead; over all four it is 128.
        o4 = nl.ndarray((PT, QT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        # All QT denominators in two instructions rather than twelve. This used to be, per query
        # tile, `nl.add` + `nl.add` + `reciprocal` on one element a partition -- three
        # instructions whose entire cost is the ~150-200 ns an instruction costs to issue on these
        # engines however few elements it touches, 12 a head and 1536 a core. `dcol` and `dws` are
        # both (PT, QT) and `esink` is the per-partition scalar `scalar_tensor_tensor`'s
        # `operand0` broadcasts for free, so the whole sum is one instruction and the reciprocal
        # is one more, both QT wide.
        dsum = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=dsum, data=dcol[0:PT, 0:QT], op0=nl.add,
                                  operand0=esinks[0:PT, h:h + 1],
                                  op1=nl.add, operand1=dws[0:PT, 0:QT])
        rq = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rq, data=dsum)
        for ti in nl.static_range(QT):
            nisa.tensor_scalar(dst=o4[0:PT, ti, 0:HEAD_DIM], data=accs[ti],
                               op0=nl.multiply, operand0=rq[0:PT, ti:ti + 1])
        o5 = o4.reshape((PT, QT, HEAD_DIM // 2, 2))
        oa = o5[0:PT, 0:QT, OGLO // 2:HEAD_DIM // 2, 0]
        ob = o5[0:PT, 0:QT, OGLO // 2:HEAD_DIM // 2, 1]

        def _ogmul(x, z):
            r = nl.ndarray((PT, QT, RHALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=x, data2=z, op=nl.multiply)
            return r

        # Four products before either write-back: `oa`/`ob` are stride-2 views of `o4`.
        oac = _ogmul(oa, ogcos[0:PT, 0:QT, 0:RHALF])
        obs = _ogmul(ob, ogsin[0:PT, 0:QT, 0:RHALF])
        oas = _ogmul(oa, ogsin[0:PT, 0:QT, 0:RHALF])
        obc = _ogmul(ob, ogcos[0:PT, 0:QT, 0:RHALF])
        orot = nl.ndarray((PT, QT, RHALF, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=orot[0:PT, 0:QT, 0:RHALF, 0], data1=oac, data2=obs,
                           op=nl.add)
        nisa.tensor_tensor(dst=orot[0:PT, 0:QT, 0:RHALF, 1], data1=obc, data2=oas,
                           op=nl.subtract)
        nisa.tensor_copy(dst=o4[0:PT, 0:QT, OGLO:HEAD_DIM],
                         src=orot.reshape((PT, QT, ROPE_DIM)))
        for ti in nl.static_range(QT):
            nl.store(og3_hbm[qg0 + ti * PT:qg0 + (ti + 1) * PT, h:h + 1, 0:HEAD_DIM],
                     value=o4[0:PT, ti, 0:HEAD_DIM])

    NQB = QG // GQ
    QPH = NQB // N_HEADS
    ncof = {gb: _live_pos_blocks((gb + 1) * QT - 1) for gb in gbs}
    nrtof = {gb: _live_rank_tiles((gb + 1) * QT - 1) for gb in gbs}

    key_prep(gbs[0], nrtof[gbs[0]])
    for s0 in nl.static_range(NQB):
        mask_slice(nrtof[gbs[0]], ncof[gbs[0]], s0, maskTs[0])

    for i, gb in enumerate(gbs):
        win_prep(gb)
        rope_prep(gb)
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
                for k in range(QPH + 1):
                    if (k * _ncur) // (QPH + 1) != c:
                        continue
                    if k < QPH:
                        mask_compare(_nrt, _nc, _h * QPH + k, k % 2)
                    if k >= 1:
                        mask_scatter(_nrt, _nc, _h * QPH + k - 1, (k - 1) % 2, _m)

            head_group(gb, h, ncur, wadds[0], maskTs[i % 2], inject=inject)


def _sparse_attention(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs, ogf_hbm):
    return _sparse_attention_impl(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs,
                                  ogf_hbm)


CORES = 2


def _core_runs(pid, cores=CORES):
    ng = SEQ // QG
    per = ng // (2 * cores)
    a = (pid * per, pid * per + per)
    b = (ng - (pid + 1) * per, ng - pid * per)
    runs = [(a[0], b[1])] if a[1] == b[0] else [a, b]
    return [(g0 * QG, (g1 - g0) * QG) for g0, g1 in runs], \
           [list(range(g0, g1)) for g0, g1 in runs]


@nki.jit
def kernel(hidden_states, state_shared_attn_candidates, state_shared_attn_compress_kv, state_shared_attn_index_k, state_shared_attn_topk_idxs, layers_3_attn_attn_sink, layers_3_attn_freqs_cis, layers_3_attn_kv_norm_weight, layers_3_attn_q_norm_weight, layers_3_attn_window_kv_cache, layers_3_attn_wkv_scale, layers_3_attn_wkv_weight, layers_3_attn_wo_a_scale, layers_3_attn_wo_a_weight, layers_3_attn_wo_b_scale, layers_3_attn_wo_b_weight, layers_3_attn_wq_a_scale, layers_3_attn_wq_a_weight, layers_3_attn_wq_b_scale, layers_3_attn_wq_b_weight, layers_3_attn_norm_weight, replica_group):
    # `hidden_states` and `freqs_cis` are replicated inputs, so a rank's own quarter of the
    # sequence is a *dynamic* row range of them: one NEFF runs on all four ranks, so the offset
    # cannot be a trace-time constant. `ncc.rank_id()` is the runtime scalar that makes it one.
    xsp_hbm = hidden_states.reshape((WORLD_SIZE, SP_SEQ, DIM))
    fsp_hbm = layers_3_attn_freqs_cis.reshape(
        (FREQ_ROWS // SP_SEQ, SP_SEQ, FREQ_WIDTH))
    ckv_hbm = state_shared_attn_compress_kv.reshape((CACHE_ROWS, HEAD_DIM))
    topk_hbm = state_shared_attn_topk_idxs.reshape((SEQ, INDEX_TOPK))
    # The module's output, and the only IO tensor the collective may not touch.
    out = nl.ndarray((1, SEQ, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out2 = out.reshape((SEQ, DIM))
    # The seam's two buffers. Named, both nl.shared_hbm, neither an IO tensor.
    cc_src = nl.ndarray((SEQ, DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((SEQ, DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_dst")

    pid = nl.program_id(0)
    runs, run_gbs = _core_runs(pid)

    QKV_N = Q_LORA + HEAD_DIM
    wqkv = nl.ndarray((QKV_N, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="wqkv")
    wqb = nl.ndarray((N_HEADS * HEAD_DIM, Q_LORA), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="wqb")
    wo_a = nl.ndarray((O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_a")
    wo_b = nl.ndarray((DIM, O_GROUPS * O_LORA), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_b")
    if pid == 0:
        _dequant_weight(layers_3_attn_wq_a_weight, layers_3_attn_wq_a_scale, wqkv,
                        Q_LORA, DIM)
        _dequant_weight(layers_3_attn_wkv_weight, layers_3_attn_wkv_scale, wqkv,
                        HEAD_DIM, DIM, dst_row=Q_LORA)
        _dequant_weight(layers_3_attn_wo_b_weight, layers_3_attn_wo_b_scale, wo_b,
                        DIM, O_GROUPS * O_LORA)
    else:
        _dequant_weight(layers_3_attn_wq_b_weight, layers_3_attn_wq_b_scale, wqb,
                        N_HEADS * HEAD_DIM, Q_LORA)
        _dequant_weight(layers_3_attn_wo_a_weight, layers_3_attn_wo_a_scale, wo_a,
                        O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM)
    for wbuf in (wqkv, wqb, wo_a, wo_b):
        nisa.core_barrier(data=wbuf, cores=tuple(range(CORES)))

    # Sequence-parallel prefix buffers: SP_SEQ rows, not SEQ. Each of the rank's two physical
    # cores owns half of the rank's quarter.
    SP_PER_CORE = SP_SEQ // CORES
    sp0 = pid * SP_PER_CORE
    xloc = nl.ndarray((SP_SEQ, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    floc = nl.ndarray((SP_SEQ, FREQ_WIDTH), dtype=nl.float32, buffer=nl.private_hbm)
    xn_q = nl.ndarray((SP_SEQ, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    qakv = nl.ndarray((SP_SEQ, QKV_N), dtype=nl.bfloat16, buffer=nl.private_hbm)
    # The two latents this rank contributes to the gather, and the gathered wholes. All four are
    # shared_hbm: the gather's result has to be visible to both physical cores, which private_hbm
    # is not, and a collective's operands may not mix buffer kinds.
    qr_loc = nl.ndarray((SP_SEQ, Q_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="qr_loc")
    kv_loc = nl.ndarray((SP_SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="kv_loc")
    qr_q = nl.ndarray((SEQ, Q_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qr_all")
    # A collective's operand access pattern must start at offset zero (`Coalesced ccop: AP
    # offsets must be zero`), so the window-KV gather lands at row 0 of its own buffer and
    # `win_prep` applies the window's -PT offset as an index rather than a copy.
    kvg = nl.ndarray((SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kv_all")
    q = nl.ndarray((N_HEADS * SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    og3 = nl.ndarray((SEQ, N_HEADS, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    og = og3.reshape((SEQ, N_HEADS * HEAD_DIM))
    z_q = nl.ndarray((SEQ, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)

    # This rank's quarter of `hidden_states` and of the RoPE table, staged once through a
    # scalar-indirect DMA whose offset is the runtime rank id. Everything downstream indexes
    # these with trace-time constants, so no other instruction has to become dynamic.
    rid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rid, ncc.rank_id())
    for off, sz in _blocks(SP_PER_CORE, CC_ROWS):
        nisa.dma_copy(dst=xloc[sp0 + off:sp0 + off + sz, 0:DIM],
                      src=xsp_hbm.ap(pattern=[[DIM, sz], [1, DIM]],
                                     offset=(sp0 + off) * DIM,
                                     scalar_offset=rid, indirect_dim=0))
    nisa.dma_copy(dst=floc[sp0:sp0 + SP_PER_CORE, 0:FREQ_WIDTH],
                  src=fsp_hbm.ap(pattern=[[FREQ_WIDTH, SP_PER_CORE], [1, FREQ_WIDTH]],
                                 offset=sp0 * FREQ_WIDTH,
                                 scalar_offset=rid, indirect_dim=0))
    floc3 = floc.reshape((SP_SEQ, FREQ_HALF, 2))

    # The prefix, over this rank's quarter only. No sliding-window overlap rows are needed any
    # more: the gather below hands every core the whole window-KV latent.
    _norm_quant_rows(xloc, layers_3_attn_norm_weight, xn_q, SP_PER_CORE, DIM, row=sp0)
    _gemm(xn_q, wqkv, qakv, SP_PER_CORE, DIM, QKV_N, x_row=sp0, dst_row=sp0)
    _kv_rope_quant(qakv, layers_3_attn_kv_norm_weight, floc3, kv_loc, SP_PER_CORE,
                   src_col=Q_LORA, row=sp0, dst_row=sp0)
    _norm_quant_rows(qakv, layers_3_attn_q_norm_weight, qr_loc, SP_PER_CORE, Q_LORA, row=sp0)

    # Both cores' halves of this rank's quarter have to be in shared HBM before the gather reads
    # them, and the gathered wholes have to be complete before either core reads a row it did not
    # write. The gather is coalesced: one collective for both latents.
    nisa.core_barrier(data=qr_loc, cores=tuple(range(CORES)))
    nisa.core_barrier(data=kv_loc, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[qr_loc, kv_loc], dsts=[qr_q, kvg],
                   replica_group=replica_group, collective_dim=0)
    nisa.core_barrier(data=qr_q, cores=tuple(range(CORES)))
    nisa.core_barrier(data=kvg, cores=tuple(range(CORES)))

    # From here the rank is head-parallel again: 16 of 64 heads over all 8192 queries.
    for r0, m in runs:
        _gemm_head_major(qr_q, wqb, q, m, Q_LORA, N_HEADS * HEAD_DIM, HEAD_DIM,
                         row=r0, hstride=SEQ, rope_freqs=layers_3_attn_freqs_cis)

    allgbs = sorted([gb for gl in run_gbs for gb in gl], reverse=True)
    _sparse_attention(q, kvg, ckv_hbm, topk_hbm, layers_3_attn_attn_sink, og3, allgbs,
                      layers_3_attn_freqs_cis)

    for r0, m in runs:
        for g in range(O_GROUPS):
            _gemm(og, wo_a, z_q, m, HEADS_PER_GROUP * HEAD_DIM, O_LORA,
                  x_row=r0, x_col=g * HEADS_PER_GROUP * HEAD_DIM, w_row=g * O_LORA,
                  dst_row=r0, dst_col=g * O_LORA, qdq=True)
        # `wo_b` is RowParallelLinear: this writes *this rank's partial sum* of the module's
        # output, over the 2048 of 8192 reduction columns this rank holds.
        _gemm(z_q, wo_b, cc_src, m, O_GROUPS * O_LORA, DIM, x_row=r0, dst_row=r0,
              out_dtype=CC_DTYPE)

    # The two physical cores of this rank's logical core filled disjoint row runs of `cc_src`.
    # `core_barrier` between those writes into shared HBM and the collective's read of them is
    # what makes the other core's rows visible.
    nisa.core_barrier(data=cc_src, cores=tuple(range(CORES)))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)
    # After the reduce every rank holds the whole module's output, rank 0 included. The copy is
    # split the same way the compute was, so each core moves half of it.
    for r0, m in runs:
        for off, sz in _blocks(m, CC_ROWS):
            nisa.dma_copy(dst=out2[r0 + off:r0 + off + sz, :],
                          src=cc_dst[r0 + off:r0 + off + sz, :])
    nisa.core_barrier(data=out, cores=tuple(range(CORES)))
    return out
