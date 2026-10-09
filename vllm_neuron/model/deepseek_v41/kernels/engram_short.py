"""All of `layers.1.engram` on 4 ranks, rejoined by one `nki.collectives.all_reduce`.

    embed(this core's 12 hash cols) -> act_quant -> half of a_t, per physical core
    ---- dma_copy(a_sh <- own half); core_barrier(a_sh); dma_copy(other half) ----
    a_t                          all 24 hash cols, 6144 contraction cols
    a_tv = a_t[12*rank ...]      this rank's `ngram` quarter, a runtime slice of a_t
    key_chunk = wkv[0:20480][chunk 2*rank + pid] @ a_t      ONE of the 8 key chunks
      -> rsq, rdot, rh           this chunk's share of the gate's three reductions
    value_r = wkv[20480:] @ a_tv this rank's 1536 contraction cols -- a PARTIAL
    ---- dma_copy(cc_src <- value_r), per physical core, disjoint halves ----
    ---- dma_copy(cc_src[5120 + 12*pid ...] <- the 3 gate partials, in rank's slot) ----
    ---- core_barrier(cc_src, cores=(0, 1)) --------------------------------
    ncc.all_reduce(cc_src -> cc_dst, op=add)   the four ranks' partials, float32
    ---- dma_copy(value <- cc_dst[:5120]); dma_copy(gpad <- cc_dst[5120:]) ----
    gate[c] = sigmoid(...)       from slot(c, core 0) + slot(c, core 1), on every rank
    out[:, c, :] = bfloat16(h[:, c, :] + gate[c] * value)

`ngram` is the contraction axis of `wkv`, so the cut rejoins by summation. `gate` is a
nonlinear function of the *summed* key, so no rank can form it from a quarter of the
n-grams: the gate's **contraction** is indivisible. Its **output rows** are not. The
three things the gate needs are sums over the key's 20480 rows, and a sum's output rows
split for free -- so the key's 8 chunks of 2560 map one-to-one onto this device's 8
physical cores (4 ranks x 2 cores), chunk `2 * rank + pid`, and the 24 scalars that
result ride the same `all_reduce` as 24 extra float32 columns rather than a second
collective. Each core writes all 12 columns of its own half of that pad, its rank's three
values in slot `3 * rank` and an exact 0.0 in the other nine, so three of the four ranks
add exact zeros to any one slot. hc copy `c` is chunks `2c` and `2c + 1`, which are
rank `c`'s core 0 and core 1, and the tail adds them in that order -- the order a single
core's loop over `ci = 2c, 2c + 1` used -- so the gate is bit-identical to forming the
whole key half on one core.

The value half enters the output linearly, so
`sum_r (gate * value_r) == gate * sum_r value_r` and it is the 2.6 MB `value` that
crosses the wire, not the 10.5 MB product. The residual `h` is added once, after the
ranks rejoin -- folding it into the partials would count it four times.

This host runs LNC=2, so each rank's logical NeuronCore is two physical cores and the
kernel is launched as `kernel[2]`. Core `p` takes key chunk `2 * rank + p`, hash columns
`12p .. 12p + 11` of the prefix, half the value projection's output columns, and hc
copies `2p`, `2p + 1` of the output. Every split is a split of a *sum's output rows* or
of independent 32-wide quantization blocks, never of a sum, so the arithmetic is
bit-identical to doing all of it on one core.

A rank's identity reaches this code two ways. Its *values* -- its quarter of
`wkv.value_weight` / `wkv.value_scale` -- come from `inference.py`. Its *index* comes from
`ncc.rank_id()`, which is a runtime `VirtualRegister`: it cannot size a tile or bound a
loop, so every loop bound and tile width here is still a trace-time constant identical on
all four ranks, and it is used only to move a column offset, through
`.ap(..., scalar_offset=..., indirect_dim=...)`. That offset is int32 for the DMA path and
**uint32** for `nisa.tensor_copy`'s dynamic src, which otherwise fails BIR verification
with `Expect argument datatype to be of type uint32`. One NEFF still runs on all four
ranks, which is what lets `neuron-explorer capture` replay it on four workers and keeps
the four ranks' schedules identical.

Six things about `nki.collectives` on this toolchain, each of which otherwise fails with
an internal compiler error that names something else:

1. the kernel is launched as `kernel[2]`, because this host is LNC=2 and a LNC=1 NEFF
   leaves the collective's buffer unallocated on the logical core's second physical core
   (`NCC_ILLC059`);
2. `cc_src` / `cc_dst` carry `name=`, or DRAM allocation fails (`NCC_IBIR440`);
3. they are scratch, because a collective may not read or write an IO tensor
   (`NCC_INLA001`) -- hence the `dma_copy` in and the `dma_copy` out;
4. both are `nl.shared_hbm`, because a collective's operands may not mix buffer kinds;
5. the `ReplicaGroup` is built by `inference.py` and passed in, because the tracer
   rejects `range` in a traced body;
6. the inputs are built on CPU and `.to(device)`'d, because `torch.full(..., device=xla)`
   emits a broadcast HLO that hits `NCC_ISMP902` -- and that error names the broadcast,
   not the collective.

The arithmetic between `_pow2` and the end of the value loop is the optimized single-rank
kernel at its best accepted commit (`7e5462f`), bit for bit, with two instruction choices
changed under it: `_pow2` builds its power of two by writing a float32's exponent field
instead of walking the index's six bits, and `_act_quant`'s halve-before-the-fp8-cast and
double-after-it have moved off the 128x3072 path into the 128x96 scales they multiply.
Both are exact rather than close -- see `_pow2`'s own docstring. What the assembly changed:
`hsq` is accumulated in the key loop where `h` is already in SBUF rather than reloaded in
the tail; the loop bounds are split across the two physical cores; the key half is cut to
one chunk per core with the gate's partials in the collective; the prefix is cut in half
per core and traded through `a_sh`; and `a_tv` is a runtime slice of `a_t` rather than a
second embed + act_quant + transpose, which is exact because act_quant's blocks are 32
contiguous features and 256 is a multiple of 32, so no block straddles a hash column.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

TOK = 128
DIM = 5120
HC = 4
NHC = 24
HD = 256
KDIM = NHC * HD
NKEY = DIM * HC
NVAL = DIM

# The floorplan's width and dimension: ngram x 4. `inference.py` checks this against its
# own WORLD_SIZE and builds the ReplicaGroup from it.
WORLD_SIZE = 4
NHC_SHARD = NHC // WORLD_SIZE
KSL = KDIM // WORLD_SIZE

# The two physical cores of one logical NeuronCore at LNC=2. `inference.py` checks this
# against the width it launches the kernel at.
N_CORES = 2
HC_PER_CORE = HC // N_CORES

# The gate was the one term `ngram` does not divide: every rank formed all 20480 key rows,
# 85% of a rank's weight bytes, four times over. It *is* divisible -- by its own output
# axis. The key's 8 chunks of 2560 map one-to-one onto the device's 8 physical cores
# (4 ranks x 2 cores), chunk `2 * rank + pid`, so one core forms one chunk and the three
# scalars that chunk contributes to the gate ride the existing all_reduce as 24 extra
# columns: slot `GSLOT * (WORLD_SIZE * pid + rank)`. Three of the four ranks contribute an
# exact 0.0 to any one slot, and the two chunks of a copy are summed core 0 then core 1 --
# the same order the single-core loop used -- so the gate is bit-identical.
# The replicated prefix divides too: by hash column, between the two physical cores.
NHC_CORE = NHC // N_CORES
KDIM_CORE = NHC_CORE * HD

GSLOT = 3
GPAD_CORE = GSLOT * WORLD_SIZE
GPAD = GPAD_CORE * N_CORES
CCW = NVAL + GPAD

EPS = 1e-20
CLAMP = 1e-6
DIM_RSQRT = 5120 ** -0.5

ABLK = 32
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
LOG2E = 1.4426950408889634

PAR = 128
NMAX = 512
NW = 2560
NSL = NW // NMAX
NWB = NW // ABLK
RBLK = PAR // ABLK

NKT = KDIM // PAR
NKT_CORE = NKT // N_CORES
NVT = KSL // PAR
NKC = NKEY // NW
NKC_CORE = NKC // N_CORES

EXP_BIAS = 32
FP32_EXP_BIAS = 127.0
FP32_EXP_LSB = 8388608.0

BF16_EXP_LSB = 128
E8M0_SHIFT = 128


def _pow2(e_int, m, n):
    """2 ** clamp(e_int, -EXP_BIAS, 63 - EXP_BIAS), in two instructions.

    The bit-walk this replaces built the same power out of a 6-bit index and six
    conditional multiplies -- 13 ops on a (128, 96) tile, where the cost is per
    instruction and not per element. A float32's exponent field *is* the answer:
    `(e + 127) * 2**23`, written into a uint32 and read back as a float32, is exactly
    `2**e`. Both steps are exact -- `(e + 127) * 2**23` is a small integer times a power
    of two, so float32 holds it with no rounding, and 255 * 2**23 is inside uint32 --
    and the clamp is the same clamp the index form applied, so every value this returns
    is the value the bit-walk returned.
    """
    idx = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=idx, data=e_int,
                       op0=nl.maximum, operand0=float(-EXP_BIAS),
                       op1=nl.minimum, operand1=float(63 - EXP_BIAS))
    res = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res.view(nl.uint32), data=idx,
                       op0=nl.add, operand0=FP32_EXP_BIAS,
                       op1=nl.multiply, operand1=FP32_EXP_LSB)
    return res


def _scratch(m, n):
    out = []
    for _ in range(2):
        out.append(nl.ndarray((m, n), dtype=nl.uint8, buffer=nl.sbuf))
    out.append(nl.ndarray((m, n // ABLK, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf))
    return out


def _e8m0_to_bf16x2(raw, dst_u16):
    for k in range(2):
        nisa.tensor_scalar(dst=dst_u16[:, :, k:k + 1], data=raw,
                           op0=nl.multiply, operand0=BF16_EXP_LSB,
                           op1=nl.add, operand1=E8M0_SHIFT)


def _fp8_dequant(byte, pw3, dst3, m, n, s):
    _fix_byte(byte, m, n, s)
    _apply_scale(pw3, dst3, m, n, s)


def _fix_byte(byte, m, n, s):
    nisa.tensor_scalar(dst=s[0].view(nl.uint16), data=byte.view(nl.uint16),
                       op0=nl.bitwise_and, operand0=0x8080)
    nisa.scalar_tensor_tensor(dst=s[1], data=byte, op0=nl.subtract, operand0=8.0,
                              op1=nl.maximum, operand1=s[0])


def _apply_scale(pw32, dst3, m, n, s):
    nb = n // ABLK
    f3 = s[1].view(nl.float8_e4m3).reshape((m, nb, ABLK))
    nisa.tensor_copy(dst=dst3, src=f3, engine=nisa.engine.scalar)
    nisa.tensor_copy(dst=s[2].view(nl.uint32), src=pw32.broadcast(2, ABLK // 2),
                     engine=nisa.engine.scalar)
    nisa.tensor_tensor(dst=dst3, data1=dst3, op=nl.multiply, data2=s[2])


def _stage1(w, i, n0, bytes_, scrs, nrow=0, roff=None):
    d = i % len(bytes_)
    if roff is None:
        nisa.dma_copy(dst=bytes_[d], src=w[i * PAR:(i + 1) * PAR, n0:n0 + NW])
    else:
        # The column run this core owns is a runtime value, so the static slice becomes an
        # access pattern with a dynamic scalar offset on the unit-stride dimension. Same
        # 128 descriptors of 2560 contiguous bytes the static slice generated.
        nisa.dma_copy(dst=bytes_[d],
                      src=w.ap(pattern=[[nrow, PAR], [1, NW]], offset=i * PAR * nrow,
                               scalar_offset=roff, indirect_dim=1))
    _fix_byte(bytes_[d], PAR, NW, scrs[d])


def _embed(ids2, ncol, ew, es, s, su32, stride=None, base=0):
    """The n-gram lookup for `ncol` hash columns from `base`: indirect DMA, fp8 -> bf16."""
    st = ncol if stride is None else stride
    emb = nl.ndarray((TOK, ncol, HD // ABLK, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(ncol):
        idx = nl.ndarray((TOK, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=ids2.ap(pattern=[[st, TOK], [1, 1]], offset=base + j))

        wb = nl.ndarray((TOK, HD), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=wb, src=ew.ap(pattern=[[HD, TOK], [1, HD]], offset=0,
                                        vector_offset=idx, indirect_dim=0))
        sb = nl.ndarray((TOK, HD // ABLK, 1), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=sb, src=es.ap(pattern=[[HD // ABLK, TOK], [1, HD // ABLK]],
                                        offset=0, vector_offset=idx, indirect_dim=0))

        _e8m0_to_bf16x2(sb, su32.view(nl.uint16))
        _fp8_dequant(wb, su32, emb[:, j, :, :], TOK, HD, s)
    return emb.reshape((TOK, ncol * HD))


def _act_quant(x_bf16, m, k):
    nb = k // ABLK
    x2 = nl.copy(x_bf16, dtype=nl.float32)
    xf = x2.reshape((m, nb, ABLK))
    amax = nl.maximum(nl.max(nl.abs(xf), axis=2), AMAX_FLOOR)

    ratio = nl.multiply(amax, 1.0 / FP8_MAX)
    e = nl.copy(nl.ceil(nl.multiply(nl.log(ratio), LOG2E)), dtype=nl.int32)
    # The halve-before-the-cast and double-after-it that keep the quantized byte off fp8
    # exponent 15 were two full-width passes over 128x3072 each. They are powers of two,
    # so they fold into the per-block scales instead -- two more ops on a 128x96 tile,
    # which costs instruction issue and no bandwidth. Exact: scaling by a power of two
    # rounds nothing, and `clamp(x * ib, +-FP8_MAX) * 0.5` is `clamp(x * ib * 0.5,
    # +-FP8_MAX * 0.5)` because the bound scales with the value.
    sc = nl.multiply(_pow2(e, m, nb), 2.0)
    inv = nl.multiply(_pow2(nl.subtract(0, e), m, nb), 0.5)

    sb = nl.broadcast_to(sc.reshape((m, nb, 1)), (m, nb, ABLK))
    ib = nl.broadcast_to(inv.reshape((m, nb, 1)), (m, nb, ABLK))
    v = nl.minimum(nl.maximum(nl.multiply(xf, ib), -FP8_MAX * 0.5), FP8_MAX * 0.5)
    q = nl.copy(nl.copy(v, dtype=nl.float8_e4m3), dtype=nl.float32)
    dq = nl.multiply(q, sb)
    out = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    out[...] = nl.copy(dq.reshape((m, k)), dtype=nl.bfloat16)
    return out


def _transpose_stack(t, k, out=None, t0=0):
    nt = k // PAR
    if out is None:
        out = nl.ndarray((PAR, nt, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(nt):
        out[:, t0 + i, :] = nl.transpose(t[:, i * PAR:(i + 1) * PAR])
    return out


def _wkv_scale_all(ws, nrow, n0, nti, praw, pw32, roff=None):
    ws3 = ws.reshape((1, RBLK * nti, nrow))
    for r in range(RBLK):
        if roff is None:
            src = ws3.ap(pattern=[[0, ABLK], [RBLK * nrow, nti], [1, NWB]],
                         offset=r * nrow + n0 // ABLK)
        else:
            src = ws3.ap(pattern=[[0, ABLK], [RBLK * nrow, nti], [1, NWB]],
                         offset=r * nrow, scalar_offset=roff, indirect_dim=2)
        nisa.dma_copy(dst=praw[r * ABLK:(r + 1) * ABLK, 0:nti, :, :], src=src)
    h16 = pw32.view(nl.uint16)
    for k in range(2):
        nisa.tensor_scalar(dst=h16[:, 0:nti, :, k:k + 1], data=praw[:, 0:nti, :, :],
                           op0=nl.multiply, operand0=BF16_EXP_LSB,
                           op1=nl.add, operand1=E8M0_SHIFT)


@nki.jit
def kernel(input_0, input_1, input_1_shard,
           layers_1_engram_k_weight, layers_1_engram_q_weight,
           layers_1_engram_wkv_key_scale, layers_1_engram_wkv_key_weight,
           layers_1_engram_wkv_value_scale, layers_1_engram_wkv_value_weight,
           layers_1_engram_embed_scale, layers_1_engram_embed_weight,
           replica_group):
    # The module's output, and the only IO tensor here. bfloat16, because the reference
    # rounds the sum back to the input's dtype (`.to(x.dtype)`). The collective may not
    # touch an IO tensor, so the reduce lands in cc_dst and is copied out below.
    out = nl.ndarray((1, TOK, HC, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # The seam's two buffers. Named, both nl.shared_hbm, neither an IO tensor. float32,
    # because the reference's `value` enters the sum as float32 and a bfloat16 reduction
    # would spend the whole of this module's 1-ulp error budget on four roundings.
    # The two physical cores' halves of the transposed activation stack meet here.
    a_sh = nl.ndarray((PAR, NKT, TOK), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                      name="a_sh")
    cc_src = nl.ndarray((TOK, CCW), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((TOK, CCW), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")

    # Which of this rank's two physical cores this instruction stream is. A trace-time
    # constant, so the split below is static and both cores' schedules are fixed.
    pid = nl.program_id(0)

    # Which rank this is. A *runtime* value -- one NEFF serves all four ranks, so this
    # cannot size a tile or bound a loop. It can address one: every loop below keeps its
    # trace-time bounds and only the column run it reads moves.
    rk_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rk_sb, ncc.rank_id())

    koff_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    koff_sb[...] = nl.copy(nl.add(nl.multiply(rk_sb, N_CORES * NW), pid * NW),
                           dtype=nl.int32)

    # `a_t` tile index of this rank's `ngram` quarter. `indirect_dim=1` is not the last
    # dim, so its offset is in units of that dim's stride -- tiles, not elements.
    # uint32, not int32: the DMA path takes an int32 `scalar_offset` but
    # `TensorCopyDynamicSrc` rejects it -- "Expect argument datatype to be of type uint32".
    voff_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    voff_sb[...] = nl.copy(nl.multiply(rk_sb, NVT), dtype=nl.uint32)

    soff_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    soff_sb[...] = nl.copy(nl.add(nl.multiply(rk_sb, N_CORES * NWB), pid * NWB),
                           dtype=nl.int32)

    # `rank == j` for j in 0..3, on every partition. The compare is free; getting a (1, 1)
    # onto 128 partitions is a 1x128 by 1x4 matmul, which is the cheapest broadcast the
    # hardware has.
    ones1 = nl.ndarray((1, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones1, 1.0)
    iot = nl.ndarray((1, WORLD_SIZE), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iot, pattern=[[1, WORLD_SIZE]], channel_multiplier=0)
    oh1 = nl.ndarray((1, WORLD_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    oh1[...] = nl.copy(nl.equal(iot, rk_sb.broadcast(1, WORLD_SIZE)), dtype=nl.bfloat16)
    ohp = nl.ndarray((TOK, WORLD_SIZE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ohp, stationary=ones1, moving=oh1)
    onehot = nl.ndarray((TOK, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=onehot, src=ohp, engine=nisa.engine.scalar)

    h2 = input_0.reshape((TOK, HC * DIM))
    out2 = out.reshape((TOK, HC * DIM))
    ids2 = input_1.reshape((TOK, NHC))
    idsv = input_1_shard.reshape((TOK, NHC_SHARD))
    qw2 = layers_1_engram_q_weight.reshape((1, HC * DIM))
    kw2 = layers_1_engram_k_weight.reshape((1, HC * DIM))

    # One activation, from the replicated table: `a_t`, all 24 hash columns, which the key
    # half contracts over in full. act_quant's blocks are 32 contiguous features and 256 is
    # a multiple of 32, so no block straddles a hash column and this rank's `ngram` quarter
    # -- what the value half contracts over -- is *bit-identical* to tiles
    # `[NVT * rank, NVT * rank + NVT)` of `a_t`. iter-0 had to recompute it from
    # `input_1_shard` because a shared NEFF cannot slice at a compile-time `SHARD`; with
    # `ncc.rank_id()` the slice is a runtime scalar offset on a non-partition dim, so the
    # second embed, act_quant and 12 transposes are simply deleted.
    esc = _scratch(TOK, HD)
    su32 = nl.ndarray((TOK, HD // ABLK, 1), dtype=nl.uint32, buffer=nl.sbuf)
    a_t = nl.ndarray((PAR, NKT, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Core `pid` builds hash columns `[NHC_CORE * pid, NHC_CORE * (pid + 1))`, which are
    # `a_t` tiles `[NKT_CORE * pid, ...)`, and trades halves with the other core through
    # HBM. act_quant's 32-wide blocks are independent and never straddle a hash column, so
    # a half computed alone is bit-identical to that half of the whole. Each core puts its
    # own half on the wire once and reads only the half it does not have: 1.57 MB of
    # exchange against a prefix that was running twice.
    emb = _embed(ids2, NHC_CORE, layers_1_engram_embed_weight,
                 layers_1_engram_embed_scale, esc, su32,
                 stride=NHC, base=NHC_CORE * pid)
    _transpose_stack(_act_quant(emb, TOK, KDIM_CORE), KDIM_CORE,
                     out=a_t, t0=NKT_CORE * pid)
    nisa.dma_copy(dst=a_sh[:, NKT_CORE * pid:NKT_CORE * (pid + 1), :],
                  src=a_t[:, NKT_CORE * pid:NKT_CORE * (pid + 1), :])
    nisa.core_barrier(data=a_sh, cores=(0, 1))
    oq = NKT_CORE * (N_CORES - 1 - pid)
    nisa.dma_copy(dst=a_t[:, oq:oq + NKT_CORE, :], src=a_sh[:, oq:oq + NKT_CORE, :])
    a_tv = nl.ndarray((PAR, NVT, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=a_tv,
                     src=a_t.ap(pattern=[[NKT * TOK, PAR], [TOK, NVT], [1, TOK]],
                                offset=0, scalar_offset=voff_sb, indirect_dim=1),
                     engine=nisa.engine.scalar)

    praw = nl.ndarray((PAR, NKT, NWB, 1), dtype=nl.uint8, buffer=nl.sbuf)
    pw32 = nl.ndarray((PAR, NKT, NWB, 1), dtype=nl.uint32, buffer=nl.sbuf)
    DEPTH = 2
    wdq = []
    wdq2 = []
    scrs = []
    bytes_ = []
    for _ in range(DEPTH):
        t = nl.ndarray((PAR, NWB, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf)
        wdq.append(t)
        wdq2.append(t.reshape((PAR, NW)))
        scrs.append(_scratch(PAR, NW))
        bytes_.append(nl.ndarray((PAR, NW), dtype=nl.uint8, buffer=nl.sbuf))
    kvb = nl.ndarray((TOK, NW), dtype=nl.bfloat16, buffer=nl.sbuf)
    wqb = nl.ndarray((TOK, NW), dtype=nl.bfloat16, buffer=nl.sbuf)
    wkb = nl.ndarray((TOK, NW), dtype=nl.bfloat16, buffer=nl.sbuf)
    rsq = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    rdot = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    rh = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    gate_scr = []
    for _ in range(3):
        gate_scr.append(nl.ndarray((TOK, NW), dtype=nl.float32, buffer=nl.sbuf))

    # ----------------------------------------------------------------- the key half
    # One of the key's 8 output chunks of 2560 -- chunk `2 * rank + pid`, so the device's
    # 8 physical cores cover all 20480 key rows exactly once between them instead of four
    # times over. All 6144 contraction columns, because the gate needs the summed key.
    # Every tile shape and loop bound is the same on every rank; only `koff` moves.
    if True:
        acc = []
        for sl in range(NSL):
            acc.append(nl.ndarray((TOK, NMAX), dtype=nl.float32, buffer=nl.psum))

        _wkv_scale_all(layers_1_engram_wkv_key_scale, NKEY // ABLK, 0, NKT, praw, pw32,
                       roff=soff_sb)
        _stage1(layers_1_engram_wkv_key_weight, 0, 0, bytes_, scrs, NKEY, koff_sb)
        for i in range(NKT):
            d = i % DEPTH
            if i + 1 < NKT:
                _stage1(layers_1_engram_wkv_key_weight, i + 1, 0, bytes_, scrs,
                        NKEY, koff_sb)
            _apply_scale(pw32[:, i, :, :], wdq[d], PAR, NW, scrs[d])
            for sl in range(NSL):
                nisa.nc_matmul(dst=acc[sl], stationary=a_t[:, i, :],
                               moving=wdq2[d][:, sl * NMAX:(sl + 1) * NMAX],
                               accumulate=(i > 0))

        for sl in range(NSL):
            nisa.tensor_copy(dst=kvb[:, sl * NMAX:(sl + 1) * NMAX], src=acc[sl],
                             engine=nisa.engine.scalar)

        keyf = gate_scr[0]
        hf = gate_scr[1]
        hw = gate_scr[2]
        hf[...] = nl.load(h2.ap(pattern=[[HC * DIM, TOK], [1, NW]], offset=0,
                                scalar_offset=koff_sb, indirect_dim=1),
                          dtype=nl.float32)
        nisa.dma_copy(dst=wqb, src=qw2.ap(pattern=[[0, TOK], [1, NW]], offset=0,
                                          scalar_offset=koff_sb, indirect_dim=1))
        nisa.dma_copy(dst=wkb, src=kw2.ap(pattern=[[0, TOK], [1, NW]], offset=0,
                                          scalar_offset=koff_sb, indirect_dim=1))
        nisa.tensor_tensor(dst=hw, data1=wqb, op=nl.multiply, data2=wkb)
        hw[...] = nl.multiply(hw, hf)
        nisa.tensor_tensor(dst=hw, data1=hw, op=nl.multiply, data2=kvb)
        nisa.activation(dst=keyf, op=nl.square, data=kvb, reduce_op=nl.add,
                        reduce_res=rsq, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        # `h` is in SBUF here for the dot product, so the residual's own row sum is one
        # more free-accumulator pass rather than a second 2.6 MB read in the tail.
        nisa.activation(dst=keyf, op=nl.square, data=hf, reduce_op=nl.add,
                        reduce_res=rh, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.activation(dst=keyf, op=nl.copy, data=hw, reduce_op=nl.add,
                        reduce_res=rdot, reduce_cmd=nisa.reduce_cmd.reset_reduce)

    # This core's three gate partials, placed in the slot its *rank* owns and zero in the
    # other three. `onehot` is exactly 1.0 or 0.0, so a slot carries one rank's value
    # unrounded and the reduce adds three exact zeros to it.
    gpart = (rsq, rdot, rh)
    pad = nl.ndarray((TOK, GPAD_CORE), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(WORLD_SIZE):
        for k in range(GSLOT):
            pad[:, j * GSLOT + k:j * GSLOT + k + 1] = nl.multiply(onehot[:, j:j + 1],
                                                                  gpart[k])

    # --------------------------------------------------------------- the value half
    # This rank's 1536 contraction columns -- its quarter of `ngram` -- over the output
    # columns this physical core owns. A partial sum of the module's `value`.
    value = nl.ndarray((TOK, NVAL), dtype=nl.float32, buffer=nl.sbuf)
    v0 = pid * (NVAL // N_CORES)
    for ci in range(NVAL // N_CORES // NW):
        n0 = v0 + ci * NW
        acc = []
        for sl in range(NSL):
            acc.append(nl.ndarray((TOK, NMAX), dtype=nl.float32, buffer=nl.psum))

        _wkv_scale_all(layers_1_engram_wkv_value_scale, NVAL // ABLK, n0, NVT, praw, pw32)
        _stage1(layers_1_engram_wkv_value_weight, 0, n0, bytes_, scrs)
        for i in range(NVT):
            d = i % DEPTH
            if i + 1 < NVT:
                _stage1(layers_1_engram_wkv_value_weight, i + 1, n0, bytes_, scrs)
            _apply_scale(pw32[:, i, :, :], wdq[d], PAR, NW, scrs[d])
            for sl in range(NSL):
                nisa.nc_matmul(dst=acc[sl], stationary=a_tv[:, i, :],
                               moving=wdq2[d][:, sl * NMAX:(sl + 1) * NMAX],
                               accumulate=(i > 0))

        for sl in range(NSL):
            nisa.tensor_copy(dst=value[:, n0 + sl * NMAX:n0 + (sl + 1) * NMAX],
                             src=acc[sl], engine=nisa.engine.scalar)

    # -------------------------------------------------------------------- the seam
    # The one point in the chain where a rank holds a tensor of the module's `value`
    # shape carrying an incomplete sum, and therefore the only place the collective can
    # sit. The two physical cores fill disjoint column runs of `cc_src` -- the same runs
    # they computed -- and the barrier is what makes the other core's columns visible to
    # the read the collective is about to do.
    nisa.dma_copy(dst=cc_src[:, v0:v0 + NVAL // N_CORES],
                  src=value[:, v0:v0 + NVAL // N_CORES])
    # 24 more columns, 0.5% more payload: the gate's partials, which are the only thing
    # the key-half split needs to put on the wire.
    nisa.dma_copy(dst=cc_src[:, NVAL + pid * GPAD_CORE:NVAL + (pid + 1) * GPAD_CORE],
                  src=pad)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], op=nl.add, replica_group=replica_group)
    # After the reduce every rank holds the module's whole `value`, rank 0 included, and
    # all 8 (rank, core) gate partials.
    nisa.dma_copy(dst=value, src=cc_dst[:, 0:NVAL])
    gpad = nl.ndarray((TOK, GPAD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gpad, src=cc_dst[:, NVAL:NVAL + GPAD])

    # ------------------------------------------------------------------- the output
    # `h + gate * value`, rounded to the module's bfloat16 by the store's own DMA. Each
    # physical core writes the two hc copies whose gate it formed.
    for cc in range(HC_PER_CORE):
        c = pid * HC_PER_CORE + cc
        # hc copy `c` is the two key chunks `2c` and `2c + 1`, which rank `c`'s core 0 and
        # core 1 formed. Summed in that order -- the order the one-core loop used.
        b0 = c * GSLOT
        b1 = GPAD_CORE + c * GSLOT
        ksqc = nl.add(gpad[:, b0:b0 + 1], gpad[:, b1:b1 + 1])
        dotc = nl.add(gpad[:, b0 + 1:b0 + 2], gpad[:, b1 + 1:b1 + 2])
        hsqc = nl.add(gpad[:, b0 + 2:b0 + 3], gpad[:, b1 + 2:b1 + 3])
        hsq = nl.multiply(hsqc, 1.0 / DIM)
        kms = nl.multiply(ksqc, 1.0 / DIM)
        rstd = nl.multiply(nl.rsqrt(nl.add(hsq, EPS)), nl.rsqrt(nl.add(kms, EPS)))
        dot = nl.multiply(nl.multiply(dotc, rstd), DIM_RSQRT)
        mag = nl.sqrt(nl.maximum(nl.abs(dot), CLAMP))
        sgn = nl.subtract(nl.multiply(nl.copy(nl.greater_equal(dot, 0.0),
                                              dtype=nl.float32), 2.0), 1.0)
        gate = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
        gate[...] = nl.sigmoid(nl.multiply(mag, sgn))
        for sl in range(DIM // NW):
            n0 = c * DIM + sl * NW
            hf = gate_scr[0]
            yv = gate_scr[1]
            hf[...] = nl.load(h2[:, n0:n0 + NW], dtype=nl.float32)
            nisa.activation(dst=yv, op=nl.copy,
                            data=value[:, sl * NW:(sl + 1) * NW], scale=gate)
            nisa.tensor_tensor(dst=yv, data1=yv, op=nl.add, data2=hf)
            nl.store(out2[:, n0:n0 + NW], value=yv)

    return out
