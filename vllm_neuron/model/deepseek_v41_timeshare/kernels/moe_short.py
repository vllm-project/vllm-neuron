"""All of `layers.1.ffn` on 4 ranks, rejoining through `nki.collectives`.

Every rank runs the whole `ffn_norm -> router` prologue (replicated), the SwiGLU experts for
its own 96-expert shard, and one quarter of the shared expert. It adds all of that into one
float32 accumulator, and `ncc.all_reduce` sums the four accumulators. The sum is the module's
output for every rank, rounded to the module's bfloat16.

The routed-expert body is `submodule/source.py` at its best accepted commit, unchanged.

Two things here are in neither predecessor.

**The shared expert is sharded, not replicated.** `MoE.forward` adds it after `all_reduce`, so
the submodule left it out of a rank's partial entirely. The reduce is a sum, so at module scale
the shared expert only has to appear across the four partials exactly once in total -- and
splitting it beats replicating it, because SwiGLU is elementwise along the intermediate
dimension and `w2` contracts over it. Rank r owns intermediate columns `r*640 .. r*640+639`,
computes `w1`/`w3` into them, the SwiGLU on them, and the `w2` rows they contract with; the four
`w2` partials sum to the whole shared expert. The intermediate is padded 2304 -> 2560 so that all
four ranks get the same shape (and so one NEFF serves all four); the padding is zero weights,
which contribute zero.

**The fp8 weights are decoded by hand.** `module/source.py` decodes each fp8 weight tile with about
fifty elementwise operations -- a sign/exponent/mantissa reconstruction, and a `_pow2` assembled
from six conditional multiplies; `_fp8_dequant` builds the bf16 bit pattern directly in four,
two of them on a uint32 view that carries two weights per lane, and `_scale_rows` folds the
decode's `2**-120` back into the e8m0 scale's exponent so the correction is free. The hardware's
own fp8 converter is not an option: `nl.float8_e4m3` and `nl.float8_e4m3fn` both read exponent 15
as inf/NaN on this toolchain, and 0.6% of these weight bytes (61399 of w1, 71502 of w2, 78556 of
w3 -- every `|w| >= 256`) live there, so converting through it produces NaN rather than a small
error.

**The rank's work is split across the two physical cores of its logical core.** This host is
LNC=2, so the kernel is launched as `kernel[2]` and a rank's logical core is two physical cores.
They used to run the identical program -- all 96 experts and the whole shared-expert slice,
twice, for one copy of the answer, which showed up as 3.911 GB of HBM reads against 1.82 GB of
real inputs. Now core `r` runs routed experts `r*48 .. r*48+47` and intermediate columns
`SH_C0[r] .. SH_C0[r]+SH_CW[r]-1` of the shared-expert slice, and the two partials are rejoined
inside the logical core by `nisa.core_barrier` before the collective. That is worth 1.95x and
costs nothing in accuracy: the partials are summed in float32, exactly as the four ranks' are.

Two things make the split cheap. `nl.program_id(0)` is a **trace-time constant** -- the launch
grid is traced once per physical core and the NEFF carries both programs -- so each core's share
is chosen with ordinary Python, with no dynamic trip count and no predication. And the shared
expert splits by intermediate column for the same reason it splits across ranks: SwiGLU is
elementwise along that dimension and `w2` contracts over it, so any whole number of 128-row GEMM
tiles is a self-contained partial. The 96 experts split 48/48 exactly; the 640 intermediate
columns split 256/384, the nearest split into whole tiles, which leaves core 0 waiting about
0.2 ms at the barrier -- a tenth of the shared expert rather than half of it.

**The routed GEMM's k-tile loop is a software pipeline.** Dequantising one fp4 weight tile is a
chain across four engines -- the DMA of its bytes, the nibble isolate on Vector, the expand into
bf16 bit patterns on Scalar, the mask and block-scale multiply back on Vector, and the matmul on
the PE -- and the submodule's schedule emitted the whole chain for one k-tile before starting the
next, relying on the four (column block x nibble) slices *inside* a k-tile to keep the engines
busy past each other. That is why the chain is written as four narrow instructions there: the
profile of the version that coalesces them into one shows every widened instruction is cheaper per
element (the chain's engine time falls 31.5 -> 27.0 ms per core) and the kernel 1.8x slower,
because with one slice per k-tile matmul/dequant overlap collapses from 89% to 2.3%. `STOFF` takes
the coalesced instructions and gets the overlap from the other direction: at group `g` the six
stages act on six *different* k-tiles, so nothing in a group depends on anything else in it and
each engine runs at its own pace. The k-tiles are still processed in order and each PSUM tile
still accumulates in the same order, so the output is bit-identical to the unpipelined kernel.

The kernel is rank-agnostic on purpose -- one NEFF runs on all four ranks. A rank's identity
lives entirely in the values `inference.py` hands it: its own 96 experts' weights, its slice of
the shared expert, and a `gate_wt`/`gate_b` whose 384 columns are permuted so that this rank's
experts are columns 0..95. Top-6 selection and the renormalising sum are invariant under a
column permutation, so the router still scores all 384 experts; only the column at which a
local expert's weight lands moves. Which of a rank's 96 experts a *core* owns is likewise not in
the weights: it is `nl.program_id(0)` times 48, added to the same static expert offset the
unroll already used.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

DIM = 5120
INTER = 2304
INTER_PAD = 2560          # 2304 rounded up to a multiple of 4 * 128, so all ranks match
SHARDS = 4
INTER_L = INTER_PAD // SHARDS
N_EXPERTS = 384
TOPK = 6
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
NORM_EPS = 1e-20
SUM_EPS = 1e-20
TOK = 128

PAR = 128
NMAX = 512
ABLK = 32
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
LOG2E = 1.4426950408889634

CH = 96
CORES = 2                 # LNC=2: the rank's logical core is two physical cores, kernel[2]
CHH = CH // CORES         # routed experts per physical core
# The two cores also split this rank's shared-expert slice. INTER_L is 640 = 5 * PAR, so the
# nearest split into whole 128-row GEMM tiles is 2/3; the routed experts are 48/48 either way,
# so the residual imbalance is 1/10 of the shared expert, not 1/2 of it.
SH_C0 = (0, 256)
SH_CW = (256, INTER_L - 256)
BPB = ABLK // 2
KBLK1 = DIM // ABLK
KBLK2 = INTER // ABLK
KB1 = DIM // 2
KB2 = INTER // 2
NW1 = 1152
NW2 = 1280
NBG1 = 2
NBG2 = 2
KSP1 = 1
KSP2 = 1

SCENG1 = (nisa.engine.scalar,)
SCENG2 = (nisa.engine.scalar,)
DRNENG = (nisa.engine.scalar,)
TRENG = (nisa.engine.scalar,)
PRENG1 = (nisa.engine.scalar,)
PRENG2 = (nisa.engine.scalar,)

# Which engine isolates each fp4 nibble, cycled over (k-tile, job): the low-nibble mask and the
# high-nibble shift+mask. The split kernel's profile has Vector at 92% of wall against GpSimd's
# 38%, so moving a share off Vector looked free; `((gpsimd, vector), (vector,))` measured
# 17.8764 ms against 17.8774 -- flat, and it compiled to the same HLO module id, so the
# `engine=` hint on a uint16-view tensor_scalar appears to be dropped rather than honoured
# (the `tensor_tensor` hint below does change it). Vector for all of it, as the submodule left it.
NIBENG = ((nisa.engine.vector,),
          (nisa.engine.vector,))

# Which engine multiplies the dequantised nibble by its block scale, cycled over
# (k-tile, job, nibble). This is the largest Vector line in the profile (8.28 ms of Vector's
# 16.41 ms busy) and GpSimd idles for 10.3 ms of the loop, so a tenth of it looked worth
# GpSimd's penalty. It is not: `(gpsimd,) + (vector,) * 9` measured 29.1412 ms against
# 17.8774 -- 0.83 ms of Vector work cost ~11.3 ms of wall, so GpSimd is far worse here than the
# ~10x per element that `_fp8_dequant`'s own GpSimd line implies, and its 10.3 ms of idle is not
# usable capacity for this chain. Vector for all of it is the measured best.
MULENG = (nisa.engine.vector,)

# Which engine scales the decoded fp8 weight tile in `_fp8_dequant`. This one only runs in the
# shared expert, and that window is latency-bound, not throughput-bound: the profile finds a
# quarter of it with every engine idle. GpSimd, which the bootstrapped module used here, sits on
# the DMA -> dequant -> matmul dependency chain for 366 us of core 1's 985 us window.
# Moving it to Vector is worth 0.30 ms: 17.5789 ms against 17.8774. Two neighbours of it were
# measured and left alone -- `_scale_rows`'s bit-build on Scalar (17.5881, a real regression:
# 320 instructions of 8-12 elements each, so its cost is per-instruction overhead that Scalar
# does not carry any better) and `_fp8_dequant`'s magnitude mask on Scalar (no effect, and the
# same HLO module id, so that hint was dropped -- see NIBENG).
FP8ENG = nisa.engine.vector

EXP_BIAS = 32
BF_BIAS = 127
F32_MANT = 23
F32_MANT_MAX = (1 << F32_MANT) - 1
RATIO_MAX = float(1 << (EXP_BIAS - 1))
RATIO_MIN_BITS = ((BF_BIAS - EXP_BIAS - 1) << F32_MANT) + 1
BF_STEP = 128.0
BF_HALF_BIAS = float((2 * BF_BIAS - 1) << 7)

NIB_MUL = 4160.0
PAT_MASK2 = -2118090304
# The nibble isolate's lane width. Vector charges per *element* and barely for bytes -- the fit
# in iter-2's notes is `27 ns + elem * (0.205 + 0.070 * bytes_per_elem)`, and the profile measures
# the 4-byte mask at 0.607 ns/element against the 2-byte scale multiply's 0.572 -- so the way to
# make a bitwise line cheaper is to put *more bytes in each element*, not fewer elements' worth of
# bytes. The isolate is a per-byte mask, so it is correct at any lane width: `& 0x0F0F0F0F` keeps
# each byte's low nibble, and `(x >> 4) & 0x0F0F0F0F` drops the four bits the shift dragged in
# from the next byte up exactly as the 2-byte version dropped them from the next byte within the
# pair. Four bytes per lane halves the element count for byte-identical output.
QUAD_NIB_MASK = 0x0F0F0F0F
HI_NIB_SHIFT = 4
# How many routed experts share one `fori_loop` body. Each loop boundary is a scheduling barrier
# that drains the k-tile pipeline and costs about 12 us, so raising this from the submodule's 4 is
# worth 0.11 ms across a core's 48 experts: 4 / 6 / 8 / 12 / 24 measured 14.967 / 14.9155 /
# 14.8936 / 14.8703 / 14.8539 ms, monotone. It stops there -- 48, which unrolls the loop away
# entirely, regresses to 14.9483, so past 24 the basic block costs more than the barrier saves.
# Nothing else changes: an expert's instructions and `y`'s accumulation order are untouched, so
# every value here is bit-identical to the 4-deep version.
UNROLL = 24
# The routed GEMM's k-tile loop is a hand-written software pipeline: at group `g` the six
# stages act on six *different* k-tiles, `g - STOFF[st]` each. `(0,) * 6` is the submodule's
# schedule, where all six stages of one k-tile are emitted together and the chain
# DMA -> isolate(Vector) -> pre(Scalar) -> mask+scale(Vector) -> matmul(PE) runs end to end
# inside one group; the profile of the coalesced variant showed that chain costing 9 us a group
# against 4.9 us of Vector work in it. Skewing the stages leaves no intra-group dependency at
# all, so every engine works on its own k-tile and the group costs what the slowest engine
# costs. The offsets are bounded by how many buffers rotate: a k-tile's weight bytes live from its
# DMA to its isolate (`DDEP` of them), its nibbles from isolate to pre (`NDEP`), and its bf16
# patterns from pre to matmul (`pdep`), so no single buffer may be needed for longer than its own
# stage's depth.
#
# Two offsets are sensitive. How far the matmul trails its own dequant was swept in iter-2 at
# 3/4/5/6 groups (16.6775 / 15.9226 / 16.2168 / 16.1592 ms) and how far the mask trails the expand
# that feeds it was swept here: 2 groups beats 1 by 0.14 ms and 3 loses again by 0.04, which is why
# the last two offsets are 3 and 5. Shortening the *span* from the other end -- one group of DMA
# lead instead of two -- costs 0.16 ms, and iter-2's finding that extra DMA lead buys nothing still
# holds, so the span is pinned from both sides.
#
# One reason recorded for the matmul offset in iter-2 does not survive a profile of this kernel:
# the PE is *not* nearly as loaded as Vector. Measured here, PE is 8.08 ms busy (53%) against
# Vector's 14.34 ms (94%), so the elasticity the PE wants is about its matmuls arriving in lumpy
# 13-deep bursts, not about the engine being near capacity. The offset itself is a measurement and
# stands; the explanation was wrong.
STOFF = (-2, 0, 0, 1, 3, 5)
# How deep each rotating stage's buffers are. These used to all be `pdep`, the widest of the
# three, which is both too much and too little: the weight and scale bytes are dead once the
# isolate and the scale build have read them, and the nibbles once the expand has, so tying them to
# the *patterns'* liveness (expand to matmul, the longest of the three) spends SBUF where it cannot
# help and leaves none for widening the skews.
#
# Decoupling them is what let the expand -> mask skew grow, but the surprise is that the right depth
# is **not** the minimum. Cutting each to exactly its own liveness (DDEP 3, NDEP 2 here) costs
# 0.29 ms: depth past the minimum is write-after-read elasticity, because a buffer of depth `D`
# written by stage `A` and read by stage `B` only gives its writer `A - B + D` groups of slack
# before it waits on the reader `D` positions back, and the asserts below guarantee just 1. The
# extra 2 and 3 are measured: 5/5 with `pdep` 6/7 is worth 0.30 ms over the minimum, and 6/6 with
# `pdep` 7/8 is flat, so this is bracketed from both sides.
DDEP = 1 + STOFF[1] - STOFF[0] + 2
NDEP = 1 + STOFF[3] - STOFF[1] + 3
assert 1 + STOFF[2] - STOFF[0] <= DDEP
assert CH % UNROLL == 0
assert CHH % UNROLL == 0
PDEP1 = 6
PDEP2 = 7
CHSP1 = True
CHSP2 = True
SC_MUL = 128.0
SC_ADD = 16128.0

# fp8 e4m3fn weight decode, see _fp8_dequant / _scale_rows
BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00
E4M3_SGN2 = 0x80008000 - (1 << 32)          # 0x80008000 written as a signed int32
CCH = 1024                                  # column chunk of the cross-core partial add
assert DIM % CCH == 0
SC_MUL8 = 128.0
SC_ADD8 = float(120 << 7)                   # folds the decode's 2**-120 back in


def _act_quant(x_bf16, m, k, chunks):
    nb = k // ABLK
    xf = x_bf16.reshape((m, nb, ABLK))
    mx = nl.ndarray((m, nb), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.max, data=xf, axis=2)
    nmn = nl.ndarray((m, nb), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=nmn, op=nl.min, data=xf, axis=2, negate=True)
    amax = nl.ndarray((m, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=amax, data=mx, op0=nl.maximum, operand0=AMAX_FLOOR,
                              op1=nl.maximum, operand1=nmn)

    ratio = nl.ndarray((m, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ratio, data=amax, op0=nl.multiply, operand0=1.0 / FP8_MAX,
                       op1=nl.minimum, operand1=RATIO_MAX, engine=nisa.engine.vector)
    eb = nl.ndarray((m, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eb, data=ratio.view(nl.int32), op0=nl.maximum,
                       operand0=RATIO_MIN_BITS, op1=nl.add, operand1=F32_MANT_MAX,
                       engine=nisa.engine.vector)
    ef = nl.ndarray((m, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ef, data=eb, op0=nl.right_shift, operand0=F32_MANT,
                       engine=nisa.engine.vector)

    sb = nl.ndarray((m, nb), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sb, data=ef, op0=nl.multiply, operand0=BF_STEP,
                       engine=nisa.engine.vector)
    s = sb.view(nl.bfloat16)
    ivb = nl.ndarray((m, nb), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ivb, data=ef, op0=nl.multiply, operand0=-BF_STEP,
                       op1=nl.add, operand1=BF_HALF_BIAS, engine=nisa.engine.vector)
    inv = ivb.view(nl.bfloat16)

    ib = inv.reshape((m, nb, 1)).broadcast(2, ABLK)

    q = nl.ndarray((m, nb, ABLK), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q, data1=xf, data2=ib, op=nl.multiply,
                       engine=nisa.engine.vector)
    q4 = q.reshape((m, nb, BPB, 2))
    fls = []
    for c in range(len(chunks)):
        kb0 = chunks[c][0]
        nk = chunks[c][1]
        s2 = nl.multiply(s[:, kb0:kb0 + nk], 2.0).reshape((m, 1, 1, nk))
        fl = nl.ndarray((m, 2, BPB, nk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=fl,
                           data1=q4[:, kb0:kb0 + nk, :, :].permute((0, 3, 2, 1)),
                           data2=s2.broadcast(1, 2).broadcast(2, BPB),
                           op=nl.multiply, engine=nisa.engine.vector)
        fls.append(fl)
    return fls


def _transpose_stack(t, k):
    nt = k // PAR
    out = nl.ndarray((PAR, nt, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(nt):
        out[:, i, :] = nl.transpose(t[:, i * PAR:(i + 1) * PAR])
    return out


def _swiglu(g_bf16, u_bf16, wcol, dst):
    m, k = g_bf16.shape
    g = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g, data=g_bf16, op0=nl.minimum, operand0=SWIGLU_LIMIT,
                       engine=nisa.engine.vector)
    u = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=u_bf16, op0=nl.maximum, operand0=-SWIGLU_LIMIT,
                       op1=nl.minimum, operand1=SWIGLU_LIMIT, engine=nisa.engine.vector)
    sg = nl.ndarray((m, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    nisa.scalar_tensor_tensor(dst=dst, data=sg, op0=nl.multiply, operand0=wcol,
                              op1=nl.multiply, operand1=u)


def _flat_chunks(kblk, split):
    if not split:
        return [(0, kblk, 0, kblk * BPB // PAR)]
    ch = []
    kb0 = 0
    while kb0 < kblk:
        nk = PAR
        while nk > kblk - kb0:
            nk //= 2
        assert nk * BPB % PAR == 0
        ch.append((kb0, nk, PAR // nk, nk * BPB // PAR))
        kb0 += nk
    return ch


def _flat_plan(kblk, split):
    tiles = []
    if not split:
        nt = kblk * BPB // PAR
        for _ in range(nt):
            tiles.append([])
        for off in range(BPB):
            kb = 0
            while kb < kblk:
                f = off * kblk + kb
                t = f // PAR
                p = f - t * PAR
                ln = min(kblk - kb, PAR - p)
                tiles[t].append((p, off, kb, ln))
                kb += ln
        return tiles
    chs = _flat_chunks(kblk, True)
    for c in range(len(chs)):
        kb0 = chs[c][0]
        nk = chs[c][1]
        nof = chs[c][2]
        for a in range(chs[c][3]):
            segs = []
            for i in range(nof):
                segs.append((i * nk, a * nof + i, kb0, nk))
            tiles.append(segs)
    return tiles


def _assert_covered(segs, name):
    for t in range(len(segs)):
        cov = []
        for _ in range(PAR):
            cov.append(0)
        for si in range(len(segs[t])):
            p = segs[t][si][0]
            ln = segs[t][si][3]
            for i in range(p, p + ln):
                cov[i] += 1
        for i in range(PAR):
            assert cov[i] == 1, \
                "%s tile %d partition %d covered %d times, not once" % (name, t, i, cov[i])


def _scale_plan(segs):
    keys = []
    idx = []
    for t in range(len(segs)):
        key = []
        for si in range(len(segs[t])):
            key.append((segs[t][si][0], segs[t][si][2], segs[t][si][3]))
        key = tuple(key)
        if key not in keys:
            keys.append(key)
        idx.append(keys.index(key))
    order = []
    grp = []
    for p in range(len(keys)):
        for t in range(len(segs)):
            if idx[t] == p:
                order.append(t)
                grp.append(p)
    return len(keys), order, grp


CHK1 = _flat_chunks(KBLK1, CHSP1)
CHK2 = _flat_chunks(KBLK2, CHSP2)
SEG1 = _flat_plan(KBLK1, CHSP1)
SEG2 = _flat_plan(KBLK2, CHSP2)
_assert_covered(SEG1, "SEG1")
_assert_covered(SEG2, "SEG2")
SSC1 = _scale_plan(SEG1)
SSC2 = _scale_plan(SEG2)


def _flat_tr(fls, chunks):
    nch = len(chunks)
    nt = 0
    for c in range(nch):
        nt += chunks[c][3]
    out = nl.ndarray((PAR, 2 * nt, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    ci = 0
    t = 0
    for c in range(nch):
        flat = fls[c].reshape((TOK, 2, BPB * chunks[c][1]))
        for a in range(chunks[c][3]):
            for high in range(2):
                tp = nl.ndarray((PAR, TOK), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=flat[:, high, PAR * a:PAR * a + PAR])
                nisa.tensor_copy(dst=out[:, 2 * t + high, :], src=tp,
                                 engine=TRENG[ci % len(TRENG)])
                ci += 1
            t += 1
    return out


def _quant_layout(x, m, k, chunks):
    return _flat_tr(_act_quant(x, m, k, chunks), chunks)


def _fp4_dequant(q, sc, gw, mul_eng=nisa.engine.vector):
    """In place: clear `q`'s garbage bits and scale, for a whole (job-group, nibble) tile.

    `q` is (PAR, 2 * gw) uint16 -- the low nibbles' bf16 patterns then the high nibbles' -- and
    `sc` is the group's (PAR, gw) block-scale tile. Both nibbles of a weight byte sit in the same
    32-element scale block, so one stride-0 broadcast of `sc` along the nibble axis serves both:
    the mask is one instruction over the whole tile and so is the scale multiply, where the
    per-(job, nibble) version issued four of each. Bit-identical to that version.
    """
    nisa.tensor_scalar(dst=q.view(nl.uint32), data=q.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=PAT_MASK2, engine=nisa.engine.vector)
    dq = q.view(nl.bfloat16).reshape((PAR, 2, gw))
    nisa.tensor_tensor(dst=dq, data1=dq,
                       data2=sc.view(nl.bfloat16).reshape((PAR, 1, gw)).broadcast(1, 2),
                       op=nl.multiply, engine=mul_eng)


def _dma_buf(nw, pdep):
    bf = []
    for _ in range(pdep):
        bf.append([nl.zeros((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf),
                   nl.zeros((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)])
    return bf


def _alloc_db(nbg, nw, nj, pdep):
    if nbg > 1:
        bf = []
        for _ in range(nj // nbg):
            bf.append(_dma_buf(nbg * nw, pdep))
        return bf
    bf = []
    for _ in range(nj):
        bf.append(_dma_buf(nw, pdep))
    return bf


def _gemm_fp4(ws, ss, j, kbytes, kblk, n, nw, nbg, segs, ssc, alay, dbufs, outs, yacc,
              sceng, preng, ksp, pdep, d):
    ng = len(ws)
    widths = []
    for q in range(0, nw, NMAX):
        widths.append(min(NMAX, nw - q))
    nt = len(segs)
    nbt = n // nw
    assert nt % ksp == 0
    ntc = nt // ksp
    # A *pass* is one matrix's one group of `nbg` column blocks -- the most output columns whose
    # float32 accumulators fit PSUM at once, which is what forces w1, w3 and the two halves of w2
    # to be four passes rather than one (2304 or 2560 columns of the 4096 a partition holds).
    # Each pass used to be its own pipeline and paid its own fill and drain: 7 groups of ramp
    # against w1's 20 k-tiles of work and w2's 9, 192 times per core. Their k-tile sequences are
    # concatenated into one pipeline here, so the next pass's DMA, isolate and expand issue during
    # the previous pass's last masks and matmuls, and only the two ends of the whole expert ramp.
    # Only the PSUM accumulators stay per-pass; a pass's last matmul is issued one group before
    # the next pass's first, so their live ranges are disjoint and the allocator reuses the banks.
    # Positions are still processed in order and each accumulator still takes its (k-tile, nibble)
    # contributions in the same order, so the output is bit-identical.
    passes = []
    for gm in range(ng):
        for nb0 in range(0, nbt, nbg):
            passes.append((gm, nb0))
    npass = len(passes)
    npos = npass * ntc
    jnb = []
    jt = []
    ja = []
    for i in range(nbg):
        for kc in range(ksp):
            jnb.append(i * nw)
            jt.append(kc * ntc)
            ja.append(i)
    njob = len(jnb)
    merge = (nbg > 1)
    # Jobs whose weight bytes already live in one DMA buffer are dequantised by *one*
    # instruction each, not one per job: with `merge`, the jobs sharing a `kc` hold adjacent
    # column blocks of `dbufs[gm][kc]`, so a group's nibbles, block scales and bf16 patterns can
    # be one contiguous tile and the chain issues 4 instructions per k-tile -- 2 nibble
    # isolates, 1 mask, 1 scale multiply -- where the per-job version issued 12. What that
    # buys is per *element*, not per instruction: Vector charges roughly
    # `27 ns + elem * (0.205 + 0.070 * bytes_per_elem)`, so widening takes the mask from 0.787
    # to 0.588 ns/element, the scale multiply from 0.650 to 0.554 and the Scalar expand from
    # 0.994 to 0.913 -- 27.0 ms of engine time for the chain against 31.5. Bits are unchanged
    # (verified standalone against the per-job path). Wider is only a win together with
    # `STOFF`: on its own it measured 1.8x slower, because those four narrow slices per k-tile
    # were the schedule's only source of overlap.
    if merge:
        groups = []
        for kc in range(ksp):
            gl = []
            for i in range(nbg):
                gl.append(kc + ksp * i)
            groups.append(gl)
    else:
        groups = []
        for q in range(njob):
            groups.append([q])
    ngp = len(groups)
    gwid = []
    for gi in range(ngp):
        gwid.append(len(groups[gi]) * nw)
    jobgrp = []
    for q in range(njob):
        for gi in range(ngp):
            for qi in range(len(groups[gi])):
                if groups[gi][qi] == q:
                    jobgrp.append((gi, qi))
    assert len(jobgrp) == njob
    nib = []
    pre = []
    for _ in range(NDEP):
        gn = []
        for gi in range(ngp):
            # One tile for both nibbles, low half then high, so the expand that reads it is one
            # instruction over 2 * gwid elements instead of two over gwid. Same elements, one
            # fewer Scalar instruction per k-tile -- and Scalar has only ~9% slack over Vector
            # in this loop, so its instruction count is worth something on its own.
            gn.append(nl.ndarray((PAR, 2 * gwid[gi]), dtype=nl.uint8, buffer=nl.sbuf))
        nib.append(gn)
    for _ in range(pdep):
        gp = []
        for gi in range(ngp):
            gp.append(nl.ndarray((PAR, 2 * gwid[gi]), dtype=nl.uint16, buffer=nl.sbuf))
        pre.append(gp)
    if ksp == 1 and ssc[0] < nt:
        torder = ssc[1]
        tgrp = ssc[2]
    else:
        torder = []
        tgrp = []
        for t in range(ntc):
            torder.append(t)
            tgrp.append(t)
    nkey = tgrp[ntc - 1] + 1
    scb = []
    for _ in range(pdep):
        gs = []
        for gi in range(ngp):
            gs.append(nl.ndarray((PAR, gwid[gi]), dtype=nl.uint16, buffer=nl.sbuf))
        scb.append(gs)
    dsc = []
    dwt = []
    for gm in range(ng):
        hs = []
        hw = []
        for gi in range(ngp):
            gs = []
            gw = []
            q0 = groups[gi][0]
            kc = q0 % ksp
            for c in range(DDEP):
                if merge:
                    gs.append(dbufs[gm][kc][c][0])
                    gw.append(dbufs[gm][kc][c][1].view(nl.uint32))
                else:
                    gs.append(dbufs[gm][q0][c][0])
                    gw.append(dbufs[gm][q0][c][1].view(nl.uint32))
            hs.append(gs)
            hw.append(gw)
        dsc.append(hs)
        dwt.append(hw)
    accs = []
    for _ in range(npass):
        accs.append(None)
    # Coalescing the dequant chain into one instruction per (job-group, nibble) makes every
    # instruction cheaper per element -- the profile puts the whole chain at 27.0 ms of
    # engine time against 31.5 ms for the per-slice version -- but it deletes the
    # *intra*-k-tile parallelism the narrow version relied on: with one slice per k-tile
    # there is nothing left for the PE to chew on while Vector masks and scales, and
    # matmul/Vector overlap collapsed from 89% to 2.3% of matmul busy (31.42 ms wall).
    # Issuing each stage on its own k-tile (STOFF) restores the overlap across k-tiles
    # instead -- the PE works on one k-tile while Vector dequantises another and Scalar
    # expands a third, out of different `pre`/`nib` buffers. Positions are still processed in
    # order and each PSUM tile still accumulates its (k-tile, nibble) contributions in the
    # same order, so the result is bit-identical.
    #
    # Per-buffer lifetimes, in groups: the weight/scale bytes from their DMA to the isolate
    # and scale-build that read them, the nibbles from isolate to pre, the bf16 patterns from
    # pre to matmul. Each must fit in the buffers that rotate for it.
    assert 1 + STOFF[1] - STOFF[0] <= DDEP and 1 + STOFF[2] - STOFF[0] <= DDEP
    assert 1 + STOFF[3] - STOFF[1] <= NDEP
    assert 1 + STOFF[5] - STOFF[3] <= pdep
    # The block-scale tiles are the one buffer indexed by scale *group* rather than by position,
    # and a group's tile has to live from its build until the last mask that reads it, which is
    # several positions into the following groups. Giving every (pass, group) pair in the whole
    # pipeline its own slot is the simple way to be sure of that, and it fits: both calls need 4
    # slots of the 6 and 7 they have. A future change that adds passes or splits the scale plan
    # finer should fail here rather than quietly alias two groups onto one tile.
    assert npass * nkey <= pdep
    for g in range(min(STOFF), npos + max(STOFF)):
        for st in range(6):
            pos = g - STOFF[st]
            if pos < 0 or pos >= npos:
                continue
            p = pos // ntc
            lp = pos - p * ntc
            gm = passes[p][0]
            nb0 = passes[p][1]
            t = torder[lp]
            c = pos % pdep
            cd = pos % DDEP
            cn = pos % NDEP
            sb = (p * nkey + tgrp[lp]) % pdep
            sfill = (lp == 0 or tgrp[lp] != tgrp[lp - 1])
            if st == 0:
                for u in range(ksp if merge else njob):
                    nwd = nbg * nw if merge else nw
                    jn0 = nb0 * nw if merge else nb0 * nw + jnb[u]
                    sg = segs[jt[u] + t]
                    for si in range(len(sg)):
                        pp = sg[si][0]
                        off = sg[si][1]
                        kb0 = sg[si][2]
                        ln = sg[si][3]
                        if sfill:
                            nisa.dma_copy(
                                dst=dbufs[gm][u][cd][0][pp:pp + ln, :],
                                src=ss[gm].ap(pattern=[[kblk * n, 1], [n, ln], [1, nwd]],
                                              offset=kb0 * n + jn0 + d * kblk * n,
                                              scalar_offset=j,
                                              indirect_dim=0).squeeze_dim(0))
                        nisa.dma_copy(
                            dst=dbufs[gm][u][cd][1][pp:pp + ln, :],
                            src=ws[gm].ap(pattern=[[kbytes * n, 1], [BPB * n, ln],
                                                   [1, nwd]],
                                          offset=(BPB * kb0 + off) * n + jn0
                                          + d * kbytes * n,
                                          scalar_offset=j,
                                          indirect_dim=0).squeeze_dim(0))
            elif st == 1:
                for gi in range(ngp):
                    b16 = dwt[gm][gi][cd]
                    gwd = gwid[gi]
                    ix = t * ngp + gi
                    nisa.tensor_scalar(dst=nib[cn][gi][:, 0:gwd].view(nl.uint32), data=b16,
                                       op0=nl.bitwise_and, operand0=QUAD_NIB_MASK,
                                       engine=NIBENG[0][ix % len(NIBENG[0])])
                    nisa.tensor_scalar(dst=nib[cn][gi][:, gwd:2 * gwd].view(nl.uint32),
                                       data=b16,
                                       op0=nl.right_shift, operand0=HI_NIB_SHIFT,
                                       op1=nl.bitwise_and, operand1=QUAD_NIB_MASK,
                                       engine=NIBENG[1][ix % len(NIBENG[1])])
            elif st == 2:
                for gi in range(ngp):
                    if sfill:
                        nisa.tensor_scalar(dst=scb[sb][gi], data=dsc[gm][gi][cd],
                                           op0=nl.multiply, operand0=SC_MUL,
                                           op1=nl.add, operand1=SC_ADD,
                                           engine=sceng[(t * ngp + gi) % len(sceng)])
            elif st == 3:
                for gi in range(ngp):
                    nisa.tensor_scalar(dst=pre[c][gi], data=nib[cn][gi],
                                       op0=nl.multiply, operand0=NIB_MUL,
                                       engine=preng[(2 * gi) % len(preng)])
            elif st == 4:
                for gi in range(ngp):
                    _fp4_dequant(pre[c][gi], scb[sb][gi], gwid[gi],
                                 MULENG[(t * ngp + gi) % len(MULENG)])
            else:
                if accs[p] is None:
                    ac = []
                    for _ in range(nbg):
                        ga = []
                        for wi in range(len(widths)):
                            ga.append(nl.ndarray((TOK, widths[wi]), dtype=nl.float32,
                                                 buffer=nl.psum))
                        ac.append(ga)
                    accs[p] = ac
                for q in range(njob):
                    gi = jobgrp[q][0]
                    qw0 = jobgrp[q][1] * nw
                    dqf = pre[c][gi].view(nl.bfloat16)
                    for high in range(2):
                        base = high * gwid[gi] + qw0
                        for wi in range(len(widths)):
                            qw = base + wi * NMAX
                            nisa.nc_matmul(dst=accs[p][ja[q]][wi],
                                           stationary=alay[:, 2 * (jt[q] + t) + high, :],
                                           moving=dqf[:, qw:qw + widths[wi]],
                                           accumulate=(lp > 0 or high > 0 or jt[q] > 0))
        for p in range(npass):
            if g == p * ntc + ntc - 1 + STOFF[5]:
                gm = passes[p][0]
                nb0 = passes[p][1]
                di = 0
                for a in range(nbg):
                    for wi in range(len(widths)):
                        qw = wi * NMAX
                        jn0 = (nb0 + a) * nw
                        sl = slice(jn0 + qw, jn0 + qw + widths[wi])
                        deng = DRNENG[di % len(DRNENG)]
                        di += 1
                        if yacc is None:
                            nisa.tensor_copy(dst=outs[gm][:, sl], src=accs[p][a][wi],
                                             engine=deng)
                        else:
                            # Vector adds the float32 accumulator straight out of PSUM. The
                            # bfloat16 landing tile this used to round through cost Scalar a
                            # copy of every one of w2's 5120 output columns per expert -- 0.24 ms
                            # per core on the engine the routed pipeline's mask stage waits on,
                            # and Scalar has only ~9% slack over Vector in this loop. Dropping it
                            # is the one change here that is not bit-identical: it removes a
                            # rounding, so the partial is strictly closer to the float32 sum the
                            # reference takes, and `max_abs_err` is reported below.
                            nisa.tensor_tensor(dst=yacc[:, sl], data1=yacc[:, sl],
                                               data2=accs[p][a][wi],
                                               op=nl.add, engine=nisa.engine.vector)


def _route_col(j, base, weff):
    wcol = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wcol,
                  src=weff.ap(pattern=[[N_EXPERTS, TOK], [1, 1]], offset=base,
                              scalar_offset=j, indirect_dim=1))
    return wcol


def _expert_step(j, d, wcol, w1, s1, w3, s3, w2, s2, seg1, alay1, seg2, db, y):
    g = nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf)
    u = nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf)
    _gemm_fp4([w1, w3], [s1, s3], j, KB1, KBLK1, INTER, NW1, NBG1, seg1, SSC1, alay1,
              [db[0], db[1]], [g, u], None, SCENG1, PRENG1, KSP1, PDEP1, d)

    t = nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf)
    _swiglu(g, u, wcol, t)

    _gemm_fp4([w2], [s2], j, KB2, KBLK2, DIM, NW2, NBG2, seg2, SSC2,
              _quant_layout(t, TOK, INTER, CHK2),
              [db[2]], None, y, SCENG2, PRENG2, KSP2, PDEP2, d)


def _pow2(e_int, m, n):
    """2**clamp(e_int, -EXP_BIAS, 63 - EXP_BIAS) as float32, in one instruction.

    The bootstrapped module built this as a product over the six bits of `idx`: thirteen
    instructions and, worse, a chain of thirteen dependent ones, inside the shared expert's
    window, which the profile finds latency-bound rather than throughput-bound. The exponent
    field *is* the answer -- `2**(idx - EXP_BIAS)` is the float32 whose bit pattern is
    `(idx - EXP_BIAS + BF_BIAS) << 23` -- so one `tensor_scalar` on the integer index writes it
    directly. Every intermediate is `k << 23` with `k <= 158`, six significant bits, so the
    engine's float32 arithmetic is exact and the value is bit-identical to the product form.
    """
    idx = nl.minimum(nl.maximum(nl.add(e_int, EXP_BIAS), 0), 63)
    res = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res.view(nl.uint32), data=idx,
                       op0=nl.multiply, operand0=float(1 << F32_MANT),
                       op1=nl.add, operand1=float((BF_BIAS - EXP_BIAS) << F32_MANT))
    return res


def _act_quant8(x_bf16, m, k):
    """`act_quant(x, 32, "ue8m0")` followed by its dequant, exactly as module/source.py.

    The activation the shared expert's GEMMs consume: quantised to fp8 e4m3 in blocks of 32
    with the 1e-4 amax floor and a power-of-two scale, then dequantised back to bf16. The
    halve-then-double around the fp8 cast keeps every value under 240, which is where this
    toolchain's fp8 converter stops being finite.
    """
    nb = k // ABLK
    x2 = nl.copy(x_bf16, dtype=nl.float32)
    xf = x2.reshape((m, nb, ABLK))
    amax = nl.maximum(nl.max(nl.abs(xf), axis=2), AMAX_FLOOR)

    ratio = nl.multiply(amax, 1.0 / FP8_MAX)
    e = nl.copy(nl.ceil(nl.multiply(nl.log(ratio), LOG2E)), dtype=nl.int32)
    s = _pow2(e, m, nb)
    inv = _pow2(nl.subtract(0, e), m, nb)

    s3 = s.reshape((m, nb, 1))
    i3 = inv.reshape((m, nb, 1))
    # These two materialise (m, k) float32 tiles where `_act_quant` and `_fp8_dequant` both use a
    # stride-0 `.broadcast(2, ABLK)` view instead. Substituting the views here compiles and is
    # byte-identical, but measured 17.2103 ms against 17.2109 -- flat, so it is left as the
    # bootstrapped module wrote it rather than changed for nothing.
    sb = nl.broadcast_to(s3, (m, nb, ABLK))
    ib = nl.broadcast_to(i3, (m, nb, ABLK))

    v = nl.multiply(nl.minimum(nl.maximum(nl.multiply(xf, ib), -FP8_MAX), FP8_MAX), 0.5)
    q = nl.copy(nl.copy(v, dtype=nl.float8_e4m3), dtype=nl.float32)
    dq = nl.multiply(nl.multiply(q, 2.0), sb)
    return nl.copy(dq, dtype=nl.bfloat16)


def _scale_rows(s, nt, nbtot, kb0=0, b0=0):
    """One (128, nbtot) bf16 tile per K-tile of an fp8 GEMM, built once per GEMM.

    Entry [32*r + q, c] is 2**(s[4*i + r, c] - 7): the e8m0 block scale with the 2**120 that
    `_fp8_dequant` leaves behind already folded into its exponent, which is what SC_ADD8's
    120 << 7 is.

    The replication across a k-block's 32 partitions is done **by the DMA**, with stride 0 on the
    partition axis: one transfer reads scale row `kb` into 32 partitions at once. That is what
    `nl.broadcast_to` was doing, except `broadcast_to` can only source partition 0, so it forced
    a (1, nbtot) landing tile and therefore a separate load, bit-build and copy per row -- 12
    instructions per 128-partition tile, each touching 8 to 12 elements, which is all
    per-instruction overhead. Now it is 4 transfers and one (128, nbtot) bit-build per tile.

    `kb0`/`b0` name the first k-block row and the first n-block column of the slice this core
    owns, in units of ABLK: the two physical cores split the shared expert's intermediate.
    """
    row = s.shape[1]
    tiles = []
    for i in range(nt):
        raw = nl.ndarray((PAR, nbtot), dtype=nl.uint8, buffer=nl.sbuf)
        for r in range(PAR // ABLK):
            kb = kb0 + i * (PAR // ABLK) + r
            nisa.dma_copy(dst=raw[r * ABLK:(r + 1) * ABLK, :],
                          src=s.ap(pattern=[[0, ABLK], [1, nbtot]], offset=kb * row + b0))
        out = nl.ndarray((PAR, nbtot), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=out.view(nl.uint16), data=raw, op0=nl.multiply, operand0=SC_MUL8,
                           op1=nl.add, operand1=SC_ADD8, engine=nisa.engine.vector)
        tiles.append(out)
    return tiles


def _fp8_dequant(byte, sc, m, nw, nb):
    """fp8 e4m3fn weight bytes -> scaled bf16, in four elementwise instructions.

    The bf16 bit pattern of code b is (b & 0x80) << 8 | (b & 0x7f) << 4, and that pattern
    reads as the value times 2**-120 -- uniformly, normals and subnormals alike, because bf16
    subnormals reach far enough down to hold b's. `_scale_rows` carries the 2**120 back.

    The hardware's fp8 converter cannot stand in for this: it reads exponent 15 as inf/NaN,
    and 0.6% of these weight bytes are exponent 15 (|w| >= 256, which e4m3fn represents and
    the reference's fp8_gemm consumes -- the largest magnitude code present is 0x7e, 448).

    The two mask/shift steps run on a uint32 view, so each lane carries two weights; the
    masks are doubled accordingly and E4M3_SGN2 is written as a signed int32.
    """
    t = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=byte, op0=nl.multiply, operand0=BYTE_HI,
                       engine=nisa.engine.scalar)
    hi = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi.view(nl.uint32), data=t.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=E4M3_MAG2,
                       op1=nl.right_shift, operand1=4, engine=nisa.engine.vector)
    bits = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=bits.view(nl.uint32), data=t.view(nl.uint32),
                              op0=nl.bitwise_and, operand0=E4M3_SGN2,
                              op1=nl.bitwise_or, operand1=hi.view(nl.uint32))
    wdq = nl.ndarray((m, nw), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=wdq.reshape((m, nb, ABLK)),
                       data1=bits.view(nl.bfloat16).reshape((m, nb, ABLK)),
                       data2=sc.reshape((m, nb, 1)).broadcast(2, ABLK),
                       op=nl.multiply, engine=FP8ENG)
    return wdq


def _gemm_fp8(a_t, w, s, k, n, k0=0, n0off=0):
    """(TOK, k) x (k, n) with fp8 e4m3fn weights and e8m0 block scales, bf16 out.

    `k0`/`n0off` name the first row / first column of `w` this core owns; both are multiples
    of PAR / ABLK respectively, so every slice is a whole number of GEMM tiles and scale blocks.
    """
    nt = k // PAR
    nbtot = n // ABLK
    sctiles = _scale_rows(s, nt, nbtot, k0 // ABLK, n0off // ABLK)
    out = nl.ndarray((TOK, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    for n0 in range(0, n, NMAX):
        nw = min(NMAX, n - n0)
        nb = nw // ABLK
        acc = nl.ndarray((TOK, nw), dtype=nl.float32, buffer=nl.psum)
        for i in range(nt):
            byte = nl.ndarray((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=byte, src=w[k0 + i * PAR:k0 + (i + 1) * PAR,
                                          n0off + n0:n0off + n0 + nw])
            wdq = _fp8_dequant(byte, sctiles[i][:, n0 // ABLK:n0 // ABLK + nb],
                               PAR, nw, nb)
            nisa.nc_matmul(dst=acc, stationary=a_t[:, i, :], moving=wdq,
                           accumulate=(i > 0))
        nisa.tensor_copy(dst=out[:, n0:n0 + nw], src=acc, engine=nisa.engine.scalar)
    return out


def _swiglu8(g_bf16, u_bf16):
    g = nl.minimum(nl.copy(g_bf16, dtype=nl.float32), SWIGLU_LIMIT)
    u = nl.minimum(nl.maximum(nl.copy(u_bf16, dtype=nl.float32), -SWIGLU_LIMIT),
                   SWIGLU_LIMIT)
    return nl.multiply(nl.silu(g), u)


def _shared_shard(h, sw1, ss1, sw2, ss2, sw3, ss3, c0, cw):
    """One core's piece of this rank's quarter of the shared expert.

    `inference.py` hands over intermediate columns `r*640 .. r*640+639` of w1/w3 and exactly
    the matching rows of w2, so nothing here knows which rank it is. The same argument that
    splits it across ranks splits it again across the two physical cores: SwiGLU is elementwise
    along the intermediate dimension and w2 contracts over it, so columns `c0 .. c0+cw-1` of
    this rank's slice are a self-contained partial. Both `c0` and `cw` are multiples of PAR, so
    every w2 contraction is a whole number of 128-row GEMM tiles and no scale block is split
    (ABLK divides PAR, so `_act_quant8`'s blocks land the same either way).
    """
    hq = _act_quant8(h, TOK, DIM)
    hq_t = _transpose_stack(hq.reshape((TOK, DIM)), DIM)
    sg = _gemm_fp8(hq_t, sw1, ss1, DIM, cw, n0off=c0)
    su = _gemm_fp8(hq_t, sw3, ss3, DIM, cw, n0off=c0)
    st = nl.copy(_swiglu8(sg, su), dtype=nl.bfloat16)
    stq = _act_quant8(st, TOK, cw)
    return _gemm_fp8(_transpose_stack(stq.reshape((TOK, cw)), cw),
                     sw2, ss2, cw, DIM, k0=c0)


@nki.jit
def kernel(hidden, norm_w, gate_wt, gate_b, w1, s1, w3, s3, w2, s2,
           sw1, ss1, sw2, ss2, sw3, ss3, replica_group):
    out = nl.ndarray((TOK, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # name= on the collective's src/dst, or DRAM allocation fails with NCC_IBIR440; both
    # shared_hbm, because a collective may not mix buffer kinds; both scratch, because a
    # collective may not read or write an IO tensor (NCC_INLA001).
    cc_src = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    # The two physical cores of this rank's logical core split the rank's 96 routed experts and
    # its shared-expert slice. Each drops its float32 partial in its own half of `part`;
    # `nisa.core_barrier` is the handshake that makes the other half readable. See the seam below.
    part = nl.ndarray((CORES * TOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="part")

    # nl.program_id is a trace-time constant here -- the launch grid is traced once per physical
    # core -- so a core's share of the work is chosen statically, and the NEFF carries both.
    pid = nl.program_id(0)
    pbase = pid * CHH

    x = nl.load(hidden, dtype=nl.float32)
    gw = nl.load(norm_w, dtype=nl.float32)
    var = nl.multiply(nl.sum(nl.multiply(x, x), axis=1, keepdims=True), 1.0 / DIM)
    hf = nl.multiply(x, nl.rsqrt(nl.add(var, NORM_EPS)))
    h = nl.copy(nl.multiply(hf, nl.broadcast_to(gw, (TOK, DIM))), dtype=nl.bfloat16)

    h_t = _transpose_stack(h, DIM)
    logits = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.psum)
    for i in range(DIM // PAR):
        nisa.nc_matmul(dst=logits, stationary=h_t[:, i, :],
                       moving=nl.load(gate_wt[i * PAR:(i + 1) * PAR, :]),
                       accumulate=(i > 0))
    scores = nl.sqrt(nl.softplus(nl.copy(logits, dtype=nl.float32)))

    sel = nl.add(scores, nl.broadcast_to(nl.load(gate_b, dtype=nl.float32),
                                         (TOK, N_EXPERTS)))
    best = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=best, src=sel)
    sixth = nl.copy(best[:, TOPK - 1:TOPK])
    keep = nl.greater_equal(sel, nl.broadcast_to(sixth, (TOK, N_EXPERTS)))
    picked = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    picked[...] = nl.multiply(scores, nl.copy(keep, dtype=nl.float32))
    total = nl.add(nl.sum(picked, axis=1, keepdims=True), SUM_EPS)
    inv = nl.multiply(nl.reciprocal(total), ROUTE_SCALE)
    weff = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.private_hbm)
    nl.store(weff, value=nl.multiply(picked, nl.broadcast_to(inv, (TOK, N_EXPERTS))))

    y = nl.zeros((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)

    alay1 = _quant_layout(h, TOK, DIM, CHK1)
    dba = _alloc_db(NBG1, NW1, NBG1 * KSP1, DDEP)
    dbb = _alloc_db(NBG1, NW1, NBG1 * KSP1, DDEP)
    db2 = _alloc_db(NBG2, NW2, NBG2 * KSP2, DDEP)
    db = (dba, dbb, db2)

    def body(j):
        for d in range(UNROLL):
            _expert_step(j, d + pbase, _route_col(j, d + pbase, weff), w1, s1, w3, s3, w2, s2,
                         SEG1, alay1, SEG2, db, y)

    nl.fori_loop(0, CHH, body, step=UNROLL)

    # The shared expert is latency-bound rather than throughput-bound: inside its window the
    # profile finds ~0.2 ms (core 0) / ~0.26 ms (core 1) where all five engines are idle at once
    # and DMA is moving 5 GB/s. Pulling the last 4 experts out of the loop so they share a basic
    # block with it -- with `nisa.register_alloc(44)` standing in for the loop index, since
    # `scalar_offset` must be a NkiTensor or VirtualRegister and rejects a Python int -- did not
    # fill those holes: fastest rank 17.8680 ms against 17.8688, and the rank spread went from
    # 0.017 ms to 0.252 ms with two of the four ranks landing at 18.12. Left as a loop.
    shared = _shared_shard(h, sw1, ss1, sw2, ss2, sw3, ss3, SH_C0[pid], SH_CW[pid])
    y[...] = nl.add(y, nl.copy(shared, dtype=nl.float32))

    # The intra-logical-core seam. Both cores publish their partial, barrier, and then add the
    # other core's partial into their own -- so both cores hold this rank's whole partial and
    # write `cc_src` with the same bytes. That is deliberate: the collective then depends only on
    # its own core's writes, and the one cross-core ordering this kernel relies on is the
    # documented one, `core_barrier` between a `dma_copy` into shared HBM and a later read of it.
    # Publishing only half each and letting the two cores fill disjoint halves of `cc_src` was
    # measured at 17.8639 ms against 17.8774 here -- 13 us, not worth resting the collective's
    # input on a barrier ordering a DMA against `all_reduce` rather than against a load.
    oth = (1 - pid) * TOK
    nisa.dma_copy(dst=part[pid * TOK:pid * TOK + TOK, :], src=y)
    nisa.core_barrier(data=part, cores=(0, 1))
    for c in range(0, DIM, CCH):
        nisa.dma_copy(dst=cc_src[:, c:c + CCH],
                      src=nl.add(y[:, c:c + CCH], nl.load(part[oth:oth + TOK, c:c + CCH])))

    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)
    nl.store(out, value=nl.copy(nl.load(cc_dst), dtype=nl.bfloat16))
    return out
